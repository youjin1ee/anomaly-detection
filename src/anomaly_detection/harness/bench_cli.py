"""`python -m anomaly_detection.harness.bench_cli`: SMD and PSM benchmark.

    run     score dimensions with the models and keep the test scores (resumable)
    report  metrics and markdown tables from the kept scores

The data directories are arguments, nothing is downloaded or stored in the repository.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .bench_data import Machine, label_events, load_psm, load_smd
from .bench_metrics import (
    DEFAULT_RULE,
    average_precision,
    cause_hits,
    event_f1_at,
    oracle_event_f1,
    pa_f1_at,
    pa_oracle_f1,
    server_scores,
)
from .bench_runner import BENCH_MODELS, Job, run_jobs, score_path, stratified_subset
from .metrics import Summary

SUBSETS = {"all": None, "s8": 8, "s3": 3}  # s3 is chosen inside s8, so the sets are nested


def subset_names(machines: list[Machine], name: str) -> list[str]:
    names = [m.name for m in machines]
    if name == "all" or len(names) == 1:
        return names
    rates = {m.name: float(m.labels.mean()) for m in machines}
    s8 = stratified_subset(names, [rates[n] for n in names], 8)
    if name == "s8":
        return s8
    return stratified_subset(s8, [rates[n] for n in s8], SUBSETS[name])


def dataset_key(dataset: str, test_rows: int | None) -> str:
    return dataset if test_rows is None else f"{dataset}-first{test_rows}"


def load_machines(
    dataset: str, data_dir: Path, test_rows: int | None, subset: str = "all"
) -> list[Machine]:
    """The machines of a subset, with only the first `test_rows` test rows if given. The subset
    is chosen on the full data, so it does not depend on `test_rows`."""
    ms = load_smd(data_dir) if dataset == "smd" else load_psm(data_dir)
    keep = set(subset_names(ms, subset))
    ms = [m for m in ms if m.name in keep]
    if test_rows is None:
        return ms
    return [
        Machine(
            m.name,
            m.train,
            m.test[:test_rows],
            m.labels[:test_rows],
            m.train_ts,
            m.test_ts[:test_rows],
            m.interpretation,
        )
        for m in ms
    ]


def usable_dims(m: Machine) -> list[int]:
    const = set(m.constant_dims())
    return [j for j in range(m.dims) if j not in const]


def cmd_run(args: argparse.Namespace) -> int:
    machines = load_machines(args.dataset, args.data_dir, args.test_rows, args.machines)
    key = dataset_key(args.dataset, args.test_rows)
    models = [x for x in args.models.split(",") if x]
    for x in models:
        if x not in BENCH_MODELS:
            raise SystemExit(f"unknown model {x!r}; known: {', '.join(BENCH_MODELS)}")
    jobs: list[Job] = []
    points = 0
    for m in machines:
        for j in usable_dims(m):
            points += len(m.test)
            for x in models:
                jobs.append((x, m.name, j, m.train[:, j], m.test[:, j], m.train_ts, m.test_ts))
    # longest jobs first keeps the pool busy until the end
    cost = {"rcf": 6, "tspulse": 7, "toto-4m": 2, "awsrcf": 0.4, "awsrcf-nab": 0.2}
    jobs.sort(key=lambda j: -cost.get(j[0], 0.02) * len(j[4]))
    t0 = time.time()
    spent = run_jobs(jobs, args.out, key, args.workers)
    wall = time.time() - t0
    rec = {
        "dataset": key,
        "machines": args.machines,
        "models": models,
        "test_points_per_model": points,
        "core_seconds": spent,
        "wall_seconds": wall,
    }
    with (args.out / "timing.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    return 0


def _load_dims(
    out: Path, key: str, base: str, model: str, m: Machine, dims: list[int]
) -> np.ndarray | None:
    """Scores of the usable dimensions. A scope of the first N test rows reads its own scores
    if they exist and otherwise the first N rows of the full run, which are the same numbers
    because every model sees the rows in order."""
    cols = []
    for j in dims:
        f = score_path(out, key, model, m.name, j)
        if f.exists():
            cols.append(np.load(f))
            continue
        f = score_path(out, base, model, m.name, j)
        if not f.exists():
            return None
        cols.append(np.load(f)[: len(m.test)])
    return np.stack(cols, axis=1).astype(float)


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x:.{nd}f}"


def evaluate_scope(
    out: Path, key: str, base: str, machines: list[Machine], model: str, seed: int = 0
) -> dict | None:
    """All metrics of one model on one set of machines, or None if scores are missing."""
    per_server: list[tuple[np.ndarray, np.ndarray]] = []
    hits = n_ev = chance = 0.0
    hits_hot = n_hot = 0.0
    excluded = {}
    rng = np.random.default_rng(seed)
    for m in machines:
        dims = usable_dims(m)
        excluded[m.name] = m.dims - len(dims)
        if model == "random":
            dim_sc = rng.random((len(m.test), len(dims)))
        else:
            dim_sc = _load_dims(out, key, base, model, m, dims)
            if dim_sc is None:
                return None
        per_server.append((server_scores(dim_sc), m.labels))
        full = np.full((len(m.test), m.dims), np.nan)
        full[:, dims] = dim_sc
        h, n, c = cause_hits(full, m.labels, m.interpretation, dims)
        hits, n_ev, chance = hits + h, n_ev + n, chance + c
        h2, n2, _ = cause_hits(full, m.labels, m.interpretation, dims, min_score=0.5)
        hits_hot, n_hot = hits_hot + h2, n_hot + n2
    step = float(machines[0].test_ts[1] - machines[0].test_ts[0])
    op: Summary = event_f1_at(per_server, step, DEFAULT_RULE)
    f1o, thr, summ_o = oracle_event_f1(per_server, step)
    pooled_s = np.concatenate([s for s, _ in per_server])
    pooled_y = np.concatenate([y for _, y in per_server])
    pa_o, pa_thr = pa_oracle_f1(per_server)
    return {
        "model": model,
        "events": op.events,
        "detected": op.detected,
        "alerts": op.alerts,
        "precision": op.precision,
        "recall": op.recall,
        "f1": op.f1,
        "fa_per_day": op.false_alarms_per_day,
        "delay_median_min": op.median_delay_min,
        "oracle_f1": f1o,
        "oracle_thr": thr,
        "oracle_precision": summ_o.precision,
        "oracle_recall": summ_o.recall,
        "ap": average_precision(pooled_s, pooled_y),
        "pa_f1_05": pa_f1_at(per_server, 0.5),
        "pa_oracle_f1": pa_o,
        "pa_oracle_thr": pa_thr,
        "cause_hits": hits,
        "cause_events": n_ev,
        "cause_chance": chance,
        "cause_hits_hot": hits_hot,
        "cause_events_hot": n_hot,
        "excluded_dims": excluded,
    }


def render_table(rows: list[dict]) -> str:
    out = [
        "| model | events | detected | alerts | precision | recall | F1 | false alarms/day |"
        " delay median (min) | oracle F1 (thr) | AP |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['model']} | {r['events']} | {r['detected']} | {r['alerts']} |"
            f" {_f(r['precision'])} | {_f(r['recall'])} | {_f(r['f1'])} |"
            f" {_f(r['fa_per_day'], 2)} | {_f(r['delay_median_min'], 1)} |"
            f" {_f(r['oracle_f1'])} ({r['oracle_thr']:.4g}) | {_f(r['ap'])} |"
        )
    out += [
        "",
        "point adjust F1, reference only",
        "",
        "| model | PA F1 at 0.5 | PA F1 oracle (thr) |",
        "|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['model']} | {_f(r['pa_f1_05'])} |"
            f" {_f(r['pa_oracle_f1'])} ({r['pa_oracle_thr']:.4g}) |"
        )
    if not any(r["cause_events"] for r in rows):
        return "\n".join(out)
    out += [
        "",
        "cause: share of events whose highest scoring dimension is a listed dimension",
        "",
        "| model | events | hits | share | chance | events with score >= 0.5 | share there |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        n = r["cause_events"]
        nh = r["cause_events_hot"]
        out.append(
            f"| {r['model']} | {n} | {r['cause_hits']:.1f} |"
            f" {_f(r['cause_hits'] / n if n else float('nan'))} |"
            f" {_f(r['cause_chance'] / n if n else float('nan'))} | {nh} |"
            f" {_f(r['cause_hits_hot'] / nh if nh else float('nan'))} |"
        )
    return "\n".join(out)


def cmd_report(args: argparse.Namespace) -> int:
    key = dataset_key(args.dataset, args.test_rows)
    result = {}
    parts = []
    default = "all,s8,s3" if args.dataset == "smd" else "all"
    for sub in (args.machines or default).split(","):
        scope = load_machines(args.dataset, args.data_dir, args.test_rows, sub)
        names = {m.name for m in scope}
        rows = []
        for model in (*BENCH_MODELS, "random"):
            r = evaluate_scope(args.out, key, args.dataset, scope, model)
            if r is not None:
                rows.append(r)
        result[sub] = {"machines": sorted(names), "rows": rows}
        parts.append(f"### {key}, machines {sub} ({len(names)}): {', '.join(sorted(names))}\n")
        parts.append(render_table(rows) + "\n")
        parts.append(f"label events in scope: {sum(len(label_events(m.labels)) for m in scope)}\n")
    (args.out / f"report-{key}.json").write_text(json.dumps(result, indent=1, default=float))
    text = "\n".join(parts)
    (args.out / f"report-{key}.md").write_text(text)
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench_cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("run", cmd_run), ("report", cmd_report)):
        s = sub.add_parser(name)
        s.add_argument("--dataset", choices=("smd", "psm"), required=True)
        s.add_argument("--data-dir", type=Path, required=True)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--test-rows", type=int, default=None, help="first N test rows only")
        s.set_defaults(fn=fn)
    r = sub.choices["run"]
    r.add_argument("--models", default=",".join(BENCH_MODELS))
    r.add_argument("--machines", choices=tuple(SUBSETS), default="all")
    r.add_argument("--workers", type=int, default=8)
    sub.choices["report"].add_argument("--machines", default=None)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
