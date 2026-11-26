"""Score a lab dump with every model and break the result down for choosing a model.

Builds on `lab.replay` (same alert rule, same stretching of the compressed day) and adds:

- models beyond the harness defaults (subpca, tspulse, toto-4m) and scores cached per model and
  signal, computed in parallel;
- recall with a 95 percent Wilson interval, by scenario, by time of day band (peak, trough)
  and by event order inside a scenario (does a model stop reacting after the first event);
- false alarms per day on the compressed day and on a 24 hour lab day, and where in the day
  they fall;
- what each model did during stretches of the trace that look anomalous by themselves
  (`natural_candidates` of the label file). Those stretches lie before the evaluation start, so
  they never count as true events or false alarms in the other tables.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..harness.metrics import AlertRule, CaseResult, alerts, summarize
from ..harness.runner import ModelSpec, parse_model
from . import replay
from .signals import SIGNALS

# Models the harness runner does not list. Paths of the checkpoints come from the command line.
EXTRA_MODELS = {"subpca": "subpca", "tspulse": "tspulse", "toto-4m": "toto"}


def resolve(name: str, paths: dict[str, str]) -> ModelSpec:
    if name not in EXTRA_MODELS:
        return parse_model(name)
    kind = EXTRA_MODELS[name]
    params: dict = {}
    if kind == "toto":
        params = {"model_dir": paths["toto"], "threads": 1}
    elif kind == "tspulse":
        params = {"model_path": paths["tspulse"], "threads": 1}
    return ModelSpec(name, kind, params)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 percent interval of a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _job(a: tuple) -> str:
    spec, run, step, scale, path = a
    scores = replay.score(spec.name, spec.kind, spec.params, run, step, scale)
    np.save(path, scores)
    return path


def compute_scores(
    runs: dict[str, replay.SignalRun],
    models: list[str],
    paths: dict[str, str],
    step: int,
    scale: float,
    cache: Path,
    workers: int,
) -> dict[tuple[str, str], np.ndarray]:
    """(model, signal) -> scores, from the cache where present."""
    cache.mkdir(parents=True, exist_ok=True)
    todo, out = [], {}
    for m in models:
        spec = resolve(m, paths)
        for sig, run in runs.items():
            f = cache / f"{m}.{sig}.npy"
            if f.exists():
                out[(m, sig)] = np.load(f)
            else:
                todo.append((spec, run, step, scale, str(f)))
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, maxtasksperchild=1) as pool:
            for done in pool.imap_unordered(_job, todo):
                print("scored", done, flush=True)
        for m in models:
            for sig in runs:
                out[(m, sig)] = np.load(cache / f"{m}.{sig}.npy")
    return out


@dataclass
class EventRow:
    model: str
    scenario: str
    order: int  # 1 based, chronological inside the scenario
    band: str
    detected: bool
    delay: float | None  # lab seconds
    max_score: float


def event_rows(
    model: str,
    run: replay.SignalRun,
    res: CaseResult,
    scores: np.ndarray,
    bands: list[str],
    step: int,
    rule: AlertRule,
) -> list[EventRow]:
    rows = []
    for k, (s, e) in enumerate(run.own):
        hits = [f for f in res.fires if s <= f <= e + rule.grace]
        window = scores[s : e + rule.grace + 1]
        mx = float(np.nanmax(window)) if np.isfinite(window).any() else float("nan")
        rows.append(
            EventRow(
                model,
                run.case.scenario,
                k + 1,
                bands[k],
                bool(hits),
                (hits[0] - s) * step if hits else None,
                mx,
            )
        )
    return rows


def _f(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def _recall_cell(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {_f(k / n if n else float('nan'))} [{_f(lo)}, {_f(hi)}]"


def _delay_cells(delays: list[float], scale: float) -> tuple[str, str]:
    """median lab seconds, median minutes on the replayed trace (compressed time stretched)."""
    if not delays:
        return "n/a", "n/a"
    m = float(np.median(delays))
    return _f(m, 0), _f(m * scale / 60, 0)


def fa_band(fire_t: float, meta: dict) -> str:
    frac = ((fire_t - meta["epoch"]) / meta["day_seconds"]) % 1.0
    for name, (lo, hi) in meta.get("bands", {}).items():
        if lo <= frac < hi:
            return name
    return "other"


def render_all(
    rows: list[replay.Row],
    evs: list[EventRow],
    meta: dict,
    model_order: list[str],
    scale: float,
    step: int,
    fa_bands: dict[tuple[str, str], list[str]],
) -> str:
    day = meta["day_seconds"]
    out: list[str] = []

    out += ["", "### Pooled over scenarios", ""]
    out += [
        "| model | events | detected | recall [95% CI] | alerts | false alarms | precision | F1 "
        "| delay median (lab s) | delay median (trace min) "
        "| false alarms per signal per compressed day "
        "| false alarms per signal per 24 h lab day | false alarms per compressed day, 3 signals |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in model_order:
        mine = [r.result for r in rows if r.model == m]
        if not mine:
            continue
        s = summarize(mine)
        d_lab, d_trace = _delay_cells(s.delays, scale)
        cdays = s.eval_days * 86400 / day
        nsig = len(mine)
        out.append(
            f"| {m} | {s.events} | {s.detected} | {_recall_cell(s.detected, s.events)} "
            f"| {s.alerts} | {s.false_alarms} | {_f(s.precision)} | {_f(s.f1)} "
            f"| {d_lab} | {d_trace} "
            f"| {_f(s.false_alarms / cdays)} | {_f(s.false_alarms / s.eval_days)} "
            f"| {_f(s.false_alarms / cdays * nsig)} |"
        )

    out += ["", "### By scenario", ""]
    out += [
        "| scenario | signal | model | events | detected | recall [95% CI] | alerts | false alarms "
        "| of which during other injections | precision | F1 | delay median (lab s) "
        "| delay p90 (lab s) | false alarms per compressed day | false alarms per 24 h lab day |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r.scenario, model_order.index(r.model))):
        s = summarize([r.result])
        p90 = _f(float(np.percentile(s.delays, 90)), 0) if s.delays else "n/a"
        med = _f(float(np.median(s.delays)), 0) if s.delays else "n/a"
        out.append(
            f"| {r.scenario} | {r.signal} | {r.model} | {s.events} | {s.detected} "
            f"| {_recall_cell(s.detected, s.events)} | {s.alerts} | {s.false_alarms} "
            f"| {r.cross_alarms} | {_f(s.precision)} | {_f(s.f1)} | {med} | {p90} "
            f"| {_f(s.false_alarms / (r.result.eval_days * 86400 / day))} "
            f"| {_f(s.false_alarms / r.result.eval_days)} |"
        )

    out += ["", "### By time of day band (injection started in the peak or trough band)", ""]
    out += [
        "| scenario | band | model | events | detected | recall [95% CI] | delay median (lab s) |",
        "|---|---|---|---|---|---|---|",
    ]
    scen = [*sorted({e.scenario for e in evs}), "all"]
    for sc in scen:
        for band in ("peak", "trough"):
            for m in model_order:
                sel = [
                    e
                    for e in evs
                    if e.model == m and e.band == band and (sc == "all" or e.scenario == sc)
                ]
                if not sel:
                    continue
                k = sum(e.detected for e in sel)
                d_lab, _ = _delay_cells([e.delay for e in sel if e.delay is not None], scale)
                out.append(
                    f"| {sc} | {band} | {m} | {len(sel)} | {k} | {_recall_cell(k, len(sel))} "
                    f"| {d_lab} |"
                )

    out += ["", "### Where false alarms fell (all signals, unmatched alerts)", ""]
    out += ["| model | peak band | trough band | other times | total |", "|---|---|---|---|---|"]
    for m in model_order:
        cnt: dict[str, int] = defaultdict(int)
        for (mm, _), bands in fa_bands.items():
            if mm == m:
                for b in bands:
                    cnt[b] += 1
        out.append(
            f"| {m} | {cnt['peak']} | {cnt['trough']} | {cnt['other']} | {sum(cnt.values())} |"
        )
    return "\n".join(out) + "\n"


def render_order(evs: list[EventRow], model_order: list[str]) -> str:
    """Highest score inside each event window by order of the event inside its scenario. A
    model that learns the injected state as normal shows high scores first and low later."""
    out = ["", "### Highest score inside the event, by event order", ""]
    n = max((e.order for e in evs), default=0)
    out += [
        "| scenario | model | " + " | ".join(f"#{i}" for i in range(1, n + 1)) + " | mean #1 "
        "| mean #2 onwards | detected #1 | detected #2 onwards |",
        "|---|---|" + "---|" * (n + 4),
    ]
    for sc in [*sorted({e.scenario for e in evs}), "all"]:
        for m in model_order:
            sel = [e for e in evs if e.model == m and (sc == "all" or e.scenario == sc)]
            if not sel:
                continue
            cells = []
            for i in range(1, n + 1):
                x = [e.max_score for e in sel if e.order == i]
                cells.append(_f(float(np.mean(x))) if x else "n/a")
            first = [e for e in sel if e.order == 1]
            rest = [e for e in sel if e.order > 1]
            out.append(
                f"| {sc} | {m} | " + " | ".join(cells) + " "
                f"| {_f(float(np.mean([e.max_score for e in first])) if first else float('nan'))} "
                f"| {_f(float(np.mean([e.max_score for e in rest])) if rest else float('nan'))} "
                f"| {sum(e.detected for e in first)}/{len(first)} "
                f"| {sum(e.detected for e in rest)}/{len(rest)} |"
            )
    return "\n".join(out) + "\n"


def natural_rows(
    meta: dict,
    natural: list[dict],
    runs: dict[str, replay.SignalRun],
    scores: dict[tuple[str, str], np.ndarray],
    models: list[str],
    step: int,
    rule: AlertRule,
) -> str:
    out = [
        "",
        "### Natural candidates (low traffic stretches of the trace)",
        "",
        "Alerts anywhere from the start of the dump, not only after the evaluation start. A model "
        "with no score yet in the stretch shows n/a.",
        "",
        "| stretch | signal | model | alerts inside | first alert after start (lab s) "
        "| share of samples at score 0.5 or more | highest score |",
        "|---|---|---|---|---|---|---|",
    ]
    for nat in natural:
        for sig, run in runs.items():
            ts = run.case.ts
            s = replay.index_of(ts, nat["start"], step)
            e = replay.index_of(ts, nat["end"], step)
            for m in models:
                sc = scores[(m, sig)]
                win = sc[s : e + 1]
                if not np.isfinite(win).any():
                    out.append(f"| days {nat['days']} | {sig} | {m} | n/a | n/a | n/a | n/a |")
                    continue
                fires = [f for f in alerts(sc, rule, 0) if s <= f <= e + rule.grace]
                hot = float(np.mean(np.nan_to_num(win, nan=0.0) >= rule.threshold))
                first = _f((fires[0] - s) * step, 0) if fires else "n/a"
                out.append(
                    f"| days {nat['days']} | {sig} | {m} | {len(fires)} | {first} "
                    f"| {_f(hot)} | {_f(float(np.nanmax(win)))} |"
                )
    return "\n".join(out) + "\n"


def effect_rows(signals: dict, events: list[dict], settle: float = 60.0, before: float = 180.0):
    """Per event: median of its own signal from settle seconds after the start to the end,
    divided by the median of the `before` seconds ahead."""
    own = {s.scenario: s.name for s in SIGNALS}
    rows = []
    for e in events:
        ts, vals = signals[own[e["scenario"]]]
        s, t_end = e["actual_start"], e["actual_end"]
        inside = vals[(ts >= s + settle) & (ts <= t_end) & np.isfinite(vals)]
        ahead = vals[(ts >= s - before) & (ts < s) & np.isfinite(vals)]
        if len(inside) and len(ahead) and np.median(ahead) != 0:
            rows.append(
                (e["scenario"], e.get("band", ""), float(np.median(inside) / np.median(ahead)))
            )
    return rows


def render_effects(rows: list[tuple[str, str, float]]) -> str:
    out = [
        "",
        "### Effect of the injection on its own signal",
        "",
        "Median during (from 60 s after the start) divided by median of the 180 s before.",
        "",
        "| scenario | band | events | ratio median | ratio min | ratio max |",
        "|---|---|---|---|---|---|",
    ]
    for sc in sorted({r[0] for r in rows}):
        for band in ("all", "peak", "trough"):
            x = [r[2] for r in rows if r[0] == sc and (band == "all" or r[1] == band)]
            if x:
                out.append(
                    f"| {sc} | {band} | {len(x)} | {np.median(x):.2f} "
                    f"| {min(x):.2f} | {max(x):.2f} |"
                )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m anomaly_detection.lab.azure_report", description=__doc__
    )
    p.add_argument("--signals", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--raw", type=Path, help="raw.csv: adds the live scores of the sidecar")
    p.add_argument(
        "--models",
        default="fixed,seasonal,rcf,awsrcf-nab,awsrcf,tspulse,toto-4m,subpca",
    )
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--for-points", type=int, default=3)
    p.add_argument("--cache-dir", type=Path, required=True, help="scores per model and signal")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--toto-dir", default="")
    p.add_argument("--tspulse-model", default="")
    p.add_argument("--out", type=Path)
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)

    doc = json.loads(args.labels.read_text())
    meta = doc["meta"]
    natural = doc.get("natural_candidates", [])
    events = [e for e in doc["events"] if e["status"] == "completed"]
    step = args.step
    signals = replay.load_signals(args.signals, step)
    own = {e["scenario"] for e in events}
    names = [s.name for s in SIGNALS if s.scenario in own]
    runs = {n: replay.build_run(n, *signals[n], events, meta, step) for n in names}
    scale = replay.DAY / meta["day_seconds"]
    rule = AlertRule(threshold=args.threshold, for_points=args.for_points)
    models = [m for m in args.models.split(",") if m]
    paths = {"toto": args.toto_dir, "tspulse": args.tspulse_model}
    scores = compute_scores(runs, models, paths, step, scale, args.cache_dir, args.workers)
    order = list(models)
    if args.raw:
        live = replay.load_live_scores(args.raw, step)
        for n in names:
            ts = runs[n].case.ts
            arr = np.full(len(ts), np.nan)
            first = round(ts[0] / step)
            for k, v in live.get(n, {}).items():
                if 0 <= k - first < len(ts):
                    arr[k - first] = v
            scores[("sidecar-live", n)] = arr
        order.append("sidecar-live")

    rows: list[replay.Row] = []
    evs: list[EventRow] = []
    fa: dict[tuple[str, str], list[str]] = {}
    for m in order:
        for n in names:
            run = runs[n]
            sc = scores[(m, n)]
            row = replay._row(m, n, run, sc, step, rule)
            rows.append(row)
            ts = run.case.ts
            fa[(m, n)] = [
                fa_band(float(ts[f]), meta)
                for f in row.result.fires
                if not any(s <= f <= e + rule.grace for s, e in run.own)
            ]
            bands = [e.get("band", "") for e in events if e["scenario"] == run.case.scenario]
            evs += event_rows(m, run, row.result, sc, bands, step, rule)

    head = (
        f"Completed events: {len(events)}. Lab day: {meta['day_seconds']:.0f}s, warm-up "
        f"{meta['warmup_days']} days, step {step}s, alert rule: score >= {args.threshold} for "
        f"{args.for_points} consecutive samples. Trace minutes are lab seconds times "
        f"{scale:.0f} / 60.\n"
    )
    text = (
        head
        + render_all(rows, evs, meta, order, scale, step, fa)
        + render_order(evs, order)
        + natural_rows(meta, natural, runs, scores, order, step, rule)
        + render_effects(effect_rows(signals, events))
    )
    print(text)
    if args.out:
        args.out.write_text(text)
    if args.json:
        args.json.write_text(json.dumps([e.__dict__ for e in evs], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
