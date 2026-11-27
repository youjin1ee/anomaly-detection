"""`anomaly-detection-boom`: replay the Datadog BOOM subset and count alerts per series per day.

    anomaly-detection-boom /path/to/boom --cache-dir cache --out-dir out --tag all \
        --models fixed,seasonal,subpca --workers 4
    anomaly-detection-boom /path/to/boom --cache-dir cache --out-dir out --tag subset \
        --max-days 7 --subset 24,48

`--subset N1,N5` takes N1 series at 1 minute and N5 at 5 minutes (round robin
over the strata). Models run one after the other, each on a pool of `--workers` processes, so
the wall time of every model is its own. Raw scores are cached, so guards, rules and
calibration can be re-evaluated without running the models again.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from .boom_data import BoomMeta, load_boom
from .boom_replay import (
    LIMIT,
    MODELS,
    WARMUP_DAYS,
    calibration_rows,
    concurrency,
    group_spread,
    model_spec,
    run_boom,
    spread,
    stratified_subset,
)
from .metrics import AlertRule
from .replay import CONTEXTS, Guards, RecordedSeries, cached_run, evaluate_series
from .runner import ModelSpec

Job = tuple[ModelSpec, RecordedSeries, int, int, Path]


def _job(job: Job) -> float:
    spec, series, step, warmup_end, cache = job
    return cached_run(spec, series, step, warmup_end, cache, run=run_boom).seconds


def run_model_pool(
    spec: ModelSpec,
    series: list[RecordedSeries],
    meta: dict[str, BoomMeta],
    cache: Path,
    workers: int,
) -> tuple[float, float]:
    """Replay every series with one model. Returns wall seconds and summed per series seconds."""
    jobs: list[Job] = []
    for s in series:
        step = meta[s.key].step
        jobs.append((spec, s, step, int(WARMUP_DAYS * 86400 / step), cache))
    jobs.sort(key=lambda j: -len(j[1].values))  # long series first keeps the pool busy
    t0 = time.time()
    total = 0.0
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=64) as pool:
        for i, sec in enumerate(pool.imap_unordered(_job, jobs, chunksize=1), 1):
            total += sec
            if i % 25 == 0 or i == len(jobs):
                print(f"{spec.name} {i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return time.time() - t0, total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anomaly-detection-boom", description=__doc__)
    p.add_argument("root", type=Path, help="directory of the BOOM subset (MANIFEST.csv, ds-*)")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--max-days", type=float, help="keep only the first days of every series")
    p.add_argument("--subset", help="series at 1 minute and at 5 minutes, for example 24,48")
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tag", required=True, help="name of the output files")
    p.add_argument("--budget", type=float, default=LIMIT, help="false alarms per series per day")
    p.add_argument("--cal-days", type=float, default=2.0, help="calibration days after warmup")
    p.add_argument("--skip-run", action="store_true", help="evaluate cached scores only")
    args = p.parse_args(argv)

    series, meta = load_boom(args.root, args.max_days, rebase_days=WARMUP_DAYS)
    if args.subset:
        n1, n5 = (int(x) for x in args.subset.split(","))
        series = stratified_subset(series, meta, {"T": n1, "5T": n5}, args.subset_seed)
    print(f"{len(series)} series, {sum(len(s.values) for s in series)} points", flush=True)
    by_key = {s.key: s for s in series}
    steps = {s.key: meta[s.key].step for s in series}
    specs = [model_spec(m) for m in args.models.split(",") if m]
    rule = AlertRule()
    guard_sets = {"off": Guards(), "on": Guards(0.05, 0.05, 0.05)}
    result: dict = {
        "series": len(series),
        "points": int(sum(len(s.values) for s in series)),
        "args": {k: str(v) for k, v in vars(args).items()},
        "models": {},
    }
    for sp in specs:
        wall, cpu = (
            (0.0, 0.0)
            if args.skip_run
            else run_model_pool(sp, series, meta, args.cache_dir, args.workers)
        )
        entry: dict = {"wall_seconds": wall, "series_seconds": cpu, "guards": {}}
        raws = {
            s.key: cached_run(
                sp,
                s,
                steps[s.key],
                int(WARMUP_DAYS * 86400 / steps[s.key]),
                args.cache_dir,
                run=run_boom,
            )
            for s in series
        }
        if args.skip_run:
            entry["series_seconds"] = sum(r.seconds for r in raws.values())
        for gname, guards in guard_sets.items():
            rows = [
                evaluate_series(
                    s,
                    raws[s.key],
                    sp.name,
                    steps[s.key],
                    int(WARMUP_DAYS * 86400 / steps[s.key]),
                    guards,
                    rule,
                )
                for s in series
            ]
            ready = [r for r in rows if r.ready]
            ctx = {c: sum(r.contexts.count(c) for r in rows) for c in CONTEXTS}
            first = {k: int(WARMUP_DAYS * 86400 / v) for k, v in steps.items()}
            entry["guards"][gname] = {
                "scored": len(ready),
                "failed": sum(1 for r in rows if r.error),
                "alerts": sum(len(r.spans) for r in rows),
                "pooled_per_series_day": sum(len(r.spans) for r in ready)
                / max(sum(r.observed_days for r in ready), 1e-9),
                "overall": spread(np.array([r.per_day for r in ready])),
                "by_resolution": group_spread(rows, meta, lambda m: m.freq),
                "by_metric_type": group_spread(rows, meta, lambda m: m.metric_type),
                "by_domain": group_spread(rows, meta, lambda m: m.domain),
                "contexts": ctx,
                "concurrency": concurrency(rows, by_key, first),
                "per_series": [
                    {
                        "key": r.key,
                        "per_day": r.per_day,
                        "alerts": len(r.spans),
                        "ready": r.ready,
                        "error": r.error,
                    }
                    for r in rows
                ],
            }
            # calibration on the stretch after the warmup, evaluation after that
            entry["guards"][gname]["calibration"] = {}
            for floor in (0.0, 0.5):
                cal = calibration_rows(
                    series,
                    raws,
                    steps,
                    lambda st: int(WARMUP_DAYS * 86400 / st),
                    guards,
                    rule,
                    args.budget,
                    args.cal_days,
                    floor,
                )
                usable = [c for c in cal if c["calibrated"] and c["base"].ready]
                entry["guards"][gname]["calibration"][str(floor)] = {
                    "series": len(usable),
                    "calibrated": spread(np.array([c["cal"].per_day for c in usable])),
                    "plain_rule": spread(np.array([c["base"].per_day for c in usable])),
                    "pooled_calibrated": sum(len(c["cal"].spans) for c in usable)
                    / max(sum(c["cal"].observed_days for c in usable), 1e-9),
                    "pooled_plain": sum(len(c["base"].spans) for c in usable)
                    / max(sum(c["base"].observed_days for c in usable), 1e-9),
                    "threshold_median": float(np.median([c["threshold"] for c in usable]))
                    if usable
                    else float("nan"),
                    "at_floor": sum(1 for c in usable if c["threshold"] <= floor + 1e-9),
                    # no threshold up to 1 keeps the budget on the calibration stretch
                    "silenced": sum(1 for c in usable if c["threshold"] > 1.0),
                }
        result["models"][sp.name] = entry
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.tag}.json").write_text(json.dumps(result, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
