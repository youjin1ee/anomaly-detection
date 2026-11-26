"""`anomaly-detection-replay`: replay recorded metric dumps and count alerts per series."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

from .metrics import AlertRule
from .replay import (
    CONTEXTS,
    Guards,
    RecordedSeries,
    SeriesAlerts,
    cached_run,
    evaluate_series,
    load_dumps,
    summarize_model,
)
from .runner import ModelSpec, parse_model

Job = tuple[ModelSpec, RecordedSeries, int, int, Path | None]


def _job(job: Job) -> None:
    spec, series, step, warmup_end, cache = job
    cached_run(spec, series, step, warmup_end, cache)


def run_models(
    specs: list[ModelSpec],
    series: list[RecordedSeries],
    step: int,
    warmup_end: int,
    cache: Path,
    workers: int,
) -> None:
    """Run every model on every series and store the raw scores in the cache."""
    jobs: list[Job] = [(sp, s, step, warmup_end, cache) for sp in specs for s in series]
    jobs.sort(key=lambda j: j[0].kind != "rcf")  # slow jobs first
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=8) as pool:
        for _ in pool.imap_unordered(_job, jobs, chunksize=1):
            pass


def render(summaries: dict[tuple[str, str], dict], limit: float) -> str:
    head = (
        "| model | guards | series | scored | failed | alerts | per series per day "
        "| median | p90 | max "
        f"| series over {limit}/day | " + " | ".join(CONTEXTS) + " |"
    )
    lines = [head, "|" + "---|" * (11 + len(CONTEXTS))]
    for (model, guard), s in summaries.items():
        lines.append(
            f"| {model} | {guard} | {s['series']} | {s['series_scored']} | {s['series_failed']} | "
            f"{s['alerts']} | "
            f"{s['alerts_per_series_day']:.3f} | {s['median_per_day']:.3f} | "
            f"{s['p90_per_day']:.3f} | {s['max_per_day']:.2f} | {s['series_over_limit']} | "
            + " | ".join(str(s["contexts"][c]) for c in CONTEXTS)
            + " |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anomaly-detection-replay", description=__doc__)
    p.add_argument("paths", nargs="+", type=Path, help="dump files or directories of csv(.gz)")
    p.add_argument(
        "--models", default="fixed,seasonal,rcf,rcf-time", help="as anomaly-detection-eval"
    )
    p.add_argument("--step", type=int, default=60)
    p.add_argument("--warmup-days", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--for-points", type=int, default=3)
    p.add_argument("--limit", type=float, default=0.1, help="false alarm limit per series per day")
    p.add_argument("--min-expected-frac", type=float, default=0.05)
    p.add_argument("--tolerance-frac", type=float, default=0.05)
    p.add_argument("--tolerance-ratio", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache-dir", type=Path, required=True, help="raw model scores are kept here")
    p.add_argument("--out", type=Path, help="write the summary table here")
    p.add_argument("--json", type=Path, help="write per series alerts here")
    args = p.parse_args(argv)

    specs = [parse_model(m) for m in args.models.split(",") if m]
    series = load_dumps(args.paths, args.step)
    if not series:
        print("no series found", file=sys.stderr)
        return 1
    warmup_end = int(args.warmup_days * 86400 / args.step)
    rule = AlertRule(threshold=args.threshold, for_points=args.for_points)
    t0 = time.time()
    run_models(specs, series, args.step, warmup_end, args.cache_dir, args.workers)
    guard_sets = {
        "off": Guards(),
        "on": Guards(args.min_expected_frac, args.tolerance_frac, args.tolerance_ratio),
    }
    summaries: dict[tuple[str, str], dict] = {}
    per_series: list[dict] = []
    ts = series[0].ts
    for sp in specs:
        for gname, guards in guard_sets.items():
            rows: list[SeriesAlerts] = []
            for s in series:
                raw = cached_run(sp, s, args.step, warmup_end, args.cache_dir)
                rows.append(evaluate_series(s, raw, sp.name, args.step, warmup_end, guards, rule))
            summaries[(sp.name, gname)] = summarize_model(rows, args.limit)
            per_series += [
                {
                    "model": sp.name,
                    "guards": gname,
                    "query": r.query,
                    "labels": r.labels,
                    "observed_days": r.observed_days,
                    "scored": r.ready,
                    "error": r.error,
                    "alerts": [
                        {
                            "first": int(ts[a]),
                            "fire": int(ts[b]),
                            "last": int(ts[c]),
                            "context": ctx,
                        }
                        for (a, b, c), ctx in zip(r.spans, r.contexts, strict=True)
                    ],
                }
                for r in rows
            ]
    text = (
        f"Series {len(series)}, step {args.step}s, warmup {args.warmup_days} days, "
        f"alert rule: score >= {args.threshold} for {args.for_points} consecutive samples, "
        f"guards on = min_expected {args.min_expected_frac}, tolerance {args.tolerance_frac}, "
        f"tolerance ratio {args.tolerance_ratio}. Wall time {time.time() - t0:.0f}s.\n\n"
        + render(summaries, args.limit)
    )
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(per_series))
    return 0


if __name__ == "__main__":
    sys.exit(main())
