"""`anomaly-detection-eval`: replay synthetic silent failures through the models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .metrics import AlertRule
from .report import render
from .runner import parse_model, run_all


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anomaly-detection-eval", description=__doc__)
    p.add_argument(
        "--models",
        default="fixed,seasonal,rcf",
        help="comma separated; NAME=BASE:key=value,... makes a parameter variant. "
        "Bases: fixed, seasonal, rcf, rcf-time. Use ';' between variants whose "
        "parameters contain commas.",
    )
    p.add_argument("--variants", type=int, default=20, help="variants per scenario")
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--train-days", type=float, default=14)
    p.add_argument("--eval-days", type=float, default=7)
    p.add_argument("--normal-cases", type=int, default=5, help="normal series per signal")
    p.add_argument("--normal-eval-days", type=float, default=14)
    p.add_argument("--step", type=int, default=60, help="seconds between samples")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--for-points", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--scenarios", default="", help="comma separated subset, default all")
    p.add_argument("--out", type=Path, help="write the markdown report here")
    p.add_argument("--json", type=Path, help="write per-case results here")
    args = p.parse_args(argv)

    sep = ";" if ";" in args.models else ","
    specs = [parse_model(m) for m in args.models.split(sep) if m]
    rule = AlertRule(threshold=args.threshold, for_points=args.for_points)
    t0 = time.time()
    rows = run_all(
        specs,
        args.variants,
        args.seed_base,
        args.train_days,
        args.eval_days,
        args.normal_cases,
        args.normal_eval_days,
        args.step,
        rule,
        args.workers,
        [s for s in args.scenarios.split(",") if s] or None,
    )
    header = (
        f"Parameters: variants={args.variants}, seed_base={args.seed_base}, "
        f"train_days={args.train_days}, eval_days={args.eval_days}, step={args.step}s, "
        f"alert rule: score >= {args.threshold} for {args.for_points} consecutive samples, "
        f"normal series per signal={args.normal_cases} x {args.normal_eval_days} days. "
        f"Wall time {time.time() - t0:.0f}s.\n\n"
    )
    text = header + render(rows, [s.name for s in specs])
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([{**r, "result": asdict(r["result"])} for r in rows], default=float)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
