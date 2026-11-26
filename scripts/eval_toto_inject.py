"""Score a finished fault-injection lab dump with Toto 2.0 next to fixed, seasonal and rcf.

Reads `signals.csv` and `labels.json` of a lab run (read only) and applies the harness alert
rule and event matching. Each signal has the direction of its scenario (request rate, output
length and decode speed below, client latency above). Models other than Toto see the stretched
time of the lab (one lab day as 86400 s), as the lab replay does, so seasonal runs with its
daily defaults; Toto is not time aware and sees the 5 second samples as they are.

    CUDA_VISIBLE_DEVICES= python scripts/eval_toto_inject.py RUN_DIR --toto-dir MODELS/toto \
        --sizes 4m,22m --out inject.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from anomaly_detection.harness.metrics import AlertRule, evaluate_case, summarize
from anomaly_detection.harness.runner import fixed_thresholds
from anomaly_detection.models import create_model
from anomaly_detection.models.toto import get_engine, score_batch
from anomaly_detection.pipeline import SeriesPipeline, apply_guards

STEP = 5
DAY = 86400.0
SIGNALS = {  # signal -> (scenario, direction)
    "request_rate": ("request_drop", "below"),
    "output_tokens": ("output_length_drop", "below"),
    "decode_speed": ("throughput_drop", "below"),
    "client_latency": ("latency_rise", "above"),
}


def load_signals(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    raw: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open() as f:
        for row in csv.DictReader(f):
            name = re.search(r'signal="([^"]*)"', row["labels"]).group(1)
            v = float(row["value"])
            if math.isfinite(v):
                raw[name][round(float(row["timestamp"]) / STEP)] = v
    out = {}
    for name, pts in raw.items():
        lo, hi = min(pts), max(pts)
        ts = np.arange(lo, hi + 1) * float(STEP)
        vals = np.full(len(ts), np.nan)
        for k, v in pts.items():
            vals[k - lo] = v
        out[name] = (ts, vals)
    return out


def idx(ts: np.ndarray, t: float) -> int:
    return round((t - ts[0]) / STEP)


def score_stream_model(kind, params, ts, vals, direction, scale):
    model = create_model(kind, params, STEP * scale, seed=1)
    pipe = SeriesPipeline(model, direction=direction)
    scores = np.full(len(ts), np.nan)
    for i in range(len(ts)):
        if np.isnan(vals[i]):
            continue
        res = pipe.update(float(ts[i] - ts[0]) * scale, float(vals[i]))
        if res is not None:
            scores[i] = res.score
    return scores


def score_toto(engine, ts, vals, direction):
    keep = np.flatnonzero(np.isfinite(vals))
    res = score_batch(engine, vals[keep])
    scores = np.full(len(ts), np.nan)
    for k, i in enumerate(keep):
        if np.isfinite(res["score"][k]):
            scores[i] = apply_guards(
                float(res["score"][k]), float(vals[i]), float(res["expected"][k]), direction
            )
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--toto-dir", type=Path, required=True)
    ap.add_argument("--sizes", default="4m,22m")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--toto-thresholds",
        default="0.3,0.2,0.1",
        help="extra alert thresholds for Toto only, an exploratory sensitivity table",
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    doc = json.loads((args.run_dir / "labels.json").read_text())
    meta = doc["meta"]
    events = [e for e in doc["events"] if e["status"] == "completed"]
    signals = load_signals(args.run_dir / "signals.csv")
    scale = DAY / meta["day_seconds"]
    rule = AlertRule()
    engines = {s: get_engine(args.toto_dir / s, args.threads) for s in args.sizes.split(",") if s}
    models = ["fixed", "seasonal", "rcf", *[f"toto-{s}" for s in engines]]

    lines = [
        f"Completed events: {len(events)}. Lab day {meta['day_seconds']:.0f} s, warm-up "
        f"{meta['warmup_days']} days, step {STEP} s, alert rule: score >= {rule.threshold} "
        f"for {rule.for_points} consecutive samples.",
        "",
        "| scenario | signal | model | events | detected | recall | alerts | false alarms "
        "| precision | F1 | delay median (s) | delay p90 (s) | false alarms per lab day |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def f(x: float, nd: int = 2) -> str:
        return "n/a" if x != x else f"{x:.{nd}f}"

    totals = {m: [] for m in models}
    toto_scores: dict[tuple[str, str], tuple[np.ndarray, tuple, int]] = {}
    for name, (scenario, direction) in SIGNALS.items():
        if name not in signals:
            continue
        ts, vals = signals[name]
        eval_start = idx(ts, meta["eval_start"])
        own = tuple(
            (idx(ts, e["actual_start"]), idx(ts, e["actual_end"]))
            for e in events
            if e["scenario"] == scenario
        )
        for m in models:
            if m == "fixed":
                p = fixed_thresholds(vals[:eval_start].copy(), direction)
                s = score_stream_model("static", p, ts, vals, direction, scale)
            elif m == "seasonal":
                p = {"seasonality": ["daily"]}
                s = score_stream_model("seasonal", p, ts, vals, direction, scale)
            elif m == "rcf":
                s = score_stream_model("rcf", {}, ts, vals, direction, scale)
            else:
                s = score_toto(engines[m.removeprefix("toto-")], ts, vals, direction)
            if m.startswith("toto-"):
                toto_scores[(m, name)] = (s, own, eval_start)
            res = evaluate_case(s, own, eval_start, STEP, rule)
            totals[m].append(res)
            sm = summarize([res])
            lab_days = res.eval_days * DAY / meta["day_seconds"]
            lines.append(
                f"| {scenario} | {name} | {m} | {sm.events} | {sm.detected} | {f(sm.recall)} "
                f"| {sm.alerts} | {sm.false_alarms} | {f(sm.precision)} | {f(sm.f1)} "
                f"| {f(sm.median_delay_min * 60, 0)} | {f(sm.p90_delay_min * 60, 0)} "
                f"| {f(sm.false_alarms / lab_days)} |"
            )
    lines += [
        "",
        "All four signals together:",
        "",
        "| model | events | detected | recall | alerts | false alarms | precision | F1 "
        "| delay median (s) | false alarms per lab day per signal |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in models:
        sm = summarize(totals[m])
        lab_days = sm.eval_days * DAY / meta["day_seconds"]
        lines.append(
            f"| {m} | {sm.events} | {sm.detected} | {f(sm.recall)} | {sm.alerts} "
            f"| {sm.false_alarms} | {f(sm.precision)} | {f(sm.f1)} "
            f"| {f(sm.median_delay_min * 60, 0)} | {f(sm.false_alarms / lab_days)} |"
        )
    extra = [float(x) for x in args.toto_thresholds.split(",") if x]
    if extra:
        lines += [
            "",
            "Toto at other alert thresholds (exploratory, chosen after seeing the scores, so "
            "optimistic). All four signals together:",
            "",
            "| model | threshold | events | detected | recall | alerts | false alarms | precision "
            "| F1 | false alarms per lab day per signal |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for m in [m for m in models if m.startswith("toto-")]:
            for thr in extra:
                rs = [
                    evaluate_case(sc, own, es, STEP, AlertRule(threshold=thr))
                    for (mm, _), (sc, own, es) in toto_scores.items()
                    if mm == m
                ]
                sm = summarize(rs)
                lab_days = sm.eval_days * DAY / meta["day_seconds"]
                lines.append(
                    f"| {m} | {thr} | {sm.events} | {sm.detected} | {f(sm.recall)} | {sm.alerts} "
                    f"| {sm.false_alarms} | {f(sm.precision)} | {f(sm.f1)} "
                    f"| {f(sm.false_alarms / lab_days)} |"
                )
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
