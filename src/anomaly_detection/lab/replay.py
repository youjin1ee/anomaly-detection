"""Replay a dump of the lab through the harness models and score it against the labels.

The compressed day is stretched back to 86400 seconds before the models see it (timestamps and
step are multiplied by 86400 / day_seconds), so every model runs with its normal daily period
and defaults. The alert rule, matching and metric definitions are the harness ones
(docs/evaluation.md); times in the report are real seconds of the lab, not stretched ones.

One extra column separates false alarms that fired while another scenario was being injected
(a cross effect, for example a GPU competitor also raising client latency) from those that
fired in quiet periods.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..harness.metrics import AlertRule, CaseResult, Summary, evaluate_case, summarize
from ..harness.runner import fixed_thresholds, parse_model
from ..harness.scenarios import Case
from ..models import create_model
from ..pipeline import SeriesPipeline
from .signals import SIGNALS

DAY = 86400.0
MODEL_PARAMS = {"seasonal": {"seasonality": ["daily"]}}


def load_signals(path: Path, step: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """signal name -> (timestamps on a regular grid, values with NaN where missing)."""
    raw: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open() as f:
        for row in csv.DictReader(f):
            name = next(
                kv.split("=", 1)[1].strip('"')
                for kv in row["labels"].split(",")
                if kv.startswith("signal=")
            )
            v = float(row["value"])
            if math.isfinite(v):
                raw[name][round(float(row["timestamp"]) / step)] = v
    out = {}
    for name, points in raw.items():
        lo, hi = min(points), max(points)
        ts = np.arange(lo, hi + 1) * float(step)
        vals = np.full(len(ts), np.nan)
        for k, v in points.items():
            vals[k - lo] = v
        out[name] = (ts, vals)
    return out


def load_labels(path: Path) -> tuple[dict, list[dict]]:
    doc = json.loads(path.read_text())
    return doc["meta"], [e for e in doc["events"] if e["status"] == "completed"]


def index_of(ts: np.ndarray, t: float, step: int) -> int:
    return round((t - ts[0]) / step)


@dataclass
class SignalRun:
    case: Case
    own: list[tuple[int, int]]  # events of this signal's scenario
    others: list[tuple[int, int]]  # events of other scenarios


def build_run(
    name: str,
    ts: np.ndarray,
    vals: np.ndarray,
    events: list[dict],
    meta: dict,
    step: int,
) -> SignalRun:
    sig = next(s for s in SIGNALS if s.name == name)
    eval_start = index_of(ts, meta["eval_start"], step)
    own, others = [], []
    for e in events:
        span = (index_of(ts, e["actual_start"], step), index_of(ts, e["actual_end"], step))
        (own if e["scenario"] == sig.scenario else others).append(span)
    case = Case(
        sig.scenario,
        name,
        sig.direction,
        0,
        0,
        ts,
        vals,
        tuple(own),
        eval_start,
        vals[:eval_start].copy(),
    )
    return SignalRun(case, own, others)


def score(spec_name: str, kind: str, params: dict, run: SignalRun, step: int, scale: float):
    case = run.case
    p = dict(params)
    if kind == "static":
        p = fixed_thresholds(case.train_values, case.direction)
    # Stretch time so that one lab day is one real day for the model.
    model = create_model(kind, p, float(step) * scale, seed=1)
    pipe = SeriesPipeline(model, direction=case.direction)
    scores = np.full(len(case.ts), np.nan)
    for i, (t, v) in enumerate(zip(case.ts, case.values, strict=True)):
        if np.isnan(v):
            continue
        res = pipe.update(float(t - case.ts[0]) * scale, float(v))
        if res is not None:
            scores[i] = res.score
    return scores


@dataclass
class Row:
    model: str
    signal: str
    scenario: str
    result: CaseResult
    cross_alarms: int  # false alarms that fired during another scenario's event
    quiet_alarms: int


def evaluate(
    signals: dict[str, tuple[np.ndarray, np.ndarray]],
    meta: dict,
    events: list[dict],
    models: list[str],
    step: int,
    rule: AlertRule,
    only: list[str] | None = None,
) -> list[Row]:
    scale = DAY / meta["day_seconds"]
    rows: list[Row] = []
    for name, (ts, vals) in signals.items():
        if only and name not in only:
            continue
        run = build_run(name, ts, vals, events, meta, step)
        for m in models:
            spec = parse_model(m)
            params = {**MODEL_PARAMS.get(spec.kind, {}), **spec.params}
            scores = score(spec.name, spec.kind, params, run, step, scale)
            res = evaluate_case(scores, run.case.anomalies, run.case.eval_start, step, rule)
            matched = {f for f in res.fires for s, e in run.own if s <= f <= e + rule.grace}
            cross = sum(
                1
                for f in res.fires
                if f not in matched and any(s <= f <= e + rule.grace for s, e in run.others)
            )
            rows.append(
                Row(spec.name, name, run.case.scenario, res, cross, res.false_alarms - cross)
            )
    return rows


def _f(x: float, nd: int = 2) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def render(rows: list[Row], model_order: list[str], day_seconds: float) -> str:
    out = [
        "| scenario | signal | model | events | detected | recall | alerts | false alarms "
        "| of which during other injections | precision | F1 | delay median (s) | delay p90 (s) "
        "| false alarms per lab day |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r.scenario, model_order.index(r.model))):
        s: Summary = summarize([r.result])
        delay = lambda x: _f(x * 60, 0)  # noqa: E731  Summary reports minutes
        out.append(
            f"| {r.scenario} | {r.signal} | {r.model} | {s.events} | {s.detected} "
            f"| {_f(s.recall)} | {s.alerts} | {s.false_alarms} | {r.cross_alarms} "
            f"| {_f(s.precision)} | {_f(s.f1)} | {delay(s.median_delay_min)} "
            f"| {delay(s.p90_delay_min)} "
            f"| {_f(s.false_alarms / (r.result.eval_days * 86400 / day_seconds))} |"
        )
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anomaly-detection-replay", description=__doc__)
    p.add_argument("--signals", type=Path, required=True, help="signals.csv from the dump")
    p.add_argument("--labels", type=Path, required=True, help="labels.json from the injector")
    p.add_argument("--models", default="fixed,seasonal,rcf,rcf-time")
    p.add_argument("--step", type=int, default=5, help="seconds between samples in the dump")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--for-points", type=int, default=3)
    p.add_argument("--only", default="", help="comma separated signal names")
    p.add_argument("--out", type=Path)
    p.add_argument("--json", type=Path)
    args = p.parse_args(argv)

    meta, events = load_labels(args.labels)
    signals = load_signals(args.signals, args.step)
    models = [m for m in args.models.split(",") if m]
    rule = AlertRule(threshold=args.threshold, for_points=args.for_points)
    rows = evaluate(
        signals, meta, events, models, args.step, rule, [s for s in args.only.split(",") if s]
    )
    order = [parse_model(m).name for m in models]
    text = (
        f"Completed events: {len(events)}. Lab day: {meta['day_seconds']:.0f}s, "
        f"warm-up {meta['warmup_days']} days, step {args.step}s, alert rule: score >= "
        f"{args.threshold} for {args.for_points} consecutive samples.\n\n"
        + render(rows, order, meta["day_seconds"])
    )
    print(text)
    if args.out:
        args.out.write_text(text)
    if args.json:
        args.json.write_text(
            json.dumps(
                [
                    {
                        "model": r.model,
                        "signal": r.signal,
                        "scenario": r.scenario,
                        "events": r.result.events,
                        "detected": r.result.detected,
                        "alerts": len(r.result.fires),
                        "false_alarms": r.result.false_alarms,
                        "cross_alarms": r.cross_alarms,
                        "delays_s": r.result.delays,
                    }
                    for r in rows
                ],
                indent=1,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
