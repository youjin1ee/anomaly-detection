"""Highest model score inside each injected event, in event order, for the lab dump.

Shows whether a model keeps reacting to later events of the same kind or learns the failure
as the new normal (the first event scores high, the next ones low). Uses the replay code of
`anomaly_detection.lab.replay`, same time stretch and parameters.

    python scripts/inject_event_scores.py --signals signals.csv --labels labels.json \
        --models rcf,awsrcf-nab,awsrcf --signal-names request_rate,output_tokens
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from anomaly_detection.harness.runner import parse_model
from anomaly_detection.lab import replay as rp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--signals", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--models", default="rcf,awsrcf-nab,awsrcf")
    p.add_argument("--signal-names", default="request_rate,output_tokens")
    p.add_argument("--step", type=int, default=5)
    a = p.parse_args()
    meta, events = rp.load_labels(a.labels)
    signals = rp.load_signals(a.signals, a.step)
    scale = rp.DAY / meta["day_seconds"]
    print("| signal | model | event 1 | event 2 | event 3 | event 4 | event 5 | quiet p99 |")
    print("|---|---|---|---|---|---|---|---|")
    for name in a.signal_names.split(","):
        ts, vals = signals[name]
        run = rp.build_run(name, ts, vals, events, meta, a.step)
        spans = sorted(run.own)
        in_event = np.zeros(len(ts), bool)
        for s, e in run.own + run.others:
            in_event[s : e + 6] = True
        for m in a.models.split(","):
            spec = parse_model(m)
            params = {**rp.MODEL_PARAMS.get(spec.kind, {}), **spec.params}
            scores = rp.score(spec.name, spec.kind, params, run, a.step, scale)
            peaks = [
                float(np.nanmax(scores[s : e + 6]))
                if np.isfinite(scores[s : e + 6]).any()
                else float("nan")
                for s, e in spans
            ]
            quiet = scores[run.case.eval_start :][~in_event[run.case.eval_start :]]
            q99 = float(np.nanpercentile(quiet, 99)) if np.isfinite(quiet).any() else float("nan")
            cells = " | ".join(f"{x:.2f}" for x in peaks)
            print(f"| {name} | {spec.name} | {cells} | {q99:.2f} |")


if __name__ == "__main__":
    main()
