"""Score the injection lab dump with TSPulse and report it like the lab replay does.

Needs the `anomaly_detection.lab` package of the injection lab branch (it holds the labels,
signal definitions and the replay report), and the tspulse extra.

    TSPULSE_MODEL_DIR=/path/to/model python scripts/inject_tspulse_eval.py \
        --signals run1/signals.csv --labels run1/labels.json --out tspulse-replay.md
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from anomaly_detection.harness.metrics import AlertRule
from anomaly_detection.lab import replay as lab
from anomaly_detection.models.tspulse import TSPulseModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--scores", type=Path, help="write the raw scores per signal here (npz)")
    a = ap.parse_args()

    meta, events = lab.load_labels(a.labels)
    signals = lab.load_signals(a.signals, a.step)
    rule = AlertRule(threshold=0.5, for_points=3)
    rows = []
    kept = {}
    for name, (ts, vals) in signals.items():
        run = lab.build_run(name, ts, vals, events, meta, a.step)
        ok = np.flatnonzero(np.isfinite(vals))
        model = TSPulseModel(model_path=os.environ["TSPULSE_MODEL_DIR"], threads=a.threads)
        outs, heads = model.score_offline(vals[ok])
        variants = {"tspulse": [np.nan if o is None else o.score for o in outs]}
        for h in ("time", "fft", "pred"):
            variants[f"tspulse-{h}"] = heads[h]
        # The pipeline applies the signal's direction with the model's expected value; every
        # variant uses the expected value of the full model, so the heads stay comparable.
        exp = np.full(len(ts), np.nan)
        exp[ok] = [np.nan if o is None else o.expected for o in outs]
        dev = vals - exp
        wrong = dev >= 0 if run.case.direction == "below" else dev <= 0
        for vname, sc in variants.items():
            scores = np.full(len(ts), np.nan)
            scores[ok] = sc
            kept[f"{vname}|{name}"] = scores.copy()
            scores[wrong & np.isfinite(scores)] = 0.0
            rows.append(lab._row(vname, name, run, scores, a.step, rule))
    order = ["tspulse", "tspulse-time", "tspulse-fft", "tspulse-pred"]
    text = (
        f"Completed events: {len(events)}. Lab day: {meta['day_seconds']:.0f}s, step {a.step}s.\n\n"
        + lab.render(rows, order, meta["day_seconds"])
        + "\nAll scenarios together:\n\n"
        + lab.render_pooled(rows, order, meta["day_seconds"])
    )
    print(text)
    if a.out:
        a.out.write_text(text)
    if a.scores:
        np.savez_compressed(a.scores, **{k.replace("|", "__"): v for k, v in kept.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
