"""Check the reimplemented scorer against the official results shipped in the NAB repository.

The NAB repository keeps, per detector, the anomaly score of every record
(`results/<detector>/<category>/<detector>_<file>.csv`), the thresholds the official optimiser
found (`config/thresholds.json`) and the normalised scores (`results/final_results.json`).
Feeding the same anomaly scores to `nab_score` has to give the same threshold and score.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .nab_data import NabSeries
from .nab_score import (
    PROFILES,
    best_threshold,
    build_events,
    normalize,
    null_and_perfect,
    raw_at,
)


def official_scores(results_dir: Path, detector: str, series: NabSeries) -> np.ndarray:
    cat, fname = series.path.split("/", 1)
    path = results_dir / detector / cat / f"{detector}_{fname}"
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = header.index("anomaly_score")
        values = [float(row[col]) for row in reader]
    if len(values) != len(series.values):
        raise ValueError(f"{path}: {len(values)} rows, data file has {len(series.values)}")
    return np.array(values)


@dataclass
class Check:
    detector: str
    profile: str
    threshold: float
    official_threshold: float
    score: float
    official_score: float
    score_at_official_threshold: float


def verify(corpus: list[NabSeries], nab_dir: Path, detectors: list[str]) -> list[Check]:
    results_dir = nab_dir / "results"
    final = json.loads((results_dir / "final_results.json").read_text())
    thresholds = json.loads((nab_dir / "config" / "thresholds.json").read_text())
    windows = [s.windows for s in corpus]
    out: list[Check] = []
    for det in detectors:
        arrays = [official_scores(results_dir, det, s) for s in corpus]
        for pname, profile in PROFILES.items():
            events = build_events(arrays, windows, profile)
            thr, raw = best_threshold(events)
            null, perfect = null_and_perfect(events, profile)
            official_thr = float(thresholds[det][pname]["threshold"])
            out.append(
                Check(
                    det,
                    pname,
                    thr,
                    official_thr,
                    normalize(raw, null, perfect),
                    float(final[det][pname]),
                    normalize(raw_at(events, official_thr), null, perfect),
                )
            )
    return out


def render(checks: list[Check]) -> str:
    lines = [
        "| detector | profile | official score | ours at the official threshold | difference "
        "| ours at our best threshold | our threshold | official threshold |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in checks:
        lines.append(
            f"| {c.detector} | {c.profile} | {c.official_score:.4f} "
            f"| {c.score_at_official_threshold:.4f} "
            f"| {c.score_at_official_threshold - c.official_score:+.2e} "
            f"| {c.score:.4f} | {c.threshold:.6f} | {c.official_threshold:.6f} |"
        )
    return "\n".join(lines) + "\n"
