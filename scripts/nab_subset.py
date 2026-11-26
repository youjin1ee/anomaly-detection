"""NAB scores on the files that are long enough for TSPulse to score as early as the other models.

TSPulse needs 512 + 50 samples before its first score, so it scores from record 563 on. The
other models score from the end of the probationary period (15 percent of the file, at most
750 records). Where the probationary period is shorter than 563 the TSPulse run spends part
of the scored range warming up. This script reports the corpus score for the files where it
does not, and for the rest, next to the full corpus.

    python scripts/nab_subset.py --nab-dir NAB --scores-dir DIR --models tspulse,rcf,seasonal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from anomaly_detection.harness.nab_data import load_corpus
from anomaly_detection.harness.nab_report import PROFILE_COLUMNS, nab_numbers
from anomaly_detection.harness.nab_score import probation_length

FIRST_TSPULSE_SCORE = 563


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nab-dir", type=Path, required=True)
    ap.add_argument("--scores-dir", type=Path, required=True)
    ap.add_argument("--models", default="tspulse,rcf,seasonal")
    a = ap.parse_args()
    corpus = load_corpus(a.nab_dir)
    names = a.models.split(",")
    scores = {}
    for m in names:
        with np.load(a.scores_dir / f"{m}.npz") as z:
            scores[m] = {s.path: z[s.path.replace("/", "__")] for s in corpus}
    long_ = [s for s in corpus if probation_length(len(s.values)) >= FIRST_TSPULSE_SCORE]
    short = [s for s in corpus if s not in long_]
    print(
        f"files: {len(corpus)}, probation >= {FIRST_TSPULSE_SCORE}: {len(long_)}, "
        f"shorter: {len(short)}"
    )
    print(
        f"windows: long {sum(len(s.windows) for s in long_)}, "
        f"short {sum(len(s.windows) for s in short)}"
    )
    # scored records and records in scored range of the other models, per group
    for label, group in (("long", long_), ("short", short)):
        scored = {m: sum(int(np.isfinite(scores[m][s.path]).sum()) for s in group) for m in names}
        print(f"scored records {label}: {scored}")
    print()
    print(
        "| subset | model | standard | threshold | reward_low_FP_rate | threshold "
        "| reward_low_FN_rate | threshold |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for label, group in (
        ("all 58", corpus),
        (f"probation >= {FIRST_TSPULSE_SCORE}", long_),
        ("shorter", short),
    ):
        for m in names:
            n = nab_numbers(group, {s.path: scores[m][s.path] for s in group})
            cells = [f"{n.optimized[p][1]:.1f} | {n.optimized[p][0]:.4f}" for p in PROFILE_COLUMNS]
            print(f"| {label} ({len(group)} files) | {m} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
