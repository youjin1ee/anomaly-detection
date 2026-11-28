"""NAB score, reimplemented from the published method (no NAB code is copied).

Method as described in the NAB paper (Lavin and Ahmad 2015, Ahmad et al. 2017) and checked
against the behaviour of the reference implementation (see docs/results/m2-nab.md for how the
reimplementation was compared with the official results):

- Every labelled anomaly has a window. A detector answers with a score in [0, 1] per record,
  and one threshold per detector and profile turns scores into detections.
- Within a window only the best detection counts, which is the earliest one: the weight is
  `2 * sigmoid(-5 * position) - 1` with position -1 at the left edge and 0 at the right edge.
  A window without detection costs `-fn_weight`.
- A detection outside any window is a false positive. Its weight is the same sigmoid with the
  position measured past the right edge of the previous window in units of that window's
  width, times `fp_weight`. Beyond 3 window widths, or before the first window, it is -1.
- The first `min(15 percent of the records, 750)` records of each file are a probationary
  period and are ignored.
- Raw score is the sum over the corpus. It is normalised so that the null detector (no
  detections) scores 0 and a perfect detector (every window detected at its left edge, no
  false positive) scores 100: `100 * (raw - null) / (perfect - null)`.
- The threshold is the single value, per profile and per detector, that maximises the raw
  score over the whole corpus. This is what the official `--optimize` step does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

PROBATION_PERCENT = 0.15
PROBATION_CAP = PROBATION_PERCENT * 5000  # 750 records


@dataclass(frozen=True)
class Profile:
    name: str
    tp_weight: float
    fn_weight: float
    fp_weight: float


PROFILES: dict[str, Profile] = {
    "standard": Profile("standard", 1.0, 1.0, 0.11),
    "reward_low_FP_rate": Profile("reward_low_FP_rate", 1.0, 1.0, 0.22),
    "reward_low_FN_rate": Profile("reward_low_FN_rate", 1.0, 2.0, 0.11),
}


def scaled_sigmoid(x: np.ndarray | float) -> np.ndarray:
    """2 * sigmoid(-5x) - 1, and -1 for x above 3."""
    x = np.asarray(x, dtype=float)
    out = 2.0 / (1.0 + np.exp(5.0 * np.minimum(x, 3.0))) - 1.0
    return np.where(x > 3.0, -1.0, out)


_MAX_TP = float(scaled_sigmoid(-1.0))


def probation_length(n: int) -> int:
    return int(min(math.floor(PROBATION_PERCENT * n), PROBATION_CAP))


class FileEvents:
    """Everything needed to score one file at any threshold, for one profile.

    `ev_scores` and `ev_deltas` are the steps of the raw score as the threshold falls: a
    false positive adds its (negative) weight, a window rises from `-fn_weight` to the
    best weight among the detections inside it.
    """

    def __init__(
        self,
        anomaly_scores: np.ndarray,
        windows: Sequence[tuple[int, int]],
        profile: Profile,
    ):
        n = len(anomaly_scores)
        scores = np.nan_to_num(np.asarray(anomaly_scores, dtype=float), nan=0.0)
        start_scored = probation_length(n)
        idx = np.arange(n)
        win = np.full(n, -1)
        prev_right = np.full(n, -1)
        prev_width = np.zeros(n)
        for k, (s, e) in enumerate(windows):
            win[s : e + 1] = k
            prev_right[e + 1 :] = e
            prev_width[e + 1 :] = e - s + 1
        in_win = win >= 0

        weight = np.full(n, np.nan)
        # Outside a window.
        out = ~in_win
        has_prev = out & (prev_right >= 0)
        weight[out] = -1.0 * profile.fp_weight
        pos_past = np.abs(prev_right[has_prev] - idx[has_prev]) / (prev_width[has_prev] - 1.0)
        weight[has_prev] = scaled_sigmoid(pos_past) * profile.fp_weight
        # Inside a window.
        for s, e in windows:
            width = float(e - s + 1)
            pos = -(e - idx[s : e + 1] + 1) / width
            weight[s : e + 1] = scaled_sigmoid(pos) * profile.tp_weight / _MAX_TP

        scored = idx >= start_scored
        fp = scored & out
        ev_s = [scores[fp]]
        ev_d = [weight[fp]]
        self.base = 0.0
        self.n_windows = 0  # windows that have scored records
        self.n_windows_total = len(windows)
        for s, e in windows:
            if e < start_scored:
                continue  # the whole window is probationary: it does not exist for scoring
            lo = max(s, start_scored)
            ws = scores[lo : e + 1]
            wv = weight[lo : e + 1]
            order = np.argsort(-ws, kind="stable")
            cur = np.maximum.accumulate(wv[order])
            cur = np.maximum(cur, -profile.fn_weight)
            ev_s.append(ws[order])
            ev_d.append(np.diff(cur, prepend=-profile.fn_weight))
            self.base -= profile.fn_weight
            self.n_windows += 1
        self.ev_scores = np.concatenate(ev_s)
        self.ev_deltas = np.concatenate(ev_d)

    def raw(self, threshold: float) -> float:
        """Raw score when every record with anomaly score >= threshold is a detection."""
        return self.base + float(self.ev_deltas[self.ev_scores >= threshold].sum())


def build_events(
    scores: Sequence[np.ndarray],
    windows: Sequence[Sequence[tuple[int, int]]],
    profile: Profile,
) -> list[FileEvents]:
    return [FileEvents(s, w, profile) for s, w in zip(scores, windows, strict=True)]


def threshold_curve(files: Sequence[FileEvents]) -> tuple[np.ndarray, np.ndarray]:
    """Candidate thresholds (descending, starting with 1.1 = no detection) and the raw score
    of the corpus at each."""
    s = np.concatenate([f.ev_scores for f in files])
    d = np.concatenate([f.ev_deltas for f in files])
    base = sum(f.base for f in files)
    order = np.argsort(-s, kind="stable")
    s, d = s[order], d[order]
    cum = np.cumsum(d)
    last = np.append(s[1:] != s[:-1], True) if len(s) else np.zeros(0, dtype=bool)
    thresholds = np.concatenate([[1.1], s[last]])
    totals = np.concatenate([[base], base + cum[last]])
    return thresholds, totals


def best_threshold(files: Sequence[FileEvents]) -> tuple[float, float]:
    """(threshold, raw score) with the highest raw score. Ties go to the higher threshold."""
    thresholds, totals = threshold_curve(files)
    i = int(np.argmax(totals))
    return float(thresholds[i]), float(totals[i])


def best_detecting_threshold(files: Sequence[FileEvents]) -> tuple[float, float]:
    """Like `best_threshold` but never the no-detection threshold, so it shows how close a
    detector gets to the null detector when the official optimum is the null itself."""
    thresholds, totals = threshold_curve(files)
    if len(thresholds) < 2:
        return float(thresholds[0]), float(totals[0])
    i = 1 + int(np.argmax(totals[1:]))
    return float(thresholds[i]), float(totals[i])


def raw_at(files: Sequence[FileEvents], threshold: float) -> float:
    return sum(f.raw(threshold) for f in files)


def null_and_perfect(files: Sequence[FileEvents], profile: Profile) -> tuple[float, float]:
    """Raw score of the null detector and of a perfect detector."""
    # Like the official normalisation, perfect counts every labelled window of the files.
    return sum(f.base for f in files), sum(f.n_windows_total for f in files) * profile.tp_weight


def normalize(raw: float, null: float, perfect: float) -> float:
    return 100.0 * (raw - null) / (perfect - null)
