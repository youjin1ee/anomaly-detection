"""Fast versions of the alert rule of `harness/metrics.py` (same results, tested against it).

`alert_spans` there walks every sample in Python. The ensemble evaluation runs the rule tens of
times over millions of samples, so the runs are found with numpy and only the runs are looped.
"""

from __future__ import annotations

import numpy as np

from ..harness.metrics import AlertRule


def _runs(hot: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], hot.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(s), int(e) - 1) for s, e in zip(starts, ends, strict=True)]


def fast_alert_spans(
    scores: np.ndarray, rule: AlertRule, first_index: int = 0
) -> list[tuple[int, int, int]]:
    """(first hot sample, firing sample, last hot sample) per alert. Same as `alert_spans`."""
    hot = np.nan_to_num(np.asarray(scores, dtype=float), nan=-1.0) >= rule.threshold
    merged: list[tuple[int, int, list[tuple[int, int]]]] = []
    for s, e in _runs(hot):
        if merged and s - merged[-1][1] - 1 <= rule.merge_gap:
            merged[-1] = (merged[-1][0], e, merged[-1][2] + [(s, e)])
        else:
            merged.append((s, e, [(s, e)]))
    spans = []
    for first, last, parts in merged:
        for s, e in parts:
            if e - s + 1 >= rule.for_points:
                fire = s + rule.for_points - 1
                fire = max(fire, first_index) if last >= first_index else fire
                if fire >= first_index:
                    spans.append((first, fire, last))
                break
    return spans


def hold(x: np.ndarray, points: int) -> np.ndarray:
    """Smallest of the last `points` values, NaN counting as 0. Thresholding this at any level
    is the `for: points` clause of an alert rule."""
    v = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    out = v.copy()
    for k in range(1, points):
        out[k:] = np.minimum(out[k:], v[:-k])
        out[:k] = np.minimum(out[:k], 0.0)
    return out
