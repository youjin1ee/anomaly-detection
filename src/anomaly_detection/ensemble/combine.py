"""Ways to turn the surprise of several models into one.

All inputs are surprise values (`tail.ScoreCalibrator.surprise`), which share one scale, so
they can be compared and averaged. NaN means the model gave nothing for the sample.

- `max`: any model alone raises the score (OR).
- `mean`: the average of the models that gave a score.
- `vote`: the m-th highest, so the score is high only when at least m models are.
- routing: the combination and models depend on the kind of the metric (`ROUTES`).
"""

from __future__ import annotations

import numpy as np

# kind -> (combination, vote count, members). Chosen from the single model results of
# docs/results (fixed and awsrcf find drops of request metrics, awsrcf-nab and fixed alert least
# on latency and error metrics, subpca and awsrcf-nab follow resource metrics) before this
# evaluation and not changed after it.
ROUTES: dict[str, tuple[str, int, tuple[str, ...]]] = {
    "request": ("max", 1, ("fixed", "awsrcf")),
    "latency": ("vote", 2, ("awsrcf-nab", "subpca", "fixed")),
    "error": ("max", 1, ("fixed", "awsrcf-nab")),
    "resource": ("max", 1, ("subpca", "awsrcf-nab")),
    "other": ("max", 1, ("awsrcf-nab", "subpca", "fixed")),
}


def combine(xs: list[np.ndarray], how: str, vote: int = 2) -> np.ndarray:
    """Combine surprise arrays of equal length. Empty input gives an empty (NaN) result."""
    if not xs:
        return np.array([])
    a = np.stack(xs)  # (k, n)
    valid = np.isfinite(a)
    count = valid.sum(axis=0)
    if how == "max":
        out = np.where(valid, a, -np.inf).max(axis=0)
        need = 1
    elif how == "mean":
        out = np.where(valid, a, 0.0).sum(axis=0) / np.maximum(count, 1)
        need = 1
    elif how == "vote":
        srt = np.sort(np.where(valid, a, -np.inf), axis=0)[::-1]  # descending
        out = srt[min(vote, len(xs)) - 1]
        need = min(vote, len(xs))
    else:
        raise ValueError(f"unknown combination {how!r}")
    return np.where(count >= need, out, np.nan)
