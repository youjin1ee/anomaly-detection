"""Score calibration and budget thresholds.

Two small tail estimators, both numpy only.

`ScoreCalibrator` turns the raw scores of one model on one series into a surprise
`x = -log10 p`, where `p` is the probability that the series scores at least this high on a
normal day. The body (up to the 90th percentile of the calibration scores) is empirical. Above
it a generalised Pareto tail is fitted to the excesses (probability weighted moments, Hosking
and Wallis 1987), so a score above everything seen in a short calibration window still gets a
finite `x` that grows with the score instead of the same value as the maximum. The shape is
clamped to [0, 0.5]: bounded tails (scores are capped at 1) would need a negative shape, but
a heavier tail only makes the extrapolated `p` larger, which is the safe side for alerts.

`budget_threshold` picks the threshold of one alert stage: the smallest value at which the
combined score alerts at most `budget` times per day. The alert counts of a short calibration
window are too coarse for a budget below one alert per window, and the maximum of a short window
does not represent the tail (BOOM, docs/results/m4-boom.md), so the peaks of the alert clusters
are fitted with an exponential tail (peaks over threshold) and the threshold is read from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

X_CAP = 8.0  # surprise is capped here (p = 1e-8)
BODY_QUANTILE = 0.9
MIN_TAIL_POINTS = 20
SHAPE_MAX = 0.5


def fit_gpd_pwm(excess: np.ndarray) -> tuple[float, float]:
    """(shape xi, scale sigma) of a generalised Pareto fitted to positive excesses by
    probability weighted moments, xi clamped to [0, SHAPE_MAX]."""
    y = np.sort(np.asarray(excess, dtype=float))
    n = len(y)
    a0 = float(y.mean())
    a1 = float(np.mean((1.0 - (np.arange(1, n + 1) - 0.35) / n) * y))
    denom = a0 - 2.0 * a1
    xi = 2.0 - a0 / denom if denom > 1e-300 else SHAPE_MAX
    xi = min(max(xi, 0.0), SHAPE_MAX)
    return xi, a0 * (1.0 - xi)


def _gpd_survival(y: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    if xi < 1e-9:
        return np.exp(-y / sigma)
    return np.power(1.0 + xi * y / sigma, -1.0 / xi)


@dataclass(frozen=True)
class ScoreCalibrator:
    """Maps raw scores of one model on one series to surprise. `ok` is False when the
    calibration window had too few scores; such a model must be left out for the series."""

    cal: np.ndarray  # sorted finite calibration scores
    u0: float = 0.0
    p_u0: float = 0.0
    xi: float = 0.0
    sigma: float = 0.0
    tail: bool = False

    @property
    def ok(self) -> bool:
        return len(self.cal) > 0

    @classmethod
    def fit(
        cls, scores: np.ndarray, min_points: int = 100, extrapolate: bool = True
    ) -> ScoreCalibrator:
        s = np.sort(scores[np.isfinite(scores)])
        if len(s) < min_points:
            return cls(np.empty(0))
        if not extrapolate:
            return cls(s)
        u0 = float(np.quantile(s, BODY_QUANTILE))
        excess = s[s > u0] - u0
        if len(excess) < MIN_TAIL_POINTS or excess.max() - excess.min() < 1e-12:
            return cls(s)  # discrete or too few scores: empirical body only
        xi, sigma = fit_gpd_pwm(excess)
        return cls(s, u0, len(excess) / (len(s) + 1), xi, sigma, True)

    def surprise(self, raw: np.ndarray) -> np.ndarray:
        """x = -log10 p, NaN where raw is NaN. p is at least 1e-8 and, without a fitted tail,
        at least 1/(n + 1) (the rank of a score at or above the calibration maximum)."""
        raw = np.asarray(raw, dtype=float)
        x = np.full(raw.shape, np.nan)
        ok = np.isfinite(raw)
        r = raw[ok]
        n = len(self.cal)
        count_ge = n - np.searchsorted(self.cal, r, side="left")
        p = (count_ge + 1.0) / (n + 1.0)
        if self.tail:
            above = r > self.u0
            p[above] = self.p_u0 * _gpd_survival(r[above] - self.u0, self.xi, self.sigma)
        x[ok] = np.minimum(-np.log10(np.maximum(p, 10.0**-X_CAP)), X_CAP)
        return x


def alert_peaks(h: np.ndarray, x_min: float, merge_gap: int) -> np.ndarray:
    """Peak of every cluster of the held score `h` at or above `x_min`, clusters being runs
    closer than `merge_gap + 1` samples (the merge rule of the alert), descending."""
    hot = h >= x_min
    d = np.diff(np.concatenate([[0], hot.astype(np.int8), [0]]))
    clusters: list[list[int]] = []  # [start, end exclusive]
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=False):
        if clusters and s - clusters[-1][1] <= merge_gap:
            clusters[-1][1] = int(e)
        else:
            clusters.append([int(s), int(e)])
    peaks = np.array([h[s:e].max() for s, e in clusters])
    return np.sort(peaks)[::-1]


def budget_threshold(
    h: np.ndarray,
    days: float,
    budget: float,
    x_min: float = 1.0,
    merge_gap: int = 10,
    extrapolate: bool = True,
    top: int = 10,
    default_excess: float = 1.0,
) -> float:
    """Threshold of the held score `h` (calibration window of `days` days) that alerts at
    most `budget` times per day. `inf` when the threshold would exceed `X_CAP`.

    Extrapolating: the `m = min(top, clusters)` highest cluster peaks above `u0` (the next
    peak, or `x_min` when there is none) have an exponential tail with the mean excess as scale,
    so the alert rate at threshold t is `m / days * exp(-(t - u0) / scale)`, solved for the
    budget. With fewer than 3 peaks the scale is `default_excess` (one decade of surprise).
    Not extrapolating: the order statistic of the peaks (alerts counted in the window)."""
    allowed = budget * days
    peaks = alert_peaks(h, x_min, merge_gap)
    k = math.floor(allowed)
    if not extrapolate:
        thr = x_min if allowed >= len(peaks) else float(np.nextafter(peaks[k], math.inf))
    else:
        m = min(top, len(peaks))
        if m >= 1 and allowed >= m:  # budget covers every peak of the fit: read it from the order statistics
            thr = x_min if allowed >= len(peaks) else float(np.nextafter(peaks[k], math.inf))
        else:
            u0 = float(peaks[m]) if len(peaks) > m else x_min
            scale = float(np.mean(peaks[:m] - u0)) if m >= 3 else default_excess
            scale = max(scale, 0.05)
            m_eff = max(m, 1)
            thr = u0 + scale * math.log(m_eff / allowed)
            thr = max(thr, x_min)
    return thr if thr <= X_CAP else math.inf
