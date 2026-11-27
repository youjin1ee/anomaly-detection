"""Rule layer on top of the model scores.

Ideas taken from the decision rules of Grafana's promql-anomaly-detection (Apache-2.0, no code
copied): direction by metric kind, no scoring of sparse request metrics, a minimum band width,
and a drop rule for request metrics that only looks at zero.

- direction: latency and error series count rises only. Other kinds count both ways.
- minimum band: a deviation from the expected value of at most `tol_frac` of the series scale
  (99th percentile of the calibration window) or `tol_ratio` of the expected value is ignored,
  and so is a fall while the expected value is below `min_expected_frac` of the scale.
- sparse: a request series that averages fewer than `sparse_per_min` events per minute in the
  calibration window is left out. Needs `SeriesInfo.per_minute` (events per minute per unit of
  value), so it applies only where the unit is known.
- zero drop: a request series that is exactly 0 where the same time one day earlier was above 0
  fires at once (surprise infinite), whatever the models say. Needs `zero_is_real`.

The expected value is the median of K samples spread over the last day, so the rules work on
any series and do not depend on the model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

DIRECTION = {"latency": "above", "error": "above"}


@dataclass(frozen=True)
class SeriesInfo:
    kind: str = "other"  # request, latency, error, resource, other
    spd: int = 1440  # samples per day
    per_minute: float | None = None  # events per minute = value * per_minute
    zero_is_real: bool = False  # a value of 0 means nothing happened (not a shifted scale)


@dataclass(frozen=True)
class RuleConfig:
    tol_frac: float = 0.05
    tol_ratio: float = 0.05
    min_expected_frac: float = 0.05
    sparse_per_min: float = 5.0
    points: int = 48  # samples behind the expected value, spread over one day


@dataclass
class RuleResult:
    suppress: np.ndarray  # bool, the score of this sample is set to 0
    force: np.ndarray  # bool, the sample alerts at once
    excluded: bool  # the whole series is left out (sparse)


def stride_of(spd: int, points: int) -> int:
    return max(1, round(spd / points))


def expected_series(values: np.ndarray, spd: int, points: int = 48) -> np.ndarray:
    """Median of values[t-1-j*stride] for j < points (the finite ones), NaN where none."""
    n, stride = len(values), stride_of(spd, points)
    cols = []
    for j in range(points):
        lag = 1 + j * stride
        col = np.full(n, np.nan)
        if lag < n:
            col[lag:] = values[: n - lag]
        cols.append(col)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(cols, axis=1), axis=1)


def series_scale(values: np.ndarray, cal: tuple[int, int]) -> float:
    seen = values[cal[0] : cal[1]]
    seen = seen[np.isfinite(seen)]
    return float(np.quantile(seen, 0.99)) if seen.size else 0.0


def suppressed(
    value: float, expected: float, direction: str, scale: float, cfg: RuleConfig
) -> bool:
    """One sample, the same rules as `apply_rules` on arrays (a test keeps them in step)."""
    if not (np.isfinite(value) and np.isfinite(expected)):
        return False
    dev = value - expected
    if (direction == "above" and dev <= 0) or (direction == "below" and dev >= 0):
        return True
    if dev < 0 and expected < cfg.min_expected_frac * scale:
        return True
    return abs(dev) <= cfg.tol_frac * scale or abs(dev) <= cfg.tol_ratio * abs(expected)


def is_excluded(
    values: np.ndarray, info: SeriesInfo, cal: tuple[int, int], cfg: RuleConfig
) -> bool:
    if info.kind != "request" or info.per_minute is None:
        return False
    seen = values[cal[0] : cal[1]]
    seen = seen[np.isfinite(seen)]
    return bool(seen.size) and float(np.mean(seen)) * info.per_minute < cfg.sparse_per_min


def zero_drop_now(value: float, yesterday: float) -> bool:
    return value == 0.0 and np.isfinite(yesterday) and yesterday > 0.0


def apply_rules(
    values: np.ndarray, info: SeriesInfo, cal: tuple[int, int], cfg: RuleConfig
) -> RuleResult:
    exp = expected_series(values, info.spd, cfg.points)
    scale = series_scale(values, cal)
    direction = DIRECTION.get(info.kind, "both")
    ok = np.isfinite(values) & np.isfinite(exp)
    dev = np.where(ok, values - exp, 0.0)
    sup = np.zeros(len(values), dtype=bool)
    if direction == "above":
        sup |= dev <= 0
    elif direction == "below":
        sup |= dev >= 0
    sup |= (dev < 0) & (exp < cfg.min_expected_frac * scale)
    sup |= (np.abs(dev) <= cfg.tol_frac * scale) | (np.abs(dev) <= cfg.tol_ratio * np.abs(exp))
    sup &= ok
    force = np.zeros(len(values), dtype=bool)
    if info.kind == "request" and info.zero_is_real and info.spd < len(values):
        prev = np.full(len(values), np.nan)
        prev[info.spd :] = values[: -info.spd]
        force = (values == 0.0) & np.isfinite(prev) & (prev > 0.0)
    return RuleResult(sup, force, is_excluded(values, info, cal, cfg))
