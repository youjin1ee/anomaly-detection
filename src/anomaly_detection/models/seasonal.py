"""Seasonal robust baseline.

For the current time slot, collect what the series did at the same time of day (and/or the
same weekday and time) in the past, and use the median and MAD of those values as the
expected value and spread. This is the same family as recording-rule baselines in PromQL,
but it also works in a few days of history and keeps no state beyond a ring buffer.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from .base import ModelOutput, check_params, logistic, register_model

_PERIOD_SECONDS = {"daily": 86400.0, "weekly": 604800.0}
_MAD_TO_SIGMA = 1.4826


def _period_seconds(item: Any) -> float:
    if isinstance(item, str) and item in _PERIOD_SECONDS:
        return _PERIOD_SECONDS[item]
    if isinstance(item, int | float) and not isinstance(item, bool) and item > 0:
        return float(item)
    raise ValueError(f"seasonality entries must be 'daily', 'weekly' or seconds, got {item!r}")


class SeasonalBaseline:
    """Parameters:

    seasonality    list of periods, 'daily', 'weekly' or a number of seconds. The longest
                   period that already has enough history wins; shorter ones are fallbacks.
    lookback       how many past periods to look at (default 7 for periods up to a day,
                   4 for longer ones)
    slot_window    neighbour slots on each side of the same slot that are pooled in, which
                   makes the estimate usable after a few days (default 8)
    min_points     minimum pooled values needed for a period to be used (default 18)
    band_k         band half width in robust sigmas, and the score is 0.5 at the band edge
                   (default 6.0)
    min_scale_ratio  floor of the spread as a fraction of the expected value, so a perfectly
                   flat history does not make every wiggle an anomaly (default 0.02)
    """

    _ALLOWED: ClassVar[set[str]] = {
        "seasonality",
        "lookback",
        "slot_window",
        "min_points",
        "band_k",
        "min_scale_ratio",
    }

    def __init__(
        self,
        interval: float,
        seasonality: list[Any] | None = None,
        lookback: int | None = None,
        slot_window: int = 8,
        min_points: int = 18,
        band_k: float = 6.0,
        min_scale_ratio: float = 0.02,
    ):
        if interval <= 0:
            raise ValueError("interval must be positive")
        periods = sorted({_period_seconds(s) for s in (seasonality or ["daily", "weekly"])})
        self._interval = interval
        # (period in slots, number of periods to look back), longest first.
        self._periods: list[tuple[int, int]] = []
        for seconds in reversed(periods):
            slots = round(seconds / interval)
            if slots < 2 * slot_window + 2:
                raise ValueError(
                    f"period {seconds}s is too short for interval {interval}s and "
                    f"slot_window {slot_window}"
                )
            look = lookback if lookback is not None else (7 if seconds <= 86400 else 4)
            self._periods.append((slots, look))
        self._w = slot_window
        self._min_points = min_points
        self._k = band_k
        self._min_ratio = min_scale_ratio
        cap = max(p * look for p, look in self._periods) + slot_window + 2
        self._cap = cap
        self._values = np.full(cap, np.nan)
        self._slots = np.full(cap, -1, dtype=np.int64)
        self._offsets = np.arange(-slot_window, slot_window + 1)
        self.samples = 0

    def _gather(self, n: int, period: int, look: int) -> np.ndarray:
        ids = n - (np.arange(1, look + 1)[:, None] * period) + self._offsets[None, :]
        pos = ids % self._cap
        ok = (self._slots[pos] == ids) & (ids >= 0)
        return self._values[pos][ok]

    def update(self, timestamp: float, value: float) -> ModelOutput | None:
        n = round(timestamp / self._interval)
        result = None
        for period, look in self._periods:
            ref = self._gather(n, period, look)
            if ref.size >= self._min_points:
                result = self._score(value, ref)
                break
        # Learn after scoring, so a point never explains itself.
        pos = n % self._cap
        self._values[pos] = value
        self._slots[pos] = n
        self.samples += 1
        return result

    def _score(self, value: float, ref: np.ndarray) -> ModelOutput:
        med = float(np.median(ref))
        mad = float(np.median(np.abs(ref - med)))
        scale = max(_MAD_TO_SIGMA * mad, self._min_ratio * abs(med), 1e-12)
        z = abs(value - med) / scale
        # Logistic around the band edge: 0.5 at k sigmas, about 0.98 at 2k.
        score = logistic((z - self._k) / (self._k / 4.0))
        return ModelOutput(score, med, med - self._k * scale, med + self._k * scale)


def _factory(params: dict[str, Any], interval: float, seed: int) -> SeasonalBaseline:
    check_params("seasonal", params, SeasonalBaseline._ALLOWED)
    return SeasonalBaseline(interval, **params)


register_model("seasonal", _factory)
