"""Per-series pipeline: model plus the guards that apply to every model."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Model


@dataclass(frozen=True, slots=True)
class Result:
    timestamp: float
    value: float
    score: float
    expected: float
    lower: float | None
    upper: float | None
    tier: int | None = None


def apply_guards(
    score: float,
    value: float,
    expected: float,
    direction: str = "both",
    min_expected: float = 0.0,
    tolerance_amount: float = 0.0,
    tolerance_ratio: float = 0.0,
) -> float:
    """Guards shared by every model: direction, low-traffic floor and tolerance.

    Pure function, so replays can apply different guard settings to stored model scores."""
    dev = value - expected
    if (direction == "above" and dev <= 0) or (direction == "below" and dev >= 0):
        return 0.0
    # Low traffic protection: a tiny expected value makes downward moves meaningless noise.
    # A fall from a large expected value to zero is not affected, because expected is large.
    if dev < 0 and expected < min_expected:
        return 0.0
    if abs(dev) <= tolerance_amount or abs(dev) <= tolerance_ratio * abs(expected):
        return 0.0
    return score


class SeriesPipeline:
    def __init__(
        self,
        model: Model,
        direction: str = "both",
        warmup_samples: int = 0,
        min_expected: float = 0.0,
        tolerance_amount: float = 0.0,
        tolerance_ratio: float = 0.0,
    ):
        self._model = model
        self._direction = direction
        self._warmup = warmup_samples
        self._min_expected = min_expected
        self._tol_amount = tolerance_amount
        self._tol_ratio = tolerance_ratio
        self.samples = 0
        self.last_timestamp = float("-inf")
        self.last_result: Result | None = None

    @property
    def ready(self) -> bool:
        return self.last_result is not None

    def update(self, timestamp: float, value: float) -> Result | None:
        """Feed one sample. Returns None while warming up (no score exists yet).

        Out-of-order or repeated timestamps are ignored, so overlapping backfill and live
        evaluation cannot double count a sample.
        """
        if timestamp <= self.last_timestamp:
            return None
        self.last_timestamp = timestamp
        out = self._model.update(timestamp, value)
        self.samples += 1
        if out is None or self.samples < self._warmup:
            self.last_result = None
            return None

        score = apply_guards(
            out.score,
            value,
            out.expected,
            self._direction,
            self._min_expected,
            self._tol_amount,
            self._tol_ratio,
        )

        self.last_result = Result(
            timestamp, value, score, out.expected, out.lower, out.upper, out.tier
        )
        return self.last_result
