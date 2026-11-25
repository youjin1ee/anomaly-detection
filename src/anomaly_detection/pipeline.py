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

        score = out.score
        dev = value - out.expected
        if (self._direction == "above" and dev <= 0) or (self._direction == "below" and dev >= 0):
            score = 0.0
        # Low traffic protection: a tiny expected value makes downward moves meaningless noise.
        # A fall from a large expected value to zero is not affected, because expected is large.
        if dev < 0 and out.expected < self._min_expected:
            score = 0.0
        if abs(dev) <= self._tol_amount or abs(dev) <= self._tol_ratio * abs(out.expected):
            score = 0.0

        self.last_result = Result(timestamp, value, score, out.expected, out.lower, out.upper)
        return self.last_result
