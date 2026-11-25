"""Fixed threshold model. Useful as a comparison baseline and for hard limits."""

from __future__ import annotations

from typing import Any

from .base import ModelOutput, check_params, register_model


class StaticThreshold:
    def __init__(self, lower: float | None, upper: float | None):
        if lower is None and upper is None:
            raise ValueError("static model needs lower and/or upper")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower must not exceed upper")
        self._lower = lower
        self._upper = upper

    def update(self, timestamp: float, value: float) -> ModelOutput:
        out = (self._lower is not None and value < self._lower) or (
            self._upper is not None and value > self._upper
        )
        # Expected is the nearest in-range value, so the deviation sign tells the direction.
        expected = value
        if self._lower is not None:
            expected = max(expected, self._lower)
        if self._upper is not None:
            expected = min(expected, self._upper)
        return ModelOutput(1.0 if out else 0.0, expected, self._lower, self._upper)


def _factory(params: dict[str, Any], interval: float, seed: int) -> StaticThreshold:
    check_params("static", params, {"lower", "upper"})
    return StaticThreshold(params.get("lower"), params.get("upper"))


register_model("static", _factory)
