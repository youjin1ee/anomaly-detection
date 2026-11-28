"""Common model interface and registry.

A model sees one time series, one sample at a time, and answers with a score in [0, 1] plus
an expected value and an optional band. Everything shared between models (direction,
tolerance, low-traffic floor, warm-up guard) lives in the pipeline, not in the models, so a
new model only has to implement `update`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """score is in [0, 1], 1 meaning certainly anomalous. lower and upper are None for
    models that cannot produce a band (for example RCF)."""

    score: float
    expected: float
    lower: float | None = None
    upper: float | None = None
    # alert stage of the ensemble model: 0 none, 1 ai analysis, 2 human. None for other models.
    tier: int | None = None


class Model(Protocol):
    def update(self, timestamp: float, value: float) -> ModelOutput | None:
        """Learn from one sample and score it. Returns None while the model is not ready.

        Timestamps are strictly increasing. Values are finite floats: missing data is never
        passed in (and never passed in as 0).
        """
        ...


# factory(params, interval_seconds, seed) -> Model
ModelFactory = Callable[[dict[str, Any], float, int], Model]

_REGISTRY: dict[str, ModelFactory] = {}


def register_model(kind: str, factory: ModelFactory) -> None:
    """Register a model kind. This is the extension point for forecast-band models and
    other detectors that are too heavy to be core dependencies."""
    _REGISTRY[kind] = factory


def create_model(kind: str, params: dict[str, Any], interval: float, seed: int = 0) -> Model:
    try:
        factory = _REGISTRY[kind]
    except KeyError:
        raise ValueError(f"unknown model kind {kind!r}; available: {sorted(_REGISTRY)}") from None
    return factory(params, interval, seed)


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def check_params(kind: str, params: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(
            f"model {kind}: unknown parameters {sorted(unknown)}; allowed {sorted(allowed)}"
        )


def logistic(x: float) -> float:
    # Avoid overflow for very large negative inputs.
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))
