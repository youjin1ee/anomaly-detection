"""Toto 2.0 forecast-band model (Datadog, Apache-2.0 code and weights).

Optional: needs the `toto` extra (`pip install anomaly-detection[toto]`), which brings torch and
the `toto-2` package. Importing this module does not import torch; it is loaded the first time
a model is created.

Per sample: the last `context_length` values of the series (before the new one) go into the
model, which forecasts the next step as nine quantiles (0.1 to 0.9). lower is the 0.1 quantile,
upper the 0.9 quantile and expected the median. The new value is scored against that band and
then appended to the context, so the score never sees the value it judges.

Score: with d the distance of the value outside [lower, upper] (0 inside) and w the band width,
`r = d / w` and `score = r / (r + score_mid)`. Inside the band the score is 0, one band width
beyond its edge it is `1 / (1 + score_mid)`, and it tends to 1 for far outliers. The mapping is
monotone in r, so a threshold chosen on the score is a threshold on r. `score_mid` is the
deviation, in band widths beyond the edge, at which the score is 0.5, the level alerts use.
The default 1.0 means 0.5 is reached about 3.8 sigma from the median for Gaussian noise
(the 0.1 to 0.9 band is 2.56 sigma wide).

The model sees a window that is a whole number of patches (32 samples). While the buffer is
shorter than `context_length` the window is the longest whole number of patches that fits, and
nothing is returned before `min_context` samples have been seen.

Two code paths give the same numbers: `TotoModel.update` (one series, one step, the sidecar
path) and `score_batch` (many steps of one series in one forward pass per batch, for offline
evaluation).
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np

from .base import ModelOutput, check_params, register_model

PATCH_SIZE = 32
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
_LOWER, _MEDIAN, _UPPER = 0, 4, 8


class Forecaster(Protocol):
    """Next-step quantile forecaster. `windows` is (batch, length) float32, length a whole
    number of patches, no missing values. Returns (batch, 9) quantiles, ascending."""

    def forecast(self, windows: np.ndarray) -> np.ndarray: ...


class TotoEngine:
    """The Toto 2.0 network on CPU, shared by every series that uses the same weights."""

    def __init__(self, model_dir: str | Path, threads: int | None = None):
        try:
            import torch
            from toto2 import Toto2Model
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "the toto model needs the optional dependencies: "
                "pip install 'anomaly-detection[toto]'"
            ) from exc
        self._torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.model_dir = str(model_dir)
        self.model = Toto2Model.from_pretrained(self.model_dir, map_location="cpu").eval()
        self.model = self.model.to(torch.device("cpu"))

    def forecast(self, windows: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32)).unsqueeze(1)
        batch = x.shape[0]
        inputs = {
            "target": x,
            "target_mask": torch.ones_like(x, dtype=torch.bool),
            "series_ids": torch.zeros(batch, 1, dtype=torch.long),
        }
        with torch.inference_mode():
            q = self.model.forecast(inputs, horizon=1, has_missing_values=False)
        # (9, batch, variates, horizon) -> (batch, 9)
        return q[:, :, 0, 0].transpose(0, 1).float().numpy()


_ENGINES: dict[tuple[str, int | None], TotoEngine] = {}
_ENGINES_LOCK = threading.Lock()


def get_engine(model_dir: str | Path, threads: int | None = None) -> TotoEngine:
    """One engine per weights directory: the network is loaded once, not once per series."""
    key = (str(model_dir), threads)
    with _ENGINES_LOCK:
        if key not in _ENGINES:
            _ENGINES[key] = TotoEngine(model_dir, threads)
        return _ENGINES[key]


def window_length(seen: int, context_length: int) -> int:
    """Length of the context window after `seen` samples: the longest whole number of patches
    that fits in the buffer, at most `context_length`."""
    return (min(seen, context_length) // PATCH_SIZE) * PATCH_SIZE


def band_score(
    value: np.ndarray | float,
    quantiles: np.ndarray,
    score_mid: float,
    min_scale_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score, expected, lower and upper from the quantiles (last axis, 9 values)."""
    q = np.asarray(quantiles, dtype=float)
    lower, expected, upper = q[..., _LOWER], q[..., _MEDIAN], q[..., _UPPER]
    # A flat history gives a zero-width band, so the width has a floor, like the other models.
    width = np.maximum.reduce(
        [upper - lower, min_scale_ratio * np.abs(expected), np.full_like(expected, 1e-9)]
    )
    dist = np.maximum.reduce([lower - value, value - upper, np.zeros_like(expected)])
    r = dist / width
    return r / (r + score_mid), expected, lower, upper


class TotoModel:
    """Parameters:

    model_dir        directory with Toto 2.0 `config.json` and `model.safetensors`
                     (a downloaded Hugging Face checkpoint). Required unless `engine` is given
    context_length   most recent samples fed to the model, a multiple of 32 (default 512)
    min_context      samples needed before the first score, a multiple of 32 (default 128)
    score_mid        deviation beyond the band, in band widths, at which the score is 0.5
                     (default 1.0)
    min_scale_ratio  floor of the band width as a fraction of the median (default 0.02)
    threads          torch CPU threads (default: torch default)

    `engine` is for tests and for callers that load the network themselves.
    """

    _ALLOWED: ClassVar[set[str]] = {
        "model_dir",
        "context_length",
        "min_context",
        "score_mid",
        "min_scale_ratio",
        "threads",
    }

    def __init__(
        self,
        model_dir: str | Path | None = None,
        context_length: int = 512,
        min_context: int = 128,
        score_mid: float = 1.0,
        min_scale_ratio: float = 0.02,
        threads: int | None = None,
        engine: Forecaster | None = None,
    ):
        validate_lengths(context_length, min_context)
        if score_mid <= 0:
            raise ValueError("score_mid must be positive")
        if engine is None:
            if model_dir is None:
                raise ValueError("model toto needs model_dir (a Toto 2.0 checkpoint directory)")
            engine = get_engine(model_dir, threads)
        self._engine = engine
        self._ctx = context_length
        self._min = min_context
        self._mid = score_mid
        self._floor = min_scale_ratio
        self._buf: deque[float] = deque(maxlen=context_length)
        self.samples = 0

    def update(self, timestamp: float, value: float) -> ModelOutput | None:
        out = None
        if self.samples >= self._min:
            n = window_length(len(self._buf), self._ctx)
            window = np.fromiter(self._buf, dtype=np.float32, count=len(self._buf))[-n:]
            q = self._engine.forecast(window[None, :])[0]
            if np.all(np.isfinite(q)):
                score, expected, lower, upper = band_score(value, q, self._mid, self._floor)
                out = ModelOutput(float(score), float(expected), float(lower), float(upper))
        self._buf.append(value)
        self.samples += 1
        return out


def validate_lengths(context_length: int, min_context: int) -> None:
    if context_length < PATCH_SIZE or context_length % PATCH_SIZE:
        raise ValueError(f"context_length must be a positive multiple of {PATCH_SIZE}")
    if min_context < PATCH_SIZE or min_context % PATCH_SIZE or min_context > context_length:
        raise ValueError(
            f"min_context must be a multiple of {PATCH_SIZE} and at most context_length"
        )


def score_batch(
    engine: Forecaster,
    values: Sequence[float] | np.ndarray,
    context_length: int = 512,
    min_context: int = 128,
    score_mid: float = 1.0,
    min_scale_ratio: float = 0.02,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Same numbers as feeding `values` one by one to `TotoModel.update`, with many steps per
    forward pass. NaN where the model gives nothing. Returns arrays score, expected, lower and
    upper, one entry per value."""
    validate_lengths(context_length, min_context)
    xv = np.asarray(values, dtype=float)
    x = xv.astype(np.float32)
    n = len(x)
    out = {k: np.full(n, np.nan) for k in ("score", "expected", "lower", "upper")}
    # Window at step i holds x[i - len : i] with len from `window_length`; steps with the same
    # length go through the model together (all of them, once the buffer is full).
    by_len: dict[int, list[int]] = {}
    for i in range(min_context, n):
        by_len.setdefault(window_length(i, context_length), []).append(i)
    for length, steps in by_len.items():
        for lo in range(0, len(steps), batch_size):
            idx = np.asarray(steps[lo : lo + batch_size])
            windows = x[idx[:, None] + np.arange(-length, 0)[None, :]]
            q = engine.forecast(windows)
            ok = np.all(np.isfinite(q), axis=1)
            score, expected, lower, upper = band_score(
                xv[idx], q, score_mid, min_scale_ratio
            )
            for key, arr in (
                ("score", score),
                ("expected", expected),
                ("lower", lower),
                ("upper", upper),
            ):
                out[key][idx[ok]] = arr[ok]
    return out


def _factory(params: dict[str, Any], interval: float, seed: int) -> TotoModel:
    check_params("toto", params, TotoModel._ALLOWED)
    return TotoModel(**params)


register_model("toto", _factory)
