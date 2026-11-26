"""Toto 2.0 on NAB.

Parameter rule: none per file. Every file gets the same context length (512) and minimum
context (128), the model defaults. The shortest file has a probationary period of 169 records,
so the model scores from the end of the probationary period in every file. Nothing depends on
values, labels, results or the sampling interval.

The offline path scores many steps per forward pass (`score_batch`) and gives the same numbers
as feeding the records one by one through the model and the pipeline (`stream_series`); a test
checks that.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..models.toto import Forecaster, TotoModel, score_batch
from ..pipeline import SeriesPipeline
from .nab_data import NabSeries

TOTO_SIZES = ("4m", "22m")
TOTO_CONTEXT = 512
TOTO_MIN_CONTEXT = 128


def toto_name(size: str) -> str:
    return f"toto-{size}"


def checkpoint_dir(root: str | Path, size: str) -> Path:
    return Path(root) / size


def fed_indices(series: NabSeries) -> np.ndarray:
    """Records the stream path feeds to the model: finite values whose timestamp is later than
    the previous fed one (the pipeline skips duplicates and backward steps)."""
    keep: list[int] = []
    last = float("-inf")
    for i, (t, v) in enumerate(zip(series.timestamps, series.values, strict=True)):
        if not np.isfinite(v):
            continue
        if t <= last:
            continue
        last = float(t)
        keep.append(i)
    return np.asarray(keep, dtype=int)


def batch_series(
    engine: Forecaster, series: NabSeries, batch_size: int = 256
) -> tuple[np.ndarray, float]:
    """Scores per record (NaN where the model gave none) and microseconds per fed record."""
    n = len(series.values)
    keep = fed_indices(series)
    t0 = time.perf_counter()
    res = score_batch(
        engine,
        series.values[keep],
        context_length=TOTO_CONTEXT,
        min_context=TOTO_MIN_CONTEXT,
        batch_size=batch_size,
    )
    us = (time.perf_counter() - t0) / max(len(keep), 1) * 1e6
    scores = np.full(n, np.nan)
    scores[keep] = res["score"]
    return scores, us


def stream_series(engine: Forecaster, series: NabSeries) -> np.ndarray:
    """The same scores through the one-sample-at-a-time path, as the sidecar runs it."""
    model = TotoModel(
        engine=engine, context_length=TOTO_CONTEXT, min_context=TOTO_MIN_CONTEXT
    )
    pipe = SeriesPipeline(model, direction="both")
    scores = np.full(len(series.values), np.nan)
    for i in range(len(series.values)):
        v = series.values[i]
        if not np.isfinite(v):
            continue
        res = pipe.update(float(series.timestamps[i]), float(v))
        if res is not None:
            scores[i] = res.score
    return scores
