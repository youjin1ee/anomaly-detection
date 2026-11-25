"""Random Cut Forest model on shingled, robustly normalised values (rrcf, MIT).

Pipeline per sample: normalise with the trailing median and MAD (computed before the new
value is added, so a jump stands out), stack the last `shingle_size` normalised values into
one point, insert it into every tree, and read the collusive displacement (CoDisp). The raw
CoDisp has no fixed scale, so it is divided by a high quantile of its own recent history and
squashed into [0, 1].

The model has no band: RCF scores shape, not a value range. `expected` is the trailing
median of the last `reference_window` samples and is only meant for direction, tolerance and
low-traffic checks.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Any, ClassVar

import numpy as np

from .base import ModelOutput, check_params, logistic, register_model

with warnings.catch_warnings():
    # rrcf imports pkg_resources, which is deprecated in setuptools.
    warnings.simplefilter("ignore")
    import rrcf

_MAD_TO_SIGMA = 1.4826


class RCFModel:
    """Parameters:

    shingle_size       consecutive samples per point (default 8, AMP default)
    num_trees          trees in the forest (default 40)
    sample_size        points kept per tree, sliding window (default 256, AMP minimum)
    norm_window        samples used for the trailing median and MAD (default 1440)
    reference_window   samples used for `expected` (default 60)
    time_features      add sin/cos of time of day (and week) as two more dimensions, which
                       lets the forest see seasonality (default false)
    time_feature_weight  scale of those dimensions after normalisation (default 1.0)
    score_ref_quantile quantile of the CoDisp history used as the reference (default 0.99)
    score_mid          CoDisp divided by the reference at which score is 0.5 (default 2.0)
    min_scale_ratio    floor of the normalisation scale as a fraction of the median
                       (default 0.02)
    """

    _ALLOWED: ClassVar[set[str]] = {
        "shingle_size",
        "num_trees",
        "sample_size",
        "norm_window",
        "reference_window",
        "time_features",
        "time_feature_weight",
        "score_ref_quantile",
        "score_mid",
        "min_scale_ratio",
    }

    def __init__(
        self,
        seed: int = 0,
        shingle_size: int = 8,
        num_trees: int = 40,
        sample_size: int = 256,
        norm_window: int = 1440,
        reference_window: int = 60,
        time_features: bool = False,
        time_feature_weight: float = 1.0,
        score_ref_quantile: float = 0.99,
        score_mid: float = 2.0,
        min_scale_ratio: float = 0.02,
    ):
        if shingle_size < 1 or num_trees < 1 or sample_size < 8:
            raise ValueError("shingle_size >= 1, num_trees >= 1 and sample_size >= 8 required")
        self._shingle = shingle_size
        self._size = sample_size
        self._time_features = time_features
        self._tw = time_feature_weight
        self._q = score_ref_quantile
        self._mid = score_mid
        self._min_ratio = min_scale_ratio
        # One seeded generator per tree keeps runs reproducible, which the evaluation needs.
        self._trees = [
            rrcf.RCTree(random_state=np.random.RandomState(seed * 100003 + i))
            for i in range(num_trees)
        ]
        self._hist: deque[float] = deque(maxlen=norm_window)
        self._norm: deque[float] = deque(maxlen=shingle_size)
        self._ref_window = reference_window
        self._codisp: deque[float] = deque(maxlen=2000)
        self._next_index = 0
        self._indices: deque[int] = deque()
        self.samples = 0

    def update(self, timestamp: float, value: float) -> ModelOutput | None:
        self.samples += 1
        hist = self._hist
        if len(hist) < max(self._shingle, 30):
            hist.append(value)
            return None
        arr = np.fromiter(hist, dtype=float, count=len(hist))
        loc = float(np.median(arr))
        scale = max(
            _MAD_TO_SIGMA * float(np.median(np.abs(arr - loc))),
            self._min_ratio * abs(loc),
            1e-12,
        )
        expected = float(np.median(arr[-self._ref_window :]))
        hist.append(value)
        self._norm.append((value - loc) / scale)
        if len(self._norm) < self._shingle:
            return None

        point = np.fromiter(self._norm, dtype=float, count=self._shingle)
        if self._time_features:
            day = 2 * np.pi * (timestamp % 86400.0) / 86400.0
            week = 2 * np.pi * (timestamp % 604800.0) / 604800.0
            point = np.concatenate(
                [point, self._tw * np.array([np.sin(day), np.cos(day), np.sin(week), np.cos(week)])]
            )

        idx = self._next_index
        self._next_index += 1
        total = 0.0
        evict = len(self._indices) >= self._size
        old = self._indices.popleft() if evict else None
        for tree in self._trees:
            if old is not None:
                tree.forget_point(old)
            tree.insert_point(point, index=idx)
            total += tree.codisp(idx)
        self._indices.append(idx)
        codisp = total / len(self._trees)

        ready = len(self._indices) >= self._size and len(self._codisp) >= 50
        ref = float(np.quantile(np.fromiter(self._codisp, float), self._q)) if ready else 0.0
        self._codisp.append(codisp)
        if not ready:
            return None
        ratio = codisp / max(ref, 1e-9)
        score = logistic((ratio - self._mid) / (self._mid / 4.0))
        return ModelOutput(score, expected)


def _factory(params: dict[str, Any], interval: float, seed: int) -> RCFModel:
    check_params("rcf", params, RCFModel._ALLOWED)
    return RCFModel(seed=seed, **params)


register_model("rcf", _factory)
