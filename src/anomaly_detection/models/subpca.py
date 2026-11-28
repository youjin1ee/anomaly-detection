"""Sub-PCA: reconstruction error of sliding subsequences against a PCA of recent normal ones.

Streaming version of the method that ranks first for univariate series in TSB-AD (see
docs/design.md, "Sub-PCA"). Written with numpy only; no TSB-AD code is copied.

Per sample: normalise with the trailing median and MAD of the training span, cut the last
`window` values into one subsequence, project it on the top principal components of recent
subsequences and take the root mean square of what the projection misses. That error is
divided by a high quantile of the training errors (r) and mapped into [0, 1) by
log(1+r) / (log(1+r) + log(1+mid)), which is 0.5 at r = mid.

The PCA is refitted every `retrain_every` samples on the last `train_size` subsequences.
With `exclude_score` set (off by default), subsequences that scored at or above it when
they arrived are left out of the fit, so an anomaly does not become the new normal. A run
of such subsequences longer than `exclude_max_run` is a level shift that lasts, not an anomaly, and
is trained on after all; so is everything if too few subsequences (`min_train`) would be left.

No band: `expected` is the PCA reconstruction of the newest value.
"""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar

import numpy as np

from .base import ModelOutput, check_params, register_model

_MAD_TO_SIGMA = 1.4826


class SubPCAModel:
    """Parameters:

    window             subsequence length N (default 16)
    n_components       int: components kept; float in (0, 1): smallest number explaining
                       that share of the variance (default 0.75)
    train_size         most recent subsequences used for a fit (default 1024)
    min_train          subsequences needed before the first fit (default 256)
    retrain_every      samples between fits (default 64)
    exclude_score      subsequences scored at or above this are not trained on; None keeps
                       all (default None: it cost NAB score, see docs/design.md)
    exclude_max_run    a run of consecutive excluded subsequences longer than this is taken to
                       be a new normal and trained on after all (default 2 * window)
    score_ref_quantile quantile of the training errors used as the reference (default 0.99)
    score_mid          error divided by the reference at which score is 0.5 (default 2.0)
    min_scale_ratio    floor of the normalisation scale as a fraction of the median
                       (default 0.02)
    """

    _ALLOWED: ClassVar[set[str]] = {
        "window",
        "n_components",
        "train_size",
        "min_train",
        "retrain_every",
        "exclude_score",
        "exclude_max_run",
        "score_ref_quantile",
        "score_mid",
        "min_scale_ratio",
    }
    _REF_FLOOR = 0.05  # in normalised units: a residual below this is never an anomaly

    def __init__(
        self,
        window: int = 16,
        n_components: float = 0.75,
        train_size: int = 1024,
        min_train: int = 256,
        retrain_every: int = 64,
        exclude_score: float | None = None,
        exclude_max_run: int | None = None,
        score_ref_quantile: float = 0.99,
        score_mid: float = 2.0,
        min_scale_ratio: float = 0.02,
    ):
        if window < 4 or retrain_every < 1 or min_train < 2 or train_size < min_train:
            raise ValueError("window >= 4, retrain_every >= 1, train_size >= min_train >= 2")
        if n_components <= 0 or (n_components >= 1 and int(n_components) != n_components):
            raise ValueError("n_components is a positive integer or a float in (0, 1)")
        self._w = window
        self._k = n_components
        self._min_train = min_train
        self._every = retrain_every
        self._excl = exclude_score
        self._max_run = 2 * window if exclude_max_run is None else exclude_max_run
        self._q = score_ref_quantile
        self._mid = score_mid
        self._min_ratio = min_scale_ratio
        self._vals: deque[float] = deque(maxlen=train_size + window - 1)
        # score of the subsequence ending at each buffered value (nan: not scored yet)
        self._scores: deque[float] = deque(maxlen=train_size + window - 1)
        self._since_fit = 0
        self._fitted = False
        self.fits = 0

    def _fit(self) -> None:
        x = np.fromiter(self._vals, float, len(self._vals))
        loc = float(np.median(x))
        scale = max(
            _MAD_TO_SIGMA * float(np.median(np.abs(x - loc))), self._min_ratio * abs(loc), 1e-12
        )
        subs = np.lib.stride_tricks.sliding_window_view((x - loc) / scale, self._w)
        sc = np.fromiter(self._scores, float, len(self._scores))[self._w - 1 :]
        if self._excl is not None:
            high = sc >= self._excl  # nan (never scored) is False
            # A run of high scores longer than `max_run` is a new normal, not an anomaly.
            edges = np.flatnonzero(np.diff(np.concatenate([[0], high.astype(int), [0]])))
            for a, b in zip(edges[::2], edges[1::2], strict=True):
                if b - a > self._max_run:
                    high[a:b] = False
            keep = ~high
            if keep.sum() >= self._min_train:
                subs = subs[keep]
        mean = subs.mean(axis=0)
        _, s, vt = np.linalg.svd(subs - mean, full_matrices=False)
        if self._k < 1:
            share = np.cumsum(s**2) / max(float((s**2).sum()), 1e-300)
            k = int(np.searchsorted(share, self._k) + 1)
        else:
            k = int(self._k)
        self._basis = vt[: min(k, len(s))]
        self._mean, self._loc, self._scale = mean, loc, scale
        err = self._error(subs)
        self._ref = max(float(np.quantile(err, self._q)), self._REF_FLOOR)
        self._fitted = True
        self.fits += 1

    def _error(self, subs: np.ndarray) -> np.ndarray:
        c = subs - self._mean
        resid = c - (c @ self._basis.T) @ self._basis
        return np.sqrt((resid**2).mean(axis=-1))

    def update(self, timestamp: float, value: float) -> ModelOutput | None:
        self._vals.append(value)
        self._scores.append(np.nan)
        self._since_fit += 1
        n = len(self._vals)
        out = None
        if self._fitted and n >= self._w:
            sub = (np.fromiter(self._vals, float, n)[-self._w :] - self._loc) / self._scale
            c = sub - self._mean
            rec = (c @ self._basis.T) @ self._basis
            ratio = float(np.sqrt(((c - rec) ** 2).mean())) / self._ref
            # log scale: error ratios reach millions on NAB and a logistic or a rational map
            # would saturate at exactly 1.0, which ties records that a threshold must rank
            lr = np.log1p(ratio)
            score = float(lr / (lr + np.log1p(self._mid)))
            self._scores[-1] = score
            out = ModelOutput(score, float((rec[-1] + self._mean[-1]) * self._scale + self._loc))
        due = not self._fitted or self._since_fit >= self._every
        if n - self._w + 1 >= self._min_train and due:
            self._fit()
            self._since_fit = 0
        return out


def _factory(params: dict[str, Any], interval: float, seed: int) -> SubPCAModel:
    check_params("subpca", params, SubPCAModel._ALLOWED)
    return SubPCAModel(**params)


register_model("subpca", _factory)
