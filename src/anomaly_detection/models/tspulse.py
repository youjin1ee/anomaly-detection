"""IBM Granite TSPulse (zero-shot, Apache-2.0) as a streaming model.

TSPulse is a 1M parameter pre-trained model with a context of 512 points. It has three ways to
say that a point is unusual, called heads here:

    time   masked reconstruction in the time domain: the patch that holds the newest point is
           hidden and rebuilt from the other 504 points
    fft    the same masked reconstruction, read from the frequency branch
    pred   the one step forecast made from the window that ended one sample earlier

Each head gives a squared error in standardised units. The IBM pipeline picks heads with a
labelled validation set (triangulation). A sidecar has no labels, so this model does what the
paper reports for the label-free case: it normalises every head against its own history and
takes the maximum.

Per sample:

1. Keep the last `context_length` (512) values. Standardise the window with the mean and the
   standard deviation of its older part (everything except the newest `aggregation_length`
   points), so a jump in the newest points stands out instead of widening the scale. The IBM
   pipeline standardises the whole series at once, which a stream cannot do.
2. One batched forward pass: the unmasked window for the forecast, and one copy of the window
   per patch of the newest `aggregation_length` points (default 32 points, 4 patches of 8)
   with that patch masked, for the two reconstruction heads.
3. Head error: mean squared error over the newest `aggregation_length` points (time and fft,
   the IBM default of 32) or of the newest point against the forecast (pred).
4. Normalise: divide the error by a high quantile of the head's own history (taken before the
   new error is added) and squash with a logistic, the same mapping as the RCF model. The
   score is the maximum over the heads.
5. `expected` is the one step forecast of the pred head. The band is `expected` plus and
   minus `band_k` robust standard deviations of the pred head's recent standardised residuals.
   TSPulse itself only outputs scores, so the band is built here.

A score exists from the (`context_length` + 1 + `min_history`)-th sample on. The IBM model
card asks for 1536 to 2048 points for stable anomaly detection. That figure belongs to its
offline pipeline, which min-max scales scores over the whole series. A stream has no such
scaling, so the window is the only hard minimum here and the history needed for the
normalisation is the rest. The cost is that a short file spends most of its life warming up.

`score_offline` runs the same state machine with batched inference. Inference never feeds
back into the state that prepares the next window, so the results match the streaming path.
Dependencies come from the `tspulse` extra. Importing this module does not import torch.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .base import ModelOutput, check_params, logistic, register_model

HEADS = ("time", "fft", "pred")
DEFAULT_MODEL = "ibm-granite/granite-timeseries-tspulse-r1"
_MAD_TO_SIGMA = 1.4826


class TSPulseEngine:
    """The torch model on CPU. One instance is shared by every series of a process."""

    def __init__(self, model_path: str = DEFAULT_MODEL, threads: int | None = None):
        try:
            import torch
            from tsfm_public.models.tspulse import TSPulseForReconstruction
            from tsfm_public.models.tspulse.utils.helpers import patchwise_stitched_reconstruction
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "the tspulse model needs the optional dependencies: pip install 'anomaly-detection[tspulse]'"
            ) from exc
        self._torch = torch
        self._stitch = patchwise_stitched_reconstruction
        if threads is not None:
            torch.set_num_threads(threads)
        # mask_type="user" is what the IBM anomaly detection example loads. With the pre-training
        # default the model masks random patches on every call and ignores ours, which makes
        # the output change from call to call.
        self.model = TSPulseForReconstruction.from_pretrained(
            model_path, num_input_channels=1, mask_type="user"
        ).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        self.context_length: int = int(cfg.context_length)
        self.patch_length: int = int(cfg.patch_length)
        self._lock = threading.Lock()

    def infer(
        self, windows: np.ndarray, aggregation_length: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """windows: (B, context_length) standardised values.

        Returns the next value forecast (B,), and the squared error of the time and fft
        reconstructions averaged over the newest `aggregation_length` points (B,)."""
        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32)).unsqueeze(-1)
        length = self.context_length
        with self._lock, torch.inference_mode():
            forecast = self.model(x)["forecast_output"][:, 0, 0]
            stitched = self._stitch(
                model=self.model,
                past_values=x,
                patch_size=self.patch_length,
                keys_to_stitch=["reconstruction_outputs", "reconstructed_ts_from_fft"],
                keys_to_aggregate=[],
                reconstruct_start=length - aggregation_length,
                reconstruct_end=length,
            )
            tail = slice(length - aggregation_length, length)
            err_time = ((x[:, tail, 0] - stitched["reconstruction_outputs"][:, tail, 0]) ** 2).mean(1)
            err_fft = (
                (x[:, tail, 0] - stitched["reconstructed_ts_from_fft"][:, tail, 0]) ** 2
            ).mean(1)
        return forecast.numpy(), err_time.numpy(), err_fft.numpy()


_ENGINES: dict[tuple[str, int | None], TSPulseEngine] = {}
_ENGINES_LOCK = threading.Lock()


def get_engine(model_path: str = DEFAULT_MODEL, threads: int | None = None) -> TSPulseEngine:
    """Load the model once per process and path."""
    key = (model_path, threads)
    with _ENGINES_LOCK:
        if key not in _ENGINES:
            _ENGINES[key] = TSPulseEngine(model_path, threads)
        return _ENGINES[key]


@dataclass(slots=True)
class _Pending:
    """What `_prepare` keeps until the inference result for the same sample arrives."""

    value: float
    loc: float
    scale: float


class _HeadScale:
    """Turns a raw error into a [0, 1] score against the head's own history."""

    def __init__(self, history: int, min_history: int, quantile: float, mid: float):
        self._hist: deque[float] = deque(maxlen=history)
        self._min = min_history
        self._q = quantile
        self._mid = mid

    def score(self, raw: float) -> float | None:
        ready = len(self._hist) >= self._min
        ref = float(np.quantile(np.fromiter(self._hist, float), self._q)) if ready else 0.0
        self._hist.append(raw)
        if not ready:
            return None
        ratio = raw / max(ref, 1e-12)
        return logistic((ratio - self._mid) / (self._mid / 4.0))


class TSPulseModel:
    """Parameters:

    model_path          directory with config.json and model.safetensors, or a Hugging Face id
                        (default ibm-granite/granite-timeseries-tspulse-r1, revision main, the
                        variant the card recommends for anomaly detection)
    aggregation_length  newest points that the reconstruction heads score (default 32, IBM
                        default, a multiple of the patch length 8)
    heads               which heads feed the maximum (default time, fft and pred)
    history             raw errors kept per head for the normalisation (default 2000)
    min_history         errors needed before a head scores (default 50)
    score_ref_quantile  quantile of the head history used as reference (default 0.99)
    score_mid           error divided by the reference at which a head scores 0.5 (default 2.0)
    band_k              half width of the band in robust standard deviations of the pred
                        head's recent residuals (default 4.0)
    residual_window     residuals kept for the band (default 512)
    min_scale_ratio     floor of the window standard deviation as a fraction of its mean
                        (default 0.02)
    threads             torch threads, None keeps the torch default
    """

    _ALLOWED: ClassVar[set[str]] = {
        "model_path",
        "aggregation_length",
        "heads",
        "history",
        "min_history",
        "score_ref_quantile",
        "score_mid",
        "band_k",
        "residual_window",
        "min_scale_ratio",
        "threads",
    }

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        aggregation_length: int = 32,
        heads: tuple[str, ...] | list[str] = HEADS,
        history: int = 2000,
        min_history: int = 50,
        score_ref_quantile: float = 0.99,
        score_mid: float = 2.0,
        band_k: float = 4.0,
        residual_window: int = 512,
        min_scale_ratio: float = 0.02,
        threads: int | None = None,
        engine: TSPulseEngine | None = None,
    ):
        self._engine = engine or get_engine(model_path, threads)
        self._ctx = self._engine.context_length
        if aggregation_length % self._engine.patch_length or not (
            0 < aggregation_length < self._ctx
        ):
            raise ValueError("aggregation_length must be a multiple of the patch length (8) < 512")
        heads = tuple(heads)
        if not heads or set(heads) - set(HEADS):
            raise ValueError(f"heads must be a non-empty subset of {HEADS}")
        self._agg = aggregation_length
        self._heads = heads
        self._band_k = band_k
        self._min_ratio = min_scale_ratio
        self._window: deque[float] = deque(maxlen=self._ctx)
        self._scales = {h: _HeadScale(history, min_history, score_ref_quantile, score_mid) for h in HEADS}
        self._residuals: deque[float] = deque(maxlen=residual_window)
        self._min_resid = min_history
        # forecast made at the previous sample: (value, scale of that window)
        self._prev_forecast: tuple[float, float] | None = None
        self.samples = 0
        self.last_heads: dict[str, float | None] = {}

    @property
    def context_length(self) -> int:
        return self._ctx

    # -- state machine, shared by the streaming and the offline path ------------------------

    def _prepare(self, value: float) -> tuple[np.ndarray, _Pending] | None:
        """Take one sample. Returns the standardised window to infer on, or None while the
        window is not full."""
        self.samples += 1
        self._window.append(value)
        if len(self._window) < self._ctx:
            return None
        arr = np.fromiter(self._window, float, count=self._ctx)
        older = arr[: self._ctx - self._agg]
        loc = float(older.mean())
        scale = max(float(older.std()), self._min_ratio * abs(loc), 1e-9)
        return (arr - loc) / scale, _Pending(value, loc, scale)

    def _finish(
        self, pending: _Pending, forecast: float, err_time: float, err_fft: float
    ) -> ModelOutput | None:
        """Combine the inference result of one sample into a score."""
        raw: dict[str, float | None] = {"time": err_time, "fft": err_fft, "pred": None}
        expected = None
        sigma = None
        prev_scale = 1.0
        if self._prev_forecast is not None:
            prev_fc, prev_scale = self._prev_forecast
            resid = (pending.value - prev_fc) / prev_scale
            raw["pred"] = resid**2
            expected = prev_fc
            if len(self._residuals) >= self._min_resid:
                r = np.fromiter(self._residuals, float)
                sigma = _MAD_TO_SIGMA * float(np.median(np.abs(r - np.median(r))))
            self._residuals.append(resid)
        self._prev_forecast = (pending.loc + pending.scale * forecast, pending.scale)

        head_scores: dict[str, float | None] = {}
        for h in HEADS:
            r_h = raw[h]
            head_scores[h] = None if r_h is None else self._scales[h].score(float(r_h))
        self.last_heads = head_scores
        chosen = [head_scores[h] for h in self._heads]
        if expected is None or sigma is None or any(s is None for s in chosen):
            return None
        half = self._band_k * max(sigma, 1e-6) * prev_scale
        return ModelOutput(max(chosen), expected, expected - half, expected + half)  # type: ignore[type-var]

    # -- public paths ----------------------------------------------------------------------

    def update(self, timestamp: float, value: float) -> ModelOutput | None:
        prep = self._prepare(value)
        if prep is None:
            return None
        window, pending = prep
        fc, et, ef = self._engine.infer(window[None, :], self._agg)
        return self._finish(pending, float(fc[0]), float(et[0]), float(ef[0]))

    def score_offline(
        self, values: np.ndarray, batch_size: int = 64
    ) -> tuple[list[ModelOutput | None], dict[str, np.ndarray]]:
        """The same computation on a whole series, with the forward passes batched.

        Returns the output per sample and, per head, its [0, 1] score (NaN where it has none)
        so that the heads can be compared. Samples must be finite and already filtered the
        way the pipeline filters them."""
        outs: list[ModelOutput | None] = []
        heads = {h: np.full(len(values), np.nan) for h in HEADS}
        chunk: list[tuple[int, np.ndarray, _Pending]] = []

        def flush() -> None:
            if not chunk:
                return
            fc, et, ef = self._engine.infer(np.stack([c[1] for c in chunk]), self._agg)
            for j, (idx, _, pend) in enumerate(chunk):
                out = self._finish(pend, float(fc[j]), float(et[j]), float(ef[j]))
                outs[idx] = out
                for h in HEADS:
                    s = self.last_heads[h]
                    if s is not None:
                        heads[h][idx] = s
            chunk.clear()

        for i, v in enumerate(values):
            outs.append(None)
            prep = self._prepare(float(v))
            if prep is None:
                continue
            chunk.append((i, prep[0], prep[1]))
            if len(chunk) >= batch_size:
                flush()
        flush()
        return outs, heads


def update_many(
    models: list[TSPulseModel], values: list[float]
) -> list[ModelOutput | None]:
    """One step for many series with a single forward pass (all series of one engine).

    This is what a sidecar with hundreds of series per interval should call."""
    engine = models[0]._engine
    agg = models[0]._agg
    preps = [m._prepare(v) for m, v in zip(models, values, strict=True)]
    ready = [i for i, p in enumerate(preps) if p is not None]
    outs: list[ModelOutput | None] = [None] * len(models)
    if not ready:
        return outs
    fc, et, ef = engine.infer(np.stack([preps[i][0] for i in ready]), agg)  # type: ignore[index]
    for j, i in enumerate(ready):
        outs[i] = models[i]._finish(preps[i][1], float(fc[j]), float(et[j]), float(ef[j]))  # type: ignore[index]
    return outs


def _factory(params: dict[str, Any], interval: float, seed: int) -> TSPulseModel:
    check_params("tspulse", params, TSPulseModel._ALLOWED)
    return TSPulseModel(**params)


register_model("tspulse", _factory)
