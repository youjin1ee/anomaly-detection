"""TSPulse model: state machine with a stand-in engine, and the real model when available.

The real model is not part of the repository. Set TSPULSE_MODEL_DIR to a directory with
config.json and model.safetensors (and install the `tspulse` extra) to run those tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from anomaly_detection.harness.nab_data import NabSeries
from anomaly_detection.harness.nab_runner import accepted_records
from anomaly_detection.models import available_models, create_model
from anomaly_detection.models.tspulse import (
    TSPulseModel,
    _HeadScale,
    update_many,
)

CTX = 64
AGG = 16


class FakeEngine:
    """Deterministic stand-in with the engine interface: naive forecast, and reconstruction
    errors from differences of neighbouring points."""

    context_length = CTX
    patch_length = 8

    def infer(self, windows: np.ndarray, aggregation_length: int):
        w = np.asarray(windows, dtype=float)
        forecast = w[:, -1] * 0.9
        tail = w[:, -aggregation_length:]
        err_time = ((tail - w[:, -aggregation_length - 1 : -1]) ** 2).mean(1)
        err_fft = ((tail - w.mean(1, keepdims=True)) ** 2).mean(1) * 0.5
        return forecast, err_time, err_fft


def make(**kw) -> TSPulseModel:
    return TSPulseModel(engine=FakeEngine(), aggregation_length=AGG, min_history=20, **kw)


def series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    v = 10 + np.sin(np.arange(n) / 7.0) + rng.normal(0, 0.1, n)
    v[300:310] += 8
    return v


def test_registered():
    assert "tspulse" in available_models()


def test_unknown_param_and_bad_heads_rejected():
    with pytest.raises(ValueError, match="unknown parameters"):
        create_model("tspulse", {"nope": 1}, 60.0)
    with pytest.raises(ValueError):
        make(heads=("bogus",))
    with pytest.raises(ValueError):
        TSPulseModel(engine=FakeEngine(), aggregation_length=10)


def test_warmup_and_first_score_index():
    m = make()
    outs = [m.update(float(i), float(v)) for i, v in enumerate(series())]
    first = next(i for i, o in enumerate(outs) if o is not None)
    # window of CTX samples, one sample for the first forecast to compare against, then the
    # min_history raw errors of the pred head
    assert first == CTX - 1 + 1 + 20
    assert all(o is None for o in outs[:first])


def test_output_ranges_and_band():
    m = make()
    outs = [o for i, v in enumerate(series()) if (o := m.update(float(i), float(v))) is not None]
    assert outs
    for o in outs:
        assert 0.0 <= o.score <= 1.0
        assert o.lower is not None and o.upper is not None
        assert o.lower < o.expected < o.upper


def test_jump_scores_high():
    m = make()
    scores = {}
    for i, v in enumerate(series()):
        o = m.update(float(i), float(v))
        if o is not None:
            scores[i] = o.score
    quiet = np.mean([s for i, s in scores.items() if i < 290])
    assert max(scores[i] for i in range(300, 306)) > 0.9 > quiet + 0.3


def test_offline_equals_streaming():
    v = series()
    stream = make()
    a = [stream.update(float(i), float(x)) for i, x in enumerate(v)]
    for batch in (1, 7, 64):
        b, heads = make().score_offline(v, batch_size=batch)
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert (x is None) == (y is None)
            if x is not None:
                assert x == y
        assert set(heads) == {"time", "fft", "pred"}


def test_update_many_equals_single():
    vs = [series(seed=s) for s in range(3)]
    singles = [make() for _ in vs]
    fleet = [make() for _ in vs]
    for i in range(len(vs[0])):
        one = [m.update(float(i), float(v[i])) for m, v in zip(singles, vs, strict=True)]
        many = update_many(fleet, [float(v[i]) for v in vs])
        assert one == many


def test_head_subset_changes_score_source():
    v = series()
    full, _ = make().score_offline(v)
    only_time, heads = make(heads=("time",)).score_offline(v)
    i = 305
    assert full[i] is not None and only_time[i] is not None
    assert full[i].score == pytest.approx(max(heads[h][i] for h in ("time", "fft", "pred")))
    assert only_time[i].score == pytest.approx(heads["time"][i])


def test_head_scale_maps_reference_to_low_and_large_to_high():
    h = _HeadScale(history=100, min_history=10, quantile=0.99, mid=2.0)
    assert all(h.score(1.0) is None for _ in range(10))
    assert h.score(1.0) == pytest.approx(0.1192, abs=1e-3)
    assert h.score(2.0) == pytest.approx(0.5)
    assert h.score(1000.0) > 0.99


def test_accepted_records_skips_nan_duplicates_and_backward():
    ts = np.array([0, 10, 10, 20, 15, 30, 40], dtype=float)
    vals = np.array([1, 2, 3, np.nan, 5, 6, 7], dtype=float)
    s = NabSeries("x/y.csv", ts, vals, ())
    assert accepted_records(s).tolist() == [0, 1, 4, 5, 6]


MODEL_DIR = os.environ.get("TSPULSE_MODEL_DIR")
needs_model = pytest.mark.skipif(not MODEL_DIR, reason="TSPULSE_MODEL_DIR is not set")


@pytest.fixture(scope="module")
def real_series():
    rng = np.random.default_rng(1)
    n = 700
    v = 50 + 5 * np.sin(np.arange(n) * 2 * np.pi / 96) + rng.normal(0, 0.5, n)
    v[640:650] -= 40
    return v


@needs_model
def test_real_model_is_deterministic_and_batch_independent():
    pytest.importorskip("tsfm_public")
    from anomaly_detection.models.tspulse import get_engine

    eng = get_engine(MODEL_DIR, 1)
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 512))
    a = eng.infer(w, 32)
    b = eng.infer(w, 32)
    c = eng.infer(w[2:3], 32)
    for x, y in zip(a, b, strict=True):
        assert np.allclose(x, y, atol=1e-5)
    for x, y in zip(a, c, strict=True):
        assert np.allclose(x[2], y[0], atol=1e-4)


@needs_model
def test_real_model_offline_equals_streaming(real_series):
    pytest.importorskip("tsfm_public")
    v = real_series
    stream = TSPulseModel(model_path=MODEL_DIR, threads=1)
    a = [stream.update(float(i), float(x)) for i, x in enumerate(v)]
    b, _ = TSPulseModel(model_path=MODEL_DIR, threads=1).score_offline(v, batch_size=16)
    scored = 0
    for x, y in zip(a, b, strict=True):
        assert (x is None) == (y is None)
        if x is not None:
            scored += 1
            assert x.score == pytest.approx(y.score, abs=1e-3)
            assert x.expected == pytest.approx(y.expected, abs=1e-3)
            assert x.lower == pytest.approx(y.lower, abs=1e-2)
    assert scored > 100
    # the drop at 640 is seen
    hit = max(o.score for o in b[640:650] if o is not None)
    assert hit > 0.9


@needs_model
def test_real_model_update_many_equals_single(real_series):
    pytest.importorskip("tsfm_public")
    vs = [real_series, real_series[::-1].copy()]
    singles = [TSPulseModel(model_path=MODEL_DIR, threads=1) for _ in vs]
    fleet = [TSPulseModel(model_path=MODEL_DIR, threads=1) for _ in vs]
    for i in range(580):
        one = [m.update(float(i), float(v[i])) for m, v in zip(singles, vs, strict=True)]
        many = update_many(fleet, [float(v[i]) for v in vs])
        for x, y in zip(one, many, strict=True):
            assert (x is None) == (y is None)
            if x is not None:
                assert x.score == pytest.approx(y.score, abs=1e-3)
