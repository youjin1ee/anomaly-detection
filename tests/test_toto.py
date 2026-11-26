import os

import numpy as np
import pytest

from anomaly_detection.models import available_models, create_model
from anomaly_detection.models.toto import (
    PATCH_SIZE,
    TotoModel,
    band_score,
    score_batch,
    window_length,
)


class FakeEngine:
    """Quantiles from the window mean and standard deviation. Records every window it sees."""

    z = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])

    def __init__(self):
        self.windows: list[np.ndarray] = []

    def forecast(self, windows):
        self.windows.extend(np.array(w) for w in windows)
        mu = windows.mean(axis=1, keepdims=True)
        sd = windows.std(axis=1, keepdims=True)
        return mu + sd * self.z[None, :]


def series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return 100 + 5 * np.sin(np.arange(n) / 9.0) + rng.normal(0, 1, n)


def test_registered_and_needs_model_dir():
    assert "toto" in available_models()
    with pytest.raises(ValueError, match="model_dir"):
        create_model("toto", {}, 60.0)
    with pytest.raises(ValueError, match="unknown parameters"):
        create_model("toto", {"model_dir": "x", "nope": 1}, 60.0)


@pytest.mark.parametrize(
    "kwargs",
    [{"context_length": 100}, {"min_context": 100}, {"context_length": 64, "min_context": 96}],
)
def test_lengths_must_be_whole_patches(kwargs):
    with pytest.raises(ValueError):
        TotoModel(engine=FakeEngine(), **kwargs)


def test_window_length_is_whole_patches_and_capped():
    assert window_length(127, 512) == 96
    assert window_length(128, 512) == 128
    assert window_length(1000, 512) == 512
    assert all(window_length(n, 512) % PATCH_SIZE == 0 for n in range(0, 700))


def test_warmup_returns_nothing_then_a_band():
    m = TotoModel(engine=FakeEngine(), context_length=256, min_context=64)
    outs = [m.update(float(i), v) for i, v in enumerate(series(100))]
    assert all(o is None for o in outs[:64])
    assert all(o is not None for o in outs[64:])
    o = outs[80]
    assert o.lower < o.expected < o.upper


def test_the_model_never_sees_the_value_it_scores():
    eng = FakeEngine()
    m = TotoModel(engine=eng, context_length=128, min_context=64)
    x = series(100)
    for i, v in enumerate(x):
        m.update(float(i), v)
    # the first window is the first 64 values, the value at index 64 is not in it
    assert len(eng.windows[0]) == 64
    np.testing.assert_allclose(eng.windows[0], x[:64].astype(np.float32))
    assert all(len(w) % PATCH_SIZE == 0 and len(w) <= 128 for w in eng.windows)


def test_band_score_mapping():
    q = np.array([[0.0, 1, 2, 3, 5, 6, 7, 8, 10]])  # band [0, 10], width 10, median 5
    score, expected, lower, upper = band_score(np.array([5.0]), q, 1.0, 0.02)
    assert (score[0], expected[0], lower[0], upper[0]) == (0.0, 5.0, 0.0, 10.0)
    inside = band_score(np.array([10.0]), q, 1.0, 0.02)[0]
    assert inside[0] == 0.0  # on the edge
    one_width = band_score(np.array([20.0]), q, 1.0, 0.02)[0]
    assert one_width[0] == pytest.approx(0.5)
    below = band_score(np.array([-10.0]), q, 1.0, 0.02)[0]
    assert below[0] == pytest.approx(0.5)
    far = band_score(np.array([1e6]), q, 1.0, 0.02)[0]
    assert 0.99 < far[0] <= 1.0
    # a score_mid of 2 halves the deviation weight
    assert band_score(np.array([20.0]), q, 2.0, 0.02)[0][0] == pytest.approx(1 / 3)


def test_flat_band_has_a_width_floor():
    q = np.full((1, 9), 100.0)
    score = band_score(np.array([100.5]), q, 1.0, 0.02)[0]
    # floor 2.0, deviation 0.5 -> r 0.25
    assert score[0] == pytest.approx(0.25 / 1.25)


def test_batch_path_matches_the_streaming_path():
    x = series(500)
    kw = {"context_length": 192, "min_context": 64}
    m = TotoModel(engine=FakeEngine(), **kw)
    stream = [m.update(float(i), v) for i, v in enumerate(x)]
    batch = score_batch(FakeEngine(), x, batch_size=37, **kw)
    for i, o in enumerate(stream):
        if o is None:
            assert np.isnan(batch["score"][i])
            continue
        np.testing.assert_allclose(
            [batch["score"][i], batch["expected"][i], batch["lower"][i], batch["upper"][i]],
            [o.score, o.expected, o.lower, o.upper],
            rtol=1e-5,
            atol=1e-6,
        )
    assert np.isnan(batch["score"][:64]).all() and np.isfinite(batch["score"][64:]).all()


def test_spike_scores_high_and_normal_low():
    x = series(300)
    x[250] += 60
    s = score_batch(FakeEngine(), x, context_length=192, min_context=64)["score"]
    assert s[250] > 0.8
    assert np.nanmedian(s) < 0.05


@pytest.mark.skipif(
    not os.environ.get("TOTO_MODEL_DIR"), reason="set TOTO_MODEL_DIR to a Toto 2.0 checkpoint"
)
def test_real_model_batch_equals_streaming():
    pytest.importorskip("toto2")
    from anomaly_detection.models.toto import get_engine

    eng = get_engine(os.environ["TOTO_MODEL_DIR"], threads=int(os.environ.get("TOTO_THREADS", 4)))
    x = series(190, seed=3)
    x[170] += 40
    kw = {"context_length": 96, "min_context": 64}
    m = TotoModel(engine=eng, **kw)
    stream = [m.update(float(i), v) for i, v in enumerate(x)]
    batch = score_batch(eng, x, batch_size=16, **kw)
    assert all(o is None for o in stream[:64])
    for i in range(64, len(x)):
        o = stream[i]
        scale = max(abs(o.upper - o.lower), 1.0)
        for key, ref in (("lower", o.lower), ("expected", o.expected), ("upper", o.upper)):
            assert abs(batch[key][i] - ref) < 1e-3 * scale, (i, key)
        assert abs(batch["score"][i] - o.score) < 1e-3, i
    assert batch["score"][170] > 0.5  # the injected spike is far outside the band

