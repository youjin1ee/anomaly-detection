import numpy as np
import pytest

from anomaly_detection.models import create_model


def wave(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10 + 3 * np.sin(2 * np.pi * t / 40) + rng.normal(0, 0.2, n)


def feed(model, values):
    return [model.update(float(i), float(v)) for i, v in enumerate(values)]


def test_silent_until_trained_then_quiet_on_normal_data():
    m = create_model("subpca", {"window": 16, "min_train": 100}, 1.0)
    outs = feed(m, wave(600))
    assert all(o is None for o in outs[:100])
    scored = [o.score for o in outs[200:]]
    assert max(scored) < 0.5
    assert abs(outs[-1].expected - wave(600)[-1]) < 1.5  # reconstruction follows the wave


def test_shape_break_scores_high_without_leaving_the_range():
    x = wave(600)
    x[500:510] = 10.0  # flat stretch inside the normal range, only the shape is wrong
    outs = feed(create_model("subpca", {"window": 16, "min_train": 100}, 1.0), x)
    assert max(o.score for o in outs[500:516]) > 0.5


def test_excluding_high_scores_keeps_an_anomaly_from_becoming_normal():
    x = wave(1200)
    burst = 10 + 3 * np.sin(2 * np.pi * np.arange(30) / 8)  # wrong frequency
    x[400:430] = burst
    x[800:830] = burst  # the same break again
    kw = {"window": 16, "min_train": 100, "retrain_every": 10, "exclude_max_run": 64}
    peak = {}
    for name, excl in (("on", 0.5), ("off", None)):
        m = create_model("subpca", {**kw, "exclude_score": excl}, 1.0)
        outs = feed(m, x)
        peak[name] = max(o.score for o in outs[800:846])
    assert peak["on"] > 0.6
    assert peak["off"] < 0.5  # learned the burst as normal


def test_rejects_bad_parameters():
    with pytest.raises(ValueError, match="unknown parameters"):
        create_model("subpca", {"bogus": 1}, 1.0)
    with pytest.raises(ValueError, match="n_components"):
        create_model("subpca", {"n_components": 2.5}, 1.0)
