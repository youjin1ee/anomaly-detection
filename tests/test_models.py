import math

import numpy as np
import pytest

from anomaly_detection.models import ModelOutput, available_models, create_model, register_model
from anomaly_detection.models.static import StaticThreshold

STEP = 300.0
DAY = int(86400 / STEP)


def synth(days: int, seed: int = 1, weekend: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = days * DAY
    t = np.arange(n)
    daily = 1 + 0.4 * np.sin(2 * np.pi * (t % DAY) / DAY - 1.5)
    week = np.where((t // DAY) % 7 >= 5, weekend, 1.0)
    return 100 * daily * week * (1 + rng.normal(0, 0.04, n))


def feed(model, values, start=0):
    outs = []
    for i, v in enumerate(values):
        outs.append(model.update((start + i) * STEP, float(v)))
    return outs


def test_registry_lists_builtin_and_rejects_unknown():
    assert {"seasonal", "rcf", "static"} <= set(available_models())
    with pytest.raises(ValueError, match="unknown model"):
        create_model("nope", {}, 60)
    with pytest.raises(ValueError, match="unknown parameters"):
        create_model("seasonal", {"bogus": 1}, 60)


def test_registry_accepts_a_new_model_kind():
    """Forecast-band models plug in through register_model without core changes."""

    class Const:
        def update(self, timestamp, value):
            return ModelOutput(0.0, 1.0, 0.0, 2.0)

    register_model("const-test", lambda params, interval, seed: Const())
    m = create_model("const-test", {}, 60)
    assert m.update(0, 1.0).upper == 2.0


def test_seasonal_returns_none_until_enough_history():
    m = create_model("seasonal", {"seasonality": ["daily"]}, STEP)
    outs = feed(m, synth(3))
    assert outs[0] is None
    assert outs[DAY - 1] is None  # first day: nothing to compare with
    assert outs[DAY + DAY // 2] is None  # one day gives only 2*slot_window+1 = 17 points
    assert outs[2 * DAY + DAY // 2] is not None  # two days are enough for min_points=18


def test_seasonal_quiet_on_normal_data_and_flags_a_drop():
    m = create_model("seasonal", {"seasonality": ["daily"]}, STEP)
    outs = feed(m, synth(10))
    normal = [o.score for o in outs[5 * DAY :] if o]
    # heavy-tailed pooled estimates allow a rare crossing, but not a stream of them
    assert sum(x > 0.5 for x in normal) <= 0.005 * len(normal)
    start = 10 * DAY
    test = synth(11)[start : start + DAY]
    test[100:112] *= 0.2
    outs2 = feed(m, test, start)
    assert all(o.score > 0.9 for o in outs2[100:112])


def test_seasonal_weekly_beats_daily_on_weekend_pattern():
    daily = create_model("seasonal", {"seasonality": ["daily"]}, STEP)
    both = create_model("seasonal", {"seasonality": ["daily", "weekly"]}, STEP)
    values = synth(28, weekend=0.5)
    od = feed(daily, values)
    ob = feed(both, values)
    fp_daily = sum(1 for o in od[-7 * DAY :] if o and o.score > 0.5)
    fp_both = sum(1 for o in ob[-7 * DAY :] if o and o.score > 0.5)
    assert fp_both < fp_daily


def test_seasonal_gaps_do_not_break_scoring():
    m = create_model("seasonal", {"seasonality": ["daily"]}, STEP)
    values = synth(6)
    outs = []
    for i, v in enumerate(values):
        if 2 * DAY < i < 2 * DAY + 40:
            continue  # missing samples stay missing
        outs.append(m.update(i * STEP, float(v)))
    assert outs[-1] is not None and math.isfinite(outs[-1].score)


def test_flat_history_does_not_explode():
    m = create_model("seasonal", {"seasonality": ["daily"]}, STEP)
    outs = feed(m, np.full(5 * DAY, 10.0))
    assert max(o.score for o in outs if o) < 0.5


def test_static_threshold():
    m = StaticThreshold(lower=5.0, upper=None)
    assert m.update(0, 4.0).score == 1.0
    assert m.update(1, 6.0).score == 0.0
    assert m.update(2, 4.0).expected == 5.0


def test_rcf_warmup_then_scores_and_reacts_to_a_level_drop():
    params = {"num_trees": 20, "sample_size": 128, "norm_window": 600}
    m = create_model("rcf", params, STEP, seed=3)
    outs = feed(m, synth(6))
    assert outs[10] is None
    ready = [i for i, o in enumerate(outs) if o is not None]
    assert ready and ready[0] > 128  # waits for a full sample
    normal = [o.score for o in outs[-DAY:] if o]
    start = 6 * DAY
    test = synth(7)[start : start + 60]
    test[20:40] = 1.0
    outs2 = feed(m, test, start)
    peak = max(o.score for o in outs2[20:30] if o)
    assert peak > max(normal)
    assert peak > 0.5


def test_rcf_is_reproducible_with_the_same_seed():
    p = {"num_trees": 10, "sample_size": 64}
    a = feed(create_model("rcf", p, STEP, seed=7), synth(3))
    b = feed(create_model("rcf", p, STEP, seed=7), synth(3))
    assert [o.score if o else None for o in a] == [o.score if o else None for o in b]


def test_rcf_handles_constant_series():
    m = create_model("rcf", {"num_trees": 10, "sample_size": 32}, STEP)
    outs = feed(m, np.full(400, 5.0))
    assert all(math.isfinite(o.score) for o in outs if o)


def _bursty_idle_history(days=3):
    """Idle series with one burst of 50 per day, at a different time each day."""
    values = np.zeros(days * DAY)
    for d, offset in enumerate((36, 108, 216)[:days]):
        values[d * DAY + offset : d * DAY + offset + 6] = 50.0
    return values


def _score_burst(params, burst_value):
    m = create_model("seasonal", {"seasonality": ["daily"], **params}, STEP)
    history = _bursty_idle_history()
    feed(m, history)
    # day four, 12:00: idle at this time of day in every earlier day
    t = len(history) + DAY // 2
    out = None
    for i in range(len(history), t + 1):
        out = m.update(i * STEP, burst_value if i == t else 0.0)
    return out


def test_seasonal_idle_scale_ratio_accepts_a_burst_of_known_size():
    assert _score_burst({}, 50.0).score > 0.9  # default: any burst from idle is an anomaly
    kept = _score_burst({"idle_scale_ratio": 0.2}, 50.0)
    assert kept.score < 0.5  # a burst as large as earlier ones is not
    assert _score_burst({"idle_scale_ratio": 0.2}, 150.0).score > 0.5  # a much larger one is


def test_seasonal_idle_scale_ratio_does_not_touch_busy_slots():
    def collapse(params):
        m = create_model("seasonal", {"seasonality": ["daily"], **params}, STEP)
        values = 100 + np.random.default_rng(0).normal(0, 1, 3 * DAY)
        feed(m, values)
        return m.update(3 * DAY * STEP, 0.0)

    assert collapse({"idle_scale_ratio": 0.2}).score > 0.99
    assert collapse({"idle_scale_ratio": 0.2}).score == collapse({}).score


def test_rcf_rebuilds_a_forest_that_exceeds_the_recursion_limit(monkeypatch):
    import rrcf

    m = create_model("rcf", {"sample_size": 64, "num_trees": 3, "shingle_size": 2}, STEP)
    calls = {"n": 0}
    real = rrcf.RCTree.insert_point

    def flaky(self, point, index, tolerance=None):
        calls["n"] += 1
        if calls["n"] == 50:
            raise RecursionError
        return real(self, point, index, tolerance)

    monkeypatch.setattr(rrcf.RCTree, "insert_point", flaky)
    outs = feed(m, synth(2, seed=4)[:500])
    assert m.resets == 1
    assert outs[-1] is not None  # scores again once the new forest has refilled
