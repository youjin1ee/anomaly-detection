import math

import numpy as np
import pytest

from anomaly_detection.ensemble.combine import combine
from anomaly_detection.ensemble.core import (
    FORCED,
    EnsembleConfig,
    EnsembleSpec,
    prepare_series,
    score01,
    tier_of,
    unit_score,
)
from anomaly_detection.ensemble.rules import (
    RuleConfig,
    SeriesInfo,
    apply_rules,
    expected_series,
    series_scale,
    suppressed,
)


def test_combine_methods_with_missing_scores():
    a, b, c = np.array([1.0, 5.0, np.nan]), np.array([3.0, 1.0, np.nan]), np.array([2.0, 2.0, 4.0])
    assert combine([a, b, c], "max").tolist() == [3.0, 5.0, 4.0]
    assert combine([a, b, c], "mean").tolist() == pytest.approx([2.0, 8 / 3, 4.0])
    v = combine([a, b, c], "vote", 2)
    assert v[:2].tolist() == [2.0, 2.0] and np.isnan(v[2])


def test_rule_arrays_equal_the_single_sample_rule():
    rng = np.random.default_rng(3)
    spd = 96
    t = np.arange(spd * 6)
    v = 50 + 10 * np.sin(2 * np.pi * t / spd) + rng.normal(0, 2, len(t))
    v[400:420] = 0.0
    info = SeriesInfo("latency", spd)
    cfg = RuleConfig()
    cal = (spd, 3 * spd)
    res = apply_rules(v, info, cal, cfg)
    exp, scale = expected_series(v, spd, cfg.points), series_scale(v, cal)
    for i in range(len(v)):
        assert res.suppress[i] == suppressed(v[i], exp[i], "above", scale, cfg)
    assert res.suppress.sum() > 0 and not res.force.any()


def test_zero_drop_and_sparse_rules():
    spd = 48
    v = np.full(spd * 4, 10.0)
    v[150:160] = 0.0
    r = apply_rules(v, SeriesInfo("request", spd, None, True), (spd, 2 * spd), RuleConfig())
    assert r.force[150:160].all() and r.force.sum() == 10
    # without a real zero the rule is off, and a series that was 0 yesterday is not forced
    assert not apply_rules(v, SeriesInfo("request", spd), (spd, 2 * spd), RuleConfig()).force.any()
    v2 = np.full(spd * 4, 0.01)  # 0.01 per second = 0.6 per minute, below 5
    assert apply_rules(v2, SeriesInfo("request", spd, 60.0), (spd, 2 * spd), RuleConfig()).excluded
    assert not apply_rules(
        v, SeriesInfo("request", spd, 60.0), (spd, 2 * spd), RuleConfig()
    ).excluded


def _scores(n, spd=96, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    v = 20 + 5 * np.sin(2 * np.pi * t / spd) + rng.normal(0, 0.5, n)
    a = np.clip(rng.beta(2, 20, n), 0, 1)  # model scores, mostly small
    b = np.clip(rng.beta(2, 30, n), 0, 1)
    return v, {"a": a, "b": b}


def test_unit_score_thresholds_and_forced_alert():
    spd = 96
    v, raws = _scores(spd * 8)
    cfg = EnsembleConfig()
    cal = (spd, 4 * spd)
    prep = [prepare_series(raws, v, SeriesInfo("request", spd, None, True), cal, cfg)]
    sc = unit_score(prep, EnsembleSpec("or", ("a", "b")), cal, cfg)
    assert sc.thresholds[0] <= sc.thresholds[1]
    assert not any(sc.silent)
    v2 = v.copy()
    v2[600:610] = 0.0
    prep = [prepare_series(raws, v2, SeriesInfo("request", spd, None, True), cal, cfg)]
    sc = unit_score(prep, EnsembleSpec("or", ("a", "b"), rules=True), cal, cfg)
    assert (sc.x[600:610] == FORCED).all()


def test_two_stage_score_mapping():
    assert score01(0.0, 2.0, 3.0) == 0.0
    assert score01(2.0, 2.0, 3.0) == pytest.approx(0.5)
    assert score01(3.0, 2.0, 3.0) == pytest.approx(0.9)
    assert score01(1.0, 2.0, 3.0) < score01(2.5, 2.0, 3.0) < score01(5.0, 2.0, 3.0) < 1.0
    assert score01(FORCED, 2.0, 3.0) == 1.0
    assert [tier_of(x, 2.0, 3.0) for x in (1.0, 2.0, 3.5)] == [0, 1, 2]
    assert tier_of(FORCED, math.inf, math.inf) == 2  # a forced alert passes a silent stage
