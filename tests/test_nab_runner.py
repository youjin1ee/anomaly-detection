"""NAB per-file model rule and event metrics."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection.harness.metrics import AlertRule
from anomaly_detection.harness.nab_data import NabSeries
from anomaly_detection.harness.nab_runner import (
    estimate_interval,
    evaluate_events,
    fixed_limits,
    hold_for,
    model_params,
    score_series,
)
from anomaly_detection.harness.nab_score import (
    PROFILES,
    best_threshold,
    build_events,
    normalize,
    null_and_perfect,
)

STD = PROFILES["standard"]


def test_estimate_interval_ignores_duplicates_and_gaps():
    ts = np.array([0, 300, 300, 600, 900, 4500, 4800], dtype=float)
    assert estimate_interval(ts) == 300.0


def test_seasonal_rule_follows_resolution():
    assert model_params("seasonal", 300.0, 4000) == {"slot_window": 8, "min_points": 18}
    assert model_params("seasonal", 60.0, 4000)["slot_window"] == 8
    assert model_params("seasonal", 1800.0, 4000) == {"slot_window": 1, "min_points": 6}
    assert model_params("seasonal", 3600.0, 1600) == {"slot_window": 1, "min_points": 6}


def test_rcf_rule_scales_with_probation():
    small = model_params("rcf", 3600.0, 1538)  # probation 230
    assert small["sample_size"] == 180
    assert small["norm_window"] == 288
    big = model_params("rcf", 300.0, 22695)  # probation 750
    assert big["sample_size"] == 700 and big["norm_window"] == 288
    assert model_params("rcf", 300.0, 400000)["sample_size"] == 700
    assert model_params("rcf-time", 300.0, 4032)["time_features"] is True
    with pytest.raises(ValueError):
        model_params("nope", 300.0, 1000)


def test_fixed_limits_use_probation_only():
    v = np.concatenate([np.linspace(-10, 10, 100), np.full(100, 1000.0)])
    lim = fixed_limits(v, 100)
    assert lim == {"lower": -20.0, "upper": 20.0}


def _daily(n=4032, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.arange(n) * 300.0
    values = 50 + 20 * np.sin(2 * np.pi * ts / 86400.0) + rng.normal(0, 1.0, n)
    values[3000:3100] = 0.0  # flatline anomaly
    return NabSeries("artificialWithAnomaly/synthetic.csv", ts, values, ((2950, 3150),))


def test_seasonal_detects_flatline_and_scores_high():
    s = _daily()
    scores, us = score_series("seasonal", s)
    assert us > 0
    assert np.nanmax(scores[3000:3100]) > 0.9
    res = evaluate_events(s, scores, AlertRule())
    assert (res.events, res.detected) == (1, 1)
    assert res.delays[0] > 0
    ev = build_events([scores], [s.windows], STD)
    _thr, raw = best_threshold(ev)
    null, perfect = null_and_perfect(ev, STD)
    assert normalize(raw, null, perfect) > 50


def test_fixed_model_flags_out_of_range():
    s = _daily()
    scores, _ = score_series("fixed", s)
    assert scores[3050] == 1.0
    assert scores[1000] == 0.0


def test_hold_for_is_the_for_clause():
    x = np.array([0.9, 0.9, 0.2, 0.9, 0.9, 0.9, np.nan, 0.9, 0.9])
    got = hold_for(x, 3)
    assert got.tolist() == [0.0, 0.0, 0.2, 0.2, 0.2, 0.9, 0.0, 0.0, 0.0]
