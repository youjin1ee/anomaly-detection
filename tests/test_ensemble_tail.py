import math

import numpy as np
import pytest

from anomaly_detection.ensemble.alerts import fast_alert_spans, hold
from anomaly_detection.ensemble.tail import (
    X_CAP,
    ScoreCalibrator,
    alert_peaks,
    budget_threshold,
    fit_gpd_pwm,
)
from anomaly_detection.harness.metrics import AlertRule, alert_spans
from anomaly_detection.harness.nab_runner import hold_for


def test_fast_alert_spans_equals_the_reference_rule():
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = rng.random(rng.integers(5, 120))
        s[rng.random(len(s)) < 0.05] = np.nan
        rule = AlertRule(
            float(rng.uniform(0.2, 0.9)), int(rng.integers(1, 5)), int(rng.integers(0, 12)), 5
        )
        first = int(rng.integers(0, len(s)))
        assert fast_alert_spans(s, rule, first) == alert_spans(s, rule, first)


def test_hold_equals_hold_for():
    x = np.random.default_rng(1).random(50)
    x[7] = np.nan
    assert np.allclose(hold(x, 3), hold_for(x, 3))


def test_gpd_fit_recovers_an_exponential_tail():
    y = np.random.default_rng(0).exponential(2.0, 5000)
    xi, sigma = fit_gpd_pwm(y)
    assert xi < 0.1 and sigma == pytest.approx(2.0, rel=0.1)


def test_surprise_extrapolates_above_the_calibration_maximum():
    rng = np.random.default_rng(0)
    cal = rng.exponential(1.0, 2000)
    c = ScoreCalibrator.fit(cal)
    big = np.array([cal.max(), cal.max() * 1.5, cal.max() * 2.0])
    x = c.surprise(big)
    assert x[0] < x[1] < x[2] <= X_CAP
    # exponential tail: p(5) = exp(-5) = 0.0067, so about 2.17
    assert c.surprise(np.array([5.0]))[0] == pytest.approx(2.17, abs=0.35)
    plain = ScoreCalibrator.fit(cal, extrapolate=False).surprise(big)
    assert plain[1] == plain[2]  # without a tail everything above the maximum ties


def test_calibrator_handles_binary_scores_and_short_windows():
    c = ScoreCalibrator.fit(np.zeros(500))
    assert not c.tail
    assert c.surprise(np.array([0.0, 1.0])).tolist() == pytest.approx(
        [-math.log10(1.0), math.log10(501)]
    )
    assert not ScoreCalibrator.fit(np.zeros(10)).ok
    assert np.isnan(c.surprise(np.array([np.nan]))[0])


def test_budget_threshold_orders_and_extrapolates():
    rng = np.random.default_rng(2)
    h = np.zeros(4000)
    for s in rng.integers(0, 3980, 30):  # 30 clusters in 3 days, exponential peaks above 1
        h[s : s + 4] = 1.0 + rng.exponential(0.5)
    t1 = budget_threshold(h, 3.0, 1.0)
    t01 = budget_threshold(h, 3.0, 0.1)
    assert t1 < t01 <= X_CAP
    assert t01 > alert_peaks(h, 1.0, 10).max() or t01 > t1  # beyond the window maximum
    flat = budget_threshold(h, 3.0, 0.1, extrapolate=False)
    assert flat == pytest.approx(np.nextafter(alert_peaks(h, 1.0, 10).max(), np.inf))
    assert budget_threshold(np.zeros(100), 3.0, 0.1) == pytest.approx(1.0 + math.log(1 / 0.3))
    assert budget_threshold(np.zeros(100), 3.0, 1e-12) == math.inf
