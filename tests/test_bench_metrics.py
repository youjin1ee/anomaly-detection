"""Multivariate benchmark metrics."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detection.harness.bench_metrics import (
    DEFAULT_RULE,
    average_precision,
    cause_hits,
    event_f1_at,
    oracle_event_f1,
    pa_f1_at,
    pa_oracle_f1,
    point_adjust_counts,
    server_scores,
)


def test_server_scores_is_max_and_keeps_all_nan_rows():
    x = np.array([[0.1, 0.9], [np.nan, 0.2], [np.nan, np.nan]])
    out = server_scores(x)
    assert out[0] == 0.9 and out[1] == 0.2 and np.isnan(out[2])


def test_average_precision():
    y = np.array([0, 0, 1, 1])
    assert average_precision(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    # ranking neg, pos, neg, pos: 0.5 * 1/2 + 0.5 * 2/4
    assert average_precision(np.array([0.9, 0.2, 0.1, 0.8]), y) == 0.5
    # equal scores are one step: precision is the base rate
    assert average_precision(np.full(4, 0.5), y) == 0.5
    assert np.isnan(average_precision(np.ones(3), np.zeros(3)))
    # NaN counts as 0
    assert average_precision(np.array([np.nan, 1.0]), np.array([0, 1])) == 1.0


def _case():
    y = np.zeros(100, dtype=np.int8)
    y[40:50] = 1
    s = np.full(100, 0.1)
    s[42:48] = 0.9  # found
    s[80:85] = 0.6  # false alarm at 0.6
    return s, y


def test_event_f1_at_operating_point():
    s, y = _case()
    summ = event_f1_at([(s, y)], 60.0, DEFAULT_RULE)
    assert (summ.events, summ.detected, summ.alerts, summ.false_alarms) == (1, 1, 2, 1)
    assert summ.precision == 0.5 and summ.recall == 1.0
    assert summ.median_delay_min == pytest.approx(4 / 1.0)  # fires at row 44, event starts at 40


def test_oracle_picks_threshold_that_removes_false_alarm():
    s, y = _case()
    f1, thr, summ = oracle_event_f1([(s, y)], 60.0)
    assert f1 == 1.0 and thr > 0.6 and summ.false_alarms == 0


def test_point_adjust_inflates():
    s = np.zeros(20)
    y = np.zeros(20, dtype=np.int8)
    y[5:15] = 1
    s[7] = 1.0  # one hit in a 10 row event
    assert point_adjust_counts(s, y, 0.5) == (10, 0, 0)
    assert pa_f1_at([(s, y)], 0.5) == 1.0
    f1, _ = pa_oracle_f1([(s, y)])
    assert f1 == 1.0


def test_cause_hits_counts_ties_as_expected_share():
    d = 4
    y = np.zeros(10, dtype=np.int8)
    y[3:6] = 1
    sc = np.zeros((10, d))
    sc[4, 1] = 0.9  # dimension 1 is the highest inside the event
    interp = ((3, 6, (1, 2)),)
    hits, n, chance = cause_hits(sc, y, interp, [0, 1, 2, 3])
    assert (hits, n) == (1.0, 1) and chance == 0.5
    sc[:] = 0.0  # all tied: dimensions 1 and 2 of 4 are right
    hits, n, _ = cause_hits(sc, y, interp, [0, 1, 2, 3])
    assert hits == 0.5
    # nothing reaches 0.5 here, so the event is skipped
    assert cause_hits(sc, y, interp, [0, 1, 2, 3], min_score=0.5)[1] == 0
    # an unusable listed dimension is dropped; an event with no usable listed dimension is skipped
    assert cause_hits(sc, y, ((3, 6, (3,)),), [0, 1, 2])[1] == 0
