"""NAB score reimplementation."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from anomaly_detection.harness.nab_data import load_corpus
from anomaly_detection.harness.nab_score import (
    PROFILES,
    FileEvents,
    best_threshold,
    normalize,
    null_and_perfect,
    probation_length,
    raw_at,
    scaled_sigmoid,
    threshold_curve,
)

STD = PROFILES["standard"]


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def test_scaled_sigmoid_reference_points():
    # values quoted in the NAB documentation of the scoring function
    assert float(scaled_sigmoid(-1.0)) == pytest.approx(0.98661, abs=1e-5)
    assert float(scaled_sigmoid(-0.5)) == pytest.approx(0.84828, abs=1e-5)
    assert float(scaled_sigmoid(0.0)) == 0.0
    assert float(scaled_sigmoid(1.0)) == pytest.approx(-0.98661, abs=1e-5)
    assert float(scaled_sigmoid(3.5)) == -1.0


def test_probation_length():
    assert probation_length(1000) == 150
    assert probation_length(4032) == 604
    assert probation_length(22695) == 750


def _naive_raw(scores, windows, profile, threshold):
    """Straightforward per-record reference written from the method description."""
    n = len(scores)
    start = probation_length(n)
    max_tp = 2 * _sig(5.0) - 1
    total = 0.0
    for _k, (s, e) in enumerate(windows):
        if e < start:
            continue
        width = e - s + 1
        best = -profile.fn_weight
        for i in range(max(s, start), e + 1):
            if scores[i] >= threshold:
                w = (2 * _sig(5.0 * (e - i + 1) / width) - 1) / max_tp * profile.tp_weight
                best = max(best, w)
        total += best
    for i in range(start, n):
        if any(s <= i <= e for s, e in windows) or scores[i] < threshold:
            continue
        prev = [e for _, e in windows if e < i]
        if not prev:
            total += -1.0 * profile.fp_weight
            continue
        pe = prev[-1]
        pw = next(e - s + 1 for s, e in windows if e == pe)
        pos = (i - pe) / (pw - 1)
        val = -1.0 if pos > 3.0 else 2 * _sig(-5.0 * pos) - 1
        total += val * profile.fp_weight
    return total


def test_single_window_perfect_and_null():
    n = 200
    windows = [(100, 119)]
    none = np.zeros(n)
    hit = np.zeros(n)
    hit[100] = 1.0
    ev_none = FileEvents(none, windows, STD)
    ev_hit = FileEvents(hit, windows, STD)
    null, perfect = null_and_perfect([ev_none], STD)
    assert (null, perfect) == (-1.0, 1.0)
    # nothing detected at a threshold above every score
    assert ev_none.raw(1.1) == -1.0
    # a detection at the left edge of the window gives the full weight
    assert ev_hit.raw(1.0) == pytest.approx(1.0)
    thr, raw = best_threshold([ev_hit])
    assert thr == 1.0
    assert normalize(raw, null, perfect) == pytest.approx(100.0)


def test_false_positive_weights():
    n = 400
    windows = [(100, 119)]  # width 20
    s = np.zeros(n)
    s[150] = 0.9  # 30 records after the right edge: position 30/19
    s[70] = 0.8  # before any window
    ev = FileEvents(s, windows, STD)
    pos = (150 - 119) / 19
    expected = -1.0 + (2 * _sig(-5 * pos) - 1) * 0.11 + -1.0 * 0.11
    assert ev.raw(0.8) == pytest.approx(expected)


def test_probationary_records_are_ignored():
    n = 400  # probation 60
    s = np.zeros(n)
    s[:60] = 1.0
    ev = FileEvents(s, [(200, 219)], STD)
    assert ev.raw(0.5) == ev.raw(1.1) == -1.0


def test_window_inside_probation_is_not_scored():
    n = 400
    ev = FileEvents(np.zeros(n), [(10, 20), (200, 210)], STD)
    assert ev.base == -1.0
    assert ev.n_windows_total == 2


@pytest.mark.parametrize("profile", list(PROFILES))
def test_vectorised_score_matches_reference(profile):
    rng = np.random.default_rng(7)
    n = 600
    windows = [(130, 160), (300, 330), (480, 500)]
    scores = np.round(rng.random(n), 2)  # ties on purpose
    ev = FileEvents(scores, windows, PROFILES[profile])
    thr, totals = threshold_curve([ev])
    assert thr[0] == 1.1
    for t, total in zip(thr, totals, strict=True):
        assert total == pytest.approx(_naive_raw(scores, windows, PROFILES[profile], t), abs=1e-9)
        assert raw_at([ev], t) == pytest.approx(total, abs=1e-9)


def test_corpus_threshold_is_shared_across_files():
    n = 500
    a = np.zeros(n)
    a[300] = 0.9
    b = np.zeros(n)
    b[300] = 0.4
    events = [FileEvents(a, [(290, 310)], STD), FileEvents(b, [(290, 310)], STD)]
    thr, raw = best_threshold(events)
    # 0.4 detects both windows, 0.9 only one; each window is a score of about 0.9 at its centre
    assert thr == pytest.approx(0.4)
    assert raw > raw_at(events, 0.9)


def test_profiles_weights():
    assert (PROFILES["standard"].fp_weight, PROFILES["standard"].fn_weight) == (0.11, 1.0)
    assert PROFILES["reward_low_FP_rate"].fp_weight == 0.22
    assert PROFILES["reward_low_FN_rate"].fn_weight == 2.0


@pytest.mark.skipif("NAB_DIR" not in os.environ, reason="set NAB_DIR to a NAB checkout")
def test_scorer_reproduces_official_results():
    from anomaly_detection.harness.nab_verify import verify

    nab = Path(os.environ["NAB_DIR"])
    corpus = load_corpus(nab)
    checks = verify(corpus, nab, ["null", "random", "numenta", "randomCutForest"])
    assert len(checks) == 12
    for c in checks:
        # same anomaly scores and the official threshold must give the official score
        assert c.score_at_official_threshold == pytest.approx(c.official_score, abs=1e-6), c
        # and our own threshold search cannot do worse than the official one
        assert c.score >= c.official_score - 1e-6, c
