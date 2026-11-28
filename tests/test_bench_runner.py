"""SMD and PSM scoring jobs (the cheap models; tspulse and toto are tested in their own files)."""

from __future__ import annotations

import numpy as np

from anomaly_detection.harness.bench_cli import subset_names
from anomaly_detection.harness.bench_data import Machine
from anomaly_detection.harness.bench_runner import (
    fixed_scores,
    score_dim,
    stratified_subset,
)
from anomaly_detection.models import create_model


def test_fixed_scores_match_the_static_model():
    rng = np.random.default_rng(1)
    train = rng.random(200)
    test = rng.random(50) * 3 - 1
    test[3] = np.nan
    got = fixed_scores(train, test, 0.5)
    lo, hi = train.min(), train.max()
    model = create_model(
        "static", {"lower": lo - 0.5 * (hi - lo), "upper": hi + 0.5 * (hi - lo)}, 60.0
    )
    want = [np.nan if np.isnan(v) else model.update(0.0, float(v)).score for v in test]
    assert np.array_equal(np.nan_to_num(got, nan=-1), np.nan_to_num(want, nan=-1))
    assert np.nansum(got) > 0 and np.isnan(got[3])
    # margin 0 is tighter: everything the wide limits flag, it flags too
    assert (fixed_scores(train, test, 0.0)[~np.isnan(test)] >= got[~np.isnan(test)]).all()


def _series(n_train=3 * 1440, n_test=1440):
    t = np.arange(n_train + n_test)
    x = (
        0.5
        + 0.2 * np.sin(2 * np.pi * t / 1440)
        + 0.005 * np.random.default_rng(0).standard_normal(len(t))
    )
    x[n_train + 700 : n_train + 720] += 0.6
    return x[:n_train], x[n_train:], t[:n_train] * 60.0, t[n_train:] * 60.0


def test_seasonal_and_subpca_find_a_spike_and_skip_missing():
    tr, te, tr_t, te_t = _series()
    tr = tr.copy()
    tr[10] = np.nan  # missing warm-up value is skipped, not fed
    for name in ("seasonal", "subpca"):
        sc = score_dim(name, tr, te, tr_t, te_t)
        assert sc.shape == te.shape and sc.dtype == np.float32
        assert np.nanmax(sc[700:720]) > 0.5
        assert np.nanmean(sc[:600] >= 0.5) < 0.05


def test_stratified_subset_is_deterministic_proportional_and_nested():
    names = [f"machine-{g}-{i}" for g, n in ((1, 8), (2, 9), (3, 11)) for i in range(1, n + 1)]
    rates = [float(i % 7) for i in range(len(names))]
    s8 = stratified_subset(names, rates, 8)
    assert len(s8) == 8 and s8 == stratified_subset(names, rates, 8)
    assert sorted({n.rsplit("-", 1)[0] for n in s8}) == ["machine-1", "machine-2", "machine-3"]
    s3 = stratified_subset(s8, [rates[names.index(n)] for n in s8], 3)
    assert set(s3) <= set(s8) and len(s3) == 3


def test_subset_names_single_machine_is_whole():
    z = np.zeros((4, 2))
    m = Machine("psm", z, z, np.zeros(4, np.int8), np.arange(4.0), np.arange(4.0, 8))
    assert subset_names([m], "s3") == ["psm"]
