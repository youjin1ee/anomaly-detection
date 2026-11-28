"""SMD and PSM loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anomaly_detection.harness.bench_data import (
    label_events,
    load_psm,
    load_smd,
    parse_interpretation,
)


def test_label_events():
    assert label_events(np.array([0, 1, 1, 0, 0, 1, 0, 1, 1])) == [(1, 2), (5, 5), (7, 8)]
    assert label_events(np.zeros(4)) == []


def test_parse_interpretation_is_zero_based_with_exclusive_end():
    got = parse_interpretation("15-18:1,9\n20-21:38\n")
    assert got == ((15, 18, (0, 8)), (20, 21, (37,)))


def _write_smd(root: Path) -> None:
    for d in ("train", "test", "test_label", "interpretation_label"):
        (root / d).mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in ("machine-1-1", "machine-2-1"):
        np.savetxt(root / "train" / f"{name}.txt", rng.random((20, 3)), delimiter=",")
        te = rng.random((12, 3))
        te[:, 2] = 0.5  # constant dimension
        np.savetxt(root / "test" / f"{name}.txt", te, delimiter=",")
        np.savetxt(root / "test_label" / f"{name}.txt", [0] * 5 + [1] * 3 + [0] * 4, fmt="%d")
        (root / "interpretation_label" / f"{name}.txt").write_text("5-8:1,2\n")


def test_load_smd(tmp_path):
    _write_smd(tmp_path)
    ms = load_smd(tmp_path)
    assert [m.name for m in ms] == ["machine-1-1", "machine-2-1"]
    m = ms[0]
    assert m.train.shape == (20, 3) and m.test.shape == (12, 3) and m.dims == 3
    assert m.test_ts[0] == 20 * 60.0 and m.train_ts[1] - m.train_ts[0] == 60.0
    assert m.interpretation == ((5, 8, (0, 1)),)
    assert label_events(m.labels) == [(5, 7)]
    assert load_smd(tmp_path, ["machine-2-1"])[0].name == "machine-2-1"
    with pytest.raises(ValueError, match="unknown"):
        load_smd(tmp_path, ["machine-9-9"])


def test_constant_dims_counts_train_and_test_together(tmp_path):
    _write_smd(tmp_path)
    m = load_smd(tmp_path)[0]
    # the test column 2 is constant but the train column varies, so it is not constant
    assert m.constant_dims() == []
    flat = np.zeros((5, 2))
    from anomaly_detection.harness.bench_data import Machine

    both = Machine("x", flat, flat.copy(), np.zeros(5, np.int8), np.arange(5.0), np.arange(5.0, 10))
    assert both.constant_dims() == [0, 1]


def test_load_psm_drops_duplicate_minute_and_keeps_nan(tmp_path):
    (tmp_path / "train.csv").write_text(
        "timestamp_(min),f0,f1\n0.0,1.0,\n1.0,2.0,3.0\n2.0,4.0,5.0\n"
    )
    (tmp_path / "test.csv").write_text("timestamp_(min),f0,f1\n2.0,7.0,8.0\n3.0,9.0,1.0\n")
    (tmp_path / "test_label.csv").write_text("timestamp_(min),label\n2.0,0\n3.0,1\n")
    (m,) = load_psm(tmp_path)
    assert m.train_ts.tolist() == [0.0, 60.0] and np.isnan(m.train[0, 1])
    assert m.test_ts.tolist() == [120.0, 180.0] and m.labels.tolist() == [0, 1]
    assert m.train_ts[-1] < m.test_ts[0]
