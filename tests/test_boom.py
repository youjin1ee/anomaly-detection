import numpy as np
import pytest

from anomaly_detection.harness.boom_replay import (
    concurrency,
    fixed_limits,
    model_params,
    model_spec,
    run_boom,
    stratified_subset,
)
from anomaly_detection.harness.replay import RecordedSeries, SeriesAlerts


def test_parameter_rule():
    assert model_params("rcf", 60, 2880)["sample_size"] == 1024
    assert model_params("rcf", 300, 576)["sample_size"] == 526  # forest full at warmup end
    assert model_params("rcf", 300, 576)["norm_window"] == 288
    assert model_params("rcf", 60, 2880)["reference_window"] == 60
    assert model_params("awsrcf", 300, 576)["mode"] == "caster"
    assert model_params("awsrcf-nab", 60, 2880)["num_trees"] == 100
    assert fixed_limits(np.array([0.0, 4.0])) == {"lower": -2.0, "upper": 6.0}


def _series(values, step=60):
    ts = np.arange(len(values), dtype=np.int64) * step
    return RecordedSeries("q", "", ts, np.asarray(values, float))


def test_fixed_needs_a_day_of_warmup_and_fires_outside_the_limits():
    short = _series(np.zeros(500))
    assert np.isnan(run_boom(model_spec("fixed"), short, 60, 500).score).all()
    values = np.concatenate([np.tile([0.0, 1.0], 800), [50.0, 0.5]])
    raw = run_boom(model_spec("fixed"), _series(values), 60, 1600)
    assert raw.score[-2] == 1.0 and raw.score[-1] == 0.0  # limits are -0.5 and 1.5


def test_stratified_subset_one_variate_per_series_and_deterministic(tmp_path):
    from anomaly_detection.harness.boom_data import BoomMeta

    series, meta = [], {}
    for q in range(10):
        for k in range(3):
            s = RecordedSeries(f"ds-{q}", f'{{"variate": {k}}}', np.arange(3), np.zeros(3))
            series.append(s)
            meta[s.key] = BoomMeta("T" if q < 6 else "5T", 60, f"d{q % 3}", "gauge")
    a = stratified_subset(series, meta, {"T": 4, "5T": 3}, seed=1)
    b = stratified_subset(series, meta, {"T": 4, "5T": 3}, seed=1)
    assert [s.key for s in a] == [s.key for s in b]
    assert len({s.query for s in a}) == 7  # never two variates of one series
    assert sum(meta[s.key].freq == "T" for s in a) == 4
    assert len(stratified_subset(series, meta, {"T": 99}, seed=1)) == 6  # asks for more than exist


def test_concurrency_counts_series_in_the_same_ten_minute_window():
    ts = np.arange(100, dtype=np.int64) * 60
    by_key = {k: RecordedSeries(k, "", ts, np.zeros(100)) for k in "abc"}
    rows = [
        SeriesAlerts("a", "a", "", "m", 1.0, spans=[(10, 12, 14)]),
        SeriesAlerts("b", "b", "", "m", 1.0, spans=[(11, 13, 14), (50, 52, 54)]),
        SeriesAlerts("c", "c", "", "m", 1.0, spans=[]),
    ]
    out = concurrency(rows, by_key, {"a": 5, "b": 5, "c": 5})
    # fires at minutes 12 and 13 share the window 0-10 min? 12 min and 13 min are in window 1
    assert out["windows"] == 2 and out["max"] == 2
    assert out["max_share"] == pytest.approx(2 / 3)
    assert out["alerts_in_5plus"] == 0.0
