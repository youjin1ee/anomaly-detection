import gzip

import numpy as np

from anomaly_detection.harness.metrics import AlertRule, alert_spans, alerts
from anomaly_detection.harness.replay import (
    Guards,
    RawOutput,
    RecordedSeries,
    alert_context,
    cached_run,
    evaluate_series,
    guarded_scores,
    load_dumps,
    run_model,
    summarize_model,
)
from anomaly_detection.harness.runner import parse_model
from anomaly_detection.pipeline import apply_guards


def _write(path, rows, gz=False):
    text = "timestamp,labels,value\n" + "\n".join(f'{t},"{lab}",{v}' for t, lab, v in rows) + "\n"
    if gz:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        path.write_text(text)


def test_load_dumps_grid_missing_and_labels(tmp_path):
    _write(
        tmp_path / "a_rate.csv.gz", [(0, '{""x"":""1""}', 1), (120, '{""x"":""1""}', 3)], gz=True
    )
    _write(tmp_path / "b.csv", [(60, "", 5), (180, "", "nan")])
    s = {x.query: x for x in load_dumps([tmp_path], 60)}
    assert set(s) == {"a_rate", "b"}
    assert list(s["a_rate"].ts) == [0, 60, 120]  # one grid for the whole dump
    assert np.isnan(s["a_rate"].values[1])  # a missing step is NaN, not zero
    assert s["a_rate"].values[2] == 3
    assert len(s["b"].values) == 3 and np.isnan(s["b"].values[2])  # non finite rows dropped
    assert s["a_rate"].labels == '{"x":"1"}'


def test_alert_spans_match_alerts():
    s = np.zeros(30)
    s[5:9] = 1
    spans = alert_spans(s, AlertRule(for_points=3))
    assert spans == [(5, 7, 8)]
    assert alerts(s, AlertRule(for_points=3)) == [7]


def test_apply_guards_matches_pipeline_rules():
    assert apply_guards(1.0, 0.0, 3.0, "both", min_expected=10) == 0.0
    assert apply_guards(1.0, 9.0, 3.0, "both", min_expected=10) == 1.0
    assert apply_guards(1.0, 96.0, 100.0, tolerance_amount=5) == 0.0
    assert apply_guards(1.0, 150.0, 100.0, "below") == 0.0


def test_alert_context_classes():
    idle = np.zeros(100)
    idle[60:] = 40
    assert alert_context(idle, 60, 0.1) == "idle_to_burst"
    busy = np.full(100, 40.0)
    busy[60:] = 0
    assert alert_context(busy, 60, 0.1) == "burst_end"
    gap = np.full(100, 40.0)
    gap[50:58] = np.nan
    gap[60] = 90
    assert alert_context(gap, 60, 0.1) == "gap_resume"
    up = np.full(100, 40.0)
    up[60:] = 90
    assert alert_context(up, 60, 0.1) == "other_up"
    down = np.full(100, 40.0)
    down[60:] = 10
    assert alert_context(down, 60, 0.1) == "other_down"
    assert alert_context(up, 0, 0.1) == "no_history"


def test_guards_suppress_scores_on_stored_output():
    values = np.array([0.0, 0.5, 100.0, 0.0])
    raw = RawOutput(np.ones(4), np.array([0.0, 0.0, 0.0, 100.0]))
    off = guarded_scores(raw, values, Guards(), scale=100.0)
    assert list(off) == [0.0, 1.0, 1.0, 1.0]  # no deviation, burst, burst, collapse
    on = guarded_scores(raw, values, Guards(tolerance_frac=0.05), scale=100.0)
    assert list(on) == [0.0, 0.0, 1.0, 1.0]  # a 0.5 wiggle on a scale of 100 is ignored


def _series(values, query="q"):
    ts = np.arange(len(values), dtype=np.int64) * 60
    return RecordedSeries(query, "", ts, np.asarray(values, float))


def test_run_model_skips_missing_and_caches(tmp_path):
    n = 3 * 1440
    rng = np.random.default_rng(1)
    values = 100 + rng.normal(0, 1, n)
    values[2000:2010] = np.nan
    s = _series(values)
    spec = parse_model('seasonal=seasonal:seasonality=["daily"]')
    raw = run_model(spec, s, 60, warmup_end=1440)
    assert np.isnan(raw.score[2000:2010]).all()
    assert np.isfinite(raw.score[3500])
    a = cached_run(spec, s, 60, 1440, tmp_path)
    b = cached_run(spec, s, 60, 1440, tmp_path)  # second call reads the file
    assert len(list(tmp_path.glob("*.npz"))) == 1
    assert np.array_equal(a.score, b.score, equal_nan=True)


def test_fixed_model_needs_history():
    s = _series(np.full(500, 3.0))
    raw = run_model(parse_model("fixed"), s, 60, warmup_end=300)
    assert np.isnan(raw.score).all()


def test_evaluate_counts_alerts_after_warmup_only():
    n = 3 * 1440
    values = np.full(n, 50.0)
    values[100:110] = 500  # inside warmup: not counted
    values[3000:3010] = 500
    s = _series(values)
    raw = run_model(parse_model('seasonal=seasonal:seasonality=["daily"]'), s, 60, 1440)
    res = evaluate_series(s, raw, "seasonal", 60, 1440, Guards(), AlertRule())
    assert len(res.spans) == 1 and res.spans[0][0] >= 3000
    assert res.ready
    summary = summarize_model([res])
    assert summary["alerts"] == 1 and summary["series_over_limit"] == 1


def test_model_error_is_recorded_not_raised(monkeypatch):
    class Boom:
        def update(self, ts, value):
            if ts > 100:
                raise RecursionError
            return None

    monkeypatch.setattr("anomaly_detection.harness.replay.create_model", lambda *a, **k: Boom())
    raw = run_model(parse_model("seasonal"), _series(np.ones(10)), 60, 0)
    assert raw.error == "RecursionError at sample 2"
    res = evaluate_series(_series(np.ones(10)), raw, "seasonal", 60, 0, Guards(), AlertRule())
    assert res.error and summarize_model([res])["series_failed"] == 1
