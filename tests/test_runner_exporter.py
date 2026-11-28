import urllib.error
import urllib.request

import pytest
from prometheus_client.parser import text_string_to_metric_families

from anomaly_detection.config import Config
from anomaly_detection.exporter import make_server, serve_in_thread
from anomaly_detection.prom import PromClient, QueryError
from anomaly_detection.runner import Runner

from .conftest import matrix, vector

STEP = 60


def make_config(url, **det):
    detector = {
        "name": "req",
        "query": "sum(rate(x[5m]))",
        "interval": STEP,
        "backfill": "3d",
        "model": {"kind": "seasonal", "seasonality": ["daily"]},
        **det,
    }
    return Config.model_validate({"datasource": {"url": url}, "detectors": [detector]})


def history(days=3, level=100.0, labels=None):
    n = days * 1440
    end = 1_700_000_000 // STEP * STEP
    return end, [((end - (n - 1 - i) * STEP), str(level + (i % 7))) for i in range(n)]


def build(fake, **det):
    cfg = make_config(fake.url, **det)
    runner = Runner(cfg, PromClient(cfg.datasource))
    return cfg, runner


def scrape(runner):
    server = make_server("127.0.0.1:0", runner)
    serve_in_thread(server)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    return server, base


def samples(text):
    out = {}
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            out.setdefault(s.name, []).append(s)
    return out


def test_backfill_then_evaluate_exports_series(fake_prom, monkeypatch):
    end, vals = history()

    def handler(path, q):
        if path.endswith("/query_range"):
            return 200, matrix(({"model": "a"}, [(t, v) for t, v in vals]))
        return 200, vector(({"model": "a"}, float(q["time"]), "100"))

    fake_prom.handler = handler
    _cfg, runner = build(fake_prom)
    st = runner.states[0]
    runner.backfill(st, end + 1)
    assert st.backfill_complete and len(st.pipelines) == 1
    runner.evaluate(st, end + STEP)
    import time as _t

    monkeypatch.setattr(_t, "time", lambda: end + STEP + 1)
    server, base = scrape(runner)
    try:
        text = urllib.request.urlopen(base + "/metrics").read().decode()
    finally:
        server.shutdown()
    s = samples(text)
    assert s["anomaly_score"][0].labels == {"detector": "req", "model": "a"}
    assert 0.0 <= s["anomaly_score"][0].value <= 1.0
    assert "anomaly_upper_band" in s and "anomaly_lower_band" in s
    assert s["anomaly_detector_ready"][0].value == 1.0
    assert s["anomaly_detector_managed_series"][0].value == 1.0
    assert s["anomaly_detector_backfill_complete"][0].value == 1.0


def test_failed_and_empty_queries_are_not_zeros(fake_prom):
    end, vals = history()
    fake_prom.handler = lambda p, q: (
        (200, matrix(({"model": "a"}, vals)))
        if p.endswith("/query_range")
        else (500, {"status": "error", "error": "boom"})
    )
    _cfg, runner = build(fake_prom)
    st = runner.states[0]
    runner.backfill(st, end + 1)
    (pipe,) = st.pipelines.values()
    before = pipe.samples
    runner.evaluate(st, end + STEP)
    assert st.query_failures == 1 and pipe.samples == before  # nothing was fed

    fake_prom.handler = lambda p, q: (200, vector())
    runner.evaluate(st, end + 2 * STEP)
    assert st.query_empty == 1 and pipe.samples == before


def test_stale_results_stop_being_exported(fake_prom, monkeypatch):
    end, vals = history()
    fake_prom.handler = lambda p, q: (200, matrix(({"model": "a"}, vals)))
    _cfg, runner = build(fake_prom)
    st = runner.states[0]
    runner.backfill(st, end + 1)
    import time as _t

    monkeypatch.setattr(_t, "time", lambda: end + 10 * STEP)  # ten intervals without data
    server, base = scrape(runner)
    try:
        text = urllib.request.urlopen(base + "/metrics").read().decode()
    finally:
        server.shutdown()
    s = samples(text)
    assert "anomaly_score" not in s
    assert s["anomaly_detector_ready"][0].value == 0.0


def test_series_limit_drops_and_reports(fake_prom):
    end, vals = history(days=1)
    many = [({"i": str(i)}, vals) for i in range(5)]
    fake_prom.handler = lambda p, q: (200, matrix(*many))
    _cfg, runner = build(fake_prom, max_series=3)
    st = runner.states[0]
    runner.backfill(st, end + 1)
    assert len(st.pipelines) == 3 and st.series_dropped == 2


def test_backfill_failure_raises_and_readyz_stays_503(fake_prom):
    fake_prom.handler = lambda p, q: (500, {"status": "error", "error": "down"})
    _cfg, runner = build(fake_prom)
    with pytest.raises(QueryError):
        runner.backfill(runner.states[0], 1_700_000_000)
    server, base = scrape(runner)
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/readyz")
        assert e.value.code == 503
        assert urllib.request.urlopen(base + "/healthz").status == 200
    finally:
        server.shutdown()


def test_restart_recovers_from_backfill_only(fake_prom):
    """Two runners built from the same history reach the same scores: no local state."""
    end, vals = history()
    fake_prom.handler = lambda p, q: (200, matrix(({"model": "a"}, vals)))
    results = []
    for _ in range(2):
        _cfg, runner = build(fake_prom)
        runner.backfill(runner.states[0], end + 1)
        (pipe,) = runner.states[0].pipelines.values()
        results.append(pipe.last_result)
    assert results[0] is not None and results[0] == results[1]


def test_empty_startup_backfill_is_repaired_by_late_backfill(fake_prom):
    """The store may not serve history yet when the detector starts (just seeded data)."""
    end, vals = history()
    state = {"ready": False}

    def handler(path, q):
        if path.endswith("/query_range"):
            return 200, matrix(({"model": "a"}, vals)) if state["ready"] else matrix()
        return 200, vector(({"model": "a"}, float(q["time"]), "100"))

    fake_prom.handler = handler
    _cfg, runner = build(fake_prom)
    st = runner.states[0]
    runner.backfill(st, end + 1)
    assert st.backfill_complete and not st.pipelines

    runner.evaluate(st, end + STEP)  # live sample creates a cold series
    (pipe,) = st.pipelines.values()
    assert pipe.last_result is None and st.cold
    state["ready"] = True
    runner.evaluate(st, end + 3 * STEP)  # two intervals later the history is replayed
    (pipe,) = st.pipelines.values()
    assert pipe.last_result is not None and not st.cold
