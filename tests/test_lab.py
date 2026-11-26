import csv
import http.client
import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import numpy as np

from anomaly_detection.lab import gpuexporter, loadgen, replay, timescale
from anomaly_detection.lab.dump import dump_signals
from anomaly_detection.lab.injector import Event, Injector, build_schedule, write_labels
from anomaly_detection.lab.signals import SCENARIOS, SIGNALS

from .conftest import matrix


def test_daily_factor_has_unit_mean_and_peaks_in_the_afternoon():
    fr = np.linspace(0, 1, 2880, endpoint=False)
    f = np.array([timescale.daily_factor(x) for x in fr])
    assert abs(f.mean() - 1.0) < 0.01
    assert 14 / 24 < fr[f.argmax()] < 16 / 24
    assert f.min() > 0.3


def test_compressed_day_fraction_wraps():
    assert timescale.day_fraction(1000 + 1440 * 3 + 360, 1000, 1440) == 0.25
    r = timescale.target_rate(1000, 1000, 1440, 2.0)
    assert r == 2.0 * timescale.daily_factor(0.0)


def test_schedule_covers_every_scenario_without_overlap_and_is_repeatable():
    a = build_schedule(10_000, 5, 600, (120, 300), 180, seed=3)
    b = build_schedule(10_000, 5, 600, (120, 300), 180, seed=3)
    assert [(e.scenario, e.start) for e in a] == [(e.scenario, e.start) for e in b]
    assert len(a) == 5 * len(SCENARIOS)
    for s in SCENARIOS:
        assert sum(e.scenario == s for e in a) == 5
    for prev, nxt in itertools.pairwise(a):
        assert nxt.start - prev.end >= 180 - 1e-6
    assert all(120 <= e.end - e.start <= 300 for e in a)
    c = build_schedule(10_000, 5, 600, (120, 300), 180, seed=4)
    assert [e.scenario for e in a] != [e.scenario for e in c]


def test_schedule_rejects_a_slot_that_is_too_short():
    import pytest

    with pytest.raises(ValueError):
        build_schedule(0, 1, 300, (120, 300), 180, seed=1)


class _Engine(BaseHTTPRequestHandler):
    bodies: ClassVar[list[dict]] = []

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        _Engine.bodies.append(json.loads(self.rfile.read(n)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(4):
            self.wfile.write(f'data: {{"i": {i}}}\n\n'.encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


def _args(**kw):
    import argparse

    d = dict(
        prompt_words_median=20, prompt_sigma=0.3, prompt_words_max=60,
        max_tokens_median=50, max_tokens_sigma=0.3, timeout=5,
    )  # fmt: skip
    return argparse.Namespace(**{**d, **kw})


def test_requests_follow_faults_and_are_counted():
    import random

    engine = ThreadingHTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=engine.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{engine.server_address[1]}"
    state = loadgen.State()
    rng = random.Random(1)
    _Engine.bodies.clear()
    loadgen.one_request(url, "m", rng, state, _args())
    assert state.sent["ok"] == 1 and state.dur_count == 1 and state.tokens_received == 4
    state.fault = {"kind": "output_drop"}
    loadgen.one_request(url, "m", rng, state, _args())
    assert _Engine.bodies[-1]["max_tokens"] <= 3
    state.fault = {"kind": "latency_rise", "delay": 0.02}
    loadgen.one_request(url, "m", rng, state, _args())
    assert state.dur_sum > 0.1  # 5 chunks plus the first byte, 20 ms each
    text = state.render()
    assert 'lab_requests_sent_total{outcome="ok"} 3' in text
    assert 'lab_fault_active{kind="latency_rise"} 1' in text
    engine.shutdown()
    engine.server_close()
    # unreachable engine: counted as an error, never as a success
    loadgen.one_request("http://127.0.0.1:1", "m", rng, state, _args())
    assert state.sent["error"] == 1


def test_control_api_sets_and_clears_faults_and_serves_metrics():
    state = loadgen.State()
    srv = loadgen.serve_control(state, "127.0.0.1:0")
    port = srv.server_address[1]

    def call(method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", port)
        c.request(method, path, json.dumps(body) if body is not None else None)
        r = c.getresponse()
        return r.status, r.read().decode()

    assert call("POST", "/fault", {"kind": "request_drop", "factor": 0.1})[0] == 200
    assert state.fault == {"kind": "request_drop", "factor": 0.1}
    assert call("POST", "/fault", {"kind": "nope"})[0] == 400
    status, text = call("GET", "/metrics")
    assert status == 200 and 'lab_fault_active{kind="request_drop"} 1' in text
    assert call("DELETE", "/fault")[0] == 200 and state.fault is None
    srv.shutdown()
    srv.server_close()


def test_injector_applies_and_clears_and_writes_labels(tmp_path):
    state = loadgen.State()
    srv = loadgen.serve_control(state, "127.0.0.1:0")
    inj = Injector(f"http://127.0.0.1:{srv.server_address[1]}", ["sleep", "30"])
    ev = Event("output_length_drop", 0, 1)
    inj.apply(ev)
    assert state.fault["kind"] == "output_drop"
    inj.clear()
    assert state.fault is None
    gpu = Event("throughput_drop", 0, 1, {"duty": 0.5})
    # `sleep` rejects the extra --duty arguments, which is enough to see the process start
    inj.contender_cmd = ["python3", "-c", "import time; time.sleep(30)", "x"]
    inj.apply(gpu)
    assert inj._proc is not None and inj._proc.poll() is None
    inj.clear()
    assert inj._proc is None
    ev.actual_start, ev.actual_end, ev.status = 5.0, 9.0, "completed"
    p = tmp_path / "labels.json"
    write_labels(p, {"epoch": 0}, [ev])
    doc = json.loads(p.read_text())
    assert doc["events"][0]["expected_signal"] == "output_tokens"
    rows = list(csv.DictReader(p.with_suffix(".csv").open()))
    assert rows[0]["start"] == "5.0" and rows[0]["scenario"] == "output_length_drop"
    srv.shutdown()
    srv.server_close()


def test_dump_signals_writes_one_labelled_series_per_signal(tmp_path, fake_prom):
    fake_prom.handler = lambda p, q: (200, matrix(({}, [(100.0, "1.5"), (105.0, "2.5")])))
    out = tmp_path / "signals.csv"
    n = dump_signals(fake_prom.url, out, 100, 105, 5)
    assert n == 2 * len(SIGNALS)
    names = {r["labels"] for r in csv.DictReader(out.open())}
    assert names == {f'signal="{s.name}"' for s in SIGNALS}
    assert {q["params"]["query"] for q in fake_prom.requests} == {s.query for s in SIGNALS}


def test_gpu_exporter_skips_unsupported_fields():
    lines = gpuexporter.parse("95, 2496, 60, 40.5, [N/A]\n")
    assert 'gpu_utilization_percent{gpu="0"} 95.0' in lines
    assert 'gpu_power_watts{gpu="0"} 40.5' in lines
    assert not any(line.startswith("gpu_memory_used_bytes") for line in lines)


def test_load_average_reads_the_first_two_fields(tmp_path):
    p = tmp_path / "loadavg"
    p.write_text("1.50 2.25 3.00 1/200 123\n")
    assert gpuexporter.load_average(str(p)) == ["host_load1 1.5", "host_load5 2.25"]
    assert gpuexporter.load_average(str(tmp_path / "missing")) == []


def _lab_signals(tmp_path, drop: bool):
    """Four compressed days of a daily request rate, then 40 minutes with one drop."""
    step, day = 5, 1440
    rng = np.random.default_rng(0)
    n = int((4 * day + 2400) / step)
    t0 = 1_800_000_000.0
    ts = t0 + np.arange(n) * step
    vals = np.array([timescale.daily_factor(((t - t0) / day) % 1.0) for t in ts]) * 1.5
    vals *= np.exp(rng.normal(0, 0.05, n))
    start, end = t0 + 4 * day + 1200, t0 + 4 * day + 1500
    if drop:
        vals[(ts >= start) & (ts < end)] *= 0.03
    sig = tmp_path / "signals.csv"
    with sig.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "labels", "value"])
        for t, v in zip(ts, vals, strict=True):
            w.writerow([t, 'signal="request_rate"', v])
    ev = {
        "scenario": "request_drop", "actual_start": start, "actual_end": end,
        "status": "completed",
    }  # fmt: skip
    meta = {"epoch": t0, "day_seconds": day, "warmup_days": 4, "eval_start": t0 + 4 * day}
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"meta": meta, "events": [ev]}))
    return sig, labels


def test_replay_detects_a_labelled_drop_with_the_seasonal_model(tmp_path):
    sig, labels = _lab_signals(tmp_path, drop=True)
    meta, events = replay.load_labels(labels)
    signals = replay.load_signals(sig, 5)
    rows = replay.evaluate(signals, meta, events, ["fixed", "seasonal"], 5, replay.AlertRule())
    by = {r.model: r for r in rows}
    assert by["seasonal"].result.events == 1 and by["seasonal"].result.detected == 1
    assert by["seasonal"].result.false_alarms == 0
    assert by["seasonal"].result.delays[0] <= 60  # seconds in lab time
    text = replay.render(rows, ["fixed", "seasonal"], 1440)
    assert "request_drop" in text and "seasonal" in text


def test_replay_of_a_quiet_run_has_no_alarms(tmp_path):
    sig, labels = _lab_signals(tmp_path, drop=False)
    meta, _ = replay.load_labels(labels)
    signals = replay.load_signals(sig, 5)
    (row,) = replay.evaluate(signals, meta, [], ["seasonal"], 5, replay.AlertRule())
    assert row.result.fires == []


def test_lab_example_config_uses_the_signal_queries_and_the_lab_day():
    from pathlib import Path

    import yaml

    from anomaly_detection.config import load_config

    root = Path(__file__).resolve().parent.parent / "examples" / "inject-lab"
    cfg = load_config(root / "config.yaml")
    assert {d.name: d.query for d in cfg.detectors} == {s.name: s.query for s in SIGNALS}
    assert {d.name: d.direction for d in cfg.detectors} == {s.name: s.direction for s in SIGNALS}
    for d in cfg.detectors:
        assert d.model.params["seasonality"] == [1440]
    for name in ("compose.yaml", "scrape.yml"):
        assert yaml.safe_load((root / name).read_text())
