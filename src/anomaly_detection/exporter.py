"""Prometheus exposition and the small HTTP server (/metrics, /healthz, /readyz)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import CollectorRegistry, ProcessCollector
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.exposition import CONTENT_TYPE_LATEST, generate_latest

from .runner import DetectorState, Runner

log = logging.getLogger(__name__)


def _labels(detector: str, key: tuple[tuple[str, str], ...]) -> tuple[list[str], list[str]]:
    names, values = ["detector"], [detector]
    for k, v in key:
        # Keep the original label when it collides with ours, like Prometheus does for
        # exported_* labels on scrape.
        names.append("exported_detector" if k == "detector" else k)
        values.append(v)
    return names, values


class DetectorCollector:
    """Builds the exposition from detector state on every scrape.

    Series are generated at scrape time instead of being stored as gauges, so a series that
    vanishes from the source or goes stale simply stops being exported. No value means no
    sample, never a zero.
    """

    def __init__(self, runner: Runner):
        self._runner = runner

    def collect(self) -> Iterator[Metric]:
        now = time.time()
        # Label sets differ between detectors, so one family per metric and label-name set
        # would be awkward. Prometheus exposition allows the same name with different label
        # names inside one family, so we use one family per metric.
        fam = {
            "anomaly_score": GaugeMetricFamily(
                "anomaly_score", "Anomaly score from 0 (normal) to 1 (certainly anomalous)"
            ),
            "anomaly_value": GaugeMetricFamily("anomaly_value", "Observed value that was scored"),
            "anomaly_expected": GaugeMetricFamily(
                "anomaly_expected", "Expected value according to the model"
            ),
            "anomaly_upper_band": GaugeMetricFamily(
                "anomaly_upper_band", "Upper edge of the expected band (models with a band only)"
            ),
            "anomaly_lower_band": GaugeMetricFamily(
                "anomaly_lower_band", "Lower edge of the expected band (models with a band only)"
            ),
            "anomaly_tier": GaugeMetricFamily(
                "anomaly_tier",
                "Alert stage of an ensemble model: 0 none, 1 ai analysis, 2 human",
            ),
            "anomaly_detector_ready": GaugeMetricFamily(
                "anomaly_detector_ready",
                "1 when the series is scored, 0 while warming up or without a fresh value",
            ),
        }
        det = {
            "managed": GaugeMetricFamily(
                "anomaly_detector_managed_series",
                "Series with a model instance",
                labels=["detector"],
            ),
            "dropped": GaugeMetricFamily(
                "anomaly_detector_series_dropped",
                "Series skipped in the last evaluation because of the series limits",
                labels=["detector"],
            ),
            "failures": CounterMetricFamily(
                "anomaly_detector_query_failures",
                "Queries that failed (HTTP, timeout or API error)",
                labels=["detector"],
            ),
            "empty": CounterMetricFamily(
                "anomaly_detector_query_empty",
                "Queries that succeeded but returned no series",
                labels=["detector"],
            ),
            "evals": CounterMetricFamily(
                "anomaly_detector_evaluations", "Evaluations run", labels=["detector"]
            ),
            "lag": GaugeMetricFamily(
                "anomaly_detector_evaluation_lag_seconds",
                "Seconds between the scheduled evaluation time and its completion",
                labels=["detector"],
            ),
            "duration": GaugeMetricFamily(
                "anomaly_detector_evaluation_duration_seconds",
                "Duration of the last evaluation",
                labels=["detector"],
            ),
            "last_ok": GaugeMetricFamily(
                "anomaly_detector_last_success_timestamp_seconds",
                "Evaluation time of the last successful query",
                labels=["detector"],
            ),
            "backfill": GaugeMetricFamily(
                "anomaly_detector_backfill_complete",
                "1 once the startup backfill finished",
                labels=["detector"],
            ),
            "backfill_samples": GaugeMetricFamily(
                "anomaly_detector_backfill_samples",
                "Samples replayed by the startup backfill",
                labels=["detector"],
            ),
        }

        for st in self._runner.states:
            self._collect_series(st, now, fam)
            name = st.cfg.name
            det["managed"].add_metric([name], len(st.pipelines))
            det["dropped"].add_metric([name], st.series_dropped)
            det["failures"].add_metric([name], st.query_failures)
            det["empty"].add_metric([name], st.query_empty)
            det["evals"].add_metric([name], st.evaluations)
            det["lag"].add_metric([name], st.evaluation_lag)
            det["duration"].add_metric([name], st.evaluation_duration)
            det["last_ok"].add_metric([name], st.last_evaluation_ts)
            det["backfill"].add_metric([name], 1 if st.backfill_complete else 0)
            det["backfill_samples"].add_metric([name], st.backfill_samples)

        yield from fam.values()
        yield from det.values()

    @staticmethod
    def _collect_series(st: DetectorState, now: float, fam: dict[str, GaugeMetricFamily]) -> None:
        fresh_for = 2 * st.interval + 5
        with st.lock:
            items = [(k, p.last_result, p.last_timestamp) for k, p in st.pipelines.items()]
        for key, result, last_ts in items:
            names, values = _labels(st.cfg.name, key)
            if result is None or now - last_ts > fresh_for:
                # Warming up, or no fresh sample: report state only, no score.
                fam["anomaly_detector_ready"].add_sample(
                    "anomaly_detector_ready", dict(zip(names, values, strict=True)), 0.0
                )
                continue
            lab = dict(zip(names, values, strict=True))
            fam["anomaly_detector_ready"].add_sample("anomaly_detector_ready", lab, 1.0)
            fam["anomaly_score"].add_sample("anomaly_score", lab, result.score)
            if result.tier is not None:
                fam["anomaly_tier"].add_sample("anomaly_tier", lab, float(result.tier))
            fam["anomaly_value"].add_sample("anomaly_value", lab, result.value)
            fam["anomaly_expected"].add_sample("anomaly_expected", lab, result.expected)
            if result.upper is not None:
                fam["anomaly_upper_band"].add_sample("anomaly_upper_band", lab, result.upper)
            if result.lower is not None:
                fam["anomaly_lower_band"].add_sample("anomaly_lower_band", lab, result.lower)


def build_registry(runner: Runner) -> CollectorRegistry:
    registry = CollectorRegistry()
    registry.register(DetectorCollector(runner))  # type: ignore[arg-type]
    try:
        ProcessCollector(registry=registry)  # type: ignore[no-untyped-call]
    except Exception:  # not available on every platform
        log.debug("process metrics unavailable")
    return registry


def make_server(listen: str, runner: Runner) -> ThreadingHTTPServer:
    host, _, port = listen.rpartition(":")
    registry = build_registry(runner)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "text/plain; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/metrics":
                self._send(200, generate_latest(registry), CONTENT_TYPE_LATEST)
            elif path == "/healthz":
                ok = runner.healthy
                self._send(200 if ok else 503, b"ok\n" if ok else b"a detector thread died\n")
            elif path == "/readyz":
                ok = runner.ready
                self._send(200 if ok else 503, b"ready\n" if ok else b"backfill not complete\n")
            else:
                self._send(404, b"not found\n")

        def log_message(self, format: str, *args: object) -> None:
            log.debug("http: " + format, *args)

    server = ThreadingHTTPServer((host or "0.0.0.0", int(port)), Handler)
    server.daemon_threads = True
    return server


def serve_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    t = threading.Thread(target=server.serve_forever, name="http", daemon=True)
    t.start()
    return t
