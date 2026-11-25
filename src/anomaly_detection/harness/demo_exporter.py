"""Synthetic metric exporter for demos (docker compose). It is not part of the sidecar.

Live mode serves /metrics with four gauges that follow the same shapes as the evaluation
harness. `POST /fault?kind=request_drop&seconds=900` injects one of the silent failures so
the sidecar's reaction can be watched; `POST /fault/clear` removes it.

Seed mode writes history before the stack starts, so the sidecar has something to learn from:
  --seed-victoriametrics URL   push the history to a VictoriaMetrics import endpoint
  --seed-openmetrics FILE      write an OpenMetrics file for `promtool tsdb create-blocks-from`
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np

from .synth import SIGNALS, STEP_TOKENS, generate, missing_mask, shape

METRICS = {
    "request_rate": ("demo_request_rate", "Requests per second"),
    "output_tokens": ("demo_output_tokens_mean", "Mean output tokens per response"),
    "throughput": ("demo_training_throughput", "Training tokens per second"),
    "step_time": ("demo_step_time_seconds", "Seconds per training step"),
}
# Same labels the store attaches when it scrapes the live exporter, so seeded history and live
# samples are one series instead of two that a sum() would add up.
SCRAPE_LABELS = '{job="demo",instance="generator:9100"}'
FAULTS = ("request_drop", "token_length_drop", "throughput_drop")


def history(end: float, days: float, step: int = 60, seed: int = 1):
    ts = np.arange(int((end - days * 86400) // step) * step, int(end // step) * step + 1, step)
    ts = ts.astype(float)
    skip = missing_mask(len(ts), seed)
    out = {}
    for sig in METRICS:
        values = generate(sig, ts, seed)
        out[sig] = [(float(t), float(v)) for t, v, s in zip(ts, values, skip, strict=True) if not s]
    return out


def seed_victoriametrics(url: str, end: float, days: float) -> None:
    lines = []
    for sig, points in history(end, days).items():
        name = METRICS[sig][0]
        lines += [f"{name}{SCRAPE_LABELS} {v} {int(t * 1000)}" for t, v in points]
    resp = httpx.post(
        f"{url.rstrip('/')}/api/v1/import/prometheus", content="\n".join(lines) + "\n"
    )
    resp.raise_for_status()
    print(f"seeded {len(lines)} samples into {url}", file=sys.stderr)


def seed_openmetrics(path: str, end: float, days: float) -> None:
    data = history(end, days)
    with open(path, "w") as f:
        for sig, points in data.items():
            name = METRICS[sig][0]
            f.write(f"# TYPE {name} gauge\n")
            f.writelines(f"{name}{SCRAPE_LABELS} {v} {t:.0f}\n" for t, v in points)
        f.write("# EOF\n")
    print(f"wrote {path}", file=sys.stderr)


class State:
    def __init__(self) -> None:
        self.faults: dict[str, float] = {}
        self.lock = threading.Lock()

    def active(self, now: float) -> set[str]:
        with self.lock:
            self.faults = {k: v for k, v in self.faults.items() if v > now}
            return set(self.faults)


def current_values(now: float, active: set[str]) -> dict[str, float]:
    rng = np.random.default_rng(int(now // 15))
    ts = np.array([now])
    out = {}
    for sig in ("request_rate", "output_tokens", "throughput"):
        out[sig] = float(shape(sig, ts)[0] * (1 + rng.normal(0, SIGNALS[sig].noise)))
    if "request_drop" in active:
        out["request_rate"] *= 0.02
    if "token_length_drop" in active:
        out["output_tokens"] = 2.0
    if "throughput_drop" in active:
        out["throughput"] *= 0.5
    out["step_time"] = STEP_TOKENS / out["throughput"]
    return out


def serve(listen: str) -> None:
    state = State()
    host, _, port = listen.rpartition(":")

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: str) -> None:
            raw = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if urlparse(self.path).path != "/metrics":
                return self._send(404, "not found\n")
            now = time.time()
            values = current_values(now, state.active(now))
            lines = []
            for sig, (name, help_) in METRICS.items():
                lines += [f"# HELP {name} {help_}", f"# TYPE {name} gauge", f"{name} {values[sig]}"]
            self._send(200, "\n".join(lines) + "\n")

        def do_POST(self) -> None:
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/fault/clear":
                with state.lock:
                    state.faults.clear()
                return self._send(200, "cleared\n")
            if u.path == "/fault":
                kind = q.get("kind", [""])[0]
                if kind not in FAULTS:
                    return self._send(400, f"kind must be one of {FAULTS}\n")
                seconds = float(q.get("seconds", ["900"])[0])
                with state.lock:
                    state.faults[kind] = time.time() + seconds
                return self._send(200, f"{kind} for {seconds:.0f}s\n")
            return self._send(404, "not found\n")

        def log_message(self, *a: object) -> None:
            pass

    ThreadingHTTPServer((host or "0.0.0.0", int(port)), H).serve_forever()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--listen", default="0.0.0.0:9100")
    p.add_argument("--seed-days", type=float, default=14)
    p.add_argument("--seed-victoriametrics", metavar="URL")
    p.add_argument("--seed-openmetrics", metavar="FILE")
    p.add_argument(
        "--seed-end-offset", type=float, default=0, help="seconds before now where the seed ends"
    )
    args = p.parse_args(argv)
    end = time.time() - args.seed_end_offset
    if args.seed_victoriametrics:
        seed_victoriametrics(args.seed_victoriametrics, end, args.seed_days)
        return 0
    if args.seed_openmetrics:
        seed_openmetrics(args.seed_openmetrics, end, args.seed_days)
        return 0
    serve(args.listen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
