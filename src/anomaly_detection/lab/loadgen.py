"""Open loop load generator for an OpenAI compatible chat endpoint.

- Arrivals are a non-homogeneous Poisson process whose rate follows a compressed day (see
  `timescale`), multiplied by a slowly varying lognormal noise factor.
- Prompt length and the output token cap are drawn from lognormal distributions.
- Faults are switched through a small HTTP control API, so an injector can start and stop
  them at scheduled times:  POST /fault {"kind": ..., ...}  and  DELETE /fault.
- Own metrics on /metrics (Prometheus text format): sent requests by outcome, client side
  request duration, in flight requests, target rate, active fault.

Faults (all keep HTTP 200 where the endpoint does):
  request_drop     arrival rate is multiplied by `factor` (default 0.03)
  output_drop      the output token cap becomes 1 to 3 tokens
  latency_rise     `delay` seconds are added before every received chunk and before the first
                   byte: a slow path between client and server
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .timescale import target_rate

WORDS = (  # noqa: SIM905  a word list reads better as one string
    "alpha river system request cluster memory kernel latency window token model queue "
    "service policy record budget signal vector batch route shard cache index stream "
    "thread worker config deploy rollout metric alert region network storage volume "
    "gateway account invoice ticket report schedule backup restore review release"
).split()
DURATION_BUCKETS = (0.5, 1, 2, 4, 8, 16, 32, 64)


def lognormal(rng: random.Random, median: float, sigma: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, median * math.exp(rng.gauss(0.0, sigma))))


def make_prompt(rng: random.Random, words: int) -> str:
    body = " ".join(rng.choice(WORDS) for _ in range(words))
    return f"Notes: {body}\nSummarise the notes above and say what to do next."


class State:
    """Shared between the arrival loop, the request threads and the control API."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.fault: dict | None = None
        self.sent = {"ok": 0, "error": 0}
        self.dur_sum = 0.0
        self.dur_count = 0
        self.dur_buckets = [0] * len(DURATION_BUCKETS)
        self.inflight = 0
        self.rate = 0.0
        self.tokens_received = 0

    def observe(self, ok: bool, seconds: float, chunks: int) -> None:
        with self.lock:
            self.sent["ok" if ok else "error"] += 1
            if ok:
                self.dur_sum += seconds
                self.dur_count += 1
                self.tokens_received += chunks
                for i, b in enumerate(DURATION_BUCKETS):
                    if seconds <= b:
                        self.dur_buckets[i] += 1

    def render(self) -> str:
        with self.lock:
            f = self.fault or {}
            lines = [
                "# TYPE lab_requests_sent_total counter",
                f'lab_requests_sent_total{{outcome="ok"}} {self.sent["ok"]}',
                f'lab_requests_sent_total{{outcome="error"}} {self.sent["error"]}',
                "# TYPE lab_request_duration_seconds histogram",
            ]
            for b, n in zip(DURATION_BUCKETS, self.dur_buckets, strict=True):
                # observe() increments every bucket whose bound is >= the value (cumulative)
                lines.append(f'lab_request_duration_seconds_bucket{{le="{b}"}} {n}')
            lines += [
                f'lab_request_duration_seconds_bucket{{le="+Inf"}} {self.dur_count}',
                f"lab_request_duration_seconds_sum {self.dur_sum}",
                f"lab_request_duration_seconds_count {self.dur_count}",
                "# TYPE lab_stream_chunks_total counter",
                f"lab_stream_chunks_total {self.tokens_received}",
                "# TYPE lab_inflight_requests gauge",
                f"lab_inflight_requests {self.inflight}",
                "# TYPE lab_target_rate gauge",
                f"lab_target_rate {self.rate}",
                "# TYPE lab_fault_active gauge",
            ]
            for kind in ("request_drop", "output_drop", "latency_rise"):
                lines.append(f'lab_fault_active{{kind="{kind}"}} {int(f.get("kind") == kind)}')
        return "\n".join(lines) + "\n"


def one_request(url: str, model: str, rng: random.Random, state: State, args) -> None:
    u = urlparse(url)
    with state.lock:
        fault = dict(state.fault) if state.fault else {}
        state.inflight += 1
    words = int(
        lognormal(rng, args.prompt_words_median, args.prompt_sigma, 8, args.prompt_words_max)
    )
    if fault.get("kind") == "output_drop":
        max_tokens = rng.randint(1, 3)
    else:
        max_tokens = int(lognormal(rng, args.max_tokens_median, args.max_tokens_sigma, 8, 512))
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": make_prompt(rng, words)}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    delay = float(fault.get("delay", 0.0)) if fault.get("kind") == "latency_rise" else 0.0
    t0 = time.monotonic()
    ok, chunks = False, 0
    try:
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=args.timeout)
        if delay:
            time.sleep(delay)
        conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status == 200:
            for raw in resp:
                if raw.startswith(b"data: ") and not raw.startswith(b"data: [DONE]"):
                    chunks += 1
                    if delay:
                        time.sleep(delay)
            ok = True
        conn.close()
    except (OSError, http.client.HTTPException):
        ok = False
    finally:
        with state.lock:
            state.inflight -= 1
        state.observe(ok, time.monotonic() - t0, chunks)


def serve_control(state: State, listen: str) -> ThreadingHTTPServer:
    host, _, port = listen.rpartition(":")

    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/metrics":
                self._send(200, state.render().encode(), "text/plain; version=0.0.4")
            elif self.path == "/fault":
                self._send(200, json.dumps(state.fault).encode())
            else:
                self._send(404, b"{}")

        def do_POST(self) -> None:
            if self.path != "/fault":
                return self._send(404, b"{}")
            n = int(self.headers.get("Content-Length", 0))
            try:
                fault = json.loads(self.rfile.read(n) or b"{}")
                if fault.get("kind") not in ("request_drop", "output_drop", "latency_rise"):
                    raise ValueError("unknown kind")
            except ValueError as e:
                return self._send(400, json.dumps({"error": str(e)}).encode())
            with state.lock:
                state.fault = fault
            self._send(200, json.dumps(fault).encode())

        def do_DELETE(self) -> None:
            with state.lock:
                state.fault = None
            self._send(200, b"{}")

        def log_message(self, *a: object) -> None:
            pass

    srv = ThreadingHTTPServer((host or "0.0.0.0", int(port)), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anomaly-detection-loadgen", description=__doc__)
    p.add_argument("--target", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="lab-model")
    p.add_argument("--listen", default="0.0.0.0:9201", help="control API and /metrics")
    p.add_argument("--day-seconds", type=float, default=1440, help="length of one compressed day")
    p.add_argument("--epoch", type=float, default=0.0, help="start of day 0 (unix seconds)")
    p.add_argument("--base-rate", type=float, default=1.5, help="requests per second, daily mean")
    p.add_argument("--noise-sigma", type=float, default=0.10, help="lognormal rate noise")
    p.add_argument("--noise-seconds", type=float, default=20, help="rate noise correlation time")
    p.add_argument("--prompt-words-median", type=float, default=120)
    p.add_argument("--prompt-sigma", type=float, default=0.8)
    p.add_argument("--prompt-words-max", type=float, default=900)
    p.add_argument("--max-tokens-median", type=float, default=96)
    p.add_argument("--max-tokens-sigma", type=float, default=0.6)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--max-inflight", type=int, default=256)
    p.add_argument("--duration", type=float, default=0, help="stop after seconds, 0 = forever")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    state = State()
    serve_control(state, args.listen)
    start = time.time()
    noise = 0.0  # AR(1) in log space, refreshed once per second
    last_noise = start
    rho = math.exp(-1.0 / args.noise_seconds)
    while args.duration <= 0 or time.time() - start < args.duration:
        now = time.time()
        if now - last_noise >= 1.0:
            noise = rho * noise + math.sqrt(1 - rho**2) * rng.gauss(0.0, args.noise_sigma)
            last_noise = now
        with state.lock:
            fault = state.fault
            mult = (
                float(fault.get("factor", 0.03))
                if fault and fault["kind"] == "request_drop"
                else 1.0
            )
        rate = target_rate(now, args.epoch, args.day_seconds, args.base_rate)
        rate *= math.exp(noise) * mult
        state.rate = rate
        # Poisson process: the wait is exponential. Waits longer than a second are cut into
        # one second slices (memoryless), so a change of rate or fault is picked up quickly.
        wait = rng.expovariate(rate) if rate > 0 else 1.0
        if wait > 1.0:
            time.sleep(1.0)
            continue
        time.sleep(wait)
        if state.inflight >= args.max_inflight:
            continue
        threading.Thread(
            target=one_request,
            args=(args.target, args.model, random.Random(rng.random()), state, args),
            daemon=True,
        ).start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
