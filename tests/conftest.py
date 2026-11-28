"""A tiny fake Prometheus HTTP API for tests."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest


class FakeProm:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        # handler(path, params) -> (status, body dict)
        self.handler: Callable[[str, dict[str, str]], tuple[int, dict]] = lambda p, q: (
            200,
            {"status": "success", "data": {"resultType": "vector", "result": []}},
        )
        self.server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        assert self.server
        return f"http://127.0.0.1:{self.server.server_address[1]}"


@pytest.fixture
def fake_prom() -> Iterator[FakeProm]:
    fake = FakeProm()

    class H(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            u = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(u.query).items()}
            fake.requests.append(
                {"path": u.path, "params": params, "headers": dict(self.headers.items())}
            )
            status, body = fake.handler(u.path, params)
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a: object) -> None:
            pass

    fake.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=fake.server.serve_forever, daemon=True)
    t.start()
    yield fake
    fake.server.shutdown()
    fake.server.server_close()


def vector(*items: tuple[dict[str, str], float, str]) -> dict:
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [{"metric": m, "value": [ts, v]} for m, ts, v in items],
        },
    }


def matrix(*items: tuple[dict[str, str], list[tuple[float, str]]]) -> dict:
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [{"metric": m, "values": [[t, v] for t, v in vals]} for m, vals in items],
        },
    }
