"""Minimal client for the Prometheus HTTP API (query and query_range)."""

from __future__ import annotations

import math
import ssl
from dataclasses import dataclass

import httpx

from .config import DatasourceConfig

LabelKey = tuple[tuple[str, str], ...]


class QueryError(Exception):
    """The data source could not answer. Callers must treat this as "no value", never as 0."""


@dataclass(frozen=True)
class InstantSample:
    labels: LabelKey
    timestamp: float
    value: float


def label_key(metric: dict[str, str]) -> LabelKey:
    # __name__ cannot be a label in the output exposition, and aggregated queries rarely
    # have it anyway.
    return tuple(sorted((k, v) for k, v in metric.items() if k != "__name__"))


def _to_float(raw: str) -> float | None:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    # NaN and Inf are "no value" for our purposes: feeding them to a model would poison it.
    return v if math.isfinite(v) else None


def _build_verify(cfg: DatasourceConfig) -> ssl.SSLContext | bool:
    tls = cfg.tls
    if tls.insecure_skip_verify:
        return False
    if tls.ca_file or tls.cert_file:
        ctx = ssl.create_default_context(cafile=str(tls.ca_file) if tls.ca_file else None)
        if tls.cert_file:
            ctx.load_cert_chain(str(tls.cert_file), str(tls.key_file) if tls.key_file else None)
        return ctx
    return True


class PromClient:
    def __init__(self, cfg: DatasourceConfig, transport: httpx.BaseTransport | None = None):
        self._cfg = cfg
        headers = dict(cfg.headers)
        auth: httpx.Auth | None = None
        if cfg.auth.type == "bearer":
            headers["Authorization"] = f"Bearer {cfg.auth.resolve_token()}"
        elif cfg.auth.type == "basic":
            auth = httpx.BasicAuth(cfg.auth.username or "", cfg.auth.resolve_password())
        self._http = httpx.Client(
            headers=headers,
            auth=auth,
            timeout=cfg.timeout,
            verify=_build_verify(cfg) if transport is None else True,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def _get(self, path: str, params: dict[str, str | float]) -> dict:
        url = f"{self._cfg.api_base}{path}"
        try:
            # POST would allow longer queries, but GET is accepted by every compatible store.
            resp = self._http.get(url, params=params)
        except httpx.HTTPError as exc:
            raise QueryError(f"request failed: {exc.__class__.__name__}: {exc}") from exc
        try:
            body = resp.json()
        except ValueError:
            body = None
        if resp.status_code != 200 or not isinstance(body, dict) or body.get("status") != "success":
            detail = body.get("error") if isinstance(body, dict) else resp.text[:200]
            raise QueryError(f"HTTP {resp.status_code}: {detail}")
        return body["data"]

    def query_instant(self, query: str, ts: float) -> list[InstantSample]:
        """Instant query at ts. An empty list means the query matched nothing."""
        data = self._get("/query", {"query": query, "time": ts})
        if data.get("resultType") != "vector":
            raise QueryError(f"expected a vector result, got {data.get('resultType')}")
        out: list[InstantSample] = []
        for item in data["result"]:
            value = _to_float(item["value"][1])
            if value is None:
                continue
            out.append(InstantSample(label_key(item["metric"]), float(item["value"][0]), value))
        return out

    def query_range(
        self, query: str, start: float, end: float, step: float
    ) -> dict[LabelKey, list[tuple[float, float]]]:
        """Range query split into chunks that respect the server's point limit.

        Points are returned sorted by time per series. Missing points stay missing.
        """
        chunk_seconds = step * (self._cfg.max_points_per_request - 1)
        result: dict[LabelKey, list[tuple[float, float]]] = {}
        cursor = start
        while cursor <= end:
            chunk_end = min(end, cursor + chunk_seconds)
            data = self._get(
                "/query_range",
                {"query": query, "start": cursor, "end": chunk_end, "step": step},
            )
            if data.get("resultType") != "matrix":
                raise QueryError(f"expected a matrix result, got {data.get('resultType')}")
            for item in data["result"]:
                points = result.setdefault(label_key(item["metric"]), [])
                for ts, raw in item["values"]:
                    value = _to_float(raw)
                    if value is not None:
                        points.append((float(ts), value))
            cursor = chunk_end + step
        for points in result.values():
            points.sort()
        return result
