"""YAML configuration: one data source, many detectors."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_DURATION_RE = re.compile(r"^(\d+)\s*(s|m|h|d|w)$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def parse_duration(value: Any) -> float:
    """Accept seconds as a number or strings such as 30s, 5m, 6h, 7d, 2w."""
    if isinstance(value, bool):
        raise ValueError("duration must be a number or a string such as '5m'")
    if isinstance(value, int | float):
        if value <= 0:
            raise ValueError("duration must be positive")
        return float(value)
    if isinstance(value, str):
        match = _DURATION_RE.match(value.strip())
        if match:
            return float(int(match.group(1)) * _UNIT_SECONDS[match.group(2)])
    raise ValueError(f"invalid duration {value!r}; use seconds or forms like 30s, 5m, 6h, 7d")


def _expand_env(node: Any) -> Any:
    """Replace ${VAR} in strings. Only used for the datasource section, because PromQL
    queries may legitimately contain dollar signs."""
    if isinstance(node, str):

        def sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name not in os.environ:
                raise ValueError(f"environment variable {name} is not set")
            return os.environ[name]

        return _ENV_RE.sub(sub, node)
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    return node


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AuthConfig(_Strict):
    type: Literal["none", "bearer", "basic"] = "none"
    token: str | None = None
    token_file: Path | None = None
    username: str | None = None
    password: str | None = None
    password_file: Path | None = None

    @model_validator(mode="after")
    def _check(self) -> AuthConfig:
        if self.type == "bearer" and not (self.token or self.token_file):
            raise ValueError("bearer auth needs token or token_file")
        if self.type == "basic" and not (
            self.username and (self.password is not None or self.password_file)
        ):
            raise ValueError("basic auth needs username and password or password_file")
        return self

    def resolve_token(self) -> str:
        if self.token_file:
            return self.token_file.read_text().strip()
        return self.token or ""

    def resolve_password(self) -> str:
        if self.password_file:
            return self.password_file.read_text().strip()
        return self.password or ""


class TLSConfig(_Strict):
    ca_file: Path | None = None
    cert_file: Path | None = None
    key_file: Path | None = None
    insecure_skip_verify: bool = False


class DatasourceConfig(_Strict):
    url: str
    # Everything between the base URL and /api/v1, for example /select/0/prometheus for a
    # VictoriaMetrics cluster vmselect, or /prometheus for Mimir.
    path_prefix: str = ""
    auth: AuthConfig = Field(default_factory=AuthConfig)
    # Extra request headers, for example X-Scope-OrgID for Mimir or Cortex tenants.
    headers: dict[str, str] = Field(default_factory=dict)
    tls: TLSConfig = Field(default_factory=TLSConfig)
    timeout: float = 30.0
    # Upper bound of points per query_range request. Prometheus rejects more than 11000.
    max_points_per_request: int = Field(default=10000, ge=10, le=11000)

    @field_validator("url")
    @classmethod
    def _url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("path_prefix")
    @classmethod
    def _prefix(cls, v: str) -> str:
        v = v.strip()
        if not v:
            return ""
        return "/" + v.strip("/")

    @property
    def api_base(self) -> str:
        return f"{self.url}{self.path_prefix}/api/v1"


class ModelConfig(_Strict):
    kind: str = "seasonal"
    # Free-form parameters validated by the model implementation itself, so that new model
    # kinds can be added without touching this file.
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, v: Any) -> Any:
        # `model: rcf` and flat parameters next to `kind` are accepted.
        if isinstance(v, str):
            return {"kind": v}
        if isinstance(v, dict) and "params" not in v:
            return {
                "kind": v.get("kind", "seasonal"),
                "params": {k: x for k, x in v.items() if k != "kind"},
            }
        return v


class ToleranceConfig(_Strict):
    # Deviations from the expected value that are ignored: absolute amount and/or a ratio of
    # the expected value (same idea as ignoreNearExpected in AMP).
    amount: float = Field(default=0.0, ge=0)
    ratio: float = Field(default=0.0, ge=0)


class DetectorConfig(_Strict):
    name: str
    query: str
    interval: float | None = None
    direction: Literal["above", "below", "both"] = "both"
    model: ModelConfig = Field(default_factory=ModelConfig)
    warmup_samples: int | None = Field(default=None, ge=1)
    # Absolute floor: when the expected value is below this, scores are suppressed
    # (low traffic protection). A drop from a large expected value to zero is still scored.
    min_expected: float = 0.0
    tolerance: ToleranceConfig = Field(default_factory=ToleranceConfig)
    backfill: float | None = None
    max_series: int | None = Field(default=None, ge=1)
    # Series that stop appearing are forgotten after this long.
    stale_after: float | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not re.match(r"^[A-Za-z0-9_.:-]+$", v):
            raise ValueError("detector name may only contain letters, digits and _ . : -")
        return v

    @field_validator("interval", "backfill", "stale_after", mode="before")
    @classmethod
    def _dur(cls, v: Any) -> float | None:
        return None if v is None else parse_duration(v)


class ServerConfig(_Strict):
    listen: str = "0.0.0.0:8080"


class Defaults(_Strict):
    interval: float = 60.0
    backfill: float = 86400.0
    max_series_per_detector: int = Field(default=100, ge=1)
    max_series_total: int = Field(default=500, ge=1)
    stale_after_intervals: int = Field(default=30, ge=1)

    @field_validator("interval", "backfill", mode="before")
    @classmethod
    def _dur(cls, v: Any) -> float:
        return parse_duration(v)


class Config(_Strict):
    datasource: DatasourceConfig
    server: ServerConfig = Field(default_factory=ServerConfig)
    defaults: Defaults = Field(default_factory=Defaults)
    detectors: list[DetectorConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Config:
        names = [d.name for d in self.detectors]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"duplicate detector names: {sorted(dup)}")
        return self

    # Resolved per-detector settings, so callers never deal with None defaults.
    def interval_of(self, d: DetectorConfig) -> float:
        return d.interval or self.defaults.interval

    def backfill_of(self, d: DetectorConfig) -> float:
        return d.backfill if d.backfill is not None else self.defaults.backfill

    def max_series_of(self, d: DetectorConfig) -> int:
        return d.max_series or self.defaults.max_series_per_detector

    def stale_after_of(self, d: DetectorConfig) -> float:
        return d.stale_after or self.interval_of(d) * self.defaults.stale_after_intervals


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    if "datasource" in raw:
        raw["datasource"] = _expand_env(raw["datasource"])
    return Config.model_validate(raw)
