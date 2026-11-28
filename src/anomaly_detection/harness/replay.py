"""Replay recorded metric dumps through the models and count alerts.

Input is any Prometheus range query dump in CSV form, one file per query (plain or gzip), with
the columns `timestamp` (unix seconds), `labels` (a JSON object, or empty) and `value`. Each
distinct label set in a file is one series. The file name without its extension is the query
name.

The first `warmup_days` of the dump are training only. Alerts are counted after that, with the
same alert rule as the synthetic harness. Model outputs are cached as raw scores, so guard
settings and alert thresholds can be changed and re-evaluated without running the models again.
There are no labels in a recorded dump: every alert is a candidate false alarm until someone
explains it.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..models import create_model
from ..pipeline import SeriesPipeline
from .metrics import AlertRule, alert_spans
from .runner import ModelSpec, fixed_thresholds

CONTEXTS = ("gap_resume", "idle_to_burst", "burst_end", "other_up", "other_down", "no_history")


@dataclass(frozen=True)
class RecordedSeries:
    query: str  # file name without extension
    labels: str  # label set as written in the file (JSON text)
    ts: np.ndarray  # int64, regular grid shared by the whole dump
    values: np.ndarray  # NaN where the series has no sample

    @property
    def key(self) -> str:
        return f"{self.query}|{self.labels}"


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return open(path, newline="")


def read_dump(path: Path) -> dict[str, tuple[list[int], list[float]]]:
    """One file to {labels: (timestamps, values)}. Rows that do not parse are skipped."""
    out: dict[str, tuple[list[int], list[float]]] = {}
    with _open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                t, v = int(float(row["timestamp"])), float(row["value"])
            except (KeyError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            ts, vs = out.setdefault(row.get("labels") or "", ([], []))
            ts.append(t)
            vs.append(v)
    return out


def load_dumps(paths: list[Path], step: int) -> list[RecordedSeries]:
    """Read files or directories of dumps and put every series on one regular grid.

    Samples are snapped to the grid (a later duplicate wins). Missing steps stay NaN and are
    skipped by the replay, never filled with zero."""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(p.glob("*.csv")) + sorted(p.glob("*.csv.gz"))
        else:
            files.append(p)
    raw: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for f in files:
        name = f.name.removesuffix(".gz").removesuffix(".csv")
        for labels, (ts, vs) in sorted(read_dump(f).items()):
            raw.append((name, labels, np.asarray(ts, dtype=np.int64), np.asarray(vs, float)))
    if not raw:
        return []
    t0 = min(int(r[2].min()) for r in raw) // step * step
    t1 = max(int(r[2].max()) for r in raw) // step * step
    grid = np.arange(t0, t1 + step, step, dtype=np.int64)
    series = []
    for name, labels, ts, vs in raw:
        values = np.full(len(grid), np.nan)
        values[np.round((ts - t0) / step).astype(np.int64)] = vs
        series.append(RecordedSeries(name, labels, grid, values))
    return series


@dataclass(frozen=True)
class Guards:
    """Low-traffic protection, derived from the warmup period of each series.

    The floors are fractions of the series' own typical peak (99th percentile of the warmup
    values), because one replay covers queries in different units. In a deployment the
    operator writes the equivalent absolute numbers as `min_expected` and `tolerance`.

    min_expected_frac  downward moves are ignored while the expected value is below this
    tolerance_frac     deviations up to this are ignored, in both directions
    tolerance_ratio    deviations up to this fraction of the expected value are ignored
    """

    min_expected_frac: float = 0.0
    tolerance_frac: float = 0.0
    tolerance_ratio: float = 0.0

    @property
    def active(self) -> bool:
        return bool(self.min_expected_frac or self.tolerance_frac or self.tolerance_ratio)


@dataclass
class RawOutput:
    """Model output before guards. NaN where the model gave nothing."""

    score: np.ndarray
    expected: np.ndarray
    error: str = ""  # the model raised; scores stop at that sample
    seconds: float = 0.0  # wall time of the replay of this series


def warmup_scale(values: np.ndarray, warmup_end: int) -> float:
    """Typical peak of the series during warmup (99th percentile), 0 if nothing was seen."""
    seen = values[:warmup_end]
    seen = seen[np.isfinite(seen)]
    return float(np.quantile(seen, 0.99)) if seen.size else 0.0


def stream_scores(
    kind: str, params: dict[str, Any], series: RecordedSeries, step: int
) -> RawOutput:
    """Feed one series to a model, one sample at a time, skipping missing samples."""
    t0 = time.perf_counter()
    seed = int(hashlib.sha1(series.key.encode()).hexdigest()[:3], 16)
    pipe = SeriesPipeline(create_model(kind, params, float(step), seed=seed))
    score = np.full(len(series.ts), np.nan)
    expected = np.full(len(series.ts), np.nan)
    error = ""
    for i, (t, v) in enumerate(zip(series.ts, series.values, strict=True)):
        if np.isnan(v):
            continue
        try:
            res = pipe.update(float(t), float(v))
        except Exception as exc:  # a model that breaks on one series must not stop the replay
            error = f"{type(exc).__name__} at sample {i}"
            break
        if res is not None:
            score[i], expected[i] = res.score, res.expected
    close = getattr(pipe._model, "close", None)
    if close is not None:
        close()  # a model that lives outside the process (awsrcf) frees its series
    return RawOutput(score, expected, error, time.perf_counter() - t0)


def run_model(spec: ModelSpec, series: RecordedSeries, step: int, warmup_end: int) -> RawOutput:
    params = dict(spec.params)
    if spec.name.split("-")[0] == "fixed" or spec.kind == "static":
        seen = series.values[:warmup_end]
        if np.isfinite(seen).sum() < 1440:  # no usable history: a fixed limit cannot be set
            return RawOutput(np.full(len(series.ts), np.nan), np.full(len(series.ts), np.nan))
        params = fixed_thresholds(seen, "both")
    return stream_scores(spec.kind, params, series, step)


def cache_name(spec: ModelSpec, series: RecordedSeries, step: int, warmup_end: int) -> str:
    data = hashlib.sha1(np.ascontiguousarray(series.values).tobytes()).hexdigest()[:12]
    ident = json.dumps(
        [spec.name, spec.kind, spec.params, series.key, step, warmup_end, data], sort_keys=True
    )
    return hashlib.sha1(ident.encode()).hexdigest() + ".npz"


def cached_run(
    spec: ModelSpec,
    series: RecordedSeries,
    step: int,
    warmup_end: int,
    cache_dir: Path | None,
    run: Callable[[ModelSpec, RecordedSeries, int, int], RawOutput] = run_model,
) -> RawOutput:
    """`run` replaces the default one-sample-at-a-time replay (for example a batched path)."""
    path = cache_dir / cache_name(spec, series, step, warmup_end) if cache_dir else None
    if path is not None and path.exists():
        with np.load(path) as z:
            sec = float(z["seconds"]) if "seconds" in z else 0.0
            return RawOutput(z["score"], z["expected"], str(z["error"]), sec)
    out = run(spec, series, step, warmup_end)
    if path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, score=out.score, expected=out.expected, error=out.error, seconds=out.seconds
        )
    return out


def guarded_scores(
    raw: RawOutput, values: np.ndarray, guards: Guards, scale: float, direction: str = "both"
) -> np.ndarray:
    """Apply direction and guards to stored model scores, the way SeriesPipeline does."""
    floor, tol = guards.min_expected_frac * scale, guards.tolerance_frac * scale
    # Same rules as apply_guards, on whole arrays (a test keeps the two in step).
    ok = np.isfinite(raw.score)
    dev = np.where(ok, values - raw.expected, 0.0)
    zero = np.zeros(len(values), dtype=bool)
    if direction == "above":
        zero |= dev <= 0
    elif direction == "below":
        zero |= dev >= 0
    zero |= (dev < 0) & (raw.expected < floor)
    zero |= (np.abs(dev) <= tol) | (np.abs(dev) <= guards.tolerance_ratio * np.abs(raw.expected))
    return np.where(ok, np.where(zero, 0.0, raw.score), np.nan)


def alert_context(values: np.ndarray, first: int, idle_level: float, lookback: int = 30) -> str:
    """Why did the alert start? Looks at the `lookback` samples before it.

    gap_resume     the series had missing samples just before
    idle_to_burst  the series was idle (at or below idle_level) and now is not
    burst_end      the series was active and is now idle
    other_up       any other upward move, other_down any other downward move
    no_history     nothing before the alert to compare with
    """
    prev = values[max(0, first - lookback) : first]
    seen = prev[np.isfinite(prev)]
    if seen.size == 0:
        return "no_history"
    if seen.size < prev.size:
        return "gap_resume"
    before, now = float(np.median(seen)), float(values[first])
    if before <= idle_level < now:
        return "idle_to_burst"
    if now <= idle_level < before:
        return "burst_end"
    return "other_up" if now > before else "other_down"


def calibrated_threshold(
    scores: np.ndarray, start: int, end: int, rule: AlertRule, allowed: float, floor: float = 0.0
) -> float:
    """Lowest threshold in [floor, 1] at which `rule` fires at most `allowed` times on
    scores[start:end] (a period with no labels, so every alert there is a false alarm).

    A false alarm budget of `b` per day over `d` days of calibration gives `allowed = b * d`.
    Alerts are counted with the real rule (consecutive samples, merged runs), not as exceedances
    of single samples. A model that scores nothing there gets `floor`."""
    window = scores[start:end]

    def count(thr: float) -> int:
        return len(alert_spans(window, replace(rule, threshold=thr)))

    if count(floor) <= allowed:
        return floor
    lo, hi = floor, 1.0 + 1e-9  # count(lo) > allowed, count(hi) == 0
    for _ in range(20):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if count(mid) <= allowed else (mid, hi)
    return hi


@dataclass
class SeriesAlerts:
    key: str
    query: str
    labels: str
    model: str
    observed_days: float  # span from the first to the last sample after warmup, at least one day
    spans: list[tuple[int, int, int]] = field(default_factory=list)  # first, fire, last index
    contexts: list[str] = field(default_factory=list)
    ready: bool = True  # the model produced at least one score after warmup
    error: str = ""  # the model raised on this series

    @property
    def per_day(self) -> float:
        return len(self.spans) / self.observed_days if self.observed_days else float("nan")


def evaluate_series(
    series: RecordedSeries,
    raw: RawOutput,
    model: str,
    step: int,
    warmup_end: int,
    guards: Guards,
    rule: AlertRule,
    idle_frac: float = 0.01,
    eval_start: int | None = None,
) -> SeriesAlerts:
    """Alerts after `eval_start` (default: the end of the warmup, which also fixes the scale
    of the guards and of the idle level)."""
    start = warmup_end if eval_start is None else eval_start
    scale = warmup_scale(series.values, warmup_end)
    scores = guarded_scores(raw, series.values, guards, scale)
    after = np.flatnonzero(np.isfinite(series.values[start:])) + start
    days = max((after[-1] - after[0] + 1) * step / 86400, 1.0) if after.size else 1.0
    out = SeriesAlerts(
        series.key,
        series.query,
        series.labels,
        model,
        days,
        ready=bool(np.isfinite(scores[start:]).any()),
        error=raw.error,
    )
    out.spans = alert_spans(scores, rule, start)
    idle = idle_frac * scale
    out.contexts = [alert_context(series.values, first, idle) for first, _, _ in out.spans]
    return out


def summarize_model(rows: list[SeriesAlerts], limit: float = 0.1) -> dict[str, Any]:
    """Alerts per series per day and where the alerts come from, for one model."""
    ready = [r for r in rows if r.ready]
    per_day = np.array([r.per_day for r in ready]) if ready else np.array([])
    total = sum(len(r.spans) for r in rows)
    days = sum(r.observed_days for r in ready)
    ctx = Counter(c for r in rows for c in r.contexts)
    return {
        "series": len(rows),
        "series_scored": len(ready),
        "series_failed": sum(1 for r in rows if r.error),
        "alerts": total,
        "alerts_per_series_day": total / days if days else float("nan"),
        "median_per_day": float(np.median(per_day)) if per_day.size else float("nan"),
        "p90_per_day": float(np.quantile(per_day, 0.9)) if per_day.size else float("nan"),
        "max_per_day": float(per_day.max()) if per_day.size else float("nan"),
        "series_over_limit": int((per_day > limit).sum()),
        "series_with_alert": sum(1 for r in ready if r.spans),
        "contexts": {c: ctx.get(c, 0) for c in CONTEXTS},
    }
