"""Replay the BOOM subset through the models: models, parameter rule, subset, statistics.

Every alert on BOOM is a candidate false alarm (no labels). The questions are how many alerts
per series per day each model gives, where they come from, and how well a per series
threshold calibrated on a quiet stretch keeps a false alarm budget.

Parameter rule (nothing depends on values after the warmup, nothing is tuned per series):

- `fixed`: limits half the warmup range below its minimum and above its maximum. The m2 rule
  (half the minimum, 1.5 times the maximum) breaks on zero and negative values, which BOOM has
  after z-scoring, and the NAB rule has the same reason. Needs one day of warmup samples.
- `seasonal`: defaults (daily and weekly, `slot_window` 8, `min_points` 18). The NAB rule gives
  the same numbers at one and five minutes.
- `rcf`, `awsrcf`: `sample_size = clamp(warmup - 50, 64, 1024)` so the forest is full when the
  warmup ends, `norm_window` and `reference_window` from the sampling interval as in NAB.
- `awsrcf-nab`, `tspulse`, `toto-4m`, `subpca`: fixed settings of the NAB evaluation.
"""

from __future__ import annotations

import os
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from ..models.toto import get_engine, score_batch
from ..models.tspulse import TSPulseModel
from .boom_data import BoomMeta
from .metrics import AlertRule
from .replay import (
    Guards,
    RawOutput,
    RecordedSeries,
    SeriesAlerts,
    calibrated_threshold,
    evaluate_series,
    guarded_scores,
    stream_scores,
    warmup_scale,
)
from .runner import BUILTIN_MODELS, ModelSpec

MODELS = ("fixed", "seasonal", "rcf", "awsrcf-nab", "awsrcf", "tspulse", "toto-4m", "subpca")
WARMUP_DAYS = 2.0
LIMIT = 0.1  # false alarms per series per day
CONCURRENCY_WINDOW = 600  # seconds


def model_spec(name: str) -> ModelSpec:
    kinds = {
        "fixed": "static",
        "rcf": "rcf",
        "seasonal": "seasonal",
        "tspulse": "tspulse",
        "toto-4m": "toto",
        "subpca": "subpca",
    }
    if name in kinds:
        return ModelSpec(name, kinds[name])
    if name in BUILTIN_MODELS:  # awsrcf, awsrcf-nab
        kind, params = BUILTIN_MODELS[name]
        return ModelSpec(name, kind, dict(params))
    raise ValueError(f"unknown model {name!r}; known: {MODELS}")


def model_params(name: str, step: int, warmup_end: int) -> dict[str, Any]:
    """Parameters that depend on the sampling interval and the warmup length only."""
    if name in ("rcf", "awsrcf"):
        shared = {
            "sample_size": int(min(1024, max(64, warmup_end - 50))),
            "reference_window": int(max(6, round(3600.0 / step))),
        }
        if name == "rcf":
            return {**shared, "norm_window": int(min(1440, max(288, round(86400.0 / step))))}
        return {**shared, "mode": "caster"}
    return dict(model_spec(name).params)


def fixed_limits(warmup: np.ndarray) -> dict[str, float]:
    lo, hi = float(warmup.min()), float(warmup.max())
    margin = 0.5 * (hi - lo)
    return {"lower": lo - margin, "upper": hi + margin}


def _scatter(n: int, idx: np.ndarray, score: np.ndarray, expected: np.ndarray) -> tuple:
    s, e = np.full(n, np.nan), np.full(n, np.nan)
    s[idx], e[idx] = score, expected
    return s, e


def _toto_batch(series: RecordedSeries) -> RawOutput:
    """Same scores as the one sample at a time path (a test in test_toto keeps them equal),
    many steps per forward pass."""
    t0 = time.perf_counter()
    idx = np.flatnonzero(np.isfinite(series.values))
    engine = get_engine(os.environ["TOTO_MODEL_DIR"], threads=1)
    res = score_batch(engine, series.values[idx], 512, 128, batch_size=256)
    s, e = _scatter(len(series.values), idx, res["score"], res["expected"])
    return RawOutput(s, e, "", time.perf_counter() - t0)


def _tspulse_batch(series: RecordedSeries) -> RawOutput:
    t0 = time.perf_counter()
    idx = np.flatnonzero(np.isfinite(series.values))
    model = TSPulseModel(model_path=os.environ["TSPULSE_MODEL_DIR"], threads=1)
    outs, _ = model.score_offline(series.values[idx], batch_size=64)
    score = np.array([np.nan if o is None else o.score for o in outs])
    expected = np.array([np.nan if o is None else o.expected for o in outs])
    s, e = _scatter(len(series.values), idx, score, expected)
    return RawOutput(s, e, "", time.perf_counter() - t0)


def run_boom(spec: ModelSpec, series: RecordedSeries, step: int, warmup_end: int) -> RawOutput:
    """Raw output of one model on one series, with the parameter rule of this module."""
    nothing = RawOutput(np.full(len(series.ts), np.nan), np.full(len(series.ts), np.nan))
    if spec.kind == "static":
        seen = series.values[:warmup_end]
        seen = seen[np.isfinite(seen)]
        if seen.size < 86400 // step:  # less than a day: a limit cannot be set
            return nothing
        return stream_scores("static", fixed_limits(seen), series, step)
    if spec.kind == "toto":
        return _toto_batch(series)
    if spec.kind == "tspulse":
        return _tspulse_batch(series)
    return stream_scores(spec.kind, model_params(spec.name, step, warmup_end), series, step)


def stratified_subset(
    series: list[RecordedSeries], meta: dict[str, BoomMeta], per_freq: dict[str, int], seed: int = 0
) -> list[RecordedSeries]:
    """At most one variate per BOOM series (variates of one query are correlated), then
    round robin over the strata (frequency, domain, metric type) in random order until
    `per_freq[freq]` series are taken for every frequency."""
    rng = np.random.default_rng(seed)
    by_query: dict[str, list[RecordedSeries]] = defaultdict(list)
    for s in series:
        by_query[s.query].append(s)
    strata: dict[tuple[str, str, str], list[RecordedSeries]] = defaultdict(list)
    for query in sorted(by_query):
        pick = by_query[query][int(rng.integers(len(by_query[query])))]
        m = meta[pick.key]
        strata[(m.freq, m.domain, m.metric_type)].append(pick)
    out: list[RecordedSeries] = []
    for freq, n in per_freq.items():
        groups = [g for k, g in sorted(strata.items()) if k[0] == freq]
        for g in groups:
            rng.shuffle(g)
        order = rng.permutation(len(groups))
        taken = 0
        while taken < n and any(groups):
            for gi in order:
                if groups[gi] and taken < n:
                    out.append(groups[gi].pop())
                    taken += 1
    return out


def spread(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p95": np.nan, "met": np.nan}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p95": float(np.quantile(v, 0.95)),
        "met": float((v <= LIMIT).mean()),  # share of series at or under the limit
    }


def group_spread(
    rows: list[SeriesAlerts], meta: dict[str, BoomMeta], by: Callable[[BoomMeta], str]
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.ready:
            groups[by(meta[r.key])].append(r.per_day)
    return {g: spread(np.array(v)) for g, v in sorted(groups.items())}


def concurrency(
    rows: list[SeriesAlerts], series: dict[str, RecordedSeries], first: dict[str, int]
) -> dict[str, Any]:
    """Alerts in the same 10 minute window of absolute time, over several series.

    BOOM series start at different dates, so only series observed in the same window can
    alert together. `active` counts the series with samples after their warmup (`first`, by key)
    in that window."""
    alerting: dict[int, set[str]] = defaultdict(set)
    n_alerts: Counter[int] = Counter()
    active: Counter[int] = Counter()
    for r in rows:
        ts = series[r.key].ts
        for b in set((ts[first[r.key] :] // CONCURRENCY_WINDOW).tolist()):
            active[b] += 1
        for _, fire, _ in r.spans:
            b = int(ts[fire] // CONCURRENCY_WINDOW)
            alerting[b].add(r.key)
            n_alerts[b] += 1
    sizes = np.array([len(v) for v in alerting.values()])
    total = sum(n_alerts.values())
    if sizes.size == 0:
        return dict(windows=0, max=0, p50=0.0, p90=0.0, alerts_in_5plus=0.0, max_share=0.0)
    in5 = sum(n_alerts[b] for b, v in alerting.items() if len(v) >= 5)
    return {
        "windows": int(sizes.size),
        "max": int(sizes.max()),
        "p50": float(np.median(sizes)),
        "p90": float(np.quantile(sizes, 0.9)),
        "alerts_in_5plus": in5 / total if total else 0.0,
        "max_share": float(max(len(v) / active[b] for b, v in alerting.items())),
    }


def calibration_rows(
    series: list[RecordedSeries],
    raws: dict[str, RawOutput],
    meta_step: dict[str, int],
    warmup_end_of: Callable[[int], int],
    guards: Guards,
    rule: AlertRule,
    budget: float,
    cal_days: float,
    floor: float,
) -> list[dict[str, Any]]:
    """Per series threshold from the calibration stretch (`cal_days` days after the warmup),
    then alerts on everything after it. `floor` is the lowest threshold allowed.
    Also gives the alerts of the plain rule on the same evaluation stretch."""
    out = []
    for s in series:
        step = meta_step[s.key]
        w = warmup_end_of(step)
        c = w + int(cal_days * 86400 / step)
        raw = raws[s.key]
        scores = guarded_scores(raw, s.values, guards, warmup_scale(s.values, w))
        thr = calibrated_threshold(scores, w, c, rule, budget * cal_days, floor)
        # a series without a score in the calibration stretch has no basis for a threshold
        calibrated = bool(np.isfinite(scores[w:c]).any())
        res = {}
        for name, r in (("cal", replace(rule, threshold=thr)), ("base", rule)):
            res[name] = evaluate_series(s, raw, "", step, w, guards, r, eval_start=c)
        out.append(
            {
                "key": s.key,
                "threshold": thr,
                "calibrated": calibrated,
                "cal": res["cal"],
                "base": res["base"],
            }
        )
    return out
