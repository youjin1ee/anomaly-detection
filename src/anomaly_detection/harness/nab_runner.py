"""Run the models over the NAB corpus and keep the per-record scores.

Parameter rule (no per-file tuning, see docs/results/m2-nab.md):

Every parameter is a function of two things only, the sampling interval estimated from the
first 100 timestamps of the file and the probationary length (15 percent of the records, at
most 750). Both are known before the first scored record. Nothing depends on values, labels
or results.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models import create_model
from ..pipeline import SeriesPipeline
from .metrics import AlertRule, CaseResult, Summary, alerts, summarize
from .nab_data import NabSeries
from .nab_score import probation_length

NAB_MODELS = ("fixed", "seasonal", "rcf", "rcf-time")
PROBE_RECORDS = 100
POOL_SECONDS = 2400.0  # pooling half width of the seasonal model at the default 5 minute data


def estimate_interval(timestamps: np.ndarray, probe: int = PROBE_RECORDS) -> float:
    """Median positive gap of the first `probe` records, in seconds."""
    gaps = np.diff(timestamps[:probe])
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        raise ValueError("cannot estimate the sampling interval")
    return float(np.median(gaps))


def model_params(name: str, interval: float, n_records: int) -> dict[str, Any]:
    """Parameters of a registered model kind for one file. `fixed` is handled separately."""
    if name == "seasonal":
        # Slots pool +-8 neighbours at 5 minutes (about +-40 minutes). The same time span is
        # kept for other resolutions, but never less than one slot on each side.
        w = int(max(1, min(8, round(POOL_SECONDS / interval))))
        return {"slot_window": w, "min_points": min(18, 2 * (2 * w + 1))}
    if name in ("rcf", "rcf-time"):
        probation = probation_length(n_records)
        params: dict[str, Any] = {
            # one day of samples, but at least 288 so that coarse data keeps a usable spread
            "norm_window": int(min(1440, max(288, round(86400.0 / interval)))),
            "reference_window": int(max(6, round(3600.0 / interval))),
            # the forest is full, and so scores, at the end of the probationary period
            "sample_size": int(min(1024, max(64, probation - 50))),
        }
        if name == "rcf-time":
            params["time_features"] = True
        return params
    raise ValueError(f"unknown NAB model {name!r}")


def fixed_limits(values: np.ndarray, probation: int) -> dict[str, float]:
    """Both limits half a range outside what the probationary period showed. The m1 fixed
    threshold (half the minimum, 1.5 times the maximum) breaks on zero and negative values."""
    seen = values[:probation][np.isfinite(values[:probation])]
    lo, hi = float(seen.min()), float(seen.max())
    margin = 0.5 * (hi - lo)
    return {"lower": lo - margin, "upper": hi + margin}


def score_series(name: str, series: NabSeries, seed: int = 0) -> tuple[np.ndarray, float]:
    """Anomaly scores per record (NaN where the model gave none) and microseconds per record."""
    n = len(series.values)
    interval = estimate_interval(series.timestamps)
    if name == "fixed":
        kind, params = "static", fixed_limits(series.values, probation_length(n))
    else:
        kind, params = ("rcf" if name.startswith("rcf") else name), model_params(name, interval, n)
    model = create_model(kind, params, interval, seed=seed)
    pipe = SeriesPipeline(model, direction="both")
    scores = np.full(n, np.nan)
    t0 = time.perf_counter()
    fed = 0
    for i in range(n):
        v = series.values[i]
        if not np.isfinite(v):
            continue
        fed += 1
        res = pipe.update(float(series.timestamps[i]), float(v))
        if res is not None:
            scores[i] = res.score
    return scores, (time.perf_counter() - t0) / max(fed, 1) * 1e6


def _init_worker(mem_limit_gb: float) -> None:
    if mem_limit_gb > 0:
        import resource

        limit = int(mem_limit_gb * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _run(job: tuple[str, NabSeries, int]) -> tuple[str, str, np.ndarray, float]:
    name, series, seed = job
    scores, us = score_series(name, series, seed)
    return name, series.path, scores, us


@dataclass
class ModelScores:
    scores: dict[str, np.ndarray]  # path -> scores
    us_per_record: dict[str, float]  # path -> microseconds


def run_models(
    names: Sequence[str],
    corpus: Sequence[NabSeries],
    workers: int,
    seed: int = 0,
    mem_limit_gb: float = 4.0,
) -> dict[str, ModelScores]:
    jobs = [(name, s, seed) for name in names for s in corpus]
    # Biggest, slowest jobs first so the pool is busy until the end.
    jobs.sort(key=lambda j: (not j[0].startswith("rcf"), -len(j[1].values)))
    out = {name: ModelScores({}, {}) for name in names}
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        workers, initializer=_init_worker, initargs=(mem_limit_gb,), maxtasksperchild=4
    ) as pool:
        for name, path, scores, us in pool.imap_unordered(_run, jobs, chunksize=1):
            out[name].scores[path] = scores
            out[name].us_per_record[path] = us
    return out


def hold_for(scores: np.ndarray, points: int = 3) -> np.ndarray:
    """Smallest score of the last `points` records (missing scores count as 0).

    Thresholding this at any level is the `for: N` clause of an alert rule: it is high only
    when the model score stayed at or above the level for `points` consecutive records.
    """
    x = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0)
    out = x.copy()
    for k in range(1, points):
        shifted = np.concatenate([np.zeros(k), x[:-k]])
        out = np.minimum(out, shifted)
    return out


def evaluate_events(series: NabSeries, scores: np.ndarray, rule: AlertRule) -> CaseResult:
    """Event metrics of one file. Alerts must fire after the probationary period. A window
    is a true event; windows that end inside the probationary period are dropped and one that
    starts inside it is cut at its end. Delay is measured from the window start (the NAB
    window is centred on the labelled anomaly, so a detection before the anomaly itself still
    counts and the delay is not the delay after the anomaly)."""
    n = len(series.values)
    first = probation_length(n)
    fires = alerts(scores, rule, first)
    events = [(max(s, first), e) for s, e in series.windows if e >= first]
    ts = series.timestamps
    used: set[int] = set()
    delays: list[float] = []
    detected = 0
    for start, end in events:
        hits = [f for f in fires if start <= f <= end + rule.grace]
        if hits:
            detected += 1
            delays.append(float(ts[hits[0]] - ts[start]))
            used.update(hits)
    return CaseResult(
        fires=fires,
        events=len(events),
        detected=detected,
        matched_alerts=len(used),
        false_alarms=len(fires) - len(used),
        delays=delays,
        eval_days=float(ts[-1] - ts[first]) / 86400.0,
    )


def event_summaries(
    corpus: Sequence[NabSeries], scores: dict[str, np.ndarray], rule: AlertRule
) -> dict[str, Summary]:
    """Per category (and `all`) event metrics."""
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for s in corpus:
        res = evaluate_events(s, scores[s.path], rule)
        groups[s.category].append(res)
        groups["all"].append(res)
    return {k: summarize(v) for k, v in groups.items()}
