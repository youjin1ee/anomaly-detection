"""Metrics for the multivariate benchmarks, on top of the event metrics in `metrics.py`.

- `server_scores`: dimension scores to one score per row, the maximum over the dimensions.
- `event_f1_at` / `oracle_event_f1`: the alert rule of `metrics.py` (score at least 0.5 for 3
  rows, merge gap 10, grace 5) for the operating point, and the same event F1 at the best
  threshold for the oracle. The oracle picks its threshold on the test labels and says
  nothing about what a deployment would get.
- `average_precision`: area under the precision recall curve, row by row, ties handled by
  grouping equal scores. Chosen over VUS-PR because it is a few lines, needs no buffer
  parameter and has no sweep over buffer lengths.
- `point_adjust_f1`: the point adjust protocol (one hit inside a true event marks the whole
  event as found). Reference only, it is known to inflate results.
- `cause_hits`: how often the dimension with the highest score inside an event is one of the
  dimensions listed for that event.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .bench_data import label_events
from .metrics import AlertRule, CaseResult, Summary, evaluate_case, summarize

DEFAULT_RULE = AlertRule(threshold=0.5, for_points=3, merge_gap=10, grace=5)
# thresholds of the oracle sweep: even steps plus a fine grid near 1, where saturating scores
# (logistic mappings) separate
ORACLE_GRID = np.unique(
    np.concatenate([np.linspace(0.05, 0.95, 19), [0.97, 0.98, 0.99, 0.995, 0.999, 0.9999]])
)


def server_scores(dim_scores: np.ndarray) -> np.ndarray:
    """Maximum over dimensions (axis 1), NaN where no dimension has a score."""
    out = np.full(dim_scores.shape[0], np.nan)
    has = ~np.all(np.isnan(dim_scores), axis=1)
    out[has] = np.nanmax(dim_scores[has], axis=1)
    return out


def event_case(scores: np.ndarray, labels: np.ndarray, step: float, rule: AlertRule) -> CaseResult:
    return evaluate_case(scores, tuple(label_events(labels)), 0, int(step), rule)


def event_f1_at(
    per_server: Sequence[tuple[np.ndarray, np.ndarray]], step: float, rule: AlertRule
) -> Summary:
    """Event summary pooled over the servers (sums of events and alerts)."""
    return summarize([event_case(s, y, step, rule) for s, y in per_server])


def _with_threshold(rule: AlertRule, thr: float) -> AlertRule:
    return AlertRule(thr, rule.for_points, rule.merge_gap, rule.grace)


def oracle_event_f1(
    per_server: Sequence[tuple[np.ndarray, np.ndarray]],
    step: float,
    rule: AlertRule = DEFAULT_RULE,
    grid: np.ndarray = ORACLE_GRID,
) -> tuple[float, float, Summary]:
    """(best F1, its threshold, summary there). One threshold for all servers."""
    best: tuple[float, float, Summary] | None = None
    for thr in grid:
        summ = event_f1_at(per_server, step, _with_threshold(rule, float(thr)))
        f1 = summ.f1
        if np.isnan(f1):
            continue
        if best is None or f1 > best[0]:
            best = (f1, float(thr), summ)
    if best is None:
        raise ValueError("no event in the data")
    return best


def _clean(scores: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Row by row AP over all rows given (concatenate the servers first). NaN scores count as
    0. Rows with equal scores are one step of the curve."""
    s = _clean(scores)
    y = np.asarray(labels, dtype=bool)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    last = np.concatenate([s[1:] != s[:-1], [True]])  # last row of each group of equal scores
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp)
    recall = tp / pos
    prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev) * precision))


def point_adjust_counts(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[int, int, int]:
    pred = _clean(scores) >= threshold
    y = np.asarray(labels, dtype=bool)
    adj = pred.copy()
    for a, b in label_events(labels):
        if pred[a : b + 1].any():
            adj[a : b + 1] = True
    tp = int((adj & y).sum())
    fp = int((adj & ~y).sum())
    fn = int((~adj & y).sum())
    return tp, fp, fn


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def pa_oracle_f1(
    per_server: Sequence[tuple[np.ndarray, np.ndarray]], grid: np.ndarray = ORACLE_GRID
) -> tuple[float, float]:
    """(best point adjust F1 over thresholds shared by the servers, threshold)."""
    best = (-1.0, float("nan"))
    for thr in grid:
        tot = np.sum([point_adjust_counts(s, y, float(thr)) for s, y in per_server], axis=0)
        f1 = _f1(*(int(v) for v in tot))
        if f1 > best[0]:
            best = (f1, float(thr))
    return best


def pa_f1_at(per_server: Sequence[tuple[np.ndarray, np.ndarray]], threshold: float) -> float:
    tot = np.sum([point_adjust_counts(s, y, threshold) for s, y in per_server], axis=0)
    return _f1(*(int(v) for v in tot))


def cause_hits(
    dim_scores: np.ndarray,
    labels: np.ndarray,
    interpretation: Sequence[tuple[int, int, tuple[int, ...]]],
    usable_dims: Sequence[int],
    min_score: float | None = None,
) -> tuple[float, int, float]:
    """For every labelled event: the dimension with the highest score inside it, and whether
    it is among the dimensions that the interpretation lists for the event (the union over
    the interpretation segments that overlap the event). Ties count as the share of tied
    dimensions that are right, which is what a random tie break would give.

    Returns (hits, events counted, chance), `chance` being the expected hits of picking a
    usable dimension at random. With `min_score` only events whose best score reaches it
    are counted. Events without any interpretation segment or without a score are skipped."""
    usable = list(usable_dims)
    hits = chance = 0.0
    n = 0
    for a, b in label_events(labels):
        truth = {
            d
            for s, e, dims in interpretation
            if s <= b and e - 1 >= a  # overlap of [s, e) and [a, b]
            for d in dims
        } & set(usable)
        if not truth:
            continue
        seg = dim_scores[a : b + 1][:, usable]
        if np.all(np.isnan(seg)):
            continue
        best = np.nan_to_num(seg, nan=-1.0).max(axis=0)  # per usable dim
        top = best.max()
        if top < 0 or (min_score is not None and top < min_score):
            continue
        tied = [d for d, v in zip(usable, best, strict=True) if v == top]
        hits += sum(d in truth for d in tied) / len(tied)
        chance += len(truth) / len(usable)
        n += 1
    return hits, n, chance
