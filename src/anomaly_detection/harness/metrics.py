"""Event level metrics.

Definitions (also written down in docs/evaluation.md):

- An alert fires when the score stays at or above `threshold` for `for_points` consecutive
  samples, like `for:` in an alerting rule. The alert fires at the last of those samples.
- Runs of high scores separated by at most `merge_gap` samples are one alert (similar to
  keep_firing_for). Only alerts that fire inside the evaluation period count.
- A true event is detected when an alert fires between its start and `grace` samples after its
  end. Detection delay is the time from event start to that alert.
- An alert that matches no true event is a false alarm. Precision is matched alerts divided by
  all alerts. Recall is detected events divided by true events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlertRule:
    threshold: float = 0.5
    for_points: int = 3
    merge_gap: int = 10
    grace: int = 5


def alert_spans(
    scores: np.ndarray, rule: AlertRule, first_index: int = 0
) -> list[tuple[int, int, int]]:
    """Alerts as (first hot sample, firing sample, last hot sample). NaN scores break a run."""
    hot = np.nan_to_num(scores, nan=-1.0) >= rule.threshold
    runs: list[list[int]] = []  # [first, last]
    i, n = 0, len(hot)
    while i < n:
        if hot[i]:
            j = i
            while j + 1 < n and hot[j + 1]:
                j += 1
            if runs and i - runs[-1][1] - 1 <= rule.merge_gap:
                runs[-1][1] = j
            else:
                runs.append([i, j])
            i = j + 1
        else:
            i += 1
    spans: list[tuple[int, int, int]] = []
    for first, last in runs:
        # the alert fires when the first stretch of for_points consecutive hot samples is complete
        k = first
        while k <= last:
            if all(hot[k : k + rule.for_points]) and k + rule.for_points - 1 <= last:
                fire = k + rule.for_points - 1
                # An alert that is already firing when the evaluation period starts counts
                # as firing at its first sample.
                fire = max(fire, first_index) if last >= first_index else fire
                if fire >= first_index:
                    spans.append((first, fire, last))
                break
            k += 1
    return spans


def alerts(scores: np.ndarray, rule: AlertRule, first_index: int = 0) -> list[int]:
    """Sample indices at which an alert fires. NaN scores (no value) break a run."""
    return [fire for _, fire, _ in alert_spans(scores, rule, first_index)]


@dataclass
class CaseResult:
    fires: list[int]
    events: int
    detected: int
    matched_alerts: int
    false_alarms: int
    delays: list[float]  # seconds, one per detected event
    eval_days: float


def evaluate_case(
    scores: np.ndarray,
    anomalies: tuple[tuple[int, int], ...],
    eval_start: int,
    step: int,
    rule: AlertRule,
) -> CaseResult:
    fires = alerts(scores, rule, eval_start)
    matched_alerts = 0
    detected = 0
    delays: list[float] = []
    used: set[int] = set()
    for start, end in anomalies:
        hits = [f for f in fires if start <= f <= end + rule.grace]
        if hits:
            detected += 1
            delays.append((hits[0] - start) * step)
            used.update(hits)
    matched_alerts = len(used)
    eval_days = (len(scores) - eval_start) * step / 86400
    return CaseResult(
        fires,
        len(anomalies),
        detected,
        matched_alerts,
        len(fires) - matched_alerts,
        delays,
        eval_days,
    )


@dataclass
class Summary:
    events: int
    detected: int
    alerts: int
    false_alarms: int
    eval_days: float
    delays: list[float]

    @property
    def precision(self) -> float:
        return (self.alerts - self.false_alarms) / self.alerts if self.alerts else float("nan")

    @property
    def recall(self) -> float:
        return self.detected / self.events if self.events else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if np.isnan(p) or np.isnan(r) or p + r == 0:
            return 0.0 if self.events else float("nan")
        return 2 * p * r / (p + r)

    @property
    def false_alarms_per_day(self) -> float:
        return self.false_alarms / self.eval_days if self.eval_days else float("nan")

    @property
    def median_delay_min(self) -> float:
        return float(np.median(self.delays)) / 60 if self.delays else float("nan")

    @property
    def p90_delay_min(self) -> float:
        return float(np.percentile(self.delays, 90)) / 60 if self.delays else float("nan")


def summarize(results: list[CaseResult]) -> Summary:
    return Summary(
        events=sum(r.events for r in results),
        detected=sum(r.detected for r in results),
        alerts=sum(len(r.fires) for r in results),
        false_alarms=sum(r.false_alarms for r in results),
        eval_days=sum(r.eval_days for r in results),
        delays=[d for r in results for d in r.delays],
    )
