"""Evaluate ensemble specs on units: two stage alerts, metrics, aggregation.

For every unit the models are calibrated on the calibration window, the specs are combined, the
two stage thresholds come from the budgets on the same window, and alerts are counted on the
evaluation window only. Labelled units give detection, misses, precision, false alarms per day,
F1 and delay per stage. Unlabelled units give false alarms per unit per day and the share of
units within the budget of the stage.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..ensemble.alerts import fast_alert_spans
from ..ensemble.combine import ROUTES
from ..ensemble.core import (
    STAGES,
    EnsembleConfig,
    EnsembleSpec,
    UnitScore,
    effective,
    prepare_series,
    unit_score,
)
from .ensemble_data import Unit
from .metrics import AlertRule


def case_counts(
    fires: list[int], events: list[tuple[int, int]], step: float, rule: AlertRule
) -> tuple[int, int, int, list[float]]:
    """(detected events, matched alerts, false alarms, delays in seconds). Same matching as
    `metrics.evaluate_case`: an alert from the event start to `grace` samples after its end
    detects it, the delay is measured from the event start."""
    used: set[int] = set()
    delays: list[float] = []
    detected = 0
    for start, end in events:
        hits = [f for f in fires if start <= f <= end + rule.grace]
        if hits:
            detected += 1
            delays.append((hits[0] - start) * step)
            used.update(hits)
    return detected, len(used), len(fires) - len(used), delays


@dataclass
class StageAgg:
    units: int = 0
    silent: int = 0
    events: int = 0
    detected: int = 0
    alerts: int = 0
    false_alarms: int = 0
    days: float = 0.0
    delays: list[float] = field(default_factory=list)
    per_unit_day: list[float] = field(default_factory=list)
    within_budget: int = 0

    def add(self, other_events: int, detected: int, alerts: int, fa: int, days: float) -> None:
        self.events += other_events
        self.detected += detected
        self.alerts += alerts
        self.false_alarms += fa
        self.days += days

    def metrics(self, labelled: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "units": self.units,
            "silent": self.silent,
            "silent_share": self.silent / self.units if self.units else math.nan,
            "alerts": self.alerts,
            "alerts_per_day": self.alerts / self.days if self.days else math.nan,
        }
        if labelled:
            p = (self.alerts - self.false_alarms) / self.alerts if self.alerts else math.nan
            r = self.detected / self.events if self.events else math.nan
            f1 = (
                2 * p * r / (p + r)
                if not (math.isnan(p) or math.isnan(r)) and p + r > 0
                else (0.0 if self.events else math.nan)
            )
            out.update(
                events=self.events,
                detected=self.detected,
                recall=r,
                miss=1 - r if not math.isnan(r) else math.nan,
                precision=p,
                f1=f1,
                false_alarms=self.false_alarms,
                fa_per_day=self.false_alarms / self.days if self.days else math.nan,
                delay_median_min=float(np.median(self.delays)) / 60 if self.delays else math.nan,
            )
        else:
            v = np.array(self.per_unit_day)
            out.update(
                fa_per_unit_day_mean=float(v.mean()) if v.size else math.nan,
                fa_per_unit_day_p90=float(np.quantile(v, 0.9)) if v.size else math.nan,
                within_budget=self.within_budget / self.units if self.units else math.nan,
            )
        return out


class Aggregator:
    """(dataset, spec name, stage) -> StageAgg."""

    def __init__(self) -> None:
        self.cells: dict[tuple[str, str, str], StageAgg] = defaultdict(StageAgg)
        self.labelled: dict[str, bool] = {}

    def add_unit(self, unit: Unit, spec_name: str, score: UnitScore, cfg: EnsembleConfig) -> None:
        self.labelled[unit.dataset] = unit.events is not None
        days = max((unit.n - unit.eval_start) / unit.spd, 1e-9)
        for i, stage in enumerate(STAGES):
            agg = self.cells[(unit.dataset, spec_name, stage)]
            agg.units += 1
            agg.silent += int(score.silent[i])
            thr = score.thresholds[i]
            rule = cfg.rule(effective(thr))
            fires = [f for _, f, _ in fast_alert_spans(score.x, rule, unit.eval_start)]
            if unit.events is not None:
                det, _matched, fa, delays = case_counts(fires, unit.events, unit.step, rule)
                agg.add(len(unit.events), det, len(fires), fa, days)
                agg.delays += delays
            else:
                agg.add(0, 0, len(fires), len(fires), days)
                agg.per_unit_day.append(len(fires) / days)
                agg.within_budget += int(len(fires) / days <= cfg.budgets[i])

    def results(self) -> dict[str, dict[str, dict[str, Any]]]:
        out: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
        for (ds, spec, stage), agg in self.cells.items():
            out[ds][spec][stage] = agg.metrics(self.labelled[ds])
        return {d: dict(s) for d, s in out.items()}


Plan = tuple[EnsembleConfig, list[EnsembleSpec], str]


def evaluate_units(units, plans: list[Plan], agg: Aggregator, log=print) -> None:
    """Add every unit's results for every plan to `agg`. A plan is a config, its specs and a
    suffix for the spec names (ablations use another config)."""
    for k, unit in enumerate(units, 1):
        for cfg, specs, suffix in plans:
            need = {m for sp in specs for m in sp.members} | {
                m for _, _, ms in ROUTES.values() for m in ms
            }
            prepared = [
                prepare_series(
                    {m: r for m, r in s.raws.items() if m in need},
                    s.values,
                    s.info,
                    unit.cal,
                    cfg,
                    any(sp.rules for sp in specs),
                )
                for s in unit.series
            ]
            for sp in specs:
                agg.add_unit(unit, sp.name + suffix, unit_score(prepared, sp, unit.cal, cfg), cfg)
        if k % 50 == 0:
            log(f"{unit.dataset} {k} units", flush=True)
