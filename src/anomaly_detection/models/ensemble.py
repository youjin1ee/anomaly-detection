"""Ensemble model: several member models, one calibrated score, two alert stages.

Streams the same steps as `ensemble/core.py` (a test keeps them equal):

1. warm-up (`warmup` samples): members learn, nothing is scored (the model returns None).
2. calibration (`cal` samples): members score, the scores are kept, still None.
3. at the end of calibration the raw scores of every member are turned into surprise values
   (`ensemble/tail.py`), combined, and the thresholds of the two stages are taken from the
   budgets on that window.
4. after that every sample gets `score` (0.5 at the ai stage threshold, 0.9 at the human stage
   threshold, 1.0 for a forced alert) and `tier` (0 none, 1 ai, 2 human). The `for:` clause and
   merging of alerts stay in the alert rule, as for every model.

State is rebuilt from the backfill after a restart, which must therefore cover warm-up and
calibration (default 2 + 3 days). The member `fixed` is the data based range: half the range of
the warm-up outside its minimum and maximum.

Parameters: members (names, default awsrcf-nab, subpca, fixed), combine (max, mean, vote),
vote, route (members and combination by metric kind), metric_kind, rules, warmup, cal,
budget_ai, budget_human, per_minute (events per minute per unit of value, for the sparse rule),
zero_is_real. `members` entries are names of a preset or {"kind": ..., "params": ...}.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..ensemble.combine import ROUTES, combine
from ..ensemble.core import (
    FORCED,
    EnsembleConfig,
    EnsembleSpec,
    fit_calibrators,
    prepare_series,
    score01,
    tier_of,
    unit_score,
)
from ..ensemble.rules import (
    DIRECTION,
    SeriesInfo,
    is_excluded,
    series_scale,
    stride_of,
    suppressed,
    zero_drop_now,
)
from .base import ModelOutput, check_params, create_model, register_model

PRESETS: dict[str, tuple[str, dict[str, Any]]] = {
    "awsrcf-nab": (
        "awsrcf",
        {
            "mode": "rcf",
            "shingle_size": 1,
            "num_trees": 100,
            "sample_size": 256,
            "time_decay": 1e-5,
            "score_div": 8.0,
        },
    ),
    "awsrcf": ("awsrcf", {"mode": "caster"}),
    "subpca": ("subpca", {}),
    "seasonal": ("seasonal", {}),
    "rcf": ("rcf", {}),
}
DEFAULT_MEMBERS = ("awsrcf-nab", "subpca", "fixed")
FIXED_MARGIN = 0.5


def fixed_limits(warmup_values: np.ndarray) -> tuple[float, float]:
    """Half the range below the minimum and above the maximum of the warm-up."""
    lo, hi = float(warmup_values.min()), float(warmup_values.max())
    return lo - FIXED_MARGIN * (hi - lo), hi + FIXED_MARGIN * (hi - lo)


class _Ring:
    """The last `size` values, `back(k)` being the value k samples before the newest (k=1)."""

    def __init__(self, size: int):
        self.buf = np.full(size, np.nan)
        self.count = 0

    def push(self, v: float) -> None:
        self.buf[self.count % len(self.buf)] = v
        self.count += 1

    def back(self, k: int) -> float:
        if k > self.count or k > len(self.buf):
            return math.nan
        return float(self.buf[(self.count - k) % len(self.buf)])


class EnsembleModel:
    def __init__(self, params: dict[str, Any], interval: float, seed: int):
        self.cfg = EnsembleConfig(
            budgets=(params.get("budget_ai", 1.0), params.get("budget_human", 0.1))
        )
        self.spd = max(1, round(86400.0 / interval))
        self.warmup = int(params.get("warmup", 2 * self.spd))
        self.cal = int(params.get("cal", 3 * self.spd))
        entries = list(params.get("members", DEFAULT_MEMBERS))
        names = [e if isinstance(e, str) else e.get("name", e["kind"]) for e in entries]
        self.spec = EnsembleSpec(
            "ensemble",
            tuple(names),
            params.get("combine", "max"),
            int(params.get("vote", 2)),
            bool(params.get("route", False)),
            bool(params.get("rules", True)),
        )
        self.info = SeriesInfo(
            params.get("metric_kind", "other"),
            self.spd,
            params.get("per_minute"),
            bool(params.get("zero_is_real", False)),
        )
        self.members: dict[str, Any] = {}
        for e, n in zip(entries, names, strict=True):
            if n != "fixed":
                kind, p = PRESETS[e] if isinstance(e, str) else (e["kind"], e.get("params", {}))
                self.members[n] = create_model(kind, dict(p), interval, seed)
        rc = self.cfg.rules
        self._stride = stride_of(self.spd, rc.points)
        self._ring = _Ring(max((rc.points - 1) * self._stride + 1, self.spd))
        self._n = 0
        self._values: list[float] = []
        self._raws: dict[str, list[float]] = {m: [] for m in (*self.members, "fixed")}
        self._lo = self._hi = math.nan
        self._cals: dict[str, Any] = {}
        self._thr = (math.inf, math.inf)
        self._scale = 0.0
        self._excluded = False

    def _expected(self) -> float:
        vals = [self._ring.back(1 + j * self._stride) for j in range(self.cfg.rules.points)]
        vals = [v for v in vals if math.isfinite(v)]
        return float(np.median(vals)) if vals else math.nan

    def _finalize(self) -> None:
        raws = {m: np.array(r) for m, r in self._raws.items() if np.isfinite(r).any()}
        values = np.array(self._values)
        cal = (self.warmup, self.warmup + self.cal)
        self._cals = fit_calibrators(raws, cal, self.cfg)
        prepared = prepare_series(raws, values, self.info, cal, self.cfg, self.spec.rules)
        self._thr = unit_score([prepared], self.spec, cal, self.cfg).thresholds
        self._scale = series_scale(values, cal)
        self._excluded = is_excluded(values, self.info, cal, self.cfg.rules)
        self._raws, self._values = {}, []  # calibration is done, nothing is kept

    def update(self, timestamp: float, value: float) -> ModelOutput | None:
        i = self._n
        self._n += 1
        raw = {}
        for name, m in self.members.items():
            out = m.update(timestamp, value)
            raw[name] = math.nan if out is None else out.score
        if i == self.warmup - 1:
            self._lo, self._hi = fixed_limits(np.array([*self._values, value]))
        raw["fixed"] = (
            math.nan
            if i < self.warmup or math.isnan(self._lo)
            else float(value < self._lo or value > self._hi)
        )
        expected, yesterday = self._expected(), self._ring.back(self.spd)
        self._ring.push(value)
        if i < self.warmup + self.cal:
            self._values.append(value)
            for m, v in raw.items():
                self._raws[m].append(v)
            if i == self.warmup + self.cal - 1:
                self._finalize()
            return None
        return self._score(raw, value, expected, yesterday)

    def _score(
        self, raw: dict[str, float], value: float, expected: float, yesterday: float
    ) -> ModelOutput:
        if self.spec.route:
            how, vote, names = ROUTES.get(self.info.kind, ROUTES["other"])
        else:
            how, vote, names = self.spec.combine, self.spec.vote, self.spec.members
        xs = [self._cals[m].surprise(np.array([raw[m]])) for m in names if m in self._cals]
        x = float(combine(xs, how, vote)[0]) if xs else math.nan
        if self.spec.rules:
            if self._excluded:
                x = math.nan
            else:
                direction = DIRECTION.get(self.info.kind, "both")
                if math.isfinite(x) and suppressed(
                    value, expected, direction, self._scale, self.cfg.rules
                ):
                    x = 0.0
                if (
                    self.info.kind == "request"
                    and self.info.zero_is_real
                    and zero_drop_now(value, yesterday)
                ):
                    x = FORCED
        ai, human = self._thr
        return ModelOutput(
            score01(x, ai, human),
            expected if math.isfinite(expected) else value,
            tier=tier_of(x, ai, human),
        )


def _factory(params: dict[str, Any], interval: float, seed: int) -> EnsembleModel:
    check_params(
        "ensemble",
        params,
        {
            "members",
            "combine",
            "vote",
            "route",
            "metric_kind",
            "rules",
            "warmup",
            "cal",
            "budget_ai",
            "budget_human",
            "per_minute",
            "zero_is_real",
        },
    )
    return EnsembleModel(params, interval, seed)


register_model("ensemble", _factory)
