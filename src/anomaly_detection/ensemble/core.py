"""Per series and per unit ensemble judgement, on arrays (the sidecar model streams the same
steps, see `models/ensemble.py`).

A unit is what an alert belongs to: one series, or one machine with several dimensions (the
unit score is the maximum over its series). The stages are two alert levels derived from the
same combined score by two budgets (alerts per unit per day on the calibration window):
`ai` (sensitive, goes to an automatic analysis step) and `human` (precise, pages a person).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..harness.metrics import AlertRule
from .alerts import hold
from .combine import ROUTES, combine
from .rules import RuleConfig, RuleResult, SeriesInfo, apply_rules
from .tail import ScoreCalibrator, budget_threshold

STAGES = ("ai", "human")
FORCED = 1e300  # surprise of a sample a rule forces to alert


def effective(thr: float) -> float:
    """A stage without a threshold still lets forced samples through."""
    return min(thr, 1e299)


@dataclass(frozen=True)
class EnsembleSpec:
    name: str
    members: tuple[str, ...] = ()
    combine: str = "max"
    vote: int = 2
    route: bool = False  # members and combination by metric kind (combine.ROUTES)
    rules: bool = False  # rule layer (rules.py)
    raw: bool = False  # reference: the first member's raw score with a fixed 0.5 threshold


@dataclass(frozen=True)
class EnsembleConfig:
    cal_days: float = 3.0  # calibration window after the warm-up, in days
    budgets: tuple[float, float] = (1.0, 0.1)  # alerts per unit per day: ai, human
    for_points: int = 3
    merge_gap: int = 10
    grace: int = 5
    min_cal: int = 100  # fewer calibration scores: the model is left out for the series
    extrapolate: bool = True  # tail fits; False is the plain maximum of the calibration window
    x_min: float = 1.0  # clusters below this surprise (p = 0.1) are not alerts
    rules: RuleConfig = field(default_factory=RuleConfig)

    def rule(self, threshold: float) -> AlertRule:
        return AlertRule(threshold, self.for_points, self.merge_gap, self.grace)


@dataclass
class PreparedSeries:
    x: dict[str, np.ndarray]  # model -> surprise
    raw: dict[str, np.ndarray]  # model -> raw score
    values: np.ndarray
    info: SeriesInfo
    rule: RuleResult | None = None


def fit_calibrators(
    raws: dict[str, np.ndarray], cal: tuple[int, int], cfg: EnsembleConfig
) -> dict[str, ScoreCalibrator]:
    """Calibrators of the models that have enough scores in the calibration window."""
    fits = {
        name: ScoreCalibrator.fit(raw[cal[0] : cal[1]], cfg.min_cal, cfg.extrapolate)
        for name, raw in raws.items()
    }
    return {name: c for name, c in fits.items() if c.ok}


def prepare_series(
    raws: dict[str, np.ndarray],
    values: np.ndarray,
    info: SeriesInfo,
    cal: tuple[int, int],
    cfg: EnsembleConfig,
    need_rules: bool = True,
) -> PreparedSeries:
    x = {name: c.surprise(raws[name]) for name, c in fit_calibrators(raws, cal, cfg).items()}
    rule = apply_rules(values, info, cal, cfg.rules) if need_rules else None
    return PreparedSeries(x, raws, values, info, rule)


def series_surprise(p: PreparedSeries, spec: EnsembleSpec) -> tuple[np.ndarray, np.ndarray | None]:
    """(combined surprise, forced samples) of one series. Forced is None without rules."""
    n = len(p.values)
    if spec.raw:
        return p.raw.get(spec.members[0], np.full(n, np.nan)), None
    if spec.route:
        how, vote, names = ROUTES.get(p.info.kind, ROUTES["other"])
    else:
        how, vote, names = spec.combine, spec.vote, spec.members
    xs = [p.x[m] for m in names if m in p.x]
    x = combine(xs, how, vote) if xs else np.full(n, np.nan)
    if not spec.rules or p.rule is None:
        return x, None
    if p.rule.excluded:
        return np.full(n, np.nan), None
    x = np.where(p.rule.suppress & np.isfinite(x), 0.0, x)
    return x, p.rule.force


def _nanmax_rows(rows: list[np.ndarray]) -> np.ndarray:
    a = np.stack(rows)
    valid = np.isfinite(a)
    return np.where(valid.any(axis=0), np.where(valid, a, -np.inf).max(axis=0), np.nan)


@dataclass
class UnitScore:
    x: np.ndarray  # final surprise of the unit (FORCED where a rule forces an alert)
    thresholds: tuple[float, float]  # ai, human; inf when no alert is possible
    silent: tuple[bool, bool]  # stage can never fire for this unit


def unit_score(
    prepared: list[PreparedSeries],
    spec: EnsembleSpec,
    cal: tuple[int, int],
    cfg: EnsembleConfig,
) -> UnitScore:
    """Combined surprise of a unit and the two stage thresholds from its calibration window."""
    parts = [series_surprise(p, spec) for p in prepared]
    model_x = _nanmax_rows([x for x, _ in parts])
    forced = [f for _, f in parts if f is not None]
    if spec.raw:
        return UnitScore(model_x, (0.5, 0.5), (False, False))
    cal_x = model_x[cal[0] : cal[1]]
    days = (cal[1] - cal[0]) / prepared[0].info.spd
    usable = bool(np.isfinite(cal_x).any())
    thr = [math.inf, math.inf]
    if usable:
        h = hold(cal_x, cfg.for_points)
        for i, b in enumerate(cfg.budgets):
            thr[i] = budget_threshold(h, days, b, cfg.x_min, cfg.merge_gap, cfg.extrapolate)
        thr[1] = max(thr[1], thr[0])
    x = model_x
    if forced:
        f = np.any(forced, axis=0)
        x = np.where(f, FORCED, model_x)
    return UnitScore(
        x, (thr[0], thr[1]), (not usable or thr[0] == math.inf, not usable or thr[1] == math.inf)
    )


def score01(x: float, thr_ai: float, thr_human: float) -> float:
    """Surprise to a 0 to 1 score: 0.5 at the ai threshold, 0.9 at the human threshold, 1
    only for a forced alert. A rule `anomaly_score >= 0.5` is the ai stage, `>= 0.9` the human."""
    if math.isnan(x) or x <= 0:
        return 0.0
    if x >= FORCED:
        return 1.0
    ai = max(thr_ai, 1e-9)
    if x < ai:
        return 0.5 * x / ai
    human = max(thr_human, ai + 1e-9)
    if x < human:
        return 0.5 + 0.4 * (x - ai) / (human - ai)
    return 0.9 + 0.1 * (1.0 - math.exp(-(x - human)))


def tier_of(x: float, thr_ai: float, thr_human: float) -> int:
    """0 none, 1 ai stage, 2 human stage (the human stage includes the ai stage)."""
    if math.isnan(x):
        return 0
    return 2 if x >= effective(thr_human) else 1 if x >= effective(thr_ai) else 0
