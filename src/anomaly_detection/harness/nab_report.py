"""NAB result tables: NAB scores per profile, per category, event metrics, public scoreboard."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .metrics import AlertRule, Summary
from .nab_data import CATEGORIES, NabSeries
from .nab_runner import event_summaries, hold_for
from .nab_score import (
    PROFILES,
    FileEvents,
    best_detecting_threshold,
    best_threshold,
    build_events,
    normalize,
    null_and_perfect,
    raw_at,
)

PROFILE_COLUMNS = ("standard", "reward_low_FP_rate", "reward_low_FN_rate")

# Published scoreboard of the NAB README at the commit the data was taken from
# (https://github.com/numenta/NAB#scoreboard). Standard, Reward Low FP, Reward Low FN.
# Scores of other detectors were produced by their authors with the official tooling.
SCOREBOARD: tuple[tuple[str, float, float, float], ...] = (
    ("Perfect", 100.0, 100.0, 100.0),
    ("ARTime", 74.9, 65.1, 80.4),
    ("Numenta HTM (NuPIC 1.0, range upper end)", 70.5, 62.6, 75.2),
    ("CAD OSE", 69.9, 67.0, 73.2),
    ("earthgecko Skyline", 58.2, 46.2, 63.9),
    ("KNN CAD", 58.0, 43.4, 64.8),
    ("Relative Entropy", 54.6, 47.6, 58.8),
    ("Random Cut Forest (AWS proprietary implementation)", 51.7, 38.4, 59.7),
    ("Twitter ADVec v1.0.0", 47.1, 33.6, 53.5),
    ("Windowed Gaussian", 39.6, 20.9, 47.4),
    ("Etsy Skyline", 35.7, 27.1, 44.5),
    ("Bayesian Changepoint", 17.7, 3.2, 32.2),
    ("EXPoSE", 16.4, 3.2, 26.9),
    ("Random (mean over seeds)", 11.0, 1.2, 19.5),
    ("Null", 0.0, 0.0, 0.0),
)


@dataclass
class ModelNab:
    """NAB numbers of one model."""

    optimized: dict[str, tuple[float, float]]  # profile -> (threshold, normalised score)
    detecting: dict[str, tuple[float, float]]  # same, restricted to thresholds that detect
    fixed: dict[str, float]  # profile -> normalised score at the fixed threshold
    by_category: dict[str, dict[str, float]]  # profile -> category -> normalised, optimized
    by_category_fixed: dict[str, dict[str, float]]  # same at the fixed threshold


def _norm(files: Sequence[FileEvents], profile_name: str, threshold: float) -> float:
    null, perfect = null_and_perfect(files, PROFILES[profile_name])
    if perfect == 0:
        return float("nan")
    return normalize(raw_at(files, threshold), null, perfect)


def nab_numbers(
    corpus: Sequence[NabSeries], scores: dict[str, np.ndarray], fixed_threshold: float = 0.5
) -> ModelNab:
    optimized: dict[str, tuple[float, float]] = {}
    detecting: dict[str, tuple[float, float]] = {}
    fixed: dict[str, float] = {}
    by_cat: dict[str, dict[str, float]] = {}
    by_cat_fixed: dict[str, dict[str, float]] = {}
    windows = [s.windows for s in corpus]
    arrays = [scores[s.path] for s in corpus]
    for pname, profile in PROFILES.items():
        events = build_events(arrays, windows, profile)
        thr, raw = best_threshold(events)
        null, perfect = null_and_perfect(events, profile)
        optimized[pname] = (thr, normalize(raw, null, perfect))
        thr_d, raw_d = best_detecting_threshold(events)
        detecting[pname] = (thr_d, normalize(raw_d, null, perfect))
        fixed[pname] = normalize(raw_at(events, fixed_threshold), null, perfect)
        by_cat[pname], by_cat_fixed[pname] = {}, {}
        for cat in CATEGORIES:
            sub = [e for e, s in zip(events, corpus, strict=True) if s.category == cat]
            by_cat[pname][cat] = _norm(sub, pname, thr_d)
            by_cat_fixed[pname][cat] = _norm(sub, pname, fixed_threshold)
    return ModelNab(optimized, detecting, fixed, by_cat, by_cat_fixed)


def _f(x: float, nd: int = 1) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def render(
    corpus: Sequence[NabSeries],
    all_scores: dict[str, dict[str, np.ndarray]],
    model_order: Sequence[str],
    rule: AlertRule,
    us_per_record: dict[str, float] | None = None,
    for_points: int | None = None,
) -> str:
    """`all_scores` may also hold `<model>+for<N>` entries (scores behind the alert rule's
    `for` clause). They get NAB scores, and no event metrics: those are the same as the
    model's, because the event rule already contains the `for` clause."""
    event_models = list(model_order)
    if for_points:
        model_order = [x for m in model_order for x in (m, f"{m}+for{for_points}")]
    numbers = {m: nab_numbers(corpus, all_scores[m]) for m in model_order}
    out: list[str] = []

    out.append("### NAB score, one threshold per model and profile (official protocol)\n")
    out.append(
        "| model | standard | threshold | reward_low_FP_rate | threshold "
        "| reward_low_FN_rate | threshold |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for m in model_order:
        o = numbers[m].optimized
        cells = [f"{_f(o[p][1])} | {o[p][0]:.4f}" for p in PROFILE_COLUMNS]
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out.append("")

    out.append("### Best NAB score among thresholds that detect at least one record\n")
    out.append(
        "| model | standard | threshold | reward_low_FP_rate | threshold "
        "| reward_low_FN_rate | threshold |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for m in model_order:
        o = numbers[m].detecting
        cells = [f"{_f(o[p][1])} | {o[p][0]:.4f}" for p in PROFILE_COLUMNS]
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out.append("")

    out.append(f"### NAB score at the fixed threshold {0.5} (no fitting to the corpus)\n")
    out.append("| model | standard | reward_low_FP_rate | reward_low_FN_rate |")
    out.append("|---|---|---|---|")
    for m in model_order:
        out.append(
            f"| {m} | " + " | ".join(_f(numbers[m].fixed[p]) for p in PROFILE_COLUMNS) + " |"
        )
    out.append("")

    out.append(
        "### Public scoreboard next to these results (standard, reward_low_FP, reward_low_FN)\n"
    )
    out.append("| detector | standard | reward_low_FP_rate | reward_low_FN_rate | source |")
    out.append("|---|---|---|---|---|")
    rows: list[tuple[float, str]] = []
    for name, a, b, c in SCOREBOARD:
        rows.append((a, f"| {name} | {a} | {b} | {c} | NAB README scoreboard |"))
    for m in model_order:
        o = numbers[m].optimized
        s = [o[p][1] for p in PROFILE_COLUMNS]
        rows.append((s[0], f"| this repo: {m} | {s[0]:.1f} | {s[1]:.1f} | {s[2]:.1f} | this run |"))
    out.extend(r for _, r in sorted(rows, key=lambda r: -r[0]))
    out.append("")

    for pname in PROFILE_COLUMNS:
        out.append(f"### NAB score per category, {pname}, best detecting corpus threshold\n")
        out.append("| model | " + " | ".join(CATEGORIES) + " |")
        out.append("|---|" + "---|" * len(CATEGORIES))
        for m in model_order:
            vals = [_f(numbers[m].by_category[pname][c]) for c in CATEGORIES]
            out.append(f"| {m} | " + " | ".join(vals) + " |")
        out.append("")

    out.append("### Event metrics per category (score >= 0.5 for 3 consecutive records)\n")
    out.append(
        "| category | model | windows | detected | recall | alerts | false alarms | precision | F1 "
        "| delay median (min) | delay p90 (min) | false alarms per day |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    summaries = {m: event_summaries(corpus, all_scores[m], rule) for m in event_models}
    for cat in (*CATEGORIES, "all"):
        for m in event_models:
            s: Summary | None = summaries[m].get(cat)
            if s is None:
                continue
            out.append(
                f"| {cat} | {m} | {s.events} | {s.detected} | {_f(s.recall, 2)} | {s.alerts} "
                f"| {s.false_alarms} | {_f(s.precision, 2)} | {_f(s.f1, 2)} "
                f"| {_f(s.median_delay_min)} | {_f(s.p90_delay_min)} "
                f"| {_f(s.false_alarms_per_day, 2)} |"
            )
    out.append("")
    if us_per_record:
        out.append("### Cost\n")
        out.append("| model | microseconds per record (single process) |")
        out.append("|---|---|")
        for m in event_models:
            if m in us_per_record:
                out.append(f"| {m} | {us_per_record[m]:.0f} |")
        out.append("")
    return "\n".join(out)


def render_seeds(
    corpus: Sequence[NabSeries],
    runs: Sequence[dict[str, dict[str, np.ndarray]]],
    models: Sequence[str],
    for_points: int = 3,
) -> str:
    """NAB scores of randomised models over several seeds: per seed, mean and range.
    `runs` holds one `model -> path -> scores` mapping per seed."""
    names = [x for m in models for x in (m, f"{m}+for{for_points}")]
    out = [
        "### NAB score over seeds, one threshold per model and profile",
        "",
        "| model | profile | "
        + " | ".join(f"seed {i}" for i in range(len(runs)))
        + " | mean | min | max |",
        "|---|---|" + "---|" * (len(runs) + 3),
    ]
    per: dict[str, list[ModelNab]] = {n: [] for n in names}
    for run in runs:
        for m in models:
            per[m].append(nab_numbers(corpus, run[m]))
            held = {p: hold_for(s, for_points) for p, s in run[m].items()}
            per[f"{m}+for{for_points}"].append(nab_numbers(corpus, held))
    for n in names:
        for pname in PROFILE_COLUMNS:
            vals = [r.optimized[pname][1] for r in per[n]]
            cells = " | ".join(_f(v) for v in vals)
            out.append(
                f"| {n} | {pname} | {cells} | {_f(float(np.mean(vals)))} "
                f"| {_f(min(vals))} | {_f(max(vals))} |"
            )
    return "\n".join(out) + "\n"
