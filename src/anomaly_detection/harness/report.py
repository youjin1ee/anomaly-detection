"""Markdown tables for harness results."""

from __future__ import annotations

from typing import Any

from .metrics import Summary
from .runner import aggregate, aggregate_normal_by_signal, cost_per_sample


def _f(x: float, nd: int = 2) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def render(rows: list[dict[str, Any]], model_order: list[str]) -> str:
    agg = aggregate(rows)
    scenarios = sorted({s for s, _ in agg if s != "normal"})
    out: list[str] = []
    out.append("### Silent failure scenarios\n")
    out.append(
        "| scenario | model | events | detected | recall | alerts | false alarms | precision | F1 "
        "| delay median (min) | delay p90 (min) | false alarms per day |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sc in scenarios:
        for m in model_order:
            s: Summary | None = agg.get((sc, m))
            if s is None:
                continue
            out.append(
                f"| {sc} | {m} | {s.events} | {s.detected} | {_f(s.recall)} | {s.alerts} "
                f"| {s.false_alarms} | {_f(s.precision)} | {_f(s.f1)} "
                f"| {_f(s.median_delay_min, 1)} "
                f"| {_f(s.p90_delay_min, 1)} | {_f(s.false_alarms_per_day)} |"
            )
    out.append("")
    out.append("### Normal behaviour (no injected anomaly, direction both)\n")
    out.append(
        "| signal | model | series | eval days per series | false alarms | per series per day |"
    )
    out.append("|---|---|---|---|---|---|")
    norm = aggregate_normal_by_signal(rows)
    for (sig, m), s in sorted(norm.items(), key=lambda kv: (kv[0][0], model_order.index(kv[0][1]))):
        series = sum(
            1 for r in rows if r["scenario"] == "normal" and r["signal"] == sig and r["model"] == m
        )
        per_series_days = s.eval_days / series
        out.append(
            f"| {sig} | {m} | {series} | {_f(per_series_days, 1)} | {s.false_alarms} "
            f"| {_f(s.false_alarms_per_day)} |"
        )
    out.append("")
    out.append("### Cost\n")
    out.append("| model | microseconds per sample (single process, includes scoring) |")
    out.append("|---|---|")
    for m, us in cost_per_sample(rows).items():
        out.append(f"| {m} | {us:.0f} |")
    return "\n".join(out) + "\n"
