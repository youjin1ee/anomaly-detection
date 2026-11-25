"""Replay synthetic cases through models and collect event metrics."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..models import create_model
from ..pipeline import SeriesPipeline
from .metrics import AlertRule, CaseResult, Summary, evaluate_case, summarize
from .scenarios import SCENARIOS, Case, build_case, build_normal_case

# name -> (kind, params). `fixed` is special: thresholds come from each case's training data.
BUILTIN_MODELS: dict[str, tuple[str, dict[str, Any]]] = {
    "fixed": ("static", {}),
    "seasonal": ("seasonal", {}),
    "rcf": ("rcf", {}),
    "rcf-time": ("rcf", {"time_features": True}),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


def parse_model(text: str) -> ModelSpec:
    """`seasonal`, or `NAME=BASE:key=value&key=value` for variants. Values are JSON."""
    name, _, rest = text.partition("=")
    if not rest:
        base, params = name, ""
    else:
        base, _, params = rest.partition(":")
    if base not in BUILTIN_MODELS:
        raise ValueError(f"unknown base model {base!r}; known: {sorted(BUILTIN_MODELS)}")
    kind, defaults = BUILTIN_MODELS[base]
    merged = dict(defaults)
    for pair in filter(None, params.split("&")):
        k, _, v = pair.partition("=")
        merged[k] = json.loads(v)
    return ModelSpec(name, kind, merged)


def fixed_thresholds(train: np.ndarray, direction: str) -> dict[str, float]:
    """What a careful operator would pick by looking at the training period: half of the lowest
    and 1.5 times the highest value seen. Tighter limits would fire at the normal daily trough."""
    finite = train[np.isfinite(train)]
    out: dict[str, float] = {}
    if direction in ("below", "both"):
        out["lower"] = float(0.5 * finite.min())
    if direction in ("above", "both"):
        out["upper"] = float(1.5 * finite.max())
    return out


def score_case(spec: ModelSpec, case: Case, step: int) -> tuple[np.ndarray, float]:
    """Returns scores (NaN where no score) and microseconds per sample."""
    params = dict(spec.params)
    if spec.name.split("-")[0] == "fixed" or spec.kind == "static":
        params = fixed_thresholds(case.train_values, case.direction)
    model = create_model(spec.kind, params, float(step), seed=case.seed)
    pipe = SeriesPipeline(model, direction=case.direction)
    scores = np.full(len(case.ts), np.nan)
    t0 = time.perf_counter()
    fed = 0
    for i, (t, v) in enumerate(zip(case.ts, case.values, strict=True)):
        if np.isnan(v):
            continue  # missing samples are skipped, not zero-filled
        fed += 1
        res = pipe.update(float(t), float(v))
        if res is not None:
            scores[i] = res.score
    return scores, (time.perf_counter() - t0) / max(fed, 1) * 1e6


Job = tuple[ModelSpec, dict[str, Any], AlertRule, int]


def _run(job: Job) -> dict[str, Any]:
    spec, case_args, rule, step = job
    if case_args["scenario"] == "normal":
        case = build_normal_case(
            case_args["signal"],
            case_args["direction"],
            case_args["variant"],
            case_args["seed"],
            case_args["train_days"],
            case_args["eval_days"],
            step,
        )
    else:
        case = build_case(
            case_args["scenario"],
            case_args["variant"],
            case_args["seed"],
            case_args["train_days"],
            case_args["eval_days"],
            step,
        )
    scores, us = score_case(spec, case, step)
    res = evaluate_case(scores, case.anomalies, case.eval_start, step, rule)
    return {
        "model": spec.name,
        "scenario": case.scenario,
        "signal": case.signal,
        "result": res,
        "us_per_sample": us,
    }


def run_all(
    specs: list[ModelSpec],
    variants: int,
    seed_base: int,
    train_days: float,
    eval_days: float,
    normal_cases: int,
    normal_eval_days: float,
    step: int,
    rule: AlertRule,
    workers: int,
    scenarios: list[str] | None = None,
) -> list[dict[str, Any]]:
    jobs: list[Job] = []
    for spec in specs:
        for scenario in scenarios or list(SCENARIOS):
            for v in range(variants):
                args = {
                    "scenario": scenario,
                    "variant": v,
                    "seed": seed_base + v,
                    "train_days": train_days,
                    "eval_days": eval_days,
                }
                jobs.append((spec, args, rule, step))
        # Normal behaviour: every signal, direction both (the worst case for false alarms).
        for signal in ("request_rate", "output_tokens", "throughput", "step_time"):
            for v in range(normal_cases):
                args = {
                    "scenario": "normal",
                    "signal": signal,
                    "direction": "both",
                    "variant": v,
                    "seed": seed_base + 500 + v,
                    "train_days": train_days,
                    "eval_days": normal_eval_days,
                }
                jobs.append((spec, args, rule, step))
    # Longest jobs first keeps the pool busy until the end.
    jobs.sort(key=lambda j: j[0].kind != "rcf")
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=4) as pool:
        return list(pool.imap_unordered(_run, jobs, chunksize=1))


def aggregate(rows: list[dict[str, Any]]) -> dict[tuple[str, str], Summary]:
    groups: dict[tuple[str, str], list[CaseResult]] = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["model"])].append(r["result"])
    return {k: summarize(v) for k, v in groups.items()}


def aggregate_normal_by_signal(rows: list[dict[str, Any]]) -> dict[tuple[str, str], Summary]:
    groups: dict[tuple[str, str], list[CaseResult]] = defaultdict(list)
    for r in rows:
        if r["scenario"] == "normal":
            groups[(r["signal"], r["model"])].append(r["result"])
    return {k: summarize(v) for k, v in groups.items()}


def cost_per_sample(rows: list[dict[str, Any]]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        acc[r["model"]].append(r["us_per_sample"])
    return {k: float(np.mean(v)) for k, v in acc.items()}
