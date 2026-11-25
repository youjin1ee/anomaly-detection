"""Silent failure scenarios injected into synthetic series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .synth import DAY, derive_step_time, generate, missing_mask, timestamps


@dataclass(frozen=True)
class Case:
    scenario: str  # scenario group, for example request_drop
    signal: str  # series the detector watches
    direction: str  # direction the detector is configured with
    variant: int
    seed: int
    ts: np.ndarray
    values: np.ndarray  # NaN where missing
    anomalies: tuple[tuple[int, int], ...]  # [start, end) sample indices, empty for normal
    eval_start: int  # first sample index that counts for the metrics
    train_values: np.ndarray  # copy of the training part before injection, for fixed thresholds


# scenario -> (signal, direction)
SCENARIOS = {
    "request_drop": ("request_rate", "below"),
    "token_length_drop": ("output_tokens", "below"),
    "throughput_drop": ("throughput", "below"),
    "step_time_rise": ("step_time", "above"),
}


def _window(rng: np.random.Generator, step: int, eval_start: int, n: int) -> tuple[int, int]:
    """Random anomaly window of 60 to 240 minutes inside the evaluation period, away from its
    first 12 hours (the model has just finished warming up) and from its end."""
    length = int(rng.integers(60, 241)) * 60 // step
    lo = eval_start + 12 * 3600 // step
    hi = n - length - 3600 // step
    start = int(rng.integers(lo, hi))
    return start, start + length


def _factor(n: int, start: int, end: int, floor: float, ramp: int) -> np.ndarray:
    """Multiplier that is 1 outside the window and `floor` inside, with an optional ramp."""
    f = np.ones(n)
    f[start:end] = floor
    for k in range(min(ramp, end - start)):
        f[start + k] = 1 + (floor - 1) * (k + 1) / (ramp + 1)
    return f


def build_case(
    scenario: str,
    variant: int,
    seed: int,
    train_days: float,
    eval_days: float,
    step: int,
) -> Case:
    signal, direction = SCENARIOS[scenario]
    ts = timestamps(train_days + eval_days, step)
    n = len(ts)
    eval_start = int(train_days * DAY / step)
    rng = np.random.default_rng(seed * 7 + variant)
    start, end = _window(rng, step, eval_start, n)
    ramp = 0 if variant % 2 == 0 else max(1, 5 * 60 // step)  # instant or five minute ramp

    if scenario == "request_drop":
        base = generate("request_rate", ts, seed)
        # even variants: near zero, odd variants: partial drop to 10-50 percent
        floor = rng.uniform(0.0, 0.05) if (variant // 2) % 2 == 0 else rng.uniform(0.10, 0.50)
        values = base * _factor(n, start, end, floor, ramp)
    elif scenario == "token_length_drop":
        base = generate("output_tokens", ts, seed)
        values = base.copy()
        # every response ends after one to three tokens
        values[start:end] = rng.uniform(1.0, 3.0) * (1 + rng.normal(0, 0.05, end - start))
    elif scenario == "throughput_drop":
        base = generate("throughput", ts, seed)
        values = base * _factor(n, start, end, rng.uniform(0.45, 0.55), ramp)
    elif scenario == "step_time_rise":
        thr = generate("throughput", ts, seed)
        thr = thr * _factor(n, start, end, rng.uniform(0.45, 0.55), ramp)
        values = derive_step_time(thr, seed)
        base = derive_step_time(generate("throughput", ts, seed), seed)
    else:
        raise ValueError(scenario)

    values = values.copy()
    values[missing_mask(n, seed)] = np.nan
    return Case(
        scenario,
        signal,
        direction,
        variant,
        seed,
        ts,
        values,
        ((start, end),),
        eval_start,
        base[:eval_start].copy(),
    )


def build_normal_case(
    signal: str,
    direction: str,
    variant: int,
    seed: int,
    train_days: float,
    eval_days: float,
    step: int,
) -> Case:
    ts = timestamps(train_days + eval_days, step)
    values = generate(signal, ts, seed).copy()
    base = values.copy()
    values[missing_mask(len(ts), seed)] = np.nan
    eval_start = int(train_days * DAY / step)
    return Case(
        "normal",
        signal,
        direction,
        variant,
        seed,
        ts,
        values,
        (),
        eval_start,
        base[:eval_start].copy(),
    )
