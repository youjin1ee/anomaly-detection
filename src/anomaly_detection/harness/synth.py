"""Synthetic metric generator.

Series have a daily shape, a weekend factor, a slow trend, autocorrelated noise and a few
missing samples. They are deliberately simple: the point is a reproducible yardstick for the
models, not a faithful copy of any production system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 2024-01-01 00:00:00 UTC, a Monday, so weekday arithmetic starts at day 0 = Monday.
T0 = 1_704_067_200
DAY = 86400


@dataclass(frozen=True)
class Signal:
    name: str
    base: float
    daily_amp: float  # relative amplitude of the first daily harmonic
    daily_amp2: float  # relative amplitude of the second harmonic
    weekend: float  # multiplier on Saturday and Sunday
    trend_per_week: float  # relative growth per week
    noise: float  # relative standard deviation of the noise
    ar: float  # autocorrelation of the noise (0 = white)
    peak_hour: float  # hour of day of the daily maximum


SIGNALS = {
    "request_rate": Signal("request_rate", 120.0, 0.45, 0.12, 0.65, 0.01, 0.06, 0.6, 15.0),
    "output_tokens": Signal("output_tokens", 280.0, 0.05, 0.02, 1.0, 0.0, 0.03, 0.5, 14.0),
    "throughput": Signal("throughput", 12000.0, 0.02, 0.0, 1.0, 0.0, 0.015, 0.3, 12.0),
}
# Training step time is derived from throughput: seconds per step = tokens per step / throughput.
STEP_TOKENS = 96000.0


def timestamps(days: float, step: int) -> np.ndarray:
    return T0 + np.arange(int(days * DAY / step)) * step


def _ar_noise(rng: np.random.Generator, n: int, sigma: float, rho: float) -> np.ndarray:
    white = rng.normal(0.0, 1.0, n)
    if rho == 0:
        return sigma * white
    out = np.empty(n)
    out[0] = white[0]
    scale = np.sqrt(1 - rho**2)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + scale * white[i]
    return sigma * out


def shape(name: str, ts: np.ndarray) -> np.ndarray:
    """Noise free expected value of a base signal at the given timestamps."""
    sig = SIGNALS[name]
    tod = (ts % DAY) / DAY
    phase = 2 * np.pi * (tod - sig.peak_hour / 24.0)
    daily = 1 + sig.daily_amp * np.cos(phase) + sig.daily_amp2 * np.cos(2 * phase + 0.8)
    weekday = ((ts - T0) // DAY) % 7
    week = np.where(weekday >= 5, sig.weekend, 1.0)
    trend = 1 + sig.trend_per_week * (ts - T0) / (7 * DAY)
    return sig.base * daily * week * trend


def generate(name: str, ts: np.ndarray, seed: int) -> np.ndarray:
    """Normal behaviour of one signal. `step_time` is derived from `throughput`."""
    if name == "step_time":
        thr = generate("throughput", ts, seed)
        return derive_step_time(thr, seed)
    sig = SIGNALS[name]
    rng = np.random.default_rng(seed)
    noise = _ar_noise(rng, len(ts), sig.noise, sig.ar)
    values = shape(name, ts) * (1 + noise)
    if name == "throughput":
        # Checkpoint dips: two samples at -30 percent once an hour. They are normal.
        step = int(ts[1] - ts[0])
        period = max(1, 3600 // step)
        offset = int(rng.integers(0, period))
        for k in range(offset, len(ts), period):
            values[k : k + 2] *= 0.7
    return values


def derive_step_time(throughput: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 7919)
    return STEP_TOKENS / throughput * (1 + rng.normal(0, 0.01, len(throughput)))


def missing_mask(n: int, seed: int, rate: float = 0.002) -> np.ndarray:
    """True where a sample is missing. Missing samples are skipped, never zero-filled."""
    return np.random.default_rng(seed + 104729).random(n) < rate
