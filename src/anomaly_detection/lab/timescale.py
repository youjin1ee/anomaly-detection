"""A day compressed into a few minutes.

The request rate follows the same daily shape as the synthetic request signal of the harness
(first harmonic with its peak at 15:00 and a smaller second harmonic), evaluated at the
compressed time of day.
"""

from __future__ import annotations

import math

# Same shape as harness.synth SIGNALS["request_rate"], without the weekend factor and trend.
DAILY_AMP = 0.45
DAILY_AMP2 = 0.12
PEAK_HOUR = 15.0


def day_fraction(t: float, epoch: float, day_seconds: float) -> float:
    """Fraction of the compressed day at wall clock time t, in [0, 1)."""
    return ((t - epoch) / day_seconds) % 1.0


def daily_factor(fraction: float) -> float:
    """Multiplier around 1.0 for a time of day given as a fraction of the day."""
    phase = 2 * math.pi * (fraction - PEAK_HOUR / 24.0)
    return 1 + DAILY_AMP * math.cos(phase) + DAILY_AMP2 * math.cos(2 * phase + 0.8)


def target_rate(t: float, epoch: float, day_seconds: float, base_rate: float) -> float:
    """Requests per second the generator aims for at wall clock time t, without noise."""
    return max(0.0, base_rate * daily_factor(day_fraction(t, epoch, day_seconds)))
