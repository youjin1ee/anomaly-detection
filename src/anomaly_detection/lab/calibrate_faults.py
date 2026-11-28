"""Find the competitor duty cycle that gives a wanted decode speed ratio.

Run against a live engine that already serves a replayed trace (so the load is the one of the
time of day under test). For every duty it starts the competitor (`lab.contender`) for
`--during` seconds and compares the decode speed (tokens per second between tokens, from the
engine's inter token latency counters over a 60 s window, the `decode_speed` signal of
`lab.signals`) during the fault with the stretch before it. The same duty slows decoding more
when more requests are in flight, so run it once per time of day band.

    python -m anomaly_detection.lab.calibrate_faults --duties 0.15,0.3,0.5 --out peak.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

WINDOW = 60.0
COUNT = "vllm:inter_token_latency_seconds_count"
TOTAL = "vllm:inter_token_latency_seconds_sum"


def read_counter(text: str, name: str) -> float:
    """Sum of a counter over its label sets in Prometheus text."""
    return sum(
        float(m.group(1))
        for m in re.finditer(rf"^{re.escape(name)}(?:\{{[^}}]*\}})? (\S+)$", text, re.M)
    )


def decode_speed(samples: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    """(time, value) of count delta over sum delta across WINDOW seconds, from (time, count,
    sum) samples."""
    out = []
    for i, (t, c, s) in enumerate(samples):
        j = next((k for k in range(i, -1, -1) if t - samples[k][0] >= WINDOW), None)
        if j is not None and s - samples[j][2] > 0:
            out.append((t, (c - samples[j][1]) / (s - samples[j][2])))
    return out


def effect(series: list[tuple[float, float]], t0: float, t1: float, before: float, settle: float):
    """Median during (t0+settle to t1) over median in the `before` seconds ahead of t0."""
    inside = [v for t, v in series if t0 + settle <= t <= t1]
    ahead = [v for t, v in series if t0 - before <= t < t0]
    if not inside or not ahead:
        return float("nan")
    return float(np.median(inside) / np.median(ahead))


def steady_trace(
    trace: Path, band: tuple[float, float], day_seconds: float, minutes: float, seed: int = 1
) -> dict[str, np.ndarray]:
    """A trace that holds one time of day band's average load: Poisson arrivals at the mean
    rate of the band over all days of `trace`, lengths drawn from the band's requests. The
    real trace changes by a factor of 3 or more within a calibration, this one does not."""
    z = np.load(trace)
    frac = (z["t"] / day_seconds) % 1.0
    sel = (frac >= band[0]) & (frac < band[1])
    days = int(z["t"].max() // day_seconds) + 1
    rate = float(sel.sum()) / (days * (band[1] - band[0]) * day_seconds)
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate * minutes * 60)
    t = np.sort(rng.uniform(0, minutes * 60, n))
    pick = rng.choice(np.flatnonzero(sel), n)
    return {"t": t, "ctx": z["ctx"][pick], "gen": z["gen"][pick], "rate": np.array(rate)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m anomaly_detection.lab.calibrate_faults")
    p.add_argument("--target", default="http://127.0.0.1:8000")
    p.add_argument("--duties", default="0.15,0.3,0.5")
    p.add_argument("--before", type=float, default=90)
    p.add_argument("--during", type=float, default=150)
    p.add_argument("--after", type=float, default=90)
    p.add_argument("--settle", type=float, default=60)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--contender-python", default=sys.executable)
    p.add_argument("--out", type=Path)
    p.add_argument("--make-steady", type=Path, help="trace npz: write a steady trace and exit")
    p.add_argument("--band", default="0.5,0.9", help="with --make-steady: day fractions lo,hi")
    p.add_argument("--day-seconds", type=float, default=3600)
    p.add_argument("--minutes", type=float, default=30)
    a = p.parse_args(argv)
    if a.make_steady:
        lo, hi = (float(x) for x in a.band.split(","))
        tr = steady_trace(a.make_steady, (lo, hi), a.day_seconds, a.minutes)
        np.savez(a.out, **tr)
        print(f"steady trace: {len(tr['t'])} requests, {float(tr['rate']):.2f} per second")
        return 0

    samples: list[tuple[float, float, float]] = []

    def sample_for(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            with urllib.request.urlopen(f"{a.target}/metrics", timeout=10) as r:
                text = r.read().decode()
            samples.append((time.time(), read_counter(text, COUNT), read_counter(text, TOTAL)))
            time.sleep(5)

    results: dict[str, list[float]] = {}
    sample_for(a.before)
    for rep in range(a.repeats):
        for duty in [float(x) for x in a.duties.split(",")]:
            t0 = time.time()
            cmd = [a.contender_python, "-m", "anomaly_detection.lab.contender", "--duty", str(duty)]
            proc = subprocess.Popen(cmd)
            try:
                sample_for(a.during)
            finally:
                proc.terminate()  # the child we started, nothing else
                proc.wait(timeout=15)
            t1 = time.time()
            series = decode_speed(samples)
            r = effect(series, t0, t1, a.before, a.settle)
            results.setdefault(str(duty), []).append(round(r, 3))
            print(f"duty {duty} repeat {rep + 1}: decode speed ratio {r:.3f}", flush=True)
            sample_for(a.after)
    doc = {"duties": results, "during": a.during, "settle": a.settle}
    print(json.dumps(doc))
    if a.out:
        a.out.write_text(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
