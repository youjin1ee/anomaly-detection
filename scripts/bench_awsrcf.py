"""Operating cost of the awsrcf models: JVM start, one-step latency, many series.

Run it with ANOMALY_AWSRCF_HOME and ANOMALY_AWSRCF_JAVA set (see models/awsrcf.py). The
series are a sine with noise at one sample per minute. Prints one JSON object.

    uv run python scripts/bench_awsrcf.py --series 500 --rounds 30
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time

import numpy as np

from anomaly_detection.models import awsrcf as mod
from anomaly_detection.models import create_model


def rss_mb(pid: int) -> float:
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def value(rng: random.Random, i: int, k: int) -> float:
    return 100 + 20 * math.sin(i / 60 + k) + rng.gauss(0, 2)


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q))


def reset() -> None:
    """Close the shared bridge so the next model starts a fresh JVM."""
    if mod._bridge is not None:
        mod._bridge.close()
    mod._bridge = None


def jvm_start(repeats: int) -> list[float]:
    out = []
    for _ in range(repeats):
        reset()
        t0 = time.perf_counter()
        b = mod.get_bridge()
        b.call("MEM")  # first round trip, the JVM answers
        out.append(time.perf_counter() - t0)
        b.close()
    return out


def one_series(kind: str, params: dict, steps: int, warm: int) -> dict:
    rng = random.Random(1)
    m = create_model(kind, params, 60.0)
    for i in range(warm):
        m.update(i * 60.0, value(rng, i, 0))
    lat = []
    for i in range(warm, warm + steps):
        v = value(rng, i, 0)
        t0 = time.perf_counter()
        m.update(i * 60.0, v)
        lat.append((time.perf_counter() - t0) * 1e3)
    return {
        "p50_ms": pct(lat, 50),
        "p99_ms": pct(lat, 99),
        "max_ms": max(lat),
        "mean_ms": statistics.fmean(lat),
    }


def many_series(kind: str, params: dict, n: int, warm: int, rounds: int) -> dict:
    rng = random.Random(2)
    t0 = time.perf_counter()
    models = [create_model(kind, params, 60.0, seed=k) for k in range(n)]
    create_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(warm):
        for k, m in enumerate(models):
            m.update(i * 60.0, value(rng, i, k))
    warm_s = time.perf_counter() - t0
    rt = []
    for i in range(warm, warm + rounds):
        t0 = time.perf_counter()
        for k, m in enumerate(models):
            m.update(i * 60.0, value(rng, i, k))
        rt.append(time.perf_counter() - t0)
    out = {
        "series": n,
        "create_s": create_s,
        "warm_samples": warm,
        "warm_s": warm_s,
        "round_p50_s": pct(rt, 50),
        "round_max_s": max(rt),
        "round_mean_s": statistics.fmean(rt),
    }
    if kind == "awsrcf":
        b = mod.get_bridge()
        out["jvm_rss_mb"] = rss_mb(b._proc.pid)
        out["jvm_heap_used_mb"] = int(b.call("MEM").split()[1]) / 1048576.0
    else:
        import os

        out["python_rss_mb"] = rss_mb(os.getpid())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--series", type=int, default=500)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--warm", type=int, default=400)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--modes", default="rcf,trcf,caster")
    p.add_argument("--sample-size", type=int, default=256)
    p.add_argument("--rrcf-series", type=int, default=0, help="also run rrcf over this many")
    a = p.parse_args()
    res: dict = {"jvm_start_s": jvm_start(5)}
    for mode in a.modes.split(","):
        params = {"mode": mode, "sample_size": a.sample_size}
        reset()
        res[f"{mode}_single"] = one_series("awsrcf", params, a.steps, a.warm)
        reset()
        res[f"{mode}_many"] = many_series("awsrcf", params, a.series, a.warm, a.rounds)
    if a.rrcf_series:
        res["rrcf_single"] = one_series("rcf", {"sample_size": a.sample_size}, 600, a.warm + 200)
        res["rrcf_many"] = many_series(
            "rcf", {"sample_size": a.sample_size}, a.rrcf_series, a.warm + 200, 3
        )
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
