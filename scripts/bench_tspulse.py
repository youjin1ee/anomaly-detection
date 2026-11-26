"""CPU cost of the TSPulse model: load time, one series per step, many series per step.

    TSPULSE_MODEL_DIR=/path/to/model python scripts/bench_tspulse.py --series 500 --threads 3

Prints one JSON object. Run it pinned to the cores you want to measure (taskset) and with
CUDA_VISIBLE_DEVICES set to an empty string.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import time

import numpy as np


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("TSPULSE_MODEL_DIR"))
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--series", type=int, default=500)
    ap.add_argument("--single-steps", type=int, default=200)
    ap.add_argument("--batch-steps", type=int, default=5)
    a = ap.parse_args()

    base_rss = rss_mb()
    t0 = time.perf_counter()
    import torch

    t_import = time.perf_counter() - t0
    from anomaly_detection.models.tspulse import TSPulseModel, get_engine, update_many

    t0 = time.perf_counter()
    engine = get_engine(a.model, a.threads)
    t_load = time.perf_counter() - t0
    rss_loaded = rss_mb()

    rng = np.random.default_rng(0)

    def stream(n: int) -> np.ndarray:
        return 100 + 10 * np.sin(np.arange(n) * 2 * np.pi / 288) + rng.normal(0, 1, n)

    # one series: a full window and a warm normalisation history, then timed steps
    warm = 512 + 80
    m = TSPulseModel(engine=engine)
    v = stream(warm + a.single_steps)
    for i in range(warm):
        m.update(float(i), float(v[i]))
    lat = []
    for i in range(warm, warm + a.single_steps):
        t = time.perf_counter()
        m.update(float(i), float(v[i]))
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()

    # many series, one forward pass per step
    models = [TSPulseModel(engine=engine) for _ in range(a.series)]
    data = [stream(warm + a.batch_steps) for _ in range(a.series)]
    for i in range(warm):
        # the window fills without inference, the last steps warm the history
        update_many(models, [float(d[i]) for d in data])
    rss_before = rss_mb()
    steps = []
    for i in range(warm, warm + a.batch_steps):
        t = time.perf_counter()
        update_many(models, [float(d[i]) for d in data])
        steps.append(time.perf_counter() - t)
    out = {
        "threads": a.threads,
        "torch": torch.__version__,
        "import_torch_s": round(t_import, 2),
        "load_model_s": round(t_load, 3),
        "rss_mb_before": round(base_rss, 1),
        "rss_mb_after_load": round(rss_loaded, 1),
        "single_step_ms": {
            "median": round(statistics.median(lat), 2),
            "p95": round(lat[int(0.95 * len(lat)) - 1], 2),
            "max": round(lat[-1], 2),
        },
        "series": a.series,
        "batch_step_s": {
            "median": round(statistics.median(steps), 2),
            "max": round(max(steps), 2),
            "all": [round(s, 2) for s in steps],
        },
        "rss_mb_with_series": round(rss_before, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
