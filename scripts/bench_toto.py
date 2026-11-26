"""CPU cost of the Toto 2.0 model: load time, one-step latency of one series, one step of many
series in one batch, and memory. One process per model size so memory numbers are clean.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python scripts/bench_toto.py MODEL_DIR \
        --threads 4 --context 512 --series 500

Prints one JSON object. Synthetic series (daily wave plus noise), the cost does not depend on
the values.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * resource.getpagesize() / 2**20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--series", type=int, default=500)
    ap.add_argument("--single-runs", type=int, default=200)
    ap.add_argument("--batch-runs", type=int, default=5)
    args = ap.parse_args()

    import numpy as np

    t0 = time.perf_counter()
    import torch

    from anomaly_detection.models.toto import TotoEngine, TotoModel

    t_import = time.perf_counter() - t0
    rss_import = rss_mb()
    t0 = time.perf_counter()
    engine = TotoEngine(args.model_dir, args.threads)
    t_load = time.perf_counter() - t0
    rss_loaded = rss_mb()

    rng = np.random.default_rng(0)
    t = np.arange(args.context + 400)
    base = 100 + 20 * np.sin(2 * np.pi * t / 288)[None, :]
    data = (base + rng.normal(0, 2, (args.series, len(t)))).astype(np.float32)

    # one series, one step: the forward pass, then the whole update() as the sidecar runs it
    w1 = data[:1, : args.context]
    for _ in range(5):
        engine.forecast(w1)
    lat = []
    for _ in range(args.single_runs):
        t0 = time.perf_counter()
        engine.forecast(w1)
        lat.append((time.perf_counter() - t0) * 1e3)
    lat.sort()
    model = TotoModel(engine=engine, context_length=args.context, min_context=128)
    for i in range(args.context):
        model.update(float(i), float(data[0, i]))
    upd = []
    for i in range(args.context, args.context + 100):
        t0 = time.perf_counter()
        model.update(float(i), float(data[0, i]))
        upd.append((time.perf_counter() - t0) * 1e3)
    upd.sort()

    # many series, one step each, in one batch
    wn = data[:, : args.context]
    engine.forecast(wn[:8])
    times = []
    for _ in range(args.batch_runs):
        t0 = time.perf_counter()
        engine.forecast(wn)
        times.append(time.perf_counter() - t0)
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    def pct(a: list[float], p: float) -> float:
        return a[min(len(a) - 1, int(p * len(a)))]

    print(
        json.dumps(
            {
                "model_dir": args.model_dir,
                "torch": torch.__version__,
                "threads": args.threads,
                "context": args.context,
                "series": args.series,
                "import_s": round(t_import, 2),
                "load_s": round(t_load, 3),
                "rss_after_import_mb": round(rss_import),
                "rss_after_load_mb": round(rss_loaded),
                "rss_peak_mb": round(rss_peak),
                "single_forward_ms": {
                    "median": round(statistics.median(lat), 2),
                    "p95": round(pct(lat, 0.95), 2),
                    "p99": round(pct(lat, 0.99), 2),
                },
                "single_update_ms": {
                    "median": round(statistics.median(upd), 2),
                    "p95": round(pct(upd, 0.95), 2),
                },
                "batch_step_s": {
                    "median": round(statistics.median(times), 2),
                    "min": round(min(times), 2),
                    "max": round(max(times), 2),
                },
            }
        )
    )


if __name__ == "__main__":
    main()
