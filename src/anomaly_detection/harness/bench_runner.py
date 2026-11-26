"""Score SMD and PSM dimension by dimension with the models of this repository.

One job is one dimension of one machine: the model sees the warm-up (train rows) and then the
test rows, one after the other, and the scores of the test rows are kept. The multivariate
part (maximum over dimensions, metrics) is in `bench_metrics.py`.

Parameter rule: the NAB rule of `nab_runner.model_params` with a sampling interval of 60
seconds, except that `sample_size` of the forest models is the model default 1024, because
the warm-up is far longer than the forest (NAB ties it to the probationary length). Nothing
depends on values, labels or results, and nothing is tuned per machine or dimension.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..models import create_model
from ..pipeline import SeriesPipeline
from .bench_data import STEP_SECONDS
from .nab_runner import model_params

STREAM_MODELS = ("seasonal", "rcf", "awsrcf-nab", "awsrcf", "subpca")
BENCH_MODELS = ("fixed", "fixed-tight", *STREAM_MODELS, "tspulse", "toto-4m")
FIXED_MARGIN = {"fixed": 0.5, "fixed-tight": 0.0}
# train rows fed before the first test row, None meaning all of them. The models with a finite
# memory (forest sample, context window, score history) do not need more than this.
WARMUP_TAIL: dict[str, int | None] = {
    "rcf": 3000,  # forest of 1024 and a 1440 row normalisation window
    "tspulse": 2600,  # 512 row context plus the 2000 error history of the score scaling
    "toto-4m": 512,  # the context window
}
TOTO_CONTEXT = 512
TOTO_MIN_CONTEXT = 128


def fixed_scores(train: np.ndarray, test: np.ndarray, margin: float) -> np.ndarray:
    """1 outside [min - margin * range, max + margin * range] of the train rows, else 0.
    Same rule as the NAB runs (margin 0.5). NaN test rows stay NaN."""
    seen = train[np.isfinite(train)]
    lo, hi = float(seen.min()), float(seen.max())
    pad = margin * (hi - lo)
    out = ((test < lo - pad) | (test > hi + pad)).astype(np.float32)
    out[~np.isfinite(test)] = np.nan
    return out


def stream_params(name: str) -> tuple[str, dict[str, Any]]:
    """(registered kind, params) of a streaming model for 60 second data."""
    params = model_params(name, STEP_SECONDS, 10**6)
    if name in ("rcf", "awsrcf"):
        params["sample_size"] = 1024
    kind = "awsrcf" if name.startswith("awsrcf") else name
    return kind, params


def stream_scores(
    name: str,
    train: np.ndarray,
    test: np.ndarray,
    train_ts: np.ndarray,
    test_ts: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Feed the warm-up and the test rows through the model and pipeline, as the sidecar
    would. Missing values are skipped, never fed."""
    kind, params = stream_params(name)
    model = create_model(kind, params, STEP_SECONDS, seed=seed)
    pipe = SeriesPipeline(model, direction="both")
    for t, v in zip(train_ts, train, strict=True):
        if math.isfinite(v):
            pipe.update(float(t), float(v))
    out = np.full(len(test), np.nan, dtype=np.float32)
    for i, (t, v) in enumerate(zip(test_ts, test, strict=True)):
        if math.isfinite(v):
            res = pipe.update(float(t), float(v))
            if res is not None:
                out[i] = res.score
    close = getattr(model, "close", None)
    if close is not None:
        close()
    return out


def _batched_scores(name: str, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """tspulse and toto: batched forward passes over warm-up plus test, same numbers as the
    streaming path (tests in test_tspulse.py and test_toto.py). Rows with a missing value are
    dropped from the stream, as the pipeline does."""
    x = np.concatenate([train, test])
    ok = np.isfinite(x)
    vals = x[ok]
    if name == "tspulse":
        from ..models.tspulse import TSPulseModel

        model = TSPulseModel(
            model_path=os.environ.get("TSPULSE_MODEL_DIR")
            or "ibm-granite/granite-timeseries-tspulse-r1",
            threads=int(os.environ.get("TSPULSE_THREADS", "1")),
        )
        outs, _ = model.score_offline(vals, batch_size=64)
        sc = np.array([np.nan if o is None else o.score for o in outs])
    else:
        from ..models.toto import get_engine, score_batch

        engine = get_engine(
            Path(os.environ["TOTO_MODEL_DIR"]), int(os.environ.get("TOTO_THREADS", "1"))
        )
        sc = score_batch(engine, vals, TOTO_CONTEXT, TOTO_MIN_CONTEXT, batch_size=256)["score"]
    out = np.full(len(x), np.nan, dtype=np.float32)
    out[ok] = sc
    return out[len(train) :]


def score_dim(
    name: str,
    train: np.ndarray,
    test: np.ndarray,
    train_ts: np.ndarray,
    test_ts: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Scores of the test rows of one dimension (float32, NaN where the model gave none)."""
    if name in FIXED_MARGIN:
        return fixed_scores(train, test, FIXED_MARGIN[name])
    tail = WARMUP_TAIL.get(name)
    if tail is not None:
        train, train_ts = train[-tail:], train_ts[-tail:]
    if name in ("tspulse", "toto-4m"):
        return _batched_scores(name, train, test)
    return stream_scores(name, train, test, train_ts, test_ts, seed)


Job = tuple[str, str, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _run(job: Job) -> tuple[str, str, int, np.ndarray, float]:
    name, machine, dim, train, test, tr_ts, te_ts = job
    t0 = time.perf_counter()
    sc = score_dim(name, train, test, tr_ts, te_ts)
    return name, machine, dim, sc, time.perf_counter() - t0


def score_path(out_dir: Path, dataset: str, model: str, machine: str, dim: int) -> Path:
    return out_dir / dataset / model / machine / f"{dim:02d}.npy"


def run_jobs(
    jobs: Iterable[Job], out_dir: Path, dataset: str, workers: int, log: Any = print
) -> dict[str, float]:
    """Run the jobs that have no result file yet. Returns seconds of compute per model."""
    todo = [j for j in jobs if not score_path(out_dir, dataset, j[0], j[1], j[2]).exists()]
    spent: dict[str, float] = {}
    if not todo:
        return spent
    # one BLAS thread per worker: SubPCA's SVD otherwise runs 8 workers x all cores and is
    # 40 times slower on a busy machine
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    ctx = mp.get_context("spawn")
    done = 0
    with ctx.Pool(workers, maxtasksperchild=64) as pool:
        for name, machine, dim, sc, sec in pool.imap_unordered(_run, todo, chunksize=1):
            f = score_path(out_dir, dataset, name, machine, dim)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, sc)
            spent[name] = spent.get(name, 0.0) + sec
            done += 1
            if done % 25 == 0 or done == len(todo):
                log(f"{dataset} {done}/{len(todo)} done", flush=True)
    return spent


def stratified_subset(
    names: Sequence[str], rates: Sequence[float], k: int, group_of: Any = None
) -> list[str]:
    """k machines chosen without looking at any model result: allocate k over the groups in
    proportion to their size (largest remainder), then take evenly spaced ranks of the
    anomaly rate inside each group. `group_of(name)` gives the group (default: the part before
    the second dash, so machine-1-x is group machine-1)."""
    group_of = group_of or (lambda n: n.rsplit("-", 1)[0])
    groups: dict[str, list[tuple[float, str]]] = {}
    for n, r in zip(names, rates, strict=True):
        groups.setdefault(group_of(n), []).append((r, n))
    total = len(names)
    quota = {g: k * len(v) / total for g, v in groups.items()}
    alloc = {g: int(q) for g, q in quota.items()}
    for g in sorted(quota, key=lambda g: quota[g] - alloc[g], reverse=True)[
        : k - sum(alloc.values())
    ]:
        alloc[g] += 1
    chosen: list[str] = []
    for g, members in groups.items():
        members.sort()
        m = alloc[g]
        for i in range(m):
            chosen.append(members[int((i + 0.5) * len(members) / m)][1])
    return sorted(chosen)
