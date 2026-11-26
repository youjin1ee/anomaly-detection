"""`anomaly-detection-nab`: run the models on the Numenta Anomaly Benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .metrics import AlertRule
from .nab_data import NAB_ARCHIVE_URL, fetch_nab, load_corpus
from .nab_report import render, render_seeds
from .nab_runner import NAB_MODELS, hold_for, run_models
from .nab_toto import TOTO_SIZES, batch_series, checkpoint_dir, toto_name
from .nab_verify import render as render_verify
from .nab_verify import verify


def _scores_file(scores_dir: Path, model: str) -> Path:
    return scores_dir / f"{model}.npz"


def cmd_fetch(args: argparse.Namespace) -> int:
    print(f"downloading {NAB_ARCHIVE_URL}")
    fetch_nab(args.dest)
    corpus = load_corpus(args.dest)
    print(f"{len(corpus)} series, {sum(len(s.windows) for s in corpus)} windows in {args.dest}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.nab_dir)
    names = [m for m in args.models.split(",") if m]
    for m in names:
        if m not in NAB_MODELS:
            raise SystemExit(f"unknown model {m!r}; known: {', '.join(NAB_MODELS)}")
    args.scores_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = run_models(names, corpus, args.workers, args.seed, args.mem_limit_gb)
    for m, res in results.items():
        arrays = {p.replace("/", "__"): s for p, s in res.scores.items()}
        np.savez_compressed(_scores_file(args.scores_dir, m), **arrays)
        us = np.mean(list(res.us_per_record.values()))
        (args.scores_dir / f"{m}.meta.json").write_text(
            json.dumps({"seed": args.seed, "us_per_record": float(us)})
        )
    print(f"scored {len(corpus)} series with {', '.join(names)} in {time.time() - t0:.0f}s")
    return 0


def cmd_run_toto(args: argparse.Namespace) -> int:
    """Toto 2.0 on every file, many steps per forward pass. Scores go to the same files `run`
    writes (`toto-<size>.npz`), so `report --models seasonal,rcf,toto-4m` reads them. One file
    per step is kept in `<scores-dir>/toto-<size>.parts`, and a rerun skips finished files."""
    from ..models.toto import get_engine

    corpus = load_corpus(args.nab_dir)
    if args.limit_files:
        corpus = corpus[: args.limit_files]
    args.scores_dir.mkdir(parents=True, exist_ok=True)
    for size in [x for x in args.sizes.split(",") if x]:
        if size not in TOTO_SIZES:
            raise SystemExit(f"unknown size {size!r}; known: {', '.join(TOTO_SIZES)}")
        name = toto_name(size)
        engine = get_engine(checkpoint_dir(args.toto_dir, size), args.threads)
        parts = args.scores_dir / f"{name}.parts"
        parts.mkdir(exist_ok=True)
        t0 = time.time()
        for k, s in enumerate(corpus):
            f = parts / (s.path.replace("/", "__") + ".npz")
            if f.exists():
                continue
            scores, us = batch_series(engine, s, args.batch_size)
            np.savez(f, scores=scores, us=us)
            print(f"{name} {k + 1}/{len(corpus)} {s.path} {us:.0f} us/record", flush=True)
        scores_by_path, us_all = {}, []
        for s in corpus:
            with np.load(parts / (s.path.replace("/", "__") + ".npz")) as z:
                scores_by_path[s.path.replace("/", "__")] = z["scores"]
                us_all.append(float(z["us"]))
        np.savez_compressed(_scores_file(args.scores_dir, name), **scores_by_path)
        (args.scores_dir / f"{name}.meta.json").write_text(
            json.dumps({"us_per_record": float(np.mean(us_all)), "threads": args.threads})
        )
        print(f"{name}: {len(corpus)} series in {time.time() - t0:.0f}s", flush=True)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.nab_dir)
    names = [m for m in args.models.split(",") if m]
    all_scores = {}
    cost = {}
    for m in names:
        with np.load(_scores_file(args.scores_dir, m)) as z:
            all_scores[m] = {s.path: z[s.path.replace("/", "__")] for s in corpus}
        meta = json.loads((args.scores_dir / f"{m}.meta.json").read_text())
        cost[m] = meta["us_per_record"]
    rule = AlertRule(threshold=args.threshold, for_points=args.for_points)
    # The same models behind the alert rule's `for` clause, derived from the kept scores.
    for m in names:
        all_scores[f"{m}+for{args.for_points}"] = {
            p: hold_for(s, args.for_points) for p, s in all_scores[m].items()
        }
    text = render(corpus, all_scores, names, rule, cost, args.for_points)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


def cmd_seeds(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.nab_dir)
    names = [m for m in args.models.split(",") if m]
    runs = []
    for d in args.scores_dirs:
        run = {}
        for m in names:
            with np.load(_scores_file(d, m)) as z:
                run[m] = {s.path: z[s.path.replace("/", "__")] for s in corpus}
        runs.append(run)
    text = render_seeds(corpus, runs, names, args.for_points)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.nab_dir)
    checks = verify(corpus, args.nab_dir, [d for d in args.detectors.split(",") if d])
    text = render_verify(checks)
    print(text)
    worst = max(abs(c.score_at_official_threshold - c.official_score) for c in checks)
    print(f"largest difference at the official thresholds: {worst:.2e}")
    return 0 if worst < args.tolerance else 1


def main(argv: list[str] | None = None) -> int:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")  # one thread per worker process
    p = argparse.ArgumentParser(prog="anomaly-detection-nab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download data/ and labels/ of the pinned NAB revision")
    f.add_argument("--dest", type=Path, required=True)
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="score every file with the models and keep the scores")
    r.add_argument("--nab-dir", type=Path, required=True, help="directory with data/ and labels/")
    r.add_argument("--models", default=",".join(NAB_MODELS))
    r.add_argument("--scores-dir", type=Path, required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--mem-limit-gb", type=float, default=4.0, help="address space per worker")
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("run-toto", help="score every file with Toto 2.0 (needs the toto extra)")
    t.add_argument("--nab-dir", type=Path, required=True)
    t.add_argument("--toto-dir", type=Path, required=True, help="has one directory per size")
    t.add_argument("--sizes", default=",".join(TOTO_SIZES))
    t.add_argument("--scores-dir", type=Path, required=True)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--limit-files", type=int, default=0, help="first N files, for a smoke run")
    t.set_defaults(func=cmd_run_toto)

    q = sub.add_parser("report", help="NAB scores and event metrics from kept scores")
    q.add_argument("--nab-dir", type=Path, required=True)
    q.add_argument("--models", default=",".join(NAB_MODELS))
    q.add_argument("--scores-dir", type=Path, required=True)
    q.add_argument("--threshold", type=float, default=0.5)
    q.add_argument("--for-points", type=int, default=3)
    q.add_argument("--out", type=Path)
    q.set_defaults(func=cmd_report)

    s = sub.add_parser("seeds", help="NAB scores of randomised models over several seeds")
    s.add_argument("--nab-dir", type=Path, required=True)
    s.add_argument("--models", default="rcf,rcf-time")
    s.add_argument("--scores-dirs", type=Path, nargs="+", required=True, help="one per seed")
    s.add_argument("--for-points", type=int, default=3)
    s.add_argument("--out", type=Path)
    s.set_defaults(func=cmd_seeds)

    v = sub.add_parser("verify", help="compare the scorer with the official NAB results")
    v.add_argument("--nab-dir", type=Path, required=True, help="a NAB checkout (with results/)")
    v.add_argument("--detectors", default="null,random,numenta,randomCutForest")
    v.add_argument("--tolerance", type=float, default=1e-6)
    v.set_defaults(func=cmd_verify)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
