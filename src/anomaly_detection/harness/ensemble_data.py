"""Inputs of the ensemble evaluation: units with cached model scores.

Nothing is recomputed here except where a cache is missing (`score_missing`, used for the
recorded production metrics). A unit is one labelled or unlabelled thing alerts belong to: a
NAB file, an SMD machine (all its dimensions), a BOOM series, a recorded series, a lab signal.
The calibration window is the stretch right after the warm-up, the evaluation window follows it.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..ensemble.rules import SeriesInfo
from .bench_data import label_events, load_psm, load_smd
from .bench_runner import score_path


@dataclass
class UnitSeries:
    values: np.ndarray
    info: SeriesInfo
    raws: dict[str, np.ndarray]  # model -> raw score (NaN where none)


@dataclass
class Unit:
    dataset: str
    name: str
    step: float  # seconds per sample (delays are reported in minutes of this)
    series: list[UnitSeries]
    cal: tuple[int, int]
    events: list[tuple[int, int]] | None = None  # None: no labels
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.series[0].values)

    @property
    def eval_start(self) -> int:
        return self.cal[1]

    @property
    def spd(self) -> int:
        return self.series[0].info.spd


MIN_CAL_SAMPLES = 300  # coarse series (hourly) get more days so the tails have something to fit


def cal_len(cal_days: float, spd: int) -> int:
    return max(int(cal_days * spd), MIN_CAL_SAMPLES)


def clip_events(events: list[tuple[int, int]], start: int) -> list[tuple[int, int]]:
    """Events that end before `start` are dropped, one that starts earlier is cut."""
    return [(max(s, start), e) for s, e in events if e >= start]


# ---- NAB -------------------------------------------------------------------------------


def nab_units(nab_dir: Path, score_files: dict[str, Path], cal_days: float) -> Iterator[Unit]:
    from .nab_data import load_corpus
    from .nab_runner import estimate_interval
    from .nab_score import probation_length

    arrays = {m: np.load(f) for m, f in score_files.items()}
    for s in load_corpus(nab_dir):
        interval = estimate_interval(s.timestamps)
        spd = round(86400 / interval)
        c0 = probation_length(len(s.values))
        c1 = c0 + cal_len(cal_days, spd)
        key = s.path.replace("/", "__")
        raws = {m: np.asarray(a[key], dtype=float) for m, a in arrays.items() if key in a}
        yield Unit(
            "nab",
            s.path,
            interval,
            [UnitSeries(s.values, SeriesInfo("other", spd), raws)],
            (c0, c1),
            clip_events(list(s.windows), c1),
            {"category": s.category},
        )


# ---- SMD and PSM ------------------------------------------------------------------------


def smd_psm_units(
    dataset: str, root: Path, out_dir: Path, models: list[str], cal_days: float
) -> Iterator[Unit]:
    machines = load_smd(root) if dataset == "smd" else load_psm(root)
    for m in machines:
        n0 = len(m.train)
        spd = 1440
        c1 = n0 + cal_len(cal_days, spd)
        series = []
        constant = set(m.constant_dims())
        for d in range(m.dims):
            if d in constant:
                continue
            raws = {}
            for model in models:
                f = score_path(out_dir, dataset, model, m.name, d)
                if f.exists():
                    sc = np.full(n0 + len(m.test), np.nan)
                    sc[n0:] = np.load(f)
                    raws[model] = sc
            values = np.concatenate([m.train[:, d], m.test[:, d]])
            series.append(UnitSeries(values, SeriesInfo("other", spd), raws))
        events = [(a + n0, b + n0) for a, b in label_events(m.labels)]
        yield Unit(dataset, m.name, 60.0, series, (n0, c1), clip_events(events, c1))


# ---- BOOM -------------------------------------------------------------------------------

BOOM_KIND = {"rate": "request", "count": "request", "distribution": "latency", "gauge": "resource"}


def boom_units(root: Path, cache: Path, models: list[str], max_days: float, cal_days: float):
    from .boom_data import load_boom
    from .boom_replay import WARMUP_DAYS, model_spec
    from .replay import cache_name

    series, meta = load_boom(root, max_days, rebase_days=WARMUP_DAYS)
    specs = {m: model_spec(m) for m in models}
    for s in series:
        step = meta[s.key].step
        w = int(WARMUP_DAYS * 86400 / step)
        spd = 86400 // step
        raws = {}
        for m, spec in specs.items():
            f = cache / cache_name(spec, s, step, w)
            if f.exists():
                with np.load(f) as z:
                    raws[m] = np.asarray(z["score"], dtype=float)
        if not raws:
            continue
        info = SeriesInfo(BOOM_KIND.get(meta[s.key].metric_type, "other"), spd)
        yield Unit(
            "boom",
            s.key,
            float(step),
            [UnitSeries(s.values, info, raws)],
            (w, w + cal_len(cal_days, spd)),
            None,
            {"freq": meta[s.key].freq, "metric_type": meta[s.key].metric_type},
        )


# ---- recorded production metrics (company data, results only under reports/) ----------------


def qa_kind(query: str) -> tuple[str, float | None]:
    """(kind, events per minute per unit of value). From the query name only."""
    if re.search(r"4xx|5xx|restarts|preemption|violation|replay|pending", query):
        return "error", None
    if re.search(r"rq_rate|request_success_rate", query):
        return "request", 60.0
    if re.search(r"p95|ttft|itl|e2e|queue_time|rq_time", query):
        return "latency", None
    if re.search(r"^(dcgm|node|k8s)_|kv_cache|requests_running|requests_waiting", query):
        return "resource", None
    return "other", None


QA_STEP = 60
QA_WARMUP_DAYS = 2.0


def qa_specs(models: list[str]):
    from .runner import ModelSpec, parse_model

    return {m: (ModelSpec(m, "subpca") if m == "subpca" else parse_model(m)) for m in models}


def _score_one(a) -> None:
    from .replay import cached_run

    spec, s, step, w, cache = a
    cached_run(spec, s, step, w, cache)


def score_missing(dumps: Path, cache: Path, models: list[str], workers: int) -> None:
    """Run the models that have no cache entry yet (same cache and keys as
    `anomaly-detection-replay`, so existing entries are reused)."""
    from .replay import cache_name, load_dumps

    series = load_dumps([dumps], QA_STEP)
    w = int(QA_WARMUP_DAYS * 86400 / QA_STEP)
    jobs = [
        (spec, s, QA_STEP, w, cache)
        for spec in qa_specs(models).values()
        for s in series
        if not (cache / cache_name(spec, s, QA_STEP, w)).exists()
    ]
    jobs.sort(key=lambda j: j[0].kind != "awsrcf")
    print(f"{len(jobs)} runs missing", flush=True)
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=8) as pool:
        for i, _ in enumerate(pool.imap_unordered(_score_one, jobs, chunksize=1), 1):
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}", flush=True)


def qa_units(dumps: Path, cache: Path, models: list[str], cal_days: float) -> Iterator[Unit]:
    from .replay import cache_name, load_dumps

    series = load_dumps([dumps], QA_STEP)
    w = int(QA_WARMUP_DAYS * 86400 / QA_STEP)
    specs = qa_specs(models)
    spd = 86400 // QA_STEP
    for s in series:
        raws = {}
        for m, spec in specs.items():
            f = cache / cache_name(spec, s, QA_STEP, w)
            if f.exists():
                with np.load(f) as z:
                    raws[m] = np.asarray(z["score"], dtype=float)
        if not raws:
            continue
        kind, per_min = qa_kind(s.query)
        info = SeriesInfo(kind, spd, per_min, zero_is_real=True)
        yield Unit(
            "qa",
            s.key,
            float(QA_STEP),
            [UnitSeries(s.values, info, raws)],
            (w, w + cal_len(cal_days, spd)),
            None,
            {"kind": kind},
        )


# ---- fault injection lab (Azure trace and any lab run with the same files) --------------------

LAB_KIND = {"request_rate": "request", "client_latency": "latency"}


def lab_units(
    dataset: str, lab_dir: Path, models: list[str], cal_days: float, step: int = 5
) -> Iterator[Unit]:
    """`lab_dir` holds signals.csv, labels.json and scores/<model>.<signal>.npy (the files of
    `anomaly_detection.lab.azure_report`). The calibration window is the end of the warm-up,
    where the run has no injected fault."""
    from ..lab import replay
    from ..lab.signals import SIGNALS

    meta, events = replay.load_labels(lab_dir / "labels.json")
    signals = replay.load_signals(lab_dir / "signals.csv", step)
    own = {e["scenario"] for e in events}
    spd = int(meta["day_seconds"] / step)
    for sig in SIGNALS:
        if sig.scenario not in own or sig.name not in signals:
            continue
        run = replay.build_run(sig.name, *signals[sig.name], events, meta, step)
        raws = {}
        for f in sorted((lab_dir / "scores").glob(f"*.{sig.name}.npy")):
            model = f.name[: -len(f".{sig.name}.npy")]
            if model in models:
                raws[model] = np.asarray(np.load(f), dtype=float)
        c1 = run.case.eval_start
        c0 = max(0, c1 - cal_len(cal_days, spd))
        info = SeriesInfo(LAB_KIND.get(sig.name, "other"), spd, None, zero_is_real=True)
        yield Unit(
            dataset,
            sig.name,
            float(step),
            [UnitSeries(run.case.values, info, raws)],
            (c0, c1),
            [(s, e) for s, e in run.own if e >= c1],
            {"scenario": sig.scenario},
        )


def lab_meta(lab_dir: Path) -> dict:
    return json.loads((lab_dir / "labels.json").read_text())["meta"]
