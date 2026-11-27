"""Fault injector: applies faults at scheduled times and writes ground truth labels.

Scenarios and how they are injected (no root needed):

  request_drop        load generator arrival rate x0.03 (upstream stops sending)
  output_length_drop  load generator output token cap 1 to 3 (HTTP 200, answers get cut)
  throughput_drop     a competing GPU matrix multiplication process (contender) on the same GPU
  latency_rise        extra delay before every chunk the load generator receives

Each scenario also has a subtle variant (key `name:subtle`, label severity "subtle"): the same
fault at a size meant to stay inside the range the signal shows on a normal day, so a fixed
limit cannot see it. The size is drawn per event from a range and the intended range is stored
in the label (`target`); the effect that really happened is measured afterwards (`slice_run`).

  request_drop:subtle        arrival rate x0.5 to 0.7
  output_length_drop:subtle  output cap set so the mean answer is 0.5 to 0.7 of normal
  throughput_drop:subtle     competitor with a duty cycle below 1, tuned per time of day band
                             so that decode speed falls to 0.7 to 0.85

The schedule is derived from a seed, so a run can be planned and repeated. Labels are written
after every event, so an interrupted run still leaves usable ground truth.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .azure import band_windows, cap_for_ratio, low_spans
from .signals import BY_SCENARIO, SCENARIOS


@dataclass
class Event:
    scenario: str
    start: float  # planned start, unix seconds
    end: float  # planned end
    params: dict = field(default_factory=dict)
    actual_start: float | None = None
    actual_end: float | None = None
    status: str = "planned"  # planned | completed | failed
    band: str = ""  # time of day band for a banded schedule (peak | trough)
    severity: str = "gross"  # gross | subtle
    block: int = 0  # which run of a collection (see slice_run) the event belongs to
    target: list[float] | None = None  # intended effect ratio range of a subtle event


def build_schedule(
    first_start: float,
    per_scenario: int,
    slot_seconds: float,
    duration_range: tuple[float, float],
    min_gap: float,
    seed: int,
    scenarios: tuple[str, ...] = SCENARIOS,
    params: dict[str, dict] | None = None,
) -> list[Event]:
    """One event per slot, scenarios shuffled so every scenario appears `per_scenario` times.

    Inside a slot the start is random, but the event always ends at least `min_gap` seconds
    before the next slot begins, so events never overlap and the metrics can recover.
    """
    rng = random.Random(seed)
    order = [s for s in scenarios for _ in range(per_scenario)]
    rng.shuffle(order)
    events: list[Event] = []
    for i, scenario in enumerate(order):
        dur = rng.uniform(*duration_range)
        if dur + min_gap > slot_seconds:
            raise ValueError("slot_seconds must exceed the longest duration plus min_gap")
        slot_start = first_start + i * slot_seconds
        offset = rng.uniform(0, slot_seconds - dur - min_gap)
        start = slot_start + offset
        events.append(Event(scenario, start, start + dur, dict((params or {}).get(scenario, {}))))
    return events


def build_banded_schedule(
    day_starts: list[float],
    day_seconds: float,
    bands: dict[str, tuple[float, float]],
    scenarios: tuple[str, ...],
    duration_range: tuple[float, float],
    min_gap: float,
    seed: int,
    params: dict[str, dict] | None = None,
) -> list[Event]:
    """Every scenario once in every band of every day, so each scenario meets each time of day
    equally often. A band is (start, end) as fractions of the day, cut into one slot per
    scenario; the order of the scenarios inside a band is shuffled. As in `build_schedule`, an
    event ends at least `min_gap` before its slot ends, so events never overlap. A scenario
    may be written `name:subtle`, which sets the severity of its events."""
    edges = sorted(bands.values())
    if any(hi > 1 for _, hi in edges) or any(a[1] > b[0] for a, b in itertools.pairwise(edges)):
        raise ValueError("bands must stay inside one day and not overlap")
    rng = random.Random(seed)
    events: list[Event] = []
    for day_start in day_starts:
        for band, (lo, hi) in sorted(bands.items(), key=lambda b: b[1]):
            slot = (hi - lo) * day_seconds / len(scenarios)
            order = list(scenarios)
            rng.shuffle(order)
            for i, scenario in enumerate(order):
                dur = rng.uniform(*duration_range)
                if dur + min_gap > slot:
                    raise ValueError("a band is too short for its slots, duration and min_gap")
                start = (
                    day_start + lo * day_seconds + i * slot + rng.uniform(0, slot - dur - min_gap)
                )
                params_ = dict((params or {}).get(scenario, {}))
                name, _, sev = scenario.partition(":")
                events.append(
                    Event(name, start, start + dur, params_, band=band, severity=sev or "gross")
                )
    return events


# Sizes of the subtle variants. duty_by_band is filled from `lab.calibrate_faults` (the competitor
# slows decoding more when many requests are in flight, so the peak needs a smaller duty).
SUBTLE_SPECS: dict[str, dict] = {
    "request_drop:subtle": {"factor": [0.5, 0.7], "target": [0.5, 0.7]},
    "output_length_drop:subtle": {"target_ratio": [0.5, 0.7], "target": [0.5, 0.7]},
    "throughput_drop:subtle": {"duty_by_band": {}, "target": [0.7, 0.85]},
}


def draw_params(
    events: list[Event], specs: dict[str, dict], gen: np.ndarray | None, seed: int
) -> None:
    """Fill the size of every subtle event from its spec: `factor` and `duty` are drawn
    uniformly from a range, `target_ratio` is turned into an output cap with `gen` (the trace's
    generated token counts). Deterministic for a seed."""
    rng = random.Random(seed)
    for ev in events:
        spec = specs.get(f"{ev.scenario}:{ev.severity}") if ev.severity != "gross" else None
        if not spec:
            continue
        ev.target = list(spec["target"])
        if "factor" in spec:
            ev.params["factor"] = round(rng.uniform(*spec["factor"]), 3)
        if "target_ratio" in spec:
            if gen is None:
                raise ValueError("target_ratio needs the trace (--trace)")
            r = rng.uniform(*spec["target_ratio"])
            ev.params.update(target_ratio=round(r, 3), cap=cap_for_ratio(gen, r))
        if "duty_by_band" in spec:
            lo_hi = spec["duty_by_band"].get(ev.band)
            if not lo_hi:
                raise ValueError(f"no duty range for band {ev.band!r}: calibrate first")
            ev.params["duty"] = round(rng.uniform(*lo_hi), 3)


def _call(url: str, method: str, body: dict | None = None) -> None:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()


class Injector:
    def __init__(self, loadgen: str, contender_cmd: list[str], env: dict[str, str] | None = None):
        self.loadgen = loadgen.rstrip("/")
        self.contender_cmd = contender_cmd
        self.env = env
        self._proc: subprocess.Popen | None = None

    def apply(self, ev: Event) -> None:
        p = ev.params
        if ev.scenario == "request_drop":
            _call(f"{self.loadgen}/fault", "POST", {"kind": "request_drop", **p})
        elif ev.scenario == "output_length_drop":
            _call(f"{self.loadgen}/fault", "POST", {"kind": "output_drop", **p})
        elif ev.scenario == "latency_rise":
            _call(f"{self.loadgen}/fault", "POST", {"kind": "latency_rise", **p})
        elif ev.scenario == "throughput_drop":
            cmd = [*self.contender_cmd, "--duty", str(p.get("duty", 1.0))]
            self._proc = subprocess.Popen(cmd, env=self.env)
        else:
            raise ValueError(ev.scenario)

    def clear(self) -> None:
        _call(f"{self.loadgen}/fault", "DELETE")
        if self._proc is not None:
            self._proc.terminate()  # SIGTERM to the child we started, nothing else
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None


def write_labels(
    path: Path, meta: dict, events: list[Event], natural: list[dict] | None = None
) -> None:
    """`natural` are stretches where the replayed trace itself looks anomalous. They are kept
    apart from the injected events (`natural_candidates`, and source=natural in the csv)."""
    rows = []
    for e in events:
        d = asdict(e)
        d["expected_signal"] = BY_SCENARIO[e.scenario].name
        d["expected_metric"] = BY_SCENARIO[e.scenario].query
        rows.append(d)
    tmp = path.with_suffix(".tmp")
    doc = {"meta": meta, "events": rows, "natural_candidates": natural or []}
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, path)
    with path.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "start",
                "end",
                "status",
                "expected_signal",
                "expected_metric",
                "band",
                "severity",
                "block",
                "source",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["scenario"],
                    r["actual_start"] or r["start"],
                    r["actual_end"] or r["end"],
                    r["status"],
                    r["expected_signal"],
                    r["expected_metric"],
                    r["band"],
                    r["severity"],
                    r["block"],
                    "injected",
                ]
            )
        for n in natural or []:
            w.writerow([n["kind"], n["start"], n["end"], "candidate", "", "", "", "", "", "natural"])


def run(
    events: list[Event],
    inj: Injector,
    labels: Path,
    meta: dict,
    natural: list[dict] | None = None,
) -> None:
    write_labels(labels, meta, events, natural)
    try:
        for ev in events:
            time.sleep(max(0.0, ev.start - time.time()))
            ev.actual_start = time.time()
            try:
                inj.apply(ev)
                print(f"{ev.actual_start:.0f} start {ev.scenario} {ev.params}", flush=True)
                time.sleep(max(0.0, ev.end - time.time()))
                ev.status = "completed"
            except Exception as e:  # keep the run going, the label says what happened
                ev.status = f"failed: {e}"
                print(f"{time.time():.0f} FAILED {ev.scenario}: {e}", file=sys.stderr, flush=True)
            finally:
                inj.clear()
                ev.actual_end = time.time()
                print(f"{ev.actual_end:.0f} end {ev.scenario}", flush=True)
                write_labels(labels, meta, events, natural)
    finally:
        inj.clear()


DEFAULT_PARAMS = {
    "request_drop": {"factor": 0.03},
    "output_length_drop": {},
    "throughput_drop": {"duty": 1.0},
    "latency_rise": {"delay": 0.06},
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anomaly-detection-injector", description=__doc__)
    p.add_argument("--loadgen", default="http://127.0.0.1:9201")
    p.add_argument("--labels", type=Path, required=True, help="labels.json (a .csv is written too)")
    p.add_argument("--day-seconds", type=float, default=1440)
    p.add_argument("--epoch", type=float, required=True, help="start of day 0 (unix seconds)")
    p.add_argument("--warmup-days", type=float, default=4)
    p.add_argument("--first-offset", type=float, default=720, help="seconds after warmup")
    p.add_argument("--per-scenario", type=int, default=5)
    p.add_argument("--slot-seconds", type=float, default=600)
    p.add_argument("--min-duration", type=float, default=120)
    p.add_argument("--max-duration", type=float, default=300)
    p.add_argument("--min-gap", type=float, default=180)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--params", default="{}", help="JSON, per scenario overrides of the defaults")
    p.add_argument(
        "--contender-python",
        default=sys.executable,
        help="python with torch and CUDA, used for throughput_drop",
    )
    p.add_argument(
        "--trace",
        type=Path,
        help="npz of the replayed trace: one event per scenario in the peak and the trough band"
        " of every day after warm-up (use --warmup-days, --inject-days), natural labels added",
    )
    p.add_argument("--inject-days", type=int, default=4, help="with --trace")
    p.add_argument("--band-hours", type=int, default=8, help="with --trace")
    p.add_argument(
        "--block-days",
        type=int,
        default=0,
        help="with --trace: cut the injection days into runs of this many days, each with its"
        " own schedule seed (seed + run number); 0 = one run",
    )
    p.add_argument(
        "--subtle", action="store_true", help="add the subtle variant of every scenario"
    )
    p.add_argument("--dry-run", action="store_true", help="print the schedule and exit")
    args = p.parse_args(argv)

    params = {k: {**v, **json.loads(args.params).get(k, {})} for k, v in DEFAULT_PARAMS.items()}
    eval_start = args.epoch + args.warmup_days * args.day_seconds
    scenarios = tuple(s for s in args.scenarios.split(",") if s)
    if args.subtle:
        scenarios += tuple(f"{s}:subtle" for s in scenarios if ":" not in s)
    specs = json.loads(json.dumps(SUBTLE_SPECS))
    for k, v in json.loads(args.params).items():
        if k in specs:
            specs[k].update(v)
    natural: list[dict] = []
    if args.trace:
        z = np.load(args.trace)
        first = int(args.warmup_days)
        bands = band_windows(z["hourly"], first, first + args.inject_days, args.band_hours)
        step = args.block_days or args.inject_days
        events = []
        for block, d0 in enumerate(range(0, args.inject_days, step)):
            part = build_banded_schedule(
                [
                    args.epoch + (first + d) * args.day_seconds
                    for d in range(d0, min(d0 + step, args.inject_days))
                ],
                args.day_seconds,
                bands,
                scenarios,
                (args.min_duration, args.max_duration),
                args.min_gap,
                args.seed + block,
                params,
            )
            draw_params(part, specs, z["gen"], args.seed + block)
            for e in part:
                e.block = block + 1
            events += part
        scale = args.day_seconds / 86400.0
        natural = [
            {
                "kind": "natural_low_traffic",
                "start": args.epoch + sp["start"] * scale,
                "end": args.epoch + sp["end"] * scale,
                "days": sp["days"],
                "ratio_to_median_day": sp["ratio_to_median_day"],
            }
            for sp in low_spans(z["hourly"])
        ]
    else:
        bands = {}
        events = build_schedule(
            eval_start + args.first_offset,
            args.per_scenario,
            args.slot_seconds,
            (args.min_duration, args.max_duration),
            args.min_gap,
            args.seed,
            scenarios,
            params,
        )
    meta = {
        "epoch": args.epoch,
        "day_seconds": args.day_seconds,
        "warmup_days": args.warmup_days,
        "eval_start": eval_start,
        "seed": args.seed,
        "planned_end": events[-1].end if events else eval_start,
        "bands": bands,
        "block_days": args.block_days,
        "subtle_specs": specs if args.subtle else {},
    }
    if args.dry_run:
        for e in events:
            print(
                f"{e.start - args.epoch:8.0f}s +{e.end - e.start:4.0f}s {e.scenario}"
                f" {e.severity} {e.band} run{e.block} {e.params}"
            )
        return 0
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    contender = [args.contender_python, "-m", "anomaly_detection.lab.contender"]
    run(events, Injector(args.loadgen, contender, env), args.labels, meta, natural)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
