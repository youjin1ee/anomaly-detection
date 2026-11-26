"""Fault injector: applies faults at scheduled times and writes ground truth labels.

Scenarios and how they are injected (no root needed):

  request_drop        load generator arrival rate x0.03 (upstream stops sending)
  output_length_drop  load generator output token cap 1 to 3 (HTTP 200, answers get cut)
  throughput_drop     a competing GPU matrix multiplication process (contender) on the same GPU
  latency_rise        extra delay before every chunk the load generator receives

The schedule is derived from a seed, so a run can be planned and repeated. Labels are written
after every event, so an interrupted run still leaves usable ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

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


def write_labels(path: Path, meta: dict, events: list[Event]) -> None:
    rows = []
    for e in events:
        d = asdict(e)
        d["expected_signal"] = BY_SCENARIO[e.scenario].name
        d["expected_metric"] = BY_SCENARIO[e.scenario].query
        rows.append(d)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"meta": meta, "events": rows}, indent=1))
    os.replace(tmp, path)
    with path.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "start", "end", "status", "expected_signal", "expected_metric"])
        for r in rows:
            w.writerow(
                [
                    r["scenario"],
                    r["actual_start"] or r["start"],
                    r["actual_end"] or r["end"],
                    r["status"],
                    r["expected_signal"],
                    r["expected_metric"],
                ]
            )


def run(events: list[Event], inj: Injector, labels: Path, meta: dict) -> None:
    write_labels(labels, meta, events)
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
                write_labels(labels, meta, events)
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
    p.add_argument("--dry-run", action="store_true", help="print the schedule and exit")
    args = p.parse_args(argv)

    params = {k: {**v, **json.loads(args.params).get(k, {})} for k, v in DEFAULT_PARAMS.items()}
    eval_start = args.epoch + args.warmup_days * args.day_seconds
    events = build_schedule(
        eval_start + args.first_offset,
        args.per_scenario,
        args.slot_seconds,
        (args.min_duration, args.max_duration),
        args.min_gap,
        args.seed,
        tuple(s for s in args.scenarios.split(",") if s),
        params,
    )
    meta = {
        "epoch": args.epoch,
        "day_seconds": args.day_seconds,
        "warmup_days": args.warmup_days,
        "eval_start": eval_start,
        "seed": args.seed,
        "planned_end": events[-1].end if events else eval_start,
    }
    if args.dry_run:
        for e in events:
            print(f"{e.start - args.epoch:8.0f}s +{e.end - e.start:4.0f}s {e.scenario} {e.params}")
        return 0
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    contender = [args.contender_python, "-m", "anomaly_detection.lab.contender"]
    run(events, Injector(args.loadgen, contender, env), args.labels, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
