"""Cut one long collection into separate runs and measure what every injection did.

A collection has `warmup_days` quiet days followed by injection days, in blocks of
`block_days` (`injector --block-days`). Run N is the warm-up plus block N, with the block moved
back by a whole number of days so the time axis is continuous and the time of day is kept. A
model replayed on run N sees the warm-up and then that block only, never the injections of the
other blocks, so one warm-up serves several runs. The runs share the warm-up, so the limits
of a fixed threshold are the same in all of them.

The labels of each run get, per event, the effect that really happened (`effect`: median of
its own signal from 60 s after the start to the end, divided by the median of the 180 s before,
as in `lab.replay.effect_sizes`), the level before it (`level_before`) and, for subtle events,
`in_target`. Writes <out-root>/run<N>/{signals.csv,labels.json,labels.csv}. The raw dump
(sidecar scores) is not cut: the live sidecar saw every block in a row.

    python -m anomaly_detection.lab.slice_run --dump-dir collect --out-root .
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np

from . import replay
from .injector import Event, write_labels
from .signals import BY_SCENARIO


def block_window(meta: dict, block: int) -> tuple[float, float, float]:
    """(start, end, shift) in unix seconds of a block's injection days and how far it moves."""
    day, w, b = meta["day_seconds"], meta["warmup_days"], meta["block_days"]
    start = meta["epoch"] + (w + (block - 1) * b) * day
    return start, start + b * day, (block - 1) * b * day


def cut_signals(src: Path, dst: Path, meta: dict, block: int, last: bool) -> None:
    warm_end = meta["eval_start"]
    lo, hi, shift = block_window(meta, block)
    with src.open() as f, dst.open("w", newline="") as g:
        r = csv.reader(f)
        w = csv.writer(g)
        w.writerow(next(r))
        for ts, labels, value in r:
            t = float(ts)
            if t < warm_end:
                w.writerow([ts, labels, value])
            elif lo <= t < hi or (last and t >= lo):
                w.writerow([f"{t - shift:.3f}" if shift else ts, labels, value])


def annotate(events: list[Event], signals: dict, settle: float = 60.0, before: float = 180.0):
    """Fill effect, level_before and in_target of every event, in place."""
    for e in events:
        ts, vals = signals[BY_SCENARIO[e.scenario].name]
        s, t_end = e.actual_start, e.actual_end
        inside = vals[(ts >= s + settle) & (ts <= t_end) & np.isfinite(vals)]
        ahead = vals[(ts >= s - before) & (ts < s) & np.isfinite(vals)]
        ok = len(inside) > 0 and len(ahead) > 0 and np.median(ahead) != 0
        e.effect = float(np.median(inside) / np.median(ahead)) if ok else None
        e.level_before = float(np.median(ahead)) if len(ahead) else None
        e.in_target = (
            None if not e.target or e.effect is None else bool(e.target[0] <= e.effect <= e.target[1])
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m anomaly_detection.lab.slice_run", description=__doc__)
    p.add_argument("--dump-dir", type=Path, required=True, help="signals.csv and labels.json")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--step", type=int, default=5)
    a = p.parse_args(argv)

    doc = json.loads((a.dump_dir / "labels.json").read_text())
    meta = doc["meta"]
    nblocks = max(e["block"] for e in doc["events"])
    for block in range(1, nblocks + 1):
        out = a.out_root / f"run{block}"
        out.mkdir(parents=True, exist_ok=True)
        _, _, shift = block_window(meta, block)
        cut_signals(a.dump_dir / "signals.csv", out / "signals.csv", meta, block, block == nblocks)
        fields = set(Event.__dataclass_fields__)
        events = [
            Event(**{k: v for k, v in e.items() if k in fields})
            for e in copy.deepcopy(doc["events"])
            if e["block"] == block
        ]
        for e in events:
            for k in ("start", "end", "actual_start", "actual_end"):
                if getattr(e, k) is not None:
                    setattr(e, k, getattr(e, k) - shift)
        sig = replay.load_signals(out / "signals.csv", a.step)
        done = [e for e in events if e.status == "completed"]
        annotate(done, sig)
        run_meta = {**meta, "run": block, "shift_seconds": shift, "of_collection": str(a.dump_dir)}
        run_meta["planned_end"] = max((e.end for e in events), default=meta["eval_start"])
        write_labels(out / "labels.json", run_meta, events, [])
        print(f"run{block}: {len(done)} completed events of {len(events)}, shift {shift:g} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
