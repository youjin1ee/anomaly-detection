import itertools
import json

import pytest

import numpy as np

from anomaly_detection.lab import injector
from anomaly_detection.lab.injector import (
    SUBTLE_SPECS,
    Event,
    build_banded_schedule,
    draw_params,
    write_labels,
)


def test_banded_schedule_puts_every_scenario_in_every_band_each_day():
    sc = ("a", "b", "c")
    bands = {"peak": (0.5, 0.8), "trough": (0.1, 0.4)}
    days = [1000 + 3600 * d for d in range(4)]
    ev = build_banded_schedule(days, 3600, bands, sc, (120, 240), 120, seed=2)
    assert len(ev) == 4 * 2 * 3
    for s, b in itertools.product(sc, bands):
        assert sum(e.scenario == s and e.band == b for e in ev) == 4
    for prev, nxt in itertools.pairwise(ev):
        assert nxt.start - prev.end >= 120 - 1e-6 or nxt.start < prev.start
    for e in ev:
        frac = ((e.start - 1000) / 3600) % 1
        lo, hi = bands[e.band]
        assert lo <= frac and (e.end - 1000) / 3600 % 1 <= hi + 1e-9
    again = build_banded_schedule(days, 3600, bands, sc, (120, 240), 120, seed=2)
    assert [(e.scenario, e.start) for e in ev] == [(e.scenario, e.start) for e in again]
    with pytest.raises(ValueError):
        build_banded_schedule(days, 3600, {"x": (0.0, 0.1)}, sc, (120, 240), 120, seed=1)
    with pytest.raises(ValueError):
        build_banded_schedule(days, 3600, {"x": (0.0, 0.6), "y": (0.5, 0.9)}, sc, (1, 2), 1, seed=1)


def test_labels_keep_natural_candidates_apart_from_injected_events(tmp_path):
    ev = Event("request_drop", 5, 9, band="peak", status="completed")
    nat = [{"kind": "natural_low_traffic", "start": 1.0, "end": 2.0, "days": [1, 2]}]
    p = tmp_path / "labels.json"
    write_labels(p, {"epoch": 0}, [ev], nat)
    doc = json.loads(p.read_text())
    assert doc["events"][0]["band"] == "peak" and doc["natural_candidates"] == nat
    rows = p.with_suffix(".csv").read_text().splitlines()
    assert rows[1].endswith(",peak,gross,0,,,injected")
    assert rows[2].endswith("candidate,,,,,,,,natural")
    assert len(rows[0].split(",")) == len(rows[1].split(",")) == len(rows[2].split(","))


def test_subtle_variants_are_balanced_and_sized_from_their_ranges():
    sc = ("a", "b", "a:subtle", "b:subtle")
    bands = {"peak": (0.5, 0.9), "trough": (0.0, 0.4)}
    days = [3600 * d for d in range(3)]
    ev = build_banded_schedule(days, 3600, bands, sc, (60, 100), 60, seed=4)
    assert len(ev) == 3 * 2 * 4
    for name, sev, band in itertools.product(("a", "b"), ("gross", "subtle"), bands):
        assert sum(e.scenario == name and e.severity == sev and e.band == band for e in ev) == 3
    specs = {
        "a:subtle": {"factor": [0.5, 0.7], "target": [0.5, 0.7]},
        "b:subtle": {"duty_by_band": {"peak": [0.1, 0.2], "trough": [0.4, 0.6]}, "target": [0.7, 0.85]},
    }  # fmt: skip
    draw_params(ev, specs, None, seed=4)
    for e in ev:
        if e.severity == "gross":
            assert e.target is None and "factor" not in e.params
        elif e.scenario == "a":
            assert 0.5 <= e.params["factor"] <= 0.7 and e.target == [0.5, 0.7]
        else:
            lo, hi = specs["b:subtle"]["duty_by_band"][e.band]
            assert lo <= e.params["duty"] <= hi
    with pytest.raises(ValueError, match="calibrate"):
        draw_params(ev, {"b:subtle": {"duty_by_band": {}, "target": [0, 1]}}, None, 1)


def test_output_cap_for_a_subtle_event_comes_from_the_trace_lengths():
    gen = np.array([2] * 50 + [8] * 30 + [60] * 20)
    (e,) = [Event("output_length_drop", 0, 1, severity="subtle", band="peak")]
    draw_params([e], SUBTLE_SPECS, gen, seed=1)
    r = e.params["target_ratio"]
    assert 0.5 <= r <= 0.7 and 1 <= e.params["cap"] <= 60
    assert np.minimum(gen, e.params["cap"]).mean() >= r * gen.mean()


def test_dry_run_lists_blocks_with_their_own_seeds(tmp_path, capsys):
    hourly = np.tile(np.r_[np.full(8, 10.0), np.full(8, 100.0), np.full(8, 50.0)], 12)
    npz = tmp_path / "t.npz"
    np.savez(npz, hourly=hourly, gen=np.array([2, 8, 60] * 10), t=np.zeros(1))
    argv = [
        "--trace", str(npz), "--labels", str(tmp_path / "l.json"), "--epoch", "0",
        "--day-seconds", "3600", "--warmup-days", "2", "--inject-days", "6", "--block-days", "3",
        "--band-hours", "8", "--subtle", "--min-duration", "100", "--max-duration", "150",
        "--min-gap", "60", "--scenarios", "request_drop,output_length_drop",
        "--params", json.dumps({"throughput_drop:subtle": {"duty_by_band": {}}}),
        "--dry-run",
    ]  # fmt: skip
    assert injector.main(argv) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 6 * 2 * 4
    assert sum(" run1 " in x for x in out) == sum(" run2 " in x for x in out) == 24
    assert sum(" subtle " in x for x in out) == 24


def test_slice_run_cuts_blocks_into_continuous_runs_and_measures_effects(tmp_path):
    from anomaly_detection.lab import slice_run

    day, step = 600, 5
    meta = {"epoch": 1000.0, "day_seconds": day, "warmup_days": 1, "block_days": 1,
            "eval_start": 1600.0}  # fmt: skip
    rows = ["timestamp,labels,value"]
    for t in range(1000, 1000 + 3 * day, step):  # warm-up day, then blocks 1 and 2
        v = 10.0 if not (2300 <= t < 2400) else 5.0  # request rate halves inside event 2
        rows.append(f'{t},"signal=""request_rate""",{v}'.replace('""', '"'))
    (tmp_path / "signals.csv").write_text("\n".join(rows))
    ev = [
        Event("request_drop", 1700, 1800, actual_start=1700, actual_end=1800, status="completed",
              band="peak", block=1, severity="subtle", target=[0.5, 0.7]),
        Event("request_drop", 2300, 2400, actual_start=2300, actual_end=2400, status="completed",
              band="trough", block=2, severity="subtle", target=[0.4, 0.6]),
    ]  # fmt: skip
    write_labels(tmp_path / "labels.json", meta, ev)
    out = tmp_path / "out"
    assert slice_run.main(["--dump-dir", str(tmp_path), "--out-root", str(out)]) == 0
    r1 = json.loads((out / "run1" / "labels.json").read_text())
    r2 = json.loads((out / "run2" / "labels.json").read_text())
    assert r1["events"][0]["effect"] == 1.0 and r1["events"][0]["in_target"] is False
    assert r2["events"][0]["actual_start"] == 1700 and r2["events"][0]["start"] == 1700
    assert r2["events"][0]["effect"] == 0.5 and r2["events"][0]["in_target"] is True
    ts = [float(x.split(",")[0]) for x in (out / "run2" / "signals.csv").read_text().splitlines()[1:]]
    assert ts == sorted(ts) and ts[0] == 1000 and ts[-1] == 2195  # warm-up, then block 2 moved
    assert set(ts) == set(range(1000, 2200, 5))
