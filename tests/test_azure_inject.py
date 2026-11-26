import itertools
import json

import pytest

from anomaly_detection.lab.injector import Event, build_banded_schedule, write_labels


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
    assert rows[1].endswith(",peak,injected") and rows[2].endswith("candidate,,,,natural")
