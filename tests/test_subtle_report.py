import json

import numpy as np

from anomaly_detection.lab import subtle_report as sr

from .test_lab import _lab_signals


def test_poisson_interval_matches_known_values():
    lo, hi = sr.poisson_ci(0)
    assert lo == 0.0 and abs(hi - 3.689) < 0.01
    lo, hi = sr.poisson_ci(10)
    assert abs(lo - 4.795) < 0.01 and abs(hi - 18.390) < 0.01


def test_paired_bootstrap_interval_brackets_the_difference():
    a = np.array([True] * 8 + [False] * 2)
    b = np.array([True] * 4 + [False] * 6)
    lo, hi = sr.paired_bootstrap(a, b, n=2000)
    assert lo < 0.4 < hi and lo > -0.1


def test_report_tables_from_one_run_with_a_subtle_and_a_gross_label(tmp_path, capsys):
    sig, labels = _lab_signals(tmp_path, drop=True)
    doc = json.loads(labels.read_text())
    doc["meta"]["bands"] = {"peak": [0.7, 0.9]}
    e = doc["events"][0]
    e.update(band="peak", severity="subtle", effect=0.04, in_target=False)
    labels.write_text(json.dumps(doc))
    out = tmp_path / "r.md"
    argv = ["--runs", str(tmp_path), "--models", "fixed,seasonal", "--workers", "1",
            "--out", str(out), "--events-csv", str(tmp_path / "e.csv")]  # fmt: skip
    assert sr.main(argv) == 0
    text = out.read_text()
    assert "| subtle | seasonal | 1 | 1/1 = 1.00" in text
    for head in ("크기별 전체", "오탐", "크기, 시나리오, 시간대별", "고정 임계 대비", "실제 효과", "일차별"):
        assert head in text
    idx = json.loads((tmp_path / "scores" / "index.json").read_text())
    assert idx["time_stretch"] == 60.0 and "request_rate" in idx["signals"]
    sc = np.load(tmp_path / "scores" / "seasonal.request_rate.npy")
    assert len(sc) == idx["signals"]["request_rate"]["n"]
    assert (tmp_path / "e.csv").read_text().splitlines()[0].startswith("run,model")
