import json

import numpy as np

from anomaly_detection.lab import azure_report

from .test_lab import _lab_signals


def test_wilson_interval_brackets_the_proportion():
    lo, hi = azure_report.wilson(6, 8)
    assert lo < 0.75 < hi and lo >= 0 and hi <= 1
    assert azure_report.wilson(0, 8)[0] == 0.0
    assert np.isnan(azure_report.wilson(0, 0)[0])


def test_azure_report_scores_cached_and_renders_every_section(tmp_path):
    sig, labels = _lab_signals(tmp_path, drop=True)
    doc = json.loads(labels.read_text())
    doc["meta"]["bands"] = {"peak": [0.7, 0.9]}
    doc["events"][0]["band"] = "peak"
    doc["natural_candidates"] = [
        {"kind": "natural_low_traffic", "start": doc["meta"]["epoch"] + 1000,
         "end": doc["meta"]["epoch"] + 1400, "days": [0, 0]}
    ]  # fmt: skip
    labels.write_text(json.dumps(doc))
    out, cache = tmp_path / "r.md", tmp_path / "cache"
    argv = [
        "--signals", str(sig), "--labels", str(labels), "--models", "fixed,seasonal",
        "--cache-dir", str(cache), "--workers", "1", "--out", str(out),
    ]  # fmt: skip
    assert azure_report.main(argv) == 0
    text = out.read_text()
    assert (cache / "seasonal.request_rate.npy").exists()
    for section in ("Pooled over scenarios", "By scenario", "By time of day band",
                    "event order", "Natural candidates", "Effect of the injection"):  # fmt: skip
        assert section in text
    assert "| seasonal | 1 | 1 | 1/1 = 1.00" in text


def test_azure_example_matches_the_lab_signals_and_the_compressed_hour():
    from pathlib import Path

    import yaml

    from anomaly_detection.config import load_config
    from anomaly_detection.lab.signals import BY_NAME

    root = Path(__file__).resolve().parent.parent / "examples" / "inject-azure"
    cfg = load_config(root / "config.yaml")
    assert {d.name for d in cfg.detectors} == {"request_rate", "output_tokens", "decode_speed"}
    for d in cfg.detectors:
        assert d.query == BY_NAME[d.name].query and d.direction == BY_NAME[d.name].direction
        assert d.model.params["seasonality"] == [3600]
    compose = yaml.safe_load((root / "compose.yaml").read_text())
    assert compose["name"] == "inject-azure"  # a project and volume of its own
    assert all("cpuset" in s for s in compose["services"].values())
    assert yaml.safe_load((root / "scrape.yml").read_text())
