from pathlib import Path

import pytest
import yaml

from anomaly_detection.config import load_config
from anomaly_detection.models import create_model

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize(
    "path",
    [
        "config.example.yaml",
        "compose/config-vm.yaml",
        "compose/config-prometheus.yaml",
    ],
)
def test_example_configs_load_and_build_models(path, monkeypatch):
    monkeypatch.setenv("METRICS_TOKEN", "x")
    cfg = load_config(EXAMPLES / path)
    for d in cfg.detectors:
        create_model(d.model.kind, dict(d.model.params), cfg.interval_of(d))


@pytest.mark.parametrize(
    "path", ["compose/compose.yaml", "compose/scrape-vm.yml", "compose/prometheus.yml"]
)
def test_example_yaml_parses(path):
    assert yaml.safe_load((EXAMPLES / path).read_text())


def test_kubernetes_manifest_embedded_config_is_valid(tmp_path):
    docs = list(yaml.safe_load_all((EXAMPLES / "kubernetes/sidecar.yaml").read_text()))
    cm = next(d for d in docs if d["kind"] == "ConfigMap")
    p = tmp_path / "c.yaml"
    p.write_text(cm["data"]["config.yaml"])
    assert load_config(p).detectors
