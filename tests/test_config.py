import pytest
import yaml

from anomaly_detection.config import load_config, parse_duration


def write(tmp_path, doc):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


BASE = {
    "datasource": {"url": "http://prom:9090/"},
    "detectors": [{"name": "d1", "query": "sum(rate(x[5m]))"}],
}


def test_parse_duration():
    assert parse_duration("90s") == 90
    assert parse_duration("5m") == 300
    assert parse_duration("2d") == 172800
    assert parse_duration(30) == 30
    for bad in ("abc", "5", "-1", 0, True):
        with pytest.raises(ValueError):
            parse_duration(bad)


def test_defaults_and_api_base(tmp_path):
    cfg = load_config(write(tmp_path, BASE))
    d = cfg.detectors[0]
    assert cfg.datasource.api_base == "http://prom:9090/api/v1"
    assert cfg.interval_of(d) == 60
    assert cfg.backfill_of(d) == 86400
    assert cfg.max_series_of(d) == 100
    assert d.direction == "both"
    assert d.model.kind == "seasonal"


def test_tenant_path_prefix(tmp_path):
    doc = {**BASE, "datasource": {"url": "http://vm:8481", "path_prefix": "select/0/prometheus/"}}
    cfg = load_config(write(tmp_path, doc))
    assert cfg.datasource.api_base == "http://vm:8481/select/0/prometheus/api/v1"


def test_env_expansion_only_in_datasource(tmp_path, monkeypatch):
    monkeypatch.setenv("TOK", "secret")
    doc = {
        "datasource": {"url": "http://p", "auth": {"type": "bearer", "token": "${TOK}"}},
        "detectors": [{"name": "d", "query": 'sum(x{a="${NOT_EXPANDED}"})'}],
    }
    cfg = load_config(write(tmp_path, doc))
    assert cfg.datasource.auth.resolve_token() == "secret"
    assert "${NOT_EXPANDED}" in cfg.detectors[0].query


def test_missing_env_is_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    doc = {**BASE, "datasource": {"url": "http://p", "headers": {"X": "${NOPE}"}}}
    with pytest.raises(ValueError, match="NOPE"):
        load_config(write(tmp_path, doc))


def test_basic_auth_requires_credentials(tmp_path):
    doc = {**BASE, "datasource": {"url": "http://p", "auth": {"type": "basic", "username": "u"}}}
    with pytest.raises(ValueError):
        load_config(write(tmp_path, doc))


def test_model_shorthands(tmp_path):
    doc = {
        **BASE,
        "detectors": [
            {"name": "a", "query": "q", "model": "rcf"},
            {"name": "b", "query": "q", "model": {"kind": "seasonal", "band_k": 5}},
        ],
    }
    cfg = load_config(write(tmp_path, doc))
    assert cfg.detectors[0].model.kind == "rcf"
    assert cfg.detectors[1].model.params == {"band_k": 5}


def test_rejects_unknown_keys_and_duplicate_names(tmp_path):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, {**BASE, "bogus": 1}))
    dup = {**BASE, "detectors": [BASE["detectors"][0], BASE["detectors"][0]]}
    with pytest.raises(ValueError, match="duplicate"):
        load_config(write(tmp_path, dup))
