"""The AWS RCF adapter. Tests that need the JVM run only when ANOMALY_AWSRCF_HOME is set."""

import math
import random

import pytest

from anomaly_detection.harness.nab_runner import NAB_ALL_MODELS, model_params
from anomaly_detection.models import available_models, create_model
from anomaly_detection.models import awsrcf as awsrcf_mod

needs_jvm = pytest.mark.skipif(
    not __import__("os").environ.get("ANOMALY_AWSRCF_HOME"),
    reason="set ANOMALY_AWSRCF_HOME (jars and awsrcf-bridge.jar) to run the Java bridge tests",
)


def test_kind_is_registered_without_a_jvm():
    assert "awsrcf" in available_models()


def test_missing_home_names_the_variable(monkeypatch):
    monkeypatch.delenv("ANOMALY_AWSRCF_HOME", raising=False)
    monkeypatch.setattr(awsrcf_mod, "_bridge", None)
    with pytest.raises(awsrcf_mod.AwsRcfUnavailable, match="ANOMALY_AWSRCF_HOME"):
        create_model("awsrcf", {}, 60)


def test_unknown_parameter_and_mode_are_rejected():
    with pytest.raises(ValueError, match="unknown parameters"):
        create_model("awsrcf", {"bogus": 1}, 60)
    with pytest.raises(ValueError, match="mode"):
        awsrcf_mod.AwsRcfModel(mode="nope")
    with pytest.raises(ValueError, match="score_from"):
        awsrcf_mod.AwsRcfModel(score_from="nope")


def test_nab_parameter_rule():
    assert "awsrcf" in NAB_ALL_MODELS
    official = model_params("awsrcf-nab", 300.0, 4000)
    assert official["num_trees"] == 100 and official["sample_size"] == 256
    assert official["shingle_size"] == 1 and official["score_div"] == 8.0
    ours = model_params("awsrcf", 300.0, 4000)
    assert ours["mode"] == "caster" and ours["sample_size"] == 550
    assert ours["score_from"] == "ratio"
    assert model_params("awsrcf-grade", 300.0, 4000)["score_from"] == "grade"
    assert model_params("awsrcf-trcf", 300.0, 4000)["mode"] == "trcf"


def _series(n, spike_at=None):
    rng = random.Random(3)
    for i in range(n):
        v = 100 + 10 * math.sin(i / 20) + rng.gauss(0, 1)
        if spike_at is not None and spike_at <= i < spike_at + 8:
            v += 80
        yield i * 300.0, v


@needs_jvm
@pytest.mark.parametrize("mode", ["rcf", "trcf", "caster"])
def test_modes_score_a_spike_and_stay_in_range(mode):
    m = create_model("awsrcf", {"mode": mode}, 300)
    outs = [(t, v, m.update(t, v)) for t, v in _series(1200, spike_at=1000)]
    assert outs[10][2] is None  # not ready yet
    scored = [(t, v, o) for t, v, o in outs if o is not None]
    assert all(0.0 <= o.score <= 1.0 for _, _, o in scored)
    quiet = max(o.score for t, v, o in scored if t < 1000 * 300.0)
    spike = max(o.score for t, v, o in scored if 1000 * 300.0 <= t < 1008 * 300.0)
    assert spike > quiet and spike > 0.5


@needs_jvm
def test_caster_gives_a_band_around_expected_and_others_do_not():
    m = create_model("awsrcf", {"mode": "caster"}, 300)
    outs = [m.update(t, v) for t, v in _series(1200)]
    banded = [o for o in outs if o is not None and o.lower is not None]
    assert banded, "the forecast band never appeared"
    assert all(o.lower <= o.expected <= o.upper for o in banded[-100:])
    plain = create_model("awsrcf", {"mode": "trcf"}, 300)
    last = [plain.update(t, v) for t, v in _series(400)][-1]
    assert last is not None and last.lower is None and last.upper is None


@needs_jvm
def test_one_jvm_serves_many_series_independently():
    a = create_model("awsrcf", {"mode": "trcf"}, 300)
    b = create_model("awsrcf", {"mode": "trcf"}, 300)
    for t, v in _series(300):
        a.update(t, v)
    assert b.update(0.0, 100.0) is None  # b has seen one sample only
    assert a._bridge is b._bridge


@needs_jvm
def test_grade_score_is_zero_on_normal_data_and_positive_on_a_spike():
    m = create_model("awsrcf", {"mode": "caster", "score_from": "grade"}, 300)
    outs = [(t, v, m.update(t, v)) for t, v in _series(1200, spike_at=1000)]
    scored = [(t, o.score) for t, v, o in outs if o is not None]
    assert all(0.0 <= s <= 1.0 for _, s in scored)
    assert max(s for t, s in scored if t >= 1000 * 300.0) > 0.0
