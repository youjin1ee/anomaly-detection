import math

import numpy as np

from anomaly_detection.ensemble.alerts import fast_alert_spans
from anomaly_detection.ensemble.core import (
    EnsembleConfig,
    EnsembleSpec,
    prepare_series,
    score01,
    tier_of,
    unit_score,
)
from anomaly_detection.ensemble.rules import (
    SeriesInfo,
)
from anomaly_detection.harness.ensemble_eval import case_counts
from anomaly_detection.harness.metrics import AlertRule, evaluate_case
from anomaly_detection.models import create_model
from anomaly_detection.models.ensemble import fixed_limits


def test_case_counts_equal_evaluate_case():
    rng = np.random.default_rng(5)
    s = rng.random(400)
    rule = AlertRule(0.8, 3, 10, 5)
    events = [(50, 70), (200, 230)]
    ref = evaluate_case(s, tuple(events), 100, 60, rule)
    fires = [f for _, f, _ in fast_alert_spans(s, rule, 100)]
    det, matched, fa, delays = case_counts(fires, [(50, 70), (200, 230)], 60.0, rule)
    assert fires == ref.fires
    assert (det, matched, fa) == (ref.detected, ref.matched_alerts, ref.false_alarms)
    assert delays == ref.delays


def test_streaming_model_equals_the_array_path():
    spd, warm, cal_n = 48, 96, 144
    rng = np.random.default_rng(7)
    t = np.arange(spd * 10)
    v = 20 + 5 * np.sin(2 * np.pi * t / spd) + rng.normal(0, 0.4, len(t))
    v[400:410] += 12.0
    params = {
        "members": ["subpca", "seasonal", "fixed"],
        "warmup": warm,
        "cal": cal_n,
        "metric_kind": "latency",
        "rules": True,
    }
    interval = 86400.0 / spd
    model = create_model("ensemble", params, interval)
    outs = [model.update(float(i) * interval, float(x)) for i, x in enumerate(v)]
    assert all(o is None for o in outs[: warm + cal_n])
    assert all(o is not None for o in outs[warm + cal_n :])

    raws = {}
    for name in ("subpca", "seasonal"):
        m = create_model(name, {}, interval)
        raws[name] = np.array(
            [
                math.nan if (o := m.update(float(i) * interval, float(x))) is None else o.score
                for i, x in enumerate(v)
            ]
        )
    lo, hi = fixed_limits(v[:warm])
    fixed = np.where((v < lo) | (v > hi), 1.0, 0.0)
    fixed[:warm] = np.nan
    raws["fixed"] = fixed
    cfg = EnsembleConfig()
    cal = (warm, warm + cal_n)
    prep = [prepare_series(raws, v, SeriesInfo("latency", spd), cal, cfg)]
    spec = EnsembleSpec("e", ("subpca", "seasonal", "fixed"), rules=True)
    sc = unit_score(prep, spec, cal, cfg)
    expected = [score01(x, *sc.thresholds) for x in sc.x[warm + cal_n :]]
    got = [o.score for o in outs[warm + cal_n :]]
    assert np.allclose(got, expected, atol=1e-9)
    assert [o.tier for o in outs[warm + cal_n :]] == [
        tier_of(x, *sc.thresholds) for x in sc.x[warm + cal_n :]
    ]
    assert max(got[400 - warm - cal_n :][:15]) > 0  # the injected rise is visible
