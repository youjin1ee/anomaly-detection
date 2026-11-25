import numpy as np

from anomaly_detection.harness.metrics import AlertRule, alerts, evaluate_case
from anomaly_detection.harness.runner import ModelSpec, parse_model, score_case
from anomaly_detection.harness.scenarios import SCENARIOS, build_case, build_normal_case
from anomaly_detection.harness.synth import generate


def test_same_seed_same_data_and_labels():
    a = build_case("request_drop", 3, 11, 3, 2, 300)
    b = build_case("request_drop", 3, 11, 3, 2, 300)
    assert np.array_equal(a.values, b.values, equal_nan=True)
    assert a.anomalies == b.anomalies
    c = build_case("request_drop", 3, 12, 3, 2, 300)
    assert not np.array_equal(a.values, c.values, equal_nan=True)


def test_injection_lands_in_eval_period_and_changes_values():
    for scenario in SCENARIOS:
        case = build_case(scenario, 1, 5, 3, 2, 300)
        ((start, end),) = case.anomalies
        assert case.eval_start < start < end <= len(case.ts)
        base = {
            "request_drop": "request_rate",
            "token_length_drop": "output_tokens",
            "throughput_drop": "throughput",
            "step_time_rise": "step_time",
        }[scenario]
        normal = generate(base, case.ts, 5)
        mid = (start + end) // 2
        ratio = case.values[mid] / normal[mid] if np.isfinite(case.values[mid]) else None
        if ratio is not None:
            assert ratio > 1.5 if scenario == "step_time_rise" else ratio < 0.7


def test_normal_case_has_no_labels():
    case = build_normal_case("request_rate", "both", 0, 1, 3, 2, 300)
    assert case.anomalies == ()


def test_alert_rule_for_points_and_merge():
    s = np.zeros(40)
    s[5:7] = 1  # only two points: no alert with for=3
    s[10:14] = 1  # fires at index 12
    s[16:20] = 1  # within merge_gap of the previous run: same alert
    s[35:39] = 1  # separate alert, fires at 37
    assert alerts(s, AlertRule(for_points=3, merge_gap=3)) == [12, 37]
    assert alerts(s, AlertRule(for_points=3, merge_gap=0)) == [12, 18, 37]


def test_always_normal_and_always_alarm_detectors_give_expected_metrics():
    rule = AlertRule(for_points=1, grace=0)
    n, step = 1000, 60
    truth = ((600, 700),)
    quiet = evaluate_case(np.zeros(n), truth, 500, step, rule)
    assert (quiet.detected, quiet.false_alarms, len(quiet.fires)) == (0, 0, 0)
    loud = evaluate_case(np.ones(n), truth, 500, step, rule)
    assert loud.detected == 0 and len(loud.fires) == 1 and loud.false_alarms == 1
    # fires at the start of the window, delay zero
    s = np.zeros(n)
    s[620:640] = 1
    hit = evaluate_case(s, truth, 500, step, rule)
    assert hit.detected == 1 and hit.delays == [20 * 60] and hit.false_alarms == 0


def test_alert_before_the_event_is_a_false_alarm():
    rule = AlertRule(for_points=1, grace=0)
    s = np.zeros(1000)
    s[550] = 1
    r = evaluate_case(s, ((600, 700),), 500, 60, rule)
    assert r.detected == 0 and r.false_alarms == 1


def test_score_case_runs_for_each_builtin_kind():
    case = build_normal_case("output_tokens", "both", 0, 1, 3, 1, 300)
    for text in ("fixed", "seasonal", "rcf=rcf:num_trees=5&sample_size=32"):
        spec = parse_model(text)
        assert isinstance(spec, ModelSpec)
        scores, us = score_case(spec, case, 300)
        assert scores.shape == case.values.shape and us > 0


def test_demo_exporter_history_and_faults():
    from anomaly_detection.harness.demo_exporter import current_values, history

    h = history(1_704_067_200 + 3 * 86400, 1, 60)
    assert set(h) == {"request_rate", "output_tokens", "throughput", "step_time"}
    assert len(h["request_rate"]) > 1400
    normal = current_values(1_704_067_200 + 3 * 86400 + 15 * 3600, set())
    faulty = current_values(1_704_067_200 + 3 * 86400 + 15 * 3600, {"request_drop"})
    assert faulty["request_rate"] < 0.1 * normal["request_rate"]
