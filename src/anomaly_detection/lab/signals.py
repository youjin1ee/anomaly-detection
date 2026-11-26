"""The signals a detector watches in the lab, and the fault each one is meant to reveal.

The PromQL here is shared by the dump (which evaluates it at a fixed step) and by the lab
detector config (examples/inject-lab/config.yaml), so a replay sees what the sidecar saw.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    name: str
    query: str
    direction: str  # below | above
    scenario: str  # fault scenario this signal is expected to show


SIGNALS: tuple[Signal, ...] = (
    Signal(
        "request_rate",
        "sum(rate(vllm:request_success_total[1m]))",
        "below",
        "request_drop",
    ),
    Signal(
        "output_tokens",
        "sum(rate(vllm:generation_tokens_total[1m])) / sum(rate(vllm:request_success_total[1m]))",
        "below",
        "output_length_drop",
    ),
    Signal(
        "decode_speed",
        "sum(rate(vllm:inter_token_latency_seconds_count[1m]))"
        " / sum(rate(vllm:inter_token_latency_seconds_sum[1m]))",
        "below",
        "throughput_drop",
    ),
    Signal(
        "client_latency",
        "sum(rate(lab_request_duration_seconds_sum[1m]))"
        " / sum(rate(lab_request_duration_seconds_count[1m]))",
        "above",
        "latency_rise",
    ),
)

SCENARIOS = tuple(s.scenario for s in SIGNALS)
BY_SCENARIO = {s.scenario: s for s in SIGNALS}
BY_NAME = {s.name: s for s in SIGNALS}

# Metric families kept in the raw dump.
RAW_MATCH = '{__name__=~"vllm:.*|lab_.*|gpu_.*|host_.*|anomaly_.*"}'
