#!/bin/bash
# Capacity check before a run: replay a stretch of a prepared trace against a running engine
# (port 8000) and print what the engine did. Used to pick the arrival rate scale of `lab.azure`.
#
#   TRACE=code-r4.npz FROM=13500 SECONDS_RUN=600 PY=$ENGINE_BIN/python ./calibrate.sh
#
# FROM is the lab time (seconds from trace day 0) to start at, for example the peak hours of a
# weekday. Output: median and 90th percentile of running and waiting requests and of the KV
# cache usage, mean time to first token and mean request time from the engine histograms,
# completed requests per second, errors seen by the load generator.
set -euo pipefail
: "${TRACE:?npz}"; : "${FROM:?lab seconds into the trace}"; PY="${PY:-python3}"
SECONDS_RUN="${SECONDS_RUN:-600}"; CPUSET="${CPUSET:-12-19}"
export PYTHONPATH="$PWD/../../src${PYTHONPATH:+:$PYTHONPATH}"
snap() { curl -sf http://127.0.0.1:8000/metrics; }
val() { awk -v n="$1" '$1 ~ "^"n"({|$)" {s+=$2} END{print s+0}'; }
EPOCH=$(( $(date +%s) - FROM ))
taskset -c "$CPUSET" "$PY" -m anomaly_detection.lab.loadgen --listen 127.0.0.1:9301 --epoch "$EPOCH" \
  --trace "$TRACE" --duration 0 > /tmp/calibrate-loadgen.$$.log 2>&1 &
LG=$!
trap 'kill $LG 2>/dev/null || true' EXIT
s0=$(snap); t0=$(date +%s)
: > /tmp/calibrate-samples.$$.txt
while [ $(( $(date +%s) - t0 )) -lt "$SECONDS_RUN" ]; do
  m=$(snap)
  echo "$(echo "$m" | val 'vllm:num_requests_running') $(echo "$m" | val 'vllm:num_requests_waiting') $(echo "$m" | val 'vllm:kv_cache_usage_perc')" >> /tmp/calibrate-samples.$$.txt
  sleep 10
done
s1=$(snap)
d() { echo "$(echo "$s1" | val "$1") $(echo "$s0" | val "$1")" | awk '{print $1-$2}'; }
n=$(d vllm:request_success_total)
echo "completed_per_s $(echo "$n $SECONDS_RUN" | awk '{printf "%.2f", $1/$2}')"
echo "mean_ttft_s $(echo "$(d vllm:time_to_first_token_seconds_sum) $(d vllm:time_to_first_token_seconds_count)" | awk '{printf "%.3f", $1/($2+1e-9)}')"
echo "mean_e2e_s $(echo "$(d vllm:e2e_request_latency_seconds_sum) $(d vllm:e2e_request_latency_seconds_count)" | awk '{printf "%.3f", $1/($2+1e-9)}')"
echo "mean_itl_ms $(echo "$(d vllm:inter_token_latency_seconds_sum) $(d vllm:inter_token_latency_seconds_count)" | awk '{printf "%.1f", 1000*$1/($2+1e-9)}')"
"$PY" - /tmp/calibrate-samples.$$.txt <<'PYEOF'
import sys
import numpy as np
a = np.loadtxt(sys.argv[1])
for i, name in enumerate(("running", "waiting", "kv_cache_usage")):
    print(name, "median", round(float(np.median(a[:, i])), 3), "p90", round(float(np.percentile(a[:, i], 90)), 3), "max", round(float(a[:, i].max()), 3))
PYEOF
curl -sf http://127.0.0.1:9301/metrics | grep -E '^lab_requests_sent_total' || true
