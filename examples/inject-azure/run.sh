#!/bin/bash
# One data collection run of the Azure trace lab: store and detector, engine, load generator
# replaying a prepared trace, GPU exporter, host monitor, injector; then a CSV dump. Everything
# it starts is stopped on exit (by PID, nothing else). Engine, load generator, exporter, host
# monitor, injector and both containers run on the cores in CPUSET.
#
#   ENGINE_BIN=/path/to/venv/bin TRACE=/path/code.npz OUT=$HOME/azure-run ./run.sh
#
# TRACE comes from `python -m anomaly_detection.lab.azure` (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
: "${ENGINE_BIN:?bin directory of an environment with vllm and torch}"
: "${OUT:?output directory for logs, labels and the dump}"
: "${TRACE:?npz made by anomaly_detection.lab.azure}"
DAY_SECONDS="${DAY_SECONDS:-3600}"
WARMUP_DAYS="${WARMUP_DAYS:-3}"
INJECT_DAYS="${INJECT_DAYS:-4}"
CPUSET="${CPUSET:-12-19}"
INJECTOR_ARGS="${INJECTOR_ARGS:-}"
export CPUSET HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" PYTHONPATH="$PWD/../../src${PYTHONPATH:+:$PYTHONPATH}"
PY="${LAB_PYTHON:-$ENGINE_BIN/python}"
mkdir -p "$OUT/logs" "$OUT/pids"

stop_all() {
  for f in "$OUT"/pids/*; do
    [ -f "$f" ] || continue
    kill "$(cat "$f")" 2>/dev/null || true
    rm -f "$f"
  done
  docker compose -f compose.yaml down 2>/dev/null || true   # keeps the volume
}
trap stop_all EXIT

start() {  # name, command...
  local name="$1"; shift
  setsid nohup taskset -c "$CPUSET" "$@" > "$OUT/logs/$name.log" 2>&1 &
  echo $! > "$OUT/pids/$name"
}

if docker volume inspect inject-azure_vmdata > /dev/null 2>&1; then
  echo "volume inject-azure_vmdata exists: old samples would mix into this run." >&2
  echo "remove it (it is this lab's own) with: docker compose -f compose.yaml down -v" >&2
  exit 1
fi
# Any process with the GPU open that is not ours would disturb the signals: say so up front.
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tee "$OUT/logs/gpu-before.txt"
docker compose -f compose.yaml up -d
ENGINE_BIN="$ENGINE_BIN" start engine ./start-engine.sh
until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do sleep 5; done
# One request so that kernels are compiled before the clock starts.
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"lab-model","prompt":"hi","max_tokens":8}' > /dev/null

EPOCH=$(date +%s)
echo "$EPOCH" > "$OUT/epoch"
start host ./hostmon.sh
start gpu "$PY" -m anomaly_detection.lab.gpuexporter --listen 127.0.0.1:9202
start loadgen "$PY" -m anomaly_detection.lab.loadgen --listen 127.0.0.1:9201 --epoch "$EPOCH" \
  --trace "$TRACE"

# shellcheck disable=SC2086
taskset -c "$CPUSET" "$PY" -m anomaly_detection.lab.injector --epoch "$EPOCH" \
  --day-seconds "$DAY_SECONDS" --warmup-days "$WARMUP_DAYS" --inject-days "$INJECT_DAYS" \
  --trace "$TRACE" --scenarios request_drop,output_length_drop,throughput_drop \
  --min-duration 120 --max-duration 240 --min-gap 120 --labels "$OUT/labels.json" \
  --contender-python "$ENGINE_BIN/python" $INJECTOR_ARGS 2>&1 | tee "$OUT/logs/injector.log"

# Wait for the end of the replayed week, then let the detector and the store catch up.
END_TRACE=$(( EPOCH + 7 * DAY_SECONDS ))
while [ "$(date +%s)" -lt "$END_TRACE" ]; do sleep 30; done
sleep 120
END=$(date +%s)
"$PY" -m anomaly_detection.lab.dump --start "$EPOCH" --end "$END" --out-dir "$OUT"
echo "done: $OUT"
