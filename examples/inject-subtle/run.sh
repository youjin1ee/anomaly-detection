#!/bin/bash
# One collection of the subtle fault lab: the Azure trace lab (`../inject-azure`) with a trace
# of weekdays only, big and subtle variants of every scenario, and many injections per day
# (every scenario and severity once in the peak and once in the trough band of each day).
# Store and detector, engine, load generator, GPU exporter, host monitor, injector; then a CSV
# dump and `slice_run`, which cuts the collection into runs. Everything it starts is stopped on
# exit (by PID, nothing else). Engine, load generator, exporter, host monitor, injector and both
# containers run on the cores in CPUSET.
#
#   ENGINE_BIN=/path/to/venv/bin TRACE=/path/code-wk.npz OUT=$HOME/collect ./run.sh
#
# TRACE comes from `python -m anomaly_detection.lab.azure --days 3,4,5,6,...` (see README.md).
# SIZES is a JSON file with the calibrated `duty_by_band` of throughput_drop:subtle.
set -euo pipefail
cd "$(dirname "$0")"
: "${ENGINE_BIN:?bin directory of an environment with vllm and torch}"
: "${OUT:?output directory for logs, labels and the dump}"
: "${TRACE:?npz made by anomaly_detection.lab.azure}"
: "${SIZES:?json with the duty_by_band of throughput_drop:subtle}"
DAY_SECONDS="${DAY_SECONDS:-3600}"
WARMUP_DAYS="${WARMUP_DAYS:-3}"
INJECT_DAYS="${INJECT_DAYS:-9}"
BLOCK_DAYS="${BLOCK_DAYS:-3}"
BAND_HOURS="${BAND_HOURS:-11}"
SEED="${SEED:-1100}"
CPUSET="${CPUSET:-12-19}"
RUNS_ROOT="${RUNS_ROOT:-$(dirname "$OUT")}"
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

if docker volume inspect inject-subtle_vmdata > /dev/null 2>&1; then
  echo "volume inject-subtle_vmdata exists: old samples would mix into this run." >&2
  echo "remove it (it is this lab's own) with: docker compose -f compose.yaml down -v" >&2
  exit 1
fi
# Any process with the GPU open that is not ours would disturb the signals: stop if there is one.
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tee "$OUT/logs/gpu-before.txt"
if grep -v gnome-remote-desktop "$OUT/logs/gpu-before.txt" | grep -q .; then
  echo "another process uses the GPU, not starting" >&2
  exit 1
fi
docker compose -f compose.yaml up -d
ENGINE_BIN="$ENGINE_BIN" start engine ../inject-azure/start-engine.sh
until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do sleep 5; done
# One request so that kernels are compiled before the clock starts.
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"lab-model","prompt":"hi","max_tokens":8}' > /dev/null

EPOCH=$(date +%s)
echo "$EPOCH" > "$OUT/epoch"
start host ../inject-azure/hostmon.sh
start gpu "$PY" -m anomaly_detection.lab.gpuexporter --listen 127.0.0.1:9202
start loadgen "$PY" -m anomaly_detection.lab.loadgen --listen 127.0.0.1:9201 --epoch "$EPOCH" \
  --trace "$TRACE"

# shellcheck disable=SC2086
taskset -c "$CPUSET" "$PY" -m anomaly_detection.lab.injector --epoch "$EPOCH" \
  --day-seconds "$DAY_SECONDS" --warmup-days "$WARMUP_DAYS" --inject-days "$INJECT_DAYS" \
  --block-days "$BLOCK_DAYS" --band-hours "$BAND_HOURS" --seed "$SEED" --subtle \
  --trace "$TRACE" --scenarios request_drop,output_length_drop,throughput_drop \
  --min-duration 120 --max-duration 180 --min-gap 90 --labels "$OUT/labels.json" \
  --params "$(cat "$SIZES")" --contender-python "$ENGINE_BIN/python" $INJECTOR_ARGS \
  2>&1 | tee "$OUT/logs/injector.log"

# Wait for the end of the replayed trace, then let the detector and the store catch up.
END_TRACE=$(( EPOCH + (WARMUP_DAYS + INJECT_DAYS) * DAY_SECONDS ))
while [ "$(date +%s)" -lt "$END_TRACE" ]; do sleep 30; done
sleep 120
END=$(date +%s)
"$PY" -m anomaly_detection.lab.dump --start "$EPOCH" --end "$END" --out-dir "$OUT"
"$PY" -m anomaly_detection.lab.slice_run --dump-dir "$OUT" --out-root "$RUNS_ROOT"
echo "done: $OUT"
