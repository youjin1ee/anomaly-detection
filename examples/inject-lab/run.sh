#!/bin/bash
# One full data collection run: store and detector, engine, load generator, GPU exporter,
# injector; then a CSV dump. Everything it starts is stopped on exit (by PID, nothing else).
#
#   ENGINE_BIN=/path/to/venv/bin OUT=$HOME/lab-run ./run.sh
#
# Needs: docker compose, an environment with vLLM and torch (ENGINE_BIN), python 3.12 with this
# package importable (PYTHONPATH=../../src is set here).
set -euo pipefail
cd "$(dirname "$0")"
: "${ENGINE_BIN:?bin directory of an environment with vllm and torch}"
: "${OUT:?output directory for logs, labels and the dump}"
DAY_SECONDS="${DAY_SECONDS:-1440}"
WARMUP_DAYS="${WARMUP_DAYS:-4}"
PER_SCENARIO="${PER_SCENARIO:-5}"
BASE_RATE="${BASE_RATE:-1.5}"
INJECTOR_ARGS="${INJECTOR_ARGS:-}"
export PYTHONPATH="$PWD/../../src${PYTHONPATH:+:$PYTHONPATH}"
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
  setsid nohup "$@" > "$OUT/logs/$name.log" 2>&1 &
  echo $! > "$OUT/pids/$name"
}

if docker volume inspect inject-lab_vmdata > /dev/null 2>&1; then
  echo "volume inject-lab_vmdata exists: old samples would mix into this run." >&2
  echo "remove it with: docker compose -f compose.yaml down -v" >&2
  exit 1
fi
docker compose -f compose.yaml up -d --build
ENGINE_BIN="$ENGINE_BIN" start engine ./start-engine.sh
until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do sleep 5; done
# One request so that kernels are compiled before the clock starts.
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"lab-model","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' > /dev/null

EPOCH=$(date +%s)
echo "$EPOCH" > "$OUT/epoch"
start gpu "$PY" -m anomaly_detection.lab.gpuexporter --listen 127.0.0.1:9202
start loadgen "$PY" -m anomaly_detection.lab.loadgen --listen 127.0.0.1:9201 --epoch "$EPOCH" \
  --day-seconds "$DAY_SECONDS" --base-rate "$BASE_RATE"

# shellcheck disable=SC2086
"$PY" -m anomaly_detection.lab.injector --epoch "$EPOCH" --day-seconds "$DAY_SECONDS" \
  --warmup-days "$WARMUP_DAYS" --per-scenario "$PER_SCENARIO" --labels "$OUT/labels.json" \
  --contender-python "$ENGINE_BIN/python" $INJECTOR_ARGS 2>&1 | tee "$OUT/logs/injector.log"

sleep 120   # let the detector and the store catch up with the last event
END=$(date +%s)
"$PY" -m anomaly_detection.lab.dump --start "$EPOCH" --end "$END" --out-dir "$OUT"
echo "done: $OUT"
