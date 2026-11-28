#!/bin/bash
# Pick the competitor duty cycles for `throughput_drop:subtle`: for the peak and the trough
# band, replay a steady load (the band's mean rate) and measure the decode speed ratio of a few
# duties (`lab.calibrate_faults`). Starts its own engine on port 8000 and stops it on exit (by
# PID). Needs a free GPU.
#
#   ENGINE_BIN=/path/to/venv/bin TRACE=code-wk.npz OUT=$HOME/calib ./calibrate-throughput.sh
set -euo pipefail
cd "$(dirname "$0")"
: "${ENGINE_BIN:?bin directory of an environment with vllm and torch}"
: "${TRACE:?npz made by anomaly_detection.lab.azure}"
: "${OUT:?output directory}"
CPUSET="${CPUSET:-12-19}"; DUTIES="${DUTIES:-0.15,0.3,0.5}"; BAND_HOURS="${BAND_HOURS:-11}"
WARMUP_DAYS="${WARMUP_DAYS:-3}"; INJECT_DAYS="${INJECT_DAYS:-9}"
export PYTHONPATH="$PWD/../../src${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1
PY="$ENGINE_BIN/python"
mkdir -p "$OUT"
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tee "$OUT/gpu-before.txt"
PIDS=()
trap 'for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done' EXIT
ENGINE_BIN="$ENGINE_BIN" PORT=8000 setsid nohup taskset -c "$CPUSET" ../inject-azure/start-engine.sh \
  > "$OUT/engine.log" 2>&1 &
PIDS+=($!)
until curl -sf http://127.0.0.1:8000/v1/models > /dev/null; do sleep 5; done
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"lab-model","prompt":"hi","max_tokens":8}' > /dev/null
BANDS=$("$PY" - "$TRACE" "$WARMUP_DAYS" "$INJECT_DAYS" "$BAND_HOURS" <<'PYEOF'
import sys
import numpy as np
from anomaly_detection.lab.azure import band_windows
b = band_windows(np.load(sys.argv[1])["hourly"], int(sys.argv[2]), int(sys.argv[2]) + int(sys.argv[3]), int(sys.argv[4]))
print(" ".join(f"{k}={lo},{hi}" for k, (lo, hi) in b.items()))
PYEOF
)
echo "bands $BANDS" | tee "$OUT/bands.txt"
for kv in $BANDS; do
  band="${kv%%=*}"; frac="${kv#*=}"
  "$PY" -m anomaly_detection.lab.calibrate_faults --make-steady "$TRACE" --band "$frac" \
    --minutes 30 --out "$OUT/steady-$band.npz"
  EPOCH=$(date +%s)
  setsid nohup taskset -c "$CPUSET" "$PY" -m anomaly_detection.lab.loadgen --listen 127.0.0.1:9301 \
    --epoch "$EPOCH" --trace "$OUT/steady-$band.npz" > "$OUT/loadgen-$band.log" 2>&1 &
  LG=$!; PIDS+=($LG)
  sleep 180   # reach a steady state before the first measurement
  taskset -c "$CPUSET" "$PY" -m anomaly_detection.lab.calibrate_faults --duties "$DUTIES" \
    --before 90 --during 150 --after 90 --repeats 1 --contender-python "$PY" \
    --out "$OUT/calib-$band.json" | tee "$OUT/calib-$band.log"
  kill "$LG" 2>/dev/null || true
  sleep 20
done
echo done
