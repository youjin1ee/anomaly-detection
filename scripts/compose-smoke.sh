#!/usr/bin/env bash
# End-to-end smoke test of the compose demo: wait for the detector, inject a silent failure
# into the generator, and wait for the score to react. Tears down only its own project.
# Usage: scripts/compose-smoke.sh [prometheus]
set -euo pipefail
cd "$(dirname "$0")/../examples/compose"

profile_args=()
detector_port=8080
if [ "${1:-}" = prometheus ]; then
  profile_args=(--profile prometheus)
  detector_port=8081
fi
cleanup() { docker compose --profile prometheus down -v --remove-orphans; }
trap cleanup EXIT

docker compose "${profile_args[@]}" up -d --build

echo "waiting for readiness on :$detector_port"
for _ in $(seq 1 60); do
  curl -fs "localhost:$detector_port/readyz" >/dev/null 2>&1 && break
  sleep 5
done
curl -fs "localhost:$detector_port/readyz" || { docker compose logs --tail 50; exit 1; }
sleep 70 # let a couple of live evaluations happen

echo "--- scores before the failure"
curl -s "localhost:$detector_port/metrics" | grep -E '^anomaly_(score|detector_ready|detector_managed)' || true

curl -fs -X POST 'localhost:9100/fault?kind=request_drop&seconds=900'
started=$(date +%s)
for _ in $(seq 1 60); do
  score=$(curl -s "localhost:$detector_port/metrics" |
    awk '/^anomaly_score\{detector="request_rate_drop"/ {print $NF}')
  echo "score=${score:-none}"
  if [ -n "${score:-}" ] && awk "BEGIN{exit !($score >= 0.8)}"; then
    echo "detected after $(($(date +%s) - started)) seconds"
    curl -s "localhost:$detector_port/metrics" | grep -E '^anomaly_(score|value|expected|lower_band|upper_band)\{detector="request_rate_drop"'
    exit 0
  fi
  sleep 10
done
echo "no detection within 10 minutes"
exit 1
