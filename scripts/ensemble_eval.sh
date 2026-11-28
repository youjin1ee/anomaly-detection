#!/usr/bin/env bash
# Ensemble evaluation on the cached model scores of the earlier runs (docs/results/m5-ensemble.md).
# Usage: ensemble_eval.sh OUT_DIR   (repository root, venv, env of the earlier runs set)
# Needs ANOMALY_DATA (data root), RUNS (output root of the m3 and m4 runs). Missing QA scores
# are computed first (`qa-score`). Cores, memory and nice level are set by the caller.
# A fault injection run is added with: LAB="name=DIR" (DIR has signals.csv, labels.json, scores/).
set -euo pipefail
OUT=$1
PY=${PYTHON:-.venv/bin/python}
CLI="$PY -m anomaly_detection.harness.ensemble_cli"
: "${ANOMALY_DATA:?}" "${RUNS:?}"
A=$ANOMALY_DATA R=$RUNS
mkdir -p "$OUT"
NAB="--nab-dir $A/nab --nab-score fixed=$A/nab-scores/seed0/fixed.npz"
NAB="$NAB --nab-score seasonal=$A/nab-scores/seed0/seasonal.npz"
NAB="$NAB --nab-score awsrcf-nab=$R/m3-poc-awsrcf-run/scores/awsrcf-nab.npz"
NAB="$NAB --nab-score awsrcf=$R/m3-poc-awsrcf-run/scores/awsrcf.npz"
NAB="$NAB --nab-score subpca=$R/m3-integrate-out/scores/subpca.npz"
$CLI run --out "$OUT/nab.json" $NAB
$CLI run --out "$OUT/benchmarks.json" --smd-root "$A/public/smd" --psm-root "$A/public/psm" \
  --bench-out "$R/m4-bench-smd-psm-out/run"
$CLI run --out "$OUT/boom.json" --boom-root "$A/public/boom" --boom-cache "$R/m4-bench-boom-out/cache"
LABS=(--lab-dir "azure=$A/inject-azure-20261127/run-code")
[ -n "${LAB:-}" ] && LABS+=(--lab-dir "$LAB")
$CLI run --out "$OUT/lab.json" "${LABS[@]}"
Q=$A/qa-cluster-20261126
$CLI qa-score --qa-dumps "$Q" --qa-cache "$Q/reports/cache" --workers "${WORKERS:-6}"
$CLI run --out "$Q/reports/m5-ensemble-qa.json" --qa-dumps "$Q" --qa-cache "$Q/reports/cache"
$CLI report "$OUT"/nab.json "$OUT"/benchmarks.json "$OUT"/boom.json "$OUT"/lab.json \
  "$Q/reports/m5-ensemble-qa.json" > "$OUT/report.md"
