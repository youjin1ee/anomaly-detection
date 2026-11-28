#!/usr/bin/env bash
# SMD and PSM benchmark, stage by stage (see docs/results/m4-smd-psm.md for the choices).
# Usage: bench_smd_psm.sh SMD_DIR PSM_DIR OUT_DIR   (run from the repository root with the venv)
# Cores, memory and nice level are set by the caller (taskset, systemd-run, nice).
# The first run was cut after stage 1 and stage 2 was restarted on the first 10000 test rows
# (the slow models took about 9 minutes per dimension on the shared machine).
set -euo pipefail
SMD=$1 PSM=$2 OUT=$3
PY=${PYTHON:-.venv/bin/python}
CLI="$PY -m anomaly_detection.harness.bench_cli"
export ANOMALY_AWSRCF_JAVA_OPTS="-Xmx1g -XX:ActiveProcessorCount=1 -XX:+UseSerialGC"
FAST=fixed,fixed-tight,seasonal,subpca,awsrcf-nab
mkdir -p "$OUT"
stage() { echo "== $(date +%T) $*"; }
stage 1a smd all, fast models
$CLI run --dataset smd --data-dir "$SMD" --out "$OUT" --machines all --models "$FAST"
stage 1b psm full, fast models and awsrcf
$CLI run --dataset psm --data-dir "$PSM" --out "$OUT" --models "$FAST,awsrcf"
stage 2 smd 3 machines, first 10000 test rows, awsrcf and slow models
$CLI run --dataset smd --data-dir "$SMD" --out "$OUT" --machines s3 --test-rows 10000 \
  --models awsrcf,toto-4m,tspulse,rcf
stage 3 psm first 20000 test rows, slow models
$CLI run --dataset psm --data-dir "$PSM" --out "$OUT" --test-rows 20000 --models toto-4m,tspulse,rcf
stage done
