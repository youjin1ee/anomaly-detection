#!/usr/bin/env bash
# SMD and PSM benchmark, stage by stage (see docs/results/m4-smd-psm.md for the choices).
# Usage: bench_smd_psm.sh SMD_DIR PSM_DIR OUT_DIR   (run from the repository root with the venv)
# Cores, memory and nice level are set by the caller (taskset, systemd-run, nice).
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
stage 2 smd 3 machines, awsrcf and slow models
$CLI run --dataset smd --data-dir "$SMD" --out "$OUT" --machines s3 --models awsrcf,toto-4m,tspulse,rcf
stage 3 psm first 40000 test rows, slow models
$CLI run --dataset psm --data-dir "$PSM" --out "$OUT" --test-rows 40000 --models toto-4m,tspulse,rcf
stage 4 smd 8 machines, awsrcf and toto
$CLI run --dataset smd --data-dir "$SMD" --out "$OUT" --machines s8 --models awsrcf,toto-4m
stage done
