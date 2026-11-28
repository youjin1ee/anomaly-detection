#!/bin/bash
# Once a minute: load average, busy share of the cores in CPUSET, and every process that has
# the GPU open (a process that is not the lab's would show up here). Output: one line each.
CPUSET="${CPUSET:-12-19}"
lo="${CPUSET%-*}"; hi="${CPUSET#*-}"
busy() { awk -v lo="$lo" -v hi="$hi" '$1 ~ /^cpu[0-9]+$/ { n=substr($1,4)+0; if (n>=lo && n<=hi) { b+=$2+$3+$4+$7+$8; t+=$2+$3+$4+$5+$6+$7+$8 } } END { print b, t }' /proc/stat; }
read -r b0 t0 < <(busy)
while sleep 60; do
  read -r b1 t1 < <(busy)
  echo "$(date +%s) load $(cut -d' ' -f1-3 /proc/loadavg) cores_busy_pct $(( 100 * (b1 - b0) / (t1 - t0 + 1) ))"
  echo "$(date +%s) gpu_procs $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | tr '\n' ';')"
  b0=$b1; t0=$t1
done
