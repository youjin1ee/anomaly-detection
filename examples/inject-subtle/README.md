# Subtle fault lab

The Azure trace lab (`../inject-azure`) with weekdays only, a big and a subtle variant of every
scenario, and many injections per day. Data: Azure LLM inference trace 2024, CC-BY, Stojkovic
et al., DynamoLLM, HPCA 2025 (read from disk, never copied into this repository).

## Steps

```sh
# 1. trace of weekdays: source days 3 to 6 (Monday to Thursday), 12 slots, repeats thinned anew
python -m anomaly_detection.lab.azure --csv AzureLLMInferenceTrace_code_1week.csv --out code-wk.npz \
  --day-seconds 3600 --mean-rate 2 --ctx-scale 0.2 --ctx-cap 1700 --gen-cap 256 \
  --days 3,4,5,6,3,4,5,6,3,4,5,6

# 2. competitor duty per time of day band (decode speed 0.7 to 0.85), writes calib-<band>.json
ENGINE_BIN=<vllm bin> TRACE=$PWD/code-wk.npz OUT=<calib dir> ./calibrate-throughput.sh

# 3. one collection: 3 warm-up days, 9 injection days, cut into 3 runs of 3 days
ENGINE_BIN=<vllm bin> TRACE=$PWD/code-wk.npz SIZES=sizes.json OUT=<dir>/collect ./run.sh

# 4. score and break down
python -m anomaly_detection.lab.subtle_report --runs <dir>/run1 <dir>/run2 <dir>/run3 \
  --out report.md --events-csv events.csv
```

`sizes.json` holds `{"throughput_drop:subtle": {"duty_by_band": {"peak": [lo, hi], "trough": [lo, hi]}}}`.

## What a run contains

`run<N>/signals.csv` (the three signals at 5 s), `labels.json` and `labels.csv` (per event:
scenario, `severity` gross or subtle, `band` peak or trough, `block`, planned `target` range,
measured `effect`, `in_target`, `level_before`), `scores/` (see below). A collection is cut into
runs by `slice_run`: warm-up plus one block of injection days, blocks moved back by whole days.
The runs share the warm-up, so fixed limits are identical in all of them.

## Score cache

`run<N>/scores/<model>.<signal>.npy`: float64 array per model and signal, aligned to the signal
grid of `signals.csv` (`ts = (first_index + arange(n)) * step` seconds, with `first_index`, `n`,
`step`, the time stretch and the alert rule in `scores/index.json`). Values in [0, 1], NaN where
the model has no score yet. Models saw the time stretched by 24 (a lab day of 3600 s as a real day
of 86400 s) with their default parameters. Alert rule: score of 0.5 or more for 3 samples in a row.
Signals: `request_rate`, `output_tokens`, `decode_speed` (`lab/signals.py`).
