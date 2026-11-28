# Azure trace lab

The injection lab (`../inject-lab`) with a real traffic shape: the load generator replays a
prepared slice of the Azure LLM inference trace 2024 instead of a synthetic daily curve.

Data: Azure LLM inference trace 2024, CC-BY, from the Azure public dataset repository. Cite
Stojkovic et al., DynamoLLM, HPCA 2025. The trace is read from disk and never copied into
this repository.

## Prepare a trace

```sh
python -m anomaly_detection.lab.azure --csv AzureLLMInferenceTrace_code_1week.csv \
  --out code.npz --day-seconds 3600 --mean-rate 3 --ctx-scale 0.2 --ctx-cap 1700 --gen-cap 256
```

- `--mean-rate`: requests per lab second averaged over the week. Each source request is kept
  with the same probability, so the daily shape, the weekend dip and the joint distribution of
  context and generated tokens are those of the trace. `calibrate.sh` shows what an engine does
  at a rate.
- `--day-seconds`: one trace day lasts this long. 3600 means the week takes 7 hours.
- `--ctx-scale`, `--ctx-cap`, `--gen-cap`: contexts are multiplied by the scale (the engine's KV
  cache is far smaller than the original contexts) and both lengths are capped. The prepared
  file stores the share of requests that hit a cap in `code.json`.
- The same step lists days far below the median day (`low_spans` in `code.json`). They are
  written to the label file as natural anomaly candidates, apart from the injected events.

## Run

```sh
ENGINE_BIN=$HOME/venvs/vllm/bin TRACE=$PWD/code.npz OUT=$HOME/azure-run CPUSET=12-19 ./run.sh
```

Differences from `../inject-lab/run.sh`:

- Scenarios are `request_drop`, `output_length_drop`, `throughput_drop`; `latency_rise` is left
  out (it was only a client side simulation and no model detected it).
- Events are placed by time of day: every scenario once in the peak band and once in the
  trough band of each injection day (bands from the trace, 8 hours each). 3 warm-up days, then
  4 injection days: 8 events per scenario.
- Engine, load generator, exporter, host monitor, injector and both containers are pinned to
  `CPUSET`. `hostmon.sh` logs the load average, the busy share of those cores and every process
  with the GPU open, once a minute (`logs/host.log`).
- The compose project is `inject-azure` (own volume). The detector image is the one built by
  `../inject-lab`; this lab only runs it with `config.yaml` (`seasonality: [3600]`).

## Score a dump

```sh
python -m anomaly_detection.lab.azure_report --signals $OUT/signals.csv --labels $OUT/labels.json \
  --raw $OUT/raw.csv --cache-dir $OUT/scores --toto-dir <toto checkpoints>/4m \
  --tspulse-model <tspulse dir> --out report.md
```

`awsrcf` models need `ANOMALY_AWSRCF_HOME` (see `docs/design.md`). Scores are cached per model
and signal in `--cache-dir`, so the tables can be rendered again without scoring.
