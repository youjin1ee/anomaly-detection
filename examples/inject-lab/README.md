# Injection lab

A real serving engine, real metrics and silent failures injected on a schedule, with ground
truth labels. It produces a dump that the harness models can be replayed on, so detection can
be compared with the synthetic results on data that has real noise.

```text
 load generator ---> vLLM (GPU) ---> /metrics --+
   (compressed day,   ^                         +--> VictoriaMetrics --> detector (sidecar)
    noise, lengths)   | competing GPU matmul     |         |
 injector -----------+  (throughput_drop)       |         +--> dump.csv + labels.json
   (writes labels.json)        nvidia-smi exporter +
```

Nothing here needs root. The engine, load generator, injector and GPU exporter are plain host
processes; the store and the detector run from `compose.yaml` with host networking.

## Faults

| scenario | how it is injected | signal that should show it |
|---|---|---|
| `request_drop` | arrival rate of the load generator x0.03 | `sum(rate(vllm:request_success_total[1m]))`, below |
| `output_length_drop` | output token cap 1 to 3, HTTP stays 200 | tokens per finished request, below |
| `throughput_drop` | competing matrix multiplication process on the same GPU (`lab.contender`, duty cycle 1.0) | tokens per second of decoding (inverse of inter token latency), below |
| `latency_rise` | 60 ms added before the first byte and before every received chunk on the client path | client side request duration, above |

Why these and not others:

- Power limits and clock locks need root (`nvidia-smi -pl`, `-lgc`). On the machine this was
  built on, power limit and clock lock were not even available to the user, so a competing
  process is the way to slow the GPU down without privileges.
- `tc netem` needs root too. The latency fault therefore sits in the client: it delays reading
  the response stream. It is a simulation of a slow path, not a slow engine. The engine side
  latency histograms do not move for this fault; the client side one does.
- A GPU competitor slows decoding for every request, so it also raises client latency. The
  replay reports alerts that fire during another scenario's window in a separate column.

## Compressed day

The load follows a daily shape (peak at 15:00 of the compressed day, a smaller second harmonic,
the same shape as the synthetic `request_rate`). One day is `DAY_SECONDS` of wall clock time,
default 1440 seconds (24 minutes, so one lab minute is one hour of the day).

Why 24 minutes: a seasonal baseline needs two or more periods of history, so the warm-up is
4 days (96 minutes), and a 5 minute fault is a fifth of a day instead of the ten to forty
minutes of the real day. Shorter days make the one minute `rate()` window a large part of the
day and blur the daily shape. The weekly pattern is not simulated.

To match a detector to the compressed day:

- Sidecar: `seasonality: [1440]` (seconds, not `daily`) and `interval: 5s`, which is 288
  samples per day (`config.yaml`). Keep `slot_window` times two plus two below the samples per
  day, and keep `backfill` shorter than the history you have.
- Harness: `anomaly-detection-replay` stretches lab time by `86400 / day_seconds` before the
  models see it, so every model runs with its normal daily period and default parameters.
  `anomaly-detection-eval` is for synthetic series and needs no change.
- If you change `DAY_SECONDS`, change `seasonality` in `config.yaml` to the same number.

## Run

Needs Docker with compose, and an environment with vLLM and torch (`ENGINE_BIN` is its `bin`
directory). vLLM and the model are not downloaded by the scripts: they must already exist, with
`HF_HUB_OFFLINE=1` if the machine should not reach the hub.

```sh
cd examples/inject-lab
ENGINE_BIN=$HOME/venvs/vllm/bin OUT=$HOME/lab-run ./run.sh
```

Defaults: 4 warm-up days, 12 minutes of quiet after warm-up, then 5 injections per scenario in
slots of 10 minutes (about 5.3 hours in all). `PER_SCENARIO`, `WARMUP_DAYS`, `DAY_SECONDS`,
`BASE_RATE` and `INJECTOR_ARGS` (see `anomaly-detection-injector --help`) change that. GPU
memory of the engine is capped by `GPU_UTIL` (default 0.2); keep it small on machines with
unified memory.

Output in `$OUT`: `labels.json` and `labels.csv` (scenario, start, end, expected signal and
metric), `signals.csv` and `raw.csv` (`timestamp,labels,value`), `logs/`.

Replay:

```sh
uv run anomaly-detection-replay --signals $OUT/signals.csv --labels $OUT/labels.json \
  --models fixed,seasonal,rcf,rcf-time --out report.md
```

Stop and clean up: `run.sh` stops what it started when it exits. The store volume
(`inject-lab_vmdata`) is kept so the dump can be repeated; remove it with
`docker compose -f compose.yaml down -v`.

## Not covered

- DCGM exporter and node exporter are not used. `lab.gpuexporter` reads `nvidia-smi` instead,
  and drops fields reported as N/A.
- One engine, one model, one GPU. Request rate and decode speed come from vLLM metrics.
- Faults never overlap and never ramp: each one starts and stops instantly.
