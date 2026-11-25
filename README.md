# anomaly-detection

A small sidecar that adds anomaly scores and expected bands to any metric store that speaks the
Prometheus HTTP API (Prometheus, VictoriaMetrics, Thanos, Mimir, Cortex and friends).

It reads PromQL results, keeps one model per time series, and exposes the result as ordinary
Prometheus metrics. Your store scrapes it, and you alert on `anomaly_score` the same way you
alert on anything else. There is no new storage, no query language and no dashboard to adopt.

The case it is built for is the silent failure: HTTP 200 everywhere, latency fine, and yet the
output length collapsed, traffic stopped arriving, or a training job runs at half speed. A fixed
threshold cannot see those, because the value that matters is "lower than usual for this time
of day", not "lower than N".

Status: first milestone, usable but young. See [Limitations](#limitations).

## How it works

```text
 Prometheus-compatible API          anomaly-detection                    your store
 (query_range at start,    --->  one model per series      --->   scrapes /metrics
  instant query every            score, expected, band             alerts on anomaly_score
  interval)
```

1. At start, each detector replays `backfill` of history with `query_range`, so models learn
   before the first score is exported. Nothing is stored on disk: a restart repeats the
   backfill.
2. Every `interval`, each detector runs its PromQL as an instant query and feeds one sample per
   returned series to that series' model.
3. A failed query or an empty result feeds nothing. It is never turned into a zero, and a
   series without a fresh value stops being exported instead of reporting a stale score.
4. The model returns a score between 0 and 1, an expected value, and (for models that have one)
   a band. Direction, tolerance and a low-traffic floor are applied on top, the same way for
   every model.

Models:

| kind | what it does | when to use |
|---|---|---|
| `seasonal` | median and MAD of the same time of day (and week) in the past, pooled over neighbouring slots | default; cheap, works after two days of history (defaults tuned on synthetic data), has a band |
| `rcf` | Random Cut Forest ([rrcf](https://github.com/kLabUM/rrcf)) over shingled, normalised values | shape changes that a band does not describe; heavier, no band |
| `static` | fixed lower and upper limit | hard limits, and as a comparison baseline |

New model kinds register with `anomaly_detection.models.register_model(kind, factory)`. A model
only has to implement `update(timestamp, value) -> ModelOutput | None`.

## Quick start

Docker compose, with a synthetic metric generator and 21 days of seeded history in
VictoriaMetrics (add `--profile prometheus` for a Prometheus variant):

```sh
cd examples/compose
docker compose up --build
curl -s localhost:8080/metrics | grep '^anomaly_score'

# silently break the generator for 15 minutes, then look at the scores again
curl -X POST 'localhost:9100/fault?kind=request_drop&seconds=900'
```

From a checkout, against your own store:

```sh
uv sync
uv run anomaly-detection validate --config examples/config.example.yaml
uv run anomaly-detection run --config my-config.yaml
```

Container image (multi-arch Dockerfile, non-root, Python 3.12 slim):

```sh
docker buildx build --platform linux/amd64,linux/arm64 -t anomaly-detection .
docker run --rm -v $PWD/my-config.yaml:/etc/anomaly-detection/config.yaml:ro -p 8080:8080 anomaly-detection
```

As a sidecar in Kubernetes: [`examples/kubernetes/sidecar.yaml`](examples/kubernetes/sidecar.yaml).
The same container also runs standalone; point `datasource.url` at the store's Service.

## Configuration

One YAML file, one data source, any number of detectors. The full annotated example is
[`examples/config.example.yaml`](examples/config.example.yaml).

```yaml
datasource:
  url: http://prometheus:9090
  path_prefix: ""            # for example /select/0/prometheus (VictoriaMetrics cluster)
  headers: {}                # for example X-Scope-OrgID for Mimir
  auth: {type: none}         # none | bearer | basic
  tls: {}                    # ca_file, cert_file, key_file, insecure_skip_verify

defaults:
  interval: 60s
  backfill: 1d

detectors:
  - name: request_rate_drop
    query: sum(rate(http_requests_total[5m]))
    direction: below         # above | below | both
    min_expected: 1          # low-traffic floor, see below
    tolerance: {ratio: 0.05} # ignore deviations within 5 percent of expected
    model:
      kind: seasonal
      seasonality: [daily, weekly]
      band_k: 6
```

Notes:

- Write aggregated queries (`sum`, `avg`, `max`, ...). The detector keeps one model per returned
  series, so a query that returns thousands of series is expensive. `max_series_per_detector`
  (default 100) and `max_series_total` (default 500) cap it. Series over the cap are skipped
  and counted in `anomaly_detector_series_dropped`, never silently.
- `min_expected` protects low-traffic series: a downward move is ignored when the expected
  value is below it. A fall from a large expected value to zero is still scored, because the
  expected value is large.
- `tolerance` ignores small deviations: `amount` is absolute, `ratio` is relative to the
  expected value.
- `warmup_samples` adds a minimum number of samples before any score is exported. Models also
  have their own readiness rules; the seasonal model needs two days of history for a daily
  pattern and two weeks for the weekly one.
- Durations are seconds or `30s`, `5m`, `6h`, `7d`, `2w`. `${VAR}` is expanded in the
  `datasource` section only, for secrets.
- `interval` is also the spacing of model samples. Backfill uses the same step, so a long
  backfill at a short interval means many samples (10000 per request, chunked automatically).
  RCF costs a few milliseconds per sample and needs `sample_size` (default 1024) samples before it scores, so keep its backfill short or its interval longer.

## Output metrics

On `/metrics`, per series (labels: `detector` plus the labels of the original series; an
original label named `detector` is exported as `exported_detector`):

| metric | meaning |
|---|---|
| `anomaly_score` | 0 normal to 1 certainly anomalous |
| `anomaly_value` | the value that was scored |
| `anomaly_expected` | expected value according to the model |
| `anomaly_upper_band`, `anomaly_lower_band` | expected band (models that have one) |
| `anomaly_detector_ready` | 1 when the series is scored, 0 while warming up or without a fresh value |

While a series is warming up, only `anomaly_detector_ready 0` is exported: no score, no zeros.

Detector level (label `detector`):

| metric | meaning |
|---|---|
| `anomaly_detector_managed_series` | series with a model |
| `anomaly_detector_series_dropped` | series skipped in the last evaluation because of the limits |
| `anomaly_detector_query_failures_total` | failed queries |
| `anomaly_detector_query_empty_total` | queries that returned nothing |
| `anomaly_detector_evaluations_total` | evaluations run |
| `anomaly_detector_evaluation_lag_seconds` | scheduled time to completion |
| `anomaly_detector_evaluation_duration_seconds` | duration of the last evaluation |
| `anomaly_detector_last_success_timestamp_seconds` | evaluation time of the last successful query |
| `anomaly_detector_backfill_complete`, `anomaly_detector_backfill_samples` | startup backfill |

Process metrics from `prometheus_client` are included. `/healthz` is 200 while all detector
threads are alive. `/readyz` is 200 once every detector finished its startup backfill.

## Alerting

Make the store scrape `:8080/metrics`, then alert on the score. The score is 0.5 at the band
edge for the seasonal model, so a threshold between 0.7 and 0.9 with a `for` clause is a
reasonable start.

```yaml
groups:
  - name: anomaly-detection
    rules:
      - alert: MetricAnomaly
        expr: anomaly_score{detector="request_rate_drop"} >= 0.8
        for: 3m
        labels: {severity: warning}
        annotations:
          summary: "{{ $labels.detector }} is outside its expected range"

      # The detector itself must not fail silently.
      - alert: AnomalyDetectorNoScore
        expr: absent_over_time(anomaly_score{detector="request_rate_drop"}[15m])
        for: 5m
      - alert: AnomalyDetectorQueryFailing
        expr: increase(anomaly_detector_query_failures_total[10m]) > 3
      - alert: AnomalyDetectorBehind
        expr: anomaly_detector_evaluation_lag_seconds > 120
      - alert: AnomalyDetectorSeriesDropped
        expr: anomaly_detector_series_dropped > 0
```

Band form, if you prefer to see the value leave the band: `anomaly_value < anomaly_lower_band`.

## Evaluation

`anomaly-detection-eval` replays synthetic series with injected silent failures (request volume
collapse, output length collapse, throughput drop with step time increase) through the models
and reports event-level precision, recall, F1, detection delay and false alarms per day, with a
fixed threshold as comparison. Definitions are in [docs/evaluation.md](docs/evaluation.md) and
the numbers of the latest run in [docs/results/](docs/results/).

```sh
uv run anomaly-detection-eval --models fixed,seasonal,rcf --variants 20 --workers 8
```

## Limitations

- Models learn from what they see. A failure that lasts long enough becomes the new normal;
  the sliding window of RCF and the median over past periods of the seasonal model bound how
  long that takes, and detection delay is part of the evaluation for that reason.
- The seasonal model keeps no trend. A fast level change (a launch, a migration) shows up as an
  anomaly until the history catches up.
- Weekly patterns need about two weeks of history before the weekly estimate is used; before
  that the daily estimate is used and weekends can look anomalous.
- RCF is pure Python (rrcf), a few milliseconds per sample and per series. Hundreds of series
  at a one minute interval are fine; a long backfill for each of them is slow. It has no band.
- rrcf imports `pkg_resources`, so the project pins `setuptools<81`.
- One series, one model: no multivariate detection and no root cause analysis.
- Forecast-band models (statsforecast and similar) are not included; the model registry is the
  place to add them.
- State is not persisted. After a restart a series is silent until its backfill is replayed.
- A series that disappears is not reported as an anomaly. Use `absent_over_time` on the
  original metric or on `anomaly_score`, as shown above.
- Scores are not calibrated probabilities. Thresholds need a look at your own data, and
  `anomaly-detection-eval` is there to help with that.
- Only the Prometheus HTTP API is read. Stores that do not implement `query_range` and
  `query` are not supported.
