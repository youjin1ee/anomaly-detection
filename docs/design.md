# Design notes

Implementation choices of the first milestone and the evidence behind them.

## RCF implementation: rrcf

Requirement: installs cleanly on linux/arm64 and linux/amd64, licence MIT, BSD or Apache-2.0,
no JVM by default.

| candidate | result |
|---|---|
| AWS Java implementation (Apache-2.0) via a JPype wrapper | needs a JVM and a jar in the image, heavier image and one more moving part for air-gapped installs. Implemented later as an optional model, see [AWS RCF (Java) integration](#aws-rcf-java-integration) |
| AWS Rust crate | no Python binding found in this investigation |
| PyPI `randomcutforest` | does not exist (404 on the PyPI JSON API) |
| `pyrcf` | unrelated project (robot control) |
| rrcf 0.4.4 (MIT, kLabUM) | pure Python, depends on numpy only |

Checked on the ARM64 test machine (Python 3.12, numpy 2.5.5):

- `uv pip install rrcf` builds from the sdist (pure Python, no compiler needed) and works.
- Import fails without `pkg_resources`. rrcf does not declare setuptools, and setuptools 81
  and newer do not ship `pkg_resources`, so the project pins `setuptools<81` and rrcf `==0.4.4`.
- Cost: about 3.4 ms per update with 40 trees, sample 256, 8 dimensions; about 1.5 ms with
  20 trees and sample 128. The evaluation measured about 5 ms per sample inside the harness.
- Duplicate points (a constant series) are handled; a test covers it.
- The release is from 2023 and the project is barely maintained. That is the main risk, and the
  reason the model sits behind the registry interface.

amd64: every compiled dependency (numpy, pydantic-core, PyYAML) has a manylinux x86_64 wheel in
`uv.lock`, and rrcf is pure Python. The amd64 image build itself was not run (the test machine
is arm64 only).

## Model design choices

- One interface, `update(timestamp, value) -> ModelOutput | None`. Direction, tolerance, the
  low-traffic floor and the warm-up guard are in `SeriesPipeline`, shared by all models.
- A model that cannot produce a band (RCF) returns `None` for lower and upper and the band
  series are simply not exported for it.
- Seasonal baseline: values at the same slot of earlier periods, pooled with neighbouring
  slots, then median and MAD. The longest period with enough points wins. Defaults came from a
  sweep on tuning seeds: a narrow pool and a small `band_k` gave many false alarms on the
  steepest signal (request rate), `band_k` 6 with `slot_window` 8 removed them without
  slowing detection.
- RCF: values are normalised with the trailing median and MAD, stacked into shingles, and the
  mean CoDisp over the trees is divided by its own recent 99th percentile and squashed. The
  model sees shape; after a failure has lasted long enough to fill the sample with the new
  level, its score decays. That limits detection to the start of a failure.
- `expected` of RCF is a trailing median, used only for direction, tolerance and the floor.

## Runtime choices

- Threads, one per detector, and the standard library HTTP server: few dependencies, enough
  for hundreds of series. Revisit if the load measurements say otherwise.
- Evaluation times are aligned to the interval grid so seasonal slots match the backfill.
- Exposition is built at scrape time from state. A series that has no fresh value is not
  exported; there is no stale gauge to clean up.
- State is not persisted. Backfill replays history at start. If the store returns nothing at
  start (for example freshly written data is not searchable yet), series that appear later get
  a history replay on a later evaluation (at most once per two intervals), so a model is not
  left cold.
- Detector-level limits (`max_series_per_detector`, `max_series_total`) are enforced at model
  creation and the number of skipped series is exported.

## Found while testing the compose demo

- Seed data without the labels that scraping adds made a second series that `sum()` doubled.
  The seed now carries the same `job` and `instance` labels.
- The first backfill returned nothing because the store had not made seeded data searchable yet.
  This led to the late backfill described above.

## AWS RCF (Java) integration

Optional model `awsrcf` (`models/awsrcf.py`, `sidecar/awsrcf/AwsRcfBridge.java`). Measurements are
in [docs/results/m3-poc-awsrcf.md](results/m3-poc-awsrcf.md).

### Where the library comes from

- Maven Central, group `software.amazon.randomcutforest`: `randomcutforest-core` and
  `randomcutforest-parkservices` 4.4.0 (Apache-2.0), plus the jackson jars that parkservices
  declares. They are not in the repository and not in the default install.
- The Rust crate in the same repository (`rcflib`, version 4.0.0 in its `Cargo.toml`) has no
  Python binding and no PyPI package. PyPI `randomcutforest` does not exist.
- The upstream `python_rcf_wrapper` directory calls the Java classes through JPype and ships
  4.0.0-SNAPSHOT jars. It is not a package either.

### How it is connected

| option | judgement |
|---|---|
| JPype, JVM in the Python process | extra native Python dependency (a wheel for each architecture), one crash or out-of-memory takes the sidecar down, JVM heap and Python heap share one process limit. One-step latency is about the same as the child process (see results) |
| Rust binding | would have to be written and built for two architectures. Not done |
| Java child process, line protocol on stdin and stdout (chosen) | no Python dependency; the JVM is a separate process with its own heap limit; one JVM serves all series of the Python process; `java` plus six jars, no compiler at run time |

The Python adapter keeps one shared child. `NEW`, `U`, `DEL` and `MEM` are the whole
protocol (see the header of `AwsRcfBridge.java`). The child ends when its stdin closes, so a
crashed or killed parent leaves no orphan. If the child dies, the next call raises and the next
model creation starts a new one; series state is lost, as after any restart.

"Optional extra" means the jars and the JVM, not a Python extra: nothing changes for a default
install, `import anomaly_detection.models` does not touch Java, and creating an `awsrcf` model
without `ANOMALY_AWSRCF_HOME` fails with a message that names the variable.

```sh
# on a machine with a JDK: jars from Maven Central into one directory, then
scripts/build-awsrcf-bridge.sh /path/to/awsrcf      # writes awsrcf-bridge.jar there
export ANOMALY_AWSRCF_HOME=/path/to/awsrcf ANOMALY_AWSRCF_JAVA=/path/to/jre/bin/java
export ANOMALY_AWSRCF_JAVA_OPTS="-Xmx2g -XX:+UseParallelGC -XX:ParallelGCThreads=2"
```

The runtime needs only the `java.base` module (checked with a `jlink` image). For a container,
that is the part to copy into the image.

### What the library gives

| class | output used here |
|---|---|
| `RandomCutForest` (core) | anomaly score of the sample before it is added. No threshold, no band |
| `ThresholdedRandomCutForest` | RCF score, anomaly grade (0 for normal, up to 1), the threshold it keeps (adapts to the score history, `anomalyRate` and `zFactor`). Expected values only when it flags an anomaly. No band |
| `RCFCaster` | the above plus a forecast of the next `forecastHorizon` samples with `lower` and `upper` per step, calibrated from the forecast errors seen so far (`SIMPLE`, `MINIMAL` or `NONE`) |

### Modes and the score mapping

- `caster` (default): the forecast made after sample t is the expected value and band of
  sample t+1, so `ModelOutput` carries `expected`, `lower` and `upper` for the sample being
  scored. Horizon 1. Until the first forecast exists the band is `None` and `expected` is a
  trailing median.
- `trcf`: no band, `expected` is the trailing median like `rcf`.
- `rcf`: plain forest. Raw score divided by `score_div` (default log2 of the sample size, the
  normalisation the NAB entry of AWS used) and capped to [0, 1].
- `trcf` and `caster` score: `logistic((raw / threshold - 1) / threshold_width)`, 0.5 at the
  threshold the library itself keeps. `score_from: grade` uses the anomaly grade instead.
  The first mapping fires at the library's own threshold and so far too often for a
  per-record alert (numbers in the results). It is kept because it is a fixed, documented rule
  that was not fitted to NAB; the grade variant is the alternative that was measured.
- Direction, tolerance and the low-traffic floor stay in `SeriesPipeline`, as for every model.

### Limits of this integration

- A series is a forest in the JVM heap, 0.9 to 1.2 MB at sample size 256, so memory, not time,
  limits how many series one child serves. Several children would be needed beyond that.
- Calls are serialised in one lock per child. Series are processed one after the other.
- State is not persisted: a restart replays the backfill, which costs about as much per sample
  as live scoring.
- Only linux/arm64 with Temurin 21 was run. amd64 was not.
