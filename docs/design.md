# Design notes

Implementation choices of the first milestone and the evidence behind them.

## RCF implementation: rrcf

Requirement: installs cleanly on linux/arm64 and linux/amd64, licence MIT, BSD or Apache-2.0,
no JVM by default.

| candidate | result |
|---|---|
| AWS Java implementation (Apache-2.0) via a JPype wrapper | needs a JVM and a jar in the image, heavier image and one more moving part for air-gapped installs. Not implemented; the model registry leaves room for it as an optional extra |
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
