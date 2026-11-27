"""Load the Datadog BOOM benchmark (Apache-2.0) as input of the replay harness.

BOOM is a forecasting benchmark of observability metrics. It has no anomaly labels, so every
alert on it is a candidate false alarm. The data is never stored in this repository: the
loader takes the directory of a local copy.

Expected layout, as written by `datasets.Dataset.save_to_disk` per series:

    <root>/MANIFEST.csv                    dir, freq, num_variates, domain, metric_type, ...
    <root>/<dir>/data-00000-of-00001.arrow  one row: start, item_id, freq, target

`target` holds one list per variate (or one flat list for a single variate). Every variate
becomes its own univariate `RecordedSeries` on a regular grid starting at `start` (UTC).
Missing values (NaN) stay NaN and are skipped by the replay, never filled with zero.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from .replay import RecordedSeries


@dataclass(frozen=True)
class BoomMeta:
    freq: str  # "T" (1 minute) or "5T" (5 minutes)
    step: int  # seconds
    domain: str  # as in the manifest, "+" joins several
    metric_type: str  # count, distribution, gauge, rate


def freq_seconds(freq: str) -> int:
    """`T` is one minute, `5T` five minutes. Other pandas frequencies are not in the subset."""
    m = re.fullmatch(r"(\d*)T", freq)
    if not m:
        raise ValueError(f"unsupported BOOM frequency {freq!r}")
    return 60 * int(m.group(1) or 1)


def read_manifest(root: Path) -> list[dict[str, str]]:
    with open(root / "MANIFEST.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def read_target(path: Path) -> tuple[str, np.ndarray]:
    """Start timestamp text and the target as a (variates, points) float array."""
    with pa.memory_map(str(path)) as src:
        try:
            table = pa.ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:  # a file-format arrow file
            table = pa.ipc.open_file(src).read_all()
        row = table.slice(0, 1).to_pylist()[0]
    target = np.asarray(row["target"], dtype=float)
    if target.ndim == 1:
        target = target[None, :]
    target[~np.isfinite(target)] = np.nan
    return str(row["start"]), target


def load_boom(
    root: Path,
    max_days: float | None = None,
    dirs: list[str] | None = None,
    rebase_days: float | None = None,
) -> tuple[list[RecordedSeries], dict[str, BoomMeta]]:
    """All variates of the selected series. `max_days` keeps only the first days of each one
    (the replay length is then the same for every series). Returns the series and their
    metadata by `RecordedSeries.key`.

    The published BOOM values are z-scored per variate, so zero means nothing: an idle counter
    sits at its mode, not at 0. `rebase_days` subtracts the 1st percentile of the first days
    (the warmup, nothing from later) so that idle is about 0 again. That is a constant offset
    per series, which the scale and shift robust models do not see, and it restores the
    meaning of the replay's relative guards and of the models' floors relative to the level."""
    series: list[RecordedSeries] = []
    meta: dict[str, BoomMeta] = {}
    for row in read_manifest(root):
        if dirs is not None and row["dir"] not in dirs:
            continue
        step = freq_seconds(row["freq"])
        start, target = read_target(root / row["dir"] / "data-00000-of-00001.arrow")
        t0 = int(np.datetime64(start, "s").astype(np.int64))
        n = target.shape[1]
        if max_days is not None:
            n = min(n, int(max_days * 86400 / step))
        ts = t0 + step * np.arange(n, dtype=np.int64)
        for k in range(target.shape[0]):
            values = target[k, :n].copy()
            if rebase_days is not None:
                seen = values[: int(rebase_days * 86400 / step)]
                seen = seen[np.isfinite(seen)]
                if seen.size:
                    values -= float(np.quantile(seen, 0.01))
            s = RecordedSeries(row["dir"], json.dumps({"variate": k}), ts, values)
            series.append(s)
            meta[s.key] = BoomMeta(row["freq"], step, row["domain"], row["metric_type"])
    return series, meta
