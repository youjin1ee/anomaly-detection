"""Loaders for the public multivariate benchmarks SMD and PSM.

The data is not part of this repository. Both loaders take the directory of the dataset as an
argument: SMD has `train/`, `test/`, `test_label/` and `interpretation_label/` with one
`machine-*.txt` per server, PSM has `train.csv`, `test.csv` and `test_label.csv`.

Each machine is one labelled unit: a (n, d) matrix of dimensions plus one 0/1 label per test
row. Time is the row index times `step` seconds (SMD has no timestamps, its documentation
says one sample per minute; PSM gives the minute index itself).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STEP_SECONDS = 60.0


@dataclass(frozen=True)
class Machine:
    name: str
    train: np.ndarray  # (n_train, d) float64, NaN where missing
    test: np.ndarray  # (n_test, d)
    labels: np.ndarray  # (n_test,) int8
    train_ts: np.ndarray  # seconds, strictly increasing, all earlier than test_ts
    test_ts: np.ndarray
    # interpretation: (start, end_exclusive, dims) with 0-based test rows and dims
    interpretation: tuple[tuple[int, int, tuple[int, ...]], ...] = field(default=())

    @property
    def dims(self) -> int:
        return self.train.shape[1]

    def constant_dims(self) -> list[int]:
        """Dimensions without any variance over train and test together (NaN ignored)."""
        both = np.concatenate([self.train, self.test])
        return [j for j in range(self.dims) if np.nanstd(both[:, j]) == 0.0]


def label_events(labels: np.ndarray) -> list[tuple[int, int]]:
    """Runs of 1 as (first, last) inclusive row indices."""
    d = np.diff(np.concatenate([[0], np.asarray(labels, dtype=np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(int(s), int(e) - 1) for s, e in zip(starts, ends, strict=True)]


def _read_matrix(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, delimiter=",", dtype=np.float64))


def parse_interpretation(text: str) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """`start-end:d1,d2,...` per line. start and end are 0-based rows with end exclusive, which
    is what matches the label runs (checked on the files, see docs/results/m4-smd-psm.md);
    dimensions are 1-based in the file and 0-based here."""
    out = []
    for line in text.strip().splitlines():
        rng, dims = line.split(":")
        a, b = (int(x) for x in rng.split("-"))
        out.append((a, b, tuple(int(x) - 1 for x in dims.split(","))))
    return tuple(out)


def load_smd(root: str | Path, names: list[str] | None = None) -> list[Machine]:
    root = Path(root)
    files = sorted(p.name for p in (root / "test").glob("machine-*.txt"))
    if names is not None:
        missing = set(names) - {f.removesuffix(".txt") for f in files}
        if missing:
            raise ValueError(f"unknown SMD machines {sorted(missing)}")
        files = [f"{n}.txt" for n in names]
    out = []
    for f in files:
        train = _read_matrix(root / "train" / f)
        test = _read_matrix(root / "test" / f)
        labels = np.loadtxt(root / "test_label" / f, dtype=np.int8)
        if len(labels) != len(test) or train.shape[1] != test.shape[1]:
            raise ValueError(f"{f}: label or dimension mismatch")
        interp_file = root / "interpretation_label" / f
        interp = parse_interpretation(interp_file.read_text()) if interp_file.exists() else ()
        n0 = len(train)
        out.append(
            Machine(
                f.removesuffix(".txt"),
                train,
                test,
                labels,
                np.arange(n0) * STEP_SECONDS,
                np.arange(n0, n0 + len(test)) * STEP_SECONDS,
                interp,
            )
        )
    return out


def _read_psm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as fh:
        rows = list(csv.reader(fh))
    body = rows[1:]
    ts = np.array([float(r[0]) for r in body]) * STEP_SECONDS
    vals = np.array([[float(x) if x != "" else np.nan for x in r[1:]] for r in body])
    return ts, vals


def load_psm(root: str | Path) -> list[Machine]:
    """The whole of PSM is one unit (the data is pooled server metrics, no server id)."""
    root = Path(root)
    tr_ts, train = _read_psm(root / "train.csv")
    te_ts, test = _read_psm(root / "test.csv")
    lab_ts, lab = _read_psm(root / "test_label.csv")
    if not np.array_equal(te_ts, lab_ts):
        raise ValueError("PSM test and test_label timestamps differ")
    # The last train row and the first test row carry the same minute index. The test row
    # is kept (it has a label), the train row is dropped.
    keep = tr_ts < te_ts[0]
    return [Machine("psm", train[keep], test, lab[:, 0].astype(np.int8), tr_ts[keep], te_ts)]
