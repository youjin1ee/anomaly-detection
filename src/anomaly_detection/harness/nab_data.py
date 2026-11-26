"""Loader for the Numenta Anomaly Benchmark (NAB) corpus.

The data is not part of this repository. Point the loader at a directory that has the layout
of the NAB repository (`data/<category>/<file>.csv`, `labels/combined_windows.json`), or let
`fetch_nab` download exactly those files from GitHub into a directory you choose.
"""

from __future__ import annotations

import csv
import itertools
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np

CATEGORIES = (
    "artificialNoAnomaly",
    "artificialWithAnomaly",
    "realAdExchange",
    "realAWSCloudwatch",
    "realKnownCause",
    "realTraffic",
    "realTweets",
)

# The commit the results in docs/results/m2-nab.md were produced with.
NAB_REF = "ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"
NAB_ARCHIVE_URL = "https://codeload.github.com/numenta/NAB/tar.gz/" + NAB_REF
EXPECTED_FILES = 58
EXPECTED_WINDOWS = 116

_WANTED_PREFIXES = ("data/", "labels/combined_windows.json", "labels/combined_labels.json")
_WANTED_FILES = ("LICENSE.txt",)


@dataclass(frozen=True)
class NabSeries:
    path: str  # relative path as used by NAB, for example realTweets/Twitter_volume_AAPL.csv
    timestamps: np.ndarray  # float seconds since the epoch (naive timestamps read as UTC)
    values: np.ndarray
    windows: tuple[tuple[int, int], ...]  # labelled anomaly windows, inclusive sample indices

    @property
    def category(self) -> str:
        return self.path.split("/", 1)[0]

    @property
    def name(self) -> str:
        return self.path.split("/", 1)[1].removesuffix(".csv")


def _to_us(texts: list[str]) -> np.ndarray:
    return np.array([t.replace(" ", "T") for t in texts], dtype="datetime64[us]").astype(np.int64)


def load_series(root: Path, path: str, windows_raw: list[list[str]]) -> NabSeries:
    stamps: list[str] = []
    values: list[float] = []
    with open(root / "data" / path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[:2] != ["timestamp", "value"]:
            raise ValueError(f"{path}: unexpected header {header}")
        for row in reader:
            stamps.append(row[0])
            values.append(float(row[1]))
    us = _to_us(stamps)
    first_index: dict[int, int] = {}
    for i, t in enumerate(us.tolist()):
        first_index.setdefault(t, i)  # NAB looks windows up with list.index, the first match
    windows: list[tuple[int, int]] = []
    for start_text, end_text in windows_raw:
        start_us, end_us = (int(x) for x in _to_us([start_text, end_text]))
        try:
            windows.append((first_index[start_us], first_index[end_us]))
        except KeyError as exc:
            raise ValueError(f"{path}: window edge {exc} is not a timestamp of the file") from exc
    windows.sort()
    for (_, e1), (s2, _) in itertools.pairwise(windows):
        if s2 <= e1:
            raise ValueError(f"{path}: overlapping windows")
    return NabSeries(path, us / 1e6, np.array(values, dtype=float), tuple(windows))


def load_corpus(root: Path | str, expect_complete: bool = True) -> list[NabSeries]:
    """All series of the corpus, sorted by path. `expect_complete` checks the 58 files and 116
    windows of the NAB version this was written against."""
    root = Path(root)
    windows_file = root / "labels" / "combined_windows.json"
    if not windows_file.is_file():
        raise FileNotFoundError(f"{windows_file} not found; see `anomaly-detection-nab fetch`")
    windows = json.loads(windows_file.read_text())
    corpus = [load_series(root, path, windows[path]) for path in sorted(windows)]
    if expect_complete:
        n_windows = sum(len(s.windows) for s in corpus)
        if len(corpus) != EXPECTED_FILES or n_windows != EXPECTED_WINDOWS:
            raise ValueError(
                f"expected {EXPECTED_FILES} files and {EXPECTED_WINDOWS} windows, "
                f"found {len(corpus)} and {n_windows}"
            )
    return corpus


def extract_nab_archive(fileobj: IO[bytes], dest: Path) -> None:
    """Extract only data/, the label files and the licence from a NAB .tar.gz into `dest`.

    Paths are checked by hand: only regular files under the wanted prefixes are written.
    """
    dest = Path(dest)
    with tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = member.name.split("/", 1)
            if len(parts) != 2:
                continue
            rel = parts[1]
            if ".." in rel.split("/") or rel.startswith("/"):
                continue
            if not (rel in _WANTED_FILES or rel.startswith(_WANTED_PREFIXES)):
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def fetch_nab(dest: Path | str, url: str = NAB_ARCHIVE_URL) -> None:
    """Download the pinned NAB revision and unpack data, labels and licence into `dest`."""
    import tempfile

    import httpx

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as tmp:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes():
                tmp.write(chunk)
        tmp.seek(0)
        extract_nab_archive(tmp, dest)
