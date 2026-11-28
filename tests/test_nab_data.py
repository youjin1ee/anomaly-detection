"""NAB loader."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from anomaly_detection.harness.nab_data import extract_nab_archive, load_corpus


def _write_corpus(root: Path) -> None:
    (root / "data" / "realTweets").mkdir(parents=True)
    (root / "labels").mkdir()
    rows = ["timestamp,value"]
    for i in range(30):
        rows.append(f"2015-01-01 {i // 12:02d}:{(i % 12) * 5:02d}:00,{i}.5")
    (root / "data" / "realTweets" / "x.csv").write_text("\n".join(rows) + "\n")
    (root / "labels" / "combined_windows.json").write_text(
        json.dumps(
            {"realTweets/x.csv": [["2015-01-01 01:00:00.000000", "2015-01-01 01:20:00.000000"]]}
        )
    )


def test_load_corpus(tmp_path):
    _write_corpus(tmp_path)
    corpus = load_corpus(tmp_path, expect_complete=False)
    assert len(corpus) == 1
    s = corpus[0]
    assert s.category == "realTweets" and s.name == "x"
    assert s.windows == ((12, 16),)
    assert s.values[0] == 0.5
    assert s.timestamps[1] - s.timestamps[0] == 300.0
    with pytest.raises(ValueError, match="expected 58"):
        load_corpus(tmp_path)


def test_load_corpus_bad_window_edge(tmp_path):
    _write_corpus(tmp_path)
    (tmp_path / "labels" / "combined_windows.json").write_text(
        json.dumps(
            {"realTweets/x.csv": [["2015-01-01 01:01:00.000000", "2015-01-01 01:20:00.000000"]]}
        )
    )
    with pytest.raises(ValueError, match="not a timestamp"):
        load_corpus(tmp_path, expect_complete=False)


def test_extract_archive_takes_only_wanted_files(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, body in {
            "NAB-abc/data/realTweets/x.csv": b"timestamp,value\n",
            "NAB-abc/labels/combined_windows.json": b"{}",
            "NAB-abc/LICENSE.txt": b"MIT",
            "NAB-abc/results/null/x.csv": b"big",
            "NAB-abc/data/../../evil.txt": b"x",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    buf.seek(0)
    extract_nab_archive(buf, tmp_path / "out")
    got = sorted(
        str(p.relative_to(tmp_path / "out")) for p in (tmp_path / "out").rglob("*") if p.is_file()
    )
    assert got == ["LICENSE.txt", "data/realTweets/x.csv", "labels/combined_windows.json"]
    assert not (tmp_path / "evil.txt").exists()
