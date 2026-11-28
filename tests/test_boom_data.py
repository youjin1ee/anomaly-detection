import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")


def _write_boom(root, name, freq, start, target):
    """One BOOM series the way `datasets.save_to_disk` stores it: a single row, arrow stream."""
    d = root / name
    d.mkdir(parents=True)
    table = pa.table({"start": [start], "item_id": [name], "freq": [freq], "target": [target]})
    sink = pa.OSFile(str(d / "data-00000-of-00001.arrow"), "wb")
    with sink, pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)


def _boom_dir(tmp_path):
    rows = [
        ("ds-1-5T", "5T", 3, "Database", "gauge"),
        ("ds-2-T", "T", 1, "Networking", "rate"),
    ]
    (tmp_path / "MANIFEST.csv").write_text(
        "dir,freq,num_variates,domain,metric_type\n"
        + "\n".join(f"{d},{f},{n},{dom},{mt}" for d, f, n, dom, mt in rows)
        + "\n"
    )
    three = [[float(i + k) for i in range(600)] for k in range(3)]
    three[1][5] = float("nan")
    _write_boom(tmp_path, "ds-1-5T", "5T", "2024-06-01T00:05:00", three)
    _write_boom(tmp_path, "ds-2-T", "T", "2024-06-01T00:01:00", [float(i) for i in range(300)])
    return tmp_path


def test_load_boom_variates_grid_and_missing(tmp_path):
    from anomaly_detection.harness.boom_data import load_boom

    series, meta = load_boom(_boom_dir(tmp_path))
    assert [s.query for s in series] == ["ds-1-5T"] * 3 + ["ds-2-T"]  # one series per variate
    five, one = series[1], series[3]
    assert five.labels == '{"variate": 1}' and len(five.values) == 600
    assert np.isnan(five.values[5]) and np.isfinite(five.values[4])  # a gap stays NaN
    assert np.diff(five.ts).tolist() == [300] * 599 and np.diff(one.ts).tolist() == [60] * 299
    assert five.ts[0] == 1717200300  # 2024-06-01T00:05:00 UTC
    assert meta[five.key].step == 300 and meta[one.key].metric_type == "rate"
    assert meta[five.key].domain == "Database"


def test_load_boom_max_days_dirs_and_rebase(tmp_path):
    from anomaly_detection.harness.boom_data import load_boom

    root = _boom_dir(tmp_path)
    series, _ = load_boom(root, max_days=0.5, dirs=["ds-2-T"])  # 0.5 day is 720 minutes
    assert len(series) == 1 and len(series[0].values) == 300  # shorter than the cap
    series, _ = load_boom(root, max_days=1.0, dirs=["ds-1-5T"], rebase_days=0.5)
    s = series[0]
    assert len(s.values) == 288  # one day of five minutes
    # the 1st percentile of the first 144 samples (0..143) is 1.43, taken from the warmup only
    assert s.values[0] == pytest.approx(-1.43)
    assert s.values[1] - s.values[0] == pytest.approx(1.0)


def test_freq_seconds():
    from anomaly_detection.harness.boom_data import freq_seconds

    assert freq_seconds("T") == 60 and freq_seconds("5T") == 300
    with pytest.raises(ValueError):
        freq_seconds("H")
