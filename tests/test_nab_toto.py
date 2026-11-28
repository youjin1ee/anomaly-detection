"""Toto 2.0 on NAB: the batch path gives the stream path's scores."""

import numpy as np

from .test_toto import FakeEngine, series


def test_nab_batch_path_equals_stream_path_with_duplicates_and_gaps():
    from anomaly_detection.harness.nab_data import NabSeries
    from anomaly_detection.harness.nab_toto import batch_series, fed_indices, stream_series

    x = series(700, seed=5)
    ts = np.arange(700) * 300.0
    ts[300] = ts[299]  # duplicate timestamp, skipped
    ts[450] = ts[440]  # backward step, skipped
    x[200] = np.nan  # missing value, skipped
    s = NabSeries("realTraffic/x.csv", ts, x, ((600, 650),))
    assert 200 not in fed_indices(s) and 300 not in fed_indices(s) and 450 not in fed_indices(s)
    batch, us = batch_series(FakeEngine(), s, batch_size=50)
    stream = stream_series(FakeEngine(), s)
    assert us > 0
    np.testing.assert_allclose(batch, stream, rtol=1e-5, atol=1e-6, equal_nan=True)
    assert np.isnan(batch[300]) and np.isnan(batch[200])
    assert np.isfinite(batch[169:]).sum() > 400
