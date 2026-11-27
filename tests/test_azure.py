import random
import threading
from http.server import ThreadingHTTPServer

import numpy as np

from anomaly_detection.lab import azure, loadgen

from .test_lab import _args, _Engine


def _trace_csv(path, days=3, per_hour=40):
    """Day 1 has a quarter of the traffic. Every request has context 1000 and 20 generated."""
    lines = ["TIMESTAMP,ContextTokens,GeneratedTokens"]
    for d in range(days):
        n = per_hour // 4 if d == 1 else per_hour
        for h in range(24):
            for i in range(n):
                sec = i * 3600 / n
                lines.append(
                    f"2024-05-{10 + d:02d} {h:02d}:{int(sec // 60):02d}:{int(sec % 60):02d}"
                    ".000000+00:00,1000,20"
                )
    path.write_text("\n".join(lines) + "\n")


def test_prepare_thins_compresses_scales_and_caps(tmp_path):
    csv_path = tmp_path / "t.csv"
    _trace_csv(csv_path)
    out = tmp_path / "t.npz"
    # the keep probability is for 7 replayed days; the file holds 3 (2160 rows), so it is 0.58
    meta = azure.prepare(csv_path, out, 3600, 0.05, 0.25, 200, 10, seed=1)
    z = np.load(out)
    assert abs(meta["keep_prob"] - 0.05 * 7 * 3600 / 2160) < 0.05
    assert (np.diff(z["t"]) >= 0).all() and z["t"].max() < 3 * 3600
    assert (z["ctx"] == 200).all()  # 1000 x 0.25 = 250, capped at 200
    assert (z["gen"] == 10).all()  # 20 capped at 10
    assert meta["ctx_clipped"] == 1.0 and meta["gen_clipped"] == 1.0
    assert z["hourly"].shape == (72,) and z["hourly"][0] == 40
    # a day of 24 compressed hours: 3600 s of trace day 1 is 150 lab seconds of hour 0
    assert (z["t"][(z["t"] >= 3600) & (z["t"] < 3750)] < 3750).all()


def test_low_spans_finds_days_far_below_the_median():
    hourly = np.concatenate([np.full(24, 100.0), np.full(24, 20.0), np.full(24 * 3, 100.0)])
    (sp,) = azure.low_spans(hourly)
    assert sp["days"] == [1, 1] and sp["start"] == 86400 and sp["end"] == 2 * 86400
    assert sp["ratio_to_median_day"] == [0.2]
    assert azure.low_spans(np.full(72, 5.0)) == []


def test_band_windows_wrap_around_the_busiest_and_quietest_hours():
    day = np.ones(24)
    day[18:20] = 10
    day[6:8] = 0.1
    w = azure.band_windows(np.tile(day, 5), 1, 5, width_hours=8)
    pk, tr = w["peak"], w["trough"]
    assert pk[0] * 24 <= 18 and pk[1] * 24 >= 20
    assert tr[0] * 24 <= 6 and tr[1] * 24 >= 8


def test_trace_request_sends_token_ids_with_exact_lengths_and_obeys_faults():
    engine = ThreadingHTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=engine.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{engine.server_address[1]}"
    state, rng = loadgen.State(), random.Random(1)
    _Engine.bodies.clear()
    loadgen.one_request(url, "m", rng, state, _args(), shape=(37, 9))
    body = _Engine.bodies[-1]
    assert len(body["prompt"]) == 37 and body["max_tokens"] == 9 and body["ignore_eos"] is True
    state.fault = {"kind": "output_drop"}
    loadgen.one_request(url, "m", rng, state, _args(), shape=(37, 9))
    assert _Engine.bodies[-1]["max_tokens"] <= 3
    engine.shutdown()
    engine.server_close()


def test_replay_trace_sends_at_trace_times_and_request_drop_thins_arrivals(tmp_path):
    import argparse
    import time

    engine = ThreadingHTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=engine.serve_forever, daemon=True).start()
    npz = tmp_path / "t.npz"
    np.savez(npz, t=0.1 + np.arange(20) * 0.05, ctx=np.full(20, 5), gen=np.full(20, 2))
    ns = argparse.Namespace(
        trace=npz,
        epoch=time.time(),
        target=f"http://127.0.0.1:{engine.server_address[1]}",
        model="m",
        max_inflight=100,
        **{k: v for k, v in vars(_args()).items()},
    )
    state = loadgen.State()
    _Engine.bodies.clear()
    loadgen.replay_trace(ns, state, random.Random(1))
    time.sleep(0.5)
    assert len(_Engine.bodies) == 20
    state.fault = {"kind": "request_drop", "factor": 0.0}
    ns.epoch = time.time()
    _Engine.bodies.clear()
    loadgen.replay_trace(ns, state, random.Random(1))
    time.sleep(0.3)
    assert _Engine.bodies == []
    engine.shutdown()
    engine.server_close()
