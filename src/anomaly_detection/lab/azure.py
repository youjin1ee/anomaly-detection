"""Azure LLM inference trace (2024) prepared for the load generator.

Source: Azure LLM inference trace 2024 (CC-BY), columns TIMESTAMP, ContextTokens,
GeneratedTokens. Cite: Stojkovic et al., DynamoLLM, HPCA 2025.

One week holds millions of requests per day, far above what one small engine serves, so
`prepare` keeps each request with the same probability (thinning). Arrival times keep the shape
of the original (daily cycle, weekend dip, bursts at the scale of the kept rate) and every kept
request keeps its own (context, generated) pair, so the joint length distribution is preserved.
Time is compressed by 86400 / day_seconds. Context lengths are multiplied by `ctx_scale` (the
engine's KV cache is far smaller than the original contexts) and both lengths are capped.

Also finds natural low traffic days (candidates for natural anomalies) from hourly counts.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

DAY = 86400.0
CHUNK = 500_000


def _hour_index(key: str, day0: datetime) -> int:
    """`key` is the first 13 characters of a timestamp, `YYYY-MM-DD HH`."""
    d = datetime.fromisoformat(key[:10])
    return (d - day0).days * 24 + int(key[11:13])


def prepare(
    csv_path: Path,
    out: Path,
    day_seconds: float,
    mean_rate: float,
    ctx_scale: float,
    ctx_cap: int,
    gen_cap: int,
    seed: int = 1,
) -> dict:
    """Write an npz with t (lab seconds from the start of trace day 0), ctx, gen, and the
    hourly request counts of the whole source. Returns the meta that is also stored."""
    with csv_path.open() as f:
        n_rows = sum(1 for _ in f) - 1
    rng = np.random.default_rng(seed)
    hours: Counter[str] = Counter()
    t_l: list[float] = []
    ctx_l: list[int] = []
    gen_l: list[int] = []
    day0 = None
    with csv_path.open() as f:
        next(f)
        while chunk := list(itertools.islice(f, CHUNK)):
            hours.update(line[:13] for line in chunk)
            if day0 is None:
                day0 = datetime.fromisoformat(chunk[0][:10]).replace(tzinfo=None)
            # keep probability is set from the target mean rate over the replayed week
            p = mean_rate * 7 * day_seconds / n_rows
            keep = rng.random(len(chunk)) < p
            for line, k in zip(chunk, keep, strict=True):
                if not k:
                    continue
                ts, c, g = line.rstrip("\n").split(",")
                dt = datetime.fromisoformat(ts)
                t_l.append((dt.replace(tzinfo=None) - day0).total_seconds() * day_seconds / DAY)
                ctx_l.append(int(c))
                gen_l.append(int(g))
    assert day0 is not None, "empty trace"
    order = np.argsort(t_l)
    t = np.asarray(t_l)[order]
    raw_ctx = np.asarray(ctx_l)[order]
    raw_gen = np.asarray(gen_l)[order]
    ctx = np.clip(np.round(raw_ctx * ctx_scale), 1, ctx_cap).astype(np.int32)
    gen = np.clip(raw_gen, 1, gen_cap).astype(np.int32)
    n_h = max(_hour_index(k, day0) for k in hours) + 1
    hourly = np.zeros(n_h)
    for k, n in hours.items():
        hourly[_hour_index(k, day0)] = n
    meta = {
        "source": csv_path.name,
        "day0": day0.isoformat(),
        "day_seconds": day_seconds,
        "source_rows": n_rows,
        "kept": len(t),
        "keep_prob": float(len(t) / n_rows),
        "mean_rate": mean_rate,
        "ctx_scale": ctx_scale,
        "ctx_cap": ctx_cap,
        "gen_cap": gen_cap,
        "ctx_clipped": float(np.mean(np.round(raw_ctx * ctx_scale) > ctx_cap)),
        "gen_clipped": float(np.mean(raw_gen > gen_cap)),
        "seed": seed,
    }
    np.savez(out, t=t, ctx=ctx, gen=gen, hourly=hourly, raw_ctx=raw_ctx, raw_gen=raw_gen)
    return meta


def low_spans(hourly: np.ndarray, ratio: float = 0.6) -> list[dict]:
    """Days whose request total is below `ratio` times the median day, merged into spans.

    The weekday pattern is not modelled here: the point is only to list stretches where the
    trace itself drops, so they can be labelled as natural anomaly candidates. Spans are in
    trace seconds from the start of day 0.
    """
    days = hourly[: len(hourly) // 24 * 24].reshape(-1, 24).sum(axis=1)
    med = float(np.median(days))
    low = days < ratio * med
    spans, i = [], 0
    while i < len(low):
        if not low[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(low) and low[j + 1]:
            j += 1
        spans.append(
            {
                "start": i * DAY,
                "end": (j + 1) * DAY,
                "days": [int(i), int(j)],
                "ratio_to_median_day": [round(float(days[k] / med), 3) for k in range(i, j + 1)],
            }
        )
        i = j + 1
    return spans


def band_windows(
    hourly: np.ndarray, first_day: int, last_day: int, width_hours: int = 8
) -> dict[str, tuple[float, float]]:
    """Time of day windows (fractions of the day) around the busiest and the quietest stretch of
    `width_hours`, from the mean hourly counts of trace days first_day..last_day-1."""
    prof = hourly[first_day * 24 : last_day * 24].reshape(-1, 24).mean(axis=0)
    ring = np.concatenate([prof, prof])
    sums = np.array([ring[h : h + width_hours].sum() for h in range(24)])
    out = {}
    for band, h in (("peak", int(sums.argmax())), ("trough", int(sums.argmin()))):
        out[band] = (h / 24, (h + width_hours) / 24)  # may exceed 1: wraps into the next day
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m anomaly_detection.lab.azure", description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="npz for loadgen --trace")
    p.add_argument("--day-seconds", type=float, default=3600)
    p.add_argument("--mean-rate", type=float, required=True, help="kept requests per lab second")
    p.add_argument("--ctx-scale", type=float, default=0.25)
    p.add_argument("--ctx-cap", type=int, default=1900)
    p.add_argument("--gen-cap", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)
    meta = prepare(
        a.csv, a.out, a.day_seconds, a.mean_rate, a.ctx_scale, a.ctx_cap, a.gen_cap, a.seed
    )
    z = np.load(a.out)
    meta["low_spans"] = low_spans(z["hourly"])
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
