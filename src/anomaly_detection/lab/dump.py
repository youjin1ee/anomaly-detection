"""Dump signals and raw metrics from a Prometheus-compatible store to CSV.

Writes two files into --out-dir, both with the columns timestamp,labels,value:

  signals.csv  the PromQL of `signals.SIGNALS` evaluated with query_range at --step, labels
               hold `signal="<name>"` so a replay can pick them up
  raw.csv      raw samples of the vllm, lab, gpu and anomaly metric families, from the
               VictoriaMetrics /api/v1/export endpoint (not available on plain Prometheus,
               use --no-raw there)
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path

from .signals import RAW_MATCH, SIGNALS


def _get(url: str, params: dict[str, str]) -> bytes:
    q = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{q}", timeout=600) as r:
        return r.read()


def labels_text(metric: dict[str, str]) -> str:
    return ",".join(f'{k}="{v}"' for k, v in sorted(metric.items()))


def dump_signals(base: str, out: Path, start: float, end: float, step: int) -> int:
    n = 0
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "labels", "value"])
        for sig in SIGNALS:
            body = _get(
                f"{base}/api/v1/query_range",
                {"query": sig.query, "start": str(start), "end": str(end), "step": str(step)},
            )
            data = json.loads(body)
            if data.get("status") != "success":
                raise RuntimeError(f"{sig.name}: {data}")
            for series in data["data"]["result"]:
                lab = labels_text({**series["metric"], "signal": sig.name})
                for ts, v in series["values"]:
                    w.writerow([ts, lab, v])
                    n += 1
    return n


def dump_raw(base: str, out: Path, start: float, end: float) -> int:
    n = 0
    body = _get(
        f"{base}/api/v1/export",
        {"match[]": RAW_MATCH, "start": str(start), "end": str(end)},
    )
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "labels", "value"])
        for line in body.splitlines():
            if not line:
                continue
            s = json.loads(line)
            lab = labels_text(s["metric"])
            for t_ms, v in zip(s["timestamps"], s["values"], strict=True):
                w.writerow([t_ms / 1000.0, lab, v])
                n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anomaly-detection-dump", description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8428")
    p.add_argument("--start", type=float, required=True, help="unix seconds")
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--no-raw", action="store_true")
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = dump_signals(args.url, args.out_dir / "signals.csv", args.start, args.end, args.step)
    print(f"signals.csv: {n} rows")
    if not args.no_raw:
        n = dump_raw(args.url, args.out_dir / "raw.csv", args.start, args.end)
        print(f"raw.csv: {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
