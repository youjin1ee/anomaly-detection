"""Minimal GPU exporter based on `nvidia-smi --query-gpu`, for machines where DCGM exporter
does not run (for example some ARM64 systems). Fields that report N/A are left out. Also
exports the host load average (`host_load1`, `host_load5`)."""

from __future__ import annotations

import argparse
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIELDS = {
    "utilization.gpu": ("gpu_utilization_percent", 1.0),
    "clocks.sm": ("gpu_sm_clock_mhz", 1.0),
    "temperature.gpu": ("gpu_temperature_celsius", 1.0),
    "power.draw": ("gpu_power_watts", 1.0),
    "memory.used": ("gpu_memory_used_bytes", 1048576.0),
}


def parse(csv_text: str) -> list[str]:
    """One line per GPU, columns in the order of FIELDS (header-less, no units)."""
    lines: list[str] = []
    for idx, line in enumerate(csv_text.strip().splitlines()):
        for name, raw in zip((m for m, _ in FIELDS.values()), line.split(","), strict=False):
            raw = raw.strip()
            try:
                value = float(raw)
            except ValueError:
                continue  # N/A and friends
            scale = next(s for m, s in FIELDS.values() if m == name)
            lines.append(f'{name}{{gpu="{idx}"}} {value * scale}')
    return lines


def load_average(path: str = "/proc/loadavg") -> list[str]:
    """Host load, because other jobs on a shared machine are a source of noise."""
    try:
        with open(path) as f:
            one, five, _ = f.read().split()[:3]
    except (OSError, ValueError):
        return []
    return [f"host_load1 {float(one)}", f"host_load5 {float(five)}"]


def collect() -> str:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout
    return "\n".join([*parse(out), *load_average()]) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anomaly-detection-gpuexporter", description=__doc__)
    p.add_argument("--listen", default="0.0.0.0:9202")
    args = p.parse_args(argv)
    host, _, port = args.listen.rpartition(":")

    class H(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            try:
                body = collect().encode()
                self.send_response(200)
            except (OSError, subprocess.SubprocessError):
                body = b""
                self.send_response(503)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: object) -> None:
            pass

    ThreadingHTTPServer((host or "0.0.0.0", int(port)), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
