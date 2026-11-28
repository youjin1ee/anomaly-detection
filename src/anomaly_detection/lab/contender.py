"""Competing GPU load: matrix multiplications with an adjustable duty cycle.

Needs torch with CUDA, so it runs in the serving engine's Python environment. It needs no
special privileges: it only shares the GPU, which time slices between processes. Memory use
is a few hundred MiB (`--size` squared times 2 bytes times three matrices).

    python -m anomaly_detection.lab.contender --duty 1.0 --seconds 300
"""

from __future__ import annotations

import argparse
import signal
import time


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anomaly-detection-contender", description=__doc__)
    p.add_argument("--duty", type=float, default=1.0, help="busy fraction of each cycle, 0 to 1")
    p.add_argument("--cycle", type=float, default=0.2, help="cycle length in seconds")
    p.add_argument("--size", type=int, default=4096, help="matrix side")
    p.add_argument("--seconds", type=float, default=0, help="stop after seconds, 0 = until SIGTERM")
    args = p.parse_args(argv)

    import torch  # imported late so --help works without it

    stop = False

    def _stop(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    a = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(args.size, args.size, device="cuda", dtype=torch.bfloat16)
    t_end = time.monotonic() + args.seconds if args.seconds > 0 else float("inf")
    while not stop and time.monotonic() < t_end:
        t0 = time.monotonic()
        busy_until = t0 + args.duty * args.cycle
        while time.monotonic() < busy_until and not stop:
            for _ in range(8):
                a @ b
            torch.cuda.synchronize()
        rest = t0 + args.cycle - time.monotonic()
        if rest > 0 and not stop:
            time.sleep(rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
