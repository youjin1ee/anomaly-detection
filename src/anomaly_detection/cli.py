"""Command line entry point: `anomaly-detection run|validate --config FILE`."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from .config import load_config
from .exporter import make_server, serve_in_thread
from .prom import PromClient
from .runner import Runner

log = logging.getLogger("anomaly_detection")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anomaly-detection")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, help="path to the YAML configuration")
        p.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    try:
        config = load_config(args.config)
        # Fail on unknown model kinds or bad parameters before starting anything.
        from .models import create_model

        for d in config.detectors:
            create_model(d.model.kind, dict(d.model.params), config.interval_of(d))
    except Exception as exc:
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return 2
    if args.command == "validate":
        print(f"ok: {len(config.detectors)} detector(s)")
        return 0

    client = PromClient(config.datasource)
    runner = Runner(config, client)
    server = make_server(config.server.listen, runner)
    serve_in_thread(server)
    runner.start()
    log.info("listening on %s with %d detector(s)", config.server.listen, len(config.detectors))

    done = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())
    done.wait()
    log.info("shutting down")
    server.shutdown()
    runner.stop()
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
