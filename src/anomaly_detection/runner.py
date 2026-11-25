"""Detector scheduling: backfill at start, then one instant query per interval."""

from __future__ import annotations

import logging
import math
import threading
import time
import zlib
from dataclasses import dataclass, field

from .config import Config, DetectorConfig
from .models import create_model
from .pipeline import SeriesPipeline
from .prom import InstantSample, LabelKey, PromClient, QueryError

log = logging.getLogger(__name__)


@dataclass
class DetectorState:
    cfg: DetectorConfig
    interval: float
    backfill: float
    max_series: int
    stale_after: float
    pipelines: dict[LabelKey, SeriesPipeline] = field(default_factory=dict)
    last_seen: dict[LabelKey, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    backfill_complete: bool = False
    backfill_samples: int = 0
    evaluations: int = 0
    query_failures: int = 0
    query_empty: int = 0
    series_dropped: int = 0
    last_evaluation_ts: float = 0.0
    evaluation_lag: float = 0.0
    evaluation_duration: float = 0.0
    thread_alive: bool = True


class Runner:
    def __init__(self, config: Config, client: PromClient):
        self._cfg = config
        self._client = client
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.states: list[DetectorState] = [
            DetectorState(
                cfg=d,
                interval=config.interval_of(d),
                backfill=config.backfill_of(d),
                max_series=config.max_series_of(d),
                stale_after=config.stale_after_of(d),
            )
            for d in config.detectors
        ]

    # ---- lifecycle -------------------------------------------------------------------

    def start(self) -> None:
        for st in self.states:
            t = threading.Thread(target=self._run, args=(st,), name=f"detector-{st.cfg.name}")
            t.daemon = True
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)

    @property
    def ready(self) -> bool:
        return all(s.backfill_complete for s in self.states)

    @property
    def healthy(self) -> bool:
        return all(t.is_alive() for t in self._threads)

    def total_series(self) -> int:
        return sum(len(s.pipelines) for s in self.states)

    # ---- per detector loop -----------------------------------------------------------

    def _run(self, st: DetectorState) -> None:
        try:
            self._backfill_until_done(st)
            self._loop(st)
        except Exception:  # a dead thread must be visible through /healthz
            log.exception("detector %s crashed", st.cfg.name)
            st.thread_alive = False
            raise

    def _backfill_until_done(self, st: DetectorState) -> None:
        delay = 5.0
        while not self._stop.is_set():
            try:
                self.backfill(st, time.time())
                return
            except QueryError as exc:
                log.warning("backfill of %s failed, retrying in %.0fs: %s", st.cfg.name, delay, exc)
                st.query_failures += 1
                self._stop.wait(delay)
                delay = min(delay * 2, 300.0)

    def _loop(self, st: DetectorState) -> None:
        while not self._stop.is_set():
            now = time.time()
            # Evaluate on the interval grid, so seasonal slots line up with the backfill.
            slot = math.floor(now / st.interval) + 1
            target = slot * st.interval
            if self._stop.wait(max(0.0, target - now)):
                return
            self.evaluate(st, target)

    # ---- work ------------------------------------------------------------------------

    def _new_pipeline(self, st: DetectorState, key: LabelKey) -> SeriesPipeline | None:
        if len(st.pipelines) >= st.max_series:
            return None
        if self.total_series() >= self._cfg.defaults.max_series_total:
            return None
        d = st.cfg
        seed = zlib.crc32(f"{d.name}|{key}".encode())
        model = create_model(d.model.kind, dict(d.model.params), st.interval, seed)
        return SeriesPipeline(
            model,
            direction=d.direction,
            warmup_samples=d.warmup_samples or 0,
            min_expected=d.min_expected,
            tolerance_amount=d.tolerance.amount,
            tolerance_ratio=d.tolerance.ratio,
        )

    def backfill(self, st: DetectorState, now: float) -> None:
        """Replay history into fresh models. Raises QueryError if the store cannot answer.

        State is rebuilt from the data source on every start, so no local state is needed.
        """
        end = math.floor(now / st.interval) * st.interval
        start = math.floor((now - st.backfill) / st.interval) * st.interval
        data = self._client.query_range(st.cfg.query, start, end, st.interval)
        fed = 0
        dropped = 0
        with st.lock:
            st.pipelines.clear()
            st.last_seen.clear()
            for key in sorted(data):
                points = data[key]
                if not points or end - points[-1][0] > st.stale_after:
                    continue  # series is already gone, do not spend a model on it
                pipe = self._new_pipeline(st, key)
                if pipe is None:
                    dropped += 1
                    continue
                for ts, value in points:
                    pipe.update(ts, value)
                fed += len(points)
                st.pipelines[key] = pipe
                st.last_seen[key] = points[-1][0]
            st.series_dropped = dropped
            st.backfill_samples = fed
            st.backfill_complete = True
        log.info(
            "backfill of %s done: %d series, %d samples, %d dropped by series limit",
            st.cfg.name,
            len(st.pipelines),
            fed,
            dropped,
        )

    def evaluate(self, st: DetectorState, ts: float) -> None:
        """One evaluation. A failed or empty query produces no values, never zeros."""
        started = time.monotonic()
        try:
            samples = self._client.query_instant(st.cfg.query, ts)
        except QueryError as exc:
            st.query_failures += 1
            log.warning("query for %s failed: %s", st.cfg.name, exc)
            self._finish(st, ts, started)
            return
        if not samples:
            st.query_empty += 1
        self._apply(st, ts, samples)
        self._finish(st, ts, started)

    def _apply(self, st: DetectorState, ts: float, samples: list[InstantSample]) -> None:
        dropped = 0
        with st.lock:
            for s in sorted(samples, key=lambda x: x.labels):
                pipe = st.pipelines.get(s.labels)
                if pipe is None:
                    pipe = self._new_pipeline(st, s.labels)
                    if pipe is None:
                        dropped += 1
                        continue
                    st.pipelines[s.labels] = pipe
                pipe.update(ts, s.value)
                st.last_seen[s.labels] = ts
            for key in [k for k, seen in st.last_seen.items() if ts - seen > st.stale_after]:
                del st.pipelines[key]
                del st.last_seen[key]
            st.series_dropped = dropped
            st.last_evaluation_ts = ts

    def _finish(self, st: DetectorState, ts: float, started: float) -> None:
        st.evaluations += 1
        st.evaluation_duration = time.monotonic() - started
        st.evaluation_lag = max(0.0, time.time() - ts)
