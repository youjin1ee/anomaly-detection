"""AWS Random Cut Forest (Java, Apache-2.0) behind the model interface.

The library runs in a small Java child process (`sidecar/awsrcf/AwsRcfBridge.java`) that
speaks a line protocol on stdin and stdout. One child hosts every series of the Python process,
so the JVM start and the heap are paid once, not per series. Nothing is imported at module
level that is not in the standard library, and the JVM is only started when the first
`awsrcf` model is created, so the default install does not change.

Where the files are:

    ANOMALY_AWSRCF_HOME   directory with randomcutforest-core, randomcutforest-parkservices,
                          the jackson jars and awsrcf-bridge.jar (scripts/build-awsrcf-bridge.sh)
    ANOMALY_AWSRCF_JAVA   java executable (default: java on PATH, JRE 17 or newer)
    ANOMALY_AWSRCF_JAVA_OPTS  extra JVM options (default -Xmx1g)

Modes (parameter `mode`):

    rcf      plain forest of the core library. Raw score divided by `score_div` and capped at 1.
             With shingle 1, 100 trees, sample 256, decay 1e-5 and score_div 8 this is the
             setting of the AWS entry in the NAB scoreboard (see docs/results/m3-poc-awsrcf.md).
    trcf     ThresholdedRandomCutForest. Score is the raw RCF score relative to the
             threshold the library keeps, squashed so that 0.5 is the threshold. No band.
    caster   RCFCaster, a ThresholdedRandomCutForest that also forecasts with calibrated
             error ranges. The forecast made after sample t is the expected value and band of
             sample t+1. Default.
"""

from __future__ import annotations

import atexit
import contextlib
import math
import os
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import Any, ClassVar

from .base import ModelOutput, check_params, logistic, register_model


class AwsRcfUnavailable(RuntimeError):
    """The JVM or the jars are missing. The message says what to set."""


def _find_java() -> str:
    """ANOMALY_AWSRCF_JAVA, else a JDK unpacked inside ANOMALY_AWSRCF_HOME (the bridge needs
    Java 17 or newer, and the java on PATH is often older), else java on PATH."""
    java = os.environ.get("ANOMALY_AWSRCF_JAVA")
    if not java:
        home = os.environ.get("ANOMALY_AWSRCF_HOME")
        bundled = sorted(Path(home).glob("jdk*/bin/java")) if home else []
        java = str(bundled[-1]) if bundled else shutil.which("java")
    if not java:
        raise AwsRcfUnavailable("no java found: set ANOMALY_AWSRCF_JAVA or put java on PATH")
    return java


def _classpath() -> str:
    home = os.environ.get("ANOMALY_AWSRCF_HOME")
    if not home or not Path(home).is_dir():
        raise AwsRcfUnavailable("set ANOMALY_AWSRCF_HOME to the directory with the RCF jars")
    jars = sorted(Path(home).glob("*.jar"))
    if not any(j.name.startswith("awsrcf-bridge") for j in jars):
        raise AwsRcfUnavailable(
            f"{home} has no awsrcf-bridge.jar; run scripts/build-awsrcf-bridge.sh {home}"
        )
    return os.pathsep.join(str(j) for j in jars)


class Bridge:
    """One Java child process, shared by every series of this Python process."""

    def __init__(self) -> None:
        cmd = [_find_java()]
        opts = os.environ.get("ANOMALY_AWSRCF_JAVA_OPTS", "-Xmx1g")
        cmd += opts.split()
        cmd += ["-cp", _classpath(), "AwsRcfBridge"]
        self._err = tempfile.TemporaryFile("w+")  # noqa: SIM115 - JVM stderr, shown only if the start fails
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._err,
            text=True,
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._next = 0
        first = self._proc.stdout.readline().strip()  # type: ignore[union-attr]
        if first != "READY":
            self._proc.wait()
            self._err.seek(0)
            tail = self._err.read().strip()[-300:]
            raise AwsRcfUnavailable(f"the Java bridge did not start ({cmd[0]}): {tail}")
        atexit.register(self.close)

    def new_id(self) -> str:
        with self._lock:
            self._next += 1
            return f"s{self._next}"

    def call(self, line: str) -> str:
        with self._lock:
            stdin, stdout = self._proc.stdin, self._proc.stdout
            stdin.write(line + "\n")  # type: ignore[union-attr]
            stdin.flush()  # type: ignore[union-attr]
            reply = stdout.readline()  # type: ignore[union-attr]
        if not reply:
            raise AwsRcfUnavailable("the Java bridge process ended")
        reply = reply.strip()
        if reply.startswith("ERR"):
            raise RuntimeError(f"awsrcf: {reply[4:]}")
        return reply

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.stdin.write("QUIT\n")  # type: ignore[union-attr]
                self._proc.stdin.flush()  # type: ignore[union-attr]
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()


_bridge: Bridge | None = None
_bridge_lock = threading.Lock()


def get_bridge() -> Bridge:
    global _bridge
    with _bridge_lock:
        if _bridge is None or _bridge._proc.poll() is not None:
            _bridge = Bridge()
        return _bridge


def _f(text: str) -> float:
    return math.nan if text == "nan" else float(text)


class AwsRcfModel:
    """Parameters:

    mode               rcf, trcf or caster (default caster)
    shingle_size       consecutive samples per point (default 8; 1 in mode rcf)
    num_trees          trees (default 50; 100 in mode rcf)
    sample_size        points kept per tree (default 256)
    time_decay         forest decay per update (library default when unset)
    output_after       updates before the forest scores (library default when unset)
    anomaly_rate       expected share of anomalies, drives the threshold (default 0.005)
    z_factor           threshold width in standard deviations (default 2.5)
    calibration        none, minimal or simple, error calibration of the band (default simple)
    error_horizon      samples of forecast error kept for the band (library default)
    score_div          mode rcf only: raw score divided by this, capped at 1 (default
                       log2(sample_size))
    threshold_width    modes trcf and caster: score is 0.5 at the threshold and
                       `1 / (1 + exp(-(raw / threshold - 1) / threshold_width))` otherwise
                       (default 0.25)
    score_from         modes trcf and caster: `ratio` (above) or `grade`, the anomaly grade of
                       the library itself, 0 for a normal sample and up to 1 (default ratio)
    reference_window   samples of the trailing median used as `expected` when the library
                       gives none (default 60)
    """

    _ALLOWED: ClassVar[set[str]] = {
        "mode",
        "shingle_size",
        "num_trees",
        "sample_size",
        "time_decay",
        "output_after",
        "anomaly_rate",
        "z_factor",
        "calibration",
        "error_horizon",
        "score_div",
        "threshold_width",
        "score_from",
        "reference_window",
    }

    def __init__(
        self,
        seed: int = 0,
        mode: str = "caster",
        shingle_size: int | None = None,
        num_trees: int | None = None,
        sample_size: int = 256,
        time_decay: float | None = None,
        output_after: int | None = None,
        anomaly_rate: float = 0.005,
        z_factor: float = 2.5,
        calibration: str = "simple",
        error_horizon: int | None = None,
        score_div: float | None = None,
        threshold_width: float = 0.25,
        score_from: str = "ratio",
        reference_window: int = 60,
    ):
        if mode not in ("rcf", "trcf", "caster"):
            raise ValueError("mode must be rcf, trcf or caster")
        if calibration not in ("none", "minimal", "simple"):
            raise ValueError("calibration must be none, minimal or simple")
        if score_from not in ("ratio", "grade"):
            raise ValueError("score_from must be ratio or grade")
        self._mode = mode
        self._from = score_from
        self._div = score_div if score_div else math.log2(sample_size)
        self._width = threshold_width
        self._recent: deque[float] = deque(maxlen=reference_window)
        args = {
            "shingle": shingle_size or (1 if mode == "rcf" else 8),
            "trees": num_trees or (100 if mode == "rcf" else 50),
            "sample": sample_size,
            "seed": seed,
        }
        if time_decay is not None:
            args["decay"] = time_decay
        if output_after is not None:
            args["output_after"] = output_after
        if mode != "rcf":
            args["anomaly_rate"] = anomaly_rate
            args["z_factor"] = z_factor
        if mode == "caster":
            args["calibration"] = calibration.upper()
            if error_horizon is not None:
                args["error_horizon"] = error_horizon
        self._bridge = get_bridge()
        self._sid = self._bridge.new_id()
        spec = " ".join(f"{k}={v}" for k, v in args.items())
        self._bridge.call(f"NEW {self._sid} {mode} {spec}")
        self.samples = 0
        self.grade = 0.0  # anomaly grade of the last sample, for inspection

    def update(self, timestamp: float, value: float) -> ModelOutput | None:
        self.samples += 1
        reply = self._bridge.call(f"U {self._sid} {int(timestamp)} {value!r}")
        _, raw_s, grade_s, thr_s, exp_s, lo_s, up_s, ready_s = reply.split()
        raw, grade, thr = float(raw_s), float(grade_s), float(thr_s)
        expected, lower, upper = _f(exp_s), _f(lo_s), _f(up_s)
        recent = self._recent
        median = _median(recent) if recent else value
        recent.append(value)
        if ready_s != "1":
            return None
        self.grade = grade
        if self._mode == "rcf":
            score = min(1.0, max(0.0, raw / self._div))
        elif self._from == "grade":
            score = min(1.0, max(0.0, grade))
        else:
            ref = max(thr, 1e-6)
            score = logistic((raw / ref - 1.0) / self._width)
        if math.isnan(expected):
            expected, lower, upper = median, None, None  # type: ignore[assignment]
        else:
            lower, upper = (None, None) if math.isnan(lower) else (lower, upper)
        return ModelOutput(score, expected, lower, upper)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._bridge.call(f"DEL {self._sid}")


def _median(values: deque[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _factory(params: dict[str, Any], interval: float, seed: int) -> AwsRcfModel:
    check_params("awsrcf", params, AwsRcfModel._ALLOWED)
    return AwsRcfModel(seed=seed, **params)


register_model("awsrcf", _factory)
