"""One-step latency of the AWS library called in process through JPype, for comparison with
the child-process bridge of models/awsrcf.py. JPype is not a dependency of the project:
install it into a scratch environment (`uv pip install jpype1`).

    ANOMALY_AWSRCF_HOME=... ANOMALY_AWSRCF_JAVA=... python scripts/bench_awsrcf_jpype.py
"""

import glob
import json
import math
import os
import random
import time

import jpype
import jpype.imports
import numpy as np

home = os.environ["ANOMALY_AWSRCF_HOME"]
t0 = time.perf_counter()
jpype.startJVM(
    os.environ["ANOMALY_AWSRCF_JAVA"].replace("/bin/java", "/lib/server/libjvm.so"),
    "-Xmx1g",
    classpath=[j for j in glob.glob(f"{home}/*.jar") if "bridge" not in j],
)
start_s = time.perf_counter() - t0
from com.amazon.randomcutforest.config import TransformMethod  # noqa: E402
from com.amazon.randomcutforest.parkservices import ThresholdedRandomCutForest  # noqa: E402

trcf = (
    ThresholdedRandomCutForest.builder()
    .dimensions(8)
    .shingleSize(8)
    .internalShinglingEnabled(True)
    .numberOfTrees(50)
    .sampleSize(256)
    .randomSeed(0)
    .outputAfter(64)
    .anomalyRate(0.005)
    .parallelExecutionEnabled(False)
    .transformMethod(TransformMethod.NORMALIZE)
    .build()
)
rng = random.Random(1)
lat = []
for i in range(3400):
    v = 100 + 20 * math.sin(i / 60) + rng.gauss(0, 2)
    t0 = time.perf_counter()
    d = trcf.process([v], i * 60)
    d.getRCFScore()
    d.getAnomalyGrade()
    if i >= 400:
        lat.append((time.perf_counter() - t0) * 1e3)
print(
    json.dumps(
        {
            "jvm_start_s": start_s,
            "p50_ms": float(np.percentile(lat, 50)),
            "p99_ms": float(np.percentile(lat, 99)),
        }
    )
)
