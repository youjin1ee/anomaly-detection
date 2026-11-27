"""Markdown tables from the JSON that `anomaly-detection-boom` writes.

python scripts/boom_report.py out/subset.json [out/all14.json ...]
"""

from __future__ import annotations

import json
import sys

import numpy as np

from anomaly_detection.harness.replay import CONTEXTS

G = ("off", "on")
GNAME = {"off": "끔", "on": "켬"}
RES = {"T": "1분", "5T": "5분"}


def cell(s: dict) -> str:
    return "-" if not s["n"] else f"{s['mean']:.3f} ({100 * s['met']:.0f}%, n={s['n']})"


def mean_ci(x: dict) -> str:
    """Bootstrap 95 percent interval of the mean over series (2000 resamples, seed 0)."""
    v = np.array([r["per_day"] for r in x["per_series"] if r["ready"]])
    rng = np.random.default_rng(0)
    means = v[rng.integers(0, len(v), (2000, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return f"{lo:.3f} - {hi:.3f}"


def table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render(d: dict) -> str:
    models = d["models"]
    out = [f"### 시계열 {d['series']}개, 점 {d['points']}개\n"]
    rows = []
    for m, e in models.items():
        for g in G:
            x = e["guards"][g]
            o = x["overall"]
            rows.append(
                [
                    m,
                    GNAME[g],
                    str(x["scored"]),
                    str(x["alerts"]),
                    f"{o['mean']:.3f}",
                    mean_ci(x),
                    f"{o['median']:.3f}",
                    f"{o['p95']:.3f}",
                    f"{100 * o['met']:.0f}%",
                    f"{x['pooled_per_series_day']:.3f}",
                ]
            )
    out.append("시계열당 하루 알림 수\n")
    out.append(
        table(
            [
                "model",
                "보호 설정",
                "scored",
                "알림",
                "평균",
                "평균 95% 구간",
                "중앙값",
                "95분위",
                "기준 충족",
                "합산 기준 하루",
            ],
            rows,
        )
    )
    for title, key in (
        ("해상도별 평균 (기준 충족 비율, 시계열 수)", "by_resolution"),
        ("지표 유형별", "by_metric_type"),
    ):
        names = sorted({k for e in models.values() for k in e["guards"]["on"][key]})
        rows = []
        for m, e in models.items():
            for g in G:
                rows.append(
                    [m, GNAME[g]] + [cell(e["guards"][g][key].get(n, {"n": 0})) for n in names]
                )
        out.append(f"\n{title}\n")
        out.append(table(["model", "보호 설정"] + [RES.get(n, n) for n in names], rows))
    names = sorted({k for e in models.values() for k in e["guards"]["on"]["by_domain"]})
    for g in G:
        rows = []
        for n in names:
            rows.append(
                [n] + [cell(e["guards"][g]["by_domain"].get(n, {"n": 0})) for e in models.values()]
            )
        out.append(f"\n도메인별, 보호 설정 {GNAME[g]}\n")
        out.append(table(["domain", *models], rows))
    rows = []
    for m, e in models.items():
        for g in G:
            c = e["guards"][g]["concurrency"]
            rows.append(
                [
                    m,
                    GNAME[g],
                    str(c["windows"]),
                    str(c["max"]),
                    f"{c['p50']:.1f}",
                    f"{c['p90']:.1f}",
                    f"{100 * c['alerts_in_5plus']:.0f}%",
                    f"{100 * c['max_share']:.0f}%",
                ]
            )
    out.append("\n동시 알림 (같은 10분 창)\n")
    out.append(
        table(
            [
                "model",
                "보호 설정",
                "알림 있는 창",
                "최대 시계열",
                "중앙값",
                "90분위",
                "5개 이상 창의 알림",
                "최대 점유",
            ],
            rows,
        )
    )
    rows = []
    for m, e in models.items():
        for g in G:
            c = e["guards"][g]["contexts"]
            rows.append([m, GNAME[g]] + [str(c[k]) for k in CONTEXTS])
    out.append("\n알림 상황 분류\n")
    out.append(table(["model", "보호 설정", *CONTEXTS], rows))
    rows = []
    for m, e in models.items():
        for fl in ("0.0", "0.5"):
            c = e["guards"]["on"]["calibration"][fl]
            rows.append(
                [
                    m,
                    fl,
                    str(c["series"]),
                    str(c["silenced"]),
                    f"{c['threshold_median']:.3f}",
                    f"{c['pooled_plain']:.3f}",
                    f"{c['pooled_calibrated']:.3f}",
                    f"{c['plain_rule']['p95']:.3f}",
                    f"{c['calibrated']['p95']:.3f}",
                    f"{100 * c['plain_rule']['met']:.0f}%",
                    f"{100 * c['calibrated']['met']:.0f}%",
                ]
            )
    out.append("\n오탐 예산 보정 (보호 설정 켬, 예산 하루 0.1건)\n")
    out.append(
        table(
            [
                "model",
                "임계 하한",
                "series",
                "침묵",
                "임계 중앙값",
                "0.5 고정 합산",
                "보정 합산",
                "0.5 고정 95분위",
                "보정 95분위",
                "0.5 고정 충족",
                "보정 충족",
            ],
            rows,
        )
    )
    rows = [
        [
            m,
            f"{e['wall_seconds']:.0f}",
            f"{e['series_seconds']:.0f}",
            f"{1000 * e['series_seconds'] / d['points']:.2f}",
        ]
        for m, e in models.items()
    ]
    out.append("\n재생 시간\n")
    out.append(table(["model", "벽시계 초", "시계열별 초 합", "점당 ms"], rows))
    return "\n".join(out)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"## {path}\n")
        with open(path) as fh:
            print(render(json.load(fh)))
        print()
