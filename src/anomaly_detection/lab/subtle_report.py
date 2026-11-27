"""Score the runs of the subtle fault lab and break the result down by size and time of day.

Reads run directories (`signals.csv` and `labels.json`, as made by `lab.slice_run`), scores every
model on every signal (cached in `<run>/scores/`, see below) with the alert rule of
`lab.azure_report`, and writes tables in Korean:

- recall (with a 95 percent Wilson interval), miss rate and detection delay by size (gross,
  subtle), scenario and time of day band (peak, trough);
- false alarms per signal per compressed day with an exact Poisson interval;
- for subtle events, every model against the fixed limit, event by event (paired): both,
  model only, fixed only, neither, and the recall difference with a bootstrap interval;
- the effect each injection really had on its signal against the size it was meant to have;
- detection by run and by injection day, to see whether models change as injections pile up.

Score cache, per run: `<run>/scores/<model>.<signal>.npy`, one float array per model and signal,
aligned to the grid of that signal in `signals.csv` (`ts = (first + arange(n)) * step`, see
`scores/index.json`), NaN where the model has no score yet. Scores are in [0, 1]; the models saw
the time stretched by `time_stretch` so a lab day was a real day for them. The alert rule is
`score >= 0.5` for 3 samples in a row.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..harness.metrics import AlertRule, alert_spans
from . import azure_report, replay
from .azure_report import wilson
from .signals import SIGNALS

SEVERITIES = ("gross", "subtle")
BANDS = ("peak", "trough")


@dataclass
class Rec:
    run: str
    model: str
    scenario: str
    severity: str
    band: str
    detected: bool
    covered: bool  # not detected, but an alert that fired before the event was still going
    delay: float | None  # lab seconds
    max_score: float
    effect: float | None
    in_target: bool | None
    day: int  # injection day inside the run, from 0
    idx: int  # event number inside the run


@dataclass
class Fa:
    run: str
    model: str
    signal: str
    count: int
    cross: int
    where: dict  # band name or other -> count
    days: float  # compressed days of this signal's evaluation period


def poisson_ci(k: int, z: float = 1.96) -> tuple[float, float]:
    """Exact (Garwood) 95 percent interval of a Poisson mean given k events, by bisection."""

    def cdf(x: int, lam: float) -> float:  # P(X <= x)
        return sum(math.exp(-lam + i * math.log(lam) - math.lgamma(i + 1)) for i in range(x + 1))

    def solve(f, lo: float, hi: float) -> float:
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2

    alpha = 0.05 if abs(z - 1.96) < 1e-9 else 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    top = 10.0 + 10 * k
    low = 0.0 if k == 0 else solve(lambda lam: 1 - cdf(k - 1, lam) < alpha / 2, 1e-9, top)
    up = solve(lambda lam: cdf(k, lam) > alpha / 2, 1e-9, top)
    return low, up


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 4000, seed: int = 1):
    """95 percent interval of mean(a) - mean(b) over paired boolean arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def _f(x, nd: int = 2) -> str:
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def _rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {_f(k / n if n else float('nan'))} [{_f(lo)}, {_f(hi)}]"


def _delay(sel: list[Rec]) -> str:
    d = [r.delay for r in sel if r.delay is not None]
    return f"{np.median(d):.0f} / {np.percentile(d, 90):.0f}" if d else "n/a"


def load_run(path: Path) -> tuple[dict, list[dict], dict]:
    doc = json.loads((path / "labels.json").read_text())
    events = [e for e in doc["events"] if e["status"] == "completed"]
    return doc["meta"], events, replay.load_signals(path / "signals.csv", 5)


def score_run(
    tag: str, path: Path, models: list[str], paths: dict, rule: AlertRule, workers: int,
    step: int = 5, cache: Path | None = None,
) -> tuple[list[Rec], list[Fa]]:  # fmt: skip
    meta, events, signals = load_run(path)
    names = [s.name for s in SIGNALS if any(e["scenario"] == s.scenario for e in events)]
    runs = {n: replay.build_run(n, *signals[n], events, meta, step) for n in names}
    scale = replay.DAY / meta["day_seconds"]
    cache = cache or path / "scores"
    scores = azure_report.compute_scores(runs, models, paths, step, scale, cache, workers)
    index = {
        "step": step, "time_stretch": scale, "day_seconds": meta["day_seconds"],
        "alert_rule": {"threshold": rule.threshold, "for_points": rule.for_points},
        "models": models,
        "signals": {
            n: {"first_index": round(runs[n].case.ts[0] / step), "n": len(runs[n].case.ts)}
            for n in names
        },
        "ts_formula": "ts = (first_index + arange(n)) * step, unix seconds of the run",
    }  # fmt: skip
    (cache / "index.json").write_text(json.dumps(index, indent=1))
    recs, fas = [], []
    for m in models:
        for n in names:
            run, sc = runs[n], scores[(m, n)]
            row = replay._row(m, n, run, sc, step, rule)
            spans = alert_spans(sc, rule, run.case.eval_start)
            own_ev = [e for e in events if e["scenario"] == run.case.scenario]
            for k, ((s, e), ev) in enumerate(zip(run.own, own_ev, strict=True)):
                hits = [f for f in row.result.fires if s <= f <= e + rule.grace]
                window = sc[s : e + rule.grace + 1]
                covered = not hits and any(
                    first <= s and fire < s and last >= s for first, fire, last in spans
                )
                recs.append(
                    Rec(
                        tag, m, ev["scenario"], ev.get("severity", "gross"), ev.get("band", ""),
                        bool(hits), covered, (hits[0] - s) * step if hits else None,
                        float(np.nanmax(window)) if np.isfinite(window).any() else float("nan"),
                        ev.get("effect"), ev.get("in_target"),
                        int((ev["actual_start"] - meta["eval_start"]) // meta["day_seconds"]), k,
                    )
                )  # fmt: skip
            ts = run.case.ts
            where: dict[str, int] = defaultdict(int)
            for f in row.result.fires:
                if not any(s <= f <= e + rule.grace for s, e in run.own):
                    where[azure_report.fa_band(float(ts[f]), meta)] += 1
            fas.append(
                Fa(tag, m, n, row.result.false_alarms, row.cross_alarms, dict(where),
                   row.result.eval_days * 86400 / meta["day_seconds"])
            )  # fmt: skip
    return recs, fas


def _sel(recs, **kw):
    return [r for r in recs if all(getattr(r, k) == v for k, v in kw.items())]


def render_by_size(recs, models) -> str:
    out = ["", "### 크기별 전체", "", "미탐 비율은 1에서 탐지율을 뺀 값. 합쳐짐은 이벤트 시작 전에 이미 울린 알림이 이어져 "
           "따로 세지 못한 건수(탐지에는 안 들어감). 지연은 중앙값 / 90분위(랩 초).", "",
           "| 크기 | 모델 | 이벤트 | 탐지율 [95% 구간] | 미탐 비율 | 앞 알림에 합쳐짐 | 지연 |",
           "|---|---|---|---|---|---|---|"]  # fmt: skip
    for sev in SEVERITIES:
        for m in models:
            sel = _sel(recs, severity=sev, model=m)
            if not sel:
                continue
            k = sum(r.detected for r in sel)
            out.append(
                f"| {sev} | {m} | {len(sel)} | {_rate(k, len(sel))} | {_f(1 - k / len(sel))} "
                f"| {sum(r.covered for r in sel)} | {_delay(sel)} |"
            )
    return "\n".join(out) + "\n"


def render_false_alarms(fas, models) -> str:
    out = ["", "### 오탐", "",
           "신호 3개를 합친 건수. 비율은 신호당 압축일당, 구간은 정확 포아송 95%. 다른 주입 중은 다른 시나리오 "
           "이벤트 구간에 울린 것. 주입이 하루의 절반 넘게 이어지므로 조용한 시간이 적음.", "",
           "| 모델 | 오탐 | 신호당 압축일당 [95% 구간] | 다른 주입 중 | 피크 시간대 | 저점 시간대 | 그 외 |",
           "|---|---|---|---|---|---|---|"]  # fmt: skip
    for m in models:
        mine = [f for f in fas if f.model == m]
        if not mine:
            continue
        n, days = sum(f.count for f in mine), sum(f.days for f in mine)
        lo, hi = poisson_ci(n)
        w = defaultdict(int)
        for f in mine:
            for b, c in f.where.items():
                w[b] += c
        out.append(
            f"| {m} | {n} | {_f(n / days)} [{_f(lo / days)}, {_f(hi / days)}] "
            f"| {sum(f.cross for f in mine)} | {w['peak']} | {w['trough']} | {w['other']} |"
        )
    return "\n".join(out) + "\n"


def render_breakdown(recs, models, by_band: bool) -> str:
    title = "### 크기, 시나리오, 시간대별" if by_band else "### 크기, 시나리오별"
    head = "| 크기 | 시나리오 | " + ("시간대 | " if by_band else "")
    out = ["", title, "", head + "모델 | 이벤트 | 탐지율 [95% 구간] | 지연 중앙 / 90분위 |",
           "|---|---|" + ("---|" if by_band else "") + "---|---|---|---|"]  # fmt: skip
    scen = sorted({r.scenario for r in recs})
    for sev in SEVERITIES:
        for sc in scen:
            for band in BANDS if by_band else (None,):
                for m in models:
                    kw = {"severity": sev, "scenario": sc, "model": m}
                    if band:
                        kw["band"] = band
                    sel = _sel(recs, **kw)
                    if not sel:
                        continue
                    k = sum(r.detected for r in sel)
                    out.append(
                        f"| {sev} | {sc} | " + (f"{band} | " if band else "")
                        + f"{m} | {len(sel)} | {_rate(k, len(sel))} | {_delay(sel)} |"
                    )  # fmt: skip
    return "\n".join(out) + "\n"


def render_paired(recs, models, baseline: str = "fixed") -> str:
    out = ["", "### 고정 임계 대비(같은 이벤트끼리 짝지어 비교)", "",
           "둘 다, 모델만, 고정만, 둘 다 아님은 이벤트 수. 차이는 모델 탐지율에서 고정 탐지율을 뺀 값이고 "
           "구간은 이벤트를 다시 뽑는 부트스트랩 95%. 동적 합집합은 fixed를 뺀 모델 중 하나라도 잡은 이벤트의 비율"
           "(오탐은 더해지므로 상한 비교용).", "",
           "| 크기 | 시나리오 | 모델 | 이벤트 | 둘 다 | 모델만 | 고정만 | 둘 다 아님 | 차이 [95% 구간] |",
           "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    dyn = [m for m in models if m != baseline]
    for sev in SEVERITIES:
        for sc in ["all", *sorted({r.scenario for r in recs})]:
            base = {
                (r.run, r.scenario, r.idx): r.detected
                for r in recs
                if r.model == baseline and r.severity == sev and (sc == "all" or r.scenario == sc)
            }
            keys = sorted(base)
            if not keys:
                continue
            fixed = np.array([base[k] for k in keys])
            per_model = {}
            for m in dyn:
                det = {
                    (r.run, r.scenario, r.idx): r.detected
                    for r in recs
                    if r.model == m and r.severity == sev
                }
                per_model[m] = np.array([det[k] for k in keys])
            per_model["동적 합집합"] = np.any(np.array(list(per_model.values())), axis=0)
            for m, d in per_model.items():
                lo, hi = paired_bootstrap(d, fixed)
                out.append(
                    f"| {sev} | {sc} | {m} | {len(keys)} | {int((d & fixed).sum())} "
                    f"| {int((d & ~fixed).sum())} | {int((~d & fixed).sum())} "
                    f"| {int((~d & ~fixed).sum())} | {d.mean() - fixed.mean():+.2f} "
                    f"[{lo:+.2f}, {hi:+.2f}] |"
                )
    return "\n".join(out) + "\n"


def render_effects(recs) -> str:
    out = ["", "### 주입이 신호에 준 실제 효과", "",
           "이벤트 시작 60초 뒤부터 끝까지 신호 중앙값을 시작 전 180초 중앙값으로 나눈 비율. 목표는 미묘한 장애에 "
           "정한 범위이고 범위 안은 그 안에 든 건수.", "",
           "| 크기 | 시나리오 | 시간대 | 이벤트 | 비율 중앙값 | 최소 | 최대 | 목표 | 범위 안 |",
           "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    seen = {(r.run, r.scenario, r.idx): r for r in recs if r.effect is not None}
    for sev in SEVERITIES:
        for sc in sorted({r.scenario for r in recs}):
            for band in ("all", *BANDS):
                x = [
                    r for r in seen.values()
                    if r.severity == sev and r.scenario == sc and (band == "all" or r.band == band)
                ]  # fmt: skip
                if not x:
                    continue
                e = [r.effect for r in x]
                tg = [r for r in x if r.in_target is not None]
                tgt = "n/a" if not tg else f"{sum(r.in_target for r in tg)}/{len(tg)}"
                out.append(
                    f"| {sev} | {sc} | {band} | {len(x)} | {np.median(e):.2f} | {min(e):.2f} "
                    f"| {max(e):.2f} | {_target(sev, sc)} | {tgt} |"
                )
    return "\n".join(out) + "\n"


TARGETS = {"request_drop": "0.5~0.7", "output_length_drop": "0.5~0.7", "throughput_drop": "0.7~0.85"}


def _target(sev: str, sc: str) -> str:
    return TARGETS[sc] if sev == "subtle" else "n/a"


def render_drift(recs, models) -> str:
    out = ["", "### 실행별, 주입 일차별 탐지율", "",
           "주입이 쌓일수록 점수가 내려가는 모델이 있는지 보려는 표. 일차는 한 실행 안에서 0부터.", "",
           "| 크기 | 모델 | " + " | ".join(
               f"{r}" for r in sorted({r.run for r in recs})) + " | 일차 0 | 일차 1 | 일차 2 |",
           "|---|---|" + "---|" * (len({r.run for r in recs}) + 3)]  # fmt: skip
    runs = sorted({r.run for r in recs})
    for sev in SEVERITIES:
        for m in models:
            sel = _sel(recs, severity=sev, model=m)
            if not sel:
                continue
            cells = []
            for run in runs:
                s = [r for r in sel if r.run == run]
                cells.append(f"{sum(r.detected for r in s)}/{len(s)}")
            for d in range(3):
                s = [r for r in sel if r.day == d]
                cells.append(f"{sum(r.detected for r in s)}/{len(s)}")
            out.append(f"| {sev} | {m} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m anomaly_detection.lab.subtle_report", description=__doc__)
    p.add_argument("--runs", nargs="+", type=Path, required=True, help="run directories")
    p.add_argument("--old", type=Path, help="run directory of the earlier lab, gross events only")
    p.add_argument("--models", default="fixed,seasonal,awsrcf,awsrcf-nab,subpca,rcf,tspulse")
    p.add_argument("--toto-dir", default="")
    p.add_argument("--tspulse-model", default="")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--out", type=Path)
    p.add_argument("--events-csv", type=Path, help="one row per event and model")
    a = p.parse_args(argv)
    models = [m for m in a.models.split(",") if m]
    paths = {"toto": a.toto_dir, "tspulse": a.tspulse_model}
    rule = AlertRule()
    recs: list[Rec] = []
    fas: list[Fa] = []
    for path in a.runs:
        r, f = score_run(path.name, path, models, paths, rule, a.workers)
        recs += r
        fas += f
    parts = [
        f"실행 {len(a.runs)}개, 이벤트 {len({(r.run, r.scenario, r.idx) for r in recs})}건. "
        "알림 규칙은 점수 0.5 이상이 3샘플 연속.\n",
        render_by_size(recs, models), render_false_alarms(fas, models),
        render_breakdown(recs, models, by_band=False), render_breakdown(recs, models, by_band=True),
        render_paired(recs, models), render_effects(recs), render_drift(recs, models),
    ]  # fmt: skip
    if a.old:
        # The earlier lab had no severity (all big) and a warm-up with the weekend in it. Its
        # scores come from the cache that run left (not rewritten), copied beside it.
        old_models = [m for m in models if (a.old / "scores" / f"{m}.request_rate.npy").exists()]
        r, f = score_run("old", a.old, old_models, paths, rule, a.workers, cache=a.old / "scores")
        pooled = [x for x in recs if x.severity == "gross"] + r
        parts.append("\n## 지난 실행(24건, 워밍업에 주말 포함)을 합친 큰 장애\n")
        parts.append(render_by_size(pooled, old_models).replace("### 크기별 전체", "### 큰 장애 합산"))
        parts.append(render_breakdown([x for x in pooled], old_models, by_band=True))
    text = "\n".join(parts)
    print(text)
    if a.out:
        a.out.write_text(text)
    if a.events_csv:
        import csv

        with a.events_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "model", "scenario", "severity", "band", "day", "idx", "detected",
                        "covered", "delay_lab_s", "max_score", "effect", "in_target"])  # fmt: skip
            for r in recs:
                w.writerow([r.run, r.model, r.scenario, r.severity, r.band, r.day, r.idx,
                            int(r.detected), int(r.covered), r.delay, r.max_score, r.effect,
                            r.in_target])  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
