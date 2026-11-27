"""`anomaly-detection-ensemble`: evaluate the ensemble judgement layer on cached model scores.

    anomaly-detection-ensemble run --out out.json \
        --nab-dir NAB --nab-score fixed=a.npz --nab-score awsrcf-nab=b.npz \
        --smd-root SMD --psm-root PSM --bench-out m4-bench-smd-psm-out/run \
        --boom-root BOOM --boom-cache boom/cache \
        --qa-dumps DUMPS --qa-cache DUMPS/reports/cache --lab-dir azure=azure/run-code
    anomaly-detection-ensemble qa-score --qa-dumps DUMPS --qa-cache CACHE --workers 8
    anomaly-detection-ensemble report out.json

Model scores come from the caches of the earlier runs; only the models that a cache lacks are
computed (`qa-score`). One run evaluates every spec of `SPECS` on every dataset given, with one
fixed configuration, nothing is tuned per dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

from ..ensemble.core import STAGES, EnsembleConfig, EnsembleSpec
from .ensemble_data import (
    boom_units,
    lab_units,
    nab_units,
    qa_units,
    score_missing,
    smd_psm_units,
)
from .ensemble_eval import Aggregator, evaluate_units

BASE = ("awsrcf-nab", "subpca", "fixed")
SINGLES = ("fixed", "seasonal", "subpca", "awsrcf-nab", "awsrcf")
ENSEMBLES: tuple[EnsembleSpec, ...] = (
    EnsembleSpec("or2", ("awsrcf-nab", "subpca")),
    EnsembleSpec("or3", BASE),
    EnsembleSpec("mean3", BASE, "mean"),
    EnsembleSpec("vote2of3", BASE, "vote", 2),
    EnsembleSpec("route", route=True),
    EnsembleSpec("or3+rules", BASE, rules=True),
    EnsembleSpec("vote2of3+rules", BASE, "vote", 2, rules=True),
    EnsembleSpec("route+rules", route=True, rules=True),
    # comparison only: seasonal is not in the default composition
    EnsembleSpec("or3+seasonal", (*BASE, "seasonal")),
)
ABLATION = ("or3", "or3+rules")  # also run without tail extrapolation (suffix @max)


def specs_for(models: list[str]) -> list[EnsembleSpec]:
    singles = [EnsembleSpec(m, (m,)) for m in SINGLES if m in models]
    return [*singles, *ENSEMBLES]


def parse_pairs(items: list[str]) -> dict[str, Path]:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k] = Path(v)
    return out


def cmd_run(a: argparse.Namespace) -> int:
    cfg = EnsembleConfig(
        cal_days=a.cal_days,
        budgets=(a.budget_ai, a.budget_human),
        min_cal=a.min_cal,
    )
    agg = Aggregator()
    models_all = ["fixed", "seasonal", "subpca", "awsrcf-nab", "awsrcf"]

    def plans(models: list[str]):
        sp = specs_for(models)
        ab = [s for s in sp if s.name in ABLATION]
        return [(cfg, sp, ""), (replace(cfg, extrapolate=False), ab, "@max")]

    if a.nab_dir:
        files = parse_pairs(a.nab_score)
        evaluate_units(nab_units(a.nab_dir, files, a.cal_days), plans(list(files)), agg)
    for ds, root in (("smd", a.smd_root), ("psm", a.psm_root)):
        if root:
            models = [m for m in models_all if m != "awsrcf" or ds == "psm"]
            evaluate_units(
                smd_psm_units(ds, root, a.bench_out, models, a.cal_days), plans(models), agg
            )
    if a.boom_root:
        evaluate_units(
            boom_units(a.boom_root, a.boom_cache, models_all, a.boom_max_days, a.cal_days),
            plans(models_all),
            agg,
        )
    if a.qa_dumps:
        evaluate_units(
            qa_units(a.qa_dumps, a.qa_cache, models_all, a.cal_days), plans(models_all), agg
        )
    for name, d in parse_pairs(a.lab_dir).items():
        evaluate_units(lab_units(name, d, models_all, a.cal_lab_days), plans(models_all), agg)
    out = {"config": {k: str(v) for k, v in vars(cfg).items()}, "results": agg.results()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("done", flush=True)
    return 0


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render(doc: dict) -> str:
    lines = []
    for ds, specs in doc["results"].items():
        any_stage = next(iter(next(iter(specs.values())).values()))
        labelled = "recall" in any_stage
        lines.append(f"\n### {ds}\n")
        for stage in STAGES:
            lines.append(f"\n{stage} stage\n")
            if labelled:
                lines.append(
                    "| spec | events | detection | miss | precision | FA/unit-day | F1 | delay min "
                    "| silent |\n|---|---|---|---|---|---|---|---|---|"
                )
                for name, st in specs.items():
                    m = st[stage]
                    lines.append(
                        f"| {name} | {m['events']} | {_f(m['recall'])} | {_f(m['miss'])} "
                        f"| {_f(m['precision'])} | {_f(m['fa_per_day'])} | {_f(m['f1'])} "
                        f"| {_f(m['delay_median_min'], 1)} | {_f(m['silent_share'], 2)} |"
                    )
            else:
                lines.append(
                    "| spec | units | alerts | pooled/unit-day | mean/unit-day | p90 "
                    "| within budget | silent |\n|---|---|---|---|---|---|---|---|"
                )
                for name, st in specs.items():
                    m = st[stage]
                    lines.append(
                        f"| {name} | {m['units']} | {m['alerts']} | {_f(m['alerts_per_day'])} "
                        f"| {_f(m['fa_per_unit_day_mean'])} | {_f(m['fa_per_unit_day_p90'])} "
                        f"| {_f(m['within_budget'], 2)} | {_f(m['silent_share'], 2)} |"
                    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anomaly-detection-ensemble", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--cal-days", type=float, default=3.0)
    r.add_argument("--cal-lab-days", type=float, default=2.0)
    r.add_argument("--budget-ai", type=float, default=1.0)
    r.add_argument("--budget-human", type=float, default=0.1)
    r.add_argument("--min-cal", type=int, default=100)
    r.add_argument("--nab-dir", type=Path)
    r.add_argument("--nab-score", action="append", default=[], metavar="MODEL=FILE")
    r.add_argument("--smd-root", type=Path)
    r.add_argument("--psm-root", type=Path)
    r.add_argument("--bench-out", type=Path)
    r.add_argument("--boom-root", type=Path)
    r.add_argument("--boom-cache", type=Path)
    r.add_argument("--boom-max-days", type=float, default=14.0)
    r.add_argument("--qa-dumps", type=Path)
    r.add_argument("--qa-cache", type=Path)
    r.add_argument("--lab-dir", action="append", default=[], metavar="NAME=DIR")
    q = sub.add_parser("qa-score")
    q.add_argument("--qa-dumps", type=Path, required=True)
    q.add_argument("--qa-cache", type=Path, required=True)
    q.add_argument("--models", default="fixed,seasonal,subpca,awsrcf-nab,awsrcf")
    q.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("report")
    s.add_argument("json", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "qa-score":
        score_missing(a.qa_dumps, a.qa_cache, a.models.split(","), a.workers)
        return 0
    print(render(json.loads(a.json.read_text())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
