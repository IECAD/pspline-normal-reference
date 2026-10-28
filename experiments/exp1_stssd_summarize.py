"""실험 1 ST-SSD형 비교 집계 (doc/실험1_ST-SSD_계획.md 3절 판정을 그대로 구현). 테스트.

PS = results/C03 (F01 합성 + cubic B-spline K_t = 12 + SAP = 제안 모델, T = 12) 의 테스트 결과.
판정: PS − 변형 짝지은 AUC-PR 평균 (D1 주입 45쌍, D2 다지표 60쌍), 허용오차 D1 0.0165 · D2 0.0171.
출력: results/exp1_stssd/summary.md, paired.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exp1_stssd"
TOL = {"d1": 0.0165, "d2": 0.0171}
VIEWS = {"d1": r"^SC\d$", "d2": r"^SC\dm$"}
COLS = ["auc_pr", "hitrate", "cell_f1", "smearing", "infer_ms_per_window"]

sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))["selection"]
lines = ["# ST-SSD형 (정적) vs 제안 모델 PS — 테스트", "",
         "자동 생성: `experiments/exp1_stssd_summarize.py`. 계획: `doc/실험1_ST-SSD_계획.md`.", ""]
rows = []
for ds, rx in VIEWS.items():
    S = pd.read_csv(OUT / f"metrics_test_{ds}.csv")
    P = pd.read_csv(ROOT / "results" / "C03" / "base" / f"metrics_{ds}_test.csv").assign(variant="PS")
    M = pd.concat([S, P], ignore_index=True)
    M = M[(M.split == "test") & M.scenario.astype(str).str.match(rx)]
    t = M.groupby("variant")[COLS].mean()
    lines += [f"## {ds} ({'주입 SC1–SC3' if ds == 'd1' else '다지표 주입 SC1m–SC3m'})", "",
              "| 모델 | 선택 (γ, K_a) | AUC-PR | HitRate | 셀 F1 | Smearing | 추론 ms/윈도우 | PS − 모델 (양수 비율) | 판정 |",
              "|---|---|---|---|---|---|---|---|---|"]
    key = ["space", "scenario", "seed"]
    ps = M[M.variant == "PS"].set_index(key).auc_pr
    for v in ["PS", "N", "G1", "G2"]:
        r = t.loc[v]
        if v == "PS":
            lines.append(f"| PS (제안) | — | {r.auc_pr:.3f} | {r.hitrate:.3f} | {r.cell_f1:.3f} | {r.smearing:.3f} | "
                         f"{r.infer_ms_per_window:.4f} | — | — |")
            continue
        x = (ps - M[M.variant == v].set_index(key).auc_pr).dropna()
        verdict = "우위" if x.mean() > TOL[ds] else ("열세" if x.mean() < -TOL[ds] else "대등")
        s = sel[v][ds]
        rows.append(dict(dataset=ds, variant=v, delta=x.mean(), frac_pos=(x > 0).mean(), n=len(x), verdict=verdict,
                         gamma=s["gamma"], Ka=s["Ka"]))
        lines.append(f"| {v} | ({s['gamma']:g}, {s['Ka']}) | {r.auc_pr:.3f} | {r.hitrate:.3f} | {r.cell_f1:.3f} | "
                     f"{r.smearing:.3f} | {r.infer_ms_per_window:.4f} | {x.mean():+.4f} ({(x > 0).mean():.2f}) | {verdict} |")
    lines.append("")
pd.DataFrame(rows).to_csv(OUT / "paired.csv", index=False, encoding="utf-8-sig")
(OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print("\n".join(lines))
