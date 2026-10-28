"""실험 R1-4 2부 요약·판정 (doc/실험R1-4_계획.md 2.4절). 허용오차 0.0171 (D6), D1 0.0165, D2 0.0171.

사용법: python experiments/expR1_4_robust_summarize.py --split val | test
출력: results/expR1/robust/summary_<split>.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "expR1" / "robust"
TOL = {"d6": 0.0171, "d1": 0.0165, "d2": 0.0171}


def pair(M, a, b, key, tol):
    x = (M[M.ref == a].set_index(key).auc_pr - M[M.ref == b].set_index(key).auc_pr).dropna()
    v = "이김" if x.mean() > tol else ("짐" if x.mean() < -tol else "대등")
    return x.mean(), (x > 0).mean(), len(x), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    D6 = pd.read_csv(OUT / f"metrics_d6_{a.split}.csv")
    L = [f"# 실험 R1-4 2부 집계 ({a.split})", ""]
    for ev in ("full", "excl"):
        for scale, lab in ((False, "주 크기"), (True, "×3")):
            m = D6[(D6["eval"] == ev) & (D6.scenario.astype(str).str.contains("x") == scale)]
            tab = m.groupby("ref").auc_pr.mean().reindex(["CG", "CI", "CMI", "NM", "NMed", "PS", "PS1", "PSH", "PS1H"])
            L += [f"## D6 {'전체 평가(주)' if ev == 'full' else '이상값 제외 평가(보조, 탐색)'} — {lab}", "",
                  "평균 셀 AUC-PR: " + ", ".join(f"{k} {v:.3f}" for k, v in tab.items()), ""]
            key = ["scan", "scenario", "seed"]
            for tag, x, y in (("H-B1", "CMI", "CI"), ("H-B2", "PSH", "CMI"), ("H-B3", "PS1H", "NMed"), ("H-B4", "PSH", "PS"),
                              ("(보고)", "NMed", "NM"), ("(보고)", "PS1H", "PSH"), ("(보고)", "CMI", "CG")):
                mu, fp, n, v = pair(m, x, y, key, TOL["d6"])
                L.append(f"- {tag} {x} − {y}: {mu:+.4f} (양 {fp:.2f}, n {n}) **{v}**")
            if ev == "excl" and not scale:
                c = m[m.ref == "PS"]
                pb, pa, nb, na = c.pos_before.sum(), c.pos_after.sum(), c.neg_before.sum(), c.neg_after.sum()
                L.append(f"- 제외 칸 수(PS 행 합계, 시나리오·seed 합): 양성 {pb:,} → {pa:,} (제외 {1 - pa / pb:.1%}), "
                         f"음성 {nb:,} → {na:,} (제외 {1 - na / nb:.1%}). 제외는 주입 전 원값으로 정함.")
            L.append("")
    p12 = OUT / f"metrics_d12_{a.split}.csv"
    if p12.exists():
        M = pd.read_csv(p12)
        L += ["## H-B5 D1·D2 깨끗한 조건 (보고)", ""]
        for ds, rx in (("d1", r"^SC\d$"), ("d2", r"^SC\dm$")):
            m = M[(M.dataset == ds) & M.scenario.astype(str).str.match(rx)]
            for x, y in (("CMI", "PS"), ("NMed", "PS")):
                mu, fp, n, v = pair(m, x, y, ["space", "scenario", "seed"], TOL[ds])
                L.append(f"- {ds} {x} − {y}: {mu:+.4f} (양 {fp:.2f}, n {n}) **{v}**")
        L.append("")
    (OUT / f"summary_{a.split}.md").write_text("\n".join(L), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
