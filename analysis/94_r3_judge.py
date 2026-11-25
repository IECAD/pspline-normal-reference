"""실험 R3-1 판정 (doc/실험R3-1_계획.md 4절의 규칙, 예측은 analysis/out/93_r3_predictions.json).

입력: results/expR3/sim/metrics.csv (칸 100–114 × 반복 100 × 참조 10), results/expR1/sim/metrics.csv (C 의 기준 칸 2·7)
출력: analysis/out/94_r3_judge.md
사용법: python analysis/94_r3_judge.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "results/expR3/sim/metrics.csv"
OLD = ROOT / "results/expR1/sim/metrics.csv"
PRED = json.load(open(ROOT / "analysis/out/93_r3_predictions.json"))
OUT = ROOT / "analysis/out/94_r3_judge.md"


def main():
    d = pd.read_csv(NEW)
    o = pd.read_csv(OLD)
    s_of = d["noise"].astype(str).str.extract(r"@([\d.]+)")[0].astype(float).fillna(1.0)
    d["Ve"] = d["sdR"] ** 2 - s_of ** 2
    per = d.groupby(["cell", "tags", "N", "miss", "r", "atype", "mag", "noise", "ref"], as_index=False).agg(
        auc=("auc_pr", "mean"), auc_sd=("auc_pr", "std"), Ve=("Ve", "mean"), n=("auc_pr", "size"))
    L = ["# 실험 R3-1 판정 — 정리 1 등록 예측 (실행 뒤, 2026-11-13)", "",
         f"예측 파일: Φ0 = {PRED['phi0']:.4f}, κ = {PRED['kappa']:+.4f}; 반복 {int(per.n.max())}.", ""]
    verdicts = []

    # ---- A ----------------------------------------------------------------
    L += ["## A. κ 없는 비율 (N 5, MCAR 0.5; 칸 100–105)", "",
          "| 칸 | 이상 | 크기 | PS − CI | PS − CF | PS − CG | PS1 − PS | A1 (PS−CF)/(PS−CI) | A2 (PS−CG)/(PS−CI) | A3 | Spearman(AUC, −V_e) | 순서 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    pa = PRED["ratios"]
    okA = []
    for cell, sub in per[per.tags == "A"].groupby("cell"):
        a = sub.set_index("ref").auc
        v = sub.set_index("ref").Ve
        dci, dcf, dcg, d1 = a["PS"] - a["CI"], a["PS"] - a["CF"], a["PS"] - a["CG"], a["PS1"] - a["PS"]
        r1, r2, r3 = dcf / dci, dcg / dci, d1 / dci
        refs6 = ["CG", "CF", "CI", "PS", "PS1", "NM"]
        rho = stats.spearmanr(a[refs6], -v[refs6])[0]
        tol = 0.5 if float(sub.mag.iloc[0]) == 1.0 else 0.25
        ok = (abs(r1 - pa["A1"]) <= tol, abs(r2 - pa["A2"]) <= tol, abs(r3 - pa["A3"]) <= tol, rho >= 0.9)
        okA.append(ok)
        order = " > ".join(a[refs6].sort_values(ascending=False).index)
        L.append(f"| {cell} | {sub.atype.iloc[0]} | {sub.mag.iloc[0]:.0f} | {dci:+.3f} | {dcf:+.3f} | {dcg:+.3f} | {d1:+.3f} | "
                 f"{r1:.2f} {'✓' if ok[0] else '✗'} | {r2:.2f} {'✓' if ok[1] else '✗'} | {r3:+.2f} {'✓' if ok[2] else '✗'} | {rho:+.2f} {'✓' if ok[3] else '✗'} | {order} |")
    okA = np.array(okA)
    L += ["", f"예측: A1 {pa['A1']:.2f}, A2 {pa['A2']:.2f}, A3 {pa['A3']:+.2f} (허용 ±0.25, 크기 1 칸 ±0.5); A4 Spearman ≥ 0.9.",
          f"맞음: A1 {okA[:, 0].sum()}/6, A2 {okA[:, 1].sum()}/6, A3 {okA[:, 2].sum()}/6, A4 {okA[:, 3].sum()}/6.", ""]
    verdicts.append(("A", okA.all()))

    # ---- B ----------------------------------------------------------------
    L += ["## B. 끝 간격 END-g (N 5; 칸 106–110)", "",
          "| g | V_e PS 관측 | V_e PS 예측 | V_e CI 관측 | V_e CI 예측 | AUC PS | AUC CI | PS − CI 관측 | 예측 | λ̂ 중앙값 | PS1 | PS1 − CI |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    pb = {int(x["g"]): x for x in PRED["end_gap"]}
    gaps = {}
    for cell, sub in per[per.tags == "B"].groupby("cell"):
        g = int(round(float(sub.r.iloc[0]))) if sub.miss.iloc[0] == "END" else 0
        a = sub.set_index("ref").auc
        v = sub.set_index("ref").Ve
        lam = d[(d.cell == cell) & (d.ref == "PS")].lam_med.median()
        gaps[g] = a["PS"] - a["CI"]
        L.append(f"| {g} | {v['PS']:.3f} | {pb[g]['Ve_PS']:.3f} | {v['CI']:.3f} | {pb[g]['Ve_CI']:.3f} | {a['PS']:.3f} | {a['CI']:.3f} | "
                 f"{gaps[g]:+.3f} | {pb[g]['gap']:+.3f} | {lam:.2f} | {a['PS1']:.3f} | {a['PS1'] - a['CI']:+.3f} |")
    gstar_obs = min([g for g in sorted(gaps) if g > 0 and gaps[g] < 0], default=None)
    gstar_pred = 3
    okB1 = gstar_obs is not None and abs(gstar_obs - gstar_pred) <= 1
    aucs = per[per.tags == "B"].pivot_table(index="ref", columns="cell", values="auc")
    mono = all(np.all(np.diff(aucs.loc[r].to_numpy()) <= 0.005) for r in ("PS", "CI"))
    signs_ok = all((gaps[g] > 0) == (pb[g]["gap"] > 0) for g in gaps if g != gstar_obs and abs(g - gstar_pred) > 1)
    okB3 = abs(gaps[0] - 0.033) <= 0.01
    L += ["", f"B1: 부호가 바뀌는 g* 관측 {gstar_obs} (예측 3, 허용 ±1) → {'맞음' if okB1 else '틀림'}.",
          f"B2: g 에 따라 PS·CI 단조 감소 {'맞음' if mono else '틀림'}; 허용 범위 밖 g 의 부호 일치 {'맞음' if signs_ok else '틀림'}.",
          f"B3: g = 0 재현 PS − CI = {gaps[0]:+.3f} (예측 +0.033 ± 0.01) → {'맞음' if okB3 else '틀림'}.", ""]
    verdicts.append(("B", okB1 and mono and signs_ok and okB3))

    # ---- C ----------------------------------------------------------------
    L += ["## C. 잡음 척도 불변 (칸 111–114)", "",
          "| 칸 | 설계 | s | 참조 | AUC 관측 | 기준(s = 1) | 차 | ±0.01 |", "|---|---|---|---|---|---|---|---|"]
    base = o[o.cell.isin([2, 7]) & o.ref.isin(["CF", "CG", "CI", "PS", "PS1"])].groupby(["cell", "ref"]).auc_pr.mean()
    hits, tot, c2 = 0, 0, []
    for cell, sub in per[per.tags == "C"].groupby("cell"):
        N, r = int(sub.N.iloc[0]), float(sub.r.iloc[0])
        ref_cell = 2 if (N == 5 and r == 0.0) else 7
        s = float(str(sub.noise.iloc[0]).split("@")[1])
        a = sub.set_index("ref").auc
        for ref in ["CF", "CG", "CI", "PS", "PS1"]:
            diff = a[ref] - base[(ref_cell, ref)]
            ok = abs(diff) <= 0.01
            hits += ok
            tot += 1
            L.append(f"| {cell} | {'N 5, r 0' if ref_cell == 2 else 'N 10, MCAR 0.5'} | {s} | {ref} | {a[ref]:.3f} | {base[(ref_cell, ref)]:.3f} | {diff:+.3f} | {'✓' if ok else '✗'} |")
        gap_ref = base[(ref_cell, "PS")] - base[(ref_cell, "CI")]
        c2.append((cell, s, a["PS"] - a["CI"], gap_ref))
    okC1 = hits >= 18
    okC2 = all(abs(x[2] - x[3]) <= 0.01 for x in c2)
    L += ["", f"C1: {hits}/{tot} 값이 ±0.01 안 (규칙 ≥ 18) → {'맞음' if okC1 else '틀림'}.",
          "C2: PS − CI " + "; ".join(f"칸 {c} (s {s}): {g:+.3f} (기준 {gr:+.3f})" for c, s, g, gr in c2) + f" → {'맞음' if okC2 else '틀림'}.", ""]
    verdicts.append(("C", okC1 and okC2))

    L += ["## 종합", ""] + [f"- {k}: {'맞음' if v else '틀림'}" for k, v in verdicts] + [""]
    OUT.write_text("\n".join(L), encoding="utf-8", newline="\n")
    print("written", OUT.relative_to(ROOT), [(k, bool(v)) for k, v in verdicts])


if __name__ == "__main__":
    main()
