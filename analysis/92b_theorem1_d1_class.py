"""92 의 저장 점(92_theorem1_d1_points.csv)으로 동치류 안 회귀 (2026-11-13).

정리 1 의 전개는 max|d_t| 가 작을 때의 것이다. 단위 채움(cell)은 빈 칸 몇 개에 큰 오차(노드–단위 격차)를 두므로
V_e 가 중간이어도 전개 영역 밖일 수 있다. 그래서 동치류 후보(cellfill, cellinterp, PS)만으로 (b) 를 다시 적합하고,
PS 더미가 설명력을 더하는지, 그리고 cell 점의 잔차가 어느 쪽으로 치우치는지 본다.
출력: analysis/out/92b_theorem1_d1_class.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "analysis/out/92_theorem1_d1_points.csv"
OUT = ROOT / "analysis/out/92b_theorem1_d1_class.md"
CLASS = ["cellfill", "cellinterp", "PS"]


def ols(y, X):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    return coef, r, float(r @ r), len(y) - X.shape[1]


def f_test(rss0, dof0, rss1, dof1):
    F = ((rss0 - rss1) / (dof0 - dof1)) / (rss1 / dof1)
    return F, 1.0 - stats.f.cdf(F, dof0 - dof1, dof1)


R = pd.read_csv(SRC).dropna(subset=["auc_test"])
L = ["# 정리 1 — D1 동치류 안 회귀 (cellfill · cellinterp · PS), 단위 채움은 전개 영역 밖", "",
     "| 방 | 점 | x 범위 | Φ0 | κ | R² | 잔차 SD | PS 더미 | interp 더미 | F | p | cell 점 평균 잔차(같은 식 기준) | cell 점 수 |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for sp, sub in R.groupby("space"):
    cls = sub[sub["basis"].isin(CLASS)]
    y = cls["auc_test"].to_numpy()
    X0 = np.column_stack([np.ones(len(cls)), cls["x"]])
    c0, r0, rss0, dof0 = ols(y, X0)
    D = pd.get_dummies(cls["basis"]).reindex(columns=["cellinterp", "PS"], fill_value=0).to_numpy(float)
    c1, r1, rss1, dof1 = ols(y, np.column_stack([X0, D]))
    F, p = f_test(rss0, dof0, rss1, dof1)
    r2 = 1 - rss0 / float(((y - y.mean()) ** 2).sum())
    cell = sub[(sub["basis"] == "cell") & (sub["x"] < 2.0)]
    resid_cell = cell["auc_test"] - (c0[0] + c0[1] * cell["x"])
    L.append(f"| {sp} | {len(cls)} | {cls['x'].min():.2f}–{cls['x'].max():.2f} | {c0[0]:.3f} | {c0[1]:+.3f} | {r2:.3f} | "
             f"{np.sqrt(rss0 / dof0):.4f} | {c1[3]:+.4f} | {c1[2]:+.4f} | {F:.2f} | {p:.3g} | {resid_cell.mean():+.4f} | {len(cell)} |")
L += ["", "x = σ_R 비율² − 1 (방마다 상수배의 V_e). cell 점 잔차는 동치류 식 Φ0 + κx 에 대한 것; 음수면 같은 x 에서 단위 채움이 더 나쁨(큰 오차가 몇 칸에 몰린 경우, 전개 영역 밖).",
      "", "## 동치류 안 PS − 보간 차이와 κ·Δx", "", "| 방 | 조건 | 시드 | x interp | x PS | Δx | AUC PS − interp (test) | κΔx |", "|---|---|---|---|---|---|---|---|"]
for sp, sub in R.groupby("space"):
    cls = sub[sub["basis"].isin(CLASS)]
    y = cls["auc_test"].to_numpy()
    c0, *_ = ols(y, np.column_stack([np.ones(len(cls)), cls["x"]]))
    piv = sub[sub["basis"].isin(["cellinterp", "PS"])].pivot_table(index=["level", "cond_seed"], columns="basis", values=["x", "auc_test"]).reset_index()
    piv.columns = ["_".join(c).strip("_") for c in piv.columns]
    for _, d in piv.iterrows():
        dx = d["x_cellinterp"] - d["x_PS"]
        L.append(f"| {sp} | {d['level']} | {d['cond_seed']} | {d['x_cellinterp']:.3f} | {d['x_PS']:.3f} | {dx:+.3f} | "
                 f"{d['auc_test_PS'] - d['auc_test_cellinterp']:+.4f} | {-c0[1] * dx:+.4f} |")
OUT.write_text("\n".join(L), encoding="utf-8", newline="\n")
print("written", OUT.relative_to(ROOT))
