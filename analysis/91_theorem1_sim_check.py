"""정리 1 회고 검증 (시뮬레이션 저장 출력, 2026-11-13).

정리 1: 행 중심화된 점수에서 참조 오차는 순위 기반 탐지 지표에 2차로, V_e 하나를 통해서만 들어온다.
검증: results/expR1/sim/metrics.csv (실험 R1-5, 43칸 × 100반복 × 참조 10종) 에서
  V_e 추정 = sdR² − σ²   (명제 4′: σ_R² = σ² + V_e; 가우스 칸은 σ = 1)
  (a) 칸 안에서 참조들의 평균 AUC-PR 순서가 평균 V_e 의 역순과 같은가 (Spearman).
  (b) 가우스·혼합 이상·크기 2·오염 없음 칸들을 모두 모았을 때 (칸, 참조) 점들이 V_e 의 한 곡선 위에 놓이는가
      — V_e 가 작은 영역에서 AUC-PR ~ Φ0 + κ V_e 를 적합하고, 참조 더미가 설명력을 더하는지 (F 검정).
  (c) 같은 V_e 에서 오차의 '자리'(빈 칸 대 관측 칸의 RMSE 분배)가 설명력을 더하는지.
출력: analysis/out/91_theorem1_sim_check.md, 91_theorem1_sim_points.csv
사용법: python analysis/91_theorem1_sim_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/expR1/sim/metrics.csv"
OUT_MD = ROOT / "analysis/out/91_theorem1_sim_check.md"
OUT_CSV = ROOT / "analysis/out/91_theorem1_sim_points.csv"

SIGMA = 1.0  # 가우스 잡음 SD (실험 R1-5 설계)
REFS_MAIN = ["CG", "CF", "CI", "PS", "PS1", "NM"]      # 제곱 손실 참조 + 노드 평균(동치류 밖)
REFS_CLASS = ["CF", "CI", "PS", "PS1"]                  # 따름정리 1 의 동치류 후보 (빈 칸을 이웃에서 잇는 참조)


def ols(y, X):
    """최소제곱: (coef, resid, rss, dof)"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    return coef, r, float(r @ r), len(y) - X.shape[1]


def f_test(rss0, dof0, rss1, dof1):
    num = (rss0 - rss1) / (dof0 - dof1)
    den = rss1 / dof1
    F = num / den
    p = 1.0 - stats.f.cdf(F, dof0 - dof1, dof1)
    return F, p


def main():
    df = pd.read_csv(SRC)
    df["Ve"] = df["sdR"] ** 2 - SIGMA ** 2
    df["mse"] = (1 - df["empty_frac"]) * df["rmse_obs"] ** 2 + df["empty_frac"].fillna(0) * df["rmse_empty"].fillna(0) ** 2
    g = df[(df["noise"] == "G") & (df["p"] == 0.0)].copy()
    lines = ["# 정리 1 회고 검증 — 시뮬레이션 저장 출력 (실험 R1-5)", "",
             f"원천: `{SRC.relative_to(ROOT)}`; V_e 추정 = sdR² − 1 (명제 4′, 가우스 σ = 1). 가우스·오염 없음 칸만.", ""]

    # ---- (a) 칸 안 순서 -------------------------------------------------
    per = (g[g["ref"].isin(REFS_MAIN)]
           .groupby(["cell", "tags", "N", "L", "miss", "r", "atype", "mag", "ref"], as_index=False)
           .agg(auc=("auc_pr", "mean"), Ve=("Ve", "mean"), mse=("mse", "mean"),
                rmse_obs=("rmse_obs", "mean"), rmse_empty=("rmse_empty", "mean"), empty=("empty_frac", "mean"),
                sdR=("sdR", "mean"), n=("auc_pr", "size")))
    per.to_csv(OUT_CSV, index=False)
    rows = []
    for cell, sub in per.groupby("cell"):
        rho, pv = stats.spearmanr(sub["auc"], -sub["Ve"])
        order_auc = " > ".join(sub.sort_values("auc", ascending=False)["ref"])
        order_ve = " < ".join(sub.sort_values("Ve")["ref"])
        d = sub.iloc[0]
        rows.append((cell, d["tags"], d["N"], d["L"], d["miss"], d["r"], d["atype"], d["mag"], f"{rho:+.2f}", order_auc, order_ve))
    lines += ["## (a) 칸 안에서 AUC-PR 순서 대 V_e 순서 (참조 " + ", ".join(REFS_MAIN) + ")", "",
              "| 칸 | 꼬리표 | N | L | 결측 | r | 이상 | 크기 | Spearman(AUC, −V_e) | AUC 순서 | V_e 순서(작은 것부터) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    rhos = np.array([float(r[8]) for r in rows])
    lines += ["", f"Spearman 중앙값 {np.median(rhos):+.2f}, 최소 {rhos.min():+.2f}; ρ ≥ 0.8 인 칸 {int((rhos >= 0.8).sum())}/{len(rhos)}.", ""]

    # ---- (b) 한 곡선 ---------------------------------------------------------
    pool = per[(per["atype"] == "mix") & (per["mag"] == 2.0)].copy()      # 같은 f, Δ, π
    small = pool[pool["Ve"] < 0.5].copy()                                    # 2차 전개 영역
    y = small["auc"].to_numpy()
    X0 = np.column_stack([np.ones(len(small)), small["Ve"]])
    c0, r0, rss0, dof0 = ols(y, X0)
    dummies = pd.get_dummies(small["ref"], drop_first=True).to_numpy(float)
    X1 = np.column_stack([X0, dummies])
    c1, r1, rss1, dof1 = ols(y, X1)
    F, p = f_test(rss0, dof0, rss1, dof1)
    r2 = 1 - rss0 / float(((y - y.mean()) ** 2).sum())
    lines += ["## (b) 가우스·혼합·크기 2 칸을 모은 (칸, 참조) 점 — AUC-PR 대 V_e", "",
              f"점 {len(pool)}개 (V_e < 0.5 인 점 {len(small)}개). V_e < 0.5 에서 AUC-PR = Φ0 + κ V_e 적합:",
              f"Φ0 = {c0[0]:.4f}, κ = {c0[1]:+.4f} (AUC-PR / V_e 단위), R² = {r2:.3f}, 잔차 SD = {np.sqrt(rss0 / dof0):.4f}.",
              f"참조 더미를 더했을 때 F = {F:.2f}, p = {p:.3g} (더미 계수: " +
              ", ".join(f"{k} {v:+.4f}" for k, v in zip(pd.get_dummies(small['ref'], drop_first=True).columns, c1[2:])) + ").", ""]
    # 2차 항 확인
    X2 = np.column_stack([X0, small["Ve"] ** 2])
    c2, r2_, rss2, dof2 = ols(y, X2)
    F2, p2 = f_test(rss0, dof0, rss2, dof2)
    lines += [f"V_e² 항 추가: 계수 {c2[2]:+.4f}, F = {F2:.2f}, p = {p2:.3g}.", ""]
    # 전체 범위 표
    lines += ["| 칸 | N | L | 결측 | r | 참조 | V_e | 빈 칸 비율 | RMSE 관측 | RMSE 빈 칸 | AUC-PR | 적합 Φ0+κV_e | 잔차 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    pool["fit"] = c0[0] + c0[1] * pool["Ve"]
    pool["resid"] = pool["auc"] - pool["fit"]
    for _, d in pool.sort_values(["Ve"]).iterrows():
        re_txt = "—" if pd.isna(d["rmse_empty"]) else f"{d['rmse_empty']:.3f}"
        lines.append(f"| {d['cell']} | {d['N']} | {d['L']} | {d['miss']} | {d['r']} | {d['ref']} | {d['Ve']:.3f} | {d['empty']:.3f} | "
                     f"{d['rmse_obs']:.3f} | {re_txt} | {d['auc']:.3f} | {d['fit']:.3f} | {d['resid']:+.3f} |")
    lines.append("")

    # ---- (c) 자리 ------------------------------------------------------------
    sm = small[small["empty"] > 0].copy()
    if len(sm) > 5:
        y = sm["auc"].to_numpy()
        Xa = np.column_stack([np.ones(len(sm)), sm["Ve"]])
        share = (sm["empty"] * sm["rmse_empty"].fillna(0) ** 2) / sm["mse"].replace(0, np.nan)
        Xb = np.column_stack([Xa, share.fillna(0)])
        ca, ra, rssa, dofa = ols(y, Xa)
        cb, rb, rssb, dofb = ols(y, Xb)
        Fc, pc = f_test(rssa, dofa, rssb, dofb)
        lines += ["## (c) 같은 V_e 에서 오차의 자리(빈 칸 몫)가 설명력을 더하는가", "",
                  f"빈 칸이 있는 점 {len(sm)}개. '빈 칸 오차 몫'(빈 칸 MSE / 전체 MSE) 항 추가: 계수 {cb[2]:+.4f}, F = {Fc:.2f}, p = {pc:.3g}.", ""]

    # ---- 동치류 안 차이 대 1/c ---------------------------------------------
    cls = per[(per["atype"] == "mix") & (per["mag"] == 2.0) & per["ref"].isin(["CI", "PS"])]
    piv = cls.pivot_table(index=["cell", "N", "miss", "r", "L"], columns="ref", values=["auc", "Ve"]).reset_index()
    piv.columns = ["_".join(c).strip("_") for c in piv.columns]
    piv["gap"] = piv["auc_PS"] - piv["auc_CI"]
    piv["dVe"] = piv["Ve_CI"] - piv["Ve_PS"]
    piv["c_eff"] = piv["N"] * (1 - piv["r"].where(piv["miss"] == "MCAR", 0.5))
    lines += ["## PS − CI 와 V_e 차이, 유효 관측 수", "",
              "| 칸 | N | 결측 | r | L | c_eff | V_e CI | V_e PS | ΔV_e | AUC PS − CI | κ·ΔV_e 예측 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, d in piv.sort_values("c_eff").iterrows():
        lines.append(f"| {d['cell']} | {d['N']} | {d['miss']} | {d['r']} | {d['L']} | {d['c_eff']:.1f} | {d['Ve_CI']:.3f} | {d['Ve_PS']:.3f} | "
                     f"{d['dVe']:+.3f} | {d['gap']:+.3f} | {-c0[1] * d['dVe']:+.3f} |")
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    msg = "\n".join(lines[:12]) + f"\n... written {OUT_MD.relative_to(ROOT)}"
    print(msg.encode("utf-8", "replace").decode("utf-8") if sys.stdout.encoding and "utf" in sys.stdout.encoding.lower()
          else msg.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
