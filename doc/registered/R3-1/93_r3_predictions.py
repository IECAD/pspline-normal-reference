"""실험 R3-1 등록 예측 계산 (2026-11-13). 실행 전에 적는 숫자만 만든다 — 새 실험은 돌리지 않는다.

정리 1 (doc/정리1_유도.md): AUC-PR = Φ0 + κ V_e + o(V_e), κ 는 (잡음, 이상, 양성률)만의 함수.
A. κ 없는 비율 예측 — 새 이상 칸(spike/shift/drift × 크기 1/3, N 5, MCAR 0.5)에서
   (PS − CF)/(PS − CI) 와 (PS − CG)/(PS − CI) 는 같은 (N, r) 의 V_e 차 비율과 같다. V_e 는 실험 R1-5 칸 28 에서.
B. 끝 간격 마스크(END-g: 마지막 g 시점이 모든 학습 창에서 빔, N 5, 다른 손실 없음) — PS·CI 의 V_e 를 식으로 예측하고
   Φ0 + κ V_e 로 AUC-PR 을 예측한다 (Φ0, κ 는 R1-5 가우스·혼합·크기 2 칸의 합동 적합).
   PS: 명제 4(1) 분산 + 편향, λ 는 R1-5 칸 2 (N 5, 손실 없음) 의 REML λ 중앙값. CI: 끝값 상수 연장의 편향 + 복사된 잡음.
   모양은 R1-5 의 L1 (위상 무작위 사인, 진폭 2 + 노드 변형 0.3·2) 에서 노이즈 없이 평균.
C. 잡음 척도 불변 — s ∈ {0.5, 2}, 이상 크기 2s: V_e/s² 와 Δ/s 가 그대로이므로 모든 참조의 AUC-PR 이 s = 1 값과 같다.
   보간의 편향²/s² 만 바뀐다(L1 에서 작음) — 그 보정량을 계산해 둔다.
출력: analysis/out/93_r3_predictions.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from expR1_5_sim import AMP, NODE_VAR, T, shapes  # noqa: E402
from src.model.pspline import bspline_basis, diff_matrix  # noqa: E402

SRC = ROOT / "results/expR1/sim/metrics.csv"
OUT = ROOT / "analysis/out/93_r3_predictions.md"
SIGMA = 1.0


def ols(y, X):
    X = np.asarray(X, float)
    coef, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    return coef


def pooled_phi0_kappa(df):
    g = df[(df.noise == "G") & (df.p == 0.0) & (df.atype == "mix") & (df.mag == 2.0) & df.ref.isin(["CG", "CF", "CI", "PS", "PS1", "NM"])]
    per = g.groupby(["cell", "ref"], as_index=False).agg(auc=("auc_pr", "mean"), Ve=("Ve", "mean"))
    per = per[per.Ve < 0.5]
    c = ols(per.auc, np.column_stack([np.ones(len(per)), per.Ve]))
    return float(c[0]), float(c[1]), len(per)


def profiles(rng, n):
    """R1-5 와 같은 참 정상 곡선(노드 수준 제외; 수준은 V_e 에 안 들어감): AMP·f + NODE_VAR·AMP·b, (n, T)."""
    return AMP * shapes(rng, "L1", n) + NODE_VAR * AMP * shapes(rng, "L1", n)


def ve_of(err_rows):
    """err (n, T) → 행별 V_e 의 평균."""
    d = err_rows - err_rows.mean(1, keepdims=True)
    return float((d ** 2).mean())


def predict_end_gap(N, lam, n_prof=2000, seed=11):
    """END-g 마스크에서 PS(2차 벌점, λ 고정)·CI 의 E V_e (편향² + 분산). σ = 1."""
    rng = np.random.default_rng(seed)
    M = profiles(rng, n_prof)
    B = bspline_basis(T, T)
    Pt = diff_matrix(T, 2).T @ diff_matrix(T, 2)
    t = np.arange(T)
    rows = []
    for g in range(0, 5):
        obs = np.ones(T, bool)
        if g:
            obs[T - g:] = False
        cnt = obs.astype(float) * N
        C = np.diag(cnt)
        H = B.T @ C @ B + lam * Pt + 1e-6 * (np.trace(B.T @ C @ B) / T) * np.eye(T)
        Hi = np.linalg.inv(H)
        S = B @ Hi @ B.T @ C                                   # 적합 사상 (T × T), 입력 = 칸 평균
        var_ps = SIGMA ** 2 * np.diag(B @ Hi @ (B.T @ C @ B) @ Hi @ B.T)
        bias_ps = (M @ S.T) - M                                # (n, T)
        # CI: 관측 칸 = 칸 평균(편향 0, 분산 σ²/N), 빈 끝 칸 = 마지막 관측값 복사
        bias_ci = np.zeros_like(M)
        if g:
            bias_ci[:, T - g:] = M[:, [T - g - 1]] - M[:, T - g:]
        var_ci = np.full(T, SIGMA ** 2 / N)
        # E V_e = mean_t(Var + b²) − (Var(ē) + b̄²); Var(ē) 는 분산의 공분산 구조가 필요 — 식으로 계산
        # PS: Cov(ê) = σ² S C^{-1}_obs S^T (관측 칸 평균의 공분산 σ²/N) ; CI: 복사 구조
        Cov_ps = SIGMA ** 2 * (B @ Hi @ B.T @ C) @ np.diag(np.where(obs, 1.0 / N, 0.0)) @ (B @ Hi @ B.T @ C).T
        Lci = np.eye(T)
        if g:
            Lci[T - g:, :] = 0.0
            Lci[T - g:, T - g - 1] = 1.0
        Cov_ci = SIGMA ** 2 * Lci @ np.diag(np.where(obs, 1.0 / N, 0.0)) @ Lci.T
        Jc = np.eye(T) - np.ones((T, T)) / T                   # 중심화
        ve_var_ps = float(np.trace(Jc @ Cov_ps @ Jc) / T)
        ve_var_ci = float(np.trace(Jc @ Cov_ci @ Jc) / T)
        rows.append(dict(g=g, Ve_PS_var=ve_var_ps, Ve_PS_bias=ve_of(bias_ps), Ve_CI_var=ve_var_ci, Ve_CI_bias=ve_of(bias_ci),
                         maxvar_PS_end=float(var_ps[T - 1]), lam=lam))
    R = pd.DataFrame(rows)
    R["Ve_PS"] = R.Ve_PS_var + R.Ve_PS_bias
    R["Ve_CI"] = R.Ve_CI_var + R.Ve_CI_bias
    return R


def main():
    df = pd.read_csv(SRC)
    df["Ve"] = df.sdR ** 2 - SIGMA ** 2
    phi0, kappa, npts = pooled_phi0_kappa(df)
    L = ["# 실험 R3-1 등록 예측 (정리 1) — 계산 2026-11-13, 실행 전", "",
         f"합동 적합(R1-5 가우스·혼합·크기 2, V_e < 0.5, {npts} 점): Φ0 = {phi0:.4f}, κ = {kappa:+.4f}.", ""]

    # ---- A ---------------------------------------------------------------
    c28 = df[(df.cell == 28)].groupby("ref").agg(Ve=("Ve", "mean"), auc=("auc_pr", "mean"))
    ve = c28.Ve
    rA_cf = (ve["CF"] - ve["PS"]) / (ve["CI"] - ve["PS"])
    rA_cg = (ve["CG"] - ve["PS"]) / (ve["CI"] - ve["PS"])
    rA_ps1 = (ve["PS1"] - ve["PS"]) / (ve["CI"] - ve["PS"])
    L += ["## A. κ 없는 비율 예측 — 새 이상 칸 (N 5, MCAR r 0.5, L1, 가우스), spike/shift/drift × 크기 1/3", "",
          "같은 (N, r) 이므로 V_e 는 칸 28 과 같다(평균, 반복 100): " + ", ".join(f"{k} {v:.3f}" for k, v in ve.items()) + ".", "",
          f"- 예측 A1: (PS − CF)/(PS − CI) = {rA_cf:.2f} (칸 28 관측값: {(c28.auc['PS'] - c28.auc['CF']) / (c28.auc['PS'] - c28.auc['CI']):.2f}).",
          f"- 예측 A2: (PS − CG)/(PS − CI) = {rA_cg:.2f} (칸 28 관측값: {(c28.auc['PS'] - c28.auc['CG']) / (c28.auc['PS'] - c28.auc['CI']):.2f}; CG 의 V_e {ve['CG']:.2f} 는 전개 영역 경계 — 관측이 작게 나오면 곡률).",
          f"- 예측 A3: (PS1 − PS)/(PS − CI) = {rA_ps1:+.2f} (PS1 이 PS 보다 V_e 가 크면 음수).",
          "- 예측 A4: 여섯 칸 모두에서 참조 순서 = V_e 역순 (PS < PS1 < CI < CF < CG, 칸 28 과 같음; NM 은 L1 에서 꼴찌).",
          "- 판정: 비율은 100 반복 평균의 ±0.25 안이면 맞음(분모가 작은 크기 1 칸은 ±0.5). 순서는 Spearman ≥ 0.9.", ""]

    # ---- B ---------------------------------------------------------------
    lam = float(df[(df.cell == 2) & (df.ref == "PS")].lam_med.median())
    R = predict_end_gap(N=5, lam=lam)
    R["auc_PS"] = phi0 + kappa * R.Ve_PS
    R["auc_CI"] = phi0 + kappa * R.Ve_CI
    R["gap"] = R.auc_PS - R.auc_CI
    L += ["## B. 끝 간격 마스크 END-g (N 5, 손실 없음, L1, 혼합 크기 2): 마지막 g 시점이 모든 학습 창에서 빔", "",
          f"PS 의 λ 는 칸 2 의 REML 중앙값 {lam:.3g} 로 고정해 식을 계산(실제 REML λ̂ 는 다를 수 있음 — 그 차이는 Remark 3 의 1.1배 수준으로 봄).",
          "V_e 는 식(명제 4(1)·4′)과 R1-5 모양 2000 개의 노이즈 없는 편향 평균. AUC-PR 예측 = Φ0 + κ V_e (전개 영역 V_e < 0.5 에서만 숫자로, 그 밖은 '0.5 초과'로 적음).", "",
          "| g | V_e PS (분산+편향²) | V_e CI (분산+편향²) | 끝 칸 Var PS | AUC PS 예측 | AUC CI 예측 | PS − CI 예측 |", "|---|---|---|---|---|---|---|"]
    for _, d in R.iterrows():
        a_ps = f"{d.auc_PS:.3f}" + ("" if d.Ve_PS < 0.5 else " (전개 밖)")
        a_ci = f"{d.auc_CI:.3f}" + ("" if d.Ve_CI < 0.5 else " (전개 밖)")
        L.append(f"| {int(d.g)} | {d.Ve_PS:.3f} ({d.Ve_PS_var:.3f}+{d.Ve_PS_bias:.3f}) | {d.Ve_CI:.3f} ({d.Ve_CI_var:.3f}+{d.Ve_CI_bias:.3f}) | "
                 f"{d.maxvar_PS_end:.2f} | {a_ps} | {a_ci} | {d.gap:+.3f} |")
    gstar = R[(R.g > 0) & (R.gap < 0)].g.min()
    L += ["", f"- 예측 B1: PS − CI 의 부호가 바뀌는 최소 g* = {int(gstar) if pd.notna(gstar) else '없음(g ≤ 4 에서 PS 가 계속 앞섬)'}.",
          "- 예측 B2: g 가 커질수록 두 참조 모두 떨어지고, 순서는 V_e 순서를 따른다(표의 부호).",
          "- 예측 B3: g = 0 은 칸 2 와 같은 설계이므로 PS − CI = +0.033 ± 0.01 (재현).",
          "- 판정: 부호 g* 가 표와 같으면 맞음(±1). 크기는 전개 영역(V_e < 0.5)에서 ±0.02.", ""]

    # ---- C ---------------------------------------------------------------
    rng = np.random.default_rng(5)
    M = profiles(rng, 2000)
    # 보간 편향: MCAR r 0.5, N 10 에서 빈 칸(확률 0.5^10 ≈ 0.001) — 무시 가능; N 5 r 0 에서는 빈 칸 없음 → 편향 0.
    # 따라서 L1 에서 C 의 편향 보정은 0 에 가깝다. 보고용으로 '빈 칸 하나' 평균 보간 편향²를 적어 둔다.
    b2 = []
    for i in range(1, T - 1):
        b2.append(((M[:, i - 1] + M[:, i + 1]) / 2 - M[:, i]) ** 2)
    b2 = float(np.mean(b2))
    base = df[(df.cell.isin([2, 7])) & df.ref.isin(["CG", "CF", "CI", "PS", "PS1"])].groupby(["cell", "ref"]).auc_pr.mean().unstack()
    L += ["## C. 잡음 척도 불변 — s ∈ {0.5, 2}, 이상 크기 2s, (N 5, r 0) 와 (N 10, MCAR 0.5)", "",
          f"내부 빈 칸 하나의 선형 보간 편향² 평균(L1, 노이즈 없음) = {b2:.4f}: s = 0.5 에서 σ² 단위로 {b2 / 0.25:.3f}, 빈 칸 비율(N 10, r 0.5: 0.001)을 곱하면 무시할 수준.",
          "- 예측 C1: 네 칸 모두에서 각 참조의 AUC-PR 이 s = 1 의 같은 설계 칸 값과 ±0.01 안에서 같다:", "",
          "| 설계 (s = 1 기준 칸) | " + " | ".join(base.columns) + " |", "|---|" + "---|" * len(base.columns)]
    for cell, row in base.iterrows():
        L.append(f"| 칸 {cell} ({'N 5, r 0' if cell == 2 else 'N 10, MCAR 0.5'}) | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    L += ["", "- 예측 C2: PS − CI 는 s 와 무관하게 칸 2: +0.033, 칸 7: +0.043 (±0.01).",
          "- 판정: 20 값 중 18 이상이 ±0.01 안이면 맞음. 틀리면 V_e/σ² 이 불변이라는 따름정리 1 의 가정(편향 무시)이 틀린 것.", ""]
    OUT.write_text("\n".join(L), encoding="utf-8", newline="\n")
    json.dump(dict(phi0=phi0, kappa=kappa, lam_cell2=lam, ratios=dict(A1=float(rA_cf), A2=float(rA_cg), A3=float(rA_ps1)),
                   end_gap=R.round(4).to_dict(orient="records")), open(ROOT / "analysis/out/93_r3_predictions.json", "w"), indent=1)
    print("written", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
