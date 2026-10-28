"""Mathematics 논문 Fig 1 (모델 개요). 합성 예시 데이터(캡션에 'illustrative' 명시), 원寸 전폭·공통 스타일(fig_style.py).
(a) 한 노드의 정상 윈도우 3개: 센서 행 × 시점 행렬, 흰 칸 = 결측
(b) 한 행: 관측, 참 패턴, 셀 평균(λ=0, 빈 칸에서 정의 안 됨), P-spline(REML λ), 노드 평균(d=1, λ→∞)
(c) 테스트 윈도우의 표준화 잔차 점수 A = |R−μ|/σ, 테두리 = 검증 97.5 백분위 초과 셀
그림 안 제목은 두지 않는다(설명은 캡션). 캡션에 쓰는 값(λ̂, 임계값, 표시 셀)은 실행 출력으로 확인한다.
사용법: python analysis/64_math_fig1.py → paper/mathematics/figures/fig1_overview.{pdf,png}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import fig_style as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from src.model.pspline import SurfaceFit, bspline_basis, diff_matrix, kron_penalties  # noqa: E402

OUT = fs.out_dir(ROOT)            # 컬러판 figures/, 흑백판(--bw) figures_bw/
fs.setup()

rng = np.random.default_rng(3)
S, T, N, NV = 4, 12, 5, 12
t = np.arange(T)
truth = np.stack([0.8 * np.sin(2 * np.pi * (t / T + ph)) + a for ph, a in zip([0.1, 0.45, 0.7, 0.2], [0.0, 0.5, -0.4, 0.2])])  # (S, T)
sig = 0.18
Xtr = truth[None] + rng.normal(0, sig, (N, S, T))
Wtr = (rng.random((N, S, T)) > 0.55).astype(float)                       # 55 % 결측 → 빈 칸 생김
Wtr[:, 1, 4] = 0                                                          # 예시용: 행 2 시점 5 를 빈 칸으로 고정
Xva = truth[None] + rng.normal(0, sig, (NV, S, T))
Xte = truth + rng.normal(0, sig, (S, T)); Xte[1, 5:9] += 1.6             # 행 2, 시점 6–9 에 이상 주입

Bt = bspline_basis(T, T)
fit = SurfaceFit(Bt, kron_penalties(S, T, diff_matrix(T, 2), {})).fit(Xtr, Wtr)
M_ps = fit.predict()
cnt = Wtr.sum(0); M_cell = np.where(cnt > 0, (Wtr * Xtr).sum(0) / np.maximum(cnt, 1), np.nan)
M_z1 = np.repeat(((Wtr * Xtr).sum((0, 2)) / Wtr.sum((0, 2)))[:, None], T, 1)
Rva = Xva - M_ps[None]; mu, sd = Rva.mean((0, 2)), Rva.std((0, 2))
A = np.abs((Xte - M_ps) - mu[:, None]) / (sd[:, None] + 1e-9)
thr = np.quantile(np.abs(Rva - mu[None, :, None]) / (sd[None, :, None] + 1e-9), 0.975)

fig = plt.figure(figsize=(fs.FULL_W, 6.6 * fs.CM), layout="constrained")
sf = fig.subfigures(1, 3, width_ratios=[1.0, 1.25, 1.0], wspace=0.02)
# (a) 정상 윈도우 3개 (세로로 쌓음)
axa = sf[0].subplots(3, 1, sharex=True)
for i, ax in enumerate(axa):
    img = np.where(Wtr[i] > 0, Xtr[i], np.nan)
    ax.imshow(img, cmap=fs.FIELD_CMAP, vmin=-1.6, vmax=1.6, aspect="auto", interpolation="nearest")
    for s in range(S):
        for k in range(T):
            if Wtr[i, s, k] == 0:
                ax.add_patch(plt.Rectangle((k - 0.5, s - 0.5), 1, 1, fill=True, fc="white", ec="#b0b0b0", lw=0.4))
    ax.set_yticks([0, 3]); ax.set_yticklabels(["1", "4"]); ax.set_ylabel(f"$i$={i + 1}", rotation=0, ha="right", va="center", labelpad=4)
    ax.tick_params(length=2)
axa[-1].set_xticks([0, 4, 8, 11]); axa[-1].set_xticklabels(["1", "5", "9", "12"]); axa[-1].set_xlabel("time step $t$")
sf[0].supylabel("window $i$, sensor row $s$", fontsize=8)
fs.panel(axa[0], "a", x=-0.18)
# (b) 한 행의 곡선
axb = sf[1].subplots(1, 1); s = 1
for i in range(N):
    m = Wtr[i, s] > 0
    axb.plot(t[m] + 1, Xtr[i, s][m], "o", ms=3, color="#555555", alpha=0.5, mew=0, label="observations" if i == 0 else None)
axb.plot(t + 1, truth[s], "-", color="#000000", lw=0.7, alpha=0.6, label="true pattern")
axb.plot(t + 1, M_cell[s], marker="o", ls="-", color=fs.STYLE["cell"]["color"], ms=3.8, lw=1.1, mfc="white", mew=1.0, label="cell mean ($\\lambda=0$)")
axb.plot(t + 1, M_ps[s], "-", color=fs.STYLE["PS"]["color"], lw=2.0, label="P-spline, REML $\\hat\\lambda$")
axb.plot(t + 1, M_z1[s], "--", color=fs.OI["black"], lw=1.1, label="node mean ($d=1,\\ \\lambda\\to\\infty$)")
axb.axvspan(4.5, 5.5, color="#d9d9d9", alpha=0.6, lw=0, zorder=0)
axb.text(5, -0.62, "empty\ncell", ha="center", va="bottom", fontsize=7, color="#555555")
axb.set_xlabel("time step $t$"); axb.set_ylabel("value, sensor row $s$ = 2"); axb.set_xticks([1, 3, 5, 7, 9, 11]); axb.set_ylim(-0.65, 1.75)
axb.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, columnspacing=1.0, handlelength=2.0)
fs.panel(axb, "b")
# (c) 점수 히트맵
axc = sf[2].subplots(1, 1)
im = axc.imshow(A, cmap=fs.SCORE_CMAP, vmin=0, vmax=max(6, A.max()), aspect="auto", interpolation="nearest")
flag = []
for s_ in range(S):
    for k in range(T):
        if A[s_, k] > thr:
            flag.append((s_ + 1, k + 1))
            fs.box(axc, k - 0.5, s_ - 0.5)
axc.set_xticks([0, 4, 8, 11]); axc.set_xticklabels(["1", "5", "9", "12"]); axc.set_yticks(range(S)); axc.set_yticklabels([str(i + 1) for i in range(S)])
axc.set_xlabel("time step $t$"); axc.set_ylabel("sensor row $s$")
cb = sf[2].colorbar(im, ax=axc, fraction=0.06, pad=0.03); cb.set_label("score $A_{st}$")
fs.panel(axc, "c")
fs.save(fig, OUT, "fig1_overview")
print(f"caption values: lambda_hat={fit.lam['t']:.2g}, threshold={thr:.2f}, flagged cells (s,t)={flag}, empty cells of row 2={[k + 1 for k in range(T) if cnt[1, k] == 0]}")
# 그래픽 초록은 본문 그림과 달라야 하므로(MDPI) 별도 스크립트 analysis/72_graphical_abstract.py 가 새로 그린다.
