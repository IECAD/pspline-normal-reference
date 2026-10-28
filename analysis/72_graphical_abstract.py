"""MDPI 그래픽 초록 (본문 그림과 다른 새 그림; MDPI: 본문 그림 그대로·하위 그림 겹치기·글 초록+그림 조합 불가, PNG/JPEG/TIFF,
최소 1100 × 560 px). 한 장으로 논문의 줄거리: 결측 있는 정상 기록 → 한 행의 세 기준(셀 평균·노드 평균·P-spline) → 70 % 손실에서의 탐지.
웹에서 작게(약 550 px 폭) 보이므로 글자를 크게, 요소를 적게. 색은 본문과 같은 D안(제안 모델만 파랑).
수치는 1차 산출물에서 읽는다(results/exp6/comparators_missing/metrics_d1_test.csv, exp6/imputation/metrics_d1_test.csv,
expR1/factorial/table_test.csv; D1 테스트, i.i.d. r = 0.7). 1차 수정(2026-10-29): 단순 대체(보간)와 보간 참조를 받은 탐지기를 더해
"단위 채움 셀 평균은 무너지지만 단순 대체가 격차 대부분을 메운다"는 수정본 결론에 맞춤. 제출본 그림은 graphical_abstract_submitted.png.
사용법: python analysis/72_graphical_abstract.py → paper/mathematics/figures/graphical_abstract.png (2200 × 1120 px)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import fig_style as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from src.model.pspline import SurfaceFit, bspline_basis, diff_matrix, kron_penalties  # noqa: E402

fs.setup()
sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "axes.linewidth": 1.2})
BLUE, BLACK, GREY = fs.STYLE["PS"]["color"], "#000000", "#8C8C8C"

# ---------------- 결과 수치 (1차 산출물) ----------------
m = pd.read_csv(ROOT / "results/exp6/comparators_missing/metrics_d1_test.csv", encoding="utf-8-sig")
m = m[m.scenario.astype(str).str.match(r"^SC\d$")]; m["level"] = m.level.astype(str)
m["key"] = np.where(m.family == "base", "base", m.family + ":" + m.level)
g = m.groupby(["key", "model"]).auc_pr.mean().unstack()
r07, r0 = g.loc["C:0.7"], g.loc["base"]
comp07 = r07.drop(["PS", "cell"]); comp0 = r0.drop(["PS", "cell"])
imp = pd.read_csv(ROOT / "results/exp6/imputation/metrics_d1_test.csv", encoding="utf-8-sig")
imp = imp[imp.scenario.astype(str).str.match(r"^SC\d$") & (imp.family == "C") & (imp.level.astype(str) == "r=0.7")]
interp07 = imp[imp.basis == "cellinterp"].auc_pr.mean()
fac = pd.read_csv(ROOT / "results/expR1/factorial/table_test.csv", encoding="utf-8-sig")
det_ci = fac[(fac.dataset == "d1") & (fac.ck == "C|0.7") & (fac.detector != "self")].CI

# ---------------- 예시 행 (합성; Fig 1 과 같은 생성 규칙이지만 새로 그린 단순화 도식) ----------------
rng = np.random.default_rng(3)
S, T, N = 4, 12, 5
t = np.arange(T)
truth = np.stack([0.8 * np.sin(2 * np.pi * (t / T + ph)) + a for ph, a in zip([0.1, 0.45, 0.7, 0.2], [0.0, 0.5, -0.4, 0.2])])
X = truth[None] + rng.normal(0, 0.18, (N, S, T)); W = (rng.random((N, S, T)) > 0.55).astype(float); W[:, 1, 4] = 0
fit = SurfaceFit(bspline_basis(T, T), kron_penalties(S, T, diff_matrix(T, 2), {})).fit(X, W)
M_ps = fit.predict(); cnt = W.sum(0)
M_cell = np.where(cnt > 0, (W * X).sum(0) / np.maximum(cnt, 1), np.nan)
M_z1 = (W * X).sum((0, 2)) / W.sum((0, 2))

fig = plt.figure(figsize=(11, 5.6), facecolor="white")
# 머리 띠: 세 단계 제목
heads = [(0.155, "Incomplete normal records"), (0.49, "One P-spline family, two limits"), (0.83, "Detection, 70 % cell loss (D1)")]
for x, h in heads:
    fig.text(x, 0.93, h, ha="center", va="center", fontsize=15, fontweight="bold")
for x0 in (0.318, 0.648):
    fig.patches.append(FancyArrowPatch((x0 - 0.014, 0.575), (x0 + 0.014, 0.575), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=28,
                                       color="#444444", lw=2.2))

# (1) 결측 있는 정상 윈도우: 한 윈도우의 센서 × 시간 격자
ax = fig.add_axes([0.035, 0.33, 0.25, 0.49])
img = np.where(W[0] > 0, X[0], np.nan)
ax.imshow(img, cmap="Greys", vmin=-2.2, vmax=1.8, aspect="auto", interpolation="nearest")
for s in range(S):
    for k in range(T):
        if W[0, s, k] == 0:
            ax.add_patch(Rectangle((k - 0.5, s - 0.5), 1, 1, fc="white", ec="#B0B0B0", lw=0.8))
ax.add_patch(Rectangle((4 - 0.5, 1 - 0.5), 1, 1, fill=False, ec=BLACK, lw=2.4))
ax.annotate("empty cell: no cell mean,\nit must be filled", xy=(4, 1.5), xytext=(5.5, 4.55), fontsize=12.5, ha="center", va="top",
            arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=1.4), annotation_clip=False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_ylabel("sensors", fontsize=13); ax.set_title("time →", fontsize=12.5, loc="right", pad=3)
for sp in ax.spines.values():
    sp.set_visible(True); sp.set_color("#888888")

# (2) 한 행: 셀 평균(빈 칸에서 끊김) · 노드 평균(평평) · P-spline(빈 칸을 이음)
ax = fig.add_axes([0.37, 0.33, 0.25, 0.49]); s = 1
ax.axvspan(4.5, 5.5, color="#E6E6E6", lw=0, zorder=0)
ax.plot(t + 1, M_cell[s], "o-", color=BLACK, mfc="white", mew=1.5, ms=6, lw=1.6, label="cell mean ($\\lambda=0$)")
ax.plot(t + 1, np.full(T, M_z1[s]), "--", color=BLACK, lw=1.8, label="node mean ($\\lambda\\to\\infty$)")
ax.plot(t + 1, M_ps[s], "-", color=BLUE, lw=3.6, label="P-spline, REML $\\hat\\lambda$")
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("one sensor row, time →", fontsize=12.5, loc="right", pad=3)
ax.set_ylim(-0.9, 1.75)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=1, fontsize=12, frameon=False, handlelength=2.6)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# (3) 결과: 70 % 손실에서 셀 AUC-PR (D1 테스트) — 단위 채움은 무너지고, 단순 대체·보간 참조는 P-spline 에 가깝다
ax = fig.add_axes([0.80, 0.33, 0.145, 0.49])
rows = [("P-spline", r07["PS"], r07["PS"], BLUE, True),
        ("cell mean\n+ interpolation", interp07, interp07, "#5A5A5A", False),
        ("4 detectors on\ninterpolated ref.", det_ci.min(), det_ci.max(), "#9A9A9A", False),
        ("cell mean\n(unit fill)", r07["cell"], r07["cell"], "#333333", False),
        ("9 comparators on\nunit-fill ref.", comp07.min(), comp07.max(), "#9A9A9A", False)]
y = np.arange(len(rows))[::-1]
for yy, (lab, lo, hi, col, bold) in zip(y, rows):
    if hi > lo:
        ax.barh(yy, hi, color="#C8C8C8", height=0.62)
    ax.barh(yy, lo, color=col, height=0.62)
    txt = f"{lo:.2f}" if hi == lo else f"{lo:.2f}–{hi:.2f}"
    ax.text(hi + 0.02, yy, txt, va="center", fontsize=12.5, fontweight="bold" if bold else "normal", zorder=3,
            bbox=dict(fc="white", ec="none", pad=0.6))
ax.axvline(r0["PS"], color="#666666", lw=1.2, ls=(0, (3, 2)), zorder=0)
ax.text(r0["PS"] - 0.03, len(rows) - 0.35, f"all ≈ {r0['PS']:.2f}\nwithout loss", ha="right", va="center", fontsize=10.5, color="#444444")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=11); ax.set_xlim(0, 1.25); ax.set_xticks([0, 0.5, 1.0])
ax.set_xlabel("cell AUC-PR", fontsize=12.5); ax.set_ylim(-0.6, len(rows) + 0.15)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

out = ROOT / "paper" / "mathematics" / "figures" / "graphical_abstract.png"
fig.savefig(out, dpi=200, facecolor="white")
print(f"saved {out.name}: {int(11 * 200)} × {int(5.6 * 200)} px | values r=0.7: PS {r07['PS']:.3f}, interp {interp07:.3f}, detectors on interp {det_ci.min():.3f}–{det_ci.max():.3f}, cell {r07['cell']:.3f}, "
      f"comparators {comp07.min():.3f}–{comp07.max():.3f}; no loss PS {r0['PS']:.3f}, cell {r0['cell']:.3f}, comparators {comp0.min():.3f}–{comp0.max():.3f}")
