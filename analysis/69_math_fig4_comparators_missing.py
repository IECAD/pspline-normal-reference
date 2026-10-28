"""Mathematics 논문 Fig 4: 결측 조건에서의 비교 모델 (실험 6 1부 추가 5, D1 테스트), 원寸·공통 스타일(fig_style.py), 본문 폭.
그림 = 절대 수준의 경향(아홉 비교 모델이 셀 평균과 함께 떨어짐). 짝지은 차이와 CI 는 Table 6 (중복 금지 원칙).
출처: results/exp6/comparators_missing/metrics_d1_<split>.csv (기본 test; --split val 로 검증본)
사용법: python analysis/69_math_fig4_comparators_missing.py [--split test]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_style as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = fs.out_dir(ROOT)            # 컬러판 figures/, 흑백판(--bw) figures_bw/
NAMES = {"OLS": "OLS", "RRR": "lagged RRR", "TRANS": "masked Transformer", "RBCPHI": "RBC$^{\\varphi}$", "GDN": "GDN", "OMNI": "OmniAnomaly",
         "LAPRPCA": "Laplacian-RPCA", "CAE2D": "2-D Conv-AE", "TRANAD": "TranAD"}
ORDER = ["base", "C:0.5", "C:0.7", "D:pi=0.7,L=4", "P:P3"]
XL = {"base": "none", "C:0.5": "i.i.d.\n$r$=0.5", "C:0.7": "i.i.d.\n$r$=0.7", "D:pi=0.7,L=4": "burst\n$\\pi$=0.7, $L$=4", "P:P3": "real loss\nP3"}

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="test"); ap.add_argument("--bw", action="store_true"); args = ap.parse_args()
fs.setup()
m = pd.read_csv(ROOT / "results/exp6/comparators_missing" / f"metrics_d1_{args.split}.csv", encoding="utf-8-sig")
m = m[m.scenario.astype(str).str.match(r"^SC\d$")]; m["level"] = m.level.astype(str)
m["key"] = np.where(m.family == "base", "base", m.family + ":" + m.level)
g = m.groupby(["key", "model"]).auc_pr.mean().unstack()
fig, ax = plt.subplots(figsize=(fs.TEXT_W, 6.4 * fs.CM), layout="constrained")
x = np.arange(len(ORDER))
comps = [k for k in NAMES if k in g.columns]
for k, mdl in enumerate(comps):
    fs.line(ax, x, [g.loc[c, mdl] for c in ORDER], "comp", label=fs.STYLE["comp"]["label"] if k == 0 else None, zorder=2)
fs.line(ax, x, [g.loc[c, "cell"] for c in ORDER], "cell", zorder=4)
fs.line(ax, x, [g.loc[c, "PS"] for c in ORDER], "PS", zorder=5)
# 가장 낮은 비교 모델 한 개만 직접 이름표 (범례로는 회색 선들을 구분할 수 없으므로)
low = min(comps, key=lambda c: g.loc["C:0.5", c])
ax.annotate(NAMES[low], xy=(0.97, g.loc["C:0.5", low]), xytext=(0.05, 0.52), fontsize=7, color="#555555", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6, shrinkA=1, shrinkB=2))
ax.set_xticks(x); ax.set_xticklabels([XL[k] for k in ORDER])
ax.set_xlim(-0.3, len(ORDER) - 0.7); ax.set_ylim(0.45, 1.0)
ax.set_ylabel(f"cell AUC-PR (D1, {args.split})"); ax.set_xlabel("training-cell loss")
h, l = ax.get_legend_handles_labels(); order = [l.index(s) for s in ("P-spline (PS)", "cell mean", "nine comparators")]
ax.legend([h[i] for i in order], [l[i] for i in order], loc="center right", bbox_to_anchor=(1.0, 0.52), borderaxespad=0.3)
stem = f"fig4_comparators_missing{'' if args.split == 'test' else '_' + args.split}"
fs.save(fig, OUT, stem)
sys.stdout.reconfigure(encoding="utf-8")
print("lowest comparator at r=0.5:", low); print(g.round(3).to_string())
