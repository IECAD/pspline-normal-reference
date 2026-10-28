"""Mathematics 논문 그림 2·3 (원寸, 공통 스타일 analysis/fig_style.py). 출처 CSV 를 읽어 PDF·PNG 로 저장한다.
Fig 2: 빈 칸 비율 예측 vs 실측, σ_R 증가율 (analysis/out/47_reference_error.csv) — 본문 폭
Fig 3: 결측 조건별 AUC-PR: cell / cell+fill / cell+interp / PS, D1·D2 (results/exp6/{conditions,missing_patterns,phy_masks}/table_test*.csv) — 전폭
(퇴역) D6 설계 변형 막대그래프: Table 10 과 같은 값의 중복이라 draft_v8 에서 뺐다. `--fig5-retired` 로만 다시 그린다.
사용법: python analysis/60_math_figures.py [--fig5-retired]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_style as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = fs.out_dir(ROOT)            # 컬러판 figures/, 흑백판(--bw) figures_bw/
fs.setup()


# ----------------------------------------------------------------------------- Fig 2
def fig2():
    d = pd.read_csv(ROOT / "analysis" / "out" / "47_reference_error.csv")
    c = d[d.family == "C"].copy(); c["r"] = c.level.astype(float)
    fig, ax = plt.subplots(1, 2, figsize=(fs.TEXT_W, 5.6 * fs.CM), layout="constrained")
    # (a) 예측 vs 실측 빈 칸 비율: D1 방 3개는 마커로 구분, D2 환경 4개는 한 계열
    cm = c[c.basis == "cell"]
    marks = {"classroom": ("o", "D1 classroom"), "lab": ("s", "D1 laboratory"), "hallway": ("^", "D1 hallway")}
    for sp, (mk, lab) in marks.items():
        g = cm[cm.space == sp].groupby("r")[["empty_pred", "empty_frac"]].mean()
        ax[0].plot(g.empty_pred, g.empty_frac, mk, ms=5, mfc="white", mec=fs.GROUP_MARK_COLOR, mew=1.1, label=lab, ls="none")   # (a) 는 셀 평균 표의 빈 칸 → 셀 평균 색
    g = cm[cm.dataset == "d2"].groupby(["space", "r"])[["empty_pred", "empty_frac"]].mean()
    ax[0].plot(g.empty_pred, g.empty_frac, "x", ms=5, color=fs.D2_MARK_COLOR, mew=1.1, ls="none", label="D2 (4 environments)")
    lim = 0.16
    ax[0].plot([0, lim], [0, lim], "--", color=fs.OI["grey"], lw=0.8, zorder=0)
    ax[0].set_xlim(-0.005, lim); ax[0].set_ylim(-0.005, lim)
    ax[0].set_xticks([0, 0.05, 0.10, 0.15]); ax[0].set_yticks([0, 0.05, 0.10, 0.15])
    ax[0].set_xlabel("predicted empty-cell fraction"); ax[0].set_ylabel("observed empty-cell fraction")
    ax[0].legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    fs.panel(ax[0], "a")
    # (b) σ_R 증가율 (로그 축), 방·환경마다 한 선
    for b in ("cell", "PS"):
        st = fs.STYLE[b]
        g = c[c.basis == b].groupby(["space", "r"]).sigma_ratio.mean().reset_index()
        for k, (sp, gg) in enumerate(g.groupby("space")):
            ax[1].plot(gg.r, gg.sigma_ratio, marker=st["marker"], ls=st["ls"], lw=1.1, ms=3.8, color=st["color"], mfc=st.get("mfc", st["color"]),
                       alpha=0.9, label=st["label"] if k == 0 else None)
    ax[1].set_yscale("log"); ax[1].set_yticks([1, 2, 5, 10]); ax[1].set_yticklabels(["1", "2", "5", "10"])
    ax[1].set_xticks([0.3, 0.5, 0.7]); ax[1].set_xlabel("loss rate $r$")
    ax[1].set_ylabel(r"$\sigma_R$ / $\sigma_R$ without loss")
    ax[1].legend(loc="upper left", borderaxespad=0.2)
    fs.panel(ax[1], "b")
    fs.save(fig, OUT, "fig2_empty_cells"); plt.close(fig)


# ----------------------------------------------------------------------------- Fig 3
def _key(fam, lev):
    if fam == "base":
        return "base"
    return f"{fam}:{str(lev).replace('r=', '')}"


def _table(path):
    t = pd.read_csv(path); t["key"] = [_key(f, l) for f, l in zip(t.family, t.level)]
    return t.set_index("key")


def fig3():
    base = ROOT / "results" / "exp6"
    cond, pat, phy = (_table(base / s / "table_test.csv") for s in ("conditions", "missing_patterns", "phy_masks"))
    imp = {w: _table(base / "imputation" / f"table_test_vs_{w}.csv") for w in ("cellfill", "cellinterp")}
    phyimp = {w: _table(base / "phy_masks" / f"table_test_vs_{w}.csv") for w in ("cellfill", "cellinterp")}
    panels = [
        ("a", "loss rate $r$", cond, imp, [("base", "0"), ("C:0.3", "0.3"), ("C:0.5", "0.5"), ("C:0.7", "0.7")]),
        ("b", "bursty loss  |  anchor outage", pat, imp,
         [("D:pi=0.3,L=4", "$\\pi$=0.3\n$L$=4"), ("D:pi=0.5,L=4", "$\\pi$=0.5\n$L$=4"), ("D:pi=0.5,L=8", "$\\pi$=0.5\n$L$=8"),
          ("D:pi=0.7,L=4", "$\\pi$=0.7\n$L$=4"), ("E:q=0.3", "$q$=0.3"), ("E:q=0.5", "$q$=0.5")]),
        ("c", "real packet-loss mask", phy, phyimp, [("P:P1", "P1"), ("P:P2", "P2"), ("P:P3", "P3")]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(fs.FULL_W, 10.8 * fs.CM), sharey="row", layout="constrained",
                             gridspec_kw={"width_ratios": [4, 6.4, 3]})
    for j, (letter, xlab, t, imps, keys) in enumerate(panels):
        ks = [k for k, _ in keys]; labels = [l for _, l in keys]
        for i, ds in enumerate(("d1", "d2")):
            ax = axes[i, j]; x = np.arange(len(ks))
            fs.line(ax, x, t.loc[ks, f"{ds}_cell"].values, "cell")
            for which in ("cellfill", "cellinterp"):
                r = imps[which][f"{ds}_ref"].reindex(ks).values
                if np.isfinite(r).any():
                    fs.line(ax, x, r, which)
            fs.line(ax, x, t.loc[ks, f"{ds}_PS"].values, "PS")
            ax.set_xticks(x); ax.set_xticklabels(labels)
            ax.set_xlim(-0.4, len(ks) - 0.6)
            if letter == "b":
                ax.axvline(3.5, color=fs.OI["grey"], lw=0.6, ls=(0, (1, 2)))
            if i == 1:
                ax.set_xlabel(xlab)
            if i == 0:
                fs.panel(ax, letter)
            if j == 0:
                ax.set_ylabel(f"{'D1' if ds == 'd1' else 'D2 (multi-metric)'} AUC-PR (test)")
    axes[0, 0].set_ylim(0.48, 0.97); axes[1, 0].set_ylim(0.862, 0.908)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4, handlelength=2.8, columnspacing=1.6)
    fs.save(fig, OUT, "fig3_missing_robustness"); plt.close(fig)


# ----------------------------------------------------------------------------- (퇴역) D6 설계 변형
def fig5_retired():
    d = pd.read_csv(ROOT / "results" / "exp10" / "design" / "compare_test.csv")
    models = ["cell", "Z1", "PS", "PS1", "PSH", "PS1H"]
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_W, 5.5 * fs.CM), layout="constrained")
    for ax, ev, letter in zip(axes, ("SD", "이상값 제외"), "ab"):
        for sc, mk, mf in (("주", "o", None), ("×3", "o", "white")):
            r = d[(d["eval"] == ev) & (d.scen == sc)].iloc[0]
            ax.plot(range(len(models)), [r[m] for m in models], mk, ms=5, ls="none", color=fs.OI["black"], mfc=mf or fs.OI["black"],
                    label="main injection" if sc == "주" else "×3 injection")
        ax.set_xticks(range(len(models))); ax.set_xticklabels(["cell", "node", "PS", "PS1", "PSH", "PS1H"])
        ax.set_ylim(0, 1.0 if ev != "SD" else 0.25); fs.panel(ax, letter)
    axes[0].set_ylabel("cell AUC-PR (D6, test)"); axes[0].legend(loc="upper left")
    fs.save(fig, OUT, "fig5_d6_design_retired"); plt.close(fig)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    fig2(); fig3()   # 흑백판: python analysis/60_math_figures.py --bw
    if "--fig5-retired" in sys.argv:
        fig5_retired()
