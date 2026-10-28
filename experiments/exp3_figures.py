"""실험 3 그림 (doc/합성방식_실험계획_최종.md 8.2절).

사용법: python experiments/exp3_figures.py --split test   (선택 확정 후 테스트를 한 번만 보고)
출력: results/exp3/fig_*.png
  fig_axis_delta_<ds>_<split>.png    축별 B0 대비 ΔAUC-PR, ΔHitRate (오차막대: 주입 시드 간 표준편차, O3 는 순서 10회 표준편차)
  fig_order_sensitivity_<split>.png  O3 10회 분포(상자) + O1, O2, O4 위치, T2 와 T1 나란히 (D1, D2 다지표)
  fig_smearing_heatmap.png           D2 SC2m 테스트 윈도우 예시: T2(B0) vs T1(E12) 앵커 점수 히트맵
  fig_anchor_count.png               D2 앵커 수 4/6/8 에서 seriation 이득, 센서 평활 이득 (거리만 주입)
"""
import argparse
import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from exp3_summarize import load_all, per_run  # noqa: E402

OUT = ROOT / "results" / "exp3"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
AX_COLOR = {"①": "#2a78d6", "②": "#eb6834", "③": "#1baf7a", "④": "#eda100", "③×④": "#e87ba4", "②×④": "#4a3aa7"}
plt.rcParams.update({
    "font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
AXIS_OF_ID = {**{f"E{i:02d}": "①" for i in range(1, 5)}, **{f"E{i:02d}": "②" for i in range(5, 9)},
              **{f"E{i:02d}": "③" for i in range(9, 12)}, **{f"E{i:02d}": "④" for i in range(12, 15)},
              **{f"E{i:02d}": "③×④" for i in range(15, 18)}, "E18": "②×④"}


def run_level(M, ds, split, scen):
    """실행(run) 단위 평균, 그리고 B0 대비 차이의 오차 (시드 또는 순서 반복 간 표준편차)."""
    d = per_run(M, ds, split, scen)
    by = d.groupby(["id", "run", "seed"])[["auc_pr", "hitrate"]].mean().reset_index()
    rows = []
    for id_, g in by.groupby("id"):
        per_run_mean = g.groupby("run")[["auc_pr", "hitrate"]].mean()
        per_seed = g.groupby("seed")[["auc_pr", "hitrate"]].mean()
        spread = per_run_mean if len(per_run_mean) > 1 else per_seed
        rows.append(dict(id=id_, auc=per_run_mean.auc_pr.mean(), hit=per_run_mean.hitrate.mean(),
                         auc_sd=spread.auc_pr.std(), hit_sd=spread.hitrate.std()))
    return pd.DataFrame(rows).set_index("id")


def fig_axis_delta(M, ds, split, scen, label):
    R = run_level(M, ds, split, scen)
    if "B0" not in R.index:
        return
    ids = [i for i in R.index if i.startswith("E")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, col, title in [(axes[0], "auc", "ΔAUC-PR (B0 대비)"), (axes[1], "hit", "ΔHitRate@100% (B0 대비)")]:
        y = np.arange(len(ids))
        d = R.loc[ids, col] - R.loc["B0", col]
        ax.barh(y, d, xerr=R.loc[ids, f"{col}_sd"].fillna(0), color=[AX_COLOR[AXIS_OF_ID[i]] for i in ids],
                height=0.6, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_yticks(y, [f"{i} ({AXIS_OF_ID[i]})" for i in ids], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=10)
    fig.suptitle(f"실험 3 축별 효과 — {label} (B0 AUC-PR {R.loc['B0', 'auc']:.3f}, HitRate {R.loc['B0', 'hit']:.3f})",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"fig_axis_delta_{ds}_{split}{'_' + scen.strip('^$').replace(chr(92) + 'd', '') if scen else ''}.png",
                dpi=130)
    plt.close(fig)


def fig_order(M, split):
    panels = [("d1", None, "D1 (거리만 주입)"), ("d2", r"^SC\dm$", "D2 (다지표 주입)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (ds, scen, label) in zip(axes, panels):
        d = per_run(M, ds, split, scen)
        r = d.groupby(["id", "run"])["auc_pr"].mean()
        for xi, (tlabel, o3, fixed) in enumerate([("T2 (블록)", "E10", {"O1": "E09", "O2": "B0", "O4": "E11"}),
                                                   ("T1 (통합)", "E17", {"O1": "E15", "O2": "E16"})]):
            if o3 in r.index.get_level_values(0):
                vals = r.loc[o3].to_numpy()
                bp = ax.boxplot([vals], positions=[xi], widths=0.4, patch_artist=True,
                                medianprops=dict(color=INK, lw=1.2), flierprops=dict(marker=".", ms=3, mec=MUTED))
                bp["boxes"][0].set(facecolor="#cde2fb", edgecolor="#2a78d6")
            for k, (o, id_) in enumerate(fixed.items()):
                if id_ in r.index.get_level_values(0):
                    v = r.loc[id_].mean()
                    mk = {"O1": "s", "O2": "D", "O4": "^"}[o]
                    ax.plot(xi + 0.32, v, mk, ms=7, color=["#eb6834", "#1baf7a", "#4a3aa7"][k], mec=SURF, mew=1,
                            label=o if xi == 0 else None)
        ax.set_xticks([0, 1], ["T2 (블록)", "T1 (통합)"])
        ax.set_xlim(-0.6, 1.8)
        lo, hi = ax.get_ylim()
        ax.set_title(f"{label} — y축 범위 {hi - lo:.4f}", loc="left", fontsize=10)
        ax.set_ylabel(f"{split} AUC-PR")
        ax.legend(fontsize=8, loc="lower right", title="고정 순서", title_fontsize=8)
    fig.suptitle("순서 민감도: 무작위 순서(O3) 10회 분포(상자)와 O1·O2·O4", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / f"fig_order_sensitivity_{split}.png", dpi=130)
    plt.close(fig)


def fig_anchor_count():
    p = OUT / "h6_anchor_count.csv"
    if not p.exists():
        return
    R = pd.read_csv(p)
    R = R[R.kind == "inject"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, sp in zip(axes, ("val", "test")):
        g = R[R.split == sp].sort_values("k")
        for col, c, lab in [("seri_gain_vs_O3", "#2a78d6", "seriation 이득 (B0 − O3 평균)"),
                            ("seri_gain_vs_O1", "#1baf7a", "seriation 이득 (B0 − O1)"),
                            ("smooth_gain_vs_T4", "#eb6834", "센서 평활 이득 (B0 − T4)")]:
            ax.plot(g.k, g[col], "o-", color=c, lw=2, ms=7, mec=SURF, label=lab)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xticks([4, 6, 8])
        ax.set_xlabel("앵커 수 (D2 부분추출, 부분집합 5개 평균)")
        ax.set_title(sp, loc="left", fontsize=10)
    axes[0].set_ylabel("ΔAUC-PR")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("H6 앵커 수 효과 (D2, 거리만 주입)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig_anchor_count.png", dpi=130)
    plt.close(fig)


def fig_smearing():
    """D2 env0 SC2m 테스트: B0(T2) 와 E12(T1) 의 앵커 점수 (2개 앵커 동시 주입 윈도우 하나)."""
    import exp3_composition as E3
    from src.data.config import load_config
    from src.data.inject import inject_windows
    from src.data.load import load_windows
    from src.model.bank import anchor_scores
    from src.model.compose import Composer
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    env = 1                                                    # NLOS 앵커가 가장 적은 환경 (NLOS 셀 26%)
    tr, va, te = (load_windows("d2", f"env{env}_ch5_path", dcfg, split=s) for s in ("train", "val", "test"))
    rows = tr["rows"]
    E3.ROWS[0] = rows
    node = lambda w: np.array([f"env{env}/{int(t)}" for t in w["meta"]["tidx0"]])
    anchors = [f"A{a}" for a in range(1, 9)]
    rix = [rows.index(f"RANGE_{a}") for a in anchors]
    nl = te["cells"]["lab_D"][:, rix]
    pool = E3.SIG[E3.SIG.env == env][["d_range", "d_rss", "d_rssfp"]].to_numpy()
    Xi, lab, _, ev = inject_windows(te["X"], rows, ecfg["datasets"]["d2"]["scenarios"]["SC2"], 42, "RANGE",
                                    (nl == 0).all(2), pool, ["RANGE", "RSS", "RSS_FP"])
    if len(ev):                                                # 주입 윈도우 중 NLOS 앵커가 가장 적은 것
        cand = ev.window.unique()
        n = int(cand[np.argmin([(nl[c] == 1).any(1).sum() for c in cand])])
    else:
        n = 0
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw=dict(width_ratios=[1, 1, 0.05]))
    vmax = None
    imgs = []
    nlos_cells = nl[n] == 1
    for ax, id_ in zip(axes[:2], ("B0", "E12")):
        code = ecfg["ids"][id_]
        comp = Composer("d2", code["V"], code["S"], code["O"], code["T"]).fit(tr["X"], tr["cells"]["mask"], rows, node(tr))
        bank = E3.fit_bank(comp, ecfg, tr["X"], tr["cells"]["mask"], node(tr), tr["X"].shape[2])
        Xv, Wv = comp.transform(va["X"], va["cells"]["mask"], rows)
        bank.calibrate(Xv, Wv, node(va))
        Xs, _ = comp.transform(Xi[[n]], te["cells"]["mask"][[n]], rows)
        A = anchor_scores(bank.score(Xs, node(te)[[n]]), comp.anchor_of_row, anchors)[0]
        Am = np.ma.masked_where(nlos_cells, A)                 # NLOS 셀(평가 제외)은 회색
        vmax = vmax or float(np.nanpercentile(A[~nlos_cells], 99))
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#d9d8d3")
        im = ax.imshow(Am, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        imgs.append(im)
        for j, t in zip(*np.nonzero(lab[n] == 1)):
            ax.add_patch(plt.Rectangle((t - 0.5, j - 0.5), 1, 1, fill=False, ec="#eb6834", lw=1.4))
        for j, t in zip(*np.nonzero((A > bank.thr) & ~nlos_cells)):
            ax.plot(t, j, ".", color=INK, ms=4)
        ax.set_yticks(range(8), anchors)
        ax.set_xlabel("시간 스텝 (경로 위치)")
        ax.set_title(f"{id_} ({code['T']}) — 테두리: 주입 셀, 점: 임계 초과", loc="left", fontsize=9)
        ax.grid(False)
    fig.colorbar(imgs[0], cax=axes[2], label="앵커 점수 A_R")
    fig.suptitle(f"번짐 비교: D2 env{env} SC2m (앵커 2개 동시 주입) 테스트 윈도우 — 회색: 기하 NLOS 셀 (평가 제외)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig_smearing_heatmap.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    M = load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_axis_delta(M, "d1", args.split, None, f"D1 {args.split} (주입 SC1-3, 거리만)")
    fig_axis_delta(M, "d2", args.split, r"^SC\dm$", f"D2 {args.split} (다지표 주입 SC1m-3m)")
    fig_order(M, args.split)
    fig_anchor_count()
    fig_smearing()
    print(sorted(p.name for p in OUT.glob("fig_*.png")))
