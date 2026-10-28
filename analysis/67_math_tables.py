"""Mathematics 원고 표 자동 생성 (template addendum §B-1·D-3: 수기 전사 제거).
1차 산출물(results/*.csv, paper/mathematics/paired_stats.csv)에서 본문 표 4·5·7·8·9·A1 과 보충 S4 전체 표를 markdown 으로 만들고,
draft 의 같은 라벨 행과 숫자를 대조해 불일치를 출력한다.
출력: paper/mathematics/tables_auto.md (본문 표), paper/mathematics/supplementary/S4_full_tables.md (전체 표)
사용법: python analysis/67_math_tables.py [--draft paper/mathematics/draft/draft_v3.md]
"""
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
PM = ROOT / "paper/mathematics"
PS_ = pd.read_csv(PM / "paired_stats.csv"); PS_["condition"] = PS_.condition.fillna("")
OUT_MAIN, OUT_S4 = [], []


def rh(x, nd=3, sign=False):
    """보고서(요약 스크립트)와 같은 half-up 반올림 문자열."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    q = Decimal(str(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    mag = f"{abs(q):.{nd}f}"
    neg = float(x) < 0                      # 반올림해 0 이 되어도 부호는 원값을 따른다 (−0.000; '+−0.000' 버그 수정 2026-10-13)
    if sign:
        return ("−" if neg else "+") + mag
    return ("−" + mag) if neg and float(q) != 0 else (("−" + mag) if neg else mag)


def f3(x, sign=False):
    return rh(x, 3, sign)


def ci(row):
    return f"[{rh(row.ci_lo, 3, True)}, {rh(row.ci_hi, 3, True)}]"


def ps_row(group, contrast, dataset, condition=""):
    s = PS_[(PS_.group == group) & (PS_.contrast == contrast) & (PS_.dataset == dataset) & (PS_.condition == condition)]
    return s.iloc[0] if len(s) else None


def d2multi(df):
    return df[df.scenario.astype(str).str.endswith("m")] if (str(df.dataset.iloc[0]) == "d2") else df


def md(title, header, rows, note=None, target=OUT_MAIN):
    L = [f"**{title}**", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    L += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if note:
        L.append(""); L.append(note)
    target.extend(L + [""])


# ----------------------------------------------------------------------------- Table 4 (comparators)
def comparator_abs():
    """모델별 테스트 절대 AUC-PR (선택 구성·D2 다지표)."""
    out = {}
    for ds in ("d1", "d2"):
        for sub, selfile in (("", "selection.json"), ("batch2/", "batch2/selection.json"), ("batch2_fix/", None), ("batch2_fix2/", None), ("batch3/", None), ("batch3_fix/", None)):
            if not (R / f"exp1_baselines/{sub}metrics_test_{ds}.csv").exists():
                continue
            m = d2multi(pd.read_csv(R / f"exp1_baselines/{sub}metrics_test_{ds}.csv"))
            sel = json.load(open(R / f"exp1_baselines/{selfile}", encoding="utf-8"))["selection"] if selfile else {}
            for mdl in m.model.unique():
                t = m[m.model == mdl]; s = sel.get(mdl, {}).get(ds)
                if s is not None:
                    t = t[(t.variant == s["variant"]) & (t.cfg.apply(lambda x: json.loads(x) == s["cfg"]))]
                if len(t):
                    out[(mdl, ds)] = t.auc_pr.mean()
        ps = d2multi(pd.read_csv(R / f"C03/base/metrics_{ds}_test.csv")); ps = ps[ps.scenario.astype(str).str.match(r"SC\d")]; out[("PS", ds)] = ps.auc_pr.mean()
        st = d2multi(pd.read_csv(R / f"exp1_stssd/metrics_test_{ds}.csv")) if (R / f"exp1_stssd/metrics_test_{ds}.csv").exists() else None
        if st is not None and "variant" in st.columns:
            st = st[st.scenario.astype(str).str.match(r"SC\d")]
            for v in st.variant.unique():
                out[(f"STSSD_{v}", ds)] = st[st.variant == v].auc_pr.mean()
        pr = pd.read_csv(R / "exp1_stssd/paired.csv")
        for v in pr.variant.unique():
            r = pr[(pr.dataset == ds) & (pr.variant == v)].iloc[0]; out[(f"STSSD_{v}_delta", ds)] = (r.delta, r.frac_pos)
        simple = pd.read_csv(R / f"exp1_simple/table_{'d1_inject' if ds == 'd1' else 'd2_inject_multi'}_test.csv").set_index("id")
        out[("Z1", ds)] = simple.loc["Z1", "auc_pr"]; out[("Z2", ds)] = simple.loc["Z2", "auc_pr"]
    return out


def table4():
    A = comparator_abs()
    pt = pd.read_csv(R / "exp1_simple/paired_test.csv")
    def simple_delta(view, prefix):
        r = pt[(pt.view == view) & (pt.contrast.str.startswith(prefix))]
        return (r["mean"].iloc[0], r["frac_pos"].iloc[0]) if len(r) else (np.nan, np.nan)
    spec = [("**PS (proposed, no tuning)**", "PS", None), ("ST-SSD-type node background (static, $\\gamma=0$)‡", "STSSD_N", None),
            ("Cell mean", "Z2", "cell"), ("Node mean", "Z1", "node"),
            ("OLS cross-sensor regression (nc)", "OLS", "PS − OLS"), ("Lagged RRR (nc, rank 1)", "RRR", "PS − RRR"),
            ("Masked Transformer (nc, $d_{model}$ 32)", "TRANS", "PS − TRANS"), ("RBC$^{\\varphi}$ (nc, var 0.8)", "RBCPHI", "PS − RBCPHI"),
            ("GDN re-implementation (nc, leak fixed)", "GDN", "PS − GDN"), ("OmniAnomaly re-implementation (nc)", "OMNI", "PS − OMNI"),
            ("Laplacian-RPCA, sensor-graph variant (nc)", "LAPRPCA", "PS − LAPRPCA"), ("2-D Conv-AE (nc)", "CAE2D", "PS − CAE2D"),
            ("1-D CAE (nc)", "CAE1D", "PS − CAE1D"), ("LSTM-AE (nc)", "LSTMAE", "PS − LSTMAE"),
            ("USAD, original (Sigmoid output, alternating optimisers)", "USADFIX2", "PS − USADFIX2"), ("Isolation forest (nc)", "IF", "PS − IF"),
            ("RPCA subspace (nc)", "RPCA", "PS − RPCA"), ("PCA-SPE (nc, var 0.8)", "PCA", "PS − PCA"),
            ("TranAD re-implementation (nc)§", "TRANAD", "PS − TRANAD"), ("Anomaly Transformer re-implementation (nc)§", "ANOMTRANSFIX", "PS − ANOMTRANSFIX"),
            ("ST-SSD-type, one global background", "STSSD_G1", "global")]
    rows = []
    for label, key, con in spec:
        cells = [label]
        for ds in ("d1", "d2"):
            tol = 0.0165 if ds == "d1" else 0.0171
            a = A.get((key, ds), np.nan); cells.append(f"**{a:.3f}**" if key == "PS" else f3(a))
            if con is None:
                cells.append("—" if key == "PS" else f"{rh(A[('STSSD_N_delta', ds)][0], 3, True)} tie")
            elif con == "cell":
                r = ps_row("exp6/conditions", "PS − cell", ds, "base 기준")
                cells.append(f"{r['mean']:+.3f} {ci(r)} ({r.frac_pos:.2f}) tie")
            elif con == "node":
                m_, fp = simple_delta("d1_inject", "시간 방향 P-spline (F01 − Z1)") if ds == "d1" else simple_delta("d2_inject_multi", "최종 M01 − Z1")
                cells.append(f"{m_:+.3f} ({fp:.2f}) tie" if abs(m_) <= tol else f"**{m_:+.3f}** ({fp:.2f})")
            elif con == "global":
                d, fp = A[("STSSD_G1_delta", ds)]; cells.append(f"**{rh(d, 3, True)}** ({fp:.2f})")
            else:
                r = ps_row("exp1_baselines", con, ds, {"OLS": "batch1", "RRR": "batch2", "TRANS": "batch2", "RBCPHI": "batch2_fix", "GDN": "batch2_fix", "OMNI": "batch2",
                                                        "LAPRPCA": "batch2_fix", "CAE2D": "batch2", "CAE1D": "batch1", "LSTMAE": "batch1", "USADFIX2": "batch2_fix2",
                                                        "IF": "batch1", "RPCA": "batch1", "PCA": "batch1", "TRANAD": "batch3", "ANOMTRANSFIX": "batch3_fix"}[key])
                if r is None:
                    cells.append("—"); continue
                v = f"{r['mean']:+.3f} {ci(r)} ({r.frac_pos:.2f})"
                cells.append(f"**{r['mean']:+.3f}** {ci(r)} ({r.frac_pos:.2f})" if r["mean"] > tol else v + " tie")
        rows.append(cells)
    md("Table 4 (auto). Comparators on the test split: cell AUC-PR, paired PS − model [95 % CI] (fraction positive); bold = beyond tolerance.",
       ["Model", "D1", "PS − model [95 % CI] (pos.)", "D2", "PS − model [95 % CI] (pos.)"], rows)


# ----------------------------------------------------------------------------- Table 5 (missing, D1)
def tab(path):
    t = pd.read_csv(path); t["key"] = ["base" if f == "base" else f"{f}:{str(l).replace('r=', '')}" for f, l in zip(t.family, t.level)]
    return t.set_index("key")


def table5():
    c = tab(R / "exp6/conditions/table_test.csv"); m = tab(R / "exp6/missing_patterns/table_test.csv"); p = tab(R / "exp6/phy_masks/table_test.csv")
    ifill, iint = tab(R / "exp6/imputation/table_test_vs_cellfill.csv"), tab(R / "exp6/imputation/table_test_vs_cellinterp.csv")
    pfill, pint = tab(R / "exp6/phy_masks/table_test_vs_cellfill.csv"), tab(R / "exp6/phy_masks/table_test_vs_cellinterp.csv")
    tol = 0.0165
    def b(x):
        return f"**{x:+.3f}**" if x > tol else rh(x, 3, True)
    spec = [("none (45 pairs)", c, "base", "exp6/conditions", "base 기준", None, None),
            ("i.i.d. $r=0.3$", c, "C:0.3", "exp6/conditions", "C 0.3", ifill, iint), ("i.i.d. $r=0.5$", c, "C:0.5", "exp6/conditions", "C 0.5", ifill, iint),
            ("i.i.d. $r=0.7$", c, "C:0.7", "exp6/conditions", "C 0.7", ifill, iint),
            ("burst $\\pi=0.3,L=4$", m, "D:pi=0.3,L=4", "exp6/missing_patterns", "D pi=0.3,L=4", ifill, iint),
            ("burst $\\pi=0.5,L=4$", m, "D:pi=0.5,L=4", "exp6/missing_patterns", "D pi=0.5,L=4", ifill, iint),
            ("burst $\\pi=0.5,L=8$", m, "D:pi=0.5,L=8", "exp6/missing_patterns", "D pi=0.5,L=8", ifill, iint),
            ("burst $\\pi=0.7,L=4$", m, "D:pi=0.7,L=4", "exp6/missing_patterns", "D pi=0.7,L=4", ifill, iint),
            ("real loss P1", p, "P:P1", "exp6/phy_masks", "P P1", pfill, pint), ("real loss P2", p, "P:P2", "exp6/phy_masks", "P P2", pfill, pint),
            ("real loss P3", p, "P:P3", "exp6/phy_masks", "P P3", pfill, pint),
            ("anchor outage $q=0.3$", m, "E:q=0.3", "exp6/missing_patterns", "E q=0.3", None, None), ("anchor outage $q=0.5$", m, "E:q=0.5", "exp6/missing_patterns", "E q=0.5", None, None)]
    # draft_v8: 절대 수준은 Figure 3 이 보여 주므로(중복 금지) 표는 짝지은 차이만 — D1 세 기준선 + D2 셀 평균
    def b2(x):
        return f"**{x:+.3f}**" if x > 0.0171 else rh(x, 3, True)
    rows = []
    for label, t, key, grp, cond, fi, it in spec:
        r = ps_row(grp, "PS − cell", "d1", cond); r2 = ps_row(grp, "PS − cell", "d2", cond)
        fill = b(fi.loc[key, "d1_delta"]) if fi is not None else "—"; interp = b(it.loc[key, "d1_delta"]) if it is not None else "—"
        d2 = f"{b2(r2['mean'])} {ci(r2)}" if r2 is not None else "—"
        rows.append([label.replace(" (45 pairs)", ""), f"{b(r['mean'])} {ci(r)}", fill, interp, d2])
    md("Table 5 (auto). Paired PS − baseline test AUC-PR under training-cell loss (bold = beyond tolerance 0.0165 D1 / 0.0171 D2).",
       ["Condition", "D1: PS − cell [95 % CI]", "D1: PS − cell+fill", "D1: PS − cell+interp", "D2: PS − cell [95 % CI]"], rows)


# ----------------------------------------------------------------------------- Table 7 (ST-SSD variants)
def table7():
    rows = [["**PS**"]]
    absd = {}
    for ds in ("d1", "d2"):
        m = d2multi(pd.read_csv(R / f"exp1_stssd/original/metrics_test_{ds}.csv"))
        for v, bg in m[["variant", "background"]].drop_duplicates().itertuples(index=False):
            t = m[(m.variant == v) & (m.background == bg)]; absd[(v, bg, ds)] = (t.auc_pr.mean(), t.win_auc_pr.mean())
    rows[0] += [f"{absd[('PS','N','d1')][0]:.3f} / {absd[('PS','N','d1')][1]:.3f}", f"{absd[('PS','N','d2')][0]:.3f} / {absd[('PS','N','d2')][1]:.3f}", "—", "—"]
    for label, v, bg in (("LRT + node background (R)", "R", "N"), ("dynamic mean, row-independent (D′)", "DT", "G1T4"),
                         ("LRT + global background (original setting)", "R", "G1"), ("dynamic mean + global smoothing (D)", "D", "G1")):
        cells = [label]
        for ds in ("d1", "d2"):
            a = absd[(v, bg, ds)]; cells.append(f"{a[0]:.3f} / {a[1]:.3f}")
        c1 = [ps_row("exp1_stssd/original", f"PS − {v}({bg}) cell", ds) for ds in ("d1", "d2")]
        w1 = [ps_row("exp1_stssd/original", f"{v}({bg}) − PS window", ds) for ds in ("d1", "d2")]
        cells.append(" / ".join(f"{r['mean']:+.3f} {ci(r)}" for r in c1)); cells.append(" / ".join(f"{r['mean']:+.3f} {ci(r)}" for r in w1))
        rows.append(cells)
    md("Table 8 (auto). ST-SSD variants: cell / window AUC-PR and paired differences [95 % CI].",
       ["Variant", "D1 cell / window", "D2 cell / window", "PS − variant, cell (D1 / D2)", "variant − PS, window (D1 / D2)"], rows)


# ----------------------------------------------------------------------------- Table 8/9 (D6)
def d6_pairs(split):
    d = pd.read_csv(R / f"exp10/metrics_{split}.csv"); d = d[~d.scenario.astype(str).str.contains("x")]
    full = d[d.cond == "full"]
    piv = full.pivot_table(index=["scan", "scenario", "seed"], columns="model", values="auc_pr")
    wpiv = full.pivot_table(index=["scan", "scenario", "seed"], columns="model", values="win_auc_pr")
    f01 = d[d.cond == "f0.1"].pivot_table(index=["scan", "scenario", "seed", "cond_seed"], columns="model", values="auc_pr")
    anc = d[(d.cond == "anchors") & (d.model == "PS")].groupby("level").auc_pr.mean()
    return piv, wpiv, f01, anc


def table8():
    pv, wv, f01v, ancv = d6_pairs("val"); pt, wt, f01t, anct = d6_pairs("test")
    def dl(p, a, b): return (p[a] - p[b]).mean()
    def pos(p, a, b): return int(((p[a] - p[b]) > 0).sum())
    rows = [["T1 window LRT (R) − PS, window AUC-PR (expected R ≥ PS)", rh(dl(wv,'R','PS'), 3, True), f"{dl(wt,'R','PS'):+.3f} {ci(ps_row('exp10','R − PS window','d6','main, full'))}"],
            ["T2 PS − cell (expected tie)", f"{dl(pv,'PS','cell'):+.3f}, {pos(pv,'PS','cell')}/180 positive", f"{dl(pt,'PS','cell'):+.3f} {ci(ps_row('exp10','PS − cell','d6','main, full'))}, {pos(pt,'PS','cell')}/180 positive"],
            ["T2 PS − cell+fill", rh(dl(pv,'PS','cellfill'), 3, True), f"{dl(pt,'PS','cellfill'):+.3f} {ci(ps_row('exp10','PS − cellfill','d6','main, full'))}"],
            ["T2 PS − node mean (expected tie, static)", rh(dl(pv,'PS','Z1'), 3, True), f"{dl(pt,'PS','Z1'):+.3f} {ci(ps_row('exp10','PS − Z1','d6','main, full'))} ({pos(pt,'PS','Z1')}/180 positive)"],
            ["T3 training 1/10 (2 windows/node), PS − cell", rh((f01v['PS']-f01v['cell']).mean(), 3, True), f"{(f01t['PS']-f01t['cell']).mean():+.3f} {ci(ps_row('exp10','PS − cell','d6','main, f=0.1'))}"],
            ["T4 anchors 4/8/16/all, PS AUC-PR", "/".join(rh(ancv.loc[k], 3) for k in (4.0, 8.0, 16.0)) + f"/{pv['PS'].mean():.3f}", "/".join(rh(anct.loc[k], 3) for k in (4.0, 8.0, 16.0)) + f"/{pt['PS'].mean():.3f}"],
            ["T5 PS − OLS / − PCA", f"{dl(pv,'PS','OLS'):+.3f} / {dl(pv,'PS','PCA'):+.3f}", f"{dl(pt,'PS','OLS'):+.3f} {ci(ps_row('exp10','PS − OLS','d6','main, full'))} / {dl(pt,'PS','PCA'):+.3f} {ci(ps_row('exp10','PS − PCA','d6','main, full'))}"]]
    md("Table 9 (auto). D6 pre-registered tests (180 pairs; tolerance 0.0171).", ["Test", "Validation", "Test (once)"], rows)
    dg = pd.read_csv(R / "exp10/design/compare_test.csv").set_index(["eval", "scen"])
    rows = []
    # draft_v8: 퇴역한 그림(D6 설계 변형)의 'SD, ×3' 값을 표로 옮기고, 수기였던 '학습 1/10' 행도 생성한다
    for ev, sc, label in (("SD", "주", "SD standardisation, main"), ("SD", "×3", "SD standardisation, ×3"),
                          ("이상값 제외", "주", "excluded, main"), ("이상값 제외", "×3", "excluded, ×3")):
        r = dg.loc[(ev, sc)]; vals = [r[k] for k in ("cell", "Z1", "PS", "PS1", "PSH", "PS1H")]; best = rh(max(vals), 3)
        rows.append([label] + [f"**{rh(v, 3)}**" if rh(v, 3) == best else rh(v, 3) for v in vals])   # 반올림 후 같은 값은 모두 굵게
    r = dg.loc[("이상값 제외", "주")]
    rows.append(["excluded, training 1/10", rh(r["f01_cell"], 3), "—", rh(r["f01_PS"], 3), "—", "—", f"**{rh(r['f01_PS1H'], 3)}**"])
    rows.append(["HitRate (excluded)", "—", "—", rh(r["hit_PS"], 2), "—", "—", rh(r["hit_PS1H"], 2)])
    md("Table 10 (auto). D6 design variants, test AUC-PR (bold = best in row).", ["Evaluation", "cell mean", "node mean", "PS ($d=2$)", "PS1 ($d=1$)", "PSH (Huber)", "PS1H"], rows)


# ----------------------------------------------------------------------------- Table 6 (comparators under missingness, exp6 추가 5)
def table_missing_comp(split="test"):
    p = R / f"exp6/comparators_missing/metrics_d1_{split}.csv"
    if not p.exists():
        return
    # draft_v8: 절대 수준은 Figure 4 가 보여 주므로(중복 금지) 표는 짝지은 PS − 모델 [95 % CI] (63 의 exp6/comparators_missing)
    names = [("cell", "Cell mean"), ("OLS", "OLS cross-sensor regression"), ("RRR", "Lagged RRR"), ("TRANS", "Masked Transformer"),
             ("RBCPHI", "RBC$^{\\varphi}$"), ("GDN", "GDN re-implementation"), ("OMNI", "OmniAnomaly re-implementation"), ("LAPRPCA", "Laplacian-RPCA, sensor-graph variant"),
             ("CAE2D", "2-D Conv-AE"), ("TRANAD", "TranAD re-implementation")]
    cols = [("base", "none"), ("C 0.5", "i.i.d. $r=0.5$"), ("C 0.7", "i.i.d. $r=0.7$"), ("D pi=0.7,L=4", "burst $\\pi=0.7$"), ("P P3", "real loss P3")]
    rows = []
    for k, lab in names:
        cells = [lab]
        for c, _ in cols:
            r = ps_row("exp6/comparators_missing", f"PS − {k}", "d1", c)
            if r is None:
                cells.append("—"); continue
            v = f"{r['mean']:+.3f}".replace("-", "−")
            cells.append(f"**{v}** {ci(r)}" if r["mean"] > 0.0165 else f"{v} {ci(r)}")
        rows.append(cells)
    md(f"Table 6 (auto). Paired PS − model D1 {split} cell AUC-PR when every model is trained under the same cell-level loss masks (bold = beyond tolerance 0.0165).",
       ["Model", *[c for _, c in cols]], rows)


# ----------------------------------------------------------------------------- Table A1 (grids)
def tableA1():
    try:
        sys.path.insert(0, str(ROOT)); from src.baselines import models as M  # noqa: E402
        grids = {}
        for g in (M.GRIDS, M.GRIDS2, M.GRIDS3, M.GRIDS4):
            grids.update(g)
        rows = []
        for k, cfgs in grids.items():
            keys = sorted({kk for c in cfgs for kk in c}); desc = "; ".join(f"{kk} ∈ {{{', '.join(str(c[kk]) for c in cfgs if kk in c and c[kk] not in [x[kk] for x in cfgs[:cfgs.index(c)] if kk in x])}}}" for kk in keys)
            rows.append([k, desc])
        md("Table A1 (auto, grids only). Search grids from `src/baselines/models.py`.", ["Model key", "Grid"], rows)
    except Exception as e:  # noqa: BLE001
        OUT_MAIN.append(f"Table A1: grid import failed ({type(e).__name__}); keep the hand-filled table.\n")


# ----------------------------------------------------------------------------- S4 full tables
def dump(path, title, cols=None, idx=None):
    if not Path(path).exists():
        return
    t = pd.read_csv(path)
    if cols:
        t = t[[c for c in cols if c in t.columns]]
    md(title, list(t.columns), show(t), target=OUT_S4)


P_COLS = ("p_wilcoxon", "wilcoxon_p")


def show(t):
    """S4 표시용 셀 문자열: 4자리 반올림, 0.0001 미만 p 는 지수 표기(반올림하면 0.0 으로 보임), 빈 값은 —. 값 자체는 바꾸지 않는다."""
    t = t.copy()
    for c in t.columns:
        if c in P_COLS:
            t[c] = [("—" if pd.isna(v) else (f"{v:.1e}" if 0 < v < 1e-4 else str(round(v, 4)))) for v in pd.to_numeric(t[c], errors="coerce")]
    t = t.round(4).astype(str).replace({"nan": "—", "None": "—"})
    return t.values.tolist()


GROUP_LAB = {"exp6/comparators_missing": "comparators under loss", "exp1_baselines": "comparators", "exp1_stssd/original": "original ST-SSD",
             "exp10": "D6 main", "exp10/design": "D6 design variants"}      # 내부 실험 폴더명 → 독자용 (값은 그대로)


def stats_table(title, groups):
    """paired_stats 의 한 묶음. group 열은 한 종류면 빼고, 여러 종류면 독자용 이름으로 바꾼다 (2026-10-13 제출 전 스윕)."""
    sub = PS_[PS_.group.isin(groups)].copy()
    if sub.group.nunique() == 1:
        sub = sub.drop(columns="group")
    else:
        sub["group"] = sub.group.map(GROUP_LAB).fillna(sub.group)
    md(title, list(sub.columns), show(sub), target=OUT_S4)


S4_KEY = """**Key to labels.** The tables keep the column names of the primary result files so that every value can be traced; the labels are defined here.

- **Datasets and evaluations.** d1, d2, d6, smd and asd are D1, D2, D6, SMD and ASD. For D2, d2 is the multi-metric injection (main) and d2dist the distance-only injection (auxiliary); d1_inject, d2_inject_multi and d2_inject_dist are the same three evaluations in the simple-baseline ladder.
- **Column patterns.** A column named after an evaluation and a model (for example d1_PS, d2_cell) is that model's cell-level AUC-PR. Suffixes: _delta is the paired mean of PS − reference, _fracpos (frac_pos, pos) the fraction of positive pairs, _n (n) the number of pairs, _ref the reference model's AUC-PR, _hitrate (hit) HitRate@100 %, _smearing (smear) the smearing rate, f1 the cell F1 and range_f1 the Range-F1 score; columns starting with d_ are paired differences and columns starting with v_ the corresponding verdicts. beats and noninferior are the pre-registered verdicts (PS beyond the tolerance at the level; PS not worse than the tolerance). mean, ci_lo and ci_hi are the paired mean difference and its bootstrap 95 % CI (B = 5000), p_wilcoxon (wilcoxon_p) the Wilcoxon signed-rank p-value, auc_pr and delta a comparator's AUC-PR and PS − comparator.
- **Missingness families and levels.** base is the clean training set; A: fraction f of training windows kept; B: window length T; C: i.i.d. cell loss at rate r; D: bursty loss (Gilbert–Elliott) with mean loss rate pi and mean burst length L; E: whole-anchor outage with probability q; P: real packet-loss masks P1–P3 (block loss rate [0.05, 0.2), [0.2, 0.5), [0.5, 1.0)). Sensitivity families: F threshold percentile p (far is the realised false-alarm rate); G1 training noise and G2 noise in all splits (multiples of the replicate noise); H anchor count; I window length on common blocks (pair 6v12 and 24v12, Tp the other T, T12 the T = 12 model); J1 unlabelled anomalies in training and J2 training without NLOS masks. reference marks the reference level.
- **Simple-baseline ladder.** Z0 one global mean per space; Z0s one global P-spline; Z1 the node mean; Z1t the node mean with the classical z-score; Z2 the node × row × time mean; F01 the P-spline surface of the ladder (K = 8, see the note to Table 3 of the main text); ZF01 its reproduction check; M01 the cell-mean surface (cell in other tables).
- **Comparators.** cell cell mean; cellfill (cell+fill) and cellinterp (cell+interp) cell mean with row-mean fill or linear interpolation of empty cells; OLS cross-sensor ridge regression; RRR lagged reduced-rank regression; PCA PCA reconstruction error; RPCA robust PCA subspace; LAPRPCA Laplacian-RPCA, sensor-graph variant; IF isolation forest; LSTMAE LSTM autoencoder; CAE1D and CAE2D 1-D and 2-D convolutional autoencoders; TRANS masked Transformer; ANOMTRANSFIX Anomaly Transformer with the minimax direction of the original (ANOMTRANS is the earlier version with the two phases reversed, Section S2, not used for judgement); TRANAD TranAD; GDN GDN; OMNI OmniAnomaly; RBCPHI the reconstruction-based contribution RBC$^{\\varphi}$; USAD, USADFIX and USADFIX2 the three USAD implementations of Section S2 (only USADFIX2 is used for judgement); TIMEHIST per-anchor range time-history test; ORACLE true-range residual z-score with surveyed coordinates; RXFP the RX−FP power threshold rule (D2). In config, nc means node-centred input (the node's cell mean subtracted) and raw the original input; the brace terms are the selected hyper-parameters. batch1, batch2, batch2_fix, batch2_fix2, batch3 and batch3_fix are the comparator batches; batch2_fix holds the corrected implementations of Section S2, batch2_fix2 the original-form USAD and batch3_fix the corrected Anomaly Transformer.
- **ST-SSD variants.** STSSD_G1 one global background (G1), STSSD_N the node background (N); R is the original multi-$\\gamma$ window likelihood-ratio statistic, D the dynamic mean, DT the dynamic mean with a row-independent global background (G1T4, exploratory). cell and window mark cell-level and window-level AUC-PR.
- **D6.** scen main and ×3 are the registered and the three-fold injection sizes; eval SD is the registered evaluation and outlier-excluded the evaluation without the outlier cells; f = 0.1 keeps one tenth of the training windows. PS1, PSH and PS1H are the first-difference, Huber and first-difference Huber P-splines; V1 MAD and V2 SD + outlier-excluded are the standardisation variants; R_win and PS_win are window-level AUC-PR of R and PS.
- **Verdicts and notes.** PS better, PS worse and tie are the pre-registered verdicts (tolerance 0.0165 for D1, 0.0171 for D2); training-bug version (excluded) and diverged version (record only) mark the two USAD versions not used for judgement, reversed-minimax version (excluded) the Anomaly Transformer version not used for judgement; batch 3 marks the third batch and batch 3, corrected its correction.
"""


def s4():
    """보충 자료 S4 전체 표. 제목은 독자용 설명(내부 실험 코드명 대신), 본문 번호는 70_supp2tex.py 가 문서 순서로 'Table S#' 를 붙인다."""
    SUB = {"conditions": "training amount, window length and i.i.d. cell loss", "missing_patterns": "bursty loss and whole-anchor outage",
           "imputation": "i.i.d. and bursty loss against the imputation baselines", "phy_masks": "real packet-loss masks P1–P3"}
    VS = {"": "PS and cell mean", "_vs_cellfill": "PS against the cell mean with row-mean fill", "_vs_cellinterp": "PS against the cell mean with linear interpolation"}
    SPLIT = {"val": "validation", "test": "test"}
    for sub in ("conditions", "missing_patterns", "imputation", "phy_masks"):
        for split in ("val", "test"):
            for suffix in ("", "_vs_cellfill", "_vs_cellinterp"):
                p = R / f"exp6/{sub}/table_{split}{suffix}.csv"
                dump(p, f"S4 — Training-cell loss, {SUB[sub]}: {VS[suffix]} ({SPLIT[split]} AUC-PR; columns ending in _delta are PS − reference, _fracpos the fraction of positive pairs)",
                     cols=["family", "level", "d1_cell", "d1_PS", "d1_ref", "d1_delta", "d1_fracpos", "d1_n", "d2_cell", "d2_PS", "d2_ref", "d2_delta", "d2_fracpos", "d2_n", "d2dist_cell", "d2dist_PS", "d2dist_delta", "d2dist_fracpos", "beats", "noninferior"])
    FAM = {"F": "threshold percentile", "G": "training noise", "H": "anchor count (D2)", "I": "window length on common blocks", "J": "training contamination"}
    for fam in ("F", "G", "H", "I", "J"):
        for split in ("val", "test"):
            dump(R / f"exp6/part2/table_{fam}_{split}.csv", f"S4 — Sensitivity: {FAM[fam]} ({SPLIT[split]})")
    # 결측 조건의 비교 모델 (본문 Table 6 의 전체 통계: n, 평균, 95 % CI, Wilcoxon p, 양수 비율)
    stats_table("S4 — Comparators under training-cell loss: paired PS − model statistics, D1 and D2 (n, mean, bootstrap 95 % CI, Wilcoxon p, fraction of positive pairs)",
                ["exp6/comparators_missing"])
    dump(R / "exp1_baselines/paired.csv", "S4 — Comparators, first batch: paired PS − model (test)")
    for sub_, lab in (("batch2", "second batch"), ("batch2_fix", "second batch, corrected implementations"), ("batch2_fix2", "USAD in its original form"), ("batch3_fix", "Anomaly Transformer with the minimax direction of the original")):
        dump(R / f"exp1_baselines/{sub_}/paired.csv", f"S4 — Comparators, {lab}: paired PS − model (test)")
    stats_table("S4 — Comparators: paired statistics for all batches (bootstrap 95 % CI, Wilcoxon p)", ["exp1_baselines"])
    stats_table("S4 — Original ST-SSD statistics: paired statistics", ["exp1_stssd/original"])
    stats_table("S4 — D6: paired statistics (main and ×3 injection; SD and outlier-excluded evaluation)", ["exp10", "exp10/design"])
    dump(R / "exp10/robust/compare_test.csv", "S4 — D6 exploratory: robust standardisation and outlier-excluded evaluation (test)")
    dump(R / "exp10/robust/compare_val.csv", "S4 — D6 exploratory: robust standardisation and outlier-excluded evaluation (validation)")
    dump(R / "exp10/design/compare_val.csv", "S4 — D6 design variants (validation)")
    dump(R / "exp10/design/compare_test.csv", "S4 — D6 design variants (test)")
    dump(R / "exp9/paired.csv", "S4 — Data without location structure (SMD, ASD): PS − model, entity-paired (test)")
    dump(R / "exp1_simple/paired_test.csv", "S4 — Simple-baseline ladder (test; Wilcoxon p from the original aggregation)", cols=["view", "contrast", "mean", "sd", "n", "frac_pos", "wilcoxon_p", "verdict"])


# ----------------------------------------------------------------------------- draft diff
def numbers(s):
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+", s.replace("−", "-"))]


HEADERS = ("Model", "Condition", "Variant", "Test", "Evaluation", "Model key")


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def gen_tables():
    """OUT_MAIN 을 표 번호별로 나눈다: {번호: {라벨: 셀 목록}} ('Table N (auto)' 캡션 기준)."""
    tabs, cur = {}, None
    for line in "\n".join(OUT_MAIN).splitlines():
        m = re.match(r"\*\*Table (A?\d+) \(auto", line)
        if m:
            cur = m.group(1); tabs[cur] = {}
        elif cur and line.startswith("| ") and line.count("|") >= 3:
            c = _cells(line)
            if c[0] not in HEADERS:
                tabs[cur][c[0]] = c
    return tabs


def draft_tables(lines):
    """원고 줄 목록을 표 번호별 행 인덱스로: {번호: [(줄 번호, 셀 목록)]} ('**Table N.**' 캡션 뒤의 표)."""
    tabs, cur = {}, None
    for k, line in enumerate(lines):
        m = re.match(r"\*\*Table (A?\d+)\.\*\*", line)
        if m:
            cur = m.group(1); tabs[cur] = []
        elif cur and line.startswith("| ") and line.count("|") >= 3:
            c = _cells(line)
            if c[0] not in HEADERS and not re.match(r"^-+$", c[0]):
                tabs[cur].append((k, c))
        elif cur and line.strip() == "" and tabs.get(cur):
            cur = None
    return tabs


def diff_against_draft(draft_path):
    lines = Path(draft_path).read_text(encoding="utf-8").splitlines()
    dt, gt = draft_tables(lines), gen_tables()
    report, n_ok, n_bad = [], 0, 0

    def subseq(g, d):
        i = 0
        for x in d:
            if i < len(g) and abs(x - g[i]) <= 0.0006:
                i += 1
        return i == len(g)
    for tno, rows in gt.items():
        if tno == "A1":                                              # 격자 전용 표 (라벨이 모델 키) — 대조 제외
            continue
        if tno not in dt:
            report.append(f"  TABLE {tno}: not in draft ({len(rows)} generated rows)"); continue
        drows = {c[0]: numbers("| " + " | ".join(c) + " |") for _, c in dt[tno]}
        for label, g in rows.items():
            if label not in drows:
                report.append(f"  TABLE {tno}: row '{label}' not in draft"); n_bad += 1; continue
            gen = numbers("| " + " | ".join(g) + " |")
            ok = subseq(gen, drows[label]); n_ok += ok; n_bad += (not ok)
            if not ok:
                report.append(f"  MISMATCH T{tno} '{label}': generated {gen[:8]} vs draft {drows[label][:8]}")
    return n_ok, n_bad, report


def apply_to_draft(draft_path):
    """같은 번호의 표 안에서 라벨이 같은 행을 생성 셀로 덮어쓴다 (원고 행이 더 길면 뒤쪽 열 보존)."""
    path = Path(draft_path); lines = path.read_text(encoding="utf-8").splitlines(); n_rep = 0
    dt, gt = draft_tables(lines), gen_tables()
    for tno, rows in gt.items():
        if tno == "A1":
            continue
        for k, cells in dt.get(tno, []):
            g = rows.get(cells[0])
            if g and len(cells) >= len(g):
                new = g + cells[len(g):]
                if new != cells:
                    lines[k] = "| " + " | ".join(new) + " |"; n_rep += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_rep


def main():
    draft = sys.argv[sys.argv.index("--draft") + 1] if "--draft" in sys.argv else str(PM / "draft/draft_v3.md")
    table4(); table5(); table_missing_comp(sys.argv[sys.argv.index('--mc-split') + 1] if '--mc-split' in sys.argv else 'test'); table7(); table8(); tableA1(); s4()
    if "--apply" in sys.argv:
        print("rows replaced in draft:", apply_to_draft(draft))
    (PM / "tables_auto.md").write_text("# 본문 표 자동 생성 (`analysis/67_math_tables.py`)\n\n" + "\n".join(OUT_MAIN), encoding="utf-8")
    (PM / "supplementary").mkdir(exist_ok=True)
    (PM / "supplementary/S4_full_tables.md").write_text("# S4. Full tables\n\nAll conditions, validation and test splits, all comparators, generated by script from the primary result files and the paired-statistics file (values unrounded to four decimals).\n\n" + S4_KEY + "\n" + "\n".join(OUT_S4), encoding="utf-8")
    ok, bad, rep = diff_against_draft(draft)
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"tables_auto.md and supplementary/S4_full_tables.md written. Draft row check: {ok} rows match, {bad} mismatch")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
