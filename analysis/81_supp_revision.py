"""보충 자료 S6 절(1차 수정에서 더한 분석, 표 S41–S49)을 1차 출력에서 만든다 (2026-10-29).

출력: paper/mathematics/supplementary/S6_revision.md (영문). 70_supp2tex.py 가 S1–S5 뒤에 붙인다.
값은 결과 파일에서 다시 계산하거나(표 S42–S49) 결과 요약 표를 옮긴다(표 S41). 표 번호는 S40 다음부터.
사용법: python analysis/81_supp_revision.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper/mathematics/supplementary/S6_revision.md"
REFS9 = ["CG", "CI", "CMI", "NM", "NMed", "PS", "PS1", "PSH", "PS1H"]
NAMES = {"CG": "cell mean (unit fill)", "CF": "cell + fill", "CI": "cell + interp", "CMI": "cellwise median + interp",
         "NM": "node mean", "NMed": "node median", "GM": "global cell mean"}


def f3(v):
    if pd.isna(v):
        return "—"
    v = 0.0 if abs(v) < 0.0005 else v
    return f"{v:.3f}"


def s3(v):
    if pd.isna(v):
        return "—"
    v = 0.0 if abs(v) < 0.0005 else v
    return ("+" if v >= 0 else "−") + f"{abs(v):.3f}"


def table(header, rows):
    L = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    L += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(L)


def cond_name(family, level):
    level = str(level).strip()
    if family == "base":
        return "none"
    if family == "C" and not level.startswith("r="):
        level = "r=" + level
    if family == "A":
        level = "f=" + level
    if family == "B":
        level = "T=" + level
    if family in ("G1", "G2"):
        level = "k=" + level
    if family == "J1":
        level = "p=" + level
    m = {"C": "i.i.d.", "D": "burst", "E": "anchor outage", "P": "real loss", "A": "training amount", "B": "window length",
         "G1": "training noise", "G2": "noise in all splits", "J1": "training contamination", "J2": "NLOS cells kept in training"}
    lv = (level.replace("r=", "r = ").replace("pi=", "π = ").replace(",L=", ", L = ").replace("q=", "q = ")
          .replace("f=", "f = ").replace("T=", "T = ").replace("k=", "k = ").replace("p=", "p = "))
    if family == "J2":
        lv = ""
    return f"{m.get(family, family)} {lv}".strip()


def cond_label(cond):
    """cluster_ci.csv 의 조건 문자열 → 영문 (실험 6 계열 코드만 바꾸고 나머지는 그대로)."""
    if pd.isna(cond):
        return ""
    cond = str(cond).replace("base 기준", "base").strip()
    m = re.match(r"^(base|A|B|C|D|E|P|G1|G2|J1|J2)(?:\s+(.*))?$", cond)
    if not m:
        return cond
    fam, lev = m.group(1), (m.group(2) or "")
    if fam == "P":
        lev = lev.replace("P ", "")
    return cond_name(fam, lev)


# ---------------------------------------------------------------- S41 학습 seed
def s41():
    md = (R / "expR1/train_seeds/summary.md").read_text(encoding="utf-8")
    rows = []
    for ds, block in re.findall(r"## (d\d)\n\n(.*?)(?=\n## |\Z)", md, flags=re.S):
        for ln in block.splitlines()[2:]:
            c = [x.strip() for x in ln.strip("|").split("|")]
            if len(c) < 6:
                continue
            verdict = c[5].replace("대등", "tie").replace("이김", "PS better").replace("짐", "PS worse")
            name = {"TRANAD": "TranAD", "GDN": "GDN", "CAE2D": "2-D Conv-AE"}[c[0]]
            rows.append([ds.upper(), name, c[1], c[2], c[3], c[4].replace("-", "−"), verdict])
    return ("**Table S41.** Training-seed variability of the deep comparators (clean test data, Table 4 configurations, "
            "training seeds 0–4). SD (training) = mean over (unit, scenario, injection seed) of the SD across the five training "
            "seeds; SD (injection) = mean over (unit, scenario) of the SD across injection seeds at training seed 0. "
            "Verdict per training seed by the pre-registered tolerance.\n\n"
            + table(["Data", "Model", "Mean AUC-PR", "SD (training)", "SD (injection)", "PS − model, seeds 0–4", "Verdict, seeds 0–4"], rows))


# ---------------------------------------------------------------- S42 군집 구간 전체
SRC = {"exp6/conditions": "Table 5, 8", "exp6/missing_patterns": "Table 5", "exp6/imputation": "Table 5",
       "exp6/phy_masks": "Table 5", "exp6/part2": "Table 8", "exp6/comparators_missing": "Table 6",
       "exp1_baselines": "Table 4", "exp1_simple/boot": "Table 3", "exp1_stssd/original": "Table 9",
       "exp10": "Table 10", "exp10/design": "Table 11"}


def s42():
    c = pd.read_csv(R / "expR1/inference/cluster_ci.csv")
    rows = []
    for r in c.itertuples():
        cond = cond_label(r.condition)
        rows.append([SRC.get(r.group, r.group), r.contrast, r.dataset.upper(), cond, r.n, r.clusters, s3(r.mean),
                     f"[{s3(r.iid_lo)}, {s3(r.iid_hi)}]", f"[{s3(r.cl_lo)}, {s3(r.cl_hi)}]", f"{r.width_ratio:.1f}",
                     "yes" if r.units_same_sign else "no"])
    return ("**Table S42.** Paired differences with pair-level (i.i.d.) and two-stage cluster bootstrap 95 % intervals "
            "(clusters: D1 rooms, D2 environments, D6 scans; 5,000 resamples; Section 3.7). Width ratio = cluster / i.i.d. "
            "interval width. 'Same sign' = every unit's mean difference has the sign of the overall mean. Per-unit means are in "
            "the repository (`results/expR1/inference/unit_means.csv`).\n\n"
            + table(["Source", "Contrast", "Data", "Condition", "Pairs", "Clusters", "Mean", "i.i.d. 95 % CI",
                     "Cluster 95 % CI", "Width ratio", "Same sign"], rows))


# ---------------------------------------------------------------- S43 참조 × 탐지기
def s43():
    t = pd.read_csv(R / "expR1/factorial/table_test.csv")
    DET = {"self": "reference's own score", "OLS": "OLS cross-sensor", "GDN": "GDN", "CAE2D": "2-D Conv-AE", "TRANAD": "TranAD"}
    order = {"self": 0, "OLS": 1, "GDN": 2, "CAE2D": 3, "TRANAD": 4}
    rows = []
    for (ds, ck), g in t.groupby(["dataset", "ck"], sort=False):
        fam, lev = ck.split("|", 1)
        for r in sorted(g.itertuples(), key=lambda x: order[x.detector]):
            rows.append([ds.upper(), cond_name(fam, lev), DET[r.detector], f3(r.CG), f3(r.CF), f3(r.CI), f3(r.PS)])
    return ("**Table S43.** Reference × detector factorial (Section 4.3, Table 7), test cell AUC-PR for every condition; "
            "configurations reselected on validation for every (condition, reference, detector). References: cell mean with unit "
            "fill (CG), with node-row fill (CF), with linear interpolation (CI), and the P-spline (PS).\n\n"
            + table(["Data", "Training-cell loss", "Detector", "CG", "CF", "CI", "PS"], rows))


# ---------------------------------------------------------------- S44 운영 임계 F1
def s44():
    rows = []
    for ds, scen in (("d1", r"^SC\d$"), ("d2", r"^SC\dm$")):
        d = pd.read_csv(R / f"exp6/imputation/metrics_{ds}_test.csv")
        d = d[d.scenario.str.match(scen)]
        g = d.groupby(["family", "level", "basis"]).cell_f1.mean().unstack("basis")
        for (fam, lev), r in g.iterrows():
            cf = r.get("cellfill", np.nan) if fam != "base" else r["cell"]
            ci = r.get("cellinterp", np.nan) if fam != "base" else r["cell"]
            rows.append([ds.upper(), cond_name(fam, lev), f3(r["PS"]), f3(r["cell"]), f3(cf), f3(ci)])
    rows.sort(key=lambda x: (x[0], x[1] != "none", x[1]))
    return ("**Table S44.** Cell-level F1 at the operating threshold (97.5th percentile of validation cell scores), test split, "
            "same runs as Table 5 (D2 multi-metric scenarios). Without loss the three cell-mean variants coincide.\n\n"
            + table(["Data", "Training-cell loss", "PS", "cell mean (unit fill)", "cell + fill", "cell + interp"], rows))


# ---------------------------------------------------------------- S45 임계 보정
def s45():
    f = pd.read_csv(R / "expR1/thresholds/far.csv")
    g = f.groupby(["dataset", "ref", "split"])[["far_cell", "far_a", "far_b", "far_c", "rec_a", "rec_b", "rec_c"]].mean().reset_index()
    rows = [[r.dataset.upper(), r.ref, r.split, f3(r.far_cell), f3(r.far_a), f3(r.far_b), f3(r.far_c),
             f3(r.rec_a), f3(r.rec_b), f3(r.rec_c)] for r in g.itertuples()]
    return ("**Table S45.** Realised false-alarm rates (nominal 0.025) and recall under three group-threshold rules: (a) the cell "
            "threshold applied to the group maximum, (b) row-wise thresholds, (c) a threshold calibrated on validation group "
            "scores (Section 3.4). Means over units, scenarios and seeds; recall on the injected test sets only.\n\n"
            + table(["Data", "Reference", "Split", "FAR cell", "FAR group (a)", "FAR group (b)", "FAR group (c)",
                     "Recall (a)", "Recall (b)", "Recall (c)"], rows))


# ---------------------------------------------------------------- S46 빈 칸 예측 분산
def s46():
    md = (R / "expR1/pred_var/summary.md").read_text(encoding="utf-8")
    rows = []
    for ln in md.splitlines():
        c = [x.strip() for x in ln.strip("|").split("|")]
        if len(c) == 8 and re.match(r"^0\.\d$", c[0]):
            rows.append([c[0], "empty" if c[1] == "예" else "observed"] + c[2:])
    tail = re.search(r"r 0\.5 ([\d.]+), r 0\.7 ([\d.]+)", md)
    return ("**Table S46.** Prediction variance of the P-spline reference (in units of σ², λ fixed at its SAP value) at observed and "
            "empty cells of the actual D1 training masks under i.i.d. loss r (Remark 3). For comparison, the cell mean's variance "
            f"1/c at observed cells has median {tail.group(1)} (r = 0.5) and {tail.group(2)} (r = 0.7); at empty cells it has no "
            "estimate.\n\n"
            + table(["r", "Cells", "Count", "Mean", "Median", "90 %", "99 %", "Max"], rows))


# ---------------------------------------------------------------- S47 시뮬레이션 전체
def s47():
    design = json.loads((R / "expR1/sim/design.json").read_text(encoding="utf-8"))
    m = pd.concat([pd.read_csv(R / "expR1/sim/metrics.csv"), pd.read_csv(R / "expR1/sim/cells43_metrics.csv")])   # 보완 칸 43–45
    piv = m.groupby(["cell", "ref"]).auc_pr.mean().unstack("ref")
    refs = ["CG", "CF", "CI", "CMI", "NM", "NMed", "PS", "PS1", "PSH", "PS1H"]
    reps = m.groupby("cell").rep.nunique()
    rows = []
    for d in design:
        c = d["cfg"]
        desc = (f"N {c['N']}, {c['L']}, {c['noise']}, {c['miss']} r {c['r']:g}, p {c['p']:g}, {c['atype']}, ×{c['mag']:g}")
        rows.append([d["id"], desc, reps.get(d["id"], 0)] + [f3(piv.loc[d["id"], k]) if d["id"] in piv.index else "—" for k in refs])
    return ("**Table S47.** Simulation study (Section 4.8): mean test cell AUC-PR per design cell and reference. Design: N windows per "
            "node; profile L0 (flat), L1, L2 (increasing within-window structure); noise G (Gaussian), T3 (t₃), CN (contaminated "
            "normal); missingness mechanism and rate r; training contamination p; anomaly type and magnitude (multiples of the "
            "noise SD). Replicates per cell in the third column.\n\n"
            + table(["Cell", "Design", "Reps"] + refs, rows))


# ---------------------------------------------------------------- S48 Wi-Fi
def s48():
    u = pd.read_csv(R / "expR1/uji/metrics_test.csv")
    refs = ["GM", "CG", "CF", "CI", "NM", "PS", "PS1", "PSH", "PS1H"]
    lab = {("N", "-"): "natural missingness", ("M", "0.5"): "+ 50 % random loss", ("M", "0.7"): "+ 70 % random loss",
           ("F", "-"): "3 training windows"}
    rows = []
    for (cond, lev), g in u.groupby(["cond", "level"], sort=False):
        a = g.groupby("ref").auc_pr.mean()
        far = g.groupby("ref").far_cell.mean()
        rows.append([lab[(cond, str(lev))], "AUC-PR"] + [f3(a.get(k)) for k in refs])
        rows.append(["", "cell FAR"] + [f3(far.get(k)) for k in refs])
    b = pd.read_csv(R / "expR1/uji/by_set_test.csv")                  # 사후 분석 expR1_6b (세트별)
    glab = {"same directions (trn14, trn15, tst01)": "training orientation (trn14, trn15, tst01)",
            "tst05 (other directions)": "Test-05 (other directions)"}
    rows2 = []
    for cond, clab in (("N", "natural missingness"), ("M", "+ 70 % random loss")):
        for g, gl in glab.items():
            sub = b[(b.cond == cond) & (b.group == g)]
            a = sub.groupby("ref").auc_pr.mean(); far = sub.groupby("ref").far_cell.mean()
            rows2.append([clab, gl, "AUC-PR"] + [f3(a.get(k)) for k in refs])
            rows2.append(["", "", "cell FAR"] + [f3(far.get(k)) for k in refs])
    s = pd.read_csv(R / "expR1/uji/shift_far.csv", encoding="utf-8-sig")  # expR1_6_uji shift (서술)
    s = s[s.month >= 2].copy()
    s["orient"] = np.where(s.set.isin(["tst05", "tst10"]), "other directions", "training orientation")
    rows3 = []
    for (dev, ori), g in s.groupby(["device", "orient"], sort=False):
        phone = {"S3": "Galaxy S3", "A5": "Galaxy A5 (2017)"}[dev]
        months = f"{g.month.min()}–{g.month.max()}" if g.month.nunique() > 1 else f"{g.month.min()}"
        rows3.append([phone, ori, ", ".join(sorted(g.set.unique())), months, len(g),
                      f3(g.far_cell.min()), f3(g.far_cell.median()), f3(g.far_cell.max())])
    return ("**Table S48.** Wi-Fi fingerprints (D7, Section 4.7), test split: mean cell AUC-PR and realised cell false-alarm rate "
            "(nominal 0.025) per condition and reference; GM = one global cell mean for all nodes. The upper part is the registered "
            "test (sets trn14, trn15, tst01, tst05). The middle part splits it by set (post hoc, same injections): Test-05 was recorded "
            "at the same points but facing the two other directions (dataset description by Mendoza-Silva et al., 2018, cited in the main text). "
            "The lower part gives the realised cell false-alarm rate of the first-month P-spline reference (natural missingness, calibrated "
            "on the validation sets) on the sets of later months at the same points, one value per set and floor; the per-set values are in "
            "results/expR1/uji/shift_far.csv of the code repository. In month 25 some sets were recorded with another phone; tst06 and tst10 "
            "are its counterparts of tst01 and tst05 (same points; per point, their RSS is closest to tst01 and to tst05 of the same month, "
            "respectively).\n\n"
            + table(["Condition", "Metric"] + refs, rows) + "\n\n"
            + table(["Condition", "Test sets", "Metric"] + refs, rows2) + "\n\n"
            + table(["Phone", "Orientation", "Sets", "Months", "Values", "Min", "Median", "Max"], rows3))


# ---------------------------------------------------------------- S49 HYMN
def s49():
    m = pd.read_csv(R / "expR1/hymn/metrics_test.csv")
    m = m[m.scenario.isin(["SC1", "SC2", "SC3"])]
    rows = []
    for (cond, size, ev), g in m.groupby(["cond", "size", "eval"], sort=False):
        a = g.groupby("ref").auc_pr.mean()
        label = ("natural" if cond == "N" else "+ 70 % training-cell loss") + f", {'main' if size == 'x1' else '×3'}, " + \
                ("complete" if ev == "full" else "outlier-excluded (exploratory)")
        rows.append([label] + [f3(a.get(k)) for k in REFS9])
    b = pd.read_csv(R / "expR1/hymn/bootstrap_test.csv")
    brow = [["natural" if r.cond == "N" else "+ 70 % loss", f"{r.a} − {r.b}", s3(r.est), f"[{s3(r.lo)}, {s3(r.hi)}]"]
            for r in b.itertuples()]
    return ("**Table S49.** HYMN (D8, Section 4.6, Table 13), test cell AUC-PR for every condition, injection size and evaluation, "
            "and paired differences at the main size under the complete evaluation with point-cluster bootstrap 95 % intervals "
            "(48 points, 2,000 resamples). Tolerance 0.0178.\n\n"
            + table(["Condition, size, evaluation"] + REFS9, rows) + "\n\n"
            + table(["Condition", "Contrast", "Mean", "Cluster 95 % CI"], brow))


def main():
    parts = ["## S6. Analyses Added in the Revision",
             "The tables below were added in the first revision. Each comes from a registered analysis (registration chronology "
             "in the repository) and is generated from the primary outputs by `analysis/81_supp_revision.py`.",
             s41(), s42(), s43(), s44(), s45(), s46(), s47(), s48(), s49()]
    text = "\n\n".join(parts) + "\n"
    if re.search(r"[가-힣]", text):
        bad = sorted(set(re.findall(r"[^\x00-\x7F]*[가-힣]+[^\x00-\x7F]*", text)))
        raise SystemExit(f"Korean left in S6: {bad[:10]}")
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(OUT, len(text), "chars;", text.count("**Table S"), "tables")


if __name__ == "__main__":
    sys.exit(main())
