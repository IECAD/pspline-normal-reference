"""Mathematics 재투고(거절 후, draft_v11) 보충 자료 S8 (영문)을 주 결과 파일에서 만든다 (2026-11-25).

표 S52: 등록 시뮬레이션 R3-1 (원고 4.8절, 표 15) — results/expR3/sim/metrics.csv, 등록 예측 analysis/out/93_r3_predictions.json,
        척도 불변 기준값 results/expR1/sim/metrics.csv (R1-5 칸 2·7). 판정 규칙은 doc/실험R3-1_계획.md 4절(94_r3_judge.py 와 같은 식).
표 S53: 정리 1 의 회고 검증 (원고 3.6절 표 1) — analysis/out/91_theorem1_sim_points.csv (R1-5 저장 출력의 칸·참조 평균),
        analysis/out/92_theorem1_d1_points.csv (D1 실험 6 보간 조건의 σ_R 비율·AUC-PR). 회귀·F 검정은 91·92b 와 같은 식으로 다시 계산한다.
출력: paper/mathematics/supplementary/S8_resubmit.md. 70_supp2tex.py 가 S7 뒤에 붙인다.
점검: 숫자가 analysis/out/91_theorem1_sim_check.md, 92b_theorem1_d1_class.md, 94_r3_judge.md 와 같은지 main() 끝에서 확인한다.
사용법: python analysis/95_supp_resubmit.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/mathematics/supplementary/S8_resubmit.md"
R3 = ROOT / "results/expR3/sim/metrics.csv"
R15 = ROOT / "results/expR1/sim/metrics.csv"
PRED = ROOT / "analysis/out/93_r3_predictions.json"
P91 = ROOT / "analysis/out/91_theorem1_sim_points.csv"
P92 = ROOT / "analysis/out/92_theorem1_d1_points.csv"

REFS6 = ["CG", "CF", "CI", "PS", "PS1", "NM"]          # 표 15·표 1 의 여섯 참조
CLASS5 = ["CF", "CG", "CI", "PS", "PS1"]               # C 묶음의 다섯 참조
CLASS_SIM = ["CF", "CI", "PS", "PS1"]                  # 따름정리 1 의 동치류 후보 (시뮬레이션)
D1_ROOMS = [("classroom", "classroom"), ("hallway", "hallway"), ("lab", "laboratory")]
D1_LEVEL = {"기준": "complete", "r=0.3": "i.i.d. $r=0.3$", "r=0.5": "i.i.d. $r=0.5$", "r=0.7": "i.i.d. $r=0.7$",
            "pi=0.3,L=4": "burst $\\pi=0.3,L=4$", "pi=0.5,L=4": "burst $\\pi=0.5,L=4$", "pi=0.5,L=8": "burst $\\pi=0.5,L=8$",
            "pi=0.7,L=4": "burst $\\pi=0.7,L=4$"}
D1_REF = {"cell": "unit fill", "cellfill": "row-mean fill", "cellinterp": "interpolation", "PS": "PS"}


def read(p):
    return pd.read_csv(p, encoding="utf-8-sig")


def f3(x, sign=False):
    return (f"{x:+.3f}" if sign else f"{x:.3f}").replace("-", "−")


def f4(x):
    return f"{x:+.4f}".replace("-", "−")


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def f2(x, sign=False):
    return (f"{x:+.2f}" if sign else f"{x:.2f}").replace("-", "−")


def ols(y, X):
    X = np.asarray(X, float); y = np.asarray(y, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    return coef, r, float(r @ r), len(y) - X.shape[1]


def f_test(rss0, dof0, rss1, dof1):
    F = ((rss0 - rss1) / (dof0 - dof1)) / (rss1 / dof1)
    return F, 1.0 - stats.f.cdf(F, dof0 - dof1, dof1)


def dummies(labels, base):
    cats = [c for c in sorted(set(labels)) if c != base]
    return np.column_stack([[1.0 if l == c else 0.0 for l in labels] for c in cats]), cats


# ----------------------------------------------------------------------------------------------------------------------
def table_s52(check):
    d = read(R3)
    pred = json.load(open(PRED, encoding="utf-8"))
    s_of = d["noise"].astype(str).str.extract(r"@([\d.]+)")[0].astype(float).fillna(1.0)
    d["Ve"] = d["sdR"] ** 2 - s_of ** 2
    per = d.groupby(["cell", "tags", "N", "miss", "r", "atype", "mag", "noise", "ref"], as_index=False).agg(
        auc=("auc_pr", "mean"), sd=("auc_pr", "std"), Ve=("Ve", "mean"), n=("auc_pr", "size"))
    piv = d.pivot_table(index=["cell", "rep"], columns="ref", values="auc_pr")
    gap_sd = (piv["PS"] - piv["CI"]).groupby(level=0).std()
    sd_min, sd_max = per.sd.min(), per.sd.max()
    L = ["**Table S52.** Registered simulation R3-1 (main text Section 4.8, Table 15): full results. Fifteen design cells, 100 replicates each, "
         "with the generator, references, scoring, calibration and metric of the simulation of Section 4.8 (12 nodes, 4 rows, $T=12$, 5 validation "
         "and 5 test windows; shape L1 with amplitude 2; Gaussian noise with SD $\\sigma$; no contamination; a mixed anomaly in half of the windows "
         "unless stated). Only two additions were made to the simulation code: a noise scale $s$ (noise SD $s$, anomaly size $2s$) and an end mask "
         "that removes the last $g$ steps from every training window and row. Entries are means over the 100 replicates of the test AUC-PR; $V_e$ is "
         f"estimated as $\\mathrm{{sd}}_R^{{2}}-\\sigma^{{2}}$ (Proposition 4′). The SD of AUC-PR across replicates is {sd_min:.3f}–{sd_max:.3f} per (cell, "
         f"reference), so the standard error of a mean is at most {sd_max / 10:.4f} and that of an unpaired difference of two means at most "
         f"{sd_max * np.sqrt(2) / 10:.4f}; the paired difference PS − CI has SD {gap_sd.min():.3f}–{gap_sd.max():.3f} across replicates (standard error "
         f"{gap_sd.min() / 10:.4f}–{gap_sd.max() / 10:.4f}). References: cell mean with unit fill (CG), with row-mean fill (CF), with linear interpolation "
         "(CI), second-order P-spline (PS), first-order P-spline (PS1), node mean (NM). The predictions, their inputs and the decision rules were "
         "fixed on 13 November 2026 before any cell was run (Section S5); the registered prediction file gives $\\Phi_0$ = "
         f"{pred['phi0']:.4f} and $\\kappa$ = {f4(pred['kappa'])}, fitted on the saved outputs of Section 4.8. "
         "(a) Group A: five windows, i.i.d. loss $r=0.5$, anomaly type × size; AUC-PR by reference. (b) Group A: $V_e$ by reference. "
         "(c) Group A: registered ratios in which $\\kappa$ cancels, and the rank correlation between AUC-PR and $-V_e$ over the six references; "
         "the verdict in brackets applies the registered tolerance. "
         "(d) Group B: end gap $g$ ($g=0$ reproduces the five-window cell of Section 4.8); observed and registered values. "
         "(e) Group B: AUC-PR of the other references. (f) Group C: noise scale $s$; each observed value against the $s=1$ cell of Section 4.8 "
         "with the same design. Verdicts follow the registered rules (Table 15).", ""]
    # (a) AUC-PR by reference, (b) V_e by reference
    A = per[per.tags == "A"]
    for part, col, title in (("a", "auc", "AUC-PR"), ("b", "Ve", "$V_e$")):
        L += ["" if part == "b" else None, f"| ({part}) Cell: anomaly, size | " + " | ".join(REFS6) + " |", "|" + "---|" * 7]
        L = [x for x in L if x is not None]
        for cell, sub in A.groupby("cell"):
            v = sub.set_index("ref")[col]
            L.append(f"| {cell}: {sub.atype.iloc[0]}, {sub.mag.iloc[0]:.0f} | " + " | ".join(f3(v[r]) for r in REFS6) + " |")
    # (c) ratios
    pa = pred["ratios"]
    L += ["", "| (c) Cell | PS − CI | A1 (PS − CF)/(PS − CI) | A2 (PS − CG)/(PS − CI) | A3 (PS1 − PS)/(PS − CI) | A4 Spearman(AUC-PR, $-V_e$) | Order by AUC-PR |",
          "|" + "---|" * 7]
    okA = []
    for cell, sub in A.groupby("cell"):
        a = sub.set_index("ref").auc; v = sub.set_index("ref").Ve
        dci = a["PS"] - a["CI"]
        r1, r2, r3 = (a["PS"] - a["CF"]) / dci, (a["PS"] - a["CG"]) / dci, (a["PS1"] - a["PS"]) / dci
        rho = stats.spearmanr(a[REFS6], -v[REFS6])[0]
        tol = 0.5 if float(sub.mag.iloc[0]) == 1.0 else 0.25
        ok = (abs(r1 - pa["A1"]) <= tol, abs(r2 - pa["A2"]) <= tol, abs(r3 - pa["A3"]) <= tol, rho >= 0.9)
        okA.append(ok)
        mark = lambda x, o: f"{x} ({'held' if o else 'failed'})"
        order = " > ".join(a[REFS6].sort_values(ascending=False).index)
        L.append(f"| {cell} | {f3(dci, True)} | {mark(f2(r1), ok[0])} | {mark(f2(r2), ok[1])} | {mark(f2(r3, True), ok[2])} | {mark(f2(rho, True), ok[3])} | {order} |")
    okA = np.array(okA)
    L.append(f"| Registered | — | {pa['A1']:.2f} ± 0.25 (± 0.5 at size 1): held {okA[:, 0].sum()}/6 | {pa['A2']:.2f}, same: held {okA[:, 1].sum()}/6 | "
             f"{f2(pa['A3'], True)}, same: held {okA[:, 2].sum()}/6 | ≥ 0.9: held {okA[:, 3].sum()}/6 | PS > PS1 > CI > CF > CG > NM |")
    check["A_ok"] = (okA[:, 0].sum(), okA[:, 1].sum(), okA[:, 2].sum(), okA[:, 3].sum())
    # (d) end gap
    pb = {int(x["g"]): x for x in pred["end_gap"]}
    B = per[per.tags == "B"]
    L += ["", "| (d) End gap $g$ | $V_e$ PS: observed / registered | $V_e$ CI: observed / registered | AUC-PR PS | AUC-PR CI | PS − CI: observed / registered | Median $\\hat{\\lambda}$ (PS) |",
          "|" + "---|" * 7]
    gaps, others, vdiff = {}, [], {}
    for cell, sub in B.groupby("cell"):
        g = int(round(float(sub.r.iloc[0]))) if sub.miss.iloc[0] == "END" else 0
        a = sub.set_index("ref").auc; v = sub.set_index("ref").Ve
        lam = d[(d.cell == cell) & (d.ref == "PS")].lam_med.median()
        gaps[g] = a["PS"] - a["CI"]
        vdiff[g] = (v["PS"] - pb[g]["Ve_PS"], v["CI"] - pb[g]["Ve_CI"])
        reg_gap = f3(pb[g]["gap"], True) if abs(pb[g]["Ve_PS"]) < 0.5 and abs(pb[g]["Ve_CI"]) < 0.5 else f"negative ({f3(pb[g]['gap'], True)}, outside the expansion)"
        L.append(f"| {g} | {f3(v['PS'])} / {f3(pb[g]['Ve_PS'])} | {f3(v['CI'])} / {f3(pb[g]['Ve_CI'])} | {f3(a['PS'])} | {f3(a['CI'])} | {f3(gaps[g], True)} / {reg_gap} | {lam:.2f} |")
        others.append((g, a, d[d.cell == cell].empty_frac.mean()))
    gstar = min([g for g in sorted(gaps) if g > 0 and gaps[g] < 0], default=None)
    check["gstar"] = gstar; check["gap0"] = gaps[0]
    L += ["", "| (e) End gap $g$ | CG | CF | PS1 | NM | PS1 − CI | Empty training cells (fraction) |", "|" + "---|" * 7]
    for g, a, emp in others:
        L.append(f"| {g} | {f3(a['CG'])} | {f3(a['CF'])} | {f3(a['PS1'])} | {f3(a['NM'])} | {f3(a['PS1'] - a['CI'], True)} | {emp:.3f} |")
    # (f) scale
    o = read(R15)
    base = o[o.cell.isin([2, 7]) & o.ref.isin(CLASS5)].groupby(["cell", "ref"]).auc_pr.mean()
    L += ["", "| (f) Cell | Design, noise scale $s$ | Reference | AUC-PR observed | AUC-PR at $s=1$ (Section 4.8 cell) | Difference | Within ±0.01 |", "|" + "---|" * 7]
    hits, tot, c2 = 0, 0, []
    for cell, sub in per[per.tags == "C"].groupby("cell"):
        N, r = int(sub.N.iloc[0]), float(sub.r.iloc[0])
        ref_cell = 2 if (N == 5 and r == 0.0) else 7
        s = float(str(sub.noise.iloc[0]).split("@")[1])
        a = sub.set_index("ref").auc
        for ref in CLASS5:
            diff = a[ref] - base[(ref_cell, ref)]
            ok = abs(diff) <= 0.01; hits += ok; tot += 1
            design = f"$N=5$, $r=0$, $s={s:g}$" if ref_cell == 2 else f"$N=10$, i.i.d. $r=0.5$, $s={s:g}$"
            L.append(f"| {cell} | {design} | {ref} | {f3(a[ref])} | {f3(base[(ref_cell, ref)])} | {f3(diff, True)} | {'yes' if ok else 'no'} |")
        c2.append((cell, s, a["PS"] - a["CI"], base[(ref_cell, "PS")] - base[(ref_cell, "CI")]))
    check["C1"] = (hits, tot)
    L += [f"| C1 rule | at least 18 of {tot} within ±0.01 | — | — | — | {hits} of {tot} | **failed** |",
          "| C2 PS − CI | " + "; ".join(f"cell {c} ($s={s:g}$): {f3(g, True)} against {f3(gr, True)}" for c, s, g, gr in c2) + " | — | — | — | all within ±0.01 | held |"]
    L += ["", "Outcome. Of the nine registered predictions, seven held and two failed by the registered rules (Table 15). A2: with anomaly size 3 the "
          "unit-fill gap exceeded the registered ratio in all three cells (1.85–2.07 against 1.56); unit fill concentrates an error of about 2.4$\\sigma$ "
          "on the 3 % empty cells, so $\\max_t|d_t|$ is not small and the second-order expansion does not apply to it (Remark 5(c)); within the class "
          "(CF, CI, PS, PS1) every ratio held. C1: 16 of the 20 values were within ±0.01 of the $s=1$ cell (rule: 18); the four misses are at most 0.013 "
          "and at most 1.8 standard errors (the unpaired standard error is about 0.0075, so the registered ±0.01 is 1.3 standard errors), and three of "
          "them share the training windows of one cell; the paired comparison C2 held. Group B, which predicted the gap at which the P-spline loses to "
          f"interpolation, held with $g^{{*}}={gstar}$ (registered 3 ± 1); at $g=2$ the observed sign differs from the registered one, which is inside "
          "the ±1 band of the rule, and $V_e$ is there nearly equal for the two references (0.376 against 0.389) while the P-spline's error sits "
          "in the end cells. The observed $V_e$ of both references agree with the registered formula within "
          f"{max(abs(x) for g in vdiff if g <= 3 for x in vdiff[g]):.2f} for $g\\le3$ (median $\\hat{{\\lambda}}$ 1.05–1.13 against the assumed 1.08); "
          f"at $g=4$ the P-spline's $V_e$ exceeds the formula by {vdiff[4][0]:.2f}. Code and outputs: experiments/expR3_1_sim.py, analysis/93_r3_predictions.py, "
          "analysis/94_r3_judge.py and results/expR3/sim/ in the repository."]
    return L


# ----------------------------------------------------------------------------------------------------------------------
def design_label(row):
    miss = {"MCAR": "i.i.d.", "BURST": "burst", "MNAR": "value-dependent"}[row.miss]
    loss = "complete" if (row.r == 0 and row.miss == "MCAR") else f"{miss} $r={row.r:g}$"
    return f"$N={int(row.N)}$, {row.L}, {loss}, {row.atype} {row.mag:g}"


def table_s53(check):
    P = read(P91)
    L = ["", "**Table S53.** Retrospective checks of Theorem 1 on the saved outputs (main text Section 3.6, Table 1). None of these outputs was produced "
         "for the theorem; the theorem was formulated after them. (a)–(c) Simulation of Section 4.8 (Table S47): per design cell the mean test AUC-PR and the "
         "mean $V_e=\\mathrm{sd}_R^{2}-1$ of six references, Gaussian noise without contamination (35 cells). (a) Within each cell, the rank correlation "
         "between AUC-PR and $-V_e$ and the two orders. (b) Pooled fit over the cells with a mixed anomaly of size 2: AUC-PR = $\\Phi_0+\\kappa V_e$ on the "
         "points with $V_e<0.5$; F tests for reference dummies, for a $V_e^{2}$ term and, on the points with empty cells, for the share of the error that "
         "sits in empty cells. (c) The difference PS − CI in each cell against its prediction $\\kappa\\,\\Delta V_e$, "
         "$\\Delta V_e=V_e(\\mathrm{PS})-V_e(\\mathrm{CI})$, with the effective number of windows per cell "
         "$c_{\\mathrm{eff}}=c(1-r)$. (d)–(e) D1, Experiment 6 interpolation conditions (Section 4.3): per room, condition and injection seed, the test "
         "AUC-PR of four references against $x=(\\sigma_R\\text{ ratio})^{2}-1$, which is $V_e$ up to a room-specific constant (Proposition 4′). (d) Fit "
         "per room on the points with $x<2$ (unit fill alone reaches $x$ up to 225), with all four references and with the class only (row-mean fill, "
         "interpolation, PS); the mean residual of the unit-fill points with $x<2$ from the class line (negative: worse than the class at equal $x$). (e) Classroom: PS − interpolation against $\\kappa\\,\\Delta x$ for every condition and seed. The 156 pooled points of (b) and the "
         "264 D1 points are in analysis/out/91_theorem1_sim_points.csv and 92_theorem1_d1_points.csv in the repository.", ""]
    # (a)
    G = P[P.ref.isin(REFS6)]
    L += ["| (a) Cell | Design | Spearman(AUC-PR, $-V_e$) | Order by AUC-PR | Order by $V_e$ (smallest first) |", "|---|---|---|---|---|"]
    rhos = []
    for cell, sub in G.groupby("cell"):
        s = sub.set_index("ref")
        rho = stats.spearmanr(s.loc[REFS6, "auc"], -s.loc[REFS6, "Ve"])[0]
        rhos.append(rho)
        L.append(f"| {cell} | {design_label(sub.iloc[0])} | {f2(rho, True)} | {' > '.join(s.auc.sort_values(ascending=False).index)} | "
                 f"{' < '.join(s.Ve.sort_values().index)} |")
    rhos = np.array(rhos)
    L.append(f"| All | 35 cells | median {np.median(rhos):+.2f}, minimum {rhos.min():+.2f}; {int((rhos >= 0.8).sum())}/35 cells with $\\rho\\ge0.8$ | — | — |")
    check["rho_med"], check["rho_min"], check["rho_n"] = np.median(rhos), rhos.min(), int((rhos >= 0.8).sum())
    # (b) pooled fit
    M = G[(G.atype == "mix") & (G.mag == 2.0)]
    F_ = M[M.Ve < 0.5]
    y = F_.auc.to_numpy(); x = F_.Ve.to_numpy()
    X0 = np.column_stack([np.ones_like(x), x])
    c0, r0, rss0, dof0 = ols(y, X0)
    D, cats = dummies(F_.ref.tolist(), "CF")
    c1, r1, rss1, dof1 = ols(y, np.column_stack([X0, D]))
    Fd, pd_ = f_test(rss0, dof0, rss1, dof1)
    c2_, r2, rss2, dof2 = ols(y, np.column_stack([X0, x ** 2]))
    Fq, pq = f_test(rss0, dof0, rss2, dof2)
    r2v = 1 - rss0 / float(((y - y.mean()) ** 2).sum())
    E = F_[F_["empty"] > 0].copy()                      # "empty" 열 (DataFrame.empty 속성과 이름이 겹침)
    share = (E.rmse_empty ** 2 * E["empty"]) / (E.mse)
    ye, xe = E.auc.to_numpy(), E.Ve.to_numpy()
    ce0, _, rsse0, dofe0 = ols(ye, np.column_stack([np.ones_like(xe), xe]))
    ce1, _, rsse1, dofe1 = ols(ye, np.column_stack([np.ones_like(xe), xe, share.to_numpy()]))
    Fe, pe = f_test(rsse0, dofe0, rsse1, dofe1)
    check.update(phi0=c0[0], kappa=c0[1], r2=r2v, sd=np.sqrt(rss0 / dof0), Fd=Fd, pd=pd_, Fq=Fq, pq=pq, Fe=Fe, pe=pe, npts=len(F_), ntot=len(M), nemp=len(E))
    L += ["", "| (b) Fit on the pooled points | Value |", "|---|---|",
          f"| Points (cells with a mixed anomaly of size 2, six references) | {len(M)}; {len(F_)} with $V_e<0.5$ |",
          f"| $\\Phi_0$, $\\kappa$ | {c0[0]:.4f}, {f4(c0[1])} |",
          f"| $R^{{2}}$, residual SD | {r2v:.3f}, {np.sqrt(rss0 / dof0):.4f} |",
          f"| Reference dummies (five, CF as base): F, p | {Fd:.2f}, {fp(pd_)}; coefficients " + ", ".join(f"{c} {f4(v)}" for c, v in zip(cats, c1[2:])) + " |",
          f"| $V_e^{{2}}$ term: coefficient, F, p | {f3(c2_[2], True)}, {Fq:.2f}, {fp(pq)} |",
          f"| Share of the error in empty cells ({len(E)} points with empty cells): coefficient, F, p | {f4(ce1[2])}, {Fe:.2f}, {fp(pe)} |"]
    # (c) PS − CI vs kappa dVe
    kappa = c0[1]
    L += ["", "| (c) Cell | Design | $c_{\\mathrm{eff}}$ | $\\Delta V_e$ = $V_e$(PS) − $V_e$(CI) | PS − CI observed | Predicted $\\kappa\\,\\Delta V_e$ |", "|" + "---|" * 6]
    rows = []
    for cell, sub in M.groupby("cell"):
        s = sub.set_index("ref")
        c_eff = int(s.N.iloc[0]) * (1 - float(s.r.iloc[0]))      # effective windows per cell, N(1 − r), as in 91
        dV = s.loc["PS", "Ve"] - s.loc["CI", "Ve"]                        # Φ(PS) − Φ(CI) = κ (V_PS − V_CI), κ < 0
        rows.append((c_eff, cell, design_label(sub.iloc[0]), dV, s.loc["PS", "auc"] - s.loc["CI", "auc"], kappa * dV))
    rows.sort()
    within = [abs(r[4] - r[5]) <= 0.01 for r in rows if r[0] >= 3]
    for c_eff, cell, lab, dV, obs, prd in rows:
        L.append(f"| {cell} | {lab} | {c_eff:.1f} | {f3(dV, True)} | {f3(obs, True)} | {f3(prd, True)} |")
    L.append(f"| Cells with $c_{{\\mathrm{{eff}}}}\\ge3$ | {len(within)} | — | — | observed − predicted within ±0.01 in {sum(within)}/{len(within)} | — |")
    check["c_within"] = (sum(within), len(within))
    # (d) D1 per room
    Q = read(P92)
    Q["level"] = Q.level.map(D1_LEVEL)
    L += ["", "| (d) Room | References in the fit | Points | Range of $x$ | $\\Phi_0$ | $\\kappa$ | $R^{2}$, residual SD |", "|" + "---|" * 7]
    L2 = ["", "| (d, continued) Room | References in the fit | Reference dummies: F, p | Dummy coefficients | Unit-fill points: mean residual from the class line | Unit-fill points |", "|" + "---|" * 6]
    for room, rname in D1_ROOMS:
        q = Q[Q.space == room]
        for fit_refs, fit_name, base in (( ["cell", "cellfill", "cellinterp", "PS"], "all four", "cell"), (["cellfill", "cellinterp", "PS"], "class only", "cellfill")):
            qq = q[q.basis.isin(fit_refs) & (q.x < 2.0)]              # 92 (b) 와 같이 x < 2 (단위 채움만 x 가 2 를 넘음)
            yy, xx = qq.auc_test.to_numpy(), qq.x.to_numpy()
            X0 = np.column_stack([np.ones_like(xx), xx])
            cc0, rr0, rs0, df0 = ols(yy, X0)
            D, cats = dummies(qq.basis.tolist(), base)
            cc1, rr1, rs1, df1 = ols(yy, np.column_stack([X0, D]))
            Fd_, pd__ = f_test(rs0, df0, rs1, df1)
            r2_ = 1 - rs0 / float(((yy - yy.mean()) ** 2).sum())
            L.append(f"| {rname} | {fit_name} | {len(qq)} | {f2(xx.min())}–{f2(xx.max())} | {cc0[0]:.3f} | {f3(cc0[1], True)} | {r2_:.3f}, {np.sqrt(rs0 / df0):.4f} |")
            if fit_name == "class only":
                qc = q[(q.basis == "cell") & (q.x < 2.0)]              # 92b 와 같이 x < 2 (전개 밖의 큰 x 제외)
                resid = qc.auc_test.to_numpy() - (cc0[0] + cc0[1] * qc.x.to_numpy())
                check[f"d1_{room}"] = (len(qq), cc0[0], cc0[1], r2_, Fd_, pd__, resid.mean(), len(qc))
                extra = f"{f4(resid.mean())} | {len(qc)} (points with $x<2$)"
            else:
                extra = "— | —"
            coefs = ", ".join(f"{D1_REF[c]} {f4(v)}" for c, v in zip(cats, cc1[2:]))
            L2.append(f"| {rname} | {fit_name} | {Fd_:.2f}, {fp(pd__)} | {coefs} | {extra} |")
    L += L2
    # (e) classroom PS − interp vs kappa dx
    q = Q[(Q.space == "classroom")]
    kap = check["d1_classroom"][2]
    L += ["", "| (e) Classroom condition | Seed | $x$ interpolation | $x$ PS | $\\Delta x$ = $x$(PS) − $x$(interpolation) | PS − interpolation observed | Predicted $\\kappa\\,\\Delta x$ |", "|" + "---|" * 7]
    e_rows = []
    for (lvl, seed), sub in q.groupby(["level", "cond_seed"], sort=False):
        s = sub.set_index("basis")
        dx = s.loc["PS", "x"] - s.loc["cellinterp", "x"]
        obs = s.loc["PS", "auc_test"] - s.loc["cellinterp", "auc_test"]
        e_rows.append((lvl, seed, s.loc["cellinterp", "x"], s.loc["PS", "x"], dx, obs, kap * dx))
    order = ["complete"] + [v for k, v in D1_LEVEL.items() if k != "기준"]
    e_rows.sort(key=lambda r: (order.index(r[0]), r[1]))
    for lvl, seed, xi, xp, dx, obs, prd in e_rows:
        L.append(f"| {lvl} | {'—' if seed < 0 else seed} | {f3(xi)} | {f3(xp)} | {f3(dx, True)} | {f4(obs)} | {f4(prd)} |")
    r07 = [(r[5], r[6]) for r in e_rows if r[0] == "i.i.d. $r=0.7$"]
    check["r07"] = r07
    L += ["", "Outcome. (a) In every cell the references rank in reverse order of $V_e$ (Spearman median +1.00, minimum +0.94). (b) One line in $V_e$ "
          f"explains the pooled points ($R^{{2}}$ {r2v:.3f}); the reference dummies add nothing (F {Fd:.2f}, p {pd_:.2f}), and at equal $V_e$ the share of "
          f"the error that sits in empty cells adds nothing (p {pe:.2f}); a $V_e^{{2}}$ term is detectable (F {Fq:.1f}), as the second-order expansion "
          "allows for $V_e$ approaching 0.5. (c) PS − CI equals $\\kappa\\,\\Delta V_e$ within ±0.01 in every cell with $c_{\\mathrm{eff}}\\ge3$; in the "
          "cells with fewer effective windows the observed gap is smaller than predicted, as the expansion no longer applies. (d) On D1 the fit with all "
          "four references has a significant reference effect in every room, and the fit with the class only has none in the classroom (F 0.12, p 0.89), "
          "where the unit-fill points lie 0.037 below the class line; in the hallway and the laboratory the class fit still shows a PS dummy of +0.002, "
          "at the level of the residual SD, and the AUC-PR is near saturation ($\\Phi_0$ 0.96–0.97). (e) In the classroom the predicted and observed PS − "
          "interpolation agree to within 0.003 at $r=0.7$ (" + "; ".join(f"{f3(o, True)} against {f3(p, True)}" for o, p in r07)
          + ") and within the injection-seed spread elsewhere."]
    return L


def main():
    check = {}
    L = ["## S8. Analyses Added for the Resubmission", "",
         "The tables below were added for the resubmission after the editorial decision of 12 November 2026. They are generated from the primary outputs by "
         "`analysis/95_supp_resubmit.py`; no new measurement and no new use of the test sets was involved. Table S52 reports the registered simulation "
         "R3-1 (Section 4.8, Table 15) and Table S53 the retrospective checks of Theorem 1 on the saved outputs (Section 3.6, Table 1). The registration "
         "record of R3-1 is in Section S5.", ""]
    L += table_s52(check)
    L += table_s53(check)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("written", OUT, len(L), "lines")
    # 점검: 91·92b·94 출력과 같은가
    exp = dict(A_ok=(6, 3, 6, 6), gstar=2, C1=(16, 20))
    for k, v in exp.items():
        assert check[k] == v, (k, check[k], v)
    assert abs(check["gap0"] - 0.034) < 0.0006, check["gap0"]
    assert abs(check["phi0"] - 0.4102) < 5e-5 and abs(check["kappa"] + 0.2784) < 5e-5, (check["phi0"], check["kappa"])
    assert abs(check["r2"] - 0.973) < 5e-4 and abs(check["Fd"] - 0.56) < 0.005 and abs(check["pe"] - 0.125) < 5e-4, (check["r2"], check["Fd"], check["pe"])
    assert (check["rho_med"], round(check["rho_min"], 2), check["rho_n"]) == (1.0, 0.94, 35), (check["rho_med"], check["rho_min"], check["rho_n"])
    cl = check["d1_classroom"]
    assert cl[0] == 66 and abs(cl[1] - 0.878) < 5e-4 and abs(cl[2] + 0.143) < 5e-4 and abs(cl[4] - 0.12) < 0.005 and abs(cl[6] + 0.0370) < 5e-5, cl
    print("checks against 91/92b/94 outputs: OK;", {k: check[k] for k in ("A_ok", "gstar", "C1", "c_within", "npts", "ntot", "nemp")})
    print("r=0.7 classroom (observed, predicted):", [(round(o, 3), round(p, 3)) for o, p in check["r07"]])


if __name__ == "__main__":
    main()
