"""Mathematics 2차 수정 보충 자료 S7 (영문)을 주 결과 파일에서 만든다 (2026-11-06).

표 S50: 측위 (원고 4.9절, 표 15) — 실험 7(results/exp7/, extra/, d2_real/)과 실험 R2-1(results/expR2/positioning_missing/).
출력: paper/mathematics/supplementary/S7_revision2.md. 70_supp2tex.py 가 S6 뒤에 붙인다.
사용법: python analysis/90_supp_revision2.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/mathematics/supplementary/S7_revision2.md"
R21 = ROOT / "results/expR2/positioning_missing"
E7 = ROOT / "results/exp7"

COND = [("base", "기준", "complete"), ("C", "0.5", "i.i.d. $r=0.5$"), ("C", "0.7", "i.i.d. $r=0.7$"),
        ("D", "pi=0.7,L=4", "burst $\\pi=0.7,L=4$"), ("P", "P3", "real loss P3")]
DET = [("PS", "PS"), ("CG", "CG"), ("CF", "CF"), ("CI", "CI")]
EST, KNOWN = "추정 노드", "정답 노드"
FILES = [("nlos_static", "classroom, static"), ("nlos_dynamic", "classroom, dynamic"),
         ("cad_dynamic_NLOS", "laboratory, dynamic"), ("hallway_dynamic_NLOS", "hallway, dynamic")]


def read(p):
    return pd.read_csv(p, encoding="utf-8-sig")


def f3(x):
    return f"{x:.3f}"


def table_s50():
    P = read(R21 / "pooled_test.csv")
    P = P[P.subset == "all"]
    L = ["**Table S50.** Positioning (main text Section 4.9, Table 15): position RMSE (m), test split. Each anchor's range is "
         "weighted by 1 if its anchor score is at most the validation threshold θ and by (θ/A)² otherwise. Positioning "
         "parameters (anchor geometry, range biases, noise SDs) come from the complete training records; only the detector's "
         "training records change between rows. Estimated node: the training node nearest to the window median of the "
         "unweighted LS fix, refined once with the weighted fix, using each detector's own scores. Known node: the tag's "
         "training node (upper reference). References: P-spline (PS), cell mean with unit fill (CG), with node-row fill (CF) "
         "and with linear interpolation (CI). (a) D1 injected anomalies, every solver, entries SC1 / SC2 / SC3; without "
         "weighting the RMSE does not depend on the row. (b) D1, complete records: detectors without a location-specific "
         "reference (LS, estimated node for PS). (c) D1 real-NLOS recordings, LS, estimated node. (d) D2 injected "
         "anomalies, LS, known node (geometric NLOS anchors removed from positioning, steps with at least four LOS anchors). "
         "(e) D2 persistent real NLOS, LS (1,872 common steps). (f) D1 injected anomalies, LS: 90th percentile of the "
         "position error (descriptive measure of reliability), entries SC1 / SC2. (g) The same runs: RMSE at steps without "
         "injected anomalies (no false-alarm cost if not higher than without weighting). Unit-level values are in the code repository.", ""]
    # (a)
    L += ["| (a) Training records, solver | none | " + " | ".join(f"{d} est." for d, _ in DET) + " | "
          + " | ".join(f"{d} known" for d, _ in DET) + " |", "|" + "---|" * 10]
    for fam, lvl, name in COND:
        for sv in ("LS", "WLS", "Huber", "EKF"):
            d = P[(P.set == "P1") & (P.family == fam) & (P.level == lvl) & (P.solver == sv)]
            cell = lambda cfg: " / ".join(f3(d[(d.scenario == s) & (d.config == cfg)].rmse.iloc[0]) for s in ("SC1", "SC2", "SC3"))
            L.append(f"| {name}, {sv} | {cell('none')} | " + " | ".join(cell(f"{k}-soft({EST})") for k, _ in DET) + " | "
                     + " | ".join(cell(f"{k}-soft({KNOWN})") for k, _ in DET) + " |")
    # (b)
    X = read(E7 / "extra/pooled_test.csv")
    X = X[(X.set == "P1") & (X.subset == "all")]
    L += ["", "| (b) Scenario | none | PS est. | within-window robust z (no location, no training) | true-range z (oracle coordinates) | residual weighting [Chen 1999], no detection |",
          "|---|---|---|---|---|---|"]
    for s in ("SC1", "SC2", "SC3"):
        g = X[(X.scenario == s)]
        v = lambda sv, cfg: f3(g[(g.solver == sv) & (g.config == cfg)].rmse.iloc[0])
        L.append(f"| {s} | {v('LS', 'none')} | {v('LS', 'PS-soft(추정 노드)')} | {v('LS', '시간이력-soft')} | "
                 f"{v('LS', '참거리z-soft(oracle)')} | {v('Rwgh', 'none')} |")
    # (c)
    L += ["", "| (c) Recording, training records | none | " + " | ".join(f"{d} est." for d, _ in DET) + " |", "|" + "---|" * 6]
    for f, fname in FILES:
        for fam, lvl, name in COND:
            d = P[(P.set == "P2") & (P.scenario == f) & (P.family == fam) & (P.level == lvl) & (P.solver == "LS")]
            v = lambda cfg: f3(d[d.config == cfg].rmse.iloc[0])
            L.append(f"| {fname}, {name} | {v('none')} | " + " | ".join(v(f"{k}-soft({EST})") for k, _ in DET) + " |")
    # (d)
    L += ["", "| (d) Training records, scenario | none | " + " | ".join(f"{d} known" for d, _ in DET) + " |", "|" + "---|" * 6]
    for fam, lvl, name in [c for c in COND if c[0] in ("base", "C") and c[1] in ("기준", "0.7")]:
        for s in ("SC1m", "SC2m", "SC3m"):
            d = P[(P.set == "P3") & (P.family == fam) & (P.level == lvl) & (P.scenario == s) & (P.solver == "LS")]
            v = lambda cfg: f3(d[d.config == cfg].rmse.iloc[0])
            L.append(f"| {name}, {s[:3]} | {v('none')} | " + " | ".join(v(f"{k}-soft({KNOWN})") for k, _ in DET) + " |")
    # (e)
    R = read(E7 / "d2_real/pooled.csv")
    R = R[(R.subset == "공통") & (R.solver == "LS")].set_index("config").rmse
    L += ["", "| (e) Configuration | RMSE |", "|---|---|", f"| none | {f3(R['none'])} |",
          f"| PS weighting, reference trained without NLOS labels | {f3(R['PS-soft(마스크 없음)'])} |"]
    # (f) P90, (g) 정상 스텝 RMSE — 신뢰도 서술 지표 (계획서 2.4절), D1 주입, LS
    for part, metric, sub, title in (("f", "p90", "all", "90th percentile of the position error (m), all steps"),
                                     ("g", "rmse", "norm", "RMSE (m) at steps without injected anomalies")):
        L += ["", f"| ({part}) {title}; training records | none | " + " | ".join(f"{d} est." for d, _ in DET) + " | "
              + " | ".join(f"{d} known" for d, _ in DET) + " |", "|" + "---|" * 10]
        Q = read(R21 / "pooled_test.csv")
        Q = Q[(Q.set == "P1") & (Q.subset == sub) & (Q.solver == "LS")]
        for fam, lvl, name in COND:
            d = Q[(Q.family == fam) & (Q.level == lvl)]
            cell = lambda cfg: " / ".join(f3(d[(d.scenario == s) & (d.config == cfg)][metric].iloc[0]) for s in ("SC1", "SC2"))
            L.append(f"| {name} | {cell('none')} | " + " | ".join(cell(f"{k}-soft({EST})") for k, _ in DET) + " | "
                     + " | ".join(cell(f"{k}-soft({KNOWN})") for k, _ in DET) + " |")
    return L


def table_s51():
    """심사자 3 (2차): 등록했지만 본문에 싣지 않은 D3 실험(실험 6 1부 추가 3)의 요약. results/exp6/d3_real_missing/."""
    D = ROOT / "results/exp6/d3_real_missing"
    M = read(D / "metrics.csv")
    C = read(D / "coverage.csv")
    tol = 0.017
    L = ["", "**Table S51.** Registered check on real missingness with a moving tag (D3; Experiment 6, addition 3; registered on "
         "10 October 2026 and run on 11 October 2026; not in the main text). Data: IDLab (imec and Ghent University) TDoA recordings "
         "with a moving tag, part \"UWB CIR Dataset for TDoA Correction and Fingerprinting\" of the Hugging Face dataset "
         "iwine-ugent/uwb-positioning-datasets (Apache-2.0; files on the imec Nextcloud linked from that page; accessed on 8 October 2026). "
         "Recordings: Random (16 anchors, median anchor observation rate 0.19; main) and Tour (8 anchors, 0.95; control). Rows: "
         "TDoA pseudo-range and RX−FP power per anchor; a window is 12 superframes. Node: the window-mean ground-truth position "
         "rounded to a 1 m grid (main) or a 0.5 m grid, so the node is assigned with the true position. Time split 60/20/20; the two "
         "held-out parts serve in turn for calibration and evaluation (directions 1 and 2), and the directions are pooled; only "
         "evaluation windows at trained nodes are scored (coverage). Injection: the D2 range-type scenarios SC1–SC3 on the range "
         "rows, five seeds per direction (30 pairs). Tolerance 0.017 (that of D2). Mean empty-cell fraction of the cell mean at "
         "training nodes: Random 0.47 (1 m) and 0.54 (0.5 m), Tour 0.00. Registered prediction: on Random the cell mean "
         "collapses and the P-spline wins; on Tour all references tie. Differences are PS − reference (fraction of positive pairs).", "",
         "| Recording, grid | Evaluated windows (dir. 1 / 2), coverage | Positive rate | AUC-PR: cell / cell + fill / cell + interp. / PS / node mean "
         "| PS − cell | PS − cell + fill | PS − cell + interp. | PS − node mean |",
         "|---|---|---|---|---|---|---|---|"]
    name = {"cell": "cell", "cellfill": "cell + fill", "cellinterp": "cell + interp.", "const": "node mean"}
    for rec, rl in (("TDoA_Random", "Random"), ("TDoA_Tour", "Tour")):
        for grid in (1.0, 0.5):
            m = M[(M.rec == rec) & (M.grid == grid)]
            c = C[(C.rec == rec) & (C.grid == grid)].sort_values("direction")
            mean = m.groupby("basis").auc_pr.mean()
            piv = m.pivot_table(index=["direction", "scenario", "seed"], columns="basis", values="auc_pr")
            cells = []
            for b in ("cell", "cellfill", "cellinterp", "const"):
                d = piv["PS"] - piv[b]
                v = "win" if d.mean() > tol else ("loss" if d.mean() < -tol else "tie")
                cells.append(f"{d.mean():+.3f}".replace("-", "−") + f" ({(d > 0).mean():.2f}) {v}")
            cov = " / ".join(str(int(x)) for x in c.covered_windows) + f", {c.covered_windows.sum() / c.eval_windows.sum():.2f}"
            L.append(f"| {rl}, {grid:g} m | {cov} | {m.pos_rate.mean():.3f} | "
                     + " / ".join(f"{mean[b]:.3f}" for b in ("cell", "cellfill", "cellinterp", "PS", "const")) + " | " + " | ".join(cells) + " |")
    L += ["", "Outcome. The registered prediction was not supported: on Random every reference was within the tolerance of the "
          "P-spline. Detection was weak for every reference (AUC-PR 0.15–0.28 at a positive rate of about 0.05). After the run we "
          "judged the data unsuitable for the question, and this judgement was made after seeing the results. The tag moves, so a "
          "node defined by a grid cell pools positions up to about the grid size apart; the spread of the normal reference within a "
          "node is then of the order of the injected anomalies, and the effect of missing training cells cannot be separated from it. "
          "The data therefore do not provide the repeated per-location normal records that the method assumes (Section 1 of the "
          "main text). Unexpectedly, on Tour with the 0.5 m grid the P-spline was ahead of the three cell-mean variants by +0.030 "
          "(all 30 pairs positive) and equal to the node mean. Code and outputs: experiments/exp6_d3.py and "
          "results/exp6/d3_real_missing/ in the repository."]
    return L


def main():
    L = ["## S7. Analyses Added in the Second Revision", "",
         "The tables below were added in the second revision. They are generated from the primary outputs by "
         "`analysis/90_supp_revision2.py`; the registration documents are in the code repository.", ""]
    L += table_s50()
    L += table_s51()
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("written", OUT, len(L), "lines")


if __name__ == "__main__":
    main()
