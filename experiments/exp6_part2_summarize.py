"""실험 6 (2부) 집계와 판정 (doc/실험6_계획.md "2부" 0–1절을 그대로 구현).

G, H, J: 수준마다 PS − cell 짝지은 AUC-PR Δ (같은 공간/환경·시나리오·주입 시드·조건 시드). D1 SC1–SC3, D2 SC1m–SC3m.
  이김: 한 데이터 이상 Δ > tol, 두 데이터 모두 Δ >= −tol / 비열등: 두 데이터 모두 Δ >= −tol (H 는 D2 만)
  조건부 주장 (계열별): 기준 + 바꾼 수준에서 한 수준 이상 이김 + 모든 수준 비열등
F: 백분위별 셀 F1·Range-F1·Smearing·실현 오경보율. PS − cell 셀 F1·Smearing 판정 (허용오차 0.01). 선택 없음
I: 공통 블록에서 PS(T') − PS(12) Δ. T' 가 "이기면" 교체, 둘 다 이기면 두 데이터 평균 Δ 가 큰 쪽. 아니면 12 유지
사용법: python experiments/exp6_part2_summarize.py [--split val|test]   (판정은 검증에서만)
출력: results/exp6/part2/summary_<split>.md, table_<계열>_<split>.csv, decision.json (검증)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exp6" / "part2"
TOL = {"d1": 0.0165, "d2": 0.0171}
TOL_LOC = 0.01
VIEWS = {"d1": ("d1", r"^SC\d$"), "d2": ("d2", r"^SC\dm$"), "d2dist": ("d2", r"^SC\d$")}
KEY = ["space", "scenario", "seed", "cond_seed"]


def load(fam, split):
    parts = [pd.read_csv(p) for ds in ("d1", "d2") if (p := OUT / f"metrics_{fam}_{ds}_{split}.csv").exists()]
    M = pd.concat(parts, ignore_index=True)
    M["level"] = M["level"].astype(str)
    return M


def sel(M, v):
    ds, rx = VIEWS[v]
    return M[(M.dataset == ds) & M.scenario.astype(str).str.match(rx)]


def paired(d, a, b, met="auc_pr", key=KEY):
    x = (d[d.basis == a].set_index(key)[met] - d[d.basis == b].set_index(key)[met]).dropna()
    return x.mean(), (x > 0).mean(), len(x)


def verdict(dd):
    dd = {k: v for k, v in dd.items() if np.isfinite(v)}
    return (any(dd[k] > TOL[k] for k in dd) and all(dd[k] >= -TOL[k] for k in dd),
            all(dd[k] >= -TOL[k] for k in dd))


def level_table(M, extra=()):
    rows = []
    for (fam, level), g in M.groupby(["family", "level"], sort=False):
        r = dict(family=fam, level=level)
        for v in VIEWS:
            d = sel(g, v)
            if d.empty:
                continue
            for b in ("cell", "PS") + tuple(extra):
                r[f"{v}_{b}"] = d[d.basis == b].auc_pr.mean()
            r[f"{v}_delta"], r[f"{v}_fracpos"], r[f"{v}_n"] = paired(d, "PS", "cell")
            for b in extra:
                r[f"{v}_delta_{b}"] = paired(d, "PS", b)[0]
                for met in ("hitrate", "smearing"):
                    r[f"{v}_{met}_{b}"] = d[d.basis == b][met].mean()
            for met in ("hitrate", "smearing"):
                r[f"{v}_{met}_PS"] = d[d.basis == "PS"][met].mean()
        r["beats"], r["noninferior"] = verdict({k: r.get(f"{k}_delta", np.nan) for k in ("d1", "d2")})
        rows.append(r)
    return pd.DataFrame(rows)


def claims(R, fams):
    out = []
    for fam in fams:
        F = R[(R.family == fam) | (R.family == "base")] if "base" in set(R.family) else R[R.family == fam]
        out.append(dict(family=fam, any_beats=bool(F.beats.any()), all_noninferior=bool(F.noninferior.all()),
                        conditional_claim=bool(F.beats.any() and F.noninferior.all()),
                        winning_levels=[lv for lv, b in zip(F.level, F.beats) if b and lv != "0.0"]))
    return out


def fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    sp = args.split
    L = [f"# 실험 6 (2부) 집계 ({sp})", "", "자동 생성: `experiments/exp6_part2_summarize.py`. 계획: `doc/실험6_계획.md` 2부.", ""]
    dec = dict(rule="doc/실험6_계획.md 2부", tol=TOL)

    # ---------------- F ----------------
    F = load("F", sp)
    L += ["## F. 임계 백분위 (선택 없음, 97.5 유지)", "",
          "| 데이터 | p | 명목 오경보 | 실현 오경보 cell / PS | 셀 F1 cell / PS | Range-F1 cell / PS | Smearing cell / PS | 셀 F1 판정 | Smearing 판정 |",
          "|---|---|---|---|---|---|---|---|---|"]
    frows = []
    for v in VIEWS:
        d = sel(F, v)
        for p, g in d.groupby("level", sort=False):
            m = g.groupby("basis")[["far", "cell_f1", "range_f1", "smearing"]].mean()
            df1 = paired(g, "PS", "cell", "cell_f1", KEY)[0]
            dsm = -paired(g, "PS", "cell", "smearing", KEY)[0]
            vf = "PS 우위" if df1 > TOL_LOC else ("PS 열세" if df1 < -TOL_LOC else "대등")
            vs = "PS 우위" if dsm > TOL_LOC else ("PS 열세" if dsm < -TOL_LOC else "대등")
            frows.append(dict(view=v, p=p, **{f"{c}_{b}": m.loc[b, c] for b in m.index for c in m.columns},
                              d_cell_f1=df1, d_smear=dsm, v_cell_f1=vf, v_smear=vs))
            L.append(f"| {v} | {p} | {1 - float(p) / 100:.3f} | {m.loc['cell', 'far']:.3f} / {m.loc['PS', 'far']:.3f} | "
                     f"{m.loc['cell', 'cell_f1']:.3f} / {m.loc['PS', 'cell_f1']:.3f} | "
                     f"{m.loc['cell', 'range_f1']:.3f} / {m.loc['PS', 'range_f1']:.3f} | "
                     f"{m.loc['cell', 'smearing']:.3f} / {m.loc['PS', 'smearing']:.3f} | {vf} ({df1:+.3f}) | {vs} ({dsm:+.3f}) |")
    pd.DataFrame(frows).round(4).to_csv(OUT / f"table_F_{sp}.csv", index=False, encoding="utf-8-sig")
    L.append("")

    # ---------------- G, J, H ----------------
    for fam, fams, extra, title in (("G", ["G1", "G2"], (), "G. 잡음 (k = σ_row 배수)"),
                                    ("J", ["J1", "J2"], (), "J. 학습 오염 (J1 p_event, J2 NLOS 마스크 없음)"),
                                    ("H", ["H"], ("OLS",), "H. 앵커 수 (D2, 수준 = 앵커 수)")):
        M = load(fam, sp)
        R = level_table(M, extra)
        R.round(4).to_csv(OUT / f"table_{fam}_{sp}.csv", index=False, encoding="utf-8-sig")
        C = claims(R, fams)
        dec[fam] = dict(levels=R.round(5).to_dict("records"), families=C)
        L += [f"## {title}", ""]
        if fam != "H":
            L += ["| 계열 | 수준 | D1 cell / PS (Δ, 양수) | D2 cell / PS (Δ, 양수) | Δ D2 거리만 | 판정 |", "|---|---|---|---|---|---|"]
            for _, r in R.iterrows():
                d1 = f"{r.d1_cell:.3f} / {r.d1_PS:.3f} ({fmt(r.d1_delta)}, {r.d1_fracpos:.2f})" if "d1_cell" in r and np.isfinite(r.get("d1_cell", np.nan)) else "—"
                d2 = f"{r.d2_cell:.3f} / {r.d2_PS:.3f} ({fmt(r.d2_delta)}, {r.d2_fracpos:.2f})"
                vd = "이김" if r.beats else ("비열등" if r.noninferior else "열세")
                L.append(f"| {r.family} | {r.level} | {d1} | {d2} | {fmt(r.d2dist_delta)} | {vd} |")
        else:
            L += ["| 앵커 수 | cell / PS / OLS (AUC-PR) | Δ PS − cell (양수) | Δ PS − OLS | HitRate PS / OLS | Smearing PS / OLS | Δ 거리만 | 판정 |",
                  "|---|---|---|---|---|---|---|---|"]
            for _, r in R.sort_values("level").iterrows():
                vd = "이김" if r.beats else ("비열등" if r.noninferior else "열세")
                L.append(f"| {r.level} | {r.d2_cell:.3f} / {r.d2_PS:.3f} / {r.d2_OLS:.3f} | {fmt(r.d2_delta)} ({r.d2_fracpos:.2f}) | "
                         f"{fmt(r.d2_delta_OLS)} | {r.d2_hitrate_PS:.3f} / {r.d2_hitrate_OLS:.3f} | "
                         f"{r.d2_smearing_PS:.3f} / {r.d2_smearing_OLS:.3f} | {fmt(r.d2dist_delta)} | {vd} |")
        L += ["", "조건부 주장: " + "; ".join(f"{c['family']} = {c['conditional_claim']} (이긴 수준 {c['winning_levels']})" for c in C), ""]

    # ---------------- I ----------------
    I = load("I", sp)
    L += ["## I. 최종 T (공통 블록, PS 기준)", "",
          "| 비교 | 데이터 | PS T' / PS 12 (Δ, 양수) | cell T' / cell 12 (Δ) | HitRate PS T' / 12 | 셀 F1 PS T' / 12 |", "|---|---|---|---|---|---|"]
    irows, sel_T = [], {}
    for pair, g in I.groupby("level", sort=False):
        Tp = 6 if pair == "6v12" else 24
        dd = {}
        for v in VIEWS:
            d = sel(g, v)
            rec = dict(pair=pair, view=v)
            for b in ("PS", "cell"):
                a = d[(d.basis == b) & (d.T_model == Tp)].set_index(["space", "scenario", "seed"])
                c = d[(d.basis == b) & (d.T_model == 12)].set_index(["space", "scenario", "seed"])
                x = (a.auc_pr - c.auc_pr).dropna()
                rec.update({f"{b}_Tp": a.auc_pr.mean(), f"{b}_T12": c.auc_pr.mean(), f"{b}_delta": x.mean(),
                            f"{b}_fracpos": (x > 0).mean(), f"{b}_n": len(x),
                            f"{b}_hit_Tp": a.hitrate.mean(), f"{b}_hit_T12": c.hitrate.mean(),
                            f"{b}_f1_Tp": a.cell_f1.mean(), f"{b}_f1_T12": c.cell_f1.mean()})
            irows.append(rec)
            if v in ("d1", "d2"):
                dd[v] = rec["PS_delta"]
            L.append(f"| {Tp} 대 12 | {v} | {rec['PS_Tp']:.3f} / {rec['PS_T12']:.3f} ({fmt(rec['PS_delta'])}, {rec['PS_fracpos']:.2f}) | "
                     f"{rec['cell_Tp']:.3f} / {rec['cell_T12']:.3f} ({fmt(rec['cell_delta'])}) | "
                     f"{rec['PS_hit_Tp']:.3f} / {rec['PS_hit_T12']:.3f} | {rec['PS_f1_Tp']:.3f} / {rec['PS_f1_T12']:.3f} |")
        b, ni = verdict(dd)
        sel_T[Tp] = dict(beats=b, noninferior=ni, mean_delta=float(np.nanmean(list(dd.values()))), deltas=dd)
    pd.DataFrame(irows).round(4).to_csv(OUT / f"table_I_{sp}.csv", index=False, encoding="utf-8-sig")
    winners = {T: s for T, s in sel_T.items() if s["beats"]}
    T_final = 12 if not winners else max(winners, key=lambda T: winners[T]["mean_delta"])
    L += ["", f"판정: " + "; ".join(f"T = {T}: 이김 {s['beats']}, 비열등 {s['noninferior']}, 평균 Δ {s['mean_delta']:+.4f}"
                                     for T, s in sel_T.items()) + f" → **최종 T = {T_final}**", ""]
    dec["I"] = dict(comparisons=sel_T, T_final=T_final)

    (OUT / f"summary_{sp}.md").write_text("\n".join(L), encoding="utf-8")
    if sp == "val":
        (OUT / "decision.json").write_text(json.dumps(dec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))
