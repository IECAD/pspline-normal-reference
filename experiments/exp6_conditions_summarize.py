"""실험 6 (1부) 집계와 판정 (doc/실험6_계획.md 2절을 그대로 구현).

수준마다 PS − cell 짝지은 AUC-PR 차이 Δ (같은 공간/환경·시나리오·주입 시드·조건 시드). D1 주입 SC1–SC3, D2 다지표 SC1m–SC3m.
  이김: 한 데이터 이상 Δ > tol, 두 데이터 모두 Δ >= −tol / 비열등: 두 데이터 모두 Δ >= −tol / 열세: 그 밖
  조건부 주장 (계열별): 한 수준 이상 이김 + 모든 수준 비열등
  기본안 교체: 모든 계열·수준 비열등 + 어디서든 이김 -> PS (K_t = T, SAP)
사용법: python experiments/exp6_conditions_summarize.py [--split val|test]   (판정은 검증에서만)
출력: results/exp6/conditions/table_<split>.csv, decision.json (검증)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exp6" / "conditions"
TOL = {"d1": 0.0165, "d2": 0.0171}
VIEWS = {"d1": ("d1", r"^SC\d$"), "d2": ("d2", r"^SC\dm$"), "d2dist": ("d2", r"^SC\d$")}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--dir", default=str(OUT), help="결과 폴더 (1부 추가: results/exp6/missing_patterns)")
    ap.add_argument("--families", nargs="*", default=["A", "B", "C"])
    ap.add_argument("--ref", default="cell", help="PS 와 비교할 기저 (1부 추가 2: cellfill, cellinterp)")
    args = ap.parse_args()
    OUT = Path(args.dir)
    M = pd.concat([pd.read_csv(OUT / f"metrics_{ds}_{args.split}.csv") for ds in ("d1", "d2")], ignore_index=True)
    M["level"] = M["level"].astype(str)
    key = ["space", "scenario", "seed", "cond_seed"]
    rows = []
    for (fam, level), g in M.groupby(["family", "level"]):
        r = dict(family=fam, level=level)
        for v, (ds, rx) in VIEWS.items():
            d = g[(g.dataset == ds) & g.scenario.astype(str).str.match(rx)]
            for b in ("cell", "PS", "PSA"):
                r[f"{v}_{b}"] = d[d.basis == b].auc_pr.mean()
            r[f"{v}_ref"] = d[d.basis == args.ref].auc_pr.mean()
            x = (d[d.basis == "PS"].set_index(key).auc_pr - d[d.basis == args.ref].set_index(key).auc_pr).dropna()
            r[f"{v}_delta"], r[f"{v}_fracpos"], r[f"{v}_n"] = x.mean(), (x > 0).mean(), len(x)
            xa = (d[d.basis == "PSA"].set_index(key).auc_pr - d[d.basis == "cell"].set_index(key).auc_pr).dropna()
            r[f"{v}_delta_PSA"] = xa.mean()
            r[f"{v}_K_PSA"] = ",".join(str(int(k)) for k in sorted(d[d.basis == "PSA"].K_used.unique()))
        dd = {ds: r[f"{ds}_delta"] for ds in ("d1", "d2")}
        r["beats"] = any(dd[k] > TOL[k] for k in dd) and all(dd[k] >= -TOL[k] for k in dd)
        r["noninferior"] = all(dd[k] >= -TOL[k] for k in dd)
        rows.append(r)
    R = pd.DataFrame(rows)
    base = R[R.family == "base"]
    fam_rows = []
    for fam in args.families:
        F = pd.concat([base, R[R.family == fam]])                       # 계열 = 기준 수준 + 바꾼 수준
        fam_rows.append(dict(family=fam, any_beats=bool(F.beats.any()), all_noninferior=bool(F.noninferior.all()),
                             conditional_claim=bool(F.beats.any() and F.noninferior.all()),
                             winning_levels=[lv for lv, b in zip(F.level, F.beats) if b]))
    FR = pd.DataFrame(fam_rows)
    switch = bool(R.noninferior.all() and R.beats.any())
    sfx = "" if args.ref == "cell" else f"_vs_{args.ref}"
    R.round(4).to_csv(OUT / f"table_{args.split}{sfx}.csv", index=False, encoding="utf-8-sig")
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    print(f"[PS − {args.ref}]")
    cols = ["family", "level", "d1_ref", "d1_PS", "d1_delta", "d1_fracpos", "d2_ref", "d2_PS", "d2_delta",
            "d2_fracpos", "d2dist_delta", "d1_delta_PSA", "d2_delta_PSA", "d1_K_PSA", "d2_K_PSA", "beats",
            "noninferior"]
    print(R[cols].round(4).to_string(index=False))
    print(FR.to_string(index=False))
    print("기본안 교체 (PS):", switch)
    if args.split == "val":
        (OUT / f"decision{sfx}.json").write_text(json.dumps(dict(
            rule="doc/실험6_계획.md 2절", tol=TOL, families=FR.to_dict("records"), switch_default_to_PS=switch,
            levels=R[["family", "level", "d1_delta", "d2_delta", "beats", "noninferior"]].round(5).to_dict("records")),
            indent=1, ensure_ascii=False, default=str), encoding="utf-8")
