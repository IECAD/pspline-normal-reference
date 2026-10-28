"""단순 기준선 사다리 집계 (doc/결정기록_실험3이후.md 2절의 사전 등록 판정 기준을 그대로 구현).

입력: results/Z*/base/metrics_*.csv, results/<compare_to>/base/metrics_*.csv (실험 3 F01)
출력: results/exp1_simple/table_<view>_<split>.csv   구성별 평균 지표
      results/exp1_simple/paired[_test].csv           F01 − 기준선, 짝지은 차이 (같은 공간·시나리오·시드)
      results/exp1_simple/summary[_test].md           판정표 (우위 / 대등 / 열세, 허용오차 = 실험 3 AUC 허용오차)
사용법: python experiments/exp1_simple_summarize.py [--splits test]
최종 모델(실험 4, results/exp4/selection.json 의 final)이 있으면 그 구성과 주 기준선의 비교도 함께 적는다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "exp1_simple"
METRICS = ["auc_pr", "vus_pr", "range_f1", "hitrate", "cell_f1", "smearing"]
VIEWS = [("d1", r"^SC\d$", "d1_inject", True), ("d2", r"^SC\dm$", "d2_inject_multi", True),
         ("d2", r"^SC\d$", "d2_inject_dist", False)]          # 마지막 값: 사전 등록 판정에 쓰는 주 결과인지
# 성분 분해 (a − b): 사전 등록 2절 표
CONTRASTS = [("ZF01", "Z1", "시간 방향 P-spline (F01 − Z1)"), ("ZF01", "Z2", "시간 방향 penalty (F01 − Z2)"),
             ("Z1", "Z0", "노드별 Bank (Z1 − Z0)"), ("ZF01", "Z0s", "P-spline 조건의 Bank (F01 − Z0s)"),
             ("Z1", "Z1t", "표준화 방식 (Z1 − Z1t)")]


def load(bcfg, splits, final_id=None):
    parts = [pd.read_csv(p) for p in RES.glob("Z*/base/metrics_*.csv") if p.stat().st_size > 5]
    srcs = [RES / bcfg["compare_to"] / "base"] + ([RES / final_id / "base"] if final_id else [])
    for d in srcs:
        for p in d.glob("metrics_d[12]*.csv"):
            parts.append(pd.read_csv(p))
    M = pd.concat(parts, ignore_index=True)
    return M[M.split.isin(splits) & ~M.scenario.astype(str).str.startswith("real")]


def paired(M, ds, scen, split, a, b, col="auc_pr"):
    d = M[(M.dataset == ds) & (M.split == split) & M.scenario.astype(str).str.match(scen)]
    key = ["space", "scenario", "seed"]
    pa = d[d.id == a].set_index(key)[col]
    pb = d[d.id == b].set_index(key)[col]
    x = (pa - pb).dropna()
    if x.empty:
        return None
    p = float(wilcoxon(x).pvalue) if (x != 0).any() else 1.0
    return dict(mean=float(x.mean()), sd=float(x.std()), n=len(x), frac_pos=float((x > 0).mean()), wilcoxon_p=p,
                by_space=x.groupby(level=0).mean().round(4).to_dict())


def verdict(m, tol):
    return "우위" if m > tol else ("열세" if m < -tol else "대등")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*")
    args = ap.parse_args()
    bcfg = yaml.safe_load((ROOT / "configs/exp1_simple.yaml").read_text(encoding="utf-8"))
    splits = args.splits or bcfg["splits"]
    selp = RES / "exp4" / "selection.json"
    final = json.loads(selp.read_text(encoding="utf-8")).get("final") if selp.exists() else None
    fid = final["id"] if final and final["id"] != "C00" else None
    M = load(bcfg, splits, fid)
    M.loc[M.id == bcfg["compare_to"], "id"] = "F01(exp3)"
    contrasts = CONTRASTS + ([(fid, "Z1", f"최종 {fid} − Z1"), (fid, "Z2", f"최종 {fid} − Z2")] if fid else [])
    OUT.mkdir(parents=True, exist_ok=True)
    name = {"val": "검증", "test": "테스트"}
    rows, lines = [], [f"# 단순 기준선 사다리 결과 ({', '.join(name.get(s, s) for s in splits)} 세트)", "",
                       "자동 생성: `experiments/exp1_simple_summarize.py`. 판정 기준: `doc/결정기록_실험3이후.md` 2절 (사전 등록).",
                       (f"최종 모델 (실험 4): {fid} = F01 + {final['spec']}" if fid else ""), ""]
    for split in splits:
        for ds, scen, tag, primary in VIEWS:
            d = M[(M.dataset == ds) & (M.split == split) & M.scenario.astype(str).str.match(scen)]
            t = d.groupby("id")[METRICS].mean().round(4)
            t.to_csv(OUT / f"table_{tag}_{split}.csv", encoding="utf-8-sig")
            tol = bcfg["tolerance"][ds]
            lines += [f"## {tag} · {split}" + ("" if primary else " (보조)"), "",
                      "| ID | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
            lines += [f"| {i} | " + " | ".join(f"{v:.3f}" for v in r) + " |" for i, r in t.iterrows()]
            lines += ["", f"짝지은 AUC-PR 차이 (허용오차 {tol}):", "",
                      "| 비교 | 평균 | SD | n | 양수 비율 | Wilcoxon p | 공간별 평균 | 판정 |", "|---|---|---|---|---|---|---|---|"]
            for a, b, lab in contrasts + [("ZF01", "F01(exp3)", "재현 확인 (ZF01 − 실험 3 F01)")]:
                r = paired(M, ds, scen, split, a, b)
                if r is None:
                    continue
                v = verdict(r["mean"], tol)
                rows.append(dict(split=split, view=tag, primary=primary, a=a, b=b, contrast=lab, tol=tol, verdict=v,
                                 **{k: r[k] for k in ("mean", "sd", "n", "frac_pos", "wilcoxon_p")},
                                 by_space=str(r["by_space"])))
                lines.append(f"| {lab} | {r['mean']:+.4f} | {r['sd']:.4f} | {r['n']} | {r['frac_pos']:.2f} | "
                             f"{r['wilcoxon_p']:.3g} | {r['by_space']} | {v} |")
            pb = bcfg["primary_baseline"][ds]
            for mid, lab in [("ZF01", "F01")] + ([(fid, f"최종 {fid}")] if fid else []):
                r = paired(M, ds, scen, split, mid, pb)
                if r and primary:
                    v = "동일 (같은 구성)" if r["sd"] == 0 and r["mean"] == 0 else verdict(r["mean"], tol)
                    lines += ["", f"**주 판정 ({ds}, 주 기준선 {pb}): {lab} − {pb} = {r['mean']:+.4f} → {v}**"]
            lines.append("")
    suf = "" if splits == ["val"] else "_" + "_".join(splits)
    pd.DataFrame(rows).to_csv(OUT / f"paired{suf}.csv", index=False, encoding="utf-8-sig")
    (OUT / f"summary{suf}.md").write_text("\n".join(lines), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))
