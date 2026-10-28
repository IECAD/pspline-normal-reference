"""실험 R1-4 1부 요약·판정 (doc/실험R1-4_계획.md 1.4절 H-A1–H-A5).

선택된 설정(selection.json: 데이터·조건·참조·탐지기마다 검증 평균 최대)의 행만 쓴다. 'self' = 참조 자체 점수(self_<ds>_<split>.csv).
짝 = (단위, 시나리오, 주입 seed, 조건 seed). 허용오차 D1 0.0165, D2 0.0171. D1 시나리오 SC1–3, D2 SC1m–3m.
사용법: python experiments/expR1_4_factorial_summarize.py --split val | test
출력: results/expR1/factorial/summary_<split>.md, contrasts_<split>.csv, table_<split>.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "expR1" / "factorial"
TOL = {"d1": 0.0165, "d2": 0.0171}
RX = {"d1": r"^SC\d$", "d2": r"^SC\dm$"}
KEY = ["space", "scenario", "seed", "cond_seed"]
REFS = ["CG", "CF", "CI", "PS"]
DETS = ["OLS", "GDN", "CAE2D", "TRANAD"]
MAIN = {"d1": ["C|0.7", "D|pi=0.7,L=4", "P|P3"], "d2": ["C|0.7"]}


def load(split):
    sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    rows = []
    for p in sorted(OUT.glob(f"metrics_d*_{split}.csv")):
        if "partial" in p.name or "quick" in p.name:
            continue
        rows.append(pd.read_csv(p, low_memory=False))
    M = pd.concat(rows, ignore_index=True)
    M["ck"] = M.family.astype(str) + "|" + M.level.astype(str)
    keep = []
    for (ds, ck, ref, det), g in M.groupby(["dataset", "ck", "ref", "detector"]):
        s = sel[ds][ck][ref][det]
        cfg = json.dumps(s["cfg"], sort_keys=True)
        keep.append(g[(g.cfg == cfg) & (g.variant == s["variant"])].assign(sel_variant=s["variant"]))
    M = pd.concat(keep, ignore_index=True)
    S = pd.concat([pd.read_csv(p, low_memory=False) for p in sorted(OUT.glob(f"self_d*_{split}.csv"))], ignore_index=True)
    S["ck"] = S.family.astype(str) + "|" + S.level.astype(str)
    S = S.rename(columns={"basis": "ref"}).assign(detector="self")
    A = pd.concat([M[["dataset", "ck"] + KEY + ["ref", "detector", "auc_pr"]], S[["dataset", "ck"] + KEY + ["ref", "detector", "auc_pr"]]],
                  ignore_index=True)
    A = A[[bool(pd.Series([sc]).str.match(RX[ds]).iloc[0]) for ds, sc in zip(A.dataset, A.scenario.astype(str))]]
    return A, sel


def pair(A, ds, ck, a, b):
    """a, b = (ref, detector). 반환 평균 차이, 양 비율, n."""
    x = A[(A.dataset == ds) & (A.ck == ck)]
    va = x[(x.ref == a[0]) & (x.detector == a[1])].set_index(KEY).auc_pr
    vb = x[(x.ref == b[0]) & (x.detector == b[1])].set_index(KEY).auc_pr
    d = (va - vb).dropna()
    return float(d.mean()), float((d > 0).mean()), int(len(d))


def verdict(m, tol):
    return "이김" if m > tol else ("짐" if m < -tol else "대등")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    split = ap.parse_args().split
    A, sel = load(split)
    tab = A.groupby(["dataset", "ck", "detector", "ref"]).auc_pr.mean().unstack("ref")[REFS]
    tab.to_csv(OUT / f"table_{split}.csv", encoding="utf-8-sig")
    L = [f"# 실험 R1-4 1부 집계 ({split})", "", "선택 설정 기준. 허용오차 D1 0.0165, D2 0.0171. 'self' = 참조 자체 점수.", "",
         "## 평균 셀 AUC-PR (행: 탐지기, 열: 참조)", ""]
    for (ds, ck), t in tab.groupby(level=[0, 1]):
        L += [f"**{ds} {ck}**", "", "| 탐지기 | " + " | ".join(REFS) + " |", "|---|" + "---|" * len(REFS)]
        for det in ["self"] + DETS:
            if (ds, ck, det) in t.index:
                r = t.loc[(ds, ck, det)]
                L.append(f"| {det} | " + " | ".join(f"{r[c]:.3f}" for c in REFS) + " |")
        L.append("")
    rows = []

    def add(hid, ds, ck, a, b):
        m, fp, n = pair(A, ds, ck, a, b)
        v = verdict(m, TOL[ds])
        rows.append(dict(H=hid, dataset=ds, cond=ck, a=f"{a[1]}[{a[0]}]", b=f"{b[1]}[{b[0]}]", mean=m, frac_pos=fp, n=n, verdict=v))
        return f"{a[1]}[{a[0]}] − {b[1]}[{b[0]}]: {m:+.4f} (양 {fp:.2f}, n {n}) **{v}**"

    L += ["## H-A1 탐지기마다 d[CI] − d[CG] (예상: 이김, +0.2 이상)", ""]
    for ds in ("d1", "d2"):
        for ck in MAIN[ds]:
            for det in ["self"] + DETS:
                L.append(f"- {ds} {ck}: " + add("H-A1", ds, ck, ("CI", det), ("CG", det)))
    L += ["", "## H-A2 탐지기마다 d[PS] − d[CI] (예상: 대등 또는 작은 이김, |Δ| ≤ 0.03)", ""]
    for ds in ("d1", "d2"):
        for ck in MAIN[ds]:
            for det in ["self"] + DETS:
                L.append(f"- {ds} {ck}: " + add("H-A2", ds, ck, ("PS", det), ("CI", det)))
    L += ["", "## H-A3 PS(self) − 최선의 (탐지기, 참조) (예상: 어떤 조합도 PS(self) 를 허용오차 넘게 이기지 않음)", ""]
    for ds in ("d1", "d2"):
        for ck in MAIN[ds] + (["C|0.5"] if ds == "d1" else []):
            t = tab.loc[(ds, ck)]
            best = max(((ref, det) for det in DETS for ref in REFS), key=lambda rd: t.loc[rd[1], rd[0]])
            L.append(f"- {ds} {ck} (최선 {best[1]}[{best[0]}]): " + add("H-A3", ds, ck, ("PS", "self"), best))
            worst = [add("H-A3all", ds, ck, ("PS", "self"), (ref, det)) for det in DETS for ref in REFS]
        L.append("")
    L += ["## H-A4 결측 없음: d[PS] − d[CG] (예상: 대등)", ""]
    for ds in ("d1", "d2"):
        for det in ["self"] + DETS:
            L.append(f"- {ds}: " + add("H-A4", ds, "base|기준", ("PS", det), ("CG", det)))
    L += ["", "## H-A5 재선택 효과: d[CG](재선택) − d[CG](원고 표 4 고정 설정, 실험 6 비교 모델 실험)", ""]
    for ds in ("d1", "d2"):
        p = ROOT / f"results/exp6/comparators_missing/metrics_{ds}_{split}.csv"
        if not p.exists():
            L.append(f"- {ds}: 기존 파일 없음")
            continue
        E = pd.read_csv(p, low_memory=False)
        E["ck"] = E.family.astype(str) + "|" + E.level.astype(str)
        E = E[E.scenario.astype(str).str.match(RX[ds])]
        for ck in MAIN[ds]:
            for det in DETS:
                a = A[(A.dataset == ds) & (A.ck == ck) & (A.ref == "CG") & (A.detector == det)].set_index(KEY).auc_pr
                b = E[(E.ck == ck) & (E.model == det)].set_index(KEY).auc_pr
                d = (a - b).dropna()
                if len(d):
                    v = verdict(d.mean(), TOL[ds])
                    rows.append(dict(H="H-A5", dataset=ds, cond=ck, a=f"{det}[CG]재선택", b=f"{det}[CG]고정", mean=d.mean(),
                                     frac_pos=(d > 0).mean(), n=len(d), verdict=v))
                    L.append(f"- {ds} {ck} {det}: {d.mean():+.4f} (양 {(d > 0).mean():.2f}, n {len(d)}) **{v}**")
    L += ["", "## 선택된 입력 변형 (raw 를 고른 경우)", ""]
    for ds in sel:
        for ck in sel[ds]:
            for ref in sel[ds][ck]:
                for det, s in sel[ds][ck][ref].items():
                    if s["variant"] == "raw":
                        L.append(f"- {ds} {ck} {ref} {det}: raw {s['cfg']}")
    pd.DataFrame(rows).to_csv(OUT / f"contrasts_{split}.csv", index=False, encoding="utf-8-sig")
    (OUT / f"summary_{split}.md").write_text("\n".join(L), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
