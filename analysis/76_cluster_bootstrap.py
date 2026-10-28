"""분석 R1-3 1절: 군집을 반영한 불확실성 (doc/분석R1-3_계획.md, 사전 등록 2026-10-28 21:32).

입력: paper/mathematics/paired_values.csv (analysis/63 이 원고 통계에 쓴 짝 값 그대로, 재실행 없음).
군집: D1 방·D2 환경(pair_key 의 space=), D6 scan(scan=). 2단계 부트스트랩 — 군집을 복원 추출하고, 뽑힌 군집 안에서 짝을 복원 추출(B = 5000).
seed = 대비 이름의 crc32 (analysis/63 과 같은 규칙). 비교용으로 기존 i.i.d. 짝 부트스트랩도 같은 seed 로 다시 낸다.
'허용오차 여유 안 동등' = 군집 95 % 구간 전체가 ±허용오차 안 (D1 0.0165, D2·D6 0.0171).
출력: results/expR1/inference/cluster_ci.csv, cluster_ci.md, unit_means.csv
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "expR1" / "inference"
B = 5000
TOL = {"d1": 0.0165, "d2": 0.0171, "d6": 0.0171}


def cluster_of(key):
    kv = dict(p.split("=", 1) for p in key.split("|"))
    return kv.get("space", kv.get("scan"))


def boot(d, cl, seed):
    rng = np.random.default_rng(seed)
    groups = [d[cl == c] for c in np.unique(cl)]
    k = len(groups)
    out_c, out_i = np.empty(B), np.empty(B)
    for b in range(B):
        pick = rng.integers(0, k, k)
        vals = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in (groups[i] for i in pick)])
        out_c[b] = vals.mean()
    rng2 = np.random.default_rng(seed)
    for b in range(B):
        out_i[b] = d[rng2.integers(0, len(d), len(d))].mean()
    return out_c, out_i, k


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(ROOT / "paper/mathematics/paired_values.csv")
    P["d"] = P.value_a - P.value_b
    P["cluster"] = P.pair_key.map(cluster_of)
    rows, units = [], []
    for (g, c, ds, cond), sub in P.groupby(["group", "contrast", "dataset", "condition"], sort=False, dropna=False):
        sub = sub.dropna(subset=["d", "cluster"])
        if len(sub) < 3:
            continue
        d = sub.d.to_numpy(); cl = sub.cluster.to_numpy()
        seed = zlib.crc32(f"{g}|{c}|{ds}|{cond}".encode("utf-8"))
        bc, bi, k = boot(d, cl, seed)
        tol = TOL.get(str(ds), np.nan)
        lo, hi = np.percentile(bc, [2.5, 97.5])
        um = sub.groupby("cluster").d.agg(["mean", "size", lambda x: (x > 0).mean()])
        um.columns = ["mean", "n", "frac_pos"]
        for cname, r in um.iterrows():
            units.append(dict(group=g, contrast=c, dataset=ds, condition=cond, cluster=cname, **r.to_dict()))
        rows.append(dict(group=g, contrast=c, dataset=ds, condition=cond, n=len(d), clusters=k, mean=d.mean(),
                         iid_lo=np.percentile(bi, 2.5), iid_hi=np.percentile(bi, 97.5), cl_lo=lo, cl_hi=hi,
                         width_ratio=(hi - lo) / max(np.percentile(bi, 97.5) - np.percentile(bi, 2.5), 1e-12),
                         units_same_sign=bool((np.sign(um["mean"]) == np.sign(d.mean())).all()),
                         equiv_within_tol=bool(np.isfinite(tol) and lo > -tol and hi < tol),
                         beyond_tol_ci=bool(np.isfinite(tol) and (lo > tol or hi < -tol))))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "cluster_ci.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(units).to_csv(OUT / "unit_means.csv", index=False, encoding="utf-8-sig")
    L = ["# 군집 부트스트랩 재분석 (분석 R1-3 1절)", "",
         "군집 = D1 방 · D2 환경 · D6 scan. 2단계 재표집 B = 5000. '구간 비' = 군집 구간 폭 / i.i.d. 구간 폭.",
         "'동등(여유 안)' = 군집 95 % 구간 전체가 ±허용오차 안. '구간이 허용오차 밖' = 구간 전체가 +tol 위 또는 −tol 아래.", ""]
    for g, sub in R.groupby("group", sort=False):
        L += [f"## {g}", "", "| 대비 | 데이터 | 조건 | n | 군집 | 평균 Δ | i.i.d. 95 % | 군집 95 % | 구간 비 | 단위 부호 일치 | 동등(여유 안) | 구간이 허용오차 밖 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            L.append(f"| {r.contrast} | {r.dataset} | {r.condition} | {r.n} | {r.clusters} | {r['mean']:+.4f} | "
                     f"[{r.iid_lo:+.4f}, {r.iid_hi:+.4f}] | [{r.cl_lo:+.4f}, {r.cl_hi:+.4f}] | {r.width_ratio:.1f} | "
                     f"{'예' if r.units_same_sign else '아니오'} | {'예' if r.equiv_within_tol else '—'} | {'예' if r.beyond_tol_ci else '—'} |")
        L.append("")
    (OUT / "cluster_ci.md").write_text("\n".join(L), encoding="utf-8")
    s = R.groupby("dataset").agg(contrasts=("mean", "size"), median_width_ratio=("width_ratio", "median"),
                                 units_same_sign=("units_same_sign", "mean"))
    sys.stdout.reconfigure(encoding="utf-8")
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
