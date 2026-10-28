"""실험 4 집계 (doc/실험4_계획.md). 선택은 exp4_select.py 가 검증으로 끝낸 뒤이고, 여기서는 표만 만든다.

출력: results/exp4/table_<view>_<split>.csv  구성별 평균 지표 + C00(F01) 대비 짝지은 ΔAUC-PR (평균, se, 양수 비율)
      results/exp4/table_d1_real_test.csv    D1 실제 NLOS (A′ 라벨, 보조)
      results/exp4/check_reproduction.txt    C00 = 실험 3 F01 확인 (검증·테스트)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "exp4"
METRICS = ["auc_pr", "vus_pr", "range_f1", "hitrate", "cell_f1", "smearing"]
VIEWS = [("d1", r"^SC\d$", "d1_inject"), ("d2", r"^SC\dm$", "d2_inject_multi"), ("d2", r"^SC\d$", "d2_inject_dist")]


def load(ids):
    parts = []
    for i in ids:
        for p in (RES / i / "base").glob("metrics_d[12]*.csv"):
            parts.append(pd.read_csv(p))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    cfg = yaml.safe_load((ROOT / "configs/exp4.yaml").read_text(encoding="utf-8"))
    ids = list(cfg["variants"]) + ["M01", "M02"]
    M = load(ids)
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        for ds, rx, tag in VIEWS:
            d = M[(M.dataset == ds) & (M.split == split) & M.scenario.astype(str).str.match(rx)]
            if d.empty:
                continue
            t = d.groupby("id")[METRICS].mean()
            g = d.groupby(["id", "space", "scenario", "seed"]).auc_pr.mean()
            for i in t.index:
                x = (g.loc[i] - g.loc["C00"]).dropna()
                t.loc[i, "d_auc_vs_C00"] = x.mean()
                t.loc[i, "se"] = x.std() / np.sqrt(len(x))
                t.loc[i, "frac_pos"] = (x > 0).mean()
            t.round(4).to_csv(OUT / f"table_{tag}_{split}.csv", encoding="utf-8-sig")
    r = M[(M.dataset == "d1") & M.scenario.astype(str).str.startswith("real") &
          M.scenario.astype(str).str.endswith("lab_A2")]
    if len(r):
        r.pivot_table(index="id", columns="scenario", values="auc_pr").round(4).to_csv(
            OUT / "table_d1_real_test.csv", encoding="utf-8-sig")
    lines = []
    F = load(["F01"])
    for split in ("val", "test"):
        for ds in ("d1", "d2"):
            a = M[(M.id == "C00") & (M.dataset == ds) & (M.split == split)]
            b = F[(F.dataset == ds) & (F.split == split)]
            m = a.merge(b, on=["space", "scenario", "seed"], suffixes=("_c", "_f"))
            mx = float(np.nanmax(np.abs(m.auc_pr_c - m.auc_pr_f))) if len(m) else np.nan
            lines.append(f"C00 vs 실험 3 F01 · {ds} · {split}: 행 {len(m)}/{len(a)}, max|ΔAUC-PR| = {mx:.2e}")
    (OUT / "check_reproduction.txt").write_text("\n".join(lines), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))
    for split in ("val", "test"):
        for _, _, tag in VIEWS:
            p = OUT / f"table_{tag}_{split}.csv"
            if p.exists():
                print(f"== {tag} · {split}")
                print(pd.read_csv(p, index_col=0)[["auc_pr", "hitrate", "cell_f1", "smearing", "d_auc_vs_C00", "se",
                                                   "frac_pos"]].to_string())
    if len(r):
        print("== d1 real (A′) test")
        print(pd.read_csv(OUT / "table_d1_real_test.csv", index_col=0).to_string())
