"""실험 6 1부 추가 2: 단순 보간 기준선 (doc/실험6_계획.md '1부 추가 2', 사전 등록 2026-10-10).

기저: cell, cellfill (빈칸 = 노드·행 평균), cellinterp (빈칸 = 시간 방향 선형 보간), PS (K_t = T, SAP)
조건: 기준 + C (무작위 결측 r 0.3 / 0.5 / 0.7) + D (버스트 결측 (π, L) 4 수준), 조건 시드 0-2. 학습 데이터에만 적용.
평가·보정은 experiments/exp6_conditions.run 과 같다.
사용법: python experiments/exp6_imputation.py [--splits val] [--datasets d1 d2]
출력: results/exp6/imputation/metrics_<데이터>_<분할>.csv
"""
import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):   # numpy 보다 먼저
    os.environ.setdefault(_v, "1")

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from exp6_conditions import apply_cond, run  # noqa: E402
from exp6_missing_patterns import BURST, apply_pattern  # noqa: E402
from src.data.config import load_config  # noqa: E402

OUT = ROOT / "results" / "exp6" / "imputation"
BASES = {"cell": {"time_basis": "cell"}, "cellfill": {"time_basis": "cellfill"},
         "cellinterp": {"time_basis": "cellinterp"}, "PS": {"K_t": "T"}}
CONDS = [("base", 1.0, None, 12, None)] + \
        [("C", 1.0, s, 12, r) for r in (0.3, 0.5, 0.7) for s in (0, 1, 2)] + \
        [("D", 1.0, s, 12, pl) for pl in BURST for s in (0, 1, 2)]


def apply_any(Xtr, Wtr, nodes, rows, c):
    fam, f, seed, _, prm = c
    if fam == "C":
        return apply_cond(Xtr, Wtr, nodes, rows, "C", f, seed, prm)
    return apply_pattern(Xtr, Wtr, nodes, rows, c)


def level_of(c):
    fam, _, _, _, prm = c
    if fam == "base":
        return "기준"
    return f"r={prm}" if fam == "C" else f"pi={prm[0]},L={prm[1]}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--datasets", nargs="*", default=["d1", "d2"])
    args = ap.parse_args()
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        t0 = time.perf_counter()
        rows = run(ds, ecfg, dcfg, args.splits, conds=CONDS, apply=apply_any, level_of=level_of, bases=BASES)
        pd.DataFrame(rows).to_csv(OUT / f"metrics_{ds}_{'_'.join(args.splits)}.csv", index=False,
                                  encoding="utf-8-sig")
        print(f"{ds}: {len(rows)} rows, {time.perf_counter() - t0:.1f}s", flush=True)
