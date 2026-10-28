"""실험 6 1부 추가 4: 실제 패킷 손실 패턴 이식 (doc/실험6_계획.md '1부 추가 4', 사전 등록 2026-10-11).

손실 라이브러리: cache/phy_loss_blocks.npz (analysis/49_phy_loss_library.py, D5 IDLab PHY Settings).
학습 윈도우 (윈도우, 앵커) 마다 손실률 구간 안 블록 하나와 무작위 시작점을 뽑아, 연속 12 회 시도의 성공 여부를 관측 마스크로 쓴다.
수준: P1 [0.05, 0.2), P2 [0.2, 0.5), P3 [0.5, 1.0). 기저: cell, cellfill, cellinterp, PS. 평가는 exp6_conditions.run 과 같다.
사용법: python experiments/exp6_phy_masks.py [--splits val] [--datasets d1 d2]
출력: results/exp6/phy_masks/metrics_<데이터>_<분할>.csv, mask_stats_<데이터>_<분할>.csv
"""
import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):   # numpy 보다 먼저
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from exp6_conditions import run  # noqa: E402
from exp6_imputation import BASES  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.model.compose import split_row  # noqa: E402

OUT = ROOT / "results" / "exp6" / "phy_masks"
LIB = np.load(ROOT / "cache" / "phy_loss_blocks.npz")
SEQ, OFF, LOSS = LIB["seq"], LIB["offset"], LIB["loss"]
STRATA = {"P1": (0.05, 0.2), "P2": (0.2, 0.5), "P3": (0.5, 1.0)}
CONDS = [("base", 1.0, None, 12, None)] + [("P", 1.0, s, 12, k) for k in STRATA for s in (0, 1, 2)]
STATS = []


def apply_phy(Xtr, Wtr, nodes, rows, c):
    fam, _, seed, T, k = c
    if fam == "base":
        return Xtr, Wtr, nodes
    lo, hi = STRATA[k]
    pool = np.flatnonzero((LOSS >= lo) & (LOSS < hi))
    rng = np.random.default_rng(5000 + seed)
    anchors = sorted({split_row(x)[1] for x in rows})
    aix = np.array([anchors.index(split_row(x)[1]) for x in rows])
    N, A = len(Xtr), len(anchors)
    obs = np.empty((N, A, T), bool)
    for n in range(N):
        for a in range(A):
            b = pool[rng.integers(len(pool))]
            seq = SEQ[OFF[b]:OFF[b + 1]]
            st = rng.integers(0, len(seq) - T + 1)
            obs[n, a] = seq[st:st + T] == 1
    miss = ~obs
    d = np.diff(np.c_[np.zeros((N * A, 1)), miss.reshape(N * A, T).astype(int), np.zeros((N * A, 1))], axis=1)
    blen = (np.flatnonzero(d.ravel() == -1) - np.flatnonzero(d.ravel() == 1))
    STATS.append(dict(level=k, cond_seed=seed, missing_rate=float(miss.mean()),
                      mean_burst=float(blen.mean()) if len(blen) else 0.0))
    return Xtr, Wtr * obs[:, aix, :], nodes


def level_of(c):
    return "기준" if c[0] == "base" else c[4]


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
        STATS.clear()
        rows = run(ds, ecfg, dcfg, args.splits, conds=CONDS, apply=apply_phy, level_of=level_of, bases=BASES)
        suf = f"{ds}_{'_'.join(args.splits)}"
        pd.DataFrame(rows).to_csv(OUT / f"metrics_{suf}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(STATS).groupby(["level", "cond_seed"]).mean().to_csv(OUT / f"mask_stats_{suf}.csv",
                                                                         encoding="utf-8-sig")
        print(f"{ds}: {len(rows)} rows, {time.perf_counter() - t0:.1f}s", flush=True)
