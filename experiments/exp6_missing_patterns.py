"""실험 6 1부 추가: 결측 패턴 — D 버스트 결측 (Gilbert-Elliott), E 앵커 끊김 (doc/실험6_계획.md '1부 추가', 사전 등록).
학습 데이터에만 적용한다. 평가·보정·기저는 experiments/exp6_conditions.py 와 같다 (run 을 그대로 쓴다).
사용법: python experiments/exp6_missing_patterns.py [--splits val] [--datasets d1 d2]
출력: results/exp6/missing_patterns/metrics_<데이터>_<분할>.csv
"""
import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):   # numpy 보다 먼저 (작은 행렬은 단일 스레드가 빠름)
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from exp6_conditions import run  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.model.compose import split_row  # noqa: E402

OUT = ROOT / "results" / "exp6" / "missing_patterns"
BURST = [(0.3, 4), (0.5, 4), (0.7, 4), (0.5, 8)]
OUTAGE = [0.3, 0.5]
CONDS = [("base", 1.0, None, 12, None)] + \
        [("D", 1.0, s, 12, pl) for pl in BURST for s in (0, 1, 2)] + \
        [("E", 1.0, s, 12, q) for q in OUTAGE for s in (0, 1, 2)]


def _anchor_index(rows):
    anchors = sorted({split_row(x)[1] for x in rows})
    return anchors, np.array([anchors.index(split_row(x)[1]) for x in rows])


def apply_pattern(Xtr, Wtr, nodes, rows, c):
    fam, _, seed, _, prm = c
    if fam == "base":
        return Xtr, Wtr, nodes
    anchors, aix = _anchor_index(rows)
    N, A, T = len(Xtr), len(anchors), Xtr.shape[2]
    if fam == "D":                                                      # 2상태 마르코프 (정상 0 / 누락 1)
        pi, L = prm
        q = 1.0 / L
        p = pi * q / (1.0 - pi)
        rng = np.random.default_rng(3000 + seed)
        st = np.zeros((N, A, T), bool)
        st[:, :, 0] = rng.random((N, A)) < pi
        for t in range(1, T):
            u = rng.random((N, A))
            st[:, :, t] = np.where(st[:, :, t - 1], u >= q, u < p)
        drop = st
    elif fam == "E":                                                    # 앵커가 윈도우 전체에서 빠짐
        rng = np.random.default_rng(4000 + seed)
        drop = np.repeat((rng.random((N, A)) < prm)[:, :, None], T, axis=2)
    else:
        raise ValueError(fam)
    return Xtr, Wtr * (~drop[:, aix, :]), nodes


def level_of(c):
    fam, _, _, _, prm = c
    if fam == "base":
        return "기준"
    return f"pi={prm[0]},L={prm[1]}" if fam == "D" else f"q={prm}"


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
        rows = run(ds, ecfg, dcfg, args.splits, conds=CONDS, apply=apply_pattern, level_of=level_of)
        pd.DataFrame(rows).to_csv(OUT / f"metrics_{ds}_{'_'.join(args.splits)}.csv", index=False,
                                  encoding="utf-8-sig")
        print(f"{ds}: {len(rows)} rows, {time.perf_counter() - t0:.1f}s", flush=True)
