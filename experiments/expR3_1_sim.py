"""실험 R3-1: 정리 1 의 등록 예측 세 묶음 (doc/실험R3-1_계획.md, 사전 등록 2026-11-13, 지문 doc/R3_등록_지문.md).

A. 새 이상 칸 6개 — N 5, MCAR r 0.5, L1, 가우스, spike/shift/drift × 크기 1/3 (κ 없는 비율 예측).
B. 끝 간격 END-g 5개 — g 0–4, N 5, 다른 손실 없음 (H2 실패; 부호가 바뀌는 g*).
C. 잡음 척도 4개 — s ∈ {0.5, 2} × {(N 5, r 0), (N 10, MCAR 0.5)}, 이상 크기 2s (척도 불변).
실험 R1-5 의 코드(expR1_5_sim.py)를 그대로 쓰고, 잡음 척도("G@s")와 END 마스크만 덧붙인다. 참조 10종·점수·보정 경로 동일.
사용법: python experiments/expR3_1_sim.py [--reps 100] [--workers 8] [--quick]
출력: results/expR3/sim/metrics.csv, design.json, run_info.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import expR1_5_sim as m  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/expR3/sim"

# ---- R1-5 코드에 덧붙이는 두 가지 (모듈 최상위: spawn 된 작업 프로세스에서도 적용된다) ----
_noise0 = m.noise


def noise(rng, kind, shape):
    """"G@0.5" 처럼 잡음 종류 뒤에 척도를 붙이면 그 배수로 키운다."""
    if "@" in kind:
        base, s = kind.split("@")
        return _noise0(rng, base, shape) * float(s)
    return _noise0(rng, kind, shape)


_mask0 = m.train_mask


def train_mask(rng, cfg, Ytr):
    """END: 마지막 g (= cfg["r"]) 시점을 모든 학습 창·행에서 비운다. 나머지는 R1-5 와 같다."""
    if cfg["miss"] == "END":
        g = int(round(cfg["r"]))
        O = np.ones(Ytr.shape, bool)
        if g:
            O[:, :, m.T - g:] = False
        return O
    return _mask0(rng, cfg, Ytr)


m.noise = noise
m.train_mask = train_mask


def design():
    cells = []

    def add(tag, **kw):
        c = dict(m.BASE)
        c.update(kw)
        cells.append({"id": 100 + len(cells), "key": json.dumps(c, sort_keys=True), "cfg": c, "tags": [tag]})

    for atype in ("spike", "shift", "drift"):
        for mag in (1.0, 3.0):
            add("A", N=5, miss="MCAR", r=0.5, atype=atype, mag=mag)
    for g in range(0, 5):
        add("B", N=5, miss="END" if g else "MCAR", r=float(g))
    for N, miss, r in ((5, "MCAR", 0.0), (10, "MCAR", 0.5)):
        for s in (0.5, 2.0):
            add("C", N=N, miss=miss, r=r, noise=f"G@{s}", mag=2.0 * s)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--quick", action="store_true", help="칸마다 반복 2 (점검용, 결과 파일은 quick_ 접두)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cells = design()
    pre = "quick_" if args.quick else ""
    (OUT / f"{pre}design.json").write_text(json.dumps([{"id": c["id"], "tags": c["tags"], "cfg": c["cfg"]} for c in cells],
                                                       ensure_ascii=False, indent=1), encoding="utf-8")
    reps = 2 if args.quick else args.reps
    tasks = [(c, r) for c in cells for r in range(reps)]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(m.replicate, tasks, chunksize=4)):
            rows.extend(res)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(tasks)} 반복, {time.time() - t0:.0f} s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"{pre}metrics.csv", index=False, encoding="utf-8-sig")
    (OUT / f"{pre}run_info.json").write_text(json.dumps(dict(cells=len(cells), reps=reps, workers=args.workers,
                                                             quick=args.quick, seconds=time.time() - t0), indent=1),
                                             encoding="utf-8")
    print(f"반복 {len(tasks)}개, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
