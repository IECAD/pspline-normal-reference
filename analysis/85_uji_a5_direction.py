"""D7 25개월째 A5 세트(tst06, tst10)의 수집 방향 확인 (사후, 2026-10-29).

Readme 에는 tst06–10 의 방향이 없다. 같은 달 S3 의 tst01(학습과 같은 Up/Down)·tst05(Left/Right)와
같은 위치 순서의 6샘플 블록 평균 RSS 를 비교해, 더 가까운 쪽의 방향으로 본다(미검출 100 -> -105 dBm).
출력: results/expR1/uji/a5_direction.txt (표 S48 아래 부분의 tst06·tst10 분류 근거)
사용법: python analysis/85_uji_a5_direction.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data_ext/D7_uji/extracted/db"
OUT = ROOT / "results/expR1/uji/a5_direction.txt"
MONTH = 25


def blocks(nm):
    X = pd.read_csv(DB / f"{MONTH:02d}" / f"{nm}rss.csv", header=None).to_numpy(float)
    C = pd.read_csv(DB / f"{MONTH:02d}" / f"{nm}crd.csv", header=None).to_numpy(float)
    X = np.where(X == 100, -105.0, X)
    n = len(X) // 6
    return X[: n * 6].reshape(n, 6, -1).mean(1), C[: n * 6: 6]


def compare(a, b):
    (A, ca), (B, cb) = a, b
    n = min(len(A), len(B))
    same_pos = np.all(np.isclose(ca[:n], cb[:n]), axis=1).mean()
    keep = (A[:n] > -105).any(0) | (B[:n] > -105).any(0)
    d = np.abs(A[:n][:, keep] - B[:n][:, keep]).mean()
    r = np.mean([np.corrcoef(A[i, keep], B[i, keep])[0, 1] for i in range(n)])
    return n, same_pos, d, r


ref = {nm: blocks(nm) for nm in ["tst01", "tst05", "trn01"]}
lines = [f"month {MONTH}: per-point 6-sample block means, compared with S3 sets of the same month"]
for q in ["tst06", "tst10", "trn02"]:
    b = blocks(q)
    for nm, a in ref.items():
        n, sp, d, r = compare(b, a)
        lines.append(f"{q} vs {nm}: blocks {n}, same position {sp:.2f}, mean |dRSS| {d:.2f} dB, mean per-block corr {r:.3f}")
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
