"""Huber IRLS (실험 10-3, `SurfaceFit._fit_blocks(robust=...)`) 의 합성 점검. 논문 명제 5 비고의 수치 출처.
합성: 행 S=4, T=12, 윈도우 N=19 (D6 와 같음). 참 곡면 = 행별 매끄러운 곡선, 잡음 N(0, 0.1), 샘플 7 % 를 ±(1–3) 이상값으로 교체.
비교: 제곱 손실 SAP (PS) vs Huber IRLS (PSH), 차분 2차. 출력: 참 곡면 대비 RMSE, 바깥 반복 수, 평균 가중.
사용법: python analysis/61_huber_synthetic_check.py  → analysis/out/61_huber_synthetic_check.txt
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model.pspline import SurfaceFit, bspline_basis, diff_matrix, kron_penalties  # noqa: E402

S, T, N, K = 4, 12, 19, 12
lines = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / (T - 1)
    truth = np.stack([np.sin(2 * np.pi * (t + rng.uniform(0, 1))) * rng.uniform(0.3, 1.0) for _ in range(S)])  # (S, T)
    X = truth[None] + rng.normal(0, 0.1, (N, S, T))
    out = rng.random((N, S, T)) < 0.07
    X[out] += rng.choice([-1, 1], out.sum()) * rng.uniform(1, 3, out.sum())
    W = np.ones((N, S, T))
    Bt = bspline_basis(T, K)
    pen = kron_penalties(S, K, diff_matrix(K, 2), {})
    res = {}
    for name, rob in (("PS", None), ("PSH", ("huber", 1.345, 5))):
        f = SurfaceFit(Bt, pen).fit(X, W, robust=rob)
        rmse = float(np.sqrt(np.mean((f.predict() - truth) ** 2)))
        res[name] = (rmse, f.info.get("robust_outer", 0), f.info.get("robust_weight_mean", 1.0), f.lam["t"])
    lines.append(f"seed {seed}: outliers {out.mean():.3f} | PS rmse {res['PS'][0]:.4f} (lam {res['PS'][3]:.3g}) | "
                 f"PSH rmse {res['PSH'][0]:.4f} (lam {res['PSH'][3]:.3g}, outer {res['PSH'][1]}, mean weight {res['PSH'][2]:.3f})")
ps = np.array([float(l.split('PS rmse ')[1].split(' ')[0]) for l in lines])
psh = np.array([float(l.split('PSH rmse ')[1].split(' ')[0]) for l in lines])
lines.append(f"mean over 5 seeds: PS rmse {ps.mean():.4f}, PSH rmse {psh.mean():.4f}, ratio {ps.mean() / psh.mean():.1f}")
txt = "\n".join(lines)
print(txt)
(ROOT / "analysis" / "out" / "61_huber_synthetic_check.txt").write_text(txt + "\n", encoding="utf-8")
