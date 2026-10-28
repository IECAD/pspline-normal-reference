"""분석 R1-3 4절: 빈 칸 예측 분산의 실데이터 계산 (doc/분석R1-3_계획.md, 사전 등록 2026-10-28 21:32). 서술용.

D1 의 실제 학습 마스크(실험 6 MCAR r = 0.5, 0.7, 조건 seed 0–2)에서, 노드·행마다 SAP 로 고른 λ 를 조건으로
예측 분산 Var(m̂_t)/σ² = [B H⁻¹ BᵀCB H⁻¹ Bᵀ]_tt, H = BᵀCB + λP + 릿지(구현과 같음)를 계산하고 빈 칸과 관측 칸으로 나눈다.
학습 데이터만 쓰며 탐지 성능은 보지 않는다.
출력: results/expR1/pred_var/pred_var.csv, summary.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from exp1_baselines import prep  # noqa: E402
from exp6_conditions import apply_cond  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.model.compose import Composer  # noqa: E402
from src.model.pspline import bspline_basis, diff_matrix  # noqa: E402
from src.model.bank import NormalSurfaceBank  # noqa: E402

OUT = ROOT / "results" / "expR1" / "pred_var"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    code = ecfg["ids"]["F01"]
    rows_out = []
    for unit, sp in ecfg["datasets"]["d1"]["spaces"].items():
        D = prep("d1", unit, sp, ecfg, dcfg, ["val"])
        tr = D["tr"]["w"]
        W0 = np.isfinite(tr["X"]).astype(float)
        for r in (0.5, 0.7):
            for cs in (0, 1, 2):
                _, Wm, _ = apply_cond(tr["X"], W0, D["tr"]["n"], D["rows"], "C", 1.0, cs, r)
                comp = Composer("d1", code.get("V_d1", code["V"]), code["S"], code["O"], code["T"]).fit(tr["X"], Wm, D["rows"], D["tr"]["n"])
                Xs, Ws = comp.transform(tr["X"], Wm, D["rows"])
                T = Xs.shape[2]
                bank = NormalSurfaceBank(T, T, {}, time_basis="pspline").fit(Xs, Ws, D["tr"]["n"])
                B = bspline_basis(T, T); Dt = diff_matrix(T, 2); Pt = Dt.T @ Dt
                for nd, fit in bank.fits.items():
                    ix = D["tr"]["n"] == nd
                    cnt = (Ws[ix] > 0).sum(0)                                   # (S, T)
                    lam = fit.lam["t"]
                    A_all = np.einsum("tk,st,tl->skl", B, cnt.astype(float), B)
                    ridge = 1e-6 * max(np.trace(A_all.sum(0)) / (T * cnt.shape[0]), 1e-12)
                    for s in range(cnt.shape[0]):
                        A = A_all[s]
                        Hi = np.linalg.inv(A + lam * Pt + ridge * np.eye(T))
                        v = np.diag(B @ Hi @ A @ Hi @ B.T)
                        for t in range(T):
                            rows_out.append(dict(unit=unit, r=r, cond_seed=cs, node=nd, row=s, t=t, count=int(cnt[s, t]),
                                                 empty=bool(cnt[s, t] == 0), lam=lam, var_over_sigma2=float(v[t]),
                                                 cellmean_var=(1.0 / cnt[s, t]) if cnt[s, t] > 0 else np.inf))
    P = pd.DataFrame(rows_out)
    P.to_csv(OUT / "pred_var.csv", index=False, encoding="utf-8-sig")
    L = ["# 빈 칸 예측 분산 (D1 실제 학습 마스크, λ 는 SAP 선택값 조건부)", "",
         "값은 σ² 단위의 예측 분산. 비교: 관측 c 회인 칸의 셀 평균 분산 = 1/c (빈 칸은 정의되지 않음 — 구현은 단위 행 평균으로 채움).", ""]
    g = P.groupby(["r", "empty"]).var_over_sigma2.describe(percentiles=[0.5, 0.9, 0.99])[["count", "mean", "50%", "90%", "99%", "max"]]
    L += ["| r | 빈 칸 | 칸 수 | 평균 | 중앙 | 90 % | 99 % | 최대 |", "|---|---|---|---|---|---|---|---|"]
    for (r, e), x in g.iterrows():
        L.append(f"| {r} | {'예' if e else '아니오'} | {int(x['count'])} | {x['mean']:.3f} | {x['50%']:.3f} | {x['90%']:.3f} | {x['99%']:.3f} | {x['max']:.3f} |")
    obs = P[~P.empty & np.isfinite(P.cellmean_var)]
    L += ["", f"관측 칸에서 셀 평균 분산 1/c 의 중앙값: r 0.5 {obs[obs.r == 0.5].cellmean_var.median():.3f}, r 0.7 {obs[obs.r == 0.7].cellmean_var.median():.3f}"]
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
