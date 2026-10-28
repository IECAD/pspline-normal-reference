"""ST-SSD형 (정적) 점수기: 사전학습 배경 M + 희소 이상 A + 잡음 (doc/실험1_ST-SSD_계획.md 1절).

원 논문(Yan, Paynabar, Shi 2018) Algorithm 1–3 과 대조함 (2026-10-12, doc/실험1_ST-SSD_계획.md "원 논문 대조"):
이상 갱신 식(APG, L = 2‖B_a‖², 연판정 γ/L)과 4.3절 정상 과정(평균 고정)이 일치한다.
다른 점: 감시 통계량(원형은 윈도우 단위 다중 γ LRT, 식 12), 평균 λ 선택(원형 GCV), 동적 평균(4.1·4.2) 미구현.
  z = (Y − M) / σ_R (행별 검증 잔차 SD, 중심 이동 없음)
  행마다 min ‖m ⊙ (z − Θ B_aᵀ)‖² + γ‖Θ‖₁   (m = 관측 셀)
    B_a = 항등 (K_a = T): 해 = sign(z) · max(|z| − γ/2, 0)
    B_a = cubic B-spline (K_a < T): FISTA (최대 max_iter 회)
  점수 = |Θ̂ B_aᵀ| (셀 단위 우도비 Â²/σ² 와 순위가 같다). 임계값 = 검증 점수 백분위.
배경 M 은 NormalSurfaceBank (노드별 또는 전역) 의 곡면을 그대로 쓴다.
"""
import time

import numpy as np

from .pspline import bspline_basis


def _soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


class StSsdScorer:
    def __init__(self, bank, gamma, Ka, T, max_iter=200, tol=1e-6, eps=1e-6):
        self.bank, self.gamma, self.T = bank, float(gamma), T
        self.Ba = None if Ka in ("I", None) or int(Ka) >= T else bspline_basis(T, int(Ka))
        self.max_iter, self.tol, self.eps = max_iter, tol, eps

    def _sparse(self, z, m):
        """z, m: (N, S, T). 반환 Â (N, S, T)."""
        if self.Ba is None:
            return _soft(z, self.gamma / 2.0)
        B = self.Ba                                                      # (T, K)
        L = 2.0 * float(np.linalg.eigvalsh(B.T @ B).max())
        th = np.zeros(z.shape[:2] + (B.shape[1],))
        y, tk = th.copy(), 1.0
        for _ in range(self.max_iter):
            grad = -2.0 * np.einsum("nst,tk->nsk", m * (z - y @ B.T), B)
            new = _soft(y - grad / L, self.gamma / L)
            tk1 = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
            y = new + (tk - 1) / tk1 * (new - th)
            done = np.max(np.abs(new - th)) < self.tol
            th, tk = new, tk1
            if done:
                break
        return th @ B.T

    def score(self, X, nodes, W=None):
        t0 = time.perf_counter()
        R = self.bank.residual(X, nodes)
        m = np.isfinite(R) if W is None else (W > 0) & np.isfinite(R)
        z = np.where(m, R / (self.sd[None, :, None] + self.eps), 0.0)
        A = np.abs(self._sparse(z, m.astype(float)))
        self.infer_ms_per_window = 1000 * (time.perf_counter() - t0) / max(len(X), 1)
        return np.where(m, A, np.nan)

    def calibrate(self, Xv, Wv, nodes_v, pct=97.5):
        R = self.bank.residual(Xv, nodes_v)
        r = np.where(Wv > 0, R, np.nan)
        sd = np.nanstd(np.moveaxis(r, 1, 0).reshape(r.shape[1], -1), 1)
        self.sd = np.where(np.isfinite(sd), sd, 0.0)
        A = self.score(Xv, nodes_v, Wv)
        self.thr = float(np.nanpercentile(A[np.isfinite(A)], pct))
        # node_model_eval 의 _info 가 읽는 속성
        self.fit_rmse, self.fit_time_per_node = self.bank.fit_rmse, self.bank.fit_time_per_node
        self.M = self.bank.M
        return self

    def lambdas(self):
        return self.bank.lambdas()
