"""정상 곡면: M = B_s Θ B_t^T, B_s = I (센서 방향 항등 기저 + 차분 penalty), B_t = cubic B-spline.

목적함수 (N 개 정상 윈도우, 관측 마스크 W):
    min_Θ  Σ_n || W_n ⊙ (Y_n − Θ B_t^T) ||_F²  +  Σ_k λ_k θ^T P_k θ  (+ ridge ε||θ||²)
θ = vec(Θ) 의 인덱스 p = k*S + s (k: 시간 기저, s: 센서 행). 모든 윈도우가 같은 Θ 를 쓰므로
충분통계(가중합)만으로 닫힌 해를 구한다. ridge 는 마스크로 완전히 빠진 행의 식별용 (값 아주 작음).

λ 추정: SAP (Rodríguez-Álvarez et al. 2015, Stat Comput 25:941; 원 매개화 형태는 2019, Stat Comput 29:483)
    ED_k = λ_k tr( P_k (Q⁺ − C⁻¹) ),   Q = Σ λ_j P_j,  C = A + Q
    σ² = RSS / (n − tr(C⁻¹ A)),         λ_k ← σ² · ED_k / (θ^T P_k θ)
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import pinvh


def bspline_basis(T: int, K: int, deg: int = 3) -> np.ndarray:
    x = np.arange(T, dtype=float)
    nseg = K - deg
    if nseg < 1:
        raise ValueError("K must be > deg")
    dx = (x[-1] - x[0]) / nseg
    knots = x[0] + dx * np.arange(-deg, nseg + deg + 1)
    B = BSpline.design_matrix(x, knots, deg).toarray()
    assert B.shape == (T, K)
    return B


def diff_matrix(n: int, order: int = 2) -> np.ndarray:
    if n <= order:
        return np.zeros((0, n))
    return np.diff(np.eye(n), order, axis=0)


def _pinv_sym(Q, rtol=1e-10):
    w, V = np.linalg.eigh(Q)
    keep = w > rtol * max(w.max(), 1e-300)
    return (V[:, keep] / w[keep]) @ V[:, keep].T


class PenaltySpectrum:
    """Kronecker 합 penalty 의 공통 고유기저 (U_t ⊗ U_s) 에서 각 penalty 의 대각.

    센서 penalty 들이 서로 교환 가능(같은 고유기저)하면 Q = Σ λ_k P_k 의 고유값이 Σ λ_k d_k 로 닫힌 형태가 되어
    SAP 의 tr(P_k Q⁺) 를 고유분해 없이 계산한다. 교환 가능하지 않으면 None (일반 경로 사용).
    """

    def __init__(self, S, K, Dt, Ps: dict, rng_seed=0):
        Pt = Dt.T @ Dt
        bt, _ = np.linalg.eigh(Pt)
        self.ok = True
        if Ps:
            mix = sum(np.random.default_rng(rng_seed + i).uniform(1, 2) * P for i, P in enumerate(Ps.values()))
            _, Us = np.linalg.eigh(mix)
            ds = {}
            for k, P in Ps.items():
                M = Us.T @ P @ Us
                if np.abs(M - np.diag(np.diag(M))).max() > 1e-8 * max(np.abs(M).max(), 1):
                    self.ok = False
                ds[f"s_{k}"] = np.kron(np.ones(K), np.diag(M))
        else:
            ds = {}
        ds["t"] = np.kron(bt, np.ones(S))
        self.d = ds

    def trace_PQplus(self, lam: dict, rtol=1e-10):
        q = sum(lam[k] * self.d[k] for k in self.d)
        keep = q > rtol * max(q.max(), 1e-300)
        return {k: float(np.sum(self.d[k][keep] / q[keep])) for k in self.d}


class SurfaceFit:
    """한 노드의 정상 곡면 적합."""

    def __init__(self, Bt, penalties: dict, ridge_rel=1e-6, spectrum: "PenaltySpectrum" = None):
        self.Bt = Bt
        self.K = Bt.shape[1]
        self.P = penalties            # {이름: (p×p)}
        self.ridge_rel = ridge_rel
        self.spec = spectrum if (spectrum is not None and spectrum.ok) else None

    def _stats(self, X, W):
        X = np.where(W > 0, np.nan_to_num(X), 0.0)
        S = X.shape[1]
        Wsum, Ysum = W.sum(0), (W * X).sum(0)                 # (S, T)
        K, Bt = self.K, self.Bt
        A = np.zeros((K * S, K * S))
        for s in range(S):
            idx = np.arange(K) * S + s
            A[np.ix_(idx, idx)] = Bt.T @ (Wsum[s][:, None] * Bt)
        b = (Bt.T @ Ysum.T).reshape(-1)
        return A, b, float((W * X * X).sum()), float(W.sum()), S

    def _fit_blocks(self, X, W, lam, sap, max_iter, tol, lam_bounds, robust=None):
        """robust = ("huber", δ, n_outer): IRLS Huber (실험 10-3). 바깥 반복마다 잔차의 행별 1.4826·MAD 로 표준화한 |r| 에
        Huber 가중 min(1, δ/|r|) 을 관측 마스크에 곱해 다시 적합한다 (λ 는 이전 반복에서 이어감). n 은 관측 수(가중치 아님)."""
        if robust is None:
            return self._fit_blocks_w(X, W, W, lam, sap, max_iter, tol, lam_bounds, n_obs=float((W > 0).sum()))
        _, delta, n_outer = robust
        Wb = (W > 0).astype(float)
        wts, lam_cur = Wb.copy(), lam
        for outer in range(n_outer):
            self._fit_blocks_w(X, Wb, wts, lam_cur, sap, max_iter, tol, lam_bounds, n_obs=float(Wb.sum()))
            lam_cur = self.lam
            r = np.where(Wb > 0, np.nan_to_num(X) - self.predict()[None], np.nan)                 # (N, S, T)
            flat = np.moveaxis(r, 1, 0).reshape(r.shape[1], -1)
            with np.errstate(all="ignore"):
                scale = 1.4826 * np.nanmedian(np.abs(flat - np.nanmedian(flat, 1)[:, None]), 1)   # (S,)
            scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.inf)                     # MAD 0 이면 가중 없음
            u = np.abs(np.nan_to_num(r)) / (delta * scale[None, :, None])
            new_w = Wb * np.minimum(1.0, 1.0 / np.maximum(u, 1e-12))
            change = float(np.abs(new_w - wts).max())
            wts = new_w
            if change < 1e-3:
                break
        self.info["robust_outer"] = outer + 1
        self.info["robust_weight_mean"] = float(wts[Wb > 0].mean())
        return self

    def _fit_blocks_w(self, X, W, Wf, lam, sap, max_iter, tol, lam_bounds, n_obs):
        """Wf: 적합 가중 (관측 마스크 × 견고 가중). n_obs: 관측 수 (σ² 분모)."""
        """시간 penalty 만 있을 때 (T4) 의 정확한 빠른 경로. θ 공간이 행별 K×K 블록으로 나뉘므로 (A, P_t ⊗ I 모두 블록 대각)
        C⁻¹, tr(C⁻¹A), tr(P_t C⁻¹), θᵀP_tθ 를 행별로 계산한다. 결과는 밀집 경로와 같다 (2026-10-12, D6 S = 40 에서 추가)."""
        X = np.where(W > 0, np.nan_to_num(X), 0.0)
        S, K, Bt = X.shape[1], self.K, self.Bt
        Wsum, Ysum = Wf.sum(0), (Wf * X).sum(0)                                 # (S, T)
        A = np.einsum("tk,st,tl->skl", Bt, Wsum, Bt)                             # (S, K, K)
        b = Ysum @ Bt                                                            # (S, K)
        yy, n = float((Wf * X * X).sum()), n_obs
        Pt = self.P["t"][np.ix_(np.arange(K) * S, np.arange(K) * S)]            # P_t 블록 (K, K)
        ridge = self.ridge_rel * max(np.trace(A.sum(0)) / (K * S), 1e-12)
        rank_t = int(np.sum(np.linalg.eigvalsh(Pt) > 1e-10 * max(np.linalg.eigvalsh(Pt).max(), 1e-300)))
        lam = dict(lam or {"t": 1.0})
        I = np.eye(K)
        hist, done = [], not sap

        def solve(l):
            C = A + l * Pt[None] + ridge * I[None]
            try:
                Cinv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                Cinv = np.stack([pinvh(c) for c in C])
            th = np.einsum("skl,sl->sk", Cinv, b)
            return Cinv, th

        for it in range(max_iter if sap else 1):
            Cinv, th = solve(lam["t"])
            rss = max(yy - 2 * float(np.sum(th * b)) + float(np.einsum("sk,skl,sl->", th, A, th)), 1e-12)
            ed_tot = float(np.einsum("skl,skl->", Cinv, A))
            sigma2 = rss / max(n - ed_tot, 1.0)
            if not sap:
                break
            trQp = S * rank_t / lam["t"]                                          # tr(P_t Q⁺), Q = λ P_t ⊗ I
            ed_k = max(lam["t"] * (trQp - float(np.einsum("kl,skl->", Pt, Cinv))), 1e-8)
            quad = max(float(np.einsum("sk,kl,sl->", th, Pt, th)), 1e-12)
            new = {"t": float(np.clip(sigma2 * ed_k / quad, *lam_bounds))}
            hist.append(new)
            done = abs(np.log(new["t"]) - np.log(lam["t"])) < tol
            lam = new
            if done:
                break
        _, th = solve(lam["t"])
        self.Theta = th                                                          # (S, K)
        self.theta = th.T.reshape(-1)                                            # p = k*S + s 순서
        self.lam = lam
        self.info = dict(iters=len(hist), converged=bool(done), sigma2=sigma2, ed_total=ed_tot, n_obs=n)
        return self

    def fit(self, X, W, lam=None, sap=True, max_iter=100, tol=1e-4, lam_bounds=(1e-8, 1e8), robust=None):
        if list(self.P) == ["t"]:
            return self._fit_blocks(X, W, lam, sap, max_iter, tol, lam_bounds, robust=robust)
        if robust is not None:
            raise NotImplementedError("견고 적합은 시간 penalty 만 있는 경우(T4)에만 구현")
        A, b, yy, n, S = self._stats(X, W)
        p = A.shape[0]
        ridge = self.ridge_rel * max(np.trace(A) / p, 1e-12) * np.eye(p)
        names = list(self.P)
        lam = dict(lam or {k: 1.0 for k in names})
        hist, done = [], not (sap and names)
        for it in range(max_iter if (sap and names) else 1):
            Q = sum(lam[k] * self.P[k] for k in names) if names else np.zeros_like(A)
            C = A + Q + ridge
            try:
                Cinv = np.linalg.inv(C)
            except np.linalg.LinAlgError:                      # 마스크로 빈 행 + 아주 작은 λ -> 수치적 특이
                Cinv = pinvh(C)
            theta = Cinv @ b
            rss = max(yy - 2 * theta @ b + theta @ A @ theta, 1e-12)
            ed_tot = float(np.sum(Cinv * A))                   # tr(C⁻¹A), 둘 다 대칭
            sigma2 = rss / max(n - ed_tot, 1.0)
            if not (sap and names):
                break
            if self.spec is not None:
                trQp = self.spec.trace_PQplus(lam)
            else:
                Qp = _pinv_sym(Q)
                trQp = {k: float(np.sum(self.P[k] * Qp)) for k in names}
            new = {}
            for k in names:
                ed_k = max(lam[k] * (trQp[k] - float(np.sum(self.P[k] * Cinv))), 1e-8)
                quad = max(float(theta @ self.P[k] @ theta), 1e-12)
                new[k] = float(np.clip(sigma2 * ed_k / quad, *lam_bounds))
            hist.append(new)
            done = all(abs(np.log(new[k]) - np.log(lam[k])) < tol for k in names)
            lam = new
            if done:
                break
        Q = sum(lam[k] * self.P[k] for k in names) if names else np.zeros_like(A)
        try:
            theta = np.linalg.solve(A + Q + ridge, b)
        except np.linalg.LinAlgError:
            theta = pinvh(A + Q + ridge) @ b
        self.theta = theta
        self.Theta = theta.reshape(self.K, S).T          # (S, K)
        self.lam = lam
        # converged: max|Δlog λ| < tol 로 멈췄는지 (2026-10-12 문헌 대조: REML 최적이 경계 λ→∞ 이면 Schall 형 반복이
        # 상한까지 갈 수 있다. 그때 λ 는 "사실상 ∞" 로 읽는다). sigma2·ed_total 은 마지막 갱신 직전 λ 기준
        self.info = dict(iters=len(hist), converged=bool(done), sigma2=sigma2, ed_total=ed_tot, n_obs=n)
        return self

    def predict(self):
        return self.Theta @ self.Bt.T                    # (S, T)


def kron_penalties(S, K, Dt, Ps: dict) -> dict:
    """센서 penalty (S×S, 이름별) 와 시간 penalty 를 θ 공간 (K*S) 으로."""
    out = {f"s_{k}": np.kron(np.eye(K), P) for k, P in Ps.items()}
    out["t"] = np.kron(Dt.T @ Dt, np.eye(S))
    return out
