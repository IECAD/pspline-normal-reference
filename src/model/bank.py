"""Normal Surface Bank: 노드(위치)마다 정상 곡면 Θ 를 저장하고, 추론은 닫힌 형태 잔차 1회.

이상점수 (BRIEF 2.5): R = Y − M_node, A_R(s,t) = |R − μ_R(s)| / (σ_R(s) + ε)
  μ_R, σ_R: 학습에 쓰지 않은 정상 검증 윈도우의 잔차로 센서(행)별 계산 (관측 셀만)
임계값 (BRIEF 2.6): 정상 검증 A_R 의 백분위 (기본 97.5%)

단순 기준선 사다리 (doc/결정기록_실험3이후.md 2절): 시간 기저와 표준화 통계만 바꾼다.
  time_basis = "pspline" (기본, cubic B-spline + diff_order 차 차분 penalty)
             | "const"   (노드×행 평균 = 1차 차분 + λ_t → ∞ 의 극한)
             | "cell"    (노드×행×시간 평균 = λ_t = 0 의 극한, 시간 기저 = 항등)
  calibrate(stats="val")   기본: 검증 잔차의 행별 μ_R, σ_R
           (stats="train") 고전 z-score: μ = 0, σ = 학습 잔차의 행별 SD (임계값은 같은 규칙으로 검증에서)

실험 4 구성요소 (doc/실험4_계획.md 1절)
  diff_order = 1 | 2 (기본)
  lam_mode   = "sap" (기본: 노드마다 SAP, 모든 penalty 공유) | "sap_row" (행마다 1D SAP, 센서 penalty 없을 때만)
             | 숫자 (모든 penalty 를 그 값으로 고정)
  score_set  = ("R",) 기본. "K" (A_K 곡률 변화), "C" (A_C 제어점 편차) 를 더하면 셀별 최댓값으로 결합.
    보조 점수는 테스트 윈도우를 같은 기저·그 노드의 λ_t 로 행마다 적합한다 (관측 셀만).
    θ_test = argmin ||w⊙(y − Bθ)||² + λ_t θᵀPθ + r||θ − θ_node||²  (r 은 아주 작음: 관측이 없는 계수를 θ_node 에 둔다)
"""
import time

import numpy as np

from .pspline import PenaltySpectrum, SurfaceFit, bspline_basis, diff_matrix, kron_penalties

TIME_BASES = ("pspline", "const", "cell", "cellfill", "cellinterp")
# cellfill / cellinterp (실험 6 1부 추가 2, 단순 보간 기준선): 셀 평균 + 관측 0 인 칸 채우기
#   cellfill  : 그 노드·행의 관측 셀 평균으로 채움
#   cellinterp: 같은 노드·행의 관측 칸 사이 시간 방향 선형 보간 (양 끝은 가장 가까운 값)


def _fill_empty(M, cnt, mode):
    """M (S, T) 셀 평균, cnt (S, T) 관측 수. 관측 0 인 칸을 채운 사본."""
    M = M.copy()
    t = np.arange(M.shape[1])
    for s in range(M.shape[0]):
        ok = cnt[s] > 0
        if ok.all() or not ok.any():
            continue
        if mode == "cellfill":
            M[s, ~ok] = np.average(M[s, ok], weights=cnt[s, ok])
        else:
            M[s, ~ok] = np.interp(t[~ok], t[ok], M[s, ok])
    return M


class _RowFits:
    """lam_mode='sap_row': 행마다 따로 적합한 1D P-spline 을 한 노드 곡면처럼 묶는다."""

    def __init__(self, fits):
        self.rows = fits
        self.Theta = np.vstack([f.Theta for f in fits])                     # (S, K)
        self.Bt = fits[0].Bt
        self.lam = {f"t_{s}": f.lam["t"] for s, f in enumerate(fits)}
        self.info = dict(ed_total=float(sum(f.info["ed_total"] for f in fits)))

    def predict(self):
        return self.Theta @ self.Bt.T


class NormalSurfaceBank:
    def __init__(self, T, K_t=8, sensor_penalties=None, sap=True, eps=1e-6, time_basis="pspline", diff_order=2,
                 lam_mode="sap", score_set=("R",), robust=None):
        if time_basis not in TIME_BASES:
            raise ValueError(time_basis)
        self.time_basis = time_basis
        if time_basis == "pspline":
            self.Bt, self.Dt = bspline_basis(T, K_t), diff_matrix(K_t, diff_order)
        elif time_basis == "const":
            self.Bt, self.Dt = np.ones((T, 1)), None
        else:                                              # cell, cellfill, cellinterp
            self.Bt, self.Dt = np.eye(T), None
        self.T, self.K = T, self.Bt.shape[1]
        self.Ps = sensor_penalties or {}
        self.sap, self.eps = sap, eps
        self.lam_mode = lam_mode
        self.score_set = tuple(score_set)
        self.robust = {"huber": ("huber", 1.345, 5)}.get(robust) if isinstance(robust, str) else robust   # 실험 10-3
        if lam_mode == "sap_row" and self.Ps:
            raise ValueError("sap_row 는 센서 penalty 가 없을 때(T4)만 쓴다")
        self.fits = {}

    def fit(self, X, W, nodes):
        S = X.shape[1]
        if self.Dt is not None:
            P = kron_penalties(S, self.K, self.Dt, self.Ps)
            spec = PenaltySpectrum(S, self.K, self.Dt, self.Ps)
        else:                                              # 시간 penalty 없음 (const, cell)
            P = {f"s_{k}": np.kron(np.eye(self.K), Pk) for k, Pk in self.Ps.items()}
            spec = None
        t0 = time.perf_counter()
        for nd in np.unique(nodes):
            ix = nodes == nd
            if self.lam_mode == "sap_row" and self.Dt is not None:
                P1 = kron_penalties(1, self.K, self.Dt, {})
                sp1 = PenaltySpectrum(1, self.K, self.Dt, {})
                self.fits[nd] = _RowFits([SurfaceFit(self.Bt, P1, spectrum=sp1).fit(X[ix][:, [s]], W[ix][:, [s]],
                                                                                     sap=self.sap) for s in range(S)])
            elif isinstance(self.lam_mode, (int, float)) and P:
                self.fits[nd] = SurfaceFit(self.Bt, P, spectrum=spec).fit(
                    X[ix], W[ix], lam={k: float(self.lam_mode) for k in P}, sap=False)
            else:
                self.fits[nd] = SurfaceFit(self.Bt, P, spectrum=spec).fit(X[ix], W[ix], sap=self.sap, robust=self.robust)
        if self.time_basis in ("cellfill", "cellinterp"):
            for nd in self.fits:
                cnt = (W[nodes == nd] > 0).sum(0)
                self.fits[nd].Theta = _fill_empty(self.fits[nd].Theta, cnt, self.time_basis)
        self.fit_time_per_node = (time.perf_counter() - t0) / max(len(self.fits), 1)
        self.M = {nd: f.predict() for nd, f in self.fits.items()}
        self.S = S
        return self

    def bic(self, X, W, nodes):
        """Σ_노드 [n log(RSS/n) + log(n) ED] (doc/실험4_계획.md 1절 adaptive K_t). 학습 데이터로 계산."""
        tot = 0.0
        for nd in np.unique(nodes):
            ix = nodes == nd
            w = W[ix] > 0
            n = max(float(w.sum()), 1.0)
            rss = float(np.sum(np.where(w, X[ix] - self.M[nd][None], 0.0) ** 2))
            tot += n * np.log(max(rss, 1e-12) / n) + np.log(n) * self.fits[nd].info["ed_total"]
        return tot

    def residual(self, X, nodes):
        """노드가 Bank 에 없으면 NaN (미학습 위치: kriging 은 실험 5)."""
        R = np.full_like(X, np.nan, dtype=float)
        for nd in np.unique(nodes):
            if nd in self.M:
                ix = nodes == nd
                R[ix] = X[ix] - self.M[nd][None]
        return R

    # ---------------- 보조 점수 (A_K, A_C) ----------------
    def _lam_t(self, nd):
        lam = self.fits[nd].lam
        if self.Dt is None:
            return np.zeros(self.S)
        if self.lam_mode == "sap_row":
            return np.array([lam[f"t_{s}"] for s in range(self.S)])
        return np.full(self.S, lam["t"])

    def _window_fit(self, X, W, nodes):
        """테스트 윈도우 행별 적합. 반환: Δθ (N,S,K), 지지 여부 (N,S,K), κ = Δ²_t(Bθ_test − M_node) (N,S,T)."""
        N, S, T = X.shape
        B, K = self.Bt, self.K
        Pt = self.Dt.T @ self.Dt if self.Dt is not None else np.zeros((K, K))
        r = 1e-6 * max(float(np.mean(np.diag(B.T @ B))), 1e-12)
        dth = np.full((N, S, K), np.nan)
        sup = np.zeros((N, S, K), bool)
        kap = np.full((N, S, T), np.nan)
        w_all = np.where(np.isfinite(X), W, 0.0) * (W > 0)
        y_all = np.nan_to_num(X)
        for nd in np.unique(nodes):
            if nd not in self.M:
                continue
            ix = np.flatnonzero(nodes == nd)
            th0 = self.fits[nd].Theta                                         # (S, K)
            lam = self._lam_t(nd)
            w, y = w_all[ix], y_all[ix]                                       # (n, S, T)
            G = np.einsum("tk,nst,tl->nskl", B, w, B) + lam[None, :, None, None] * Pt + r * np.eye(K)
            rhs = np.einsum("tk,nst->nsk", B, w * y) + r * th0[None]
            th = np.linalg.solve(G, rhs[..., None])[..., 0]
            dth[ix] = th - th0[None]
            sup[ix] = np.einsum("tk,nst->nsk", B, w) > 0
            d2 = np.diff(th @ B.T - self.M[nd][None], 2, axis=2)              # (n, S, T-2)
            kap[ix] = np.concatenate([d2[..., :1], d2, d2[..., -1:]], axis=2)
        return dth, sup, kap

    def _aux_maps(self, X, W, nodes):
        out = {}
        if not ({"K", "C"} & set(self.score_set)):
            return out
        dth, sup, kap = self._window_fit(X, W, nodes)
        if "C" in self.score_set:
            z = np.abs(dth - self.c_mu[None]) / (self.c_sd[None] + self.eps)
            z = np.where(sup, z, np.nan)
            Bw = self.Bt                                                      # (T, K)
            num = np.einsum("tk,nsk->nst", Bw, np.nan_to_num(z))
            den = np.einsum("tk,nsk->nst", Bw, np.isfinite(z).astype(float))
            out["C"] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        if "K" in self.score_set:
            out["K"] = np.abs(kap - self.k_mu[None, :, None]) / (self.k_sd[None, :, None] + self.eps)
        return out

    def _aux_stats(self, Xval, Wval, nodes_val):
        if not ({"K", "C"} & set(self.score_set)):
            return
        dth, sup, kap = self._window_fit(Xval, Wval, nodes_val)
        d = np.where(sup, dth, np.nan)
        self.c_mu = np.nan_to_num(np.nanmean(d, 0))
        sd = np.nanstd(d, 0)
        self.c_sd = np.where(np.isfinite(sd), sd, 0.0)
        k = np.where(Wval > 0, kap, np.nan)
        flat = np.moveaxis(k, 1, 0).reshape(k.shape[1], -1)
        self.k_mu = np.nan_to_num(np.nanmean(flat, 1))
        sd = np.nanstd(flat, 1)
        self.k_sd = np.where(np.isfinite(sd), sd, 0.0)

    # ---------------- 보정·점수 ----------------
    def calibrate(self, Xval, Wval, nodes_val, pct=97.5, stats="val", Xtr=None, Wtr=None, nodes_tr=None):
        R = self.residual(Xval, nodes_val)
        r = np.where(Wval > 0, R, np.nan)
        flat = np.moveaxis(r, 1, 0).reshape(r.shape[1], -1)
        if stats == "val":
            self.mu = np.nan_to_num(np.nanmean(flat, 1))
            sd = np.nanstd(flat, 1)
        elif stats == "train":                             # 고전 z-score: 학습 잔차 SD, 중심 0
            rt = np.where(Wtr > 0, self.residual(Xtr, nodes_tr), np.nan)
            self.mu = np.zeros(rt.shape[1])
            sd = np.nanstd(np.moveaxis(rt, 1, 0).reshape(rt.shape[1], -1), 1)
        elif stats == "val_mad":                           # 견고 표준화: 중앙값, 1.4826·MAD (실험 10-2, 두꺼운 꼬리 잡음)
            self.mu = np.nan_to_num(np.nanmedian(flat, 1))
            sd = 1.4826 * np.nanmedian(np.abs(flat - self.mu[:, None]), 1)
        else:
            raise ValueError(stats)
        self.sd = np.where(np.isfinite(sd), sd, 0.0)
        self._aux_stats(Xval, Wval, nodes_val)
        A = self.score(Xval, nodes_val, Wval)
        a = A[Wval > 0]
        self.thr = float(np.nanpercentile(a, pct))
        self.fit_rmse = float(np.sqrt(np.nanmean(r ** 2)))
        return self

    def score(self, X, nodes, W=None):
        """W: 관측 마스크 (보조 점수의 윈도우 적합에만 쓴다. 없으면 유한한 셀 전부)."""
        t0 = time.perf_counter()
        R = self.residual(X, nodes)
        A = np.abs(R - self.mu[None, :, None]) / (self.sd[None, :, None] + self.eps)
        if len(self.score_set) > 1 or self.score_set != ("R",):
            Wm = np.isfinite(X).astype(float) if W is None else W
            maps = self._aux_maps(X, Wm, nodes)
            parts = ([A] if "R" in self.score_set else []) + [maps[k] for k in ("K", "C") if k in maps]
            A = np.fmax.reduce(parts) if len(parts) > 1 else parts[0]
        self.infer_ms_per_window = 1000 * (time.perf_counter() - t0) / max(len(X), 1)
        return A

    def lambdas(self):
        return {nd: f.lam for nd, f in self.fits.items()}


def anchor_scores(A, anchor_of_row, anchors):
    """행 점수 (N,S,T) -> 앵커 점수 (N,|anchors|,T): 그 앵커 행들의 최댓값 (BRIEF 2.6)."""
    out = np.full((A.shape[0], len(anchors), A.shape[2]), np.nan)
    aor = np.array(anchor_of_row)
    for j, a in enumerate(anchors):
        ix = np.flatnonzero(aor == a)
        if len(ix):
            out[:, j] = np.nanmax(A[:, ix], axis=1)
    return out
