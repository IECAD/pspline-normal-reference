"""미학습 위치의 기준 곡면 보간 (BRIEF 2.7, doc/실험5_계획.md 5-1).

원천: 위치 xy (n, 2), 값 F (n, T) = 행 하나의 정상 곡선 (D2 처럼 위치마다 값이 하나면 T = 1).
  global  : 원천 평균 (위치 정보 없음, 하한)
  nearest : 가장 가까운 원천 1개
  idw     : 가장 가까운 k = 4 개, 거리^-2 가중
  kriging : ordinary functional kriging (Giraldo, Delicado, Mateu 2011). trace-variogram
            γ(h) = ½ E[mean_t (χ_i(t) − χ_j(t))²] 를 거리 분위 8 구간으로 묶고, 지수형 c0 + c1(1 − exp(−h/a)) 를
            구간 크기 가중 최소제곱으로 적합. 가중치 합 = 1. kriging variance 를 함께 반환.
            원천이 4 개 미만이거나 적합이 실패하면 idw 로 대신한다 (fallback 플래그).
보간은 모두 선형이므로 계수 Θ 를 보간하는 것과 곡면 M = Θ Bᵀ 를 보간하는 것은 같다.
"""
import numpy as np
from scipy.optimize import least_squares

METHODS = ("global", "nearest", "idw", "kriging")


def _dist(a, b):
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def exp_model(h, c0, c1, a):
    return c0 + c1 * (1.0 - np.exp(-h / a))


def fit_trace_variogram(xy, F, n_bins=8):
    n = len(xy)
    if n < 4:
        return None
    i, j = np.triu_indices(n, 1)
    h = np.linalg.norm(xy[i] - xy[j], axis=1)
    v = 0.5 * np.nanmean((F[i] - F[j]) ** 2, axis=1)
    ok = np.isfinite(v) & (h > 0)
    h, v = h[ok], v[ok]
    if len(h) < n_bins:
        return None
    edges = np.unique(np.quantile(h, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return None
    idx = np.clip(np.searchsorted(edges, h, side="right") - 1, 0, len(edges) - 2)
    hb, vb, nb = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            hb.append(h[m].mean())
            vb.append(v[m].mean())
            nb.append(m.sum())
    hb, vb, nb = np.array(hb), np.array(vb), np.array(nb, float)
    if len(hb) < 3 or not np.all(np.isfinite(vb)):
        return None
    vmax = max(float(vb.max()), 1e-12)
    lo = [0.0, 0.0, max(float(hb.min()) * 0.1, 1e-3)]
    hi = [2 * vmax, 4 * vmax, 10 * float(hb.max())]
    x0 = np.clip([float(vb.min()) * 0.5, max(float(vb.max() - vb.min()), 1e-9), float(hb.mean())], lo,
                 np.array(hi) - 1e-12)
    try:
        res = least_squares(lambda p: np.sqrt(nb) * (exp_model(hb, *p) - vb), x0, bounds=(lo, hi))
    except ValueError:
        return None
    return tuple(float(x) for x in res.x) if res.success else None


def kriging(src_xy, F, tgt_xy, params):
    """반환: 예측 (m, T), kriging variance (m,)."""
    n = len(src_xy)
    G = exp_model(_dist(src_xy, src_xy), *params)
    np.fill_diagonal(G, 0.0)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n], A[:n, n], A[n, :n] = G, 1.0, 1.0
    g0 = exp_model(_dist(src_xy, tgt_xy), *params)                       # (n, m)
    sol = np.linalg.lstsq(A, np.vstack([g0, np.ones((1, g0.shape[1]))]), rcond=None)[0]
    w, mu = sol[:n], sol[n]
    return w.T @ F, np.sum(w * g0, axis=0) + mu


def idw(src_xy, F, tgt_xy, k=4, p=2):
    d = _dist(src_xy, tgt_xy)                                            # (n, m)
    out = np.empty((d.shape[1], F.shape[1]))
    for j in range(d.shape[1]):
        nn = np.argsort(d[:, j])[:k]
        if d[nn[0], j] <= 1e-12:
            out[j] = F[nn[0]]
            continue
        w = 1.0 / d[nn, j] ** p
        out[j] = (w[:, None] * F[nn]).sum(0) / w.sum()
    return out


def interpolate(method, src_xy, F, tgt_xy, vparams=None):
    """반환: 예측 (m, T), kriging variance (m,) (kriging 이 아니면 NaN), fallback 여부."""
    m = len(tgt_xy)
    nan = np.full(m, np.nan)
    if method == "global":
        return np.tile(F.mean(0), (m, 1)), nan, False
    if method == "nearest":
        return idw(src_xy, F, tgt_xy, k=1), nan, False
    if method == "idw":
        return idw(src_xy, F, tgt_xy), nan, False
    if method == "kriging":
        if vparams is None or len(src_xy) < 4:
            return idw(src_xy, F, tgt_xy), nan, True
        pred, var = kriging(src_xy, F, tgt_xy, vparams)
        return pred, var, False
    raise ValueError(method)
