"""측위 해법 (doc/실험7_계획.md 2절). 2D 위치, 앵커별 높이 차 h 와 거리 편향 b 는 알려짐.

측정식: d_a = sqrt(|p − a_xy|² + h_a²) + b_a + ε_a.
여러 측위 시점(epoch)을 한 번에 푼다.
  r: (E, A) 측정 거리 (NaN = 관측 없음), w: (E, A) 탐지 가중치 (0 = 제외)
  anc: (A, 2) 앵커 xy, h: (E, A) 또는 (A,), b: (A,), sig: (A,) 앵커별 LOS 거리 잡음 SD
"""
import numpy as np

HUBER_DELTA = 1.345
GATE = 6.63                                   # χ²₁ 99 %
MAX_STEP = 1.0                                # Gauss-Newton 한 번에 움직이는 최대 거리 (m)
MARGIN = 3.0                                  # 위치 허용 범위 = 앵커 경계 상자 ± MARGIN (m)
# 수치 안전장치 (실험 7 검증 실행에서 발산 발견 후 추가, 테스트 전, 모든 구성에 같게 적용. doc/실험7_계획.md 2절 주):
#   보정 거리 하한 0, 한 걸음 최대 1 m, 위치를 앵커 상자 ± 3 m 안으로 자름


def _box(anc):
    return anc.min(0) - MARGIN, anc.max(0) + MARGIN


def gauss_newton(r, w, anc, h, b, sig=None, huber=None, iters=50):
    """가중 Gauss-Newton. sig 를 주면 가중치에 1/σ² 을 곱함 (WLS). huber = δ 면 IRLS Huber (표준화 잔차 기준).
    쓸 수 있는 앵커(관측 있고 w > 0)가 3개 미만인 시점은 NaN."""
    E, A = r.shape
    h = np.broadcast_to(h, (E, A))
    ok = np.isfinite(r) & (w > 0)
    s = np.ones(A) if sig is None else np.asarray(sig, float)
    base = np.where(ok, w, 0.0) / (s[None, :] ** 2 if sig is not None else 1.0)
    rr = np.where(ok, np.maximum(r - b[None, :], 0.0), 0.0)
    lo, hi = _box(anc)
    P = np.tile(anc.mean(0), (E, 1))
    for _ in range(iters):
        diff = P[:, None, :] - anc[None]                         # (E, A, 2)
        d = np.sqrt((diff ** 2).sum(-1) + h ** 2)
        J = diff / d[..., None]
        res = rr - d
        Wt = base
        if huber is not None:
            u = np.abs(res) / s[None, :]
            Wt = base * np.where(u <= huber, 1.0, huber / np.maximum(u, 1e-12))
        Wt = np.where(ok, Wt, 0.0)
        H = np.einsum("ea,eai,eaj->eij", Wt, J, J) + 1e-9 * np.eye(2)
        g = np.einsum("ea,eai,ea->ei", Wt, J, res)
        dP = np.linalg.solve(H, g[..., None])[..., 0]
        step = np.linalg.norm(dP, axis=1, keepdims=True)
        dP = dP * np.minimum(1.0, MAX_STEP / np.maximum(step, 1e-12))
        P = np.clip(P + dP, lo, hi)
        if np.max(np.abs(dP)) < 1e-6:
            break
    P[ok.sum(1) < 3] = np.nan
    return P


def ekf(r, w, anc, h, b, sig, q, x0, gate=GATE):
    """적응 EKF (랜덤워크 위치, 결합 갱신). r, w: (N, T, A) 윈도우×스텝×앵커, h: (N, T, A) 또는 (A,), x0: (N, 2).
    측정 잡음 R_a = σ_a² / w. 예측 상태에서 앵커마다 정규화 혁신 제곱이 gate 를 넘으면 gate 가 되도록 R_a 를 키운 뒤,
    모든 앵커로 한 번에 갱신한다. 반환: (N, T, 2)."""
    N, T, A = r.shape
    h = np.broadcast_to(h, (N, T, A))
    r = np.maximum(r - b, 0.0) + b
    lo, hi = _box(anc)
    P = np.where(np.isfinite(x0), x0, anc.mean(0)[None])
    S = np.tile(np.eye(2), (N, 1, 1))
    Q = q ** 2 * np.eye(2)
    out = np.full((N, T, 2), np.nan)
    for t in range(T):
        S = S + Q
        ok = np.isfinite(r[:, t]) & (w[:, t] > 0)                       # (N, A)
        diff = P[:, None, :] - anc[None]
        d = np.sqrt((diff ** 2).sum(-1) + h[:, t] ** 2)
        H = np.where(ok[..., None], diff / d[..., None], 0.0)           # (N, A, 2)
        nu = np.where(ok, r[:, t] - b[None] - d, 0.0)
        HSH = np.einsum("nai,nij,naj->na", H, S, H)
        R = sig[None] ** 2 / np.maximum(w[:, t], 1e-12)
        R = np.where(ok, np.maximum(R, nu ** 2 / gate - HSH), 1.0)
        Sv = np.einsum("nai,nij,nbj->nab", H, S, H) + R[..., None] * np.eye(A)
        K = np.einsum("nij,naj->nia", S, H) @ np.linalg.inv(Sv)          # (N, 2, A)
        P = np.clip(P + np.einsum("nia,na->ni", K, nu), lo, hi)
        S = S - K @ np.einsum("nai,nij->naj", H, S)
        out[:, t] = P
    return out


def ekf_sequential(r, w, anc, h, b, sig, q, x0, gate=GATE):
    """사전 등록 원안 (앵커별 순차 갱신, 보조). 합성 점검에서 먼저 갱신한 앵커가 상태를 끌고 가
    뒤의 정상 앵커가 게이트에 걸리는 문제가 있어 주 해법은 결합 갱신(ekf)으로 바꿨다 (doc/실험7_계획.md 2절 주)."""
    N, T, A = r.shape
    h = np.broadcast_to(h, (N, T, A))
    r = np.maximum(r - b, 0.0) + b
    lo, hi = _box(anc)
    P = np.where(np.isfinite(x0), x0, anc.mean(0)[None])
    S = np.tile(np.eye(2), (N, 1, 1))
    Q = q ** 2 * np.eye(2)
    out = np.full((N, T, 2), np.nan)
    for t in range(T):
        S = S + Q
        for a in range(A):
            ok = np.isfinite(r[:, t, a]) & (w[:, t, a] > 0)
            if not ok.any():
                continue
            diff = P - anc[a]
            d = np.sqrt((diff ** 2).sum(1) + h[:, t, a] ** 2)
            H = diff / d[:, None]
            nu = np.where(ok, r[:, t, a] - b[a] - d, 0.0)
            HSH = np.einsum("ni,nij,nj->n", H, S, H)
            R = sig[a] ** 2 / np.maximum(w[:, t, a], 1e-12)
            R = np.maximum(R, nu ** 2 / gate - HSH)
            K = np.einsum("nij,nj->ni", S, H) / (HSH + R)[:, None]
            P = np.clip(np.where(ok[:, None], P + K * nu[:, None], P), lo, hi)
            S = np.where(ok[:, None, None], S - np.einsum("ni,nj->nij", K, np.einsum("nj,njk->nk", H, S)), S)
        out[:, t] = P
    return out


def solve_all(r, w, anc, h, b, sig, q):
    """해법 4종. r, w, h: (N, T, A). 반환 {해법: (N, T, 2)}."""
    N, T, A = r.shape
    hh = np.broadcast_to(h, (N, T, A)).reshape(N * T, A)
    rf, wf = r.reshape(N * T, A), w.reshape(N * T, A)
    out = {"LS": gauss_newton(rf, wf, anc, hh, b).reshape(N, T, 2),
           "WLS": gauss_newton(rf, wf, anc, hh, b, sig=sig).reshape(N, T, 2),
           "Huber": gauss_newton(rf, wf, anc, hh, b, sig=sig, huber=HUBER_DELTA).reshape(N, T, 2)}
    out["EKF"] = ekf(r, w, anc, h, b, sig, q, out["WLS"][:, 0])
    out["EKFseq"] = ekf_sequential(r, w, anc, h, b, sig, q, out["WLS"][:, 0])
    return out
