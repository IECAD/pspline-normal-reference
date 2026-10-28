"""실험 1 비교 모델 1차 (doc/실험1_비교모델_계획.md 2절). 공통 인터페이스:
  m = Model(**cfg).fit(Z)   Z: (N, S, T) 학습 입력 (결측·마스크 셀은 호출 쪽에서 채움)
  E = m.errors(Z)           (N, S, T) 셀별 |오차| (Isolation Forest 는 시점 점수를 모든 셀에 복사)
입력 Z 는 raw (F01 합성 값) 또는 nc (노드별 셀 평균을 뺀 잔차) 다 (실행기에서 만듦).
"""
import numpy as np
from sklearn.ensemble import IsolationForest


def _flat_t(Z):
    """(N, S, T) -> (N*T, S) 시점 벡터."""
    return np.moveaxis(Z, 1, 2).reshape(-1, Z.shape[1])


def _unflat_t(V, N, T):
    return np.moveaxis(V.reshape(N, T, -1), 1, 2)


class PCASPE:
    def __init__(self, var=0.9):
        self.var = var

    def fit(self, Z):
        X = _flat_t(Z)
        self.mu = X.mean(0)
        _, s, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        ev = s ** 2 / np.sum(s ** 2)
        k = int(np.searchsorted(np.cumsum(ev), self.var) + 1)
        self.k = min(k, X.shape[1] - 1)                     # 잔차가 0 이 되지 않게
        self.P = Vt[:self.k].T
        return self

    def errors(self, Z):
        X = _flat_t(Z) - self.mu
        R = X - X @ self.P @ self.P.T
        return np.abs(_unflat_t(R, Z.shape[0], Z.shape[2]))


class RPCASub(PCASPE):
    """학습 행렬에 PCP (inexact ALM, Lin et al. 2010) -> 저랭크 L 의 SVD 부분공간에 투영."""

    def __init__(self, c=1.0, var=0.9, max_iter=500, tol=1e-7):
        self.c, self.var, self.max_iter, self.tol = c, var, max_iter, tol

    def fit(self, Z):
        X = _flat_t(Z)
        n, p = X.shape
        lam = self.c / np.sqrt(max(n, p))
        norm2 = np.linalg.norm(X, 2)
        Y = X / max(norm2, np.abs(X).max() / lam)
        mu, rho = 1.25 / norm2, 1.5
        S = np.zeros_like(X)
        for _ in range(self.max_iter):
            U, s, Vt = np.linalg.svd(X - S + Y / mu, full_matrices=False)
            L = (U * np.maximum(s - 1 / mu, 0)) @ Vt
            T_ = X - L + Y / mu
            S = np.sign(T_) * np.maximum(np.abs(T_) - lam / mu, 0)
            Rr = X - L - S
            Y += mu * Rr
            mu = min(mu * rho, 1e7)
            if np.linalg.norm(Rr) / max(np.linalg.norm(X), 1e-12) < self.tol:
                break
        self.mu = L.mean(0)
        _, s, Vt = np.linalg.svd(L - self.mu, full_matrices=False)
        ev = s ** 2 / max(np.sum(s ** 2), 1e-12)
        self.k = min(int(np.searchsorted(np.cumsum(ev), self.var) + 1), p - 1)
        self.P = Vt[:self.k].T
        return self


class OLSCross:
    """행 s 를 같은 시점의 다른 행으로 ridge 회귀 (절편 포함)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, Z):
        X = _flat_t(Z)
        S = X.shape[1]
        self.coef = []
        for s in range(S):
            o = np.delete(np.arange(S), s)
            A = np.c_[np.ones(len(X)), X[:, o]]
            reg = self.alpha * np.eye(A.shape[1])
            reg[0, 0] = 0
            self.coef.append((o, np.linalg.solve(A.T @ A + reg, A.T @ X[:, s])))
        return self

    def errors(self, Z):
        X = _flat_t(Z)
        R = np.empty_like(X)
        for s, (o, b) in enumerate(self.coef):
            R[:, s] = X[:, s] - np.c_[np.ones(len(X)), X[:, o]] @ b
        return np.abs(_unflat_t(R, Z.shape[0], Z.shape[2]))


class IForest:
    def __init__(self, max_samples=256, n_estimators=100, seed=0):
        self.kw = dict(max_samples=max_samples, n_estimators=n_estimators, random_state=seed)

    def fit(self, Z):
        self.m = IsolationForest(**self.kw).fit(_flat_t(Z))
        return self

    def errors(self, Z):
        sc = -self.m.score_samples(_flat_t(Z)).reshape(Z.shape[0], Z.shape[2])
        return np.repeat(sc[:, None, :], Z.shape[1], axis=1)


# ----------------------------------------------------------------------------- 딥러닝 (PyTorch)
def _torch():
    import torch
    return torch


class _DeepAE:
    epochs, lr, batch = 200, 1e-3, 64

    def __init__(self, seed=0, **kw):
        self.seed, self.kw = seed, kw

    def _net(self, S, T):
        raise NotImplementedError

    def _recon(self, net, x):
        return net(x)

    def fit(self, Z):
        torch = _torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.dev = self.kw.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")   # 실험 8: CPU 강제 가능
        N, S, T = Z.shape
        self.net = self._net(S, T).to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        x = torch.tensor(Z, dtype=torch.float32, device=self.dev)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for ep in range(self.epochs):
            perm = torch.randperm(N, generator=g)
            for i in range(0, N, self.batch):
                xb = x[perm[i:i + self.batch].to(self.dev)]
                loss = self._loss(xb, ep)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def _loss(self, xb, ep):
        return ((self._recon(self.net, xb) - xb) ** 2).mean()

    def errors(self, Z):
        torch = _torch()
        with torch.no_grad():
            x = torch.tensor(Z, dtype=torch.float32, device=self.dev)
            return self._err(x).cpu().numpy()

    def _err(self, x):
        return (self._recon(self.net, x) - x).abs()


class LSTMAE(_DeepAE):
    def _net(self, S, T):
        torch = _torch()
        h = self.kw.get("hidden", 16)

        class Net(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = torch.nn.LSTM(S, h, batch_first=True)
                s.dec = torch.nn.LSTM(h, h, batch_first=True)
                s.out = torch.nn.Linear(h, S)

            def forward(s, x):                                   # x: (B, S, T)
                seq = x.transpose(1, 2)                          # (B, T, S)
                _, (hn, _) = s.enc(seq)
                rep = hn[-1].unsqueeze(1).repeat(1, seq.shape[1], 1)
                y, _ = s.dec(rep)
                return s.out(y).transpose(1, 2)

        return Net()


class CAE1D(_DeepAE):
    def _net(self, S, T):
        torch = _torch()
        c = self.kw.get("channels", 16)
        nn = torch.nn
        return nn.Sequential(nn.Conv1d(S, c, 3, padding=1), nn.ReLU(), nn.Conv1d(c, max(c // 2, 2), 3, padding=1),
                             nn.ReLU(), nn.Conv1d(max(c // 2, 2), c, 3, padding=1), nn.ReLU(),
                             nn.Conv1d(c, S, 3, padding=1))


class USAD(_DeepAE):
    """Audibert et al. (2020) 의 두 복호기 적대 학습. 입력은 평탄화 윈도우, 출력은 선형 (z-score 입력이라 sigmoid 대신)."""

    def _net(self, S, T):
        torch = _torch()
        nn = torch.nn
        d, z, h = S * T, self.kw.get("latent", 8), 64

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = nn.Sequential(nn.Flatten(), nn.Linear(d, h), nn.ReLU(), nn.Linear(h, z), nn.ReLU())
                s.d1 = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))
                s.d2 = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))
                s.shape = (S, T)

            def ae(s, dec, x):
                return dec(s.enc(x)).view(-1, *s.shape)

        return Net()

    def _loss(self, xb, ep):
        n = ep + 1
        net = self.net
        w1 = net.ae(net.d1, xb)
        w2 = net.ae(net.d2, xb)
        w3 = net.ae(net.d2, w1)
        l1 = (1 / n) * ((xb - w1) ** 2).mean() + (1 - 1 / n) * ((xb - w3) ** 2).mean()
        l2 = (1 / n) * ((xb - w2) ** 2).mean() - (1 - 1 / n) * ((xb - w3) ** 2).mean()
        return l1 + l2

    def _err(self, x):
        net = self.net
        w1 = net.ae(net.d1, x)
        w3 = net.ae(net.d2, w1)
        return 0.5 * (x - w1).abs() + 0.5 * (x - w3).abs()


class USADFix(USAD):
    """USAD 수정판 (doc/실험1_비교모델_계획.md "2차 수정"). 위 USAD 는 l1 + l2 를 한 번에 역전파해 적대 항이
    상쇄되는 학습 오류판이다 (재현용으로 남김). 원 논문 Alg. 1·공개 코드처럼 옵티마이저 2개를 배치마다 번갈아 갱신:
    E + D1 ← L_AE1 = (1/n)‖W − AE1‖² + (1 − 1/n)‖W − AE2(AE1)‖²,  E + D2 ← L_AE2 = (1/n)‖W − AE2‖² − (1 − 1/n)‖W − AE2(AE1)‖²."""

    def fit(self, Z):
        torch = _torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.dev = self.kw.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        N, S, T = Z.shape
        net = self.net = self._net(S, T).to(self.dev)
        opt1 = torch.optim.Adam(list(net.enc.parameters()) + list(net.d1.parameters()), lr=self.lr)
        opt2 = torch.optim.Adam(list(net.enc.parameters()) + list(net.d2.parameters()), lr=self.lr)
        x = torch.tensor(Z, dtype=torch.float32, device=self.dev)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        mse = lambda a, b: ((a - b) ** 2).mean()
        for ep in range(self.epochs):
            n = ep + 1
            perm = torch.randperm(N, generator=g)
            for i in range(0, N, self.batch):
                xb = x[perm[i:i + self.batch].to(self.dev)]
                w1 = net.ae(net.d1, xb)
                l1 = (1 / n) * mse(xb, w1) + (1 - 1 / n) * mse(xb, net.ae(net.d2, w1))
                opt1.zero_grad()
                l1.backward()
                opt1.step()
                w1 = net.ae(net.d1, xb)
                l2 = (1 / n) * mse(xb, net.ae(net.d2, xb)) - (1 - 1 / n) * mse(xb, net.ae(net.d2, w1))
                opt2.zero_grad()
                l2.backward()
                opt2.step()
        return self


# ----------------------------------------------------------------------------- 2차 (doc/실험1_비교모델_계획.md "2차" 2절)
class LapRPCA(PCASPE):
    """그래프 정규화 RPCA: X = L + S, min ‖L‖_* + λ‖S‖₁ + γ tr(L Φ Lᵀ) (ADMM, W = L 분리).
    Φ = 센서 간 |상관| 그래프의 정규화 Laplacian. 학습 L 의 SVD 부분공간(에너지 var)에 투영."""

    def __init__(self, c=1.0, gamma=0.1, var=0.9, max_iter=300, tol=1e-6, std=True, wpos=True):
        """std: 학습 평균·SD 로 열 표준화 (2차 수정. False = LAPRPCA_v0). wpos: 가중 max(ρ, 0) (False = |ρ|)."""
        self.c, self.gamma, self.var, self.max_iter, self.tol = c, gamma, var, max_iter, tol
        self.std, self.wpos = std, wpos

    def _scale(self, X):
        return (X - self.cm) / self.cs if self.std else X

    def errors(self, Z):
        if not self.std:
            return super().errors(Z)
        Zs = (Z - self.cm[None, :, None]) / self.cs[None, :, None]
        return super().errors(Zs)

    def fit(self, Z):
        X = _flat_t(Z)
        self.cm, self.cs = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1.0)
        X = self._scale(X)
        n, p = X.shape
        lam = self.c / np.sqrt(max(n, p))
        C = np.nan_to_num(np.corrcoef(X, rowvar=False))
        A = np.maximum(C, 0) if self.wpos else np.abs(C)
        np.fill_diagonal(A, 0)
        dinv = 1 / np.sqrt(np.maximum(A.sum(1), 1e-12))
        Phi = np.eye(p) - dinv[:, None] * A * dinv[None, :]
        mu, rho = 1.25 / max(np.linalg.norm(X, 2), 1e-12), 1.5
        L, S, Z1, Z2 = (np.zeros_like(X) for _ in range(4))
        for _ in range(self.max_iter):
            U, s, Vt = np.linalg.svd(L + Z2 / mu, full_matrices=False)
            W = (U * np.maximum(s - 1 / mu, 0)) @ Vt
            T_ = X - L + Z1 / mu
            S = np.sign(T_) * np.maximum(np.abs(T_) - lam / mu, 0)
            L = (mu * (X - S) + Z1 + mu * W - Z2) @ np.linalg.inv(2 * self.gamma * Phi + 2 * mu * np.eye(p))
            r1, r2 = X - L - S, L - W
            Z1 += mu * r1
            Z2 += mu * r2
            mu = min(mu * rho, 1e7)
            if (np.linalg.norm(r1) + np.linalg.norm(r2)) / max(np.linalg.norm(X), 1e-12) < self.tol:
                break
        self.mu = L.mean(0)
        _, s, Vt = np.linalg.svd(L - self.mu, full_matrices=False)
        ev = s ** 2 / max(np.sum(s ** 2), 1e-12)
        self.k = min(int(np.searchsorted(np.cumsum(ev), self.var) + 1), p - 1)
        self.P = Vt[:self.k].T
        return self


class USADFix2(USADFix):
    """USADFIX + 원 논문의 입력 min-max [0, 1] 정규화 (학습 몫 행별) 와 복호기 출력 Sigmoid (부록 A.4).
    출력이 묶이지 않으면 AE2 최대화 단계가 발산한다 (USADFIX 검증 결과, doc/실험1_비교모델_계획.md "추가 수정")."""

    def _net(self, S, T):
        torch = _torch()
        net = super()._net(S, T)
        net.d1 = torch.nn.Sequential(*net.d1, torch.nn.Sigmoid())
        net.d2 = torch.nn.Sequential(*net.d2, torch.nn.Sigmoid())
        return net

    def _norm(self, Z):
        return (Z - self.lo[None, :, None]) / self.rg[None, :, None]

    def fit(self, Z):
        self.lo = Z.min(axis=(0, 2))
        self.rg = np.where(Z.max(axis=(0, 2)) - self.lo > 0, Z.max(axis=(0, 2)) - self.lo, 1.0)
        return super().fit(self._norm(Z))

    def errors(self, Z):
        return super().errors(self._norm(Z))


class PCARBCPhi:
    """Alcala & Qin (2009) 결합 지표 φ 의 재구성 기반 기여도 (2차 수정, 튜닝 없음).
    학습으로 열 표준화 → PCA (설명 분산 var 로 k 선택) → Φ = C̃/δ² + D/τ², C̃ = I − PPᵀ, D = PΛ⁻¹Pᵀ,
    δ² = SPE 관리 한계 (Box 근사 g·χ²_h(α)), τ² = χ²_k(α). 셀 점수 √RBC_i = |(Φx)_i| / √Φ_ii."""

    def __init__(self, var=0.8, alpha=0.99):
        self.var, self.alpha = var, alpha

    def fit(self, Z):
        from scipy.stats import chi2
        X = _flat_t(Z)
        self.cm, self.cs = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1.0)
        Xs = (X - self.cm) / self.cs
        lam, V = np.linalg.eigh(np.cov(Xs, rowvar=False))
        lam, V = np.maximum(lam[::-1], 0.0), V[:, ::-1]
        p = len(lam)
        k = min(int(np.searchsorted(np.cumsum(lam) / max(lam.sum(), 1e-12), self.var) + 1), p - 1)
        P, lk, rest = V[:, :k], np.maximum(lam[:k], 1e-12), lam[k:]
        th1, th2 = rest.sum(), (rest ** 2).sum()
        delta2 = (th2 / th1) * chi2.ppf(self.alpha, th1 ** 2 / th2) if th2 > 0 else 1.0
        tau2 = chi2.ppf(self.alpha, k)
        self.k, self.delta2, self.tau2 = k, delta2, tau2
        self.Phi = (np.eye(p) - P @ P.T) / delta2 + (P / lk) @ P.T / tau2
        self.dphi = np.sqrt(np.maximum(np.diag(self.Phi), 1e-12))
        return self

    def errors(self, Z):
        X = (_flat_t(Z) - self.cm) / self.cs
        return np.abs(_unflat_t((X @ self.Phi) / self.dphi, Z.shape[0], Z.shape[2]))


class RRRLag:
    """감소 랭크 시차 회귀 y_t = c + B y_{t-1}, rank(B) <= r (윈도우 안). 표준 RRR: OLS 적합값의 상위 r 오른쪽
    특이벡터로 B 를 사영. t = 0 은 학습 평균으로 예측."""

    def __init__(self, rank=2, ridge=1e-3):
        self.rank, self.ridge = rank, ridge

    @staticmethod
    def _pairs(Z):
        f = lambda A: np.moveaxis(A, 1, 2).reshape(-1, Z.shape[1])
        return f(Z[:, :, :-1]), f(Z[:, :, 1:])

    def fit(self, Z):
        Xp, Y = self._pairs(Z)
        self.mx, self.my, self.m0 = Xp.mean(0), Y.mean(0), _flat_t(Z).mean(0)
        Xc, Yc = Xp - self.mx, Y - self.my
        B = np.linalg.solve(Xc.T @ Xc + self.ridge * np.eye(Xc.shape[1]), Xc.T @ Yc)
        _, _, Vt = np.linalg.svd(Xc @ B, full_matrices=False)
        Vr = Vt[:min(self.rank, Vt.shape[0])].T
        self.B = B @ Vr @ Vr.T
        return self

    def errors(self, Z):
        pred = np.empty_like(Z)
        pred[:, :, 0] = self.m0[None]
        prev = np.moveaxis(Z[:, :, :-1], 1, 2)                       # (N, T-1, S)
        pred[:, :, 1:] = np.moveaxis((prev - self.mx) @ self.B + self.my, 2, 1)
        return np.abs(Z - pred)


class TransMask(_DeepAE):
    """시점 마스킹 재구성 Transformer. 학습: 시점 25 % 를 마스크 토큰으로 바꾸고 그 시점만 복원.
    점수: 시점을 하나씩 가리고 복원한 |오차| (마스킹 없는 AE 는 항등 사상을 배우기 쉬움)."""
    mask_ratio = 0.25

    def _net(self, S, T):
        torch = _torch()
        nn = torch.nn
        d = self.kw.get("d_model", 16)

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.inp, s.out = nn.Linear(S, d), nn.Linear(d, S)
                s.pos = nn.Parameter(torch.randn(T, d) * 0.02)
                s.tok = nn.Parameter(torch.zeros(d))
                s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True), 2)

            def forward(s, x, m):                                # x: (B, S, T), m: (B, T) bool
                h = s.inp(x.transpose(1, 2))
                h = torch.where(m[..., None], s.tok.expand_as(h), h) + s.pos
                return s.out(s.enc(h)).transpose(1, 2)

        return Net()

    def _loss(self, xb, ep):
        torch = _torch()
        B, _, T = xb.shape
        m = torch.rand(B, T, device=xb.device) < self.mask_ratio
        m[torch.arange(B), torch.randint(0, T, (B,), device=xb.device)] = True
        err = ((self.net(xb, m) - xb) ** 2).mean(1)
        return (err * m).sum() / m.sum()

    def _err(self, x):
        torch = _torch()
        B, _, T = x.shape
        out = torch.zeros_like(x)
        for t in range(T):
            m = torch.zeros(B, T, dtype=torch.bool, device=x.device)
            m[:, t] = True
            out[:, :, t] = (self.net(x, m)[:, :, t] - x[:, :, t]).abs()
        return out


class CAE2D(_DeepAE):
    """윈도우(센서 × 시간)를 1채널 영상으로 본 2D 합성곱 오토인코더 (stride 2 축소 후 복원)."""

    def _net(self, S, T):
        torch = _torch()
        nn = torch.nn
        c = self.kw.get("channels", 8)

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = nn.Sequential(nn.Conv2d(1, c, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(c, max(c // 2, 2), 3, stride=2, padding=1), nn.ReLU())
                s.dec = nn.Sequential(nn.ConvTranspose2d(max(c // 2, 2), c, 3, stride=2, padding=1, output_padding=1),
                                      nn.ReLU(), nn.Conv2d(c, 1, 3, padding=1))

            def forward(s, x):
                return s.dec(s.enc(x[:, None]))[:, 0, :S, :T]

        return Net()


class GDN(_DeepAE):
    """Deng & Hooi (2021) Graph Deviation Network 재구현 (원 코드 아님). 센서 임베딩 v, 코사인 top-k 그래프,
    그래프 어텐션 z_i = ReLU(Σ_j α_ij W x_j), 출력 MLP(v_i ∘ z_i) 로 다음 시점 예측. 윈도우 w, 앞쪽은 첫 값 반복."""
    w = 5

    def _net(self, S, T):
        torch = _torch()
        nn = torch.nn
        F = torch.nn.functional
        d, k, w = self.kw.get("embed", 16), min(S - 1, 5), self.w

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.v = nn.Parameter(torch.randn(S, d) * 0.1)
                s.lin = nn.Linear(w, d, bias=False)
                s.att = nn.Parameter(torch.randn(4 * d) * 0.1)
                s.out = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

            def forward(s, hist):                                # hist: (B, S, w) -> (B, S)
                B = hist.shape[0]
                vn = F.normalize(s.v, dim=1)
                sim = vn @ vn.T - 1e9 * torch.eye(S, device=hist.device)
                nb = torch.cat([torch.arange(S, device=hist.device)[:, None], sim.topk(k, dim=1).indices], 1)
                wx = s.lin(hist)                                 # (B, S, d)
                g = torch.cat([s.v.expand(B, S, d), wx], -1)     # (B, S, 2d)
                gi = g[:, :, None, :].expand(B, S, k + 1, 2 * d)
                e = F.leaky_relu(torch.cat([gi, g[:, nb, :]], -1) @ s.att, 0.2)
                a = torch.softmax(e, -1)
                z = F.relu((a[..., None] * wx[:, nb, :]).sum(2))
                return s.out(z * s.v).squeeze(-1)

        return Net()

    def _hist(self, x):
        """윈도우 앞 이력: pad='zero' (기본, 2차 수정) 는 0 채움. 'repeat' 는 첫 값 반복 (누수판 GDN_v0, 재현용)."""
        torch = _torch()
        B, S, T = x.shape
        pad = torch.zeros_like(x[:, :, :1]).repeat(1, 1, self.w) if self.kw.get("pad", "zero") == "zero" \
            else x[:, :, :1].repeat(1, 1, self.w)
        xp = torch.cat([pad, x], 2)
        hist = xp.unfold(2, self.w, 1)[:, :, :T, :]            # (B, S, T, w): x[t-w .. t-1]
        return hist.permute(0, 2, 1, 3).reshape(B * T, S, self.w), x.permute(0, 2, 1).reshape(B * T, S)

    def _loss(self, xb, ep):
        h, y = self._hist(xb)
        return ((self.net(h) - y) ** 2).mean()

    def _err(self, x):
        B, S, T = x.shape
        h, y = self._hist(x)
        return (self.net(h) - y).abs().reshape(B, T, S).permute(0, 2, 1)


class OmniAnomaly(_DeepAE):
    """Su et al. (2019) 재구현 (원 코드 아님). GRU 추론망 q(z_t | z_{t-1}, e_t) + 평면 흐름 K 개,
    선형 가우스 상태공간 사전 p(z_t | z_{t-1}) = N(O z_{t-1}, I), GRU 생성망 p(x_t | z_{≤t}). ELBO 학습.
    셀 점수 = 표준화 재구성 오차 |x − μ_x| / σ_x (L 표본 평균)."""
    K, H, L = 20, 64, 8

    def _net(self, S, T):
        torch = _torch()
        nn = torch.nn
        F = torch.nn.functional
        zd, H, K = self.kw.get("latent", 3), self.H, self.K

        class Planar(nn.Module):
            def __init__(s):
                super().__init__()
                s.u, s.w = nn.Parameter(torch.randn(zd) * 0.01), nn.Parameter(torch.randn(zd) * 0.01)
                s.b = nn.Parameter(torch.zeros(1))

            def forward(s, z):
                wu = s.w @ s.u
                u = s.u + (-1 + F.softplus(wu) - wu) * s.w / (s.w @ s.w + 1e-8)
                a = torch.tanh(z @ s.w + s.b)
                psi = (1 - a ** 2)[:, None] * s.w[None]
                return z + u[None] * a[:, None], torch.log((1 + psi @ u).abs() + 1e-8)

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.qrnn = nn.GRU(S, H, batch_first=True)
                s.qh = nn.Sequential(nn.Linear(H + zd, H), nn.ReLU())
                s.qmu, s.qsd = nn.Linear(H, zd), nn.Linear(H, zd)
                s.flows = nn.ModuleList([Planar() for _ in range(K)])
                s.O = nn.Linear(zd, zd, bias=False)
                s.prnn = nn.GRU(zd, H, batch_first=True)
                s.xmu, s.xsd = nn.Linear(H, S), nn.Linear(H, S)

            def forward(s, x):                                   # x: (B, S, T)
                seq = x.transpose(1, 2)
                e, _ = s.qrnn(seq)
                B, T_ = seq.shape[:2]
                zp = torch.zeros(B, zd, device=x.device)
                zs, kl = [], 0.0
                for t in range(T_):
                    hq = s.qh(torch.cat([e[:, t], zp], -1))
                    mu, sd = s.qmu(hq), F.softplus(s.qsd(hq)) + 1e-4
                    z0 = mu + sd * torch.randn_like(sd)
                    logq = torch.distributions.Normal(mu, sd).log_prob(z0).sum(-1)
                    z = z0
                    for f in s.flows:
                        z, ld = f(z)
                        logq = logq - ld
                    logp = torch.distributions.Normal(s.O(zp), 1.0).log_prob(z).sum(-1)
                    kl = kl + logq - logp
                    zs.append(z)
                    zp = z
                d, _ = s.prnn(torch.stack(zs, 1))
                mx, sx = s.xmu(d), F.softplus(s.xsd(d)) + 1e-4
                rec = torch.distributions.Normal(mx, sx).log_prob(seq).sum((1, 2))
                return mx.transpose(1, 2), sx.transpose(1, 2), rec - kl

        return Net()

    def _loss(self, xb, ep):
        return -self.net(xb)[2].mean() / (xb.shape[1] * xb.shape[2])

    def _err(self, x):
        torch = _torch()
        torch.manual_seed(self.seed)
        acc = torch.zeros_like(x)
        for _ in range(self.L):
            mx, sx, _ = self.net(x)
            acc += (x - mx).abs() / sx
        return acc / self.L


MODELS = {"PCA": PCASPE, "RPCA": RPCASub, "OLS": OLSCross, "IF": IForest, "LSTMAE": LSTMAE, "USAD": USAD,
          "CAE1D": CAE1D, "LAPRPCA": LapRPCA, "RRR": RRRLag, "TRANS": TransMask, "CAE2D": CAE2D, "GDN": GDN,
          "OMNI": OmniAnomaly, "USADFIX": USADFix, "USADFIX2": USADFix2, "RBCPHI": PCARBCPhi}
GRIDS = {
    "PCA": [dict(var=v) for v in (0.8, 0.9, 0.95, 0.99)],
    "RPCA": [dict(c=c, var=v) for c in (0.5, 1.0) for v in (0.9, 0.99)],
    "OLS": [dict(alpha=a) for a in (1e-3, 1.0, 10.0, 100.0)],
    "IF": [dict(max_samples=m, n_estimators=n) for m in (256, 1.0) for n in (100, 300)],
    "LSTMAE": [dict(hidden=h) for h in (16, 32)],
    "USAD": [dict(latent=z) for z in (8, 16)],
    "CAE1D": [dict(channels=c) for c in (8, 16)],
}
GRIDS2 = {                                                       # 2차 (RRR 랭크는 실행기에서 센서 수 이하만 남김)
    "LAPRPCA": [dict(c=c, gamma=g) for c in (0.5, 1.0) for g in (0.1, 1.0)],
    "RRR": [dict(rank=r) for r in (1, 2, 4, 8)],
    "TRANS": [dict(d_model=d) for d in (16, 32)],
    "CAE2D": [dict(channels=c) for c in (8, 16)],
    "GDN": [dict(embed=e) for e in (16, 32)],
    "OMNI": [dict(latent=z) for z in (3, 8)],
}
GRIDS3 = {                                                       # 2차 수정 (계획 "2차 수정"). RBCPHI 는 PCA 선택 구성을 실행기에서 넣음
    "USADFIX": [dict(latent=z) for z in (8, 16)],
    "GDN": [dict(embed=e) for e in (16, 32)],
    "LAPRPCA": [dict(c=c, gamma=g) for c in (0.5, 1.0) for g in (0.1, 1.0)],
}
GRIDS4 = {"USADFIX2": [dict(latent=z) for z in (8, 16)]}         # 추가 수정 (계획 "2차 수정" 추가 수정)


# ----------------------------------------------------------------------------- 3차 (계획 "3차", sA C4): TranAD · Anomaly Transformer
class TranAD(_DeepAE):
    """Tuli, Casale & Jennings (VLDB 2022) 재구현. 시점 토큰 [x_t ; focus_t] 임베딩 → encoder 1층 → 복호기 2개(TransformerDecoder, 학습 질의).
    2단계(focus = (O1 − x)²) 적대 학습: L1 = w‖O1−x‖² + (1−w)‖Ô2−x‖² (enc + dec1), L2 = w‖O2−x‖² − (1−w)‖Ô2−x‖² (dec2), w = 1/n.
    adversarial=False 면 단일 손실형(발산 대비, 계획 1절)."""
    adversarial = True

    def _net(self, S, T):
        torch = _torch(); nn = torch.nn
        d = self.kw.get("d_model", 16)

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.inp = nn.Linear(2 * S, d)
                s.pos = nn.Parameter(torch.randn(T, d) * 0.02)
                s.qry = nn.Parameter(torch.randn(T, d) * 0.02)
                s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True), 1)
                s.dec1 = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True), 1)
                s.dec2 = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True), 1)
                s.out1, s.out2 = nn.Linear(d, S), nn.Linear(d, S)

            def encode(s, x, focus):                                  # x, focus: (B, S, T)
                h = s.inp(torch.cat([x, focus], 1).transpose(1, 2)) + s.pos
                return s.enc(h)

            def forward(s, x):
                B = x.shape[0]
                m = s.encode(x, torch.zeros_like(x)); q = s.qry.expand(B, -1, -1)
                o1 = s.out1(s.dec1(q, m)).transpose(1, 2); o2 = s.out2(s.dec2(q, m)).transpose(1, 2)
                m2 = s.encode(x, (o1.detach() - x) ** 2)              # 2단계: focus score
                o2h = s.out2(s.dec2(q, m2)).transpose(1, 2)
                return o1, o2, o2h

        return Net()

    def fit(self, Z):
        torch = _torch()
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.dev = self.kw.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        N, S, T = Z.shape
        self.net = self._net(S, T).to(self.dev); n_ = self.net
        p1 = list(n_.inp.parameters()) + [n_.pos, n_.qry] + list(n_.enc.parameters()) + list(n_.dec1.parameters()) + list(n_.out1.parameters())
        p2 = list(n_.dec2.parameters()) + list(n_.out2.parameters())
        o1, o2 = torch.optim.Adam(p1, lr=self.lr), torch.optim.Adam(p2, lr=self.lr)
        x = torch.tensor(Z, dtype=torch.float32, device=self.dev)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        self.diverged = False
        for ep in range(self.epochs):
            w = 1.0 / (ep + 1)
            perm = torch.randperm(N, generator=g)
            for i in range(0, N, self.batch):
                xb = x[perm[i:i + self.batch].to(self.dev)]
                O1, O2, O2h = self.net(xb)
                e1, e2, e2h = ((O1 - xb) ** 2).mean(), ((O2 - xb) ** 2).mean(), ((O2h - xb) ** 2).mean()
                if self.adversarial:
                    L1 = w * e1 + (1 - w) * e2h; L2 = w * e2 - (1 - w) * e2h
                else:
                    L1 = w * e1 + (1 - w) * e2h; L2 = w * e2 + (1 - w) * e2h
                if not (torch.isfinite(L1) and torch.isfinite(L2)):
                    self.diverged = True; break
                o1.zero_grad(); o2.zero_grad()
                (L1 + L2).backward()
                torch.nn.utils.clip_grad_norm_(p1 + p2, 1.0)
                o1.step(); o2.step()
            if self.diverged:
                break
        return self

    def _err(self, x):
        torch = _torch()
        O1, _, O2h = self.net(x)
        return torch.sqrt(0.5 * (O1 - x) ** 2 + 0.5 * (O2h - x) ** 2)


class TranADSingle(TranAD):
    adversarial = False


class AnomalyTransformer(_DeepAE):
    """Xu, Wu, Wang & Long (ICLR 2022) 재구현. Anomaly-Attention 2층: series association S = softmax(QKᵀ/√d),
    prior association P = 정규화 가우스 커널(시점별 학습 σ, 헤드별). AssDis = 층·헤드 평균 [KL(P‖S) + KL(S‖P)] (시점별).
    minimax(k = 3): loss1 = rec − k‖AssDis(P, S.detach)‖, loss2 = rec + k‖AssDis(P.detach, S)‖. 점수 = sqrt(softmax_t(−AssDis_t) · (x̂ − x)²)."""
    k_ad, layers, heads = 3.0, 2, 4

    def _net(self, S, T):
        torch = _torch(); nn = torch.nn
        d, H, L = self.kw.get("d_model", 16), self.heads, self.layers
        dh = d // H

        class AnomAttn(nn.Module):
            def __init__(s):
                super().__init__()
                s.q, s.k, s.v, s.o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
                s.sig = nn.Linear(d, H)
                s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
                s.register_buffer("dist", (torch.arange(T)[:, None] - torch.arange(T)[None, :]).float().abs())

            def forward(s, h):                                       # h: (B, T, d)
                B = h.shape[0]
                q = s.q(h).view(B, T, H, dh).transpose(1, 2); k = s.k(h).view(B, T, H, dh).transpose(1, 2); v = s.v(h).view(B, T, H, dh).transpose(1, 2)
                series = torch.softmax(q @ k.transpose(-1, -2) / dh ** 0.5, dim=-1)                    # (B, H, T, T)
                sigma = torch.sigmoid(s.sig(h) * 5) + 1e-5                                             # (B, T, H) ∈ (0, 1]
                sigma = sigma.transpose(1, 2)[..., None] * 3 + 1e-5                                    # 스케일 (0, 3]
                prior = torch.exp(-0.5 * (s.dist[None, None] / sigma) ** 2) / (sigma * (2 * 3.141592653589793) ** 0.5)
                prior = prior / prior.sum(-1, keepdim=True)
                out = s.o((series @ v).transpose(1, 2).reshape(B, T, d))
                h = s.n1(h + out); h = s.n2(h + s.ff(h))
                return h, series, prior

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.inp, s.out = nn.Linear(S, d), nn.Linear(d, S)
                s.pos = nn.Parameter(torch.randn(T, d) * 0.02)
                s.layers = nn.ModuleList([AnomAttn() for _ in range(L)])

            def forward(s, x):                                       # x: (B, S, T)
                h = s.inp(x.transpose(1, 2)) + s.pos
                ser, pri = [], []
                for lay in s.layers:
                    h, se, pr = lay(h); ser.append(se); pri.append(pr)
                return s.out(h).transpose(1, 2), ser, pri

        return Net()

    @staticmethod
    def _kl(p, q):
        return (p * (torch.log(p + 1e-8) - torch.log(q + 1e-8))).sum(-1)                               # (B, H, T)

    def _assdis(self, ser, pri, detach):
        torch = _torch(); tot = 0
        for se, pr in zip(ser, pri):
            if detach == "series":
                se = se.detach()
            elif detach == "prior":
                pr = pr.detach()
            tot = tot + (self._kl(pr, se) + self._kl(se, pr)).mean(1)                                  # (B, T)
        return tot / len(ser)

    def fit(self, Z):
        global torch
        torch = _torch()
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.dev = self.kw.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        N, S, T = Z.shape
        self.net = self._net(S, T).to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        x = torch.tensor(Z, dtype=torch.float32, device=self.dev)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        self.diverged = False
        for ep in range(self.epochs):
            perm = torch.randperm(N, generator=g)
            for i in range(0, N, self.batch):
                xb = x[perm[i:i + self.batch].to(self.dev)]
                xh, ser, pri = self.net(xb)
                rec = ((xh - xb) ** 2).mean()
                loss1 = rec - self.k_ad * self._assdis(ser, pri, "series").abs().mean()               # 최소화 단계 (prior ← series)
                loss2 = rec + self.k_ad * self._assdis(ser, pri, "prior").abs().mean()                # 최대화 단계 (series ← prior)
                if not (torch.isfinite(loss1) and torch.isfinite(loss2)):
                    self.diverged = True; break
                opt.zero_grad(); loss1.backward(retain_graph=True); loss2.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step()
            if self.diverged:
                break
        return self

    def _err(self, x):
        global torch
        torch = _torch()
        xh, ser, pri = self.net(x)
        ad = self._assdis(ser, pri, None)                                                              # (B, T)
        wgt = torch.softmax(-ad, dim=-1) * x.shape[2]                                                  # 평균 1 로 스케일
        return torch.sqrt(wgt[:, None, :] * (xh - x) ** 2)


class AnomalyTransformerFix(AnomalyTransformer):
    """3차 정정 (doc/실험1_비교모델_계획.md "3차 정정", 2026-10-14): 원문 식 (5) 의 minimax 방향.
    최대화 단계 loss₁ = rec − k‖AssDis(P.detach, S)‖ (series 가 차이를 키움), 최소화 단계 loss₂ = rec + k‖AssDis(P, S.detach)‖ (prior 가 series 를 따라감).
    ANOMTRANS 와 다른 것은 이 두 detach 대상뿐 (ANOMTRANS 는 반대 방향, 기록용으로 유지)."""

    def fit(self, Z):
        global torch
        torch = _torch()
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.dev = self.kw.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        N, S, T = Z.shape
        self.net = self._net(S, T).to(self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        x = torch.tensor(Z, dtype=torch.float32, device=self.dev)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        self.diverged = False
        for ep in range(self.epochs):
            perm = torch.randperm(N, generator=g)
            for i in range(0, N, self.batch):
                xb = x[perm[i:i + self.batch].to(self.dev)]
                xh, ser, pri = self.net(xb)
                rec = ((xh - xb) ** 2).mean()
                loss1 = rec - self.k_ad * self._assdis(ser, pri, "prior").abs().mean()                # 최대화 단계: prior 고정, series 갱신
                loss2 = rec + self.k_ad * self._assdis(ser, pri, "series").abs().mean()               # 최소화 단계: series 고정, prior 갱신
                if not (torch.isfinite(loss1) and torch.isfinite(loss2)):
                    self.diverged = True; break
                opt.zero_grad(); loss1.backward(retain_graph=True); loss2.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step()
            if self.diverged:
                break
        return self


MODELS.update({"TRANAD": TranAD, "TRANADS": TranADSingle, "ANOMTRANS": AnomalyTransformer, "ANOMTRANSFIX": AnomalyTransformerFix})
GRIDS5 = {"TRANAD": [dict(d_model=d) for d in (16, 32)], "ANOMTRANS": [dict(d_model=d) for d in (16, 32)]}   # 3차 (계획 "3차")
GRIDS6 = {"ANOMTRANSFIX": [dict(d_model=d) for d in (16, 32)]}                                             # 3차 정정 (계획 "3차 정정")
