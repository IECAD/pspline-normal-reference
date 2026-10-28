"""센서 축 합성 (①변수 V, ②스케일 S, ③순서 O, ④구조 T). doc/합성방식_실험계획_최종.md 3절.

모든 통계(스케일, 상관, 순서)는 정상 학습 윈도우의 관측 셀(W > 0)에서만 계산하고 고정한다.
상관은 노드별 평균을 뺀 뒤(노드 안 공변동) 계산한다 — 노드마다 기하가 달라 평균 차이가 상관을 지배하지 않게.

데이터별 지표 이름:
  d1: DIST, RSSI, FP2 (앵커 1-4).  RX−FP 는 dBm 단위 FP 가 없어 정의할 수 없음 -> V4, V5 불가
  d2: RANGE, RSS, RSS_FP (앵커 1-8). RXFP = RSS − RSS_FP (dB)
  d4: 변수 V01.. (변수 = '앵커' 역할). ①축 없음, 블록 1개, T3 불가
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from .pspline import diff_matrix

METRICS = {
    "d1": {"V1": ["DIST"], "V2": ["DIST", "RSSI"], "V3": ["DIST", "RSSI", "FP2"], "V4": None, "V5": None},
    "d6": {"V1": ["DIST"], "V2": None, "V3": None, "V4": None, "V5": None},   # TWR-CLOUD-Bern: 거리만
    "d8": {"V1": ["DIST"], "V2": None, "V3": None, "V4": None, "V5": None},   # HYMN: 거리만 (rssi 고정값)
    "d2": {"V1": ["RANGE"], "V2": ["RANGE", "RSS"], "V3": ["RANGE", "RSS", "RSS_FP"],
           "V4": ["RANGE", "RSS", "RSS_FP", "RXFP"], "V5": ["RANGE", "RXFP"]},
}


class NotApplicable(Exception):
    pass


def split_row(name):
    """'DIST_A1' -> ('DIST', 'A1'); 'RSS_FP_A3' -> ('RSS_FP', 'A3'); 'V07' -> ('V', 'V07')."""
    if "_A" in name:
        m, a = name.rsplit("_A", 1)
        return m, f"A{a}"
    return "V", name


def select_rows(ds, V, X, rows):
    """원 윈도우 (N, S_all, T) 에서 V 코드의 지표 행을 고르고 파생 행을 만든다."""
    if ds == "d4":
        return X, list(rows)
    mets = METRICS[ds][V]
    if mets is None:
        raise NotApplicable(f"{ds} {V}: RX−FP 를 만들 dBm 단위 FP power 가 없음")
    idx = {r: i for i, r in enumerate(rows)}
    anchors = sorted({split_row(r)[1] for r in rows}, key=lambda a: int(a[1:]))
    out, names = [], []
    for m in mets:
        for a in anchors:
            if m == "RXFP":
                out.append(X[:, idx[f"RSS_{a}"]] - X[:, idx[f"RSS_FP_{a}"]])
            else:
                out.append(X[:, idx[f"{m}_{a}"]])
            names.append(f"{m}_{a}")
    return np.stack(out, 1), names


def _scale_params(S, X, W):
    x = np.where(W > 0, X, np.nan)
    flat = np.moveaxis(x, 1, 0).reshape(x.shape[1], -1)
    if S in ("S0",):
        return np.zeros(len(flat)), np.ones(len(flat))
    if S in ("S1", "S4"):
        c, d = np.nanmean(flat, 1), np.nanstd(flat, 1)
    elif S == "S2":
        c, d = np.nanmin(flat, 1), np.nanmax(flat, 1) - np.nanmin(flat, 1)
    elif S == "S3":
        q1, q2, q3 = np.nanpercentile(flat, [25, 50, 75], axis=1)
        c, d = q2, q3 - q1
    else:
        raise ValueError(S)
    d = np.where(np.isfinite(d) & (d > 0), d, 1.0)
    c = np.where(np.isfinite(c), c, 0.0)
    return c, d


def node_centered_corr(X, W, nodes):
    """노드별 행 평균을 뺀 뒤 행 간 Pearson 상관 (관측 셀만, 쌍별)."""
    x = np.where(W > 0, X, np.nan)
    xc = np.empty_like(x)
    for nd in np.unique(nodes):
        ix = nodes == nd
        m = np.nanmean(x[ix], axis=(0, 2), keepdims=True)
        xc[ix] = x[ix] - m
    flat = np.moveaxis(xc, 1, 0).reshape(x.shape[1], -1).T
    C = pd.DataFrame(flat).corr(min_periods=10).to_numpy()
    return np.nan_to_num(C, nan=0.0)


def seriate(C):
    n = len(C)
    if n <= 2:
        return np.arange(n)
    D = 1 - np.abs(C)
    np.fill_diagonal(D, 0)
    D = np.clip((D + D.T) / 2, 0, None)
    Z = linkage(squareform(D, checks=False), method="average")
    return leaves_list(optimal_leaf_ordering(Z, squareform(D, checks=False)))


class Composer:
    def __init__(self, ds, V="V3", S="S1", O="O2", T="T2", seed=0):
        if ds == "d4" and T == "T3":
            raise NotApplicable("d4: 앵커×지표 구조가 없어 T3 불가")
        self.ds, self.V, self.S, self.O, self.T, self.seed = ds, V, S, O, T, seed

    # ---------------- 학습 ----------------
    def fit(self, X_all, W_all, rows_all, nodes):
        X, names = select_rows(self.ds, self.V, X_all, rows_all)
        W = W_all if W_all.shape[1] == X.shape[1] else self._w_for(W_all, rows_all, names)
        self.names_sel = names
        self.c, self.d = _scale_params(self.S, X, W)
        Xs = (X - self.c[None, :, None]) / self.d[None, :, None]
        mets = [split_row(n)[0] for n in names]
        anchors = [split_row(n)[1] for n in names]
        self.C = node_centered_corr(Xs, W, nodes)
        rng = np.random.default_rng(self.seed)
        uniq_m = list(dict.fromkeys(mets))
        if self.ds == "d4" or self.T == "T1":
            blocks = [list(range(len(names)))]
        else:
            blocks = [[i for i, m in enumerate(mets) if m == um] for um in uniq_m]
        order, block_ids = [], []
        if self.T == "T3":
            uniq_a = list(dict.fromkeys(anchors))
            Ca = np.zeros((len(uniq_a), len(uniq_a)))         # 앵커 간 상관 = 지표별 |ρ| 평균
            for blk in blocks:
                Ca += np.abs(self.C[np.ix_(blk, blk)])
            Ca /= len(blocks)
            aord = self._order(np.arange(len(uniq_a)), Ca, rng)
            for bi, blk in enumerate(blocks):
                order += [blk[j] for j in aord]
                block_ids += [bi] * len(blk)
            self.n_anchor, self.n_metric = len(uniq_a), len(blocks)
        else:
            for bi, blk in enumerate(blocks):
                sub = self.C[np.ix_(blk, blk)]
                ordb = self._order(np.arange(len(blk)), sub, rng)
                order += [blk[j] for j in ordb]
                block_ids += [bi] * len(blk)
        self.order = np.array(order)
        self.block_ids = np.array(block_ids)
        self.rows = [names[i] for i in self.order]
        self.anchor_of_row = [split_row(r)[1] for r in self.rows]
        self.block_weight = np.ones(len(self.rows))
        return self

    def _order(self, idx, C, rng):
        if self.O == "O2":
            return seriate(C)
        if self.O == "O3":
            return rng.permutation(len(idx))
        return np.arange(len(idx))                       # O1, O4, T4(—)

    def _w_for(self, W_all, rows_all, names):
        """마스크를 선택 행에 맞춘다 (파생 행 RXFP 는 RSS 와 RSS_FP 마스크의 곱)."""
        idx = {r: i for i, r in enumerate(rows_all)}
        out = []
        for n in names:
            m, a = split_row(n)
            if m == "RXFP":
                out.append(W_all[:, idx[f"RSS_{a}"]] * W_all[:, idx[f"RSS_FP_{a}"]])
            else:
                out.append(W_all[:, idx[n]])
        return np.stack(out, 1)

    def set_block_weights(self, resid_var_by_block):
        """S4: 블록별 정상 잔차분산의 역수로 가중 (행 값에 sqrt(w) 를 곱함)."""
        w = np.array([1.0 / max(resid_var_by_block[b], 1e-12) for b in self.block_ids])
        self.block_weight = np.sqrt(w)

    # ---------------- 변환 ----------------
    def transform(self, X_all, W_all, rows_all):
        X, names = select_rows(self.ds, self.V, X_all, rows_all)
        W = W_all if W_all.shape[1] == X.shape[1] else self._w_for(W_all, rows_all, names)
        Xs = (X - self.c[None, :, None]) / self.d[None, :, None]
        Xs = Xs[:, self.order] * self.block_weight[None, :, None]
        W = W[:, self.order]
        return Xs, W

    # ---------------- 센서 방향 penalty ----------------
    def sensor_penalties(self):
        S = len(self.rows)
        if self.T == "T4":
            return {}
        if self.T == "T3":
            Da, Dm = diff_matrix(self.n_anchor, 2), diff_matrix(self.n_metric, 2)
            Pa = np.kron(np.eye(self.n_metric), Da.T @ Da)
            Pm = np.kron(Dm.T @ Dm, np.eye(self.n_anchor))
            return {"anchor": Pa, "metric": Pm} if Dm.size else {"anchor": Pa}
        P = np.zeros((S, S))
        for b in np.unique(self.block_ids):
            ix = np.flatnonzero(self.block_ids == b)
            if self.O == "O4":
                Cb = np.abs(self.C[np.ix_(self.order[ix], self.order[ix])])
                np.fill_diagonal(Cb, 0)
                Lb = np.diag(Cb.sum(1)) - Cb
                P[np.ix_(ix, ix)] = Lb
            else:
                D = diff_matrix(len(ix), 2)
                P[np.ix_(ix, ix)] = D.T @ D
        return {"sensor": P}
