"""D6: TWR-CLOUD-Bern (Xenakis, Di Maio, Braun; Univ. Bern, Zenodo 10.5281/zenodo.16476264, MIT). 원본은 읽기만 한다.

구조 (기술 문서): scan 12개 × 830 s, scan 마다 DWM3001C 노드 40개가 고정 위치(1–170)에서 서로 4-메시지 TWR.
  listening period 7개(1,5,10,20,30,40,50 ms)가 2 분씩 순서대로 → 메시지 성공률이 시간에 따라 변함 (실제 결측).
표현: 노드 = (scan, 위치), 행 = DIST_A{다른 노드의 위치} (scan 안 40개, 자기 행은 항상 결측), 시간 = 2 s 구간 평균.
  거리 [m] = ToF [ns] × c. 노드 i 의 관측 = i 가 responder 인 교환(Responders 표) + i 가 poller 인 교환(Pollers 표).
분할: listening period 블록(60 구간)마다 윈도우 5개(T = 12) → 앞 3개 학습, 4번째 검증, 5번째 테스트 (블록마다 같은 비율).
셀: mask (관측 있음), true_d (참 거리 m, GT 좌표). 출력: cache/windows/d6/scan{k}.npz + _meta.parquet
"""
import numpy as np
import pandas as pd

from .config import ROOT, load_config
from .safe_pickle import read_pandas_pickle_restricted

SRC = ROOT / "data_ext" / "D6_twr_cloud_bern"
C = 0.299792458                    # m/ns
BIN_S, T = 2.0, 12
PERIODS = [1, 5, 10, 20, 30, 40, 50]


def load_long():
    R, _ = read_pandas_pickle_restricted(SRC / "Rangings at Responders.pkl")
    P, _ = read_pandas_pickle_restricted(SRC / "Rangings at Pollers.pkl")
    a = R.rename(columns={"Responder_ID": "node", "Responder_Pos_ID": "pos", "Poller_ID": "other", "Poller_Pos_ID": "opos",
                          "Responder_X": "x", "Responder_Y": "y", "Responder_Z": "z"})
    b = P.rename(columns={"Poller_ID": "node", "Poller_Pos_ID": "pos", "Responder_ID": "other", "Responder_Pos_ID": "opos",
                          "Poller_X": "x", "Poller_Y": "y", "Poller_Z": "z"})
    cols = ["Scan_ID", "Timestamp", "Listening_Period", "node", "pos", "x", "y", "z", "other", "opos", "Node_Distance_3D", "ToF"]
    m = pd.concat([a[cols].assign(role="resp"), b[cols].assign(role="poll")], ignore_index=True)
    m["dist"] = m.ToF * C
    m["true_d"] = m.Node_Distance_3D / 100.0
    return m


def build_scan(m, scan):
    g = m[m.Scan_ID == scan].copy()
    t0 = g.Timestamp.min()
    g["bin"] = ((g.Timestamp - t0) / 1e9 // BIN_S).astype(int)
    nb = int(g.bin.max()) + 1
    positions = sorted(g.pos.unique())                                  # 이 scan 의 40 위치
    pix = {p: i for i, p in enumerate(positions)}
    S = len(positions)
    node_pos = g.groupby("node").pos.first()
    xyz = g.groupby("pos")[["x", "y", "z"]].first()
    # listening period 블록 경계: 시간순 period 값 변화
    per = g.groupby("bin").Listening_Period.agg(lambda s: s.mode().iloc[0]).reindex(range(nb)).ffill().bfill().astype(int)
    blk = (per != per.shift()).cumsum().to_numpy() - 1                  # 0..6
    agg = g.groupby(["node", "opos", "bin"]).dist.mean()
    X_all, M_all, TD_all, meta = [], [], [], []
    for nd, p in node_pos.items():
        D = np.full((S, nb), np.nan)
        sub = agg.loc[nd]
        rows_i = sub.index.get_level_values(0).map(pix).to_numpy()
        D[rows_i, sub.index.get_level_values(1).to_numpy()] = sub.to_numpy()
        td = np.sqrt(((xyz.loc[positions].to_numpy() - xyz.loc[p].to_numpy()[None]) ** 2).sum(1))   # (S,)
        for b in range(blk.max() + 1):
            bins = np.flatnonzero(blk == b)
            nw = len(bins) // T
            for w in range(nw):
                sel = bins[w * T:(w + 1) * T]
                split = "train" if w < nw - 2 else ("val" if w == nw - 2 else "test")
                X_all.append(D[:, sel]); M_all.append(np.isfinite(D[:, sel]).astype(np.float32))
                TD_all.append(np.repeat(td[:, None], T, 1))
                meta.append(dict(file=f"scan{scan}", space=f"scan{scan}", condition="mixed", node=f"scan{scan}/P{p}",
                                 device=int(nd), seg=b, period_ms=int(per.iloc[sel[0]]), split=split,
                                 t0=float(sel[0] * BIN_S), tidx0=int(sel[0])))
    X = np.stack(X_all); rows = [f"DIST_A{p}" for p in positions]
    return dict(X=X, rows=rows, meta=pd.DataFrame(meta), cells={"mask": np.stack(M_all), "true_d": np.stack(TD_all)}), xyz


def build(cfg=None):
    cfg = cfg or load_config()
    out = cfg["paths"]["cache"] / "windows" / "d6"
    out.mkdir(parents=True, exist_ok=True)
    m = load_long()
    info = []
    for scan in sorted(m.Scan_ID.unique()):
        w, xyz = build_scan(m, scan)
        np.savez_compressed(out / f"scan{scan}.npz", X=np.nan_to_num(w["X"]).astype(np.float32), rows=np.array(w["rows"]),
                            **{f"cell_{k}": v.astype(np.float32) for k, v in w["cells"].items()})
        w["meta"].to_parquet(out / f"scan{scan}_meta.parquet", index=False)
        xyz.reset_index().to_csv(out / f"scan{scan}_positions.csv", index=False)
        obs = w["cells"]["mask"]
        info.append(dict(scan=scan, windows=len(w["X"]), nodes=w["meta"].node.nunique(),
                         split=w["meta"].split.value_counts().to_dict(), obs_frac=float(obs.mean()),
                         obs_frac_by_period={int(k): round(float(obs[w["meta"].period_ms.to_numpy() == k].mean()), 3)
                                             for k in sorted(w["meta"].period_ms.unique())}))
        print(info[-1], flush=True)
    return info


if __name__ == "__main__":
    build()
