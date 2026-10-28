"""D8: HYMN UWB 부분 (TU Dresden TUD-ITVS, Zenodo 10.5281/zenodo.17979434 v1.3, 데이터 CC BY 4.0). 원본은 읽기만 한다.

구조 (data_ext/D8_hymn/README.md, doc/공개데이터_조사_2026-10-14.md §3):
  정지 지점 48개(격자 A..B.. 42 + 전이 T01–T06)에서 태그 하나가 앵커 10개와 TWR 을 3 분씩 반복.
  sqlite 의 LOG.MESSAGE = JSON 목록 = 태그 1개의 라운드 1회(앵커 10개 항목). 실패 측거는 value = −70 → 결측.
  주 태그 8121069331292423818 만 쓴다(데이터 제공자의 전처리 기본값). 지점 배정은 time_reference.csv 의 UTC 구간.
표현: 노드 = 지점, 행 = DIST_A{1..10} (앵커 UWB_01..UWB_10), 시간 = 라운드 1회(약 1.23 s), T = 12.
분할(시간 순): 지점마다 창 nw = 라운드 수 // 12 → 마지막 2개 테스트, 그 앞 2개 검증, 나머지 학습.
셀: mask (관측 있음), true_d (참 3D 거리 m: 지점 UWB2 좌표 ↔ 앵커 좌표, 데이터 제공자 전처리와 같은 좌표).
출력: cache/windows/d8/hymn.npz + _meta.parquet
내려받은 저장소의 파이썬 코드는 실행하지 않는다. 앵커 ID 매핑은 preprocess_uwb.py 의 anchor_mapping 을 글로 옮겨 적었다.
"""
import json
import sqlite3

import numpy as np
import pandas as pd

from .config import ROOT, load_config

SRC = ROOT / "data_ext" / "D8_hymn" / "x" / "TUD-ITVS-HYMN-dataset-8ae0f89" / "data"
TAG = "8121069331292423818"
T = 12
N_VAL, N_TEST = 2, 2
ANCHOR_MAP = {"AB": 6, "9A": 7, "97": 1, "94": 5, "89": 2, "92": 4, "9C": 9, "95": 3, "9B": 8, "98": 10}   # addressB 끝 두 자리(16진) -> UWB_xx
S = 10


def load_rounds():
    """주 태그의 라운드 전부: DataFrame(ts_ms, A1..A10 거리 m, 실패 NaN). 파일은 읽기 전용으로 연다."""
    recs = []
    for p in sorted((SRC / "raw" / "uwb").glob("*.sqlite")):
        con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
        try:
            for (msg,) in con.execute("SELECT MESSAGE FROM LOG ORDER BY ID"):
                m = json.loads(msg.decode("utf-8") if isinstance(msg, bytes) else msg)
                if not m or str(m[0].get("addressA")) != TAG:
                    continue
                d = np.full(S, np.nan)
                for it in m:
                    a = ANCHOR_MAP.get(hex(int(it["addressB"]))[-2:].upper())
                    v = float(it["value"])
                    if a is not None and v > 0:
                        d[a - 1] = v
                recs.append((int(m[0]["timestamp"]), p.name, *d))
        finally:
            con.close()
    cols = ["ts_ms", "file"] + [f"A{k}" for k in range(1, S + 1)]
    return pd.DataFrame(recs, columns=cols).sort_values("ts_ms").reset_index(drop=True)


def reference():
    ref = SRC / "reference" / "csv"
    tr = pd.read_csv(ref / "time_reference.csv")
    pc = pd.read_csv(ref / "point_coordinates.csv").set_index("point_id")
    ac = pd.read_csv(ref / "anchor_coordinates.csv").set_index("point_id")
    anc = np.stack([ac.loc[f"UWB_{k:02d}", ["X_LOCAL", "Y_LOCAL", "Z_LOCAL"]].to_numpy(float) for k in range(1, S + 1)])
    return tr, pc, anc


def build_all():
    r = load_rounds()
    tr, pc, anc = reference()
    t_utc = pd.to_datetime(r.ts_ms, unit="ms")
    X_all, M_all, TD_all, meta, info = [], [], [], [], []
    rows = [f"DIST_A{k}" for k in range(1, S + 1)]
    for _, row in tr.iterrows():
        pt = row["point"]
        sel = (t_utc >= pd.Timestamp(row["start_time_UTC"])) & (t_utc <= pd.Timestamp(row["end_time_UTC"]))
        D = r.loc[sel, [f"A{k}" for k in range(1, S + 1)]].to_numpy(float).T     # (S, n)
        n = D.shape[1]
        nw = n // T
        xyz = pc.loc[pt, ["X_LOCAL_UWB2", "Y_LOCAL_UWB2", "Z_LOCAL_UWB2"]].to_numpy(float)
        td = np.sqrt(((anc - xyz[None]) ** 2).sum(1))
        for w in range(nw):
            cols = slice(w * T, (w + 1) * T)
            split = "train" if w < nw - N_VAL - N_TEST else ("val" if w < nw - N_TEST else "test")
            X_all.append(D[:, cols]); M_all.append(np.isfinite(D[:, cols]).astype(np.float32))
            TD_all.append(np.repeat(td[:, None], T, 1))
            meta.append(dict(file="hymn", space="hymn", condition="static", node=pt,
                             kind="transition" if pt.startswith("T") else "grid", seg=0, split=split,
                             t0=float(w * T), tidx0=int(w * T)))
        info.append(dict(point=pt, rounds=n, windows=nw, all_fail_rounds=int((~np.isfinite(D)).all(0).sum()),
                         obs_frac=float(np.isfinite(D[:, :nw * T]).mean()) if nw else np.nan,
                         obs_frac_by_anchor=[round(float(v), 3) for v in np.isfinite(D[:, :nw * T]).mean(1)] if nw else []))
    X = np.stack(X_all)
    return dict(X=X, rows=rows, meta=pd.DataFrame(meta), cells={"mask": np.stack(M_all), "true_d": np.stack(TD_all)}), \
        pd.DataFrame(info), r


def build(cfg=None):
    cfg = cfg or load_config()
    out = cfg["paths"]["cache"] / "windows" / "d8"
    out.mkdir(parents=True, exist_ok=True)
    w, info, r = build_all()
    np.savez_compressed(out / "hymn.npz", X=np.nan_to_num(w["X"]).astype(np.float32), rows=np.array(w["rows"]),
                        **{f"cell_{k}": v.astype(np.float32) for k, v in w["cells"].items()})
    w["meta"].to_parquet(out / "hymn_meta.parquet", index=False)
    info.to_csv(out / "hymn_points.csv", index=False)
    return w, info, r


if __name__ == "__main__":
    build()
