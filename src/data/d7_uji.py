"""D7: UJI 장기 Wi-Fi 지문 데이터 로더 (doc/실험R1-6_계획.md 1절, 구조 점검 data_ext/D7_uji/STRUCTURE.md).

csv 만 읽는다(내려받은 MATLAB 스크립트는 쓰지 않음). 미수신 +100 → NaN.
한 세트 = 행이 지문(스캔)인 rss·crd·ids 파일. 같은 좌표가 연속 6행으로 나오는 블록이 기준점마다 두 개(방문 a, b).
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data_ext" / "D7_uji" / "extracted" / "db"
BLOCK = 6


def _csv(path):
    return np.loadtxt(path, delimiter=",", ndmin=2)


def load_set(month, name):
    """반환 dict(rss (n, 620) NaN=미수신, crd (n, 3), ids (n,) str) 또는 None(파일 없음)."""
    d = DB / f"{month:02d}"
    p = d / f"{name}rss.csv"
    if not p.exists():
        return None
    rss = _csv(p)
    rss = np.where(rss == 100, np.nan, rss)
    crd = _csv(d / f"{name}crd.csv")
    ids = np.loadtxt(d / f"{name}ids.csv", delimiter=",", dtype=str, ndmin=1)
    return dict(rss=rss, crd=crd, ids=ids)


def blocks(S):
    """연속 같은 좌표 6행 블록 목록 [(시작 행, 좌표 tuple)]. 블록 구조가 맞지 않으면 None."""
    crd = S["crd"]
    n = len(crd)
    if n % BLOCK:
        return None
    out = []
    for i in range(0, n, BLOCK):
        c = crd[i:i + BLOCK]
        if not np.all(c == c[0]):
            return None
        out.append((i, tuple(np.round(c[0], 6))))
    return out


def windows(month, name, point_index):
    """세트 하나 → 창 목록. point_index: {좌표: 기준점 번호} (1개월째 trn01 기준).
    반환 (X (n_win, 620, 6), node (n_win,) 'F{층}/P{번호}{a|b}', floor (n_win,)) 또는 None(구조 불일치·좌표 불일치)."""
    S = load_set(month, name)
    if S is None:
        return None
    bl = blocks(S)
    if bl is None:
        return None
    seen, X, node, floor = {}, [], [], []
    for i, c in bl:
        if c not in point_index:
            return None
        v = "a" if c not in seen else "b"
        if c in seen and seen[c] >= 2:
            return None
        seen[c] = seen.get(c, 0) + 1
        X.append(S["rss"][i:i + BLOCK].T)                                     # (620, 6)
        fl = int(c[2])
        node.append(f"F{fl}/P{point_index[c]:02d}{v}")
        floor.append(fl)
    return np.stack(X), np.array(node), np.array(floor)


def point_index_month1():
    S = load_set(1, "trn01")
    idx = {}
    for _, c in blocks(S):
        if c not in idx:
            idx[c] = len(idx)
    return idx
