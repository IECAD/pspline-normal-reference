"""분할과 윈도우 (데이터셋 공통).

입력: 긴 표 (한 행 = 한 노드의 한 시간 스텝). 필수 열: node, seg(연속 구간), tidx(구간 안 순서), edge(제외 여부).
  - 분할: 구간 안 시간 순서대로 ratios 비율 (학습 -> 검증 -> 테스트). 셔플 없음.
           정상 파일이 아니면 전부 test.
  - 윈도우: 같은 (seg, split) 안의 연속 T 스텝만. 구간·분할 경계를 넘지 않는다.
출력: X (N, S, T) 값, 같은 모양의 셀 단위 배열(라벨·마스크), meta (윈도우별 정보)
"""
import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def assign_split(df: pd.DataFrame, ratios, normal_col="normal", key="seg") -> pd.Series:
    """구간(key) 안 시간 순서 비율로 split 을 붙인다. edge 행은 'edge'."""
    out = pd.Series("test", index=df.index, dtype=object)
    core = ~df["edge"]
    r = np.cumsum(ratios)[:-1]
    for _, g in df[core & df[normal_col]].groupby(key, sort=False):
        frac = np.arange(len(g)) / len(g)
        out.loc[g.index] = np.where(frac < r[0], "train", np.where(frac < r[1], "val", "test"))
    out[df["edge"]] = "edge"
    return out


def fill_short_gaps(df, cols, key="seg", limit=1):
    """같은 구간 안에서 limit 스텝까지 앞값으로 채운다 (패킷 손실 1 사이클 수준)."""
    df = df.copy()
    df[cols] = df.groupby(key)[cols].ffill(limit=limit)
    return df


def make_windows(df, value_cols, T, stride, cell_cols=None, key="seg", max_missing=0,
                 meta_cols=("file", "space", "condition", "node", "seg", "split")):
    """윈도우 생성.

    cell_cols: {이름: [value_cols 와 같은 순서의 열 이름]} -> 셀 단위 배열 (라벨, 마스크 등)
    반환: dict(X=(N,S,T), cells={이름: (N,S,T)}, meta=DataFrame)
    """
    cell_cols = cell_cols or {}
    Xs, cells, metas = [], {k: [] for k in cell_cols}, []
    use = df[df["split"] != "edge"]
    for (seg, split), g in use.groupby([key, "split"], sort=False):
        g = g.sort_values("tidx")
        # 분할 안에서도 tidx 가 끊기면(가운데 edge 등) 연속 블록으로 나눈다
        block = (g["tidx"].diff() != 1).cumsum()
        for _, b in g.groupby(block):
            v = b[value_cols].to_numpy(float).T            # (S, L)
            L = v.shape[1]
            for s in range(0, L - T + 1, stride):
                w = v[:, s:s + T]
                if np.isnan(w).sum() > max_missing:
                    continue
                Xs.append(w)
                for k, cols in cell_cols.items():
                    cells[k].append(b[cols].to_numpy(float).T[:, s:s + T])
                m = {c: b[c].iloc[s] for c in meta_cols if c in b}
                m.update(t0=b["t"].iloc[s] if "t" in b else np.nan, tidx0=int(b["tidx"].iloc[s]))
                metas.append(m)
    X = np.stack(Xs) if Xs else np.empty((0, len(value_cols), T))
    cells = {k: (np.stack(v) if v else np.empty((0, len(value_cols), T))) for k, v in cells.items()}
    return dict(X=X, cells=cells, meta=pd.DataFrame(metas), rows=list(value_cols))
