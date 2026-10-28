"""실험 0 분석 공통: raw/ xlsx 로더 (원본은 읽기만 한다).

xlsx 파싱이 느리므로 analysis/out/cache/ 에 pickle 캐시를 둔다.
캐시는 원본 파일 크기·수정시각이 바뀌면 다시 만든다.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "analysis" / "out"
FIG = OUT / "fig"
CACHE = OUT / "cache"
for d in (OUT, FIG, CACHE):
    d.mkdir(parents=True, exist_ok=True)

FILES = [
    "los.xlsx",
    "nlos_static.xlsx",
    "nlos_dynamic.xlsx",
    "cad_static_NLOS.xlsx",
    "cad_dynamic_NLOS.xlsx",
    "hallway_los.xlsx",
    "hallway_dynamic_NLOS.xlsx",
]


def load(name: str) -> pd.DataFrame:
    src = RAW / name
    st = src.stat()
    key = f"{src.stem}_{st.st_size}_{int(st.st_mtime)}.pkl"
    cp = CACHE / key
    if cp.exists():
        return pd.read_pickle(cp)
    df = pd.read_excel(src, sheet_name=0)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.insert(0, "row", range(len(df)))  # 원본 행 순서 보존
    df.to_pickle(cp)
    return df


def load_all() -> dict:
    return {n.replace(".xlsx", ""): load(n) for n in FILES}


# --- 데이터셋 메타 (파일명 + 사용자 제공 정보, 2026-10-07) ---
SPACE = {  # 측정 공간
    "los": "classroom", "nlos_static": "classroom", "nlos_dynamic": "classroom",
    "cad_static_NLOS": "cad", "cad_dynamic_NLOS": "cad",
    "hallway_los": "hallway", "hallway_dynamic_NLOS": "hallway",
}
LOS_REF = {  # 같은 공간의 LOS 파일 (앵커 고정 여부: classroom 은 사용자 확인, hallway 는 미확인)
    "nlos_static": "los", "nlos_dynamic": "los", "hallway_dynamic_NLOS": "hallway_los",
}
# 격자 한 칸 (m). 출처: p00 논문 §V(3.84 × 6.12 m, 5×7), 데이터수집_및_변수설명.md, model_report.md §1.3.
# 2026-10-07 사용자가 처음 준 0.987 / 1.024 와 다르다 -> 13_d1_geometry.py 에서 LOS 적합으로 비교.
GRID_DX, GRID_DY = 0.96, 1.02
ANCHOR_HEIGHT_CLASSROOM = 2.0  # m, p00 논문 §V. Lab 은 같은 격자·좌표로 기록됨, Hallway 높이는 출처 없음
# 앵커 (x, y) m, 격자 원점 P00 기준. 출처: p00 §V, model_report.md §1.2, 데이터 문서 §1.2 (오프셋)
ANCHORS = {
    "classroom": {1: (-0.33, 6.33), 2: (4.17, 6.33), 3: (4.17, -0.21), 4: (-0.33, -0.21)},
    "cad": {1: (-0.33, 6.33), 2: (4.17, 6.33), 3: (4.17, -0.21), 4: (-0.33, -0.21)},  # 강의실과 다른 방, 크기·배치 같음
    # Hallway: 문서에는 오프셋(X 0, Y ±0.9)만 있음. 절대값은 그리드 범위(x 0~2, y 0~6) + 오프셋으로 계산
    "hallway": {1: (0.0, 7.02), 2: (1.92, 7.02), 3: (1.92, -0.90), 4: (0.0, -0.90)},
}
NUM_COLS = ["DIST", "RSSI", "SEQ", "LOSS", "POLL_TX", "RESP_RX", "RESP_TX", "CIR", "FP2", "MaxNoise", "SNR"]
REORDER_BY_SEQ = {"los", "nlos_static", "nlos_dynamic"}  # 초 안의 행 순서가 뒤섞인 파일 (1단계 확인)
EDGE_TRIM = 3.0  # s, 구간 시작에서 태그 이동이 기록된 시간 (3단계 확인)


def prep(name: str) -> pd.DataFrame:
    """분석용 전처리 (원본은 그대로 두고 메모리에서만 처리).

    - 비수치 값 -> NaN, DIST 가 NaN 인 행 제거
    - July 파일: 같은 초 안에서 SEQ 순으로 재정렬 (SEQ 순환은 초 안 범위가 128 을 넘으면 작은 값에 +256)
    - t: 경과 시간(s). 초 해상도라 같은 초 안의 k 번째 행을 k/n 초로 배치한 추정값
    - seg: group 연속 구간 번호 (같은 포인트를 두 번 기록한 경우가 있어 point_id 대신 group 사용)
    - t_in_seg: 구간 시작부터 경과 시간(s). edge: 구간 시작 EDGE_TRIM 초 (태그 이동이 기록된 구간, 3단계 확인)
    - x_m / y_m: 격자 원점(P00) 기준 태그 위치(m)
    """
    df = load(name + ".xlsx").copy()
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["DIST"])
    if name in REORDER_BY_SEQ:
        g = df.groupby("timestamp")["SEQ"]
        span = g.transform("max") - g.transform("min")
        s = df["SEQ"].where(~((span > 128) & (df["SEQ"] < 128)), df["SEQ"] + 256)
        df = df.assign(_s=s).sort_values(["timestamp", "_s"], kind="stable").drop(columns="_s")
    df = df.reset_index(drop=True)
    k = df.groupby("timestamp").cumcount()
    n = df.groupby("timestamp")["row"].transform("size")
    df["t"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() + k / n
    df["seg"] = (df["group"] != df["group"].shift()).cumsum()
    df["t_in_seg"] = df["t"] - df.groupby("seg")["t"].transform("min")
    df["edge"] = df["t_in_seg"] < EDGE_TRIM
    df["x_m"] = df["x"] * GRID_DX
    df["y_m"] = df["y"] * GRID_DY
    df["file"] = name
    df["space"] = SPACE[name]
    return df
