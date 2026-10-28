"""D2 (Bregar 2023, Zenodo 8280736) 로더. zip 원본은 읽기만 한다.

zip 구조: raw_data/environment{e}/data/<x_y_z>/ch{c}_A{a}.csv. 1행 버전, 2행 열 이름, 이후 데이터.
앞 26개 필드 = 메타, 나머지 = CIR (읽지 않음). 파일 1개 = 위치·앵커·채널, 반복 측정 31회 (타임스탬프 없음).

결정 (실험 0, PROJECT_BRIEF 2.4): 주 채널 ch5, 정상 학습은 앵커별 LOS 셀만 쓰는 마스크 방식.
시간 축 (configs/data.yaml d2.time_axis):
  path            : 경로 순서(0.3 m 간격) = 시간. 반복 k 번째 값을 이어 만든 의사 궤적 31개 (seg = 환경×k).
                    split 은 k 순서 (위치 안에서 앞선 반복 -> 학습).
  within_position : 위치 = 정지 노드, 반복 순서 k = 시간 (seg = 환경×위치)
"""
import io
import re
import zipfile

import numpy as np
import pandas as pd

from .config import load_config

META = ["TAG_ID", "ANCHOR_ID", "X_TAG", "Y_TAG", "Z_TAG", "X_ANCHOR", "Y_ANCHOR", "Z_ANCHOR", "NLOS", "RANGE",
        "FP_INDEX", "RSS", "RSS_FP", "FP_POINT1", "FP_POINT2", "FP_POINT3", "STDEV_NOISE", "CIR_POWER",
        "MAX_NOISE", "RXPACC", "CHANNEL_NUMBER", "FRAME_LENGTH", "PREAMBLE_LENGTH", "BITRATE", "PRFR",
        "PREAMBLE_CODE"]
KEEP = ["X_TAG", "Y_TAG", "Z_TAG", "X_ANCHOR", "Y_ANCHOR", "Z_ANCHOR", "RANGE", "FP_INDEX", "RSS", "RSS_FP",
        "FP_POINT1", "FP_POINT2", "FP_POINT3", "STDEV_NOISE", "CIR_POWER", "MAX_NOISE", "RXPACC"]
PAT = re.compile(r"data_set/raw_data/environment(\d)/data/([^/]+)/ch(\d)_A(\d)\.csv$")
ANCHOR_IDS = list(range(1, 9))


def load_meta(cfg) -> pd.DataFrame:
    """긴 표: env, pos, path_idx, ch, anchor, k(반복 순서), NLOS(bool), KEEP 열."""
    cp = cfg["paths"]["cache"] / "d2_meta.parquet"
    if cp.exists():
        return pd.read_parquet(cp)
    recs = []
    with zipfile.ZipFile(cfg["paths"]["d2_zip"]) as zf:
        pidx = {}
        for e in range(4):
            w = pd.read_csv(io.BytesIO(zf.read(f"data_set/environment{e}/walking_path.csv")))
            for i, r in w.iterrows():
                pidx[(e, f"{r.x:.2f}_{r.y:.2f}_{r.z:.2f}")] = i
        for n in zf.namelist():
            m = PAT.match(n)
            if not m:
                continue
            e, pos, ch, a = m.groups()
            for k, line in enumerate(zf.read(n).decode("utf-8", "replace").splitlines()[2:]):
                if line.strip():
                    recs.append([int(e), pos, int(ch), int(a), k] + line.split(",", len(META))[:len(META)])
    df = pd.DataFrame(recs, columns=["env", "pos", "ch", "anchor", "k"] + META)
    df["path_idx"] = [pidx[(e, p)] for e, p in zip(df["env"], df["pos"])]
    df["NLOS"] = df["NLOS"].str.upper().eq("NLOS")
    for c in KEEP:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["env", "pos", "path_idx", "ch", "anchor", "k", "NLOS"] + KEEP]
    df.to_parquet(cp, index=False)
    return df


def series(cfg, env: int, time_axis=None) -> pd.DataFrame:
    """한 환경의 넓은 표 (한 행 = seg 의 한 시간 스텝). 열: 메타 + {metric}_A{a}, NLOS_A{a}, mask_A{a}, lab_D_A{a}.
    time_axis 가 within_position 이면 split = 'all' (실험 3 위치 홀드아웃은 위치로 나눈다)."""
    d2 = dict(cfg["d2"])
    if time_axis:
        d2["time_axis"] = time_axis
    m = load_meta(cfg)
    m = m[(m["env"] == env) & (m["ch"] == d2["channel"])].copy()
    m["true_d"] = np.sqrt((m.X_TAG - m.X_ANCHOR) ** 2 + (m.Y_TAG - m.Y_ANCHOR) ** 2 + (m.Z_TAG - m.Z_ANCHOR) ** 2)
    metrics = d2["metrics"] + d2.get("extra_metrics", [])
    wide = m.pivot_table(index=["path_idx", "k"], columns="anchor", values=metrics + ["NLOS", "true_d"],
                         aggfunc="first")
    wide.columns = [f"{v}_A{a}" for v, a in wide.columns]
    wide = wide.reset_index()
    tag = m.groupby("path_idx")[["X_TAG", "Y_TAG", "Z_TAG"]].first()
    wide = wide.join(tag, on="path_idx")
    for a in ANCHOR_IDS:
        nl = wide[f"NLOS_A{a}"].astype(float)
        wide[f"lab_D_A{a}"] = nl                      # 기하 NLOS 라벨 (채널 무관, 실험 0 확인)
        wide[f"mask_A{a}"] = 1.0 - nl                 # 정상 학습에 쓸 셀 = LOS
    wide["file"] = f"D2_env{env}"
    wide["space"] = f"env{env}"
    wide["condition"] = "mixed"
    wide["edge"] = False
    wide["normal"] = True
    if d2["time_axis"] == "path":
        wide["seg"] = env * 100 + wide["k"]
        wide["tidx"] = wide["path_idx"]
        wide["t"] = wide["path_idx"].astype(float)
        wide["node"] = wide["file"] + "/path"
        kmax = wide["k"].max() + 1
        r = np.cumsum(cfg["split"]["ratios"])[:-1] * kmax
        wide["split"] = np.where(wide["k"] < r[0], "train", np.where(wide["k"] < r[1], "val", "test"))
    else:
        wide["seg"] = env * 1000 + wide["path_idx"]
        wide["tidx"] = wide["k"]
        wide["t"] = wide["k"].astype(float)
        wide["node"] = wide["file"] + "/p" + wide["path_idx"].astype(str)
        wide["split"] = "all"
    return wide.sort_values(["seg", "tidx"]).reset_index(drop=True)


if __name__ == "__main__":
    cfg = load_config()
    meta = load_meta(cfg)
    print("D2 meta rows", len(meta), "channels", sorted(meta.ch.unique()))
    for e in range(4):
        s = series(cfg, e)
        print(f"env{e}: rows={len(s)} segs={s.seg.nunique()} split={s.split.value_counts().to_dict()} "
              f"LOS cell ratio={s.filter(like='mask_A').to_numpy().mean():.3f}")
