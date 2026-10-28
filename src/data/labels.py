"""D1 앵커×시간 라벨: 방법 A (측정 - 실제 거리), A' (측정 - 같은 포인트 기준 중앙값).

근거: analysis/out/data_report.md 5절, doc/PROJECT_BRIEF.md 4.1절.
  - A: 실제 거리 = sqrt(dx^2 + dy^2 + dz^2) + b_앵커. dz, b 는 calib 파일의 학습+검증 사이클로 적합
       (태그 높이가 어느 출처에도 없어 dz 를 추정한다)
  - A': 기준 = calib 파일의 학습+검증 사이클에서 같은 포인트·앵커의 DIST 중앙값
       (Lab 은 calib = cad_static 이므로 '고정 환경 대비 변화' 라벨이 된다)
라벨은 smooth_samples 이동 중앙값이 판정 임계값을 넘으면 1. 계산 불가 셀은 NaN.
RX-FP 차이는 쓰지 않는다 (모델 입력과 순환 논리).
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ANCHOR_IDS = [1, 2, 3, 4]


def _anchor_xy(cfg, space):
    A = cfg["d1"]["anchors"][space]
    return {int(a): tuple(v) for a, v in A.items()}


def fit_geometry(cfg, cyc_calib: pd.DataFrame, robust: bool) -> dict:
    """calib 사이클(학습+검증)의 포인트×앵커 DIST 중앙값으로 dz, b_1..b_4 적합."""
    space = cyc_calib["space"].iloc[0]
    A = _anchor_xy(cfg, space)
    use = cyc_calib[cyc_calib["split"].isin(["train", "val"])]
    rows = []
    for a in ANCHOR_IDS:
        m = use.groupby("point_id").agg(x_m=("x_m", "first"), y_m=("y_m", "first"), d=(f"DIST_A{a}", "median"))
        m["a"] = a
        rows.append(m.dropna())
    pm = pd.concat(rows)
    ax = pm["a"].map(lambda a: A[a][0]).to_numpy()
    ay = pm["a"].map(lambda a: A[a][1]).to_numpy()
    hd2 = (pm["x_m"].to_numpy() - ax) ** 2 + (pm["y_m"].to_numpy() - ay) ** 2
    ai = pm["a"].to_numpy() - 1

    def resid(p):
        return np.sqrt(hd2 + p[0] ** 2) + p[1:][ai] - pm["d"].to_numpy()

    res = least_squares(resid, [1.0, 0, 0, 0, 0], loss="soft_l1" if robust else "linear", f_scale=0.1)
    return dict(space=space, calib_file=cyc_calib["file"].iloc[0], robust=robust, dz=float(abs(res.x[0])),
                bias={a: float(res.x[a]) for a in ANCHOR_IDS},
                rms=float(np.sqrt(np.mean(res.fun ** 2))), mad=float(np.median(np.abs(res.fun))), n=int(len(pm)))


def reference_medians(cyc_calib: pd.DataFrame) -> pd.DataFrame:
    use = cyc_calib[cyc_calib["split"].isin(["train", "val"])]
    return use.groupby("point_id")[[f"DIST_A{a}" for a in ANCHOR_IDS]].median()


def add_labels(cfg, cyc: pd.DataFrame, geom: dict, ref_med: pd.DataFrame) -> pd.DataFrame:
    """사이클 표에 errA_A{a}, lab_A_A{a}, offA2_A{a}, lab_A2_A{a} 열을 붙인다."""
    cyc = cyc.copy()
    space = cyc["space"].iloc[0]
    A = _anchor_xy(cfg, space)
    k = cfg["labels"]["A"]["smooth_samples"]
    thr_A = cfg["labels"]["A"]["decision_threshold"][space]
    thr_A2 = cfg["labels"]["A2"]["decision_threshold"]
    core = cyc["split"] != "edge"

    def smooth(s):
        return s.where(core).groupby(cyc["seg"]).transform(
            lambda x: x.rolling(k, center=True, min_periods=k // 2).median())

    for a in ANCHOR_IDS:
        d = cyc[f"DIST_A{a}"]
        true_d = np.sqrt((cyc["x_m"] - A[a][0]) ** 2 + (cyc["y_m"] - A[a][1]) ** 2 + geom["dz"] ** 2)
        cyc[f"errA_A{a}"] = d - true_d - geom["bias"][a]
        es = smooth(cyc[f"errA_A{a}"])
        cyc[f"lab_A_A{a}"] = np.where(es.isna(), np.nan, (es > thr_A).astype(float))
        ref = cyc["point_id"].map(ref_med[f"DIST_A{a}"]) if len(ref_med) else np.nan
        cyc[f"offA2_A{a}"] = d - ref
        os_ = smooth(cyc[f"offA2_A{a}"])
        cyc[f"lab_A2_A{a}"] = np.where(os_.isna(), np.nan, (os_ > thr_A2).astype(float))
    return cyc


def save_geometry(cfg, geoms: dict):
    p = cfg["paths"]["cache"] / "d1_geometry.json"
    p.write_text(json.dumps(geoms, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
