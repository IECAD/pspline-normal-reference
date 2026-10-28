"""D3 (IDLab UGent 2025, TDoA 녹화 5개) 로더. 원본 data_ext/D3_idlab/ 은 읽기만 한다.

구조 근거: analysis/out/d3/structure.md
  - processed.csv: 1행 = 앵커 1개가 받은 태그 blink 1개. ts = DW1000 RX 타임스탬프 (15.65 ps, 40-bit 순환)
  - gt.csv: 태그 위치 (m, 100 Hz). AnchorsIIoT-28march23.csv: 앵커 좌표 (mm)
처리:
  1. superframe 마다 중앙 ts 에서 sync_tol_ticks 보다 먼 행(다른 sync 셀)을 버린다
  2. 시간 격자: superframe 의 Logtime 중앙값을 녹화별 공칭 주기로 반올림해 tidx. 같은 칸이면 앵커가 많은 쪽
  3. 의사거리 p_a = (ts_a - superframe 기준) * tick * c
  4. 앵커 오프셋 b_a: calib 녹화의 학습+검증(정상) 또는 전체(자기 보정)에서 p - r - b - tau 를 교대 중앙값으로 추정
     (r = GT 위치와 앵커 사이 실제 거리, tau = superframe 공통항)
  5. 모델 입력 TDOA_a = p_a - b_a - median_a'(p_a' - b_a')   (GT 불필요, 앵커 3개 이상일 때)
  6. 의사 라벨 잔차 e_a = p_a - r_a - b_a - tau_sf, tau_sf = median_a(p - r - b). lab_R = e > decision_threshold
"""
import numpy as np
import pandas as pd

from .config import ROOT, load_config

TICK = 1.0 / (128 * 499.2e6)
C_AIR = 299702547.0
WRAP = 2 ** 40
USE = ["Logtime", "ts", "superframe", "anchor", "rx_power", "fp_power"]


def _root(cfg):
    return ROOT / cfg["d3"]["root"]


def raw(cfg, rec) -> pd.DataFrame:
    cp = cfg["paths"]["cache"] / "d3" / f"{rec}.parquet"
    cp.parent.mkdir(parents=True, exist_ok=True)
    if cp.exists():
        return pd.read_parquet(cp)
    df = pd.read_csv(_root(cfg) / rec / "processed.csv", usecols=USE)
    df.to_parquet(cp, index=False)
    return df


def anchors(cfg, rec) -> pd.DataFrame:
    a = pd.read_csv(_root(cfg) / rec / "AnchorsIIoT-28march23.csv")
    a[["x", "y", "z"]] = a[["x", "y", "z"]] / 1000.0
    return a.set_index("hex")


def blinks(cfg, rec) -> pd.DataFrame:
    """긴 표: tidx, Logtime(blink), anchor, p(m), rx_power, fp_power, gx, gy, gz, r(m)."""
    d3 = cfg["d3"]
    df = raw(cfg, rec).dropna(subset=["superframe", "ts"]).copy()
    df["rx_power"] = df["rx_power"].where(df["rx_power"] > -1e6)            # INT_MIN 센티널
    med = df.groupby("superframe")["ts"].transform("median")
    dts = (df["ts"] - med + WRAP // 2) % WRAP - WRAP // 2
    df = df[dts.abs() <= d3["sync_tol_ticks"]].copy()
    df["dts"] = dts[df.index]
    df = df.drop_duplicates(["superframe", "anchor"])
    sf = df.groupby("superframe").agg(t=("Logtime", "median"), n=("anchor", "size")).sort_values("t")
    period = float(np.median(np.diff(sf["t"].to_numpy())))
    sf["tidx"] = np.round((sf["t"] - sf["t"].iloc[0]) / period).astype(int)
    sf = sf.sort_values(["tidx", "n"], ascending=[True, False]).drop_duplicates("tidx")
    df = df[df["superframe"].isin(sf.index)].join(sf[["t", "tidx"]], on="superframe")
    df["p"] = df["dts"] * TICK * C_AIR
    gt = pd.read_csv(_root(cfg) / rec / "gt.csv").sort_values("Logtime")
    gtt = gt["Logtime"].to_numpy()
    tq = df["t"].to_numpy()
    j = np.clip(np.searchsorted(gtt, tq), 1, len(gtt) - 1)
    gap = np.minimum(np.abs(gtt[j] - tq), np.abs(gtt[j - 1] - tq))
    for c in ("x", "y", "z"):
        v = np.interp(tq, gtt, gt[c].to_numpy())
        df[f"g{c}"] = np.where(gap <= d3["gt_max_gap_s"], v, np.nan)
    A = anchors(cfg, rec)
    ax, ay, az = (df["anchor"].map(A[c]) for c in ("x", "y", "z"))
    df["r"] = np.sqrt((df["gx"] - ax) ** 2 + (df["gy"] - ay) ** 2 + (df["gz"] - az) ** 2)
    df["rec"], df["period_s"] = rec, period
    return df[["rec", "tidx", "t", "period_s", "anchor", "p", "rx_power", "fp_power", "gx", "gy", "gz", "r"]]


def fit_offsets(b: pd.DataFrame, min_anchors=3, iters=8) -> pd.Series:
    """p - r = b_a + tau_sf + 잡음. 교대 중앙값 (robust), 게이지: b 평균 0."""
    d = b.dropna(subset=["r"]).copy()
    d = d[d.groupby("tidx")["anchor"].transform("size") >= min_anchors]
    d["y"] = d["p"] - d["r"]
    bias = pd.Series(0.0, index=sorted(d["anchor"].unique()))
    for _ in range(iters):
        tau = (d["y"] - d["anchor"].map(bias)).groupby(d["tidx"]).median()
        bias = (d["y"] - d["tidx"].map(tau)).groupby(d["anchor"]).median()
        bias -= bias.mean()
    return bias


def series(cfg, rec, bias: pd.Series) -> pd.DataFrame:
    """넓은 표 (한 행 = tidx). 열: TDOA_{a}, rx_power_{a}, fp_power_{a}, e_{a}, lab_R_{a}, obs_{a}."""
    d3 = cfg["d3"]
    b = blinks(cfg, rec)
    b["pb"] = b["p"] - b["anchor"].map(bias)
    cnt = b.groupby("tidx")["anchor"].transform("size")
    ok = cnt >= d3["min_anchors_per_sf"]
    b["TDOA"] = (b["pb"] - b.groupby("tidx")["pb"].transform("median")).where(ok)
    y = (b["pb"] - b["r"]).where(ok)
    b["e"] = y - y.groupby(b["tidx"]).transform("median")
    b["lab_R"] = np.where(b["e"].isna(), np.nan, (b["e"] > d3["label"]["decision_threshold"]).astype(float))
    b["obs"] = b["TDOA"].notna().astype(float)
    vals = ["TDOA", "rx_power", "fp_power", "e", "lab_R", "obs"]
    anchors_ = sorted(b["anchor"].unique())
    wide = b.pivot_table(index="tidx", columns="anchor", values=vals, aggfunc="first", dropna=False)
    wide.columns = [f"{v}_{a}" for v, a in wide.columns]
    full = pd.RangeIndex(b["tidx"].min(), b["tidx"].max() + 1, name="tidx")
    wide = wide.reindex(index=full, columns=[f"{v}_{a}" for v in vals for a in anchors_])
    for a in anchors_:
        wide[f"obs_{a}"] = wide[f"obs_{a}"].fillna(0.0)
    t0 = b["t"].min()
    wide["t"] = t0 + wide.index.to_numpy() * b["period_s"].iloc[0]
    meta = cfg["d3"]["recordings"][rec]
    wide = wide.reset_index()
    wide["file"], wide["space"], wide["condition"] = rec, "d3", meta["condition"]
    wide["node"], wide["seg"], wide["edge"], wide["normal"] = rec, 0, False, bool(meta["normal"])
    return wide, anchors_


def calibrate_all(cfg):
    """녹화별 앵커 오프셋. 정상 녹화: 자기 학습+검증 몫. 비정상: calib 녹화 값 + 없는 앵커는 자기 전체로 robust 추정."""
    from .windows import assign_split
    out, info = {}, {}
    recs = cfg["d3"]["recordings"]
    for rec, m in recs.items():
        if not m["normal"]:
            continue
        b = blinks(cfg, rec)
        tmp = pd.DataFrame({"tidx": sorted(b["tidx"].unique())})
        tmp["seg"], tmp["edge"], tmp["normal"] = 0, False, True
        tmp["split"] = assign_split(tmp, cfg["split"]["ratios"])
        trval = set(tmp.loc[tmp["split"].isin(["train", "val"]), "tidx"])
        out[rec] = fit_offsets(b[b["tidx"].isin(trval)], cfg["d3"]["min_anchors_per_sf"])
        info[rec] = dict(source="self train+val", anchors=len(out[rec]))
    for rec, m in recs.items():
        if m["normal"]:
            continue
        b = blinks(cfg, rec)
        self_b = fit_offsets(b, cfg["d3"]["min_anchors_per_sf"])
        cal = out.get(m["calib"]) if m["calib"] != rec else None
        if cal is None:
            out[rec], info[rec] = self_b, dict(source="self all (robust)", anchors=len(self_b))
        else:
            common = self_b.index.intersection(cal.index)
            shift = float((self_b[common] - cal[common]).median())   # 게이지 맞춤
            merged = (self_b - shift).copy()
            merged[common] = cal[common]
            out[rec] = merged
            info[rec] = dict(source=f"{m['calib']} train+val ({len(common)}) + self robust ({len(self_b) - len(common)})",
                             anchors=len(merged))
    return out, info


if __name__ == "__main__":
    cfg = load_config()
    offs, info = calibrate_all(cfg)
    for rec in cfg["d3"]["recordings"]:
        w, an = series(cfg, rec, offs[rec])
        e = w.filter(like="e_").to_numpy()
        obs = w.filter(like="obs_").to_numpy()
        print(f"{rec:12s} steps={len(w):6d} anchors={len(an):2d} obs={obs.mean():.2f} "
              f"|e|>0.5={np.nanmean(np.abs(e) > 0.5):.3f} e>0.5={np.nanmean(e > 0.5):.3f} "
              f"IQR(e)={np.nanpercentile(e, 75) - np.nanpercentile(e, 25):.3f} calib={info[rec]['source']}")
