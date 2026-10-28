"""D3 (IDLab UGent UWB Positioning Datasets, HF: iwine-ugent/uwb-positioning-datasets) 구조 조사.

입력 (읽기만 함): data_ext/D3_idlab/
    hf_repo/README.md                 HF 데이터셋 카드
    _manifest.json                    다운로드 목록 (크기/sha256/건너뛴 파일)
    TWR_ds1, TWR_ds2                  ADS-TWR, 이동 로봇 (Nextcloud 공유에서 받음)
    TDoA_Racks, TDoA_Tour, TDoA_Random, TDoA_Grid, TDoA_Racks2   TDoA, 이동 로봇
    PHY_settings                      사무실 노드 간 TWR (메타 + 소형 샘플 1개만)
사용법: python -I 20_d3_structure.py [--root <D3 dir>] [--out <out dir>]
출력: analysis/out/d3/
    20_d3_structure.txt      전체 로그
    d3_files.csv             파일 목록·크기·행수
    d3_columns.csv           파일별 열·dtype·결측률·고유값수·예시
    d3_heads/*.csv           파일별 첫 5행 (긴 문자열은 잘라냄)
    d3_recordings.csv        녹화별 요약
    d3_anchor_stats.csv      녹화×앵커 통계 (관측률, 샘플링 주기, 전력, GT 기반 잔차)
    d3_anchor_coords.csv     앵커 좌표 파일
    d3_summary.json          위 요약 (기계 판독용)
CIR 은 길이·형식 확인과 일부 행 파싱만 한다 (특징량 계산 안 함).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REC_TWR = ["TWR_ds1", "TWR_ds2"]
REC_TDOA = ["TDoA_Racks", "TDoA_Tour", "TDoA_Random", "TDoA_Grid", "TDoA_Racks2"]
README_ENV = {"TDoA_Racks": "NLOS", "TDoA_Tour": "LOS", "TDoA_Random": "LOS", "TDoA_Grid": "LOS",
              "TDoA_Racks2": "NLOS"}  # HF README 표 (녹화 단위 라벨)
C = 299_792_458.0
DW_TICK = 1.0 / (499.2e6 * 128)  # DW1000 timestamp 단위 (s)
LABEL_PAT = re.compile(r"(^|_)(n?los|label|class|env)", re.I)


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        enc = sys.__stdout__.encoding or "utf-8"
        sys.__stdout__.write(s.encode(enc, errors="replace").decode(enc))
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def hdr(t, ch="="):
    print("\n" + ch * 100 + f"\n{t}\n" + ch * 100)


def q(x, ps=(5, 25, 50, 75, 95)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    d = {f"p{p}": float(np.percentile(x, p)) for p in ps}
    d.update(n=int(x.size), mean=float(x.mean()), min=float(x.min()), max=float(x.max()))
    return d


def fmtq(d, nd=3):
    return " ".join(f"{k}={v:.{nd}f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def trunc_df(df, n=60):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).map(lambda s: s if len(s) <= n else s[:n] + f"...(len={len(s)})")
    return out


def col_table(rec, fname, df, nrows_total):
    rows = []
    for c in df.columns:
        s = df[c]
        ex = s.dropna()
        ex = str(ex.iloc[0]) if len(ex) else ""
        rows.append(dict(recording=rec, file=fname, column=c, dtype=str(s.dtype),
                         missing_ratio=float(s.isna().mean()) if len(s) else np.nan,
                         n_unique=int(s.nunique(dropna=True)), example=ex[:60], rows_used=len(df),
                         rows_total=nrows_total))
    return rows


def read_no_cir(path, extra_skip=()):
    """CSV 를 cir 열(들)을 빼고 읽는다. 중복 열 이름은 pandas 규칙(u, u.1, ...)으로 바뀐다."""
    cols = list(pd.read_csv(path, nrows=0).columns)
    keep = [i for i, c in enumerate(cols) if c.lower() != "cir" and c not in extra_skip]
    df = pd.read_csv(path, usecols=keep, low_memory=False)
    return df, cols


def cir_info(path, offset_col=None, pp_col=None, fp_col=None, max_parse=300):
    """cir 열: hex 문자열 길이 분포, int16 LE I/Q 로 해석했을 때 |CIR| 최대 위치 vs ppindex-offset."""
    lens, parsed = [], []
    want = ["cir"] + [c for c in (offset_col, pp_col, fp_col) if c]
    for ch in pd.read_csv(path, usecols=lambda c: c in want, chunksize=20000, low_memory=False):
        s = ch["cir"].astype(str)
        lens.append(s.str.len().to_numpy())
        if len(parsed) < max_parse:
            for i in range(min(max_parse - len(parsed), len(ch))):
                h = s.iloc[i]
                if len(h) % 8 or not re.fullmatch(r"[0-9a-fA-F]+", h):
                    parsed.append(dict(ok=False))
                    continue
                v = np.frombuffer(bytes.fromhex(h), dtype="<i2").astype(float)
                z = v[0::2] + 1j * v[1::2]
                r = dict(ok=True, n=len(z), argmax=int(np.argmax(np.abs(z))))
                if pp_col:
                    r["pp"] = float(ch[pp_col].iloc[i])
                if fp_col:
                    r["fp"] = float(ch[fp_col].iloc[i])
                if offset_col:
                    r["off"] = float(ch[offset_col].iloc[i])
                parsed.append(r)
    L = np.concatenate(lens)
    P = pd.DataFrame(parsed)
    out = dict(hex_len=q(L, (0, 50, 100)), n_complex_mode=int(pd.Series(L // 8).mode().iloc[0]),
               ratio_n_complex_eq_300=float((L == 2400).mean()), n_rows=int(L.size),
               parse_ok_ratio=float(P["ok"].mean()) if len(P) else None)
    if len(P) and P["ok"].any():
        P = P[P["ok"]]
        if "pp" in P and "off" in P:
            out["argmax_eq_pp_minus_offset_ratio"] = float((P["argmax"] == (P["pp"] - P["off"])).mean())
            out["fp_minus_offset_median"] = float((P["fp"] - P["off"]).median())
        elif "pp" in P:
            d = P["pp"] - P["argmax"]
            out["pp_minus_argmax_mode"] = float(d.mode().iloc[0])
            out["pp_minus_argmax_mode_ratio"] = float((d == d.mode().iloc[0]).mean())
    return out


def count_rows(path):
    n = 0
    for ch in pd.read_csv(path, usecols=[0], chunksize=200000, low_memory=False):
        n += len(ch)
    return n


def gt_summary(gt):
    t = gt["Logtime"].to_numpy()
    P = gt[["x", "y", "z"]].to_numpy(float)
    dt = np.diff(t)
    # 0.5 s 간격 속도
    tt = np.arange(t[0], t[-1] - 0.5, 0.1)
    p0 = np.column_stack([np.interp(tt, t, P[:, k]) for k in range(3)])
    p1 = np.column_stack([np.interp(tt + 0.5, t, P[:, k]) for k in range(3)])
    v = np.linalg.norm((p1 - p0)[:, :2], axis=1) / 0.5
    path = float(np.linalg.norm(np.diff(P[::10, :2], axis=0), axis=1).sum())
    return dict(rows=len(gt), t0=float(t[0]), t1=float(t[-1]), duration_s=float(t[-1] - t[0]),
                dt=q(dt, (1, 50, 99)), x=q(P[:, 0], (0, 50, 100)), y=q(P[:, 1], (0, 50, 100)),
                z=q(P[:, 2], (0, 50, 100)), speed_xy=q(v, (5, 50, 95)),
                moving_ratio_v_gt_0p02=float((v > 0.02).mean()), path_len_xy_m=path,
                missing=float(gt[["x", "y", "z"]].isna().any(axis=1).mean()))


def interp_gt(gt, t, max_gap=0.05):
    """GT 를 UWB 시각 t 에 선형보간. inside = GT 범위 안이고 가장 가까운 GT 샘플이 max_gap(s) 이내."""
    tg = gt["Logtime"].to_numpy()
    P = np.column_stack([np.interp(t, tg, gt[k].to_numpy(float)) for k in "xyz"])
    i = np.clip(np.searchsorted(tg, t), 1, len(tg) - 1)
    near = np.minimum(np.abs(t - tg[i - 1]), np.abs(tg[i] - t))
    inside = (t >= tg[0]) & (t <= tg[-1]) & (near <= max_gap)
    return P, inside


def gt_gaps(gt, thr=0.5):
    tg = gt["Logtime"].to_numpy()
    d = np.diff(tg)
    k = np.where(d > thr)[0]
    return [(float(tg[j]), float(d[j])) for j in k]


def wrap40(x):
    """40-bit DW1000 타임스탬프 차이를 [-2^39, 2^39) 로."""
    M = 1 << 40
    return ((x + (M >> 1)) % M) - (M >> 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data_ext" / "D3_idlab"))
    ap.add_argument("--out", default=str(HERE / "out" / "d3"))
    a = ap.parse_args()
    D3, OUT = Path(a.root), Path(a.out)
    (OUT / "d3_heads").mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUT / "20_d3_structure.txt")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)

    summary = {"root": str(D3)}
    col_rows, file_rows, rec_rows, anc_rows = [], [], [], []

    # ------------------------------------------------------------------ 0. manifest / README
    hdr("0. 다운로드 목록 (_manifest.json) 과 HF README")
    man = json.loads((D3 / "_manifest.json").read_text(encoding="utf-8"))
    dl = pd.DataFrame(man["downloaded"])
    sk = pd.DataFrame(man["skipped"])
    print(f"downloaded files={len(dl)} bytes={dl.local_size.sum():,} all size_ok={bool(dl.size_ok.all())}")
    print(dl.groupby("dataset").agg(files=("path", "size"), bytes=("local_size", "sum")).to_string())
    print(f"skipped files={len(sk)} bytes={sk['size'].sum():,}")
    sk["top"] = sk["path"].str.split("/").str[1]
    print(sk.groupby(["dataset", "top"]).agg(files=("path", "size"), bytes=("size", "sum")).to_string())
    print("remote checksum 제공:", bool(dl["remote_checksum"].notna().any()))
    readme = (D3 / "hf_repo" / "README.md").read_text(encoding="utf-8")
    m = re.search(r"\| Name .*?\n((?:\|.*\n)+)", readme)
    if m:
        print("README TDoA 표:\n" + m.group(0))
    summary["download"] = dict(n=len(dl), bytes=int(dl.local_size.sum()), n_skipped=len(sk),
                               bytes_skipped=int(sk["size"].sum()))

    # ------------------------------------------------------------------ 1. anchor coordinates
    hdr("1. 앵커 좌표 파일 AnchorsIIoT-28march23.csv")
    afiles = sorted(D3.glob("*/AnchorsIIoT-28march23.csv"))
    hashes = {p.parent.name: pd.util.hash_pandas_object(pd.read_csv(p)).sum() for p in afiles}
    print("녹화별 사본 동일:", len(set(hashes.values())) == 1, f"(n={len(afiles)})")
    anc = pd.read_csv(afiles[0])
    print(anc.to_string())
    print("x,y,z 범위:", anc[["x", "y", "z"]].agg(["min", "max"]).to_dict())
    anc.to_csv(OUT / "d3_anchor_coords.csv", index=False)
    AXYZ = {r.hex: np.array([r.x, r.y, r.z], float) / 1000.0 for r in anc.itertuples()}  # mm -> m (단위는 아래 TWR 잔차로 확인)
    summary["anchors_file"] = dict(n=len(anc), ids=anc.hex.tolist())

    # ------------------------------------------------------------------ 2. recordings
    for rec in REC_TWR + REC_TDOA:
        R = D3 / rec
        if not R.exists():
            print("없음:", rec)
            continue
        kind = "TWR" if rec in REC_TWR else "TDoA"
        hdr(f"2. 녹화 {rec}  ({kind})")
        files = sorted(p for p in R.iterdir() if p.is_file())
        for p in files:
            file_rows.append(dict(recording=rec, file=p.name, bytes=p.stat().st_size, rows=np.nan))
        print("파일:", ", ".join(f"{p.name}({p.stat().st_size/1e6:.1f}MB)" for p in files))
        info = (R / "dataset_information.txt").read_text(encoding="utf-8", errors="replace")
        src = re.search(r"Information file of dataset (\S+)", info)
        print("원본 bag:", src.group(1) if src else "?")
        recs = dict(recording=rec, kind=kind, readme_env_label=README_ENV.get(rec, ""),
                    source_bag=src.group(1) if src else "")

        # ---- processed.csv
        P, pcols = read_no_cir(R / "processed.csv")
        nrows = len(P)
        print(f"\n[processed.csv] rows={nrows} cols={len(pcols)} : {pcols}")
        print("dtypes / missing ratio:")
        print(pd.DataFrame({"dtype": P.dtypes.astype(str), "missing": P.isna().mean().round(4),
                            "n_unique": P.nunique()}).T.to_string())
        h = pd.read_csv(R / "processed.csv", nrows=5, low_memory=False)
        trunc_df(h).to_csv(OUT / "d3_heads" / f"{rec}__processed.csv", index=False)
        print(trunc_df(h, 25).to_string())
        col_rows += col_table(rec, "processed.csv", P, nrows)
        col_rows.append(dict(recording=rec, file="processed.csv", column="cir", dtype="object(hex str)",
                             missing_ratio=np.nan, n_unique=np.nan, example="(hex)", rows_used=0, rows_total=nrows))
        labcols = [c for c in pcols if LABEL_PAT.search(c)]
        print("LOS/NLOS·label 후보 열:", labcols or "없음")
        recs.update(processed_rows=nrows, processed_cols=len(pcols), label_cols=";".join(labcols))

        # radio settings
        settings = {c: sorted(P[c].dropna().unique().tolist()) for c in
                    ["channel", "bitrate", "preamble", "PRF", "prf", "preamble_length", "tx_power", "tag", "version"]
                    if c in P}
        print("무선 설정 (processed):", settings)
        recs["settings_processed"] = json.dumps(settings, default=str)

        # time
        t = P["Logtime"].to_numpy(float)
        dts = np.diff(np.sort(t))
        print(f"Logtime: epoch s (UTC {pd.to_datetime(t.min(), unit='s')} ~ {pd.to_datetime(t.max(), unit='s')}),"
              f" span={t.max()-t.min():.1f}s, monotonic={P['Logtime'].is_monotonic_increasing},"
              f" min>0 diff={dts[dts>0].min():.2e}s, float64 ulp@t={np.spacing(t.max()):.1e}s")
        recs.update(t0_utc=str(pd.to_datetime(t.min(), unit="s")), uwb_span_s=float(t.max() - t.min()),
                    logtime_monotonic=bool(P["Logtime"].is_monotonic_increasing))
        if "UWB_time" in P:
            print("UWB_time 예:", P["UWB_time"].head(3).tolist(), " ts 예:", P["ts"].head(3).tolist() if "ts" in P else "-")
            if "ts" in P:
                # UWB_time 'a:b:c:d' 를 s,ms,us,ns 로 읽으면 ts*DW_TICK 과 같은지
                parts = P["UWB_time"].astype(str).str.strip().str.split(":", expand=True).astype(float)
                if parts.shape[1] == 4:
                    sec = parts[0] + parts[1] / 1e3 + parts[2] / 1e6 + parts[3] / 1e9
                    err = np.abs(sec - P["ts"] * DW_TICK)
                    print(f"UWB_time(s:ms:us:ns) vs ts*15.65ps: |diff| p99={np.percentile(err,99):.2e}s")
                print("ts 범위 ticks:", int(P["ts"].min()), int(P["ts"].max()), " = s:",
                      P["ts"].min() * DW_TICK, P["ts"].max() * DW_TICK, " (40-bit wrap=17.2s)")

        # superframe
        sf = P["superframe"]
        gsf = P.groupby("superframe")
        n_sf = sf.nunique()
        a_per_sf = gsf["anchor"].nunique()
        sf_t = gsf["Logtime"].median().sort_index()
        dsf = np.diff(sf_t.index.to_numpy(float))
        dts_sf = np.diff(sf_t.to_numpy())
        per_inc = dts_sf / np.where(dsf > 0, dsf, np.nan)
        print(f"superframe: n_unique={n_sf} range=[{sf.min()},{sf.max()}] monotonic(행순)={sf.is_monotonic_increasing}"
              f" 연속성: 관측 sf/전체 범위={n_sf/(sf.max()-sf.min()+1):.3f}")
        print(f"  앵커/superframe: {fmtq(q(a_per_sf,(0,25,50,75,100)),2)}")
        print("  분포:", a_per_sf.value_counts().sort_index().to_dict())
        print(f"  superframe 1증가당 시간(s): {fmtq(q(per_inc,(5,50,95)),4)};"
              f" 관측 superframe 간격(s): {fmtq(q(dts_sf,(5,50,95,100)),3)}")
        jumps = int(((dsf > 100) & (dts_sf < 1.0)).sum())
        print(f"  superframe 번호가 >100 뛰는데 시간 간격 <1s 인 경우: {jumps}회;"
              f" 관측 superframe 수/길이 = {n_sf/(t.max()-t.min()):.2f} 개/s")
        sft = np.sort(sf_t.to_numpy())
        par = int((np.diff(sft) < 0.02).sum())
        one = int((a_per_sf == 1).sum())
        print(f"  Logtime 순으로 이웃 superframe 간격 <0.02s (시간상 겹치는 superframe 열): {par}개;"
              f" 앵커 1개뿐인 superframe: {one}개 ({one/n_sf:.3f})")
        recs.update(sf_jumps_gt100_lt1s=jumps, sf_per_s=float(n_sf / (t.max() - t.min())),
                    sf_parallel_lt20ms=par, sf_single_anchor_ratio=one / n_sf)
        if "sync" in P:
            print(f"  sync 열: 범위 [{P['sync'].min()},{P['sync'].max()}], superframe%256==sync 비율="
                  f"{((P['superframe'] % 256) == P['sync']).mean():.3f}")
        recs.update(n_superframes=int(n_sf), sf_min=float(sf.min()), sf_max=float(sf.max()),
                    sf_kept_ratio=float(n_sf / (sf.max() - sf.min() + 1)),
                    anchors_per_sf_median=float(a_per_sf.median()), anchors_per_sf_mean=float(a_per_sf.mean()),
                    sf_increment_s_median=float(np.nanmedian(per_inc)), sf_obs_gap_s_median=float(np.median(dts_sf)))
        dup = P.duplicated(["superframe", "anchor"]).sum()
        print("  (superframe, anchor) 중복 행:", int(dup))
        if "ts" in P:
            spread = gsf["ts"].agg(lambda x: x.max() - x.min())
            print(f"  superframe 내 ts max-min (raw, wrap 미보정) ticks: {fmtq(q(spread,(50,95,100)),0)}"
                  f" -> m 환산 p50={np.median(spread)*DW_TICK*C:.2f} max={spread.max()*DW_TICK*C:.2f}")
            recs["ts_spread_m_p50"] = float(np.median(spread) * DW_TICK * C)

        # anchors
        ids = sorted(P["anchor"].astype(str).unique())
        miss = [i for i in ids if i not in AXYZ]
        print(f"앵커 ID ({len(ids)}): {ids}; 좌표 파일에 없는 ID: {miss or '없음'}")
        recs.update(n_anchors=len(ids), anchor_ids=";".join(ids), anchors_without_coords=";".join(miss))
        if "anchorid" in P:
            mp = P.groupby("anchor")["anchorid"].unique().apply(list).to_dict()
            print("anchor(hex) -> anchorid:", mp)

        # power (센티널 -2147483648 등 비정상 값은 NaN 처리 후 통계)
        for c in ["rx_power", "fp_power"]:
            bad = (P[c] < -200) | (P[c] > 0)
            print(f"{c} 비정상값(<-200 또는 >0) 행: {int(bad.sum())} ({bad.mean():.4f}) 예: {P.loc[bad, c].unique()[:3]}")
            recs[f"{c}_invalid_rows"] = int(bad.sum())
            P.loc[bad, c] = np.nan
            print(f"{c} (dBm, 유효값): {fmtq(q(P[c],(1,50,99)),2)}")
        dpow = P["rx_power"] - P["fp_power"]
        print(f"rx_power-fp_power (dB): {fmtq(q(dpow,(5,50,95)),2)};  >6dB 비율={(dpow>6).mean():.3f}, >10dB={(dpow>10).mean():.3f}")
        recs.update(rx_power_med=float(P.rx_power.median()), fp_power_med=float(P.fp_power.median()),
                    rx_minus_fp_med=float(dpow.median()), rx_minus_fp_gt6_ratio=float((dpow > 6).mean()))

        # ---- GT
        gt = pd.read_csv(R / "gt.csv")
        print(f"\n[gt.csv] cols={list(gt.columns)} dtypes={gt.dtypes.astype(str).to_dict()}")
        print(gt.head(5).to_string())
        trunc_df(gt.head(5)).to_csv(OUT / "d3_heads" / f"{rec}__gt.csv", index=False)
        col_rows += col_table(rec, "gt.csv", gt, len(gt))
        G = gt_summary(gt)
        print(f"rows={G['rows']} duration={G['duration_s']:.1f}s dt: {fmtq(G['dt'],4)}")
        print(f"x(m): {fmtq(G['x'],2)} | y(m): {fmtq(G['y'],2)} | z(m): {fmtq(G['z'],3)}")
        print(f"수평 속도(0.5s 차분, m/s): {fmtq(G['speed_xy'],3)}; v>0.02 비율={G['moving_ratio_v_gt_0p02']:.3f};"
              f" 경로길이(xy)={G['path_len_xy_m']:.1f} m; 결측={G['missing']}")
        gaps = gt_gaps(gt)
        print(f"GT 공백(>0.5s): {len(gaps)}개, 합계 {sum(g[1] for g in gaps):.1f}s; 큰 것:",
              [(round(a - G['t0'], 1), round(b, 1)) for a, b in sorted(gaps, key=lambda x: -x[1])[:5]],
              "(시작 기준 s, 길이 s)")
        Pg, inside = interp_gt(gt, t)
        print(f"UWB 행 중 GT 유효(범위 안 & 최근접 GT<=0.05s): {inside.mean():.4f};  GT 시작-UWB 시작={t.min()-G['t0']:.3f}s,"
              f" GT 끝-UWB 끝={G['t1']-t.max():.3f}s")
        recs.update(gt_gaps_n=len(gaps), gt_gaps_total_s=float(sum(g[1] for g in gaps)))
        recs.update(gt_rows=G["rows"], gt_duration_s=G["duration_s"], gt_dt_median=G["dt"]["p50"],
                    gt_speed_med=G["speed_xy"]["p50"], gt_speed_p95=G["speed_xy"]["p95"],
                    gt_moving_ratio=G["moving_ratio_v_gt_0p02"], gt_path_len_m=G["path_len_xy_m"],
                    gt_z_med=G["z"]["p50"], uwb_in_gt_ratio=float(inside.mean()))

        # ---- per-anchor stats (+ GT 기반 잔차)
        P = P.assign(_in=inside, _gx=Pg[:, 0], _gy=Pg[:, 1], _gz=Pg[:, 2])
        A = np.array([AXYZ.get(str(i), [np.nan] * 3) for i in P["anchor"].astype(str)])
        dtrue = np.linalg.norm(Pg - A, axis=1)
        P["_dtrue"] = dtrue
        if kind == "TWR":
            P["_res"] = P["distance"] / 1000.0 - dtrue
            r = P.loc[P._in, "_res"]
            print(f"\nTWR 잔차 distance/1000 - |GT-anchor| (m, 앵커 좌표 mm->m 가정): {fmtq(q(r,(5,25,50,75,95)),3)};"
                  f" |res|>0.5m 비율={(r.abs()>0.5).mean():.3f}")
            recs.update(twr_res_med=float(r.median()), twr_res_iqr=float(r.quantile(.75) - r.quantile(.25)),
                        twr_res_abs_gt0p5=float((r.abs() > 0.5).mean()))
        else:
            # (1) 시간축 진단: superframe 내 ts 중앙값 대비 차이(40-bit wrap 고려). |off|>1e5 tick(≈1.6us≈480m) = 다른 시간축 행
            ts = P["ts"].to_numpy(np.int64)
            med = P.groupby("superframe")["ts"].transform("median").round().to_numpy(np.int64)
            off = wrap40(ts - med)
            far = np.abs(off) > 100_000
            P["_far"] = far
            print(f"\nTDoA 시간축: superframe 중앙 ts 와 |차이|>1e5 tick 인 행 비율={far.mean():.4f}")
            fa = P.groupby("anchor")["_far"].mean()
            print("  앵커별 비율(>0 만):", fa[fa > 0].round(3).to_dict())
            recs["tdoa_far_row_ratio"] = float(far.mean())
            # (2) 같은 시간축 행만: r = (ts - sf중앙 ts)*c - (d_true - sf중앙 d_true), 앵커별 상수 b_a 를 빼고 남는 산포
            Q = P[~P._far].copy()
            Q["_off"] = wrap40(Q["ts"].to_numpy(np.int64)
                               - Q.groupby("superframe")["ts"].transform("median").round().to_numpy(np.int64))
            Q["_r"] = Q["_off"] * DW_TICK * C - Q["_dtrue"]
            Q = Q[Q._in & (Q.groupby("superframe")["anchor"].transform("size") >= 3)].copy()
            # r = tau_sf (송신시각) + b_a (앵커 상수) + e : 교대 중앙값 추정 (superframe 마다 앵커 구성이 달라서)
            b = pd.Series(0.0, index=Q["anchor"].unique())
            for _ in range(15):
                tau = (Q["_r"] - Q["anchor"].map(b)).groupby(Q["superframe"]).transform("median")
                b = (Q["_r"] - tau).groupby(Q["anchor"]).median()
                b = b - b.median()
            Q["_e"] = Q["_r"] - tau - Q["anchor"].map(b)
            e = Q["_e"]
            print(f"  (ts차*c - 실제거리차) 앵커별 상수 오프셋 b_a (m): {b.round(2).to_dict()}")
            print(f"  b_a 제거 후 잔차 (m, sf당 앵커>=3, GT 유효): {fmtq(q(e,(5,25,50,75,95)),3)};"
                  f" |e|>0.5m 비율={(e.abs()>0.5).mean():.3f}")
            recs.update(tdoa_res_iqr=float(e.quantile(.75) - e.quantile(.25)),
                        tdoa_res_abs_gt0p5=float((e.abs() > 0.5).mean()))
            P["_r"] = np.nan
            P.loc[Q.index, "_r"] = Q["_e"]
        sf_all = P["superframe"].nunique()
        for aid, g in P.groupby(P["anchor"].astype(str)):
            g = g.sort_values("superframe")
            dt_a = np.diff(g["Logtime"].to_numpy())
            row = dict(recording=rec, kind=kind, anchor=aid, n=len(g),
                       obs_ratio_of_sf=g["superframe"].nunique() / sf_all,
                       dt_med=float(np.median(dt_a)) if len(dt_a) else np.nan,
                       dt_p90=float(np.percentile(dt_a, 90)) if len(dt_a) else np.nan,
                       dt_max=float(dt_a.max()) if len(dt_a) else np.nan,
                       rx_med=float(g.rx_power.median()), fp_med=float(g.fp_power.median()),
                       rx_minus_fp_med=float((g.rx_power - g.fp_power).median()),
                       dtrue_med=float(g._dtrue.median()),
                       ax=AXYZ.get(aid, [np.nan] * 3)[0], ay=AXYZ.get(aid, [np.nan] * 3)[1],
                       az=AXYZ.get(aid, [np.nan] * 3)[2])
            col = "_res" if kind == "TWR" else "_r"
            gg = g[g._in & g[col].notna()]
            if kind == "TDoA":
                row["far_ratio"] = float(g["_far"].mean())
                row["n_superframes_ge3_anchors"] = int((g["superframe"].map(P.groupby("superframe")["anchor"].size()) >= 3).sum())
            if len(gg):
                row.update(res_med=float(gg[col].median()), res_iqr=float(gg[col].quantile(.75) - gg[col].quantile(.25)),
                           res_abs_gt0p5=float((gg[col].abs() > 0.5).mean()))
            anc_rows.append(row)
        AS = pd.DataFrame([x for x in anc_rows if x["recording"] == rec])
        print("앵커별 (n, 관측률, Logtime 간격 s, 전력 dBm, GT 잔차 m):")
        print(AS.drop(columns=["recording", "kind"]).round(3).to_string(index=False))
        recs.update(anchor_dt_med_median=float(AS.dt_med.median()),
                    anchor_obs_ratio_median=float(AS.obs_ratio_of_sf.median()))

        # ---- CIR
        full = R / "processed_full.csv"
        if full.exists():
            ci = cir_info(full, offset_col="cir_offset", pp_col="pp_index" if kind == "TDoA" else "ppindex",
                          fp_col="fp_index")
        else:
            ci = cir_info(R / "processed.csv", pp_col="pp_index" if "pp_index" in pcols else "ppindex")
        print(f"\nCIR: {ci}")
        recs.update(cir_n_complex_mode=ci["n_complex_mode"], cir_ratio_300=ci["ratio_n_complex_eq_300"],
                    cir_parse_ok=ci["parse_ok_ratio"], cir_check=json.dumps({k: v for k, v in ci.items()
                                                                              if k.startswith(("argmax", "pp_", "fp_"))}))

        # ---- processed_full.csv
        if full.exists():
            F, fcols = read_no_cir(full)
            print(f"\n[processed_full.csv] rows={len(F)} cols={len(fcols)}")
            print("열:", fcols)
            hf = pd.read_csv(full, nrows=5, low_memory=False)
            trunc_df(hf).to_csv(OUT / "d3_heads" / f"{rec}__processed_full.csv", index=False)
            col_rows += col_table(rec, "processed_full.csv", F, len(F))
            st = {c: sorted(F[c].dropna().unique().tolist())[:8] for c in
                  ["mode", "channel", "bitrate", "preamble", "preamble_length", "PRF", "prf", "tx_power",
                   "cir_length", "cir_offset", "antenna_delay", "model", "revision"] if c in F}
            print("설정 (processed_full):", st)
            recs["settings_full"] = json.dumps(st, default=str)
            nanc = [c for c in F.columns if F[c].isna().all()]
            print("전부 결측인 열:", nanc)
            same = len(F) == nrows
            print("processed 와 행수 동일:", same)
            for p in file_rows:
                if p["recording"] == rec and p["file"] == "processed_full.csv":
                    p["rows"] = len(F)
        else:
            print("\nprocessed_full.csv 없음")
        for p in file_rows:
            if p["recording"] == rec and p["file"] == "processed.csv":
                p["rows"] = nrows
            if p["recording"] == rec and p["file"] == "gt.csv":
                p["rows"] = len(gt)

        # ---- NUC.csv (원시 로그)
        nuc = R / "NUC.csv"
        if nuc.exists():
            n, nfail, norx, hosts = 0, 0, 0, {}
            for ch in pd.read_csv(nuc, usecols=["Time", "data"], chunksize=100000, low_memory=False):
                n += len(ch)
                d = ch["data"].astype(str)
                nfail += int(d.str.contains("FAILED").sum())
                norx += int(d.str.contains("no_rx|No UWB RX", regex=True).sum())
                hv = d.str.extract(r"^(nuc[^_@\s]+)")[0].value_counts()
                for k, v in hv.items():
                    hosts[k] = hosts.get(k, 0) + int(v)
            hn = pd.read_csv(nuc, nrows=5)
            trunc_df(hn).to_csv(OUT / "d3_heads" / f"{rec}__NUC.csv", index=False)
            col_rows += col_table(rec, "NUC.csv", hn, n)
            print(f"\n[NUC.csv] rows={n} cols={list(hn.columns)}; 'FAILED' 행={nfail} ({nfail/n:.3f}),"
                  f" no_rx 행={norx}; host 수={len(hosts)}: {dict(sorted(hosts.items()))}")
            recs.update(nuc_rows=n, nuc_failed_ratio=nfail / n, nuc_hosts=len(hosts))
            for p in file_rows:
                if p["recording"] == rec and p["file"] == "NUC.csv":
                    p["rows"] = n

        # ---- imu.csv
        imu = R / "imu.csv"
        if imu.exists():
            I = pd.read_csv(imu, usecols=["Time", "header.stamp.secs", "header.stamp.nsecs", "header.frame_id"])
            ti = I["header.stamp.secs"] + I["header.stamp.nsecs"] * 1e-9
            hi = pd.read_csv(imu, nrows=5)
            trunc_df(hi).to_csv(OUT / "d3_heads" / f"{rec}__imu.csv", index=False)
            col_rows += col_table(rec, "imu.csv", hi, len(I))
            print(f"\n[imu.csv] rows={len(I)} cols={len(hi.columns)} frame={I['header.frame_id'].unique()[:3]}"
                  f" dt median={np.median(np.diff(ti)):.4f}s span={ti.max()-ti.min():.1f}s")
            recs.update(imu_rows=len(I), imu_dt_med=float(np.median(np.diff(ti))))
            for p in file_rows:
                if p["recording"] == rec and p["file"] == "imu.csv":
                    p["rows"] = len(I)
        rec_rows.append(recs)

    # ------------------------------------------------------------------ 3. PHY settings (subset)
    R = D3 / "PHY_settings"
    if R.exists():
        hdr("3. PHY_settings (사무실, 노드 간 ADS-TWR) - 메타 + 샘플 1개만")
        for p in sorted(R.rglob("*")):
            if p.is_file():
                file_rows.append(dict(recording="PHY_settings", file=str(p.relative_to(R)), bytes=p.stat().st_size,
                                      rows=np.nan))
        try:
            xl = pd.read_excel(R / "9thFloorAnchorsLayout.xlsx", sheet_name=None)
            for k, v in xl.items():
                print(f"[9thFloorAnchorsLayout.xlsx] sheet={k} shape={v.shape}\n{v.head(20).to_string()}")
        except Exception as e:  # noqa
            print("xlsx 읽기 실패:", repr(e))
        g9 = pd.read_csv(R / "groundtruth9.csv", header=None, sep=r",\s*", engine="python")
        print(f"[groundtruth9.csv] (헤더 없음) shape={g9.shape}\n{g9.to_string(index=False)}")
        sp = R / "Processed files (csv)" / "office_lab_node_04_short.csv"
        S = pd.read_csv(sp, low_memory=False)
        print(f"[{sp.name}] rows={len(S)} cols={len(S.columns)}")
        print("열:", list(S.columns))
        big = [c for c in S.columns if S[c].dtype == object and S[c].astype(str).str.len().max() > 200]
        print("긴 문자열 열(CIR 등):", big, [int(S[c].astype(str).str.len().median()) for c in big])
        trunc_df(S.head(5)).to_csv(OUT / "d3_heads" / "PHY_settings__office_lab_node_04_short.csv", index=False)
        col_rows += col_table("PHY_settings", sp.name, S, len(S))
        for c in ["anchor_ID", "anchor", "channel", "bitrate", "preamble", "prf", "txpower"]:
            if c in S:
                print(f"  {c}: {sorted(S[c].dropna().unique().tolist())[:20]}")
        tcols = [c for c in S.columns if re.search(r"time|slot|stamp", c, re.I)]
        print("시간 관련 열:", tcols, S[tcols].head(3).to_dict("list") if tcols else "")
        print("결측률>0 열:", S.isna().mean()[S.isna().mean() > 0].round(3).to_dict())

    # ------------------------------------------------------------------ save
    hdr("4. 녹화 요약")
    RR = pd.DataFrame(rec_rows)
    show = ["recording", "kind", "readme_env_label", "t0_utc", "gt_duration_s", "uwb_span_s", "processed_rows",
            "n_superframes", "sf_per_s", "sf_kept_ratio", "sf_jumps_gt100_lt1s", "sf_parallel_lt20ms",
            "sf_single_anchor_ratio", "sf_increment_s_median",
            "n_anchors", "anchors_per_sf_median", "anchor_dt_med_median", "anchor_obs_ratio_median",
            "gt_dt_median", "gt_gaps_n", "gt_gaps_total_s", "uwb_in_gt_ratio", "gt_speed_med", "gt_moving_ratio",
            "gt_path_len_m", "rx_power_invalid_rows", "rx_minus_fp_med", "rx_minus_fp_gt6_ratio", "twr_res_med",
            "twr_res_abs_gt0p5", "tdoa_far_row_ratio", "tdoa_res_iqr", "tdoa_res_abs_gt0p5", "label_cols",
            "cir_n_complex_mode", "cir_ratio_300", "nuc_failed_ratio"]
    print(RR[[c for c in show if c in RR]].round(4).T.to_string())
    RR.to_csv(OUT / "d3_recordings.csv", index=False)
    pd.DataFrame(anc_rows).to_csv(OUT / "d3_anchor_stats.csv", index=False)
    pd.DataFrame(col_rows).to_csv(OUT / "d3_columns.csv", index=False)
    pd.DataFrame(file_rows).to_csv(OUT / "d3_files.csv", index=False)
    summary["recordings"] = rec_rows
    (OUT / "d3_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print("\n저장:", OUT)


if __name__ == "__main__":
    main()
