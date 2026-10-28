"""D2 (Bregar 2023) 구조·기초 분석. D1 실험 0 과 같은 관점 (구조, 정보 유무, 기초 통계, 라벨, 정상 데이터 양).

입력: data_ext/D2_bregar/data_set.zip (원본, 읽기만 함). raw_data/environment*/data/<x_y_z>/ch<c>_A<a>.csv
      README: 1행 버전, 2행 열 이름, 3행부터 데이터. 앞 26개 필드 = 메타, 이후 = CIR(복소수) -> CIR 은 읽지 않는다.
사용법: python 12_d2_structure.py            (로컬 zip 전체)
        python 12_d2_structure.py --remote 2  (원격 zip, 환경마다 앞 2개 위치만: 동작 시험용)
출력: analysis/out/d2/12_d2_structure.txt, analysis/out/d2/d2_*.csv, analysis/out/fig/d2_*.png
      캐시: analysis/out/cache/d2_meta.pkl
"""
import io
import re
import sys
import zipfile
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CACHE, FIG, OUT, ROOT

ZIP = ROOT / "data_ext" / "D2_bregar" / "data_set.zip"
OUT_D2 = OUT / "d2"
OUT_D2.mkdir(parents=True, exist_ok=True)
META = ["TAG_ID", "ANCHOR_ID", "X_TAG", "Y_TAG", "Z_TAG", "X_ANCHOR", "Y_ANCHOR", "Z_ANCHOR", "NLOS", "RANGE",
        "FP_INDEX", "RSS", "RSS_FP", "FP_POINT1", "FP_POINT2", "FP_POINT3", "STDEV_NOISE", "CIR_POWER",
        "MAX_NOISE", "RXPACC", "CHANNEL_NUMBER", "FRAME_LENGTH", "PREAMBLE_LENGTH", "BITRATE", "PRFR",
        "PREAMBLE_CODE"]
NUM = [c for c in META if c not in ("TAG_ID", "ANCHOR_ID", "NLOS")]
PAT = re.compile(r"data_set/raw_data/environment(\d)/data/([^/]+)/ch(\d)_A(\d)\.csv$")

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LOS_C, NLOS_C = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def open_zip(remote):
    if remote:
        from importlib import import_module
        rl = import_module("10_d2_remote_listing")
        return zipfile.ZipFile(io.BufferedReader(rl.HttpRangeFile(rl.URL), buffer_size=1 << 20))
    return zipfile.ZipFile(ZIP)


def read_meta(zf, limit=None):
    names = [n for n in zf.namelist() if PAT.match(n)]
    if limit:
        keep = {}
        for n in names:
            e, pos = PAT.match(n).group(1, 2)
            keep.setdefault(e, [])
            if pos not in keep[e] and len(keep[e]) < limit:
                keep[e].append(pos)
        names = [n for n in names if PAT.match(n).group(2) in keep[PAT.match(n).group(1)]]
    recs, bad = [], []
    for i, n in enumerate(names):
        e, pos, ch, a = PAT.match(n).groups()
        txt = zf.read(n).decode("utf-8", "replace").splitlines()
        version = txt[0].strip() if txt else ""
        for k, line in enumerate(txt[2:]):
            if not line.strip():
                continue
            f = line.split(",", len(META))[:len(META)]
            if len(f) < len(META):
                bad.append((n, k))
                continue
            recs.append([int(e), pos, int(ch), int(a), k, version] + f)
        if i % 2000 == 0:
            print(f"  read {i}/{len(names)}", flush=True)
    df = pd.DataFrame(recs, columns=["env", "pos", "ch", "anchor", "k", "version"] + META)
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, bad, len(names)


if __name__ == "__main__":
    remote = "--remote" in sys.argv
    limit = int(sys.argv[sys.argv.index("--remote") + 1]) if remote else None
    cache = CACHE / ("d2_meta_remote_test.pkl" if remote else "d2_meta.pkl")
    if cache.exists() and not remote:
        df = pd.read_pickle(cache)
        bad, n_files = [], df.groupby(["env", "pos", "ch", "anchor"]).ngroups
    else:
        with open_zip(remote) as zf:
            df, bad, n_files = read_meta(zf, limit)
            wp = {e: pd.read_csv(io.BytesIO(zf.read(f"data_set/environment{e}/walking_path.csv"))) for e in range(4)}
            anc = {e: pd.read_csv(io.BytesIO(zf.read(f"data_set/environment{e}/anchors.csv"))) for e in range(4)}
        # 경로 순서(위치 번호): walking_path.csv 의 행 순서와 폴더 이름(x_y_z) 대응
        pidx = {}
        for e, w in wp.items():
            for i, r in w.iterrows():
                pidx[(e, f"{r.x:.2f}_{r.y:.2f}_{r.z:.2f}")] = i
        df["path_idx"] = [pidx.get((e, p), np.nan) for e, p in zip(df["env"], df["pos"])]
        df.to_pickle(cache)
        pd.concat(anc, names=["env", "i"]).to_csv(OUT_D2 / "d2_anchors.csv", encoding="utf-8-sig")

    log(f"[1] 구조: 파일 {n_files}개, 행 {len(df):,}개, 파싱 실패 행 {len(bad)}")
    log(f"  file version: {df['version'].value_counts().to_dict()}")
    log(f"  NLOS 값: {df['NLOS'].value_counts().to_dict()}")
    log(f"  path_idx 대응 실패 행: {df['path_idx'].isna().sum()}")
    log("  열 결측 비율(0 아닌 것): " + str({c: round(v, 5) for c, v in df[META].isna().mean().items() if v > 0}))
    s = (df.groupby("env").agg(positions=("pos", "nunique"), anchors=("anchor", "nunique"), channels=("ch", "nunique"),
                               rows=("k", "size"))
         .join(df.groupby(["env", "pos", "ch", "anchor"]).size().groupby("env").agg(["min", "median", "max"])
               .add_prefix("rows_per_file_")))
    log(s.to_string())
    log("  채널별 설정: " + df.groupby("ch")[["PREAMBLE_LENGTH", "BITRATE", "PRFR", "PREAMBLE_CODE"]]
        .agg(lambda x: sorted(x.dropna().unique().tolist())).to_string())

    # 파일(=env×pos×ch×anchor) 안에서 NLOS 값이 하나인지, 채널 간 같은 위치·앵커의 라벨이 같은지
    per_file = df.groupby(["env", "pos", "ch", "anchor"])["NLOS"].nunique()
    log(f"  파일 안 NLOS 값이 2개 이상인 파일: {(per_file > 1).sum()}")
    lab = df.groupby(["env", "pos", "anchor", "ch"])["NLOS"].first().unstack("ch")
    log(f"  같은 위치·앵커에서 채널 간 라벨이 다른 경우: {(lab.nunique(axis=1) > 1).sum()} / {len(lab)}")

    # [2] 앵커별 NLOS 비율, 위치별 NLOS 앵커 수
    df["is_nlos"] = df["NLOS"].astype(str).str.upper().eq("NLOS")
    pa = df.groupby(["env", "pos", "anchor"]).agg(nlos=("is_nlos", "max"), path_idx=("path_idx", "first"))
    r = pa.groupby(["env", "anchor"])["nlos"].mean().unstack("anchor")
    r.to_csv(OUT_D2 / "d2_nlos_ratio_env_anchor.csv", encoding="utf-8-sig")
    log("\n[2] 위치 기준 NLOS 비율 (환경×앵커)")
    log(r.to_string(float_format=lambda v: f"{v:.2f}"))
    k = pa.groupby(["env", "pos"])["nlos"].sum().groupby("env").value_counts().unstack(fill_value=0)
    k.to_csv(OUT_D2 / "d2_nlos_count_per_position.csv", encoding="utf-8-sig")
    log("  위치별 NLOS 앵커 수 분포 (열 = NLOS 앵커 수, 값 = 위치 수)")
    log(k.to_string())

    # [3] 거리 오차 (측정 - 실제 3D 거리)
    df["true_d"] = np.sqrt((df.X_TAG - df.X_ANCHOR) ** 2 + (df.Y_TAG - df.Y_ANCHOR) ** 2 + (df.Z_TAG - df.Z_ANCHOR) ** 2)
    df["err"] = df["RANGE"] - df["true_d"]
    df["rss_minus_fp"] = df["RSS"] - df["RSS_FP"]
    rows = []
    for (e, nl), g in df.groupby(["env", "is_nlos"]):
        x = g["err"].dropna()
        rows.append(dict(env=e, label="NLOS" if nl else "LOS", n=len(x), err_median=x.median(),
                         err_p5=x.quantile(.05), err_p95=x.quantile(.95),
                         gt_0_2=(x > .2).mean(), gt_0_3=(x > .3).mean(), gt_0_5=(x > .5).mean(),
                         rss_fp_median=g["rss_minus_fp"].median()))
    E = pd.DataFrame(rows)
    E.to_csv(OUT_D2 / "d2_range_error.csv", index=False, encoding="utf-8-sig")
    log("\n[3] 거리 오차 = RANGE - 실제 3D 거리 (m), 라벨별. rss_fp_median = RSS - RSS_FP 중앙값 (dB, 기술 통계용)")
    log(E.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    # 앵커별 LOS 오차 중앙값 (고정 편향)
    b = df[~df.is_nlos].groupby(["env", "anchor"])["err"].median().unstack("anchor")
    log("  LOS 거리 오차 중앙값 (환경×앵커, m) = 앵커별 고정 편향")
    log(b.to_string(float_format=lambda v: f"{v:+.3f}"))
    # 파일 내 반복 측정 잡음
    w = df.groupby(["env", "pos", "ch", "anchor", "is_nlos"])["RANGE"].std().groupby("is_nlos").median()
    log(f"  파일 내 RANGE 표준편차 중앙값 (m): LOS {w.get(False, np.nan):.3f}, NLOS {w.get(True, np.nan):.3f}")

    # 임계값 기반 라벨(방법 A 규칙)과 D2 NLOS 라벨의 일치도 (위치×앵커×채널 중앙값 오차 기준, 앵커 편향 보정 전/후)
    fm = df.groupby(["env", "pos", "ch", "anchor"]).agg(err=("err", "median"), nlos=("is_nlos", "first"))
    fm["err_debias"] = fm["err"] - fm.join(b.stack().rename("bias"), on=["env", "anchor"])["bias"]
    arows = []
    for col in ["err", "err_debias"]:
        for t in [0.2, 0.3, 0.5]:
            pred = fm[col] > t
            tp = (pred & fm.nlos).sum(); fp = (pred & ~fm.nlos).sum(); fn = (~pred & fm.nlos).sum()
            arows.append(dict(rule=col, thr=t, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                              los_flag_rate=(pred & ~fm.nlos).sum() / max((~fm.nlos).sum(), 1)))
    AG = pd.DataFrame(arows)
    AG.to_csv(OUT_D2 / "d2_threshold_vs_label.csv", index=False, encoding="utf-8-sig")
    log("\n[4] 거리 오차 임계 규칙 vs D2 NLOS 라벨 (단위 = 위치×채널×앵커 파일의 중앙값)")
    log(AG.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # [5] 정상 데이터 양
    allos = (k[0] if 0 in k.columns else pd.Series(0, index=k.index))
    log("\n[5] 모든 앵커가 LOS 인 위치 수 (환경별): " + str(allos.to_dict()))
    log("  LOS 행 수 (환경×앵커): ")
    log(df[~df.is_nlos].groupby(["env", "anchor"]).size().unstack("anchor").to_string())

    # ---------------- 그림 ----------------
    if not remote:
        fig, axes = plt.subplots(4, 1, figsize=(12, 7.5), sharex=True)
        for e, ax in zip(range(4), axes):
            m = pa.loc[e].reset_index().pivot_table(index="anchor", columns="path_idx", values="nlos", aggfunc="max")
            ax.imshow(m.to_numpy(float), aspect="auto", cmap=plt.matplotlib.colors.ListedColormap([LOS_C, NLOS_C]),
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_yticks(range(len(m.index)), [f"A{a}" for a in m.index], fontsize=7)
            ax.set_ylabel(f"환경 {e}")
            ax.grid(False)
        axes[-1].set_xlabel("경로 순서 (위치 번호, 0.3 m 간격)")
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=LOS_C, label="LOS"), Patch(color=NLOS_C, label="NLOS")],
                   loc="upper right", ncol=2, fontsize=8)
        fig.suptitle("D2 앵커×경로 NLOS 라벨", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(FIG / "d2_nlos_map.png", dpi=130)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        bins = np.linspace(-0.5, 3, 141)
        for nl, c, lab_ in [(False, LOS_C, "LOS"), (True, NLOS_C, "NLOS")]:
            axes[0].hist(df.loc[df.is_nlos == nl, "err"].clip(-0.5, 3), bins=bins, color=c, alpha=0.6,
                         density=True, label=lab_)
            axes[1].hist(df.loc[df.is_nlos == nl, "rss_minus_fp"].clip(0, 25), bins=np.linspace(0, 25, 101),
                         color=c, alpha=0.6, density=True, label=lab_)
        for t in (0.2, 0.3, 0.5):
            axes[0].axvline(t, color=MUTED, lw=0.8, ls=":")
        axes[0].set_xlabel("RANGE − 실제 3D 거리 (m)")
        axes[1].set_xlabel("RSS − RSS_FP (dB)")
        axes[0].set_title("거리 오차 분포", loc="left", fontsize=10)
        axes[1].set_title("RSS − RSS_FP 분포 (기술 통계, 라벨에 쓰지 않음)", loc="left", fontsize=10)
        for ax in axes:
            ax.legend(fontsize=8)
            ax.set_ylabel("밀도")
        fig.suptitle("D2 라벨별 분포 (전체 환경·채널)", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(FIG / "d2_error_rssfp.png", dpi=130)
        plt.close(fig)

    (OUT_D2 / ("12_d2_structure_remote_test.txt" if remote else "12_d2_structure.txt")).write_text(
        "\n".join(lines), encoding="utf-8")
