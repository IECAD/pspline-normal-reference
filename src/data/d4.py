"""D4 범용 다변량 데이터: D4a SMD (OmniAnomaly), D4b ASD (InterFusion). 원본은 읽기만 한다.

구조 근거: analysis/out/d4/structure.md
  - SMD: data_ext/D4a_SMD/{train,test,test_label,interpretation_label}/machine-*.txt (CSV, 헤더 없음)
  - ASD: data_ext/D4b_ASD/processed/omi-*_{train,test,test_label}.pkl (numpy pickle -> safe_pickle 로 읽고 cache 에 .npy),
         data_ext/D4b_ASD/interpretation_label/omi-*.txt
  - 해석 라벨 'start-end:v1,v2,..' 변수 번호 1-based, 시간 0-based. end 규약: SMD [s, e), ASD [s, e] (test_label 대조로 확인)
표현: 행 = 변수 V01.., 열 = 시간 스텝. 분할: 공식 train 의 앞 (1-val_frac) 학습, 뒤 val_frac 검증, 공식 test = 테스트.
셀 라벨 lab_I_V{j}: 해석 라벨 구간·변수 = 1, 테스트의 나머지 = 0, 이상인데 해석 라벨이 없는 시점 = NaN.
점 라벨 lab_P: test_label.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config
from .safe_pickle import read_numpy_pickle_static

SRC = {
    "smd": dict(root=ROOT / "data_ext" / "D4a_SMD", end_inclusive=False),
    "asd": dict(root=ROOT / "data_ext" / "D4b_ASD", end_inclusive=True),
}
RE_INTERP = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*:\s*([\d,\s]+?)\s*$")


def entities(ds):
    if ds == "smd":
        return sorted((p.stem for p in (SRC[ds]["root"] / "train").glob("machine-*.txt")),
                      key=lambda s: tuple(int(x) for x in s.split("-")[1:]))
    return sorted((p.name[:-len("_train.pkl")] for p in (SRC[ds]["root"] / "processed").glob("omi-*_train.pkl")),
                  key=lambda s: int(s.split("-")[1]))


def _asd_array(cfg, ent, part):
    cdir = cfg["paths"]["cache"] / "d4b_npy"
    cdir.mkdir(parents=True, exist_ok=True)
    cp = cdir / f"{ent}_{part}.npy"
    if not cp.exists():
        np.save(cp, read_numpy_pickle_static(SRC["asd"]["root"] / "processed" / f"{ent}_{part}.pkl"))
    return np.load(cp, allow_pickle=False)


def load(cfg, ds, ent):
    """반환: train (Ntr, V), test (Nte, V), label (Nte,), interp [(a, b_inclusive, [변수 0-based])]."""
    r = SRC[ds]["root"]
    if ds == "smd":
        tr = pd.read_csv(r / "train" / f"{ent}.txt", header=None, dtype=np.float64).to_numpy()
        te = pd.read_csv(r / "test" / f"{ent}.txt", header=None, dtype=np.float64).to_numpy()
        y = pd.read_csv(r / "test_label" / f"{ent}.txt", header=None, dtype=np.float64)[0].to_numpy()
        ip = r / "interpretation_label" / f"{ent}.txt"
    else:
        tr = _asd_array(cfg, ent, "train").astype(np.float64)
        te = _asd_array(cfg, ent, "test").astype(np.float64)
        y = _asd_array(cfg, ent, "test_label").astype(np.float64).reshape(-1)
        ip = r / "interpretation_label" / f"{ent}.txt"
    interp = []
    for ln in ip.read_text(encoding="utf-8").splitlines():
        m = RE_INTERP.match(ln)
        if m:
            s, e = int(m.group(1)), int(m.group(2))
            b = e if SRC[ds]["end_inclusive"] else e - 1
            interp.append((s, b, [int(v) - 1 for v in m.group(3).split(",") if v.strip()]))
    return tr, te, y, interp


def series(cfg, ds, ent) -> pd.DataFrame:
    tr, te, y, interp = load(cfg, ds, ent)
    V = tr.shape[1]
    vcols = [f"V{j + 1:02d}" for j in range(V)]
    val_frac = cfg["d4"]["val_frac_of_train"]
    n_tr = len(tr)
    cut = int(round(n_tr * (1 - val_frac)))
    a = pd.DataFrame(tr, columns=vcols)
    a["split"] = np.where(np.arange(n_tr) < cut, "train", "val")
    a["seg"], a["tidx"] = 0, np.arange(n_tr)
    b = pd.DataFrame(te, columns=vcols)
    b["split"], b["seg"], b["tidx"] = "test", 1, np.arange(len(te))
    b["lab_P"] = y
    cell = np.zeros((len(te), V))
    covered = np.zeros(len(te), bool)
    for s, e, vs in interp:
        covered[s:e + 1] = True
        for v in vs:
            cell[s:e + 1, v] = 1.0
    cell[(y > 0) & ~covered, :] = np.nan            # 이상인데 해석 라벨이 없는 시점
    for j, c in enumerate(vcols):
        b[f"lab_I_{c}"] = cell[:, j]
    df = pd.concat([a, b], ignore_index=True)
    df["file"], df["space"], df["condition"] = f"{ds}/{ent}", ds, "official"
    df["node"] = df["file"]
    df["edge"], df["normal"] = False, True
    df["t"] = df["tidx"].astype(float)
    return df, vcols


if __name__ == "__main__":
    cfg = load_config()
    for ds in ("smd", "asd"):
        ents = entities(ds)
        df, vc = series(cfg, ds, ents[0])
        print(ds, len(ents), "entities; first:", ents[0], df["split"].value_counts().to_dict(),
              "V", len(vc), "lab_P", df["lab_P"].mean().round(4),
              "interp cell rate", np.nanmean(df.filter(like="lab_I_").to_numpy()).round(4))
