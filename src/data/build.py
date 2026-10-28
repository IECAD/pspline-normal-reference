"""윈도우 데이터셋 생성 (단계 1).

사용법: python -m src.data.build --dataset d1|d2|all [--config configs/data.yaml]
출력 (cache/windows/<dataset>/):
  <name>.npz          X (N,S,T), rows, 셀 배열 (라벨·마스크), 같은 이름의 _meta.parquet (윈도우별 file/node/split/t0)
  <name>_C_seed<s>.npz  D1 정상 파일 테스트 몫에 방법 C 를 주입한 버전 (+ _events.csv)
  build_summary.csv   윈도우 수와 라벨 비율
설정 사본과 기하 적합값(cache/d1_geometry.json)을 함께 저장해 재현할 수 있게 한다.
"""
import argparse
import json
import shutil

import numpy as np
import pandas as pd

from . import d1 as D1
from . import d2 as D2
from . import d3 as D3
from . import d4 as D4
from .config import load_config
from .inject import inject
from .labels import add_labels, fit_geometry, reference_medians, save_geometry
from .windows import assign_split, fill_short_gaps, make_windows


def _save(out_dir, name, w, extra_meta=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{name}.npz", X=w["X"].astype(np.float32), rows=np.array(w["rows"]),
                        **{f"cell_{k}": v.astype(np.float32) for k, v in w["cells"].items()})
    meta = w["meta"]
    if extra_meta:
        for k, v in extra_meta.items():
            meta[k] = v
    meta.to_parquet(out_dir / f"{name}_meta.parquet", index=False)


def _cell_map(value_cols, prefix):
    """value 열 '{metric}_A{a}' -> 같은 앵커의 '{prefix}_A{a}' (앵커 라벨을 그 앵커의 모든 지표 행에 적용)."""
    return [f"{prefix}_A{c.rsplit('_A', 1)[1]}" for c in value_cols]


def _rate(arr):
    v = arr[~np.isnan(arr)]
    return float(v.mean()) if v.size else np.nan


def build_d1(cfg):
    out = cfg["paths"]["cache"] / "windows" / "d1"
    W, rate = cfg["window"], cfg["d1"]["sample_rate_hz"]
    metrics = cfg["d1"]["metrics"] + cfg["d1"]["extra_metrics"]
    value_cols = [f"{m}_A{a}" for m in metrics for a in D1.ANCHOR_IDS]
    cyc = {}
    for name, meta in cfg["d1"]["files"].items():
        c = D1.cycles(cfg, name, metrics)
        c["normal"] = bool(meta["normal"])
        c["split"] = assign_split(c, cfg["split"]["ratios"])
        cyc[name] = fill_short_gaps(c, value_cols, limit=W["ffill_limit"])
    geoms = {}
    for name, meta in cfg["d1"]["files"].items():
        calib = meta["calib"]
        if calib not in geoms:
            geoms[calib] = fit_geometry(cfg, cyc[calib], robust=cfg["d1"]["files"][calib]["condition"] != "los")
        cyc[name] = add_labels(cfg, cyc[name], geoms[calib], reference_medians(cyc[calib]))
    save_geometry(cfg, geoms)

    cells = {"lab_A": _cell_map(value_cols, "lab_A"), "lab_A2": _cell_map(value_cols, "lab_A2"),
             "errA": _cell_map(value_cols, "errA"), "offA2": _cell_map(value_cols, "offA2")}
    summary = []
    for name, c in cyc.items():
        w = make_windows(c, value_cols, W["T"], W["stride"], cells, max_missing=W["max_missing"])
        _save(out, name, w)
        for sp, g in w["meta"].groupby("split"):
            ix = g.index.to_numpy()
            summary.append(dict(dataset="d1", name=name, variant="base", split=sp, windows=len(ix),
                                lab_A=_rate(w["cells"]["lab_A"][ix]), lab_A2=_rate(w["cells"]["lab_A2"][ix])))
        if cfg["d1"]["files"][name]["normal"]:
            ev_all = []
            for s in cfg["seeds"]:
                ci, ev = inject(c, cfg["labels"]["C"], rate, s, D1.ANCHOR_IDS, drop_cols=metrics)
                ci = ci[ci["split"] == cfg["labels"]["C"]["target_split"]]
                wc = make_windows(ci, value_cols, W["T"], W["stride"], {"lab_C": _cell_map(value_cols, "lab_C")},
                                  max_missing=W["max_missing"])
                _save(out, f"{name}_C_seed{s}", wc, {"seed": s})
                ev_all.append(ev)
                lab = wc["cells"]["lab_C"]
                summary.append(dict(dataset="d1", name=name, variant=f"C_seed{s}", split="test",
                                    windows=len(wc["meta"]), lab_C=_rate(lab),
                                    win_with_C=float(np.nanmax(lab, axis=(1, 2)).mean()) if len(lab) else np.nan))
            pd.concat(ev_all).to_csv(out / f"{name}_C_events.csv", index=False, encoding="utf-8-sig")
    return summary


def build_d2(cfg):
    out = cfg["paths"]["cache"] / "windows" / "d2"
    W = cfg["window"]
    metrics = cfg["d2"]["metrics"] + cfg["d2"].get("extra_metrics", [])
    value_cols = [f"{m}_A{a}" for m in metrics for a in D2.ANCHOR_IDS]
    summary = []
    for e in range(4):
      for axis in dict.fromkeys([cfg["d2"]["time_axis"], "within_position"]):   # 위치별 변형: 실험 3 위치 홀드아웃용
        s = D2.series(cfg, e, axis)
        cells = {"mask": _cell_map(value_cols, "mask"), "lab_D": _cell_map(value_cols, "lab_D")}
        w = make_windows(s, value_cols, W["T"], W["stride"], cells, max_missing=W["max_missing"],
                         meta_cols=("file", "space", "condition", "node", "seg", "split", "path_idx"))
        name = f"env{e}_ch{cfg['d2']['channel']}_{'within' if axis == 'within_position' else axis}"
        _save(out, name, w)
        for sp, g in w["meta"].groupby("split"):
            ix = g.index.to_numpy()
            lab = w["cells"]["lab_D"][ix]
            summary.append(dict(dataset="d2", name=name, variant="base", split=sp, windows=len(ix),
                                lab_D=_rate(lab),
                                win_all_LOS=float((np.nanmax(lab, axis=(1, 2)) == 0).mean()) if len(ix) else np.nan))
    return summary


def build_d3(cfg):
    out = cfg["paths"]["cache"] / "windows" / "d3"
    d3 = cfg["d3"]
    offs, info = D3.calibrate_all(cfg)
    (cfg["paths"]["cache"] / "d3_offsets.json").write_text(json.dumps(
        {r: dict(info=info[r], offsets_m={a: float(v) for a, v in offs[r].items()}) for r in offs},
        indent=1, ensure_ascii=False), encoding="utf-8")
    summary = []
    for rec in d3["recordings"]:
        s, an = D3.series(cfg, rec, offs[rec])
        s["split"] = assign_split(s, cfg["split"]["ratios"])
        value_cols = [f"{m}_{a}" for m in d3["metrics"] for a in an]
        anchor_of = [c.split("_", 1)[1] if c.startswith("TDOA") else c.rsplit("_", 1)[1] for c in value_cols]
        cells = {"obs": [f"obs_{a}" for a in anchor_of], "lab_R": [f"lab_R_{a}" for a in anchor_of],
                 "e": [f"e_{a}" for a in anchor_of]}
        S_ = len(value_cols)
        w = make_windows(s, value_cols, d3["T"], d3["stride"], cells,
                         max_missing=int((1 - d3["min_obs_frac"]) * S_ * d3["T"]))
        _save(out, rec, w, {"anchors": ",".join(an)})
        for sp, g in w["meta"].groupby("split"):
            ix = g.index.to_numpy()
            summary.append(dict(dataset="d3", name=rec, variant="base", split=sp, windows=len(ix),
                                obs=_rate(w["cells"]["obs"][ix]), lab_R=_rate(w["cells"]["lab_R"][ix])))
    return summary


def build_d4(cfg):
    out = cfg["paths"]["cache"] / "windows" / "d4"
    T, stride = cfg["d4"]["T"], cfg["d4"]["stride"]
    summary = []
    for ds in cfg["d4"]["datasets"]:
        for ent in D4.entities(ds):
            s, vcols = D4.series(cfg, ds, ent)
            cells = {"lab_I": [f"lab_I_{c}" for c in vcols], "lab_P": ["lab_P"]}
            s["lab_P"] = s["lab_P"].astype(float)
            w = make_windows(s, vcols, T, stride, cells, max_missing=0)
            _save(out, f"{ds}_{ent}", w)
            for sp, g in w["meta"].groupby("split"):
                ix = g.index.to_numpy()
                summary.append(dict(dataset=f"d4_{ds}", name=ent, variant="base", split=sp, windows=len(ix),
                                    lab_P=_rate(w["cells"]["lab_P"][ix]), lab_I=_rate(w["cells"]["lab_I"][ix])))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["d1", "d2", "d3", "d4", "all"])
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    summ = []
    if args.dataset in ("d1", "all"):
        summ += build_d1(cfg)
    if args.dataset in ("d2", "all"):
        summ += build_d2(cfg)
    if args.dataset in ("d3", "all"):
        summ += build_d3(cfg)
    if args.dataset in ("d4", "all"):
        summ += build_d4(cfg)
    out = cfg["paths"]["cache"] / "windows"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg["_path"], out / "data_config_used.yaml")
    S = pd.DataFrame(summ)
    prev = out / "build_summary.csv"
    if prev.exists():
        old = pd.read_csv(prev)
        S = pd.concat([old[~old["dataset"].isin(S["dataset"].unique())], S])
    S.to_csv(prev, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(S.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    geo = cfg["paths"]["cache"] / "d1_geometry.json"
    if geo.exists() and args.dataset in ("d1", "all"):
        print(json.dumps(json.loads(geo.read_text(encoding="utf-8")), indent=1, ensure_ascii=False))
