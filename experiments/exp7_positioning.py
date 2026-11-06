"""실험 7: 측위 효과 (doc/실험7_계획.md, 사전 등록 2026-10-11).

P1 D1 주입 (주): 세 공간의 검증·테스트 윈도우, SC1·SC2 (주), SC3 (보조)
P2 D1 실제 NLOS (보조, 테스트만): 실제 NLOS 파일 4개, 정상 파일 노드에 있는 윈도우, 정답 이상 앵커 = lab_A2
P3 D2 주입 (보조): SC1m·SC2m·SC3m, 기하 NLOS 앵커는 측위에서 뺌, LOS 앵커 4개 이상인 스텝만
탐지기: PS (F01, K_t = 12, SAP), cell (M01). 반영: none / X-hard / X-soft / oracle-hard
해법: LS, WLS, Huber, EKF (결합 갱신) + EKFseq (사전 등록 원안, 보조)
사용법: python experiments/exp7_positioning.py --splits val test
출력: results/exp7/metrics_<분할>.csv (단위별), pooled_<분할>.csv (단위를 합친 RMSE·P90)
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from node_model_eval import SIG, build, full_spec  # noqa: E402
from src.data import d1 as D1  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.inject import inject_windows  # noqa: E402
from src.data.load import load_windows  # noqa: E402
from src.model.bank import anchor_scores  # noqa: E402
from src.positioning.solvers import solve_all  # noqa: E402

OUT = ROOT / "results" / "exp7"
if os.environ.get("EXP7_PS1H"):
    OUT = OUT / "ps1h"
DETECTORS = {"PS": {"K_t": 12}, "cell": {"time_basis": "cell"}}
if os.environ.get("EXP7_PS1H"):                                    # 추가 비교 2 (doc/실험7_계획.md): 설계 후보 PS1H
    DETECTORS["PS1H"] = {"K_t": 12, "diff_order": 1, "robust": "huber"}
    OUT_SUB = "ps1h"
else:
    OUT_SUB = ""
Q = {"d1": 0.05, "d2": 0.3}
DM = ["RANGE", "RSS", "RSS_FP"]
POOL = defaultdict(list)                   # (set, split, scenario, solver, config, subset) -> 스텝 오차


# ----------------------------------------------------------------------------- 공통
def detector_scores(models, Xi, W, nodes, rows, anchors):
    out = {}
    for name, (comp, bank) in models.items():
        Xs, Ws = comp.transform(Xi, W, rows)
        out[name] = (anchor_scores(bank.score(Xs, nodes, Ws), comp.anchor_of_row, anchors), bank.thr)
    return out


def w_soft(A, thr, avail):
    A = np.nan_to_num(A, nan=0.0)
    return np.where(A > thr, (thr / np.maximum(A, 1e-12)) ** 2, 1.0) * avail


def w_hard(flag, rank, avail):
    """flag 앵커를 뺀다. 남은 앵커가 3개 미만이면 rank 가 낮은 3개를 쓴다. 모양 (N, A, T)."""
    keep = ~flag & avail
    need = keep.sum(1) < 3
    if need.any():
        order = np.argsort(np.where(avail, rank, np.inf), axis=1)
        top3 = np.zeros_like(keep)
        np.put_along_axis(top3, order[:, :3, :], True, axis=1)
        keep = np.where(need[:, None, :], top3 & avail, keep)
    return keep.astype(float)


def weight_sets(scores, lab, true_res, avail):
    """반영 구성별 가중치 (N, A, T)."""
    W = {"none": avail.astype(float)}
    for name, (A, thr) in scores.items():
        W[f"{name}-soft"] = w_soft(A, thr, avail)
        W[f"{name}-hard"] = w_hard(np.nan_to_num(A, nan=0.0) > thr, np.nan_to_num(A, nan=0.0), avail)
    W["oracle-hard"] = w_hard(np.nan_to_num(lab, nan=0.0) == 1, np.abs(np.nan_to_num(true_res, nan=0.0)), avail)
    return W


def summarize(e, valid, anom, key, tag):
    """e: (N, T) 오차. 단위 지표 행 목록 + POOL 누적."""
    out = []
    for sub, m in (("all", valid), ("anom", valid & anom), ("norm", valid & ~anom)):
        x = e[m & np.isfinite(e)]
        POOL[key + (sub,)].append(x)
        out.append(dict(**tag, subset=sub, n=int(x.size), n_fail=int((m & ~np.isfinite(e)).sum()),
                        rmse=float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan,
                        p90=float(np.percentile(x, 90)) if x.size else np.nan,
                        mean=float(x.mean()) if x.size else np.nan))
    return out


def position_eval(r, avail, scores, lab, true_res, pos, h, b, sig, q, valid, anom, base_tag, key):
    """r, avail, lab, true_res: (N, A, T). pos: (N, T, 2). h: (N, T, A) 또는 (A,)."""
    out = []
    for cfg, w in weight_sets(scores, lab, true_res, avail).items():
        sol = solve_all(np.moveaxis(r, 1, 2), np.moveaxis(w, 1, 2), base_tag["_anc"], h, b, sig, q)
        for sv, P in sol.items():
            e = np.sqrt(((P - pos) ** 2).sum(-1))
            tag = {k: v for k, v in base_tag.items() if not k.startswith("_")}
            out += summarize(e, valid, anom, key + (sv, cfg), dict(**tag, solver=sv, config=cfg))
    return out


def fit_models(code, ecfg, ds, X, W, rows, nodes, split):
    tr, va = split == "train", split == "val"
    models = {}
    for name, sp in DETECTORS.items():
        comp, bank, _ = build(code, full_spec(sp, ecfg), ecfg, ds, X[tr], W[tr], rows, nodes[tr],
                              X[va], W[va], nodes[va])
        models[name] = (comp, bank)
    return models


# ----------------------------------------------------------------------------- D1
def d1_points(dcfg, f):
    c = D1.cycles(dcfg, f)
    return c.groupby("point_id")[["x_m", "y_m"]].first()


def run_d1(ecfg, dcfg, code, splits, geom):
    E = ecfg["datasets"]["d1"]
    anchors = [f"A{a}" for a in range(1, 5)]
    out = []
    for space, sp in E["spaces"].items():
        g = geom[sp["normal"]]
        anc = np.array([dcfg["d1"]["anchors"][space][a] for a in range(1, 5)], float)
        dz, bias = g["dz"], np.array([g["bias"][str(a)] for a in range(1, 5)])
        full = load_windows("d1", sp["normal"], dcfg)
        X, meta, rows = full["X"], full["meta"], full["rows"]
        nodes = np.array([f"{space}/{n.split('/')[-1]}" for n in meta["node"]])
        W, split = np.isfinite(X).astype(float), meta["split"].to_numpy()
        dix = [rows.index(f"DIST_{a}") for a in anchors]
        models = fit_models(code, ecfg, "d1", X, W, rows, nodes, split)

        def geo(meta_, f):
            pts = d1_points(dcfg, f)
            xy = pts.loc[[n.split("/")[-1] for n in meta_["node"]]].to_numpy()          # (N, 2)
            td = np.sqrt(((xy[:, None, :] - anc[None]) ** 2).sum(-1) + dz ** 2)          # (N, A)
            return xy, td

        xy, td = geo(meta, sp["normal"])
        tr = split == "train"
        res = X[tr][:, dix] - td[tr][..., None] - bias[None, :, None]
        sig = np.nanstd(np.moveaxis(res, 1, 0).reshape(4, -1), 1)
        print(f"  d1 {space}: dz={dz:.2f} σ={np.round(sig, 3)}", flush=True)
        for spl in splits:
            ix = np.flatnonzero(split == spl)
            pos = np.repeat(xy[ix][:, None, :], X.shape[2], axis=1)
            for scn, sc in E["scenarios"].items():
                for s in E["injection_seeds"][spl]:
                    Xi, lab, _, _ = inject_windows(X[ix], rows, sc, s)
                    r = Xi[:, dix]
                    avail = np.isfinite(r)
                    tres = r - td[ix][..., None] - bias[None, :, None]
                    sc_ = detector_scores(models, Xi, W[ix], nodes[ix], rows, anchors)
                    out += position_eval(r, avail, sc_, lab, tres, pos, np.full(4, dz), bias, sig, Q["d1"],
                                         avail.sum(1) >= 3, (lab == 1).any(1),
                                         dict(set="P1", dataset="d1", split=spl, space=space, scenario=scn, seed=s,
                                              _anc=anc), ("P1", spl, scn))
            if spl == "test":                                              # P2 실제 NLOS
                for f in sp["real"]:
                    w = load_windows("d1", f, dcfg)
                    nd = np.array([f"{space}/{n.split('/')[-1]}" for n in w["meta"]["node"]])
                    keep = np.array([n in models["PS"][1].M for n in nd])
                    if not keep.any():
                        continue
                    Xr, mr = w["X"][keep], w["meta"][keep].reset_index(drop=True)
                    xy_r, td_r = geo(mr, f)
                    r = Xr[:, dix]
                    avail = np.isfinite(r)
                    lab = w["cells"]["lab_A2"][keep][:, dix]
                    tres = r - td_r[..., None] - bias[None, :, None]
                    sc_ = detector_scores(models, Xr, np.isfinite(Xr).astype(float), nd[keep], rows, anchors)
                    pos = np.repeat(xy_r[:, None, :], Xr.shape[2], axis=1)
                    out += position_eval(r, avail, sc_, lab, tres, pos, np.full(4, dz), bias, sig, Q["d1"],
                                         avail.sum(1) >= 3, (lab == 1).any(1),
                                         dict(set="P2", dataset="d1", split="test", space=space, scenario=f, seed=-1,
                                              coverage=float(keep.mean()), _anc=anc), ("P2", "test", f))
        print(f"  d1 {space} done", flush=True)
    return out


# ----------------------------------------------------------------------------- D2
def run_d2(ecfg, dcfg, code, splits):
    E = ecfg["datasets"]["d2"]
    anchors = [f"A{a}" for a in range(1, 9)]
    meta_all = pd.read_parquet(ROOT / "cache" / "d2_meta.parquet")
    out = []
    for env in E["envs"]:
        m = meta_all[(meta_all.env == env) & (meta_all.ch == dcfg["d2"]["channel"])]
        A = m.groupby("anchor")[["X_ANCHOR", "Y_ANCHOR", "Z_ANCHOR"]].first().reindex(range(1, 9)).to_numpy()
        tag = m.groupby("path_idx")[["X_TAG", "Y_TAG", "Z_TAG"]].first()
        anc = A[:, :2]
        full = load_windows("d2", f"env{env}_{E['channel_name']}", dcfg)
        X, meta, rows, cells = full["X"], full["meta"], full["rows"], full["cells"]
        nodes = np.array([f"env{env}/{int(t)}" for t in meta["tidx0"]])
        W, split = cells["mask"], meta["split"].to_numpy()
        T = X.shape[2]
        rix = [rows.index(f"RANGE_{a}") for a in anchors]
        models = fit_models(code, ecfg, "d2", X, W, rows, nodes, split)
        pidx = meta["tidx0"].to_numpy()[:, None] + np.arange(T)[None]                    # (N, T)
        G = tag.reindex(pidx.ravel()).to_numpy().reshape(len(X), T, 3)
        pos_all, h_all = G[..., :2], G[..., 2][..., None] - A[None, None, :, 2]          # h: (N, T, A)
        td_all = np.sqrt(((pos_all[:, :, None, :] - anc[None, None]) ** 2).sum(-1) + h_all ** 2)   # (N, T, A)
        los_all = cells["lab_D"][:, rix] == 0                                            # (N, A, T)
        tr = split == "train"
        res = np.where(los_all[tr], X[tr][:, rix] - np.moveaxis(td_all[tr], 2, 1), np.nan)
        flat = np.moveaxis(res, 1, 0).reshape(8, -1)
        bias = np.nanmedian(flat, 1)
        sig = np.nanstd(flat - bias[:, None], 1)
        print(f"  d2 env{env}: b={np.round(bias, 2)} σ={np.round(sig, 3)}", flush=True)
        pool = SIG[SIG.env == env][["d_range", "d_rss", "d_rssfp"]].to_numpy()
        for spl in splits:
            ix = np.flatnonzero(split == spl)
            nl = cells["lab_D"][ix][:, rix]
            for k, sc in E["scenarios"].items():
                for s in E["injection_seeds"][spl]:
                    Xi, lab, _, _ = inject_windows(X[ix], rows, sc, s, dist_metric="RANGE", eligible=(nl == 0).all(2),
                                                   delta_pool=pool, delta_metrics=DM)
                    lab = np.where(nl == 1, np.nan, lab)
                    r = Xi[:, rix]
                    avail = np.isfinite(r) & los_all[ix]
                    tres = r - np.moveaxis(td_all[ix], 2, 1) - bias[None, :, None]
                    sc_ = detector_scores(models, Xi, W[ix], nodes[ix], rows, anchors)
                    out += position_eval(r, avail, sc_, lab, tres, pos_all[ix], h_all[ix], bias, sig, Q["d2"],
                                         avail.sum(1) >= 4, (lab == 1).any(1),
                                         dict(set="P3", dataset="d2", split=spl, space=f"env{env}", scenario=f"{k}m",
                                              seed=s, _anc=anc), ("P3", spl, f"{k}m"))
        print(f"  d2 env{env} done", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--datasets", nargs="*", default=["d1", "d2"])
    args = ap.parse_args()
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    code = ecfg["ids"]["F01"]
    geom = json.loads((ROOT / "cache" / "d1_geometry.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = []
    if "d1" in args.datasets:
        rows += run_d1(ecfg, dcfg, code, args.splits, geom)
    if "d2" in args.datasets:
        rows += run_d2(ecfg, dcfg, code, args.splits)
    sfx = "_".join(args.splits) + ("" if set(args.datasets) == {"d1", "d2"} else "_" + "_".join(args.datasets))
    pd.DataFrame(rows).to_csv(OUT / f"metrics_{sfx}.csv", index=False, encoding="utf-8-sig")
    pooled = []
    for (st, spl, scn, sv, cfg, sub), xs in POOL.items():
        x = np.concatenate(xs)
        pooled.append(dict(set=st, split=spl, scenario=scn, solver=sv, config=cfg, subset=sub, n=int(x.size),
                           rmse=float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan,
                           p90=float(np.percentile(x, 90)) if x.size else np.nan,
                           mean=float(x.mean()) if x.size else np.nan))
    pd.DataFrame(pooled).to_csv(OUT / f"pooled_{sfx}.csv", index=False, encoding="utf-8-sig")
    (OUT / f"config_{sfx}.json").write_text(json.dumps(dict(
        plan="doc/실험7_계획.md", detectors=DETECTORS, q=Q, splits=args.splits, datasets=args.datasets,
        exp_config="configs/exp3.yaml", data_config="configs/data.yaml", elapsed_s=time.perf_counter() - t0),
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"done: {len(rows)} rows, {time.perf_counter() - t0:.1f}s", flush=True)
