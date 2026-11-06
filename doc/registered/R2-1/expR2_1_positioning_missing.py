"""실험 R2-1: 학습 기록이 불완전할 때의 측위 효과 (doc/실험R2-1_계획.md, 사전 등록 2026-11-06, 지문 doc/R2_등록_지문.md).

실험 7(doc/실험7_계획.md)의 측위 경로에서 탐지 기준의 학습 몫에만 실험 6·R1-4 의 결측 마스크를 적용한다.
측위 쪽 값(앵커 좌표, 높이 차, 편향 b, σ)은 실험 7 그대로(완전한 학습 데이터)이고, 바뀌는 것은 탐지 기준뿐이다.
P1 D1 주입 (주, 추정 노드 2단계 + 정답 노드), P2 D1 실제 NLOS (테스트, 서술), P3 D2 주입 (정답 노드, 서술).
탐지기: PS, CG (셀 평균, 단위 평균 채움), CF (행 평균 채움), CI (보간). 반영: none / X-soft.
사용법: python experiments/expR2_1_positioning_missing.py --splits val | test  [--datasets d1 d2]
출력: results/expR2/positioning_missing/units_<분할>.csv, pooled_<분할>.csv, node_accuracy_<분할>.csv,
      summary_<분할>.md, decision_<분할>.json
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
from exp6_comparators_missing import CONDS_D1, CONDS_D2, apply, level_of  # noqa: E402
from exp7_extra import nearest, window_median_pos  # noqa: E402
from exp7_positioning import DM, Q, d1_points, detector_scores, w_soft  # noqa: E402
from node_model_eval import SIG, build, full_spec  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.inject import inject_windows  # noqa: E402
from src.data.load import load_windows  # noqa: E402
from src.positioning.solvers import solve_all  # noqa: E402

OUT = ROOT / "results" / "expR2" / "positioning_missing"
DETECTORS = {"PS": {"K_t": 12}, "CG": {"time_basis": "cell"}, "CF": {"time_basis": "cellfill"},
             "CI": {"time_basis": "cellinterp"}}
SOLVERS = ["LS", "WLS", "Huber", "EKF"]
POOL = defaultdict(list)                   # (set, split, family, level, scenario, solver, config, subset) -> 스텝 오차


def fit_masked(code, ecfg, ds, X, W, rows, nodes, split, c):
    """학습 몫에만 조건 c 의 마스크를 적용해 탐지기들을 적합한다. 보정(검증) 윈도우는 그대로. 반환: models, 학습 결측률."""
    tr, va = split == "train", split == "val"
    _, Wm, _ = apply(X[tr], W[tr], nodes[tr], rows, c)
    models = {}
    for name, sp in DETECTORS.items():
        comp, bank, _ = build(code, full_spec(sp, ecfg), ecfg, ds, X[tr], Wm, rows, nodes[tr], X[va], W[va], nodes[va])
        models[name] = (comp, bank)
    return models, float(1 - Wm.sum() / max(W[tr].sum(), 1))


def record(e, valid, anom, key, tag, out):
    for sub, m in (("all", valid), ("anom", valid & anom), ("norm", valid & ~anom)):
        x = e[m & np.isfinite(e)]
        POOL[key + (sub,)].append(x)
        out.append(dict(**tag, subset=sub, n=int(x.size), sse=float(np.sum(x ** 2)),
                        rmse=float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan))


def solve_and_record(r, Wset, pos, anc, h, b, sig, q, valid, anom, key, tag, out, solvers):
    for cfg, w in Wset.items():
        sol = solve_all(np.moveaxis(r, 1, 2), np.moveaxis(w, 1, 2), anc, h, b, sig, q)
        for sv in solvers:
            e = np.sqrt(((sol[sv] - pos) ** 2).sum(-1))
            record(e, valid, anom, key + (sv, cfg), dict(**tag, solver=sv, config=cfg), out)


# ----------------------------------------------------------------------------- D1
def run_d1(ecfg, dcfg, code, splits, geom, conds):
    E = ecfg["datasets"]["d1"]
    anchors = [f"A{a}" for a in range(1, 5)]
    out, nacc = [], []
    for space, sp in E["spaces"].items():
        g = geom[sp["normal"]]
        anc = np.array([dcfg["d1"]["anchors"][space][a] for a in range(1, 5)], float)
        dz, bias = g["dz"], np.array([g["bias"][str(a)] for a in range(1, 5)])
        full = load_windows("d1", sp["normal"], dcfg)
        X, meta, rows = full["X"], full["meta"], full["rows"]
        nodes = np.array([f"{space}/{n.split('/')[-1]}" for n in meta["node"]])
        W, split = np.isfinite(X).astype(float), meta["split"].to_numpy()
        dix = [rows.index(f"DIST_{a}") for a in anchors]
        pts_train = d1_points(dcfg, sp["normal"])

        def geo(meta_, f):
            pts = d1_points(dcfg, f)
            xy = pts.loc[[n.split("/")[-1] for n in meta_["node"]]].to_numpy()
            return xy, np.sqrt(((xy[:, None, :] - anc[None]) ** 2).sum(-1) + dz ** 2)

        xy, td = geo(meta, sp["normal"])
        tr = split == "train"
        res_tr = X[tr][:, dix] - td[tr][..., None] - bias[None, :, None]                 # 측위 σ: 완전한 학습 데이터 (실험 7)
        sig = np.nanstd(np.moveaxis(res_tr, 1, 0).reshape(4, -1), 1)

        def evaluate_set(models, Xw, lab, xy_w, true_nodes, tag, key):
            r = Xw[:, dix]
            avail = np.isfinite(r)
            Ww = np.isfinite(Xw).astype(float)
            sc_true = detector_scores(models, Xw, Ww, true_nodes, rows, anchors)
            Pm1 = window_median_pos(r, avail.astype(float), anc, dz, bias)
            n1 = nearest(Pm1, pts_train, space)
            sc1 = detector_scores(models, Xw, Ww, n1, rows, anchors)
            Wset = {"none": avail.astype(float)}
            for name in DETECTORS:                                           # 탐지기마다 자기 점수로 2단계
                Pm2 = window_median_pos(r, w_soft(*sc1[name], avail), anc, dz, bias)
                n2 = nearest(Pm2, pts_train, space)
                sc2 = detector_scores({name: models[name]}, Xw, Ww, n2, rows, anchors)
                Wset[f"{name}-soft(추정 노드)"] = w_soft(*sc2[name], avail)
                Wset[f"{name}-soft(정답 노드)"] = w_soft(*sc_true[name], avail)
                nacc.append(dict(**tag, detector=name, acc_pass1=float(np.mean(n1 == true_nodes)),
                                 acc_pass2=float(np.mean(n2 == true_nodes))))
            pos = np.repeat(xy_w[:, None, :], Xw.shape[2], axis=1)
            valid, anom = avail.sum(1) >= 3, (np.nan_to_num(lab) == 1).any(1)
            solve_and_record(r, Wset, pos, anc, np.full(4, dz), bias, sig, Q["d1"], valid, anom, key, tag, out, SOLVERS)

        for c in conds:
            models, miss = fit_masked(code, ecfg, "d1", X, W, rows, nodes, split, c)
            fam, lvl, cs = c[0], str(level_of(c)), -1 if c[2] is None else c[2]
            ctag = dict(family=fam, level=lvl, cond_seed=cs, train_missing=miss)
            for spl in splits:
                ix = np.flatnonzero(split == spl)
                for scn, sc in E["scenarios"].items():
                    for s in E["injection_seeds"][spl]:
                        Xi, lab, _, _ = inject_windows(X[ix], rows, sc, s)
                        evaluate_set(models, Xi, lab, xy[ix], nodes[ix],
                                     dict(set="P1", dataset="d1", split=spl, space=space, scenario=scn, seed=s, **ctag),
                                     ("P1", spl, fam, lvl, scn))
                if spl == "test":                                              # P2 실제 NLOS (서술)
                    for f in sp["real"]:
                        w = load_windows("d1", f, dcfg)
                        nd = np.array([f"{space}/{n.split('/')[-1]}" for n in w["meta"]["node"]])
                        keep = np.array([n in models["PS"][1].M for n in nd])
                        if not keep.any():
                            continue
                        mr = w["meta"][keep].reset_index(drop=True)
                        xy_r, _ = geo(mr, f)
                        evaluate_set(models, w["X"][keep], w["cells"]["lab_A2"][keep][:, dix], xy_r, nd[keep],
                                     dict(set="P2", dataset="d1", split="test", space=space, scenario=f, seed=-1, **ctag),
                                     ("P2", "test", fam, lvl, f))
            print(f"  d1 {space} {fam} {lvl} s={cs} (train missing {miss:.3f}) done", flush=True)
    return out, nacc


# ----------------------------------------------------------------------------- D2 (정답 노드, 서술)
def run_d2(ecfg, dcfg, code, splits, conds):
    E = ecfg["datasets"]["d2"]
    anchors = [f"A{a}" for a in range(1, 9)]
    meta_all = pd.read_parquet(ROOT / "cache" / "d2_meta.parquet")
    out = []
    for env in E["envs"]:
        m = meta_all[(meta_all.env == env) & (meta_all.ch == dcfg["d2"]["channel"])]
        A = m.groupby("anchor")[["X_ANCHOR", "Y_ANCHOR", "Z_ANCHOR"]].first().reindex(range(1, 9)).to_numpy()
        tagp = m.groupby("path_idx")[["X_TAG", "Y_TAG", "Z_TAG"]].first()
        anc = A[:, :2]
        full = load_windows("d2", f"env{env}_{E['channel_name']}", dcfg)
        X, meta, rows, cells = full["X"], full["meta"], full["rows"], full["cells"]
        nodes = np.array([f"env{env}/{int(t)}" for t in meta["tidx0"]])
        W, split = cells["mask"], meta["split"].to_numpy()
        T = X.shape[2]
        rix = [rows.index(f"RANGE_{a}") for a in anchors]
        pidx = meta["tidx0"].to_numpy()[:, None] + np.arange(T)[None]
        G = tagp.reindex(pidx.ravel()).to_numpy().reshape(len(X), T, 3)
        pos_all, h_all = G[..., :2], G[..., 2][..., None] - A[None, None, :, 2]
        td_all = np.sqrt(((pos_all[:, :, None, :] - anc[None, None]) ** 2).sum(-1) + h_all ** 2)
        los_all = cells["lab_D"][:, rix] == 0
        tr = split == "train"
        res = np.where(los_all[tr], X[tr][:, rix] - np.moveaxis(td_all[tr], 2, 1), np.nan)
        flat = np.moveaxis(res, 1, 0).reshape(8, -1)
        bias = np.nanmedian(flat, 1)
        sig = np.nanstd(flat - bias[:, None], 1)
        pool = SIG[SIG.env == env][["d_range", "d_rss", "d_rssfp"]].to_numpy()
        for c in conds:
            models, miss = fit_masked(code, ecfg, "d2", X, W, rows, nodes, split, c)
            fam, lvl, cs = c[0], str(level_of(c)), -1 if c[2] is None else c[2]
            ctag = dict(family=fam, level=lvl, cond_seed=cs, train_missing=miss)
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
                        sc_ = detector_scores(models, Xi, W[ix], nodes[ix], rows, anchors)
                        Wset = {"none": avail.astype(float)}
                        Wset.update({f"{n}-soft(정답 노드)": w_soft(*sc_[n], avail) for n in DETECTORS})
                        solve_and_record(r, Wset, pos_all[ix], anc, h_all[ix], bias, sig, Q["d2"],
                                         avail.sum(1) >= 4, (lab == 1).any(1), ("P3", spl, fam, lvl, f"{k}m"),
                                         dict(set="P3", dataset="d2", split=spl, space=f"env{env}", scenario=f"{k}m",
                                              seed=s, **ctag), out, ["LS"])
            print(f"  d2 env{env} {fam} {lvl} s={cs} (train missing {miss:.3f}) done", flush=True)
    return out


# ----------------------------------------------------------------------------- 집계·판정
def cluster_ci(U, cfg_x, cfg_ref, B=5000, seed=0):
    """공간 군집 2단계 부트스트랩: RMSE_x / RMSE_ref − 1 의 95 % 구간. U: 한 (계열, 시나리오, 해법)의 단위 행(subset all)."""
    rng = np.random.default_rng(seed)
    piv = U.pivot_table(index=["space", "seed", "cond_seed"], columns="config", values=["sse", "n"], aggfunc="sum")
    spaces = piv.index.get_level_values(0).unique().to_numpy()
    blocks = [piv.xs(s_, level=0) for s_ in spaces]
    vals = []
    for _ in range(B):
        num_x = den_x = num_r = den_r = 0.0
        for j in rng.integers(0, len(blocks), len(blocks)):
            bk = blocks[j]
            k = rng.integers(0, len(bk), len(bk))
            num_x += bk[("sse", cfg_x)].to_numpy()[k].sum(); den_x += bk[("n", cfg_x)].to_numpy()[k].sum()
            num_r += bk[("sse", cfg_ref)].to_numpy()[k].sum(); den_r += bk[("n", cfg_ref)].to_numpy()[k].sum()
        vals.append(np.sqrt(num_x / den_x) / np.sqrt(num_r / den_r) - 1)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def pooled_table(U):
    P = U.groupby(["set", "split", "family", "level", "scenario", "solver", "config", "subset"])[["sse", "n"]].sum()
    P["rmse"] = np.sqrt(P.sse / P.n)
    return P.reset_index()


def judge(U, P, split):
    L, claims = [], {}
    fams = [("C", "0.5"), ("C", "0.7"), ("D", "pi=0.7,L=4"), ("P", "P3")]
    node = "추정 노드"
    for fam, lvl in [("base", "기준")] + fams:
        L += [f"## P1 {fam} {lvl} — RMSE (m), 모든 스텝", ""]
        for scn in ("SC1", "SC2", "SC3"):
            d = P[(P.set == "P1") & (P.family == fam) & (P.level == lvl) & (P.scenario == scn) & (P.subset == "all")]
            if d.empty:
                continue
            t = d.pivot_table(index="solver", columns="config", values="rmse")
            cfgs = ["none"] + [f"{n}-soft({nd})" for nd in ("추정 노드", "정답 노드") for n in DETECTORS]
            L += [f"### {scn}", "", "| 해법 | " + " | ".join(cfgs) + " |", "|" + "---|" * (len(cfgs) + 1)]
            for sv in SOLVERS:
                L.append(f"| {sv} | " + " | ".join(f"{t.loc[sv, c_]:.3f}" for c_ in cfgs) + " |")
            L.append("")
            if fam == "base" or scn == "SC3":
                continue
            for sv in ("LS", "WLS"):
                u = U[(U.set == "P1") & (U.family == fam) & (U.level == lvl) & (U.scenario == scn) & (U.solver == sv)
                      & (U.subset == "all")]
                uu = u.pivot_table(index=["space", "seed", "cond_seed"], columns="config", values="rmse")
                ps = f"PS-soft({node})"
                imp = 1 - t.loc[sv, ps] / t.loc[sv, "none"]
                frac = float((uu[ps] < uu["none"]).mean())
                ent = dict(improvement=imp, frac_units=frac, a_meets=bool(imp >= 0.05 and frac >= 0.8))
                for x in ("CG", "CF", "CI"):
                    cx = f"{x}-soft({node})"
                    rel = t.loc[sv, cx] / t.loc[sv, ps] - 1
                    ent[f"rel_{x}"] = rel
                    ent[f"verdict_{x}"] = "PS 우위" if rel > 0.02 else ("대등" if rel >= -0.02 else f"{x} 우위")
                    ent[f"ci_{x}"] = cluster_ci(u, cx, ps)
                dn = P[(P.set == "P1") & (P.family == fam) & (P.level == lvl) & (P.scenario == scn) & (P.subset == "norm")
                       & (P.solver == sv)].set_index("config").rmse
                ent["normal_step_cost"] = {x: bool(dn[f"{x}-soft({node})"] > 1.02 * dn["none"]) for x in DETECTORS}
                claims[f"{fam}|{lvl}/{scn}/{sv}"] = ent
    a_all = all(v["a_meets"] for v in claims.values())
    L += ["## 판정 값 (추정 노드, LS·WLS, SC1·SC2)", "", "```", json.dumps(claims, indent=1, ensure_ascii=False, default=float),
          "```", "", f"**R2-1a (결측에서도 측위 이득 유지) 충족: {a_all}**"]
    for st in ("P2", "P3"):
        d = P[(P.set == st) & (P.subset == "all") & (P.solver == "LS")]
        if d.empty:
            continue
        L += ["", f"## {st} (서술, LS, RMSE m)", "", "```",
              d.pivot_table(index=["family", "level", "scenario"], columns="config", values="rmse").round(3).to_string(),
              "```"]
    return L, dict(split=split, r2_1a=a_all, claims=claims)


def main():
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
    rows, nacc = [], []
    if "d1" in args.datasets:
        r1, nacc = run_d1(ecfg, dcfg, code, args.splits, geom, CONDS_D1)
        rows += r1
    if "d2" in args.datasets:
        rows += run_d2(ecfg, dcfg, code, args.splits, CONDS_D2)
    sfx = "_".join(args.splits) + ("" if set(args.datasets) == {"d1", "d2"} else "_" + "_".join(args.datasets))
    U = pd.DataFrame(rows)
    U.to_csv(OUT / f"units_{sfx}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(nacc).to_csv(OUT / f"node_accuracy_{sfx}.csv", index=False, encoding="utf-8-sig")
    P = pooled_table(U)
    p90 = {k: float(np.percentile(np.concatenate(v), 90)) if sum(len(x) for x in v) else np.nan for k, v in POOL.items()}
    P["p90"] = [p90.get((r.set, r.split, r.family, r.level, r.scenario, r.solver, r.config, r.subset), np.nan)
                for r in P.itertuples()]
    P.to_csv(OUT / f"pooled_{sfx}.csv", index=False, encoding="utf-8-sig")
    L, dec = judge(U, P, sfx)
    na = pd.DataFrame(nacc)
    if not na.empty:
        na = na.groupby(["set", "family", "level", "scenario", "detector"])[["acc_pass1", "acc_pass2"]].mean().round(3)
    L = [f"# 실험 R2-1 ({sfx})", "", "자동 생성: `experiments/expR2_1_positioning_missing.py`. 계획: `doc/실험R2-1_계획.md`.", "",
         f"실행 시간 {time.perf_counter() - t0:.0f} s.", "", "노드 지정 정확도:", "", "```",
         na.to_string() if len(na) else "-", "```", ""] + L
    (OUT / f"summary_{sfx}.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / f"decision_{sfx}.json").write_text(json.dumps(dec, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"done: {len(U)} unit rows, {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
