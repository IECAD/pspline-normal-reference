"""실험 7 추가 비교 (doc/실험7_계획.md "추가 비교", 사전 등록 2026-10-12, 사후·탐색 분석). D1 만.

구성: none, PS-soft(정답 노드, 원 실험), PS-soft(추정 노드 2단계, 주) / 1단계, cell-soft(추정 노드 2단계),
      시간이력-soft (위치·학습 없음), 참 거리 z-soft (oracle) + 해법 Rwgh (Chen 1999, 탐지 없음)
판정 (테스트): RQ6 현실 노드 = PS-soft(추정) 가 P1 SC1·SC2 에서 LS·WLS 개선율 ≥ 5 %, 짝 단위 80 % 이상.
              정상 모델 필요성 = PS-soft(추정) RMSE 가 시간이력-soft 보다 SC1·SC2, LS·WLS 모두 5 % 넘게 낮음.
사용법: python experiments/exp7_extra.py --splits val | test
출력: results/exp7/extra/metrics_<분할>.csv, pooled_<분할>.csv, summary_<분할>.md, node_accuracy_<분할>.csv
"""
import argparse
import itertools
import json
import os
import sys
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
from exp7_positioning import Q, d1_points, detector_scores, fit_models, w_soft  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.inject import inject_windows  # noqa: E402
from src.data.load import load_windows  # noqa: E402
from src.positioning.solvers import gauss_newton, solve_all  # noqa: E402

OUT = ROOT / "results" / "exp7" / "extra"
POOL = defaultdict(list)
SOLVERS = ["LS", "WLS", "Huber", "EKF"]


def flat(a):
    """(N, A, T) -> (N*T, A)."""
    return np.moveaxis(a, 1, 2).reshape(-1, a.shape[1])


def window_median_pos(r, w, anc, h, b):
    N, A, T = r.shape
    P = gauss_newton(flat(r), flat(w), anc, np.full((N * T, A), h), b).reshape(N, T, 2)
    return np.nanmedian(P, axis=1)


def nearest(Pm, pts, space):
    names, xy = pts.index.to_numpy(), pts[["x_m", "y_m"]].to_numpy()
    j = np.argmin(((Pm[:, None, :] - xy[None]) ** 2).sum(-1), axis=1)
    return np.array([f"{space}/{names[k]}" for k in j])


def rwgh(r, avail, anc, h, b):
    """Chen (1999) 잔차 가중. r, avail: (E, A). 앵커 3개 이상 부분집합마다 LS, 가중 1 / (Res / (|S| − 2))."""
    E, A = r.shape
    num, den = np.zeros((E, 2)), np.zeros(E)
    for k in range(3, A + 1):
        for sub in itertools.combinations(range(A), k):
            m = np.zeros(A, bool)
            m[list(sub)] = True
            ok = avail[:, m].all(1)
            if not ok.any():
                continue
            P = gauss_newton(r, (avail & m[None]).astype(float), anc, np.full((E, A), h), b)
            d = np.sqrt(((P[:, None, :] - anc[None]) ** 2).sum(-1) + h ** 2)
            res = np.where(m[None] & avail, (r - b[None] - d) ** 2, 0.0).sum(1)
            wt = np.where(ok, 1.0 / np.maximum(res / (k - 2), 1e-9), 0.0)
            num += wt[:, None] * np.nan_to_num(P)
            den += wt
    return np.where(den[:, None] > 0, num / np.maximum(den, 1e-300)[:, None], np.nan)


def summarize(e, valid, anom, key, tag, rows):
    for sub, m in (("all", valid), ("anom", valid & anom), ("norm", valid & ~anom)):
        x = e[m & np.isfinite(e)]
        POOL[key + (sub,)].append(x)
        rows.append(dict(**tag, subset=sub, n=int(x.size), rmse=float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan))


def run(ecfg, dcfg, code, splits, geom):
    E = ecfg["datasets"]["d1"]
    anchors = [f"A{a}" for a in range(1, 5)]
    rows, nacc = [], []
    for space, sp in E["spaces"].items():
        g = geom[sp["normal"]]
        anc = np.array([dcfg["d1"]["anchors"][space][a] for a in range(1, 5)], float)
        dz, bias = g["dz"], np.array([g["bias"][str(a)] for a in range(1, 5)])
        full = load_windows("d1", sp["normal"], dcfg)
        X, meta, rows_ = full["X"], full["meta"], full["rows"]
        nodes = np.array([f"{space}/{n.split('/')[-1]}" for n in meta["node"]])
        W, split = np.isfinite(X).astype(float), meta["split"].to_numpy()
        dix = [rows_.index(f"DIST_{a}") for a in anchors]
        models = fit_models(code, ecfg, "d1", X, W, rows_, nodes, split)
        pts_train = d1_points(dcfg, sp["normal"])

        def geo(meta_, f):
            pts = d1_points(dcfg, f)
            xy = pts.loc[[n.split("/")[-1] for n in meta_["node"]]].to_numpy()
            return xy, np.sqrt(((xy[:, None, :] - anc[None]) ** 2).sum(-1) + dz ** 2)

        xy, td = geo(meta, sp["normal"])
        tr, va = split == "train", split == "val"
        res_tr = X[tr][:, dix] - td[tr][..., None] - bias[None, :, None]
        sig = np.nanstd(np.moveaxis(res_tr, 1, 0).reshape(4, -1), 1)
        mu_t = np.nanmean(np.moveaxis(res_tr, 1, 0).reshape(4, -1), 1)
        th_score = lambda r: np.abs(r - np.nanmedian(r, 2, keepdims=True)) / (
            1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r, 2, keepdims=True)), 2, keepdims=True) + 1e-3)
        tz_score = lambda tres: np.abs(tres - mu_t[None, :, None]) / (sig[None, :, None] + 1e-6)
        rv = X[va][:, dix]
        thr_th = float(np.nanpercentile(th_score(rv), 97.5))
        thr_tz = float(np.nanpercentile(tz_score(rv - td[va][..., None] - bias[None, :, None]), 97.5))

        def evaluate_set(Xw, lab, xy_w, td_w, true_nodes, tag, key):
            r = Xw[:, dix]
            avail = np.isfinite(r)
            Ww = np.isfinite(Xw).astype(float)
            tres = r - td_w[..., None] - bias[None, :, None]
            sc_true = detector_scores(models, Xw, Ww, true_nodes, rows_, anchors)
            Pm1 = window_median_pos(r, avail.astype(float), anc, dz, bias)
            n1 = nearest(Pm1, pts_train, space)
            sc1 = detector_scores(models, Xw, Ww, n1, rows_, anchors)
            Pm2 = window_median_pos(r, w_soft(*sc1["PS"], avail), anc, dz, bias)
            n2 = nearest(Pm2, pts_train, space)
            sc2 = detector_scores(models, Xw, Ww, n2, rows_, anchors)
            nacc.append(dict(**{k: v for k, v in tag.items()}, acc_pass1=float(np.mean(n1 == true_nodes)),
                             acc_pass2=float(np.mean(n2 == true_nodes))))
            Wset = {"none": avail.astype(float),
                    "PS-soft(정답 노드)": w_soft(*sc_true["PS"], avail),
                    "PS-soft(추정 1단계)": w_soft(*sc1["PS"], avail),
                    "PS-soft(추정 노드)": w_soft(*sc2["PS"], avail),
                    "cell-soft(추정 노드)": w_soft(*sc2["cell"], avail),
                    "시간이력-soft": w_soft(th_score(r), thr_th, avail),
                    "참거리z-soft(oracle)": w_soft(tz_score(tres), thr_tz, avail)}
            pos = np.repeat(xy_w[:, None, :], Xw.shape[2], axis=1)
            valid, anom = avail.sum(1) >= 3, (np.nan_to_num(lab) == 1).any(1)
            for cfg, w in Wset.items():
                sol = solve_all(np.moveaxis(r, 1, 2), np.moveaxis(w, 1, 2), anc, np.full(4, dz), bias, sig, Q["d1"])
                for sv in SOLVERS:
                    e = np.sqrt(((sol[sv] - pos) ** 2).sum(-1))
                    summarize(e, valid, anom, key + (sv, cfg), dict(**tag, solver=sv, config=cfg), rows)
            Pr = rwgh(flat(r), flat(avail), anc, dz, bias).reshape(r.shape[0], r.shape[2], 2)
            summarize(np.sqrt(((Pr - pos) ** 2).sum(-1)), valid, anom, key + ("Rwgh", "none"),
                      dict(**tag, solver="Rwgh", config="none"), rows)

        for spl in splits:
            ix = np.flatnonzero(split == spl)
            for scn, sc in E["scenarios"].items():
                for s in E["injection_seeds"][spl]:
                    Xi, lab, _, _ = inject_windows(X[ix], rows_, sc, s)
                    evaluate_set(Xi, lab, xy[ix], td[ix], nodes[ix],
                                 dict(set="P1", split=spl, space=space, scenario=scn, seed=s), ("P1", spl, scn))
            if spl == "test":
                for f in sp["real"]:
                    w = load_windows("d1", f, dcfg)
                    nd = np.array([f"{space}/{n.split('/')[-1]}" for n in w["meta"]["node"]])
                    keep = np.array([n in models["PS"][1].M for n in nd])
                    mr = w["meta"][keep].reset_index(drop=True)
                    xy_r, td_r = geo(mr, f)
                    evaluate_set(w["X"][keep], w["cells"]["lab_A2"][keep][:, dix], xy_r, td_r, nd[keep],
                                 dict(set="P2", split="test", space=space, scenario=f, seed=-1), ("P2", "test", f))
        print(f"  {space} done", flush=True)
    return rows, nacc


def judge(P, M):
    L, claims = [], {}
    base = "none"
    for st, scns in (("P1", ["SC1", "SC2", "SC3"]), ("P2", sorted(P[P.set == "P2"].scenario.unique()))):
        for scn in scns:
            d = P[(P.set == st) & (P.scenario == scn) & (P.subset == "all")]
            if d.empty:
                continue
            t = d.pivot_table(index="solver", columns="config", values="rmse")
            cfgs = [c for c in ["none", "PS-soft(정답 노드)", "PS-soft(추정 노드)", "PS-soft(추정 1단계)",
                                "cell-soft(추정 노드)", "시간이력-soft", "참거리z-soft(oracle)"] if c in t.columns]
            L += [f"## {st} {scn} (RMSE m)", "", "| 해법 | " + " | ".join(cfgs) + " |", "|" + "---|" * (len(cfgs) + 1)]
            for sv in SOLVERS:
                L.append(f"| {sv} | " + " | ".join(f"{t.loc[sv, c]:.3f}" for c in cfgs) + " |")
            if "Rwgh" in t.index:
                L.append(f"| Rwgh (탐지 없음) | {t.loc['Rwgh', 'none']:.3f} |" + " |" * (len(cfgs) - 1))
            L.append("")
            if st == "P1":
                u = M[(M.set == st) & (M.scenario == scn) & (M.subset == "all")]
                for sv in ("LS", "WLS"):
                    uu = u[u.solver == sv].pivot_table(index=["space", "seed"], columns="config", values="rmse")
                    imp = 1 - t.loc[sv, "PS-soft(추정 노드)"] / t.loc[sv, base]
                    frac = float((uu["PS-soft(추정 노드)"] < uu[base]).mean())
                    vs_th = 1 - t.loc[sv, "PS-soft(추정 노드)"] / t.loc[sv, "시간이력-soft"]
                    claims[f"{scn}/{sv}"] = dict(improvement=imp, frac_units=frac, meets=bool(imp >= 0.05 and frac >= 0.8),
                                                 better_than_timehist=vs_th, needed=bool(vs_th > 0.05))
    rq6 = all(claims.get(f"{s}/{sv}", {}).get("meets", False) for s in ("SC1", "SC2") for sv in ("LS", "WLS"))
    need = all(claims.get(f"{s}/{sv}", {}).get("needed", False) for s in ("SC1", "SC2") for sv in ("LS", "WLS"))
    L += ["## 판정 값", "", "```", json.dumps(claims, indent=1, ensure_ascii=False, default=float), "```", "",
          f"**RQ6 (현실 노드) 충족: {rq6}. 정상 모델 필요성 (시간이력 대비 5 % 초과): {need}**"]
    return L, dict(rq6_realistic=rq6, normal_model_needed=need, claims=claims)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["val"])
    args = ap.parse_args()
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    geom = json.loads((ROOT / "cache" / "d1_geometry.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    rows, nacc = run(ecfg, dcfg, ecfg["ids"]["F01"], args.splits, geom)
    sfx = "_".join(args.splits)
    M = pd.DataFrame(rows)
    M.to_csv(OUT / f"metrics_{sfx}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(nacc).to_csv(OUT / f"node_accuracy_{sfx}.csv", index=False, encoding="utf-8-sig")
    P = pd.DataFrame([dict(set=k[0], split=k[1], scenario=k[2], solver=k[3], config=k[4], subset=k[5],
                           n=int(np.concatenate(v).size), rmse=float(np.sqrt(np.mean(np.concatenate(v) ** 2))))
                      for k, v in POOL.items()])
    P.to_csv(OUT / f"pooled_{sfx}.csv", index=False, encoding="utf-8-sig")
    L, dec = judge(P, M)
    na = pd.DataFrame(nacc).groupby(["set", "scenario"])[["acc_pass1", "acc_pass2"]].mean().round(3)
    L = [f"# 실험 7 추가 비교 ({sfx})", "", "자동 생성: `experiments/exp7_extra.py`. 계획: `doc/실험7_계획.md` \"추가 비교\".",
         "", "노드 지정 정확도:", "", "```", na.to_string(), "```", ""] + L
    (OUT / f"summary_{sfx}.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / f"decision_{sfx}.json").write_text(json.dumps(dec, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))
