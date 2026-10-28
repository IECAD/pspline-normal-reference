"""실험 6 (1부): B-spline 조건부 우위 — 정상 데이터 양 (A) · 윈도우 길이 (B) · 관측 결측 (C).
계획(사전 등록): doc/실험6_계획.md. 합성·점수·보정·평가는 실험 4 와 같고, 학습 데이터만 조건에 따라 바꾼다.

기저: cell (M01), PS (K_t = T, SAP; 주), PSA (adaptive K ∈ 5..T, 보조)
조건: A f ∈ {1, .5, .25, .1} (노드당 학습 윈도우 비율, 시드 0-2), B T ∈ {6, 12, 24} (T=12 윈도우를 나누거나 이어 붙임),
      C r ∈ {0, .3, .5, .7} (학습 셀 결측률, 앵커 단위로 모든 지표 함께, 시드 0-2). 기준 조건 (f=1, T=12, r=0) 은 한 번만 계산.
사용법: python experiments/exp6_conditions.py [--splits val] [--datasets d1 d2]
출력: results/exp6/conditions/metrics_<데이터>_<분할>.csv
"""
import argparse
import os
import sys
import time
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
from src.data.config import load_config  # noqa: E402
from src.data.inject import inject_windows  # noqa: E402
from src.data.load import load_windows  # noqa: E402
from src.metrics.anomaly import evaluate  # noqa: E402
from src.model.bank import anchor_scores  # noqa: E402
from src.model.compose import split_row  # noqa: E402

OUT = ROOT / "results" / "exp6" / "conditions"
BASES = {"cell": {"time_basis": "cell"}, "PS": {"K_t": "T"}, "PSA": {"K_t": "adaptive"}}
CONDS = [("base", 1.0, None, 12, 0.0)] + \
        [("A", f, s, 12, 0.0) for f in (0.5, 0.25, 0.1) for s in (0, 1, 2)] + \
        [("B", 1.0, None, T, 0.0) for T in (6, 24)] + \
        [("C", 1.0, s, 12, r) for r in (0.3, 0.5, 0.7) for s in (0, 1, 2)]


def derive(X, meta, cells, T):
    """T=12 윈도우 -> T=6 (반으로 나눔) 또는 T=24 (같은 구간·분할에서 이어진 두 윈도우를 붙임)."""
    if T == 12:
        return X, meta.reset_index(drop=True), cells
    if T == 6:
        m = pd.concat([meta.assign(tidx0=meta.tidx0), meta.assign(tidx0=meta.tidx0 + 6)], ignore_index=True)
        cat = lambda a: np.concatenate([a[:, :, :6], a[:, :, 6:]])
        return cat(X), m, {k: cat(v) for k, v in cells.items()}
    order = meta.sort_values(["seg", "split", "tidx0"]).index.to_numpy()
    pairs, i = [], 0
    while i < len(order) - 1:
        a, b = order[i], order[i + 1]
        if (meta.seg[a] == meta.seg[b] and meta.split[a] == meta.split[b]
                and meta.tidx0[b] == meta.tidx0[a] + 12):
            pairs.append((a, b))
            i += 2
        else:
            i += 1
    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    cat = lambda arr: np.concatenate([arr[a], arr[b]], axis=2)
    return cat(X), meta.loc[a].reset_index(drop=True), {k: cat(v) for k, v in cells.items()}


def apply_cond(Xtr, Wtr, nodes, rows, fam, f, seed, r):
    """학습 데이터에만 조건 적용. 반환: X, W, nodes (부분집합일 수 있음)."""
    if fam == "A" and f < 1:
        rng = np.random.default_rng(1000 + seed)
        keep = []
        for nd in sorted(set(nodes)):
            ix = np.flatnonzero(nodes == nd)
            keep += list(rng.choice(ix, max(1, int(round(f * len(ix)))), replace=False))
        keep = np.sort(keep)
        return Xtr[keep], Wtr[keep], nodes[keep]
    if fam == "C" and r > 0:
        rng = np.random.default_rng(2000 + seed)
        anchors = sorted({split_row(x)[1] for x in rows})
        drop = rng.random((len(Xtr), len(anchors), Xtr.shape[2])) < r              # 앵커 단위 결측
        aix = np.array([anchors.index(split_row(x)[1]) for x in rows])
        return Xtr, Wtr * (~drop[:, aix, :]), nodes
    return Xtr, Wtr, nodes


def spec_for(bid, T, bases=None):
    s = dict((bases or BASES)[bid])
    if s.get("K_t") == "T":
        s["K_t"] = T
    return s


def default_apply(Xtr, Wtr, nodes, rows, c):
    fam, f, cseed, _, r = c
    return apply_cond(Xtr, Wtr, nodes, rows, fam, f, cseed or 0, r)


def default_level(c):
    fam, f, _, T, r = c
    return {"base": "기준", "A": f, "B": T, "C": r}[fam]


def run(ds, ecfg, dcfg, splits, conds=None, apply=None, level_of=None, bases=None):
    """conds: (계열, f, 조건 시드, T, 결측 인자) 목록. apply(X, W, nodes, rows, cond), level_of(cond) 로 다른 조건도 돌린다."""
    conds, apply, level_of = conds or CONDS, apply or default_apply, level_of or default_level
    bases = bases or BASES
    code = ecfg["ids"]["F01"]
    E = ecfg["datasets"][ds]
    units = E["spaces"].items() if ds == "d1" else [(f"env{e}", None) for e in E["envs"]]
    out = []
    for unit, sp in units:
        if ds == "d1":
            full = load_windows("d1", sp["normal"], dcfg)
            cells0 = {}
        else:
            full = load_windows("d2", f"{unit}_{E['channel_name']}", dcfg)
            cells0 = {k: full["cells"][k] for k in ("mask", "lab_D")}
        rows = full["rows"]
        for T in sorted({c[3] for c in conds}):
            X, meta, cells = derive(full["X"], full["meta"], cells0, T)
            split = meta["split"].to_numpy()
            if ds == "d1":
                nodes = np.array([f"{unit}/{n.split('/')[-1]}" for n in meta["node"]])
                W = np.isfinite(X).astype(float)
            else:
                nodes = np.array([f"{unit}/{int(t)}" for t in meta["tidx0"]])
                W = cells["mask"]
            anchors = [f"A{a}" for a in range(1, 5 if ds == "d1" else 9)]
            # 평가 주입 (T 마다 한 번)
            inj = {}
            for spl in splits:
                ix = np.flatnonzero(split == spl)
                if ds == "d1":
                    for scn, sc in E["scenarios"].items():
                        for s in E["injection_seeds"][spl]:
                            Xi, lab, _, _ = inject_windows(X[ix], rows, sc, s)
                            inj[(spl, scn, s)] = (ix, Xi, lab)
                else:
                    rix = [rows.index(f"RANGE_{a}") for a in anchors]
                    nl = cells["lab_D"][ix][:, rix]
                    pool = SIG[SIG.env == int(unit[3:])][["d_range", "d_rss", "d_rssfp"]].to_numpy()
                    scen = [(k, sc, None) for k, sc in E["scenarios"].items()] + \
                           [(f"{k}m", sc, pool) for k, sc in E["scenarios"].items()]
                    for scn, sc, dp in scen:
                        for s in E["injection_seeds"][spl]:
                            Xi, lab, _, _ = inject_windows(X[ix], rows, sc, s, dist_metric="RANGE",
                                                           eligible=(nl == 0).all(2), delta_pool=dp,
                                                           delta_metrics=["RANGE", "RSS", "RSS_FP"])
                            inj[(spl, scn, s)] = (ix, Xi, np.where(nl == 1, np.nan, lab))
            tr, va = split == "train", split == "val"
            for c in conds:
                fam, f, cseed, Tc, r = c
                if Tc != T:
                    continue
                Xtr, Wtr, ntr = apply(X[tr], W[tr], nodes[tr], rows, c)
                for bid in bases:
                    spec = full_spec(spec_for(bid, T, bases), ecfg)
                    t0 = time.perf_counter()
                    comp, bank, K = build(code, spec, ecfg, ds, Xtr, Wtr, rows, ntr, X[va], W[va], nodes[va],
                                          adaptive_ks=list(range(5, T + 1)))
                    level = level_of(c)
                    for (spl, scn, s), (ix, Xi, lab) in inj.items():
                        Xs, Ws = comp.transform(Xi, W[ix], rows)
                        A = anchor_scores(bank.score(Xs, nodes[ix], Ws), comp.anchor_of_row, anchors)
                        seqs = [(A[n], lab[n]) for n in range(len(A))]
                        out.append(dict(dataset=ds, space=unit, split=spl, scenario=scn, seed=s, family=fam,
                                        level=level, cond_seed=-1 if cseed is None else cseed, T=T, basis=bid,
                                        K_used=K, n_train=len(Xtr), threshold=bank.thr,
                                        **evaluate(seqs, bank.thr, "time" if ds == "d1" else E["level"])))
                print(f"  {ds} {unit} {fam} f={f} T={T} r={r} seed={cseed}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--datasets", nargs="*", default=["d1", "d2"])
    args = ap.parse_args()
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        t0 = time.perf_counter()
        rows = run(ds, ecfg, dcfg, args.splits)
        pd.DataFrame(rows).to_csv(OUT / f"metrics_{ds}_{'_'.join(args.splits)}.csv", index=False,
                                  encoding="utf-8-sig")
        print(f"{ds}: {len(rows)} rows, {time.perf_counter() - t0:.1f}s", flush=True)
