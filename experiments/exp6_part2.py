"""실험 6 (2부): 임계 백분위 (F) · 잡음 (G) · 앵커 수 (H, D2) · 최종 T (I) · 학습 오염 (J).
계획(사전 등록): doc/실험6_계획.md "2부". 합성·점수·보정·평가는 1부(exp6_conditions.py)와 같다.

기저: cell (M01), PS (K_t = T, SAP). H 는 OLS (nc, α = 100, 실험 1 D2 선택 구성) 를 더한다.
  F  p ∈ {95, 97.5, 99, 99.5}: 기준 조건 모델 하나에서 임계값만 바꿈 (검증 셀 점수의 p 백분위). 실현 오경보율 기록
  G1 학습 잡음 k ∈ {.5, 1, 2}, G2 전체 잡음 k ∈ {.5, 1} (σ_row = 학습 반복 측정 잡음, 시드 0-2)
  H  앵커 부분집합 n_a ∈ {4, 6} (5개씩) + 8 (D2)
  I  공통 블록 평가: 6 대 12 (12스텝 주입, T=6 모델은 절반씩), 24 대 12 (24스텝 주입, T=12 모델은 절반씩)
  J1 학습 윈도우에 라벨 없는 SC1 이상 p_event ∈ {.1, .3} (시드 0-2), J2 (D2) 학습·검증에 NLOS 마스크 안 씀
사용법: python experiments/exp6_part2.py --families F G H I J [--splits val] [--datasets d1 d2]
출력: results/exp6/part2/metrics_<계열>_<데이터>_<분할>.csv
"""
import argparse
import copy
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from exp6_conditions import derive  # noqa: E402
from node_model_eval import SIG, build, full_spec  # noqa: E402
from src.baselines.models import OLSCross  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.inject import inject_windows  # noqa: E402
from src.data.load import load_windows  # noqa: E402
from src.metrics.anomaly import _flatten, cell_f1, evaluate, range_f1, smearing, to_time  # noqa: E402
from src.model.bank import anchor_scores  # noqa: E402
from src.model.compose import split_row  # noqa: E402

OUT = ROOT / "results" / "exp6" / "part2"
BASES = ("cell", "PS")
PCTS = (95.0, 97.5, 99.0, 99.5)
G_LEVELS = [("G1", k, s) for k in (0.5, 1.0, 2.0) for s in (0, 1, 2)] + \
           [("G2", k, s) for k in (0.5, 1.0) for s in (0, 1, 2)]
J_LEVELS = [("J1", p, s) for p in (0.1, 0.3) for s in (0, 1, 2)]
OLS_ALPHA = 100.0
DM = ["RANGE", "RSS", "RSS_FP"]


def spec(bid, T, ecfg):
    return full_spec({"time_basis": "cell"} if bid == "cell" else {"K_t": T}, ecfg)


def anchors_of(rows):
    return [f"A{a}" for a in sorted({int(split_row(r)[1][1:]) for r in rows})]


def load_unit(ds, unit, sp, E, dcfg):
    full = load_windows("d1", sp["normal"], dcfg) if ds == "d1" else \
        load_windows("d2", f"{unit}_{E['channel_name']}", dcfg)
    cells = {} if ds == "d1" else {k: full["cells"][k] for k in ("mask", "lab_D")}
    return full["X"], full["meta"].reset_index(drop=True), full["rows"], cells


def view(ds, unit, X, meta, cells):
    if ds == "d1":
        return (np.array([f"{unit}/{n.split('/')[-1]}" for n in meta["node"]]),
                np.isfinite(X).astype(float), meta["split"].to_numpy())
    return (np.array([f"{unit}/{int(t)}" for t in meta["tidx0"]]), cells["mask"], meta["split"].to_numpy())


def pool_of(unit):
    return SIG[SIG.env == int(unit[3:])][["d_range", "d_rss", "d_rssfp"]].to_numpy()


def inj_sets(ds, unit, E, X, rows, labD, ix, spl):
    """평가 주입 세트 [(시나리오, 시드, Xi, 라벨)]. 1부·실험 4 와 같은 시나리오·시드."""
    out = []
    if ds == "d1":
        for scn, sc in E["scenarios"].items():
            for s in E["injection_seeds"][spl]:
                Xi, lab, _, _ = inject_windows(X[ix], rows, sc, s)
                out.append((scn, s, Xi, lab))
        return out
    rix = [rows.index(f"RANGE_{a}") for a in anchors_of(rows)]
    nl = labD[ix][:, rix]
    scen = [(k, sc, None) for k, sc in E["scenarios"].items()] + \
           [(f"{k}m", sc, pool_of(unit)) for k, sc in E["scenarios"].items()]
    for scn, sc, dp in scen:
        for s in E["injection_seeds"][spl]:
            Xi, lab, _, _ = inject_windows(X[ix], rows, sc, s, dist_metric="RANGE", eligible=(nl == 0).all(2),
                                           delta_pool=dp, delta_metrics=DM)
            out.append((scn, s, Xi, np.where(nl == 1, np.nan, lab)))
    return out


def all_inj(ds, unit, E, X, rows, labD, split, splits):
    return {spl: (np.flatnonzero(split == spl), inj_sets(ds, unit, E, X, rows, labD, np.flatnonzero(split == spl), spl))
            for spl in splits}


def scores(comp, bank, Xi, W, nodes, rows, anchors):
    Xs, Ws = comp.transform(Xi, W, rows)
    return anchor_scores(bank.score(Xs, nodes, Ws), comp.anchor_of_row, anchors)


def thr_metrics(seqs, thr, level):
    s1 = [to_time(s, y) for s, y in seqs] if level == "time" else \
        [(s[a], y[a]) for s, y in seqs for a in range(s.shape[0])]
    f1, p, r = range_f1(s1, thr)
    s, y = _flatten(seqs)
    return dict(cell_f1=cell_f1(seqs, thr), smearing=smearing(seqs, thr), range_f1=f1, range_p=p, range_r=r,
                far=float((s[y == 0] > thr).mean()) if (y == 0).any() else np.nan)


def eval_bank(comp, bank, inj, W, nodes, rows, elevel, **tag):
    """elevel: 평가 단위 (D1 time, D2 cell). tag: 결과 행에 붙일 조건 (family, level, ...)."""
    anchors = anchors_of(rows)
    out = []
    for spl, (ix, sets) in inj.items():
        for scn, s, Xi, lab in sets:
            A = scores(comp, bank, Xi, W[ix], nodes[ix], rows, anchors)
            seqs = [(A[n], lab[n]) for n in range(len(A))]
            out.append(dict(split=spl, scenario=scn, seed=s, threshold=bank.thr, **tag,
                            **evaluate(seqs, bank.thr, elevel)))
    return out


def sigma_rows(X, W, nodes):
    """행별 반복 측정 잡음 SD: (노드, 행, 시점) 칸의 윈도우 간 분산 (관측 2개 이상) 평균의 제곱근."""
    x = np.where(W > 0, X, np.nan)
    V = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for nd in np.unique(nodes):
            xi = x[nodes == nd]
            if len(xi) < 2:
                continue
            V.append(np.where(np.isfinite(xi).sum(0) >= 2, np.nanvar(xi, axis=0, ddof=1), np.nan))
        sd = np.sqrt(np.nanmean(np.stack(V), axis=(0, 2)))
    return np.where(np.isfinite(sd), sd, 0.0)


# ----------------------------------------------------------------------------- 계열
def fam_F(ds, unit, data, E, ecfg, splits, code):
    X, meta, rows, cells = data
    nodes, W, split = view(ds, unit, X, meta, cells)
    tr, va = split == "train", split == "val"
    inj = all_inj(ds, unit, E, X, rows, cells.get("lab_D"), split, splits)
    anchors, out = anchors_of(rows), []
    for bid in BASES:
        comp, bank, _ = build(code, spec(bid, 12, ecfg), ecfg, ds, X[tr], W[tr], rows, nodes[tr], X[va], W[va], nodes[va])
        Xv, Wv = comp.transform(X[va], W[va], rows)
        av = bank.score(Xv, nodes[va], Wv)[Wv > 0]
        thrs = {p: float(np.nanpercentile(av, p)) for p in PCTS}
        assert abs(thrs[97.5] - bank.thr) < 1e-9
        for spl, (ix, sets) in inj.items():
            for scn, s, Xi, lab in sets:
                A = scores(comp, bank, Xi, W[ix], nodes[ix], rows, anchors)
                seqs = [(A[n], lab[n]) for n in range(len(A))]
                ev = evaluate(seqs, bank.thr, E["level"])
                for p, t in thrs.items():
                    out.append(dict(split=spl, scenario=scn, seed=s, family="F", level=p, cond_seed=-1, basis=bid,
                                    threshold=t, auc_pr=ev["auc_pr"], hitrate=ev["hitrate"],
                                    **thr_metrics(seqs, t, E["level"])))
    return out


def fam_G(ds, unit, data, E, ecfg, splits, code):
    X, meta, rows, cells = data
    nodes, W, split = view(ds, unit, X, meta, cells)
    tr, va = split == "train", split == "val"
    sig = sigma_rows(X[tr], W[tr], nodes[tr])
    labD = cells.get("lab_D")
    inj0 = all_inj(ds, unit, E, X, rows, labD, split, splits)
    out = []
    for fam, k, cs in [("base", 0.0, -1)] + G_LEVELS:
        if fam == "base":
            Xa, inj = X, inj0
            Xtr = X[tr]
        else:
            noise = np.random.default_rng((4000 if fam == "G1" else 5000) + cs).normal(size=X.shape) * \
                (k * sig)[None, :, None]
            if fam == "G1":
                Xa, inj, Xtr = X, inj0, X[tr] + noise[tr]
            else:
                Xa = X + noise
                inj, Xtr = all_inj(ds, unit, E, Xa, rows, labD, split, splits), Xa[tr]
        for bid in BASES:
            comp, bank, _ = build(code, spec(bid, 12, ecfg), ecfg, ds, Xtr, W[tr], rows, nodes[tr],
                                  Xa[va], W[va], nodes[va])
            out += eval_bank(comp, bank, inj, W, nodes, rows, E["level"], family=fam, level=k, cond_seed=cs, basis=bid)
        print(f"  G {ds} {unit} {fam} k={k} seed={cs}", flush=True)
    return out


def subsets():
    rng = np.random.default_rng(3000)
    out = [tuple(range(1, 9))]
    for na in (4, 6):
        seen = []
        while len(seen) < 5:
            c = tuple(sorted(int(a) for a in rng.choice(np.arange(1, 9), na, replace=False)))
            if c not in seen:
                seen.append(c)
        out += seen
    return out


def fam_H(ds, unit, data, E, ecfg, splits, code):
    if ds != "d2":
        return []
    X0, meta, rows0, cells0 = data
    out = []
    for si, sub in enumerate(subsets()):
        keep = [i for i, r in enumerate(rows0) if int(split_row(r)[1][1:]) in sub]
        rows = [rows0[i] for i in keep]
        X, cells = X0[:, keep], {k: v[:, keep] for k, v in cells0.items()}
        nodes, W, split = view(ds, unit, X, meta, cells)
        tr, va = split == "train", split == "val"
        inj = all_inj(ds, unit, E, X, rows, cells["lab_D"], split, splits)
        tag = dict(family="H", level=len(sub), cond_seed=si, subset="-".join(map(str, sub)))
        for bid in BASES:
            comp, bank, _ = build(code, spec(bid, 12, ecfg), ecfg, ds, X[tr], W[tr], rows, nodes[tr],
                                  X[va], W[va], nodes[va])
            out += eval_bank(comp, bank, inj, W, nodes, rows, E["level"], basis=bid, **tag)
            if bid == "cell":                                   # OLS (nc): M01 잔차 입력 (실험 1 model_input 과 같음)
                cbank, ccomp = bank, comp
        Z = lambda Xr, Wr, nd: np.nan_to_num(cbank.residual(ccomp.transform(Xr, Wr, rows)[0], nd))
        _, Ws_tr = ccomp.transform(X[tr], W[tr], rows)
        m = OLSCross(alpha=OLS_ALPHA).fit(np.where(Ws_tr > 0, Z(X[tr], W[tr], nodes[tr]), 0.0))
        _, Ws_va = ccomp.transform(X[va], W[va], rows)
        ev = np.where(Ws_va > 0, m.errors(Z(X[va], W[va], nodes[va])), np.nan)
        sd = np.nanstd(np.moveaxis(ev, 1, 0).reshape(ev.shape[1], -1), 1)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        thr = float(np.nanpercentile(ev / sd[None, :, None], 97.5))
        anchors = anchors_of(rows)
        for spl, (ix, sets) in inj.items():
            for scn, s, Xi, lab in sets:
                A = anchor_scores(m.errors(Z(Xi, W[ix], nodes[ix])) / sd[None, :, None], ccomp.anchor_of_row, anchors)
                out.append(dict(split=spl, scenario=scn, seed=s, threshold=thr, basis="OLS", **tag,
                                **evaluate([(A[n], lab[n]) for n in range(len(A))], thr, E["level"])))
        print(f"  H {ds} {unit} subset={sub}", flush=True)
    return out


def fam_I(ds, unit, data, E, ecfg, splits, code):
    X, meta, rows, cells = data
    D = {T: derive(X, meta, cells, T) for T in (6, 12, 24)}
    anchors, models = anchors_of(rows), {}
    for T, (XT, mT, cT) in D.items():
        nodes, W, split = view(ds, unit, XT, mT.reset_index(drop=True), cT)
        tr, va = split == "train", split == "val"
        for bid in BASES:
            comp, bank, _ = build(code, spec(bid, T, ecfg), ecfg, ds, XT[tr], W[tr], rows, nodes[tr],
                                  XT[va], W[va], nodes[va])
            models[(T, bid)] = (comp, bank)
    out = []
    for pair, (Tb, Th) in {"6v12": (12, 6), "24v12": (24, 12)}.items():
        XT, mT, cT = D[Tb]
        mT = mT.reset_index(drop=True)
        nodes, W, split = view(ds, unit, XT, mT, cT)
        h = Tb // 2
        for spl in splits:
            ix = np.flatnonzero(split == spl)
            if ds == "d1":
                nh = [nodes[ix], nodes[ix]]
            else:
                t0 = mT["tidx0"].to_numpy()[ix].astype(int)
                nh = [np.array([f"{unit}/{t}" for t in t0]), np.array([f"{unit}/{t + h}" for t in t0])]
            for scn, s, Xi, lab in inj_sets(ds, unit, E, XT, rows, cT.get("lab_D"), ix, spl):
                for bid in BASES:
                    cb, bb = models[(Tb, bid)]
                    ch, bh = models[(Th, bid)]
                    A_full = scores(cb, bb, Xi, W[ix], nodes[ix], rows, anchors)
                    A_half = np.concatenate([scores(ch, bh, Xi[:, :, k * h:(k + 1) * h], W[ix][:, :, k * h:(k + 1) * h],
                                                    nh[k], rows, anchors) for k in (0, 1)], axis=2)
                    for Tm, A, thr in ((Tb, A_full, bb.thr), (Th, A_half, bh.thr)):
                        out.append(dict(split=spl, scenario=scn, seed=s, family="I", level=pair, cond_seed=-1,
                                        basis=bid, T_model=Tm, threshold=thr,
                                        **evaluate([(A[n], lab[n]) for n in range(len(A))], thr, E["level"])))
        print(f"  I {ds} {unit} {pair}", flush=True)
    return out


def fam_J(ds, unit, data, E, ecfg, splits, code):
    X, meta, rows, cells = data
    nodes, W, split = view(ds, unit, X, meta, cells)
    tr, va = split == "train", split == "val"
    labD = cells.get("lab_D")
    inj = all_inj(ds, unit, E, X, rows, labD, split, splits)
    conds = [("base", 0.0, -1)] + J_LEVELS + ([("J2", 1.0, -1)] if ds == "d2" else [])
    out = []
    for fam, p, cs in conds:
        Xtr, Wtr, Wva = X[tr], W[tr], W[va]
        if fam == "J1":
            sc = copy.deepcopy(E["scenarios"]["SC1"])
            sc["p_event"] = p
            if ds == "d1":
                Xtr, _, _, _ = inject_windows(X[tr], rows, sc, 6000 + cs)
            else:
                rix = [rows.index(f"RANGE_{a}") for a in anchors_of(rows)]
                Xtr, _, _, _ = inject_windows(X[tr], rows, sc, 6000 + cs, dist_metric="RANGE",
                                              eligible=(labD[tr][:, rix] == 0).all(2), delta_pool=pool_of(unit),
                                              delta_metrics=DM)
        elif fam == "J2":
            Wall = np.isfinite(X).astype(float)
            Wtr, Wva = Wall[tr], Wall[va]
        for bid in BASES:
            comp, bank, _ = build(code, spec(bid, 12, ecfg), ecfg, ds, Xtr, Wtr, rows, nodes[tr], X[va], Wva, nodes[va])
            out += eval_bank(comp, bank, inj, W, nodes, rows, E["level"], family=fam, level=p, cond_seed=cs, basis=bid)
        print(f"  J {ds} {unit} {fam} p={p} seed={cs}", flush=True)
    return out


FAMS = {"F": fam_F, "G": fam_G, "H": fam_H, "I": fam_I, "J": fam_J}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=list(FAMS))
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--datasets", nargs="*", default=["d1", "d2"])
    args = ap.parse_args()
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    code = ecfg["ids"]["F01"]
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        E = ecfg["datasets"][ds]
        units = list(E["spaces"].items()) if ds == "d1" else [(f"env{e}", None) for e in E["envs"]]
        data = {u: load_unit(ds, u, sp, E, dcfg) for u, sp in units}
        for fam in args.families:
            if fam == "H" and ds != "d2":
                continue
            t0 = time.perf_counter()
            rows = []
            for u, _ in units:
                rows += [dict(dataset=ds, space=u, **r) for r in FAMS[fam](ds, u, data[u], E, ecfg, args.splits, code)]
            pd.DataFrame(rows).to_csv(OUT / f"metrics_{fam}_{ds}_{'_'.join(args.splits)}.csv", index=False,
                                      encoding="utf-8-sig")
            print(f"{fam} {ds}: {len(rows)} rows, {time.perf_counter() - t0:.1f}s", flush=True)
