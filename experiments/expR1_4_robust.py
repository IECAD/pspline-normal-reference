"""실험 R1-4 2부: 견고 참조 기준선 (doc/실험R1-4_계획.md 2부, 사전 등록 2026-10-28 21:32).

D6(주): 실험 10 과 같은 노드·연결 앵커·주입(주 크기, ×3)·보정(검증 SD) 경로에서 참조
CG(cell) · CI(cellinterp) · CMI(칸별 중앙값 + 보간, 새로) · NM(Z1) · NMed(노드 중앙값, 새로) · PS · PS1 · PSH · PS1H 를
전체 평가(주)와 이상값 제외 평가(보조, 실험 10 규칙: 주입 전 원값이 학습 링크 중앙값에서 1 m 넘게 벗어난 칸)로 함께 잰다.
제외한 양성·음성 칸 수를 직접 세어 저장한다.
D1·D2(보고): 깨끗한 조건에서 PS, CMI, NMed (exp1_baselines.prep 경로).

사용법: python experiments/expR1_4_robust.py --splits val | test [--part d6 d12]
출력: results/expR1/robust/metrics_d6_<split>.csv, metrics_d12_<split>.csv
"""
import argparse
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import exp10_d6 as X10  # noqa: E402
from exp1_baselines import prep  # noqa: E402
from node_model_eval import build, full_spec  # noqa: E402
from src.data import d2 as D2  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.inject import inject_windows  # noqa: E402
from src.data.load import load_windows  # noqa: E402
from src.metrics.anomaly import evaluate  # noqa: E402
from src.model.bank import NormalSurfaceBank, anchor_scores  # noqa: E402

OUT = ROOT / "results" / "expR1" / "robust"
SPEC = {"CG": {"time_basis": "cell"}, "CI": {"time_basis": "cellinterp"}, "NM": {"time_basis": "const"},
        "PS": {"K_t": 12}, "PS1": {"K_t": 12, "diff_order": 1}, "PSH": {"K_t": 12, "robust": "huber"},
        "PS1H": {"K_t": 12, "diff_order": 1, "robust": "huber"},
        "CMI": {"time_basis": "cell"}, "NMed": {"time_basis": "cell"}}       # CMI·NMed 는 셀 뱅크를 만든 뒤 M 을 바꿈
REFS_D6 = ["CG", "CI", "CMI", "NM", "NMed", "PS", "PS1", "PSH", "PS1H"]
REFS_D12 = ["PS", "CMI", "NMed"]


def median_surface(bank, ref, Xs, Ws, nodes):
    """CMI: 칸별 중앙값, 빈 칸은 시간 보간(행 전체가 비면 0 = 단위 중심). NMed: 노드·행 중앙값."""
    T = Xs.shape[2]
    t = np.arange(T)
    for nd in bank.M:
        ix = nodes == nd
        x = np.where(Ws[ix] > 0, Xs[ix], np.nan)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if ref == "CMI":
                    med = np.nanmedian(x, axis=0)
                    for s in range(med.shape[0]):
                        ok = np.isfinite(med[s])
                        if not ok.all():
                            med[s] = np.interp(t, t[ok], med[s, ok]) if ok.any() else 0.0
                    bank.M[nd] = med
                else:
                    v = np.nanmedian(x.transpose(1, 0, 2).reshape(x.shape[1], -1), axis=1)
                    bank.M[nd] = np.repeat(np.where(np.isfinite(v), v, 0.0)[:, None], T, axis=1)


def build_ref(ref, ecfg, ds, Xtr, Wtr, rows, ntr, Xva, Wva, nva):
    spec = full_spec({**SPEC[ref], "stats": "val"}, ecfg)
    comp, bank, _ = build(X10.CODE, spec, ecfg, ds, Xtr, Wtr, rows, ntr, Xva, Wva, nva)
    if ref in ("CMI", "NMed"):
        Xs, Ws = comp.transform(Xtr, Wtr, rows)
        median_surface(bank, ref, Xs, Ws, ntr)
        Xv, Wv = comp.transform(Xva, Wva, rows)
        bank.calibrate(Xv, Wv, nva, ecfg["model"]["threshold_pct"], stats="val")
    return comp, bank


# ---------------------------------------------------------------- D6
def run_scan(scan, ecfg, dcfg, splits):
    w = load_windows("d6", f"scan{scan}", dcfg)
    X, rows, meta = w["X"], w["rows"], w["meta"]
    M = w["cells"]["mask"]
    nodes = meta.node.to_numpy()
    split = meta.split.to_numpy()
    tr, va = split == "train", split == "val"
    anchors = [r.split("_", 1)[1] for r in rows]
    conn = X10.connected(M, meta, rows)
    keep_all = np.stack([conn[n] for n in nodes])
    clean = np.ones(X.shape, bool)                                   # 실험 10 의 제외 규칙(주입 전 원값)
    for nd in np.unique(nodes):
        ixn = np.flatnonzero(nodes == nd)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(np.where(M[ixn][tr[ixn]] > 0, X[ixn][tr[ixn]], np.nan), axis=(0, 2))
        clean[ixn] = ~((M[ixn] > 0) & (np.abs(X[ixn] - med[None, :, None]) > X10.OUTLIER_M))
    E = ecfg["datasets"]["d1"]
    scen = list(E["scenarios"].items()) + [(f"{k}x{X10.MAG_SCALE:g}", X10.scaled(sc, X10.MAG_SCALE))
                                          for k, sc in E["scenarios"].items()]
    inj = {}
    for spl in splits:
        ix = np.flatnonzero(split == spl)
        Wa = M[ix]
        elig = keep_all[ix] & (Wa.mean(2) >= X10.MIN_WIN_FRAC)
        for scn, sc in scen:
            for s in E["injection_seeds"][spl]:
                Xi, lab, _, _ = inject_windows(X[ix], rows, sc, s, dist_metric="DIST", eligible=elig)
                lab = np.where(keep_all[ix][:, :, None] & (Wa > 0), lab, np.nan)
                inj[(spl, scn, s)] = (ix, Xi, lab)
    out = []
    for ref in REFS_D6:
        t0 = time.perf_counter()
        comp, bank = build_ref(ref, ecfg, "d6", X[tr], M[tr], rows, nodes[tr], X[va], M[va], nodes[va])
        for (spl, scn, s), (ix, Xi, lab) in inj.items():
            Xs, Ws = comp.transform(Xi, M[ix], rows)
            A_rows = bank.score(Xs, nodes[ix], Ws)
            A = X10.mask_scores(anchor_scores(A_rows, comp.anchor_of_row, anchors), keep_all[ix], M[ix])
            lab_k = np.where(keep_all[ix][:, :, None], lab, np.nan)
            for ev_name, cmask in (("full", np.ones_like(clean[ix])), ("excl", clean[ix])):
                lk = np.where(cmask, lab_k, np.nan)
                Ak = np.where(cmask, A, np.nan)
                ev = X10.evaluate_light([(Ak[n], lk[n]) for n in range(len(Ak))], bank.thr)
                pos0 = int(np.nansum(lab_k == 1)); neg0 = int(np.nansum(lab_k == 0))
                pos1 = int(np.nansum(lk == 1)); neg1 = int(np.nansum(lk == 0))
                out.append(dict(scan=scan, split=spl, scenario=scn, seed=s, ref=ref, eval=ev_name,
                                pos_before=pos0, neg_before=neg0, pos_after=pos1, neg_after=neg1,
                                threshold=bank.thr, **ev))
        print(f"  scan{scan} {ref}: {time.perf_counter() - t0:.1f}s", flush=True)
    return out


# ---------------------------------------------------------------- D1·D2 깨끗한 조건 (보고)
def run_d12(ecfg, dcfg, splits):
    mall = D2.load_meta(dcfg); mall = mall[mall.ch == dcfg["d2"]["channel"]]
    out = []
    for ds in ("d1", "d2"):
        E = ecfg["datasets"][ds]
        units = list(E["spaces"].items()) if ds == "d1" else [(f"env{e}", None) for e in E["envs"]]
        for split in splits:
            for unit, sp in units:
                D = prep(ds, unit, sp, ecfg, dcfg, [split], mall)
                T = D["tr"]["Xs"].shape[2]
                for ref in REFS_D12:
                    kw = dict(time_basis="pspline") if ref == "PS" else dict(time_basis="cell")
                    bank = NormalSurfaceBank(T, T if ref == "PS" else 8, {}, **kw).fit(D["tr"]["Xs"], D["tr"]["Ws"], D["tr"]["n"])
                    if ref != "PS":
                        median_surface(bank, ref, D["tr"]["Xs"], D["tr"]["Ws"], D["tr"]["n"])
                    bank.calibrate(D["va"]["Xs"], D["va"]["Ws"], D["va"]["n"], ecfg["model"]["threshold_pct"], stats="val")
                    Ev = D["evals"][split]
                    for scn, s, Xi, lab in Ev["sets"]:
                        Xs, Ws = D["comp"].transform(Xi, Ev["W"], D["rows"])
                        A = bank.score(Xs, Ev["n"], Ws)
                        Aa = anchor_scores(A, D["comp"].anchor_of_row, D["anchors"])
                        ev = evaluate([(Aa[n], lab[n]) for n in range(len(Aa))], bank.thr, D["level"])
                        out.append(dict(dataset=ds, space=unit, split=split, scenario=scn, seed=s, ref=ref, **ev))
                print(f"  {ds} {unit} {split} done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--part", nargs="*", default=["d6", "d12"])
    ap.add_argument("--scans", nargs="*", type=int, default=list(X10.SCANS))
    args = ap.parse_args()
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    suf = "_".join(args.splits)
    t0 = time.perf_counter()
    if "d6" in args.part:
        rows = []
        for scan in args.scans:
            rows += run_scan(scan, ecfg, dcfg, args.splits)
            pd.DataFrame(rows).to_csv(OUT / f"metrics_d6_{suf}.csv", index=False, encoding="utf-8-sig")
    if "d12" in args.part:
        pd.DataFrame(run_d12(ecfg, dcfg, args.splits)).to_csv(OUT / f"metrics_d12_{suf}.csv", index=False, encoding="utf-8-sig")
    print(f"done {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
