"""실험 10: 독립 데이터 D6 (TWR-CLOUD-Bern) 확인 실험 (doc/실험10_계획.md, 사전 등록 2026-10-12).

노드 = (scan, 위치), 앵커 = 연결 링크 (학습 관측 ≥ 12). 주입 = D1 시나리오 (거리 행, 연결 앵커 중 윈도우 관측 ≥ 50 %).
항목: T1 윈도우 LRT(R, PS 배경) vs PS · T2 PS vs cell/Z1/cellfill · T3 학습 f = 0.1 · T4 앵커 수 4/8/16/전부 · T5 OLS·PCA(nc) · D 회전 노드
사용법: python experiments/exp10_d6.py run --splits val | test ;  python experiments/exp10_d6.py summarize --split val | test
출력: results/exp10/metrics_<분할>.csv, summary_<분할>.md, decision.json (검증)
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
from exp1_stssd_original import LRT, resid_sd, win_auc  # noqa: E402
from node_model_eval import build, full_spec  # noqa: E402
from src.baselines.models import OLSCross, PCASPE  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.inject import inject_windows  # noqa: E402
from src.data.load import load_windows  # noqa: E402
from src.metrics.anomaly import _flatten, cell_f1, hitrate, smearing  # noqa: E402


def evaluate_light(seqs, thr):
    """셀 단위 탐지·위치화 지표만 (Range-F1·VUS 는 이 실험에서 쓰지 않고, 셀 시퀀스 1만 개에서 파이썬 루프가 매우 느림)."""
    from sklearn.metrics import average_precision_score
    s, y = _flatten(seqs)
    auc = float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan
    return dict(auc_pr=auc, hitrate=hitrate(seqs), cell_f1=cell_f1(seqs, thr), smearing=smearing(seqs, thr),
                pos_rate=float(y.mean()) if len(y) else np.nan, n_eval=int(len(y)))
from src.model.bank import anchor_scores  # noqa: E402

OUT = ROOT / "results" / "exp10"
CFG = dict(calib="val", excl=False, anchors=True, tag="", models=None, comparators=True, f01_models=["PS", "cell"])      # 실험 10-2 (robust): calib="val_mad", excl=True, anchors=False
OUTLIER_M = 1.0


def row_sd(ev):
    """행별 산포 (N, S, T) -> (S,): SD, 또는 견고(1.4826·MAD, 실험 10-2 V1)."""
    flat = np.moveaxis(ev, 1, 0).reshape(ev.shape[1], -1)
    if CFG["calib"] == "val_mad":
        med = np.nanmedian(flat, 1)
        sd = 1.4826 * np.nanmedian(np.abs(flat - med[:, None]), 1)
    else:
        sd = np.nanstd(flat, 1)
    return np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
CODE = {"V": "V1", "S": "S1", "O": "O1", "T": "T4"}
BASES = {"PS": {"K_t": 12}, "cell": {"time_basis": "cell"}, "Z1": {"time_basis": "const"}, "cellfill": {"time_basis": "cellfill"},
         "PS1": {"K_t": 12, "diff_order": 1}, "PSH": {"K_t": 12, "robust": "huber"}, "PS1H": {"K_t": 12, "diff_order": 1, "robust": "huber"}}   # 10-3
DEFAULT_MODELS = ["PS", "cell", "Z1", "cellfill"]
MIN_TRAIN_OBS, MIN_WIN_FRAC = 12, 0.5
SUBSETS = (4, 8, 16)
TOL = 0.0171
SCANS = range(1, 13)
ROT_POS = {165, 166}
MAG_SCALE = 3.0                    # 보조 시나리오: D1 주입 크기 × 3 (D6 링크 잡음 SD 중앙값 1 m, 계획 "실행 전 수정 2")


def scaled(sc, f):
    import copy
    sc = copy.deepcopy(sc)
    for t in sc["types"].values():
        t["magnitude_m"] = [v * f for v in t["magnitude_m"]]
    return sc


def connected(M, meta, rows):
    """노드별 연결 앵커 (학습 관측 ≥ MIN_TRAIN_OBS, 자기 행 제외). 반환 {node: bool (S,)}."""
    tr = meta.split.to_numpy() == "train"
    out = {}
    for nd in meta.node.unique():
        ix = np.flatnonzero((meta.node.to_numpy() == nd) & tr)
        c = M[ix].sum((0, 2)) >= MIN_TRAIN_OBS
        c[[i for i, r in enumerate(rows) if r == f"DIST_A{nd.split('/P')[1]}"]] = False
        out[nd] = c
    return out


def mask_scores(A, keep, Wa):
    """앵커 점수 (N, A, T) 에서 비연결·미관측 셀을 NaN 으로."""
    return np.where(keep[:, :, None] & (Wa > 0), A, np.nan)


def comparator(name, cell_bank, comp, X, W, rows, nodes, split):
    """OLS / PCA (nc 입력 = 셀 평균 잔차, 실험 1 D2 선택 구성). 반환 score(Xi, Wi, ni) -> 행 점수."""
    tr, va = split == "train", split == "val"
    Z = lambda Xr, Wr, nd: np.nan_to_num(cell_bank.residual(comp.transform(Xr, Wr, rows)[0], nd))
    Ws_tr = comp.transform(X[tr], W[tr], rows)[1]
    m = (OLSCross(alpha=100.0) if name == "OLS" else PCASPE(var=0.8)).fit(np.where(Ws_tr > 0, Z(X[tr], W[tr], nodes[tr]), 0.0))
    Ws_va = comp.transform(X[va], W[va], rows)[1]
    ev = np.where(Ws_va > 0, m.errors(Z(X[va], W[va], nodes[va])), np.nan)
    sd = row_sd(ev)
    thr = float(np.nanpercentile(ev / sd[None, :, None], 97.5))
    return (lambda Xi, Wi, ni: m.errors(Z(Xi, Wi, ni)) / sd[None, :, None]), thr


def run_scan(scan, ecfg, dcfg, splits):
    w = load_windows("d6", f"scan{scan}", dcfg)
    X, rows, meta = w["X"], w["rows"], w["meta"]
    M = w["cells"]["mask"]
    nodes = meta.node.to_numpy()
    split = meta.split.to_numpy()
    tr, va = split == "train", split == "val"
    anchors = [r.split("_", 1)[1] for r in rows]                     # 'A79' ... (행 순서 = 앵커 순서)
    conn = connected(M, meta, rows)
    keep_all = np.stack([conn[n] for n in nodes])                     # (N, A)
    clean = np.ones(X.shape, bool)                                      # V2: 이상값 셀 (학습 링크 중앙값 대비 > 1 m) 제외
    if CFG["excl"]:
        for nd in np.unique(nodes):
            ixn = np.flatnonzero(nodes == nd)
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.where(M[ixn][tr[ixn]] > 0, X[ixn][tr[ixn]], np.nan), axis=(0, 2))   # (S,)
            clean[ixn] = ~((M[ixn] > 0) & (np.abs(X[ixn] - med[None, :, None]) > OUTLIER_M))
    E = ecfg["datasets"]["d1"]
    # 평가 주입 세트 (분할, 시나리오, 시드) -> (ix, Xi, lab)
    inj = {}
    scen = list(E["scenarios"].items()) + [(f"{k}x{MAG_SCALE:g}", scaled(sc, MAG_SCALE)) for k, sc in E["scenarios"].items()]
    for spl in splits:
        ix = np.flatnonzero(split == spl)
        Wa = M[ix]
        elig = keep_all[ix] & (Wa.mean(2) >= MIN_WIN_FRAC)
        for scn, sc in scen:
            for s in E["injection_seeds"][spl]:
                Xi, lab, _, _ = inject_windows(X[ix], rows, sc, s, dist_metric="DIST", eligible=elig)
                lab = np.where(keep_all[ix][:, :, None] & (Wa > 0), lab, np.nan)
                inj[(spl, scn, s)] = (ix, Xi, lab)
    out = []

    def record(scorer, tag, keep=None, Xtr=None, Wtr=None, ntr=None, win_from=None):
        """scorer(Xi, Wi, ni) -> (행 점수 (N,S,T), 임계값, 윈도우 점수 or None)."""
        kp = keep_all if keep is None else keep
        for (spl, scn, s), (ix, Xi, lab) in inj.items():
            A_rows, thr, win = scorer(Xi, M[ix], nodes[ix])
            A = mask_scores(anchor_scores(A_rows, comp_of[tag].anchor_of_row, anchors), kp[ix], M[ix])
            lab_k = np.where(kp[ix][:, :, None] & clean[ix], lab, np.nan)
            A = np.where(clean[ix], A, np.nan)
            seqs = [(A[n], lab_k[n]) for n in range(len(A))]
            ev = evaluate_light(seqs, thr)
            out.append(dict(scan=scan, split=spl, scenario=scn, seed=s, **tag_meta[tag],
                            win_auc_pr=win_auc(win if win is not None else A, lab_k), n_anchors=float(kp[ix].sum(1).mean()),
                            rot=float(np.mean([int(n.split("/P")[1]) in ROT_POS for n in nodes[ix]])), **ev))

    comp_of, tag_meta = {}, {}
    banks = {}
    # ---- T2 / T5 / T1: 학습 전부
    for bid in (CFG["models"] or DEFAULT_MODELS):
        spec = BASES[bid]
        comp, bank, _ = build(CODE, full_spec({**spec, "stats": CFG["calib"]}, ecfg), ecfg, "d6", X[tr], M[tr], rows, nodes[tr], X[va], M[va], nodes[va])
        banks[bid] = (comp, bank)
        tag = f"full/{bid}"
        comp_of[tag], tag_meta[tag] = comp, dict(cond="full", level="1", cond_seed=-1, model=bid)
        record(lambda Xi, Wi, ni, c=comp, b=bank: (b.score(*c.transform(Xi, Wi, rows)[:1], ni, c.transform(Xi, Wi, rows)[1]), b.thr, None), tag)
    comp, bank = banks["PS"]
    Xt, Wt = comp.transform(X[tr], M[tr], rows)
    if not CFG["comparators"]:
        Xv, Wv = comp.transform(X[va], M[va], rows)
    Xv, Wv = comp.transform(X[va], M[va], rows)
    lrt = LRT(bank, row_sd(np.where(Wv > 0, bank.residual(Xv, nodes[va]), np.nan))).calibrate((Xt, Wt, nodes[tr]), (Xv, Wv, nodes[va])) if CFG["comparators"] else None
    tag = "full/R"
    if not CFG["comparators"]:
        tag = None
    if tag is not None:
        comp_of[tag], tag_meta[tag] = comp, dict(cond="full", level="1", cond_seed=-1, model="R")
        record(lambda Xi, Wi, ni: (lambda cw: (cw[0], lrt.thr, cw[1]))(lrt.score(*comp.transform(Xi, Wi, rows), ni)), tag)
    ccomp, cbank = banks["cell"]
    for name in (("OLS", "PCA") if CFG["comparators"] else ()):
        f, thr = comparator(name, cbank, ccomp, X, M, rows, nodes, split)
        tag = f"full/{name}"
        comp_of[tag], tag_meta[tag] = ccomp, dict(cond="full", level="1", cond_seed=-1, model=name)
        record(lambda Xi, Wi, ni, f=f, thr=thr: (f(Xi, Wi, ni), thr, None), tag)
    # ---- T4: 앵커 부분집합 (노드별 연결 앵커에서 무작위, 평가만 제한)
    for k in (SUBSETS if CFG["anchors"] else ()):
        for cs in range(3):
            rng = np.random.default_rng(7000 + 10 * k + cs)
            kp = np.zeros_like(keep_all)
            sub = {}
            for nd in np.unique(nodes):
                c = np.flatnonzero(conn[nd])
                sub[nd] = c if len(c) <= k else rng.choice(c, k, replace=False)
            for i, nd in enumerate(nodes):
                kp[i, sub[nd]] = True
            tag = f"anchors/{k}/{cs}"
            comp_of[tag], tag_meta[tag] = comp, dict(cond="anchors", level=str(k), cond_seed=cs, model="PS")
            record(lambda Xi, Wi, ni: (bank.score(*comp.transform(Xi, Wi, rows)[:1], ni, comp.transform(Xi, Wi, rows)[1]), bank.thr, None), tag, keep=kp)
    # ---- T3: 학습 f = 0.1
    for cs in range(3):
        rng = np.random.default_rng(1000 + cs)
        keep = []
        for nd in np.unique(nodes):
            ix = np.flatnonzero((nodes == nd) & tr)
            keep += list(rng.choice(ix, max(1, int(round(0.1 * len(ix)))), replace=False))
        keep = np.sort(keep)
        for bid in CFG["f01_models"]:
            comp2, bank2, _ = build(CODE, full_spec({**BASES[bid], "stats": CFG["calib"]}, ecfg), ecfg, "d6", X[keep], M[keep], rows, nodes[keep], X[va], M[va], nodes[va])
            tag = f"f0.1/{bid}/{cs}"
            comp_of[tag], tag_meta[tag] = comp2, dict(cond="f0.1", level="0.1", cond_seed=cs, model=bid)
            record(lambda Xi, Wi, ni, c=comp2, b=bank2: (b.score(*c.transform(Xi, Wi, rows)[:1], ni, c.transform(Xi, Wi, rows)[1]), b.thr, None), tag)
    print(f"  scan{scan} done ({len(out)} rows)", flush=True)
    return out


def summarize(spl):
    M_all = pd.read_csv(OUT / f"metrics{CFG['tag']}_{spl}.csv")
    M = M_all[~M_all.scenario.astype(str).str.contains("x")]          # 주 판정: 원 D1 크기. ×3 는 보조 표
    key = ["scan", "scenario", "seed"]
    L = [f"# 실험 10 집계 ({spl})", "", "자동 생성: `experiments/exp10_d6.py summarize`. 계획: `doc/실험10_계획.md`. 허용오차 0.0171.", ""]
    dec = {}

    def paired(a, b, met, cond_a="full", cond_b="full", cs=None):
        da = M[(M.cond == cond_a) & (M.model == a) & ((M.cond_seed == cs) if cs is not None else True)]
        db = M[(M.cond == cond_b) & (M.model == b) & ((M.cond_seed == cs) if cs is not None else True)]
        x = (da.set_index(key)[met] - db.set_index(key)[met]).dropna()
        return x.mean(), (x > 0).mean(), len(x)

    full = M[M.cond == "full"].groupby("model")[["auc_pr", "win_auc_pr", "hitrate", "cell_f1", "smearing", "pos_rate"]].mean()
    L += ["## 학습 전부 (T1·T2·T5)", "", "| 모델 | 셀 AUC-PR | 윈도우 AUC-PR | HitRate | 셀 F1 | Smearing |", "|---|---|---|---|---|---|"]
    for m in ["PS", "R", "cell", "Z1", "cellfill", "OLS", "PCA"]:
        if m in full.index:
            r = full.loc[m]
            L.append(f"| {m} | {r.auc_pr:.3f} | {r.win_auc_pr:.3f} | {r.hitrate:.3f} | {r.cell_f1:.3f} | {r.smearing:.3f} |")
    L.append("")
    d, fp, n = paired("R", "PS", "win_auc_pr")
    dc, _, _ = paired("PS", "R", "auc_pr")
    dec["T1"] = dict(win_delta_R_minus_PS=d, frac_pos=fp, n=n, beats=bool(d > TOL), cell_delta_PS_minus_R=dc)
    L += [f"**T1** R − PS 윈도우 AUC-PR: {d:+.4f} (양수 {fp:.2f}, n={n}) → {'R 우위' if d > TOL else ('PS 우위' if d < -TOL else '대등')}; 셀 AUC-PR PS − R {dc:+.4f}", ""]
    dec["T2"] = {}
    for b in ("cell", "Z1", "cellfill"):
        d, fp, n = paired("PS", b, "auc_pr")
        v = "PS 우위" if d > TOL else ("PS 열세" if d < -TOL else "대등")
        dec["T2"][b] = dict(delta=d, frac_pos=fp, verdict=v)
        L.append(f"**T2** PS − {b} 셀 AUC-PR: {d:+.4f} (양수 {fp:.2f}) → {v}")
    dec["T5"] = {}
    for b in ("OLS", "PCA"):
        d, fp, n = paired("PS", b, "auc_pr")
        v = "PS 우위" if d > TOL else ("PS 열세" if d < -TOL else "대등")
        dec["T5"][b] = dict(delta=d, frac_pos=fp, verdict=v)
        L.append(f"**T5** PS − {b} 셀 AUC-PR: {d:+.4f} (양수 {fp:.2f}) → {v}")
    L.append("")
    xs = [paired("PS", "cell", "auc_pr", "f0.1", "f0.1", cs)[0] for cs in range(3)]
    f01 = M[M.cond == "f0.1"].groupby("model").auc_pr.mean()
    d3 = float(np.mean(xs))
    v = "PS 우위" if d3 > TOL else ("PS 열세" if d3 < -TOL else "대등")
    dec["T3"] = dict(delta=d3, per_seed=xs, verdict=v, ps=float(f01.get("PS", np.nan)), cell=float(f01.get("cell", np.nan)))
    L += [f"**T3** 학습 f = 0.1: PS {f01.get('PS', np.nan):.3f}, cell {f01.get('cell', np.nan):.3f}, Δ {d3:+.4f} (시드별 {np.round(xs, 4).tolist()}) → {v}", ""]
    A = M[M.cond == "anchors"].groupby("level")[["auc_pr", "hitrate", "smearing", "n_anchors"]].mean() if (M.cond == "anchors").any() else pd.DataFrame(columns=["auc_pr", "hitrate", "smearing", "n_anchors"])
    A.loc["all"] = M[(M.cond == "full") & (M.model == "PS")][["auc_pr", "hitrate", "smearing", "n_anchors"]].mean()
    L += ["## T4 앵커 수 (PS, 평가 제한)", "", "```", A.round(3).to_string(), "```", ""]
    X3 = M_all[M_all.scenario.astype(str).str.contains("x") & (M_all.cond == "full")]
    if len(X3):
        t3 = X3.groupby("model")[["auc_pr", "win_auc_pr", "hitrate", "smearing"]].mean().reindex(["PS", "R", "cell", "Z1", "cellfill", "OLS", "PCA"]).dropna(how="all")
        k3 = ["scan", "scenario", "seed"]
        def pr(a, b, met):
            x = (X3[X3.model == a].set_index(k3)[met] - X3[X3.model == b].set_index(k3)[met]).dropna(); return x.mean(), (x > 0).mean()
        L += ["## 보조: 주입 크기 ×3 (학습 전부)", "", "```", t3.round(3).to_string(), "```", ""]
        L += [f"R − PS 윈도우 {pr('R','PS','win_auc_pr')[0]:+.4f}; PS − cell {pr('PS','cell','auc_pr')[0]:+.4f}; PS − Z1 {pr('PS','Z1','auc_pr')[0]:+.4f}; PS − OLS {pr('PS','OLS','auc_pr')[0]:+.4f}; PS − PCA {pr('PS','PCA','auc_pr')[0]:+.4f}", ""]
        dec["x3"] = dict(R_minus_PS_win=pr('R','PS','win_auc_pr')[0], PS_minus_cell=pr('PS','cell','auc_pr')[0], PS_minus_Z1=pr('PS','Z1','auc_pr')[0], PS_minus_OLS=pr('PS','OLS','auc_pr')[0], PS_minus_PCA=pr('PS','PCA','auc_pr')[0])
    dec["T4"] = A.round(4).to_dict()
    (OUT / f"summary{CFG['tag']}_{spl}.md").write_text("\n".join(L), encoding="utf-8")
    if spl == "val":
        (OUT / f"decision{CFG['tag']}.json").write_text(json.dumps(dict(rule="doc/실험10_계획.md 3절", tol=TOL, **dec), indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "summarize"])
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--calib", default="val", choices=["val", "val_mad"])
    ap.add_argument("--exclude-outliers", action="store_true")
    ap.add_argument("--no-anchors", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--models", nargs="*", default=None, help="기저 목록 (BASES 키). 기본 PS cell Z1 cellfill")
    ap.add_argument("--no-comparators", action="store_true", help="R·OLS·PCA 생략")
    ap.add_argument("--f01-models", nargs="*", default=None)
    ap.add_argument("--tag", default="", help="출력 파일 꼬리표 추가")
    args = ap.parse_args()
    if args.out:
        OUT = OUT / args.out
    CFG.update(calib=args.calib, excl=args.exclude_outliers, anchors=not args.no_anchors,
               tag=("" if args.calib == "val" else "_mad") + ("_excl" if args.exclude_outliers else "") + (("_" + args.tag) if args.tag else ""),
               models=args.models or DEFAULT_MODELS, comparators=not args.no_comparators, f01_models=args.f01_models or ["PS", "cell"])
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "summarize":
        summarize(args.split)
        sys.exit(0)
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    t0 = time.perf_counter()
    rows = []
    for scan in SCANS:
        rows += run_scan(scan, ecfg, dcfg, args.splits)
        pd.DataFrame(rows).to_csv(OUT / f"metrics{CFG['tag']}_{'_'.join(args.splits)}.csv", index=False, encoding="utf-8-sig")
    print(f"done {len(rows)} rows, {time.perf_counter() - t0:.1f}s")
