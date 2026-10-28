"""실험 3: 합성 방식 ablation (B0, E01-E18). 계획: doc/합성방식_실험계획_최종.md

사용법:
  python experiments/exp3_composition.py --ids B0 E01 --datasets d1 d2 d4
  (--ids 생략 시 B0, E01-E18 전부. O3 는 시드 0-9 반복)
출력: results/<ID>/<run>/metrics_<datasets>.csv, lambdas_<datasets>.csv, config_<datasets>.json
      (run = base | o3seed<k>. 데이터별로 따로 실행해도 파일이 겹치지 않음)
규칙: 스케일·순서·임계값은 정상 학습/검증만으로 정한다. 테스트 결과는 선택에 쓰지 않는다 (F01/F02 는 별도 스크립트).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# 작은 행렬(≤ 400×400)에서 OpenBLAS 멀티스레드가 100배 가까이 느리다 (inv 192×192: 78 ms vs 0.8 ms, 이 PC 측정).
# numpy 를 불러오기 전에 스레드를 1로 고정하고, 대신 프로세스 단위로 병렬 실행한다.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.config import load_config  # noqa: E402
from src.data.inject import inject_windows  # noqa: E402
from src.data.load import load_windows  # noqa: E402
from src.metrics.anomaly import evaluate  # noqa: E402
from src.model.bank import NormalSurfaceBank, anchor_scores  # noqa: E402
from src.model.compose import Composer, NotApplicable, split_row  # noqa: E402

RES = ROOT / "results"
_sig = ROOT / "cache" / "d2_nlos_signature.parquet"         # analysis/42_d2_nlos_signature.py (D2 학습 몫 NLOS 서명)
SIG = pd.read_parquet(_sig) if _sig.exists() else pd.DataFrame(columns=["env", "d_range", "d_rss", "d_rssfp"])


def fit_bank(comp, ecfg, X, W, nodes, T):
    m = ecfg["model"]
    Xs, Ws = comp.transform(X, W, ROWS[0])
    bank = NormalSurfaceBank(T, m["K_t"], comp.sensor_penalties(), sap=m["sap"], eps=m["eps"]).fit(Xs, Ws, nodes)
    if comp.S == "S4":                                    # 블록별 정상 잔차분산의 역수로 가중 후 다시 적합
        R = bank.residual(Xs, nodes)
        var = {b: float(np.nanmean(np.where(Ws[:, comp.block_ids == b] > 0, R[:, comp.block_ids == b], np.nan) ** 2))
               for b in np.unique(comp.block_ids)}
        comp.set_block_weights(var)
        Xs, Ws = comp.transform(X, W, ROWS[0])
        bank = NormalSurfaceBank(T, m["K_t"], comp.sensor_penalties(), sap=m["sap"], eps=m["eps"]).fit(Xs, Ws, nodes)
    return bank


ROWS = [None]   # 현재 데이터의 원 행 이름 (fit_bank 에서 참조)


def lam_rows(bank, tag):
    return [dict(tag=tag, node=nd, **{f"lam_{k}": v for k, v in lam.items()}) for nd, lam in bank.lambdas().items()]


def base_info(bank):
    return dict(threshold=bank.thr, fit_rmse=bank.fit_rmse, fit_s_per_node=bank.fit_time_per_node,
                infer_ms_per_window=getattr(bank, "infer_ms_per_window", np.nan))


# ----------------------------------------------------------------------------- D1
def run_d1(code, ecfg, dcfg, seed):
    out, lams = [], []
    E = ecfg["datasets"]["d1"]
    for space, sp in E["spaces"].items():
        tr, va, te = (load_windows("d1", sp["normal"], dcfg, split=s) for s in ("train", "val", "test"))
        rows = tr["rows"]
        ROWS[0] = rows
        node = lambda w: np.array([f"{space}/{n.split('/')[-1]}" for n in w["meta"]["node"]])
        W1 = lambda w: np.isfinite(w["X"]).astype(float)
        comp = Composer("d1", code.get("V_d1", code["V"]), code["S"], code["O"], code["T"], seed=seed).fit(
            tr["X"], W1(tr), rows, node(tr))
        T = tr["X"].shape[2]
        bank = fit_bank(comp, ecfg, tr["X"], W1(tr), node(tr), T)
        Xv, Wv = comp.transform(va["X"], W1(va), rows)
        bank.calibrate(Xv, Wv, node(va), ecfg["model"]["threshold_pct"])
        lams += lam_rows(bank, f"d1/{space}")
        anchors = [f"A{a}" for a in range(1, 5)]
        for split, w in (("val", va), ("test", te)):
            for scn, sc in E["scenarios"].items():
                for s in E["injection_seeds"][split]:
                    Xi, lab, an_, _ = inject_windows(w["X"], rows, sc, s)
                    Xs, _ = comp.transform(Xi, W1(w), rows)
                    A = anchor_scores(bank.score(Xs, node(w)), comp.anchor_of_row, anchors)
                    seqs = [(A[n], lab[n]) for n in range(len(A))]
                    out.append(dict(dataset="d1", space=space, split=split, scenario=scn, seed=s,
                                    **evaluate(seqs, bank.thr, "time"), **base_info(bank)))
        for f in sp["real"]:                                   # 실제 NLOS (보조, 테스트만)
            w = load_windows("d1", f, dcfg)
            Xs, _ = comp.transform(w["X"], W1(w), rows)
            nd = node(w)
            A = anchor_scores(bank.score(Xs, nd), comp.anchor_of_row, anchors)
            dist_ix = [rows.index(f"DIST_{a}") for a in anchors]
            for labname in (E["real_label"], "lab_A"):
                lab = w["cells"][labname][:, dist_ix]
                seqs = [(A[n], lab[n]) for n in range(len(A)) if nd[n] in bank.M]
                out.append(dict(dataset="d1", space=space, split="test", scenario=f"real:{f}:{labname}", seed=-1,
                                coverage=float(np.mean([x in bank.M for x in nd])),
                                **evaluate(seqs, bank.thr, "time"), **base_info(bank)))
    return out, lams


# ----------------------------------------------------------------------------- D2
# 노드별 Bank 를 라벨 마스크로 학습하면 같은 노드의 테스트 NLOS 를 그대로 맞힌다 (누수, analysis/out/41_check_d2_leakage.txt).
# 그래서 D2 는 (1) LOS 셀 인공 주입 (ablation 지표) 과 (2) 위치 홀드아웃 실제 NLOS 로 평가한다 (사용자 결정 2026-10-09).
D2_SUBSET = [None]   # 앵커 부분추출 (계획 8.2 앵커 수 효과): None 이면 8개 전부


def _subset(w, keep_rows):
    ix = [w["rows"].index(r) for r in keep_rows]
    return dict(X=w["X"][:, ix], rows=list(keep_rows), meta=w["meta"],
                cells={k: v[:, ix] for k, v in w["cells"].items()})


def run_d2(code, ecfg, dcfg, seed):
    out, lams = [], []
    E = ecfg["datasets"]["d2"]
    anchors = D2_SUBSET[0] or [f"A{a}" for a in range(1, 9)]
    keep = lambda w: _subset(w, [r for r in w["rows"] if split_row(r)[1] in anchors])
    for env in E["envs"]:
        # (1) 경로 시간축 윈도우, 노드 = 경로 구간. 학습 셀 = LOS. 주입은 윈도우 전체가 LOS 인 앵커에만, NLOS 셀은 평가 제외
        name = f"env{env}_{E['channel_name']}"
        tr, va, te = (keep(load_windows("d2", name, dcfg, split=s)) for s in ("train", "val", "test"))
        rows = tr["rows"]
        ROWS[0] = rows
        node = lambda w: np.array([f"env{env}/{int(t)}" for t in w["meta"]["tidx0"]])
        comp = Composer("d2", code["V"], code["S"], code["O"], code["T"], seed=seed).fit(
            tr["X"], tr["cells"]["mask"], rows, node(tr))
        T = tr["X"].shape[2]
        bank = fit_bank(comp, ecfg, tr["X"], tr["cells"]["mask"], node(tr), T)
        Xv, Wv = comp.transform(va["X"], va["cells"]["mask"], rows)
        bank.calibrate(Xv, Wv, node(va), ecfg["model"]["threshold_pct"])
        lams += lam_rows(bank, f"d2/env{env}")
        rix = [rows.index(f"RANGE_{a}") for a in anchors]
        pool = SIG[SIG.env == env][["d_range", "d_rss", "d_rssfp"]].to_numpy()
        scen = [(n, sc, None) for n, sc in E["scenarios"].items()] + \
               [(f"{n}m", sc, pool) for n, sc in E["scenarios"].items() if E.get("multi_metric")]
        for split, w in (("val", va), ("test", te)):
            nl = w["cells"]["lab_D"][:, rix]                       # (N, 8, T) 기하 NLOS
            elig = (nl == 0).all(2)
            for scn, sc, dp in scen:                               # 'SCkm' = 다지표 주입 (실제 NLOS 서명)
                for s in E["injection_seeds"][split]:
                    Xi, lab, _, _ = inject_windows(w["X"], rows, sc, s, dist_metric="RANGE", eligible=elig,
                                                   delta_pool=dp, delta_metrics=["RANGE", "RSS", "RSS_FP"])
                    lab = np.where(nl == 1, np.nan, lab)
                    Xs, _ = comp.transform(Xi, w["cells"]["mask"], rows)
                    A = anchor_scores(bank.score(Xs, node(w)), comp.anchor_of_row, anchors)
                    seqs = [(A[n], lab[n]) for n in range(len(A))]
                    out.append(dict(dataset="d2", space=f"env{env}", split=split, scenario=scn, seed=s,
                                    **evaluate(seqs, bank.thr, E["level"]), **base_info(bank)))
        # (2) 위치 홀드아웃 실제 NLOS: path_idx % 4 in {0,2} 학습, 1 검증, 3 테스트. 홀드아웃 위치의 곡면 =
        #     경로상 양옆 가장 가까운 학습 위치 곡면의 거리 역가중 평균.
        #     주의: 이웃 위치의 라벨만으로도 AUC-PR 0.98-1.00 (NLOS 가 경로를 따라 이어짐) -> 이웃 마스크를 통한 누수.
        #     실험 3 에서는 제외 (사용자 결정 2026-10-09, configs/exp3.yaml d2.holdout). 실험 5 에서 다시 설계
        if not E.get("holdout", False):
            continue
        w = keep(load_windows("d2", f"env{env}_ch5_within", dcfg))
        pidx = w["meta"]["path_idx"].to_numpy()
        role = np.select([pidx % 4 == 1, pidx % 4 == 3], ["val", "test"], "train")
        trm = role == "train"
        compH = Composer("d2", code["V"], code["S"], code["O"], code["T"], seed=seed).fit(
            w["X"][trm], w["cells"]["mask"][trm], rows, pidx[trm])
        bankH = fit_bank(compH, ecfg, w["X"][trm], w["cells"]["mask"][trm], pidx[trm], T)
        tr_pos = np.array(sorted(bankH.M))
        for p in np.unique(pidx[~trm]):
            left, right = tr_pos[tr_pos < p], tr_pos[tr_pos > p]
            nb = ([left[-1]] if len(left) else []) + ([right[0]] if len(right) else [])
            wts = np.array([1.0 / abs(p - q) for q in nb])
            bankH.M[p] = sum(wt * bankH.M[q] for wt, q in zip(wts, nb)) / wts.sum()
        vm, tm = role == "val", role == "test"
        Xv, Wv = compH.transform(w["X"][vm], w["cells"]["mask"][vm], rows)
        bankH.calibrate(Xv, Wv, pidx[vm], ecfg["model"]["threshold_pct"])
        Xt, _ = compH.transform(w["X"][tm], w["cells"]["mask"][tm], rows)
        A = anchor_scores(bankH.score(Xt, pidx[tm]), compH.anchor_of_row, anchors)
        lab = w["cells"]["lab_D"][tm][:, rix]
        seqs = [(A[n], lab[n]) for n in range(len(A))]
        out.append(dict(dataset="d2", space=f"env{env}", split="test", scenario="real_holdout", seed=-1,
                        **evaluate(seqs, bankH.thr, E["level"]), **base_info(bankH)))
    return out, lams


# ----------------------------------------------------------------------------- D4
def run_d4(code, ecfg, dcfg, seed):
    out, lams = [], []
    E = ecfg["datasets"]["d4"]
    from src.data import d4 as D4
    for ds in E["datasets"]:
        for ent in D4.entities(ds):
            name = f"{ds}_{ent}"
            tr, va, te = (load_windows("d4", name, dcfg, split=s) for s in ("train", "val", "test"))
            rows = tr["rows"]
            ROWS[0] = rows
            node = lambda w: np.zeros(len(w["X"]), dtype=int)
            W1 = lambda w: np.isfinite(w["X"]).astype(float)
            comp = Composer("d4", "V3", code["S"], code["O"], code["T"], seed=seed).fit(tr["X"], W1(tr), rows, node(tr))
            T = tr["X"].shape[2]
            bank = fit_bank(comp, ecfg, tr["X"], W1(tr), node(tr), T)
            Xv, Wv = comp.transform(va["X"], W1(va), rows)
            bank.calibrate(Xv, Wv, node(va), ecfg["model"]["threshold_pct"])
            lams += lam_rows(bank, f"d4/{name}")
            Xs, _ = comp.transform(te["X"], W1(te), rows)
            A = anchor_scores(bank.score(Xs, node(te)), comp.anchor_of_row, rows)   # 변수 순서로 되돌림
            ix = te["meta"].sort_values("tidx0").index.to_numpy()
            seqs = [(np.concatenate(A[ix], 1), np.concatenate(te["cells"]["lab_I"][ix], 1))]
            tl = [np.concatenate(te["cells"]["lab_P"][ix][:, 0], 0)]
            out.append(dict(dataset=f"d4_{ds}", space=ent, split="test", scenario="real", seed=-1,
                            **evaluate(seqs, bank.thr, "time", time_labels=tl), **base_info(bank)))
    return out, lams


RUNNERS = {"d1": run_d1, "d2": run_d2, "d4": run_d4}


def run_one(id_, code, datasets, ecfg, dcfg, seed, tag):
    rows, lams, skipped = [], [], []
    t0 = time.perf_counter()
    for ds in datasets:
        b0 = ecfg["ids"]["B0"]
        v_only = code["V"] != b0["V"] and all(code[k] == b0[k] for k in ("S", "O", "T"))
        if ds == "d4" and "V" not in ecfg["datasets"]["d4"]["axes"] and v_only:
            skipped.append(dict(dataset=ds, reason="①축만 바꾼 구성은 D4 에서 하지 않음 (계획 5.1절)"))
            continue
        try:
            r, l = RUNNERS[ds](code, ecfg, dcfg, seed)
            rows += r
            lams += l
        except NotApplicable as e:
            skipped.append(dict(dataset=ds, reason=str(e)))
    d = RES / id_ / tag
    d.mkdir(parents=True, exist_ok=True)
    suf = "_".join(datasets)                 # 데이터별 병렬 실행 시 파일이 겹치지 않게
    M = pd.DataFrame(rows)
    if len(M):
        M.insert(0, "id", id_)
        M.insert(1, "run", tag)
        for k in ("V", "S", "O", "T"):
            M[k] = code[k]
    M.to_csv(d / f"metrics_{suf}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(lams).to_csv(d / f"lambdas_{suf}.csv", index=False, encoding="utf-8-sig")
    (d / f"config_{suf}.json").write_text(json.dumps(dict(
        id=id_, run=tag, code=code, o3_seed=seed, datasets=datasets, skipped=skipped, model=ecfg["model"],
        exp_config="configs/exp3.yaml", data_config="configs/data.yaml",
        data_version=str((dcfg["paths"]["cache"] / "windows" / "data_config_used.yaml")),
        elapsed_s=time.perf_counter() - t0), indent=1, ensure_ascii=False), encoding="utf-8")
    return M, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--datasets", nargs="*", default=["d1", "d2", "d4"])
    ap.add_argument("--config", default="configs/exp3.yaml")
    ap.add_argument("--d2-anchors", type=int, default=0, help="D2 앵커 부분추출 개수 (0 = 8개 전부)")
    ap.add_argument("--subset-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    ecfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    dcfg = load_config()
    ids = args.ids or list(ecfg["ids"])
    allM = []
    subsets = [(None, "")]
    if args.d2_anchors:
        subsets = []
        for ss in args.subset_seeds:
            ch = sorted(np.random.default_rng(1000 + ss).choice(8, args.d2_anchors, replace=False) + 1)
            subsets.append(([f"A{a}" for a in ch], f"_a{args.d2_anchors}s{ss}"))
    for id_ in ids:
        code = ecfg["ids"][id_]
        runs0 = [(s, f"o3seed{s}") for s in ecfg["random_order_seeds"]] if code["O"] == "O3" else [(0, "base")]
        runs = [(s, t + suf, sub) for s, t in runs0 for sub, suf in subsets]
        for seed, tag, sub in runs:
            D2_SUBSET[0] = sub
            t0 = time.perf_counter()
            M, sk = run_one(id_, code, args.datasets, ecfg, dcfg, seed, tag)
            allM.append(M)
            print(f"{id_} {tag}: {len(M)} rows, skipped={[s['dataset'] for s in sk]}, {time.perf_counter() - t0:.1f}s",
                  flush=True)
    print("done. 집계: python experiments/exp3_summarize.py")
