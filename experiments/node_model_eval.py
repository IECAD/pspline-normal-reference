"""노드별 정상 모델(Bank) 평가 공통 코드: 단순 기준선 사다리(exp1_simple_baselines.py)와 실험 4(exp4_components.py).

평가 절차는 experiments/exp3_composition.py 의 run_d1 / run_d2 와 같다:
  같은 정상 파일·분할·마스크, 같은 주입 시나리오·시드 (configs/exp3.yaml), 같은 임계 규칙, 앵커 점수 = 행 최댓값.
  D1 실제 NLOS (A′ 보조) 는 테스트 분할에서만 평가한다.
spec (구성 사양, 생략 시 F01 과 같음):
  node: node | global, time_basis: pspline | const | cell, stats: val | train,
  K_t: 정수 | "adaptive", diff_order: 1 | 2, lam_mode: sap | sap_row | 숫자, score_set: [R, K, C] 의 부분집합
"""
import json
import os
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.data.inject import inject_windows  # noqa: E402
from src.data.load import load_windows  # noqa: E402
from src.metrics.anomaly import evaluate  # noqa: E402
from src.model.bank import NormalSurfaceBank, anchor_scores  # noqa: E402
from src.model.compose import Composer  # noqa: E402

RES = ROOT / "results"
_sig = ROOT / "cache" / "d2_nlos_signature.parquet"         # analysis/42_d2_nlos_signature.py (D2 학습 몫 NLOS 서명)
SIG = pd.read_parquet(_sig) if _sig.exists() else pd.DataFrame(columns=["env", "d_range", "d_rss", "d_rssfp"])
DEFAULTS = dict(node="node", time_basis="pspline", stats="val", K_t=None, diff_order=2, lam_mode="sap",
                score_set=["R"], robust=None)
ADAPTIVE_K = list(range(5, 13))


def full_spec(spec, ecfg):
    s = {**DEFAULTS, **(spec or {})}
    if s["K_t"] is None:
        s["K_t"] = ecfg["model"]["K_t"]
    return s


def _bank(spec, ecfg, T, K, penalties):
    m = ecfg["model"]
    return NormalSurfaceBank(T, K, penalties, sap=m["sap"], eps=m["eps"], time_basis=spec["time_basis"],
                             diff_order=spec["diff_order"], lam_mode=spec["lam_mode"],
                             score_set=tuple(spec["score_set"]), robust=spec.get("robust"))


def build(code, spec, ecfg, ds, Xtr, Wtr, rows, nodes_tr, Xva, Wva, nodes_va, adaptive_ks=None):
    """합성 + Bank 적합 (adaptive K_t 면 학습 BIC 로 K 선택) + 보정. 반환: comp, bank, 선택한 K.
    adaptive_ks: adaptive 후보 K (기본 ADAPTIVE_K = 5..12, 실험 6 은 5..T)."""
    V = code.get("V_d1", code["V"]) if ds == "d1" else code["V"]
    comp = Composer(ds, V, code["S"], code["O"], code["T"]).fit(Xtr, Wtr, rows, nodes_tr)
    Xs, Ws = comp.transform(Xtr, Wtr, rows)
    T = Xtr.shape[2]
    if spec["K_t"] == "adaptive" and spec["time_basis"] == "pspline":
        bics = {K: _bank(spec, ecfg, T, K, comp.sensor_penalties()).fit(Xs, Ws, nodes_tr).bic(Xs, Ws, nodes_tr)
                for K in (adaptive_ks or ADAPTIVE_K)}
        K = min(bics, key=bics.get)
    else:
        K = ecfg["model"]["K_t"] if spec["K_t"] == "adaptive" else int(spec["K_t"])
    bank = _bank(spec, ecfg, T, K, comp.sensor_penalties()).fit(Xs, Ws, nodes_tr)
    Xv, Wv = comp.transform(Xva, Wva, rows)
    bank.calibrate(Xv, Wv, nodes_va, ecfg["model"]["threshold_pct"], stats=spec["stats"], Xtr=Xs, Wtr=Ws,
                   nodes_tr=nodes_tr)
    return comp, bank, K


def _info(bank, K):
    return dict(threshold=bank.thr, fit_rmse=bank.fit_rmse, fit_s_per_node=bank.fit_time_per_node,
                infer_ms_per_window=getattr(bank, "infer_ms_per_window", np.nan), K_t_used=K)


def _lam_rows(bank, tag, K):
    return [dict(tag=tag, node=nd, K_t_used=K, **{f"lam_{k}": v for k, v in lam.items()})
            for nd, lam in bank.lambdas().items()]


def run_d1(code, spec, ecfg, dcfg, splits, scorer=None):
    """scorer(bank, Xv, Wv, nodes_va) -> 점수기 (score, thr). 없으면 Bank 의 A_R (실험 1 ST-SSD형 비교에서 바꿔 끼움)."""
    out, lams = [], []
    E = ecfg["datasets"]["d1"]
    W1 = lambda w: np.isfinite(w["X"]).astype(float)
    for space, sp in E["spaces"].items():
        tr, va = (load_windows("d1", sp["normal"], dcfg, split=s) for s in ("train", "val"))
        rows = tr["rows"]
        if spec["node"] == "global":
            node = lambda w: np.array([f"{space}/all"] * len(w["X"]))
        else:
            node = lambda w: np.array([f"{space}/{n.split('/')[-1]}" for n in w["meta"]["node"]])
        comp, bank, K = build(code, spec, ecfg, "d1", tr["X"], W1(tr), rows, node(tr), va["X"], W1(va), node(va))
        if scorer is not None:
            bank = scorer(bank, *comp.transform(va["X"], W1(va), rows), node(va))
        lams += _lam_rows(bank, f"d1/{space}", K)
        anchors = [f"A{a}" for a in range(1, 5)]
        for split in splits:
            w = va if split == "val" else load_windows("d1", sp["normal"], dcfg, split=split)
            for scn, sc in E["scenarios"].items():
                for s in E["injection_seeds"][split]:
                    Xi, lab, _, _ = inject_windows(w["X"], rows, sc, s)
                    Xs, Ws = comp.transform(Xi, W1(w), rows)
                    A = anchor_scores(bank.score(Xs, node(w), Ws), comp.anchor_of_row, anchors)
                    seqs = [(A[n], lab[n]) for n in range(len(A))]
                    out.append(dict(dataset="d1", space=space, split=split, scenario=scn, seed=s,
                                    **evaluate(seqs, bank.thr, "time"), **_info(bank, K)))
        if "test" in splits:                                   # 실제 NLOS (보조, 테스트만)
            for f in sp["real"]:
                w = load_windows("d1", f, dcfg)
                Xs, Ws = comp.transform(w["X"], W1(w), rows)
                nd = node(w)
                A = anchor_scores(bank.score(Xs, nd, Ws), comp.anchor_of_row, anchors)
                dist_ix = [rows.index(f"DIST_{a}") for a in anchors]
                for labname in (E["real_label"], "lab_A"):
                    lab = w["cells"][labname][:, dist_ix]
                    seqs = [(A[n], lab[n]) for n in range(len(A)) if nd[n] in bank.M]
                    out.append(dict(dataset="d1", space=space, split="test", scenario=f"real:{f}:{labname}", seed=-1,
                                    coverage=float(np.mean([x in bank.M for x in nd])),
                                    **evaluate(seqs, bank.thr, "time"), **_info(bank, K)))
    return out, lams


def run_d2(code, spec, ecfg, dcfg, splits, scorer=None):
    out, lams = [], []
    E = ecfg["datasets"]["d2"]
    anchors = [f"A{a}" for a in range(1, 9)]
    for env in E["envs"]:
        name = f"env{env}_{E['channel_name']}"
        tr, va = (load_windows("d2", name, dcfg, split=s) for s in ("train", "val"))
        rows = tr["rows"]
        if spec["node"] == "global":
            node = lambda w: np.array([f"env{env}/all"] * len(w["X"]))
        else:
            node = lambda w: np.array([f"env{env}/{int(t)}" for t in w["meta"]["tidx0"]])
        comp, bank, K = build(code, spec, ecfg, "d2", tr["X"], tr["cells"]["mask"], rows, node(tr),
                              va["X"], va["cells"]["mask"], node(va))
        if scorer is not None:
            bank = scorer(bank, *comp.transform(va["X"], va["cells"]["mask"], rows), node(va))
        lams += _lam_rows(bank, f"d2/env{env}", K)
        rix = [rows.index(f"RANGE_{a}") for a in anchors]
        pool = SIG[SIG.env == env][["d_range", "d_rss", "d_rssfp"]].to_numpy()
        scen = [(n, sc, None) for n, sc in E["scenarios"].items()] + \
               [(f"{n}m", sc, pool) for n, sc in E["scenarios"].items() if E.get("multi_metric")]
        for split in splits:
            w = va if split == "val" else load_windows("d2", name, dcfg, split=split)
            nl = w["cells"]["lab_D"][:, rix]
            elig = (nl == 0).all(2)
            for scn, sc, dp in scen:
                for s in E["injection_seeds"][split]:
                    Xi, lab, _, _ = inject_windows(w["X"], rows, sc, s, dist_metric="RANGE", eligible=elig,
                                                   delta_pool=dp, delta_metrics=["RANGE", "RSS", "RSS_FP"])
                    lab = np.where(nl == 1, np.nan, lab)
                    Xs, Ws = comp.transform(Xi, w["cells"]["mask"], rows)
                    A = anchor_scores(bank.score(Xs, node(w), Ws), comp.anchor_of_row, anchors)
                    seqs = [(A[n], lab[n]) for n in range(len(A))]
                    out.append(dict(dataset="d2", space=f"env{env}", split=split, scenario=scn, seed=s,
                                    **evaluate(seqs, bank.thr, E["level"]), **_info(bank, K)))
    return out, lams


RUNNERS = {"d1": run_d1, "d2": run_d2}


def run_and_save(id_, code, spec, ecfg, dcfg, datasets, splits, extra_meta=None, tag="base"):
    """구성 하나를 데이터별로 실행해 results/<id>/<tag>/ 에 저장. 분할마다 파일을 나눈다 (val 결과를 덮어쓰지 않게)."""
    spec = full_spec(spec, ecfg)
    suf = "" if splits == ["val"] else "_" + "_".join(splits)
    frames = []
    for ds in datasets:
        t0 = time.perf_counter()
        rows, lams = RUNNERS[ds](code, spec, ecfg, dcfg, splits)
        d = RES / id_ / tag
        d.mkdir(parents=True, exist_ok=True)
        M = pd.DataFrame(rows)
        M.insert(0, "id", id_)
        M.insert(1, "run", tag)
        for k in ("V", "S", "O", "T"):
            M[k] = code.get("V_d1", code["V"]) if (k == "V" and ds == "d1") else code[k]
        for k, v in spec.items():
            M[k] = "+".join(v) if isinstance(v, list) else v
        M.to_csv(d / f"metrics_{ds}{suf}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(lams).to_csv(d / f"lambdas_{ds}{suf}.csv", index=False, encoding="utf-8-sig")
        (d / f"config_{ds}{suf}.json").write_text(json.dumps(dict(
            id=id_, spec=spec, code=code, dataset=ds, splits=splits, model=ecfg["model"],
            exp_config="configs/exp3.yaml", data_config="configs/data.yaml", elapsed_s=time.perf_counter() - t0,
            **(extra_meta or {})), indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{id_} {ds} {splits}: {len(M)} rows, {time.perf_counter() - t0:.1f}s", flush=True)
        frames.append(M)
    return pd.concat(frames, ignore_index=True)
