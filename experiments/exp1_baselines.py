"""실험 1 비교 모델 1차 (doc/실험1_비교모델_계획.md, 사전 등록 2026-10-11).

사용법:
  python experiments/exp1_baselines.py search     # 검증 탐색 -> results/exp1_baselines/selection.json
  python experiments/exp1_baselines.py test       # 선택 구성으로 테스트 한 번
  python experiments/exp1_baselines.py search 2   # 2차 모델 (계획 "2차") -> results/exp1_baselines/batch2/
평가는 실험 3·4 와 같다 (같은 주입·시드·임계 규칙·앵커 최댓값). 공정성 규칙(계획 1절):
  - 입력 raw (F01 합성 값) / nc (노드별 셀 평균 M01 을 뺀 잔차) 를 검증으로 고른다
  - D2 학습: NLOS(마스크 0) 셀을 정상 기준값으로 채움. 테스트 입력에는 마스크를 쓰지 않는다 (라벨 누수 방지)
  - 점수 = 셀 |오차| / 행별 검증 SD (검증 LOS 셀), 임계값 = 검증 점수 97.5 백분위
출력: results/exp1_baselines/metrics_<단계>_<데이터>.csv, selection.json
"""
import json
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
from node_model_eval import SIG  # noqa: E402
from src.baselines.models import GRIDS, GRIDS2, GRIDS3, GRIDS4, GRIDS5, GRIDS6, MODELS  # noqa: E402
from src.data import d2 as D2  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.inject import inject_windows  # noqa: E402
from src.data.load import load_windows  # noqa: E402
from src.metrics.anomaly import evaluate  # noqa: E402
from src.model.bank import NormalSurfaceBank, anchor_scores  # noqa: E402
from src.model.compose import Composer  # noqa: E402

OUT = OUT_BASE = ROOT / "results" / "exp1_baselines"
VIEWS = {"d1": r"^SC\d$", "d2": r"^SC\dm$"}
EPS = 1e-6


def prep(ds, unit, sp, ecfg, dcfg, splits, mall=None):
    """한 공간(환경)의 학습·검증 합성 데이터, M01 기준, 평가 주입 세트."""
    code = ecfg["ids"]["F01"]
    E = ecfg["datasets"][ds]
    if ds == "d1":
        tr, va = (load_windows("d1", sp["normal"], dcfg, split=s) for s in ("train", "val"))
        W = lambda w: np.isfinite(w["X"]).astype(float)
        node = lambda w: np.array([f"{unit}/{n.split('/')[-1]}" for n in w["meta"]["node"]])
        V = code.get("V_d1", code["V"])
        anchors = [f"A{a}" for a in range(1, 5)]
    else:
        name = f"{unit}_{E['channel_name']}"
        tr, va = (load_windows("d2", name, dcfg, split=s) for s in ("train", "val"))
        W = lambda w: w["cells"]["mask"]
        node = lambda w: np.array([f"{unit}/{int(t)}" for t in w["meta"]["tidx0"]])
        V = code["V"]
        anchors = [f"A{a}" for a in range(1, 9)]
    rows = tr["rows"]
    comp = Composer(ds, V, code["S"], code["O"], code["T"]).fit(tr["X"], W(tr), rows, node(tr))
    Xs_tr, Ws_tr = comp.transform(tr["X"], W(tr), rows)
    Xs_va, Ws_va = comp.transform(va["X"], W(va), rows)
    bank = NormalSurfaceBank(Xs_tr.shape[2], 8, {}, time_basis="cell").fit(Xs_tr, Ws_tr, node(tr))
    dname = "DIST" if ds == "d1" else "RANGE"
    dix = [rows.index(f"{dname}_{a}") for a in anchors]
    D = dict(ds=ds, unit=unit, comp=comp, rows=rows, anchors=anchors, bank=bank, dix=dix, level=E["level"],
             tr=dict(X=tr["X"], Xs=Xs_tr, Ws=Ws_tr, n=node(tr), w=tr), va=dict(X=va["X"], Xs=Xs_va, Ws=Ws_va, n=node(va), w=va))
    if ds == "d2":                                                   # 실제 3D 거리 (oracle 잔차용)
        m = mall[mall.env == int(unit[3:])]
        m = m.assign(true_d=np.sqrt((m.X_TAG - m.X_ANCHOR) ** 2 + (m.Y_TAG - m.Y_ANCHOR) ** 2 + (m.Z_TAG - m.Z_ANCHOR) ** 2))
        td = m.groupby(["path_idx", "anchor"]).true_d.first().unstack().reindex(columns=range(1, 9))
        D["true_d"] = td.to_numpy()                                  # (npos, 8)
    D["evals"] = {}
    for split in splits:
        w = va if split == "val" else load_windows(ds, sp["normal"] if ds == "d1" else f"{unit}_{E['channel_name']}",
                                                     dcfg, split=split)
        sets = []
        if ds == "d1":
            for scn, sc in E["scenarios"].items():
                for s in E["injection_seeds"][split]:
                    Xi, lab, _, _ = inject_windows(w["X"], rows, sc, s)
                    sets.append((scn, s, Xi, lab))
        else:
            nl = w["cells"]["lab_D"][:, dix]
            pool = SIG[SIG.env == int(unit[3:])][["d_range", "d_rss", "d_rssfp"]].to_numpy()
            scen = [(k, sc, None) for k, sc in E["scenarios"].items()] + \
                   [(f"{k}m", sc, pool) for k, sc in E["scenarios"].items()]
            for scn, sc, dp in scen:
                for s in E["injection_seeds"][split]:
                    Xi, lab, _, _ = inject_windows(w["X"], rows, sc, s, dist_metric="RANGE", eligible=(nl == 0).all(2),
                                                   delta_pool=dp, delta_metrics=["RANGE", "RSS", "RSS_FP"])
                    sets.append((scn, s, Xi, np.where(nl == 1, np.nan, lab)))
        D["evals"][split] = dict(w=w, n=node(w), W=W(w), sets=sets)
    return D


def model_input(D, variant, Xs, nodes, Ws=None, train=False):
    Z = Xs.copy() if variant == "raw" else D["bank"].residual(Xs, nodes)
    if train:
        return np.where(Ws > 0, np.nan_to_num(Z), 0.0)
    return np.nan_to_num(Z)


def run_model(D, mname, cfg, variant, split):
    """범용 모델 한 구성: 학습 -> 검증 보정 -> 평가 세트 점수. 반환: 지표 행 목록."""
    t0 = time.perf_counter()
    m = MODELS[mname](**cfg).fit(model_input(D, variant, D["tr"]["Xs"], D["tr"]["n"], D["tr"]["Ws"], train=True))
    train_s = time.perf_counter() - t0
    Ev = m.errors(model_input(D, variant, D["va"]["Xs"], D["va"]["n"]))
    ev = np.where(D["va"]["Ws"] > 0, Ev, np.nan)
    sd = np.nanstd(np.moveaxis(ev, 1, 0).reshape(ev.shape[1], -1), 1)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    thr = float(np.nanpercentile(ev / sd[None, :, None], 97.5))
    out = []
    E = D["evals"][split]
    for scn, s, Xi, lab in E["sets"]:
        Xs, _ = D["comp"].transform(Xi, E["W"], D["rows"])
        t1 = time.perf_counter()
        A = m.errors(model_input(D, variant, Xs, E["n"])) / sd[None, :, None]
        infer = 1000 * (time.perf_counter() - t1) / max(len(Xs), 1)
        Aa = anchor_scores(A, D["comp"].anchor_of_row, D["anchors"])
        out.append(dict(scenario=scn, seed=s, train_s=train_s, infer_ms=infer,
                        **evaluate([(Aa[n], lab[n]) for n in range(len(Aa))], thr, D["level"])))
    return out


def uwb_scores(D, method, Xraw, part):
    """UWB 전용 규칙 앵커 점수 (N, A, T). part: 'tr' / 'va' / eval dict (oracle 잔차에 필요한 원값·위치)."""
    rows, anchors = D["rows"], D["anchors"]
    if method == "RXFP":
        rss = Xraw[:, [rows.index(f"RSS_{a}") for a in anchors]]
        fp = Xraw[:, [rows.index(f"RSS_FP_{a}") for a in anchors]]
        return rss - fp
    y = Xraw[:, D["dix"]]
    if method == "TIMEHIST":
        med = np.nanmedian(y, axis=2, keepdims=True)
        mad = np.nanmedian(np.abs(y - med), axis=2, keepdims=True)
        return np.abs(y - med) / (1.4826 * mad + 1e-3)
    if method == "ORACLE":
        return oracle_resid(D, y, part)
    raise ValueError(method)


def oracle_resid(D, y, part):
    w = part["w"]
    if D["ds"] == "d1":                                              # 방법 A 오차 셀 (원값 기준) + 주입량
        base = w["X"][:, D["dix"]]
        return w["cells"]["errA"][:, D["dix"]] + (y - base)
    P = w["meta"]["tidx0"].to_numpy().astype(int)[:, None] + np.arange(y.shape[2])[None]
    return y - np.moveaxis(D["true_d"][P], 2, 1)


def run_uwb(D, method, split):
    tr = D["tr"]
    if method == "ORACLE":                                          # 앵커별 LOS 통계 (학습)
        r = uwb_scores(D, method, tr["X"], tr)
        lm = tr["w"]["cells"]["mask"][:, D["dix"]] if D["ds"] == "d2" else np.isfinite(r)
        rr = np.where(lm > 0, r, np.nan)
        mu, sd = np.nanmean(rr, axis=(0, 2)), np.nanstd(rr, axis=(0, 2))
        f = lambda X, part: np.abs(uwb_scores(D, method, X, part) - mu[None, :, None]) / (sd[None, :, None] + EPS)
    else:
        f = lambda X, part: uwb_scores(D, method, X, part)
    va = D["va"]
    lv = va["w"]["cells"]["mask"][:, D["dix"]] if D["ds"] == "d2" else np.ones(1)
    sv = f(va["X"], va)
    thr = float(np.nanpercentile(np.where(lv > 0, sv, np.nan), 97.5))
    out = []
    E = D["evals"][split]
    for scn, s, Xi, lab in E["sets"]:
        A = f(Xi, E)
        out.append(dict(scenario=scn, seed=s, train_s=0.0, infer_ms=np.nan,
                        **evaluate([(A[n], lab[n]) for n in range(len(A))], thr, D["level"])))
    return out


N_ROWS = {"d1": 4, "d2": 16}                                         # F01 합성 행 수 (D1 V1 4앵커, D2 V5 8앵커 × 2)


SUPERSEDED = {2: {"GDN", "LAPRPCA"}}                                 # 2차 수정(batch 3)으로 대체 -> 2차 테스트에서 뺌


def configs(ds, batch=1):
    if batch == 6:                                                   # 3차 정정 (계획 "3차 정정"): Anomaly Transformer minimax 방향
        return [(m, c, v) for m, grid in GRIDS6.items() for c in grid for v in ("raw", "nc")]
    if batch == 5:                                                   # 3차 (계획 "3차", sA C4): TranAD · Anomaly Transformer
        return [(m, c, v) for m, grid in GRIDS5.items() for c in grid for v in ("raw", "nc")]
    if batch == 4:                                                   # 추가 수정: USADFIX2 (계획 "2차 수정" 추가 수정)
        return [(m, c, v) for m, grid in GRIDS4.items() for c in grid for v in ("raw", "nc")]
    if batch == 3:                                                   # 2차 수정 (계획 "2차 수정"). RBCPHI = PCA 선택 구성
        out = [(m, c, v) for m, grid in GRIDS3.items() for c in grid for v in ("raw", "nc")]
        pca = json.loads((OUT_BASE / "selection.json").read_text(encoding="utf-8"))["selection"]["PCA"][ds]
        return out + [("RBCPHI", {"var": pca["cfg"]["var"]}, pca["variant"])]
    if batch == 2:                                                   # 2차: RRR 랭크는 행 수 이하만
        return [(m, c, v) for m, grid in GRIDS2.items() for c in grid for v in ("raw", "nc")
                if not (m == "RRR" and c["rank"] > N_ROWS[ds])]
    out = [(m, c, v) for m, grid in GRIDS.items() for c in grid for v in ("raw", "nc")]
    out += [(u, {}, "-") for u in (["RXFP"] if ds == "d2" else []) + ["TIMEHIST", "ORACLE"]]
    return out


def units(ds, ecfg):
    E = ecfg["datasets"][ds]
    return list(E["spaces"].items()) if ds == "d1" else [(f"env{e}", None) for e in E["envs"]]


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "search"
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 1               # 2 = 2차, 3 = 2차 수정 (계획 "2차", "2차 수정")
    if batch in (2, 3, 4, 5, 6):
        OUT = OUT / {2: "batch2", 3: "batch2_fix", 4: "batch2_fix2", 5: "batch3", 6: "batch3_fix"}[batch]
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    mall = D2.load_meta(dcfg)
    mall = mall[mall.ch == dcfg["d2"]["channel"]]
    split = "val" if stage == "search" else "test"
    sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))["selection"] if stage == "test" else None
    for ds in ("d1", "d2"):
        recs = []
        for unit, sp in units(ds, ecfg):
            D = prep(ds, unit, sp, ecfg, dcfg, ["val"] if split == "val" else ["test"], mall)
            if split == "test":                                     # 검증 보정용 val 세트 경로는 prep 의 va 를 쓴다
                pass
            todo = configs(ds, batch) if stage == "search" else \
                [(m, s["cfg"], s["variant"]) for m, d in sel.items() if ds in d for s in [d[ds]]
                 if m not in SUPERSEDED.get(batch, set())]
            for m, c, v in todo:
                t0 = time.perf_counter()
                rows = run_uwb(D, m, split) if v == "-" else run_model(D, m, c, v, split)
                for r in rows:
                    recs.append(dict(dataset=ds, space=unit, split=split, model=m, variant=v,
                                     cfg=json.dumps(c, sort_keys=True), **r))
                print(f"  {ds} {unit} {m} {v} {c}: {time.perf_counter() - t0:.1f}s", flush=True)
        M = pd.DataFrame(recs)
        M.to_csv(OUT / f"metrics_{stage}_{ds}.csv", index=False, encoding="utf-8-sig")
    if stage == "search":
        sel = {}
        for ds in ("d1", "d2"):
            M = pd.read_csv(OUT / f"metrics_search_{ds}.csv")
            d = M[M.scenario.astype(str).str.match(VIEWS[ds])]
            t = d.groupby(["model", "variant", "cfg"]).auc_pr.mean().reset_index()
            for m, g in t.groupby("model"):
                r = g.sort_values("auc_pr", ascending=False).iloc[0]
                sel.setdefault(m, {})[ds] = dict(variant=r.variant, cfg=json.loads(r.cfg), val_auc_pr=float(r.auc_pr))
        rule = "doc/실험1_비교모델_계획.md 1–2절" + {1: "", 2: " (2차)", 3: " (2차 수정)", 4: " (2차 수정 추가: USADFIX2)", 5: " (3차: TranAD·Anomaly Transformer)", 6: " (3차 정정: Anomaly Transformer minimax 방향)"}[batch]
        (OUT / "selection.json").write_text(json.dumps(dict(rule=rule, selection=sel),
                                                       indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(sel, indent=1, ensure_ascii=False))
