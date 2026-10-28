"""실험 1: ST-SSD형 (정적) 비교 (doc/실험1_ST-SSD_계획.md, 사전 등록 2026-10-11).

변형: G1 전역 배경(T1, O1, 시간 1차 차분) · G2 전역 배경(T1, O2 seriation) · N 노드별 배경(PS 와 같음)
탐색 (검증): γ ∈ {0,1,2,3,4} × K_a ∈ {항등, 6} = 10 구성, 데이터별 검증 AUC-PR 평균 최대를 선택
사용법:
  python experiments/exp1_stssd.py search          # 검증 탐색 -> results/exp1_stssd/selection.json
  python experiments/exp1_stssd.py test            # 선택 구성으로 테스트 한 번
출력: results/exp1_stssd/metrics_<단계>_<데이터>.csv, selection.json
"""
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from node_model_eval import full_spec, run_d1, run_d2  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.model.stssd import StSsdScorer  # noqa: E402

OUT = ROOT / "results" / "exp1_stssd"
GRID = [(g, ka) for g in (0, 1, 2, 3, 4) for ka in ("I", 6)]
VIEWS = {"d1": r"^SC\d$", "d2": r"^SC\dm$"}


def variants(ecfg):
    f01 = dict(ecfg["ids"]["F01"])
    g1 = {**f01, "T": "T1", "O": "O1"}
    g2 = {**f01, "T": "T1", "O": "O2"}
    glob = {"node": "global", "K_t": 12, "diff_order": 1}
    return {"G1": (g1, glob), "G2": (g2, glob), "N": (f01, {"K_t": 12})}


def run(ds, code, spec, ecfg, dcfg, splits, g, ka):
    scorer = lambda bank, Xv, Wv, nv: StSsdScorer(bank, g, ka, Xv.shape[2]).calibrate(
        Xv, Wv, nv, ecfg["model"]["threshold_pct"])
    rows, _ = (run_d1 if ds == "d1" else run_d2)(code, full_spec(spec, ecfg), ecfg, dcfg, splits, scorer=scorer)
    return pd.DataFrame(rows).assign(gamma=g, Ka=str(ka))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "search"
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    V = variants(ecfg)
    if stage == "search":
        sel = {}
        for ds in ("d1", "d2"):
            parts = []
            for vid, (code, spec) in V.items():
                for g, ka in GRID:
                    t0 = time.perf_counter()
                    parts.append(run(ds, code, spec, ecfg, dcfg, ["val"], g, ka).assign(variant=vid))
                    print(f"  {ds} {vid} γ={g} Ka={ka}: {time.perf_counter() - t0:.1f}s", flush=True)
            M = pd.concat(parts, ignore_index=True)
            M.to_csv(OUT / f"metrics_search_{ds}.csv", index=False, encoding="utf-8-sig")
            d = M[M.scenario.astype(str).str.match(VIEWS[ds])]
            t = d.groupby(["variant", "gamma", "Ka"]).auc_pr.mean().reset_index()
            for vid in V:
                r = t[t.variant == vid].sort_values("auc_pr", ascending=False).iloc[0]
                sel.setdefault(vid, {})[ds] = dict(gamma=float(r.gamma), Ka=r.Ka, val_auc_pr=float(r.auc_pr))
        (OUT / "selection.json").write_text(json.dumps(dict(rule="doc/실험1_ST-SSD_계획.md 2절", selection=sel),
                                                       indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(sel, indent=1, ensure_ascii=False))
    else:
        sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))["selection"]
        for ds in ("d1", "d2"):
            parts = []
            for vid, (code, spec) in V.items():
                s = sel[vid][ds]
                ka = s["Ka"] if s["Ka"] == "I" else int(s["Ka"])
                parts.append(run(ds, code, spec, ecfg, dcfg, ["test"], s["gamma"], ka).assign(variant=vid))
                print(f"  test {ds} {vid} γ={s['gamma']} Ka={ka}", flush=True)
            pd.concat(parts, ignore_index=True).to_csv(OUT / f"metrics_test_{ds}.csv", index=False,
                                                       encoding="utf-8-sig")
