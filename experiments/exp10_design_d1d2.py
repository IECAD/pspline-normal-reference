"""실험 10-3: 설계 변형(PS1, PSH, PS1H)이 D1·D2 를 해치지 않는지 (doc/실험10_계획.md 10-3). 검증·테스트 한 번씩.
ID: P01 = {K_t 12, diff 1}, H03 = {K_t 12, Huber}, H01 = {K_t 12, diff 1, Huber}. 비교 = C03 (PS). 허용오차 D1 0.0165 · D2 0.0171.
사용법: python experiments/exp10_design_d1d2.py [run|summarize]
출력: results/<ID>/base/, results/exp10/design/d1d2_summary.md
"""
import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from node_model_eval import run_and_save  # noqa: E402
from src.data.config import load_config  # noqa: E402

IDS = {"P01": {"K_t": 12, "diff_order": 1}, "H03": {"K_t": 12, "robust": "huber"}, "H01": {"K_t": 12, "diff_order": 1, "robust": "huber"}}
TOL = {"d1": 0.0165, "d2": 0.0171}
VIEWS = {"d1": r"^SC\d$", "d2": r"^SC\dm$"}
KEY = ["space", "scenario", "seed"]

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "run"
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    if stage == "run":
        for id_, spec in IDS.items():
            run_and_save(id_, ecfg["ids"]["F01"], spec, ecfg, dcfg, ["d1", "d2"], ["val"], extra_meta={"plan": "doc/실험10_계획.md 10-3"})
            run_and_save(id_, ecfg["ids"]["F01"], spec, ecfg, dcfg, ["d1", "d2"], ["test"], extra_meta={"plan": "doc/실험10_계획.md 10-3"})
    L = ["# 실험 10-3: D1·D2 에서 설계 변형 vs PS (C03)", "", "짝지은 AUC-PR Δ (변형 − PS). 허용오차 D1 0.0165 · D2 0.0171.", "",
         "| ID | 변형 | 분할 | D1 PS / 변형 (Δ, 양수) | D2 PS / 변형 (Δ, 양수) | 판정 |", "|---|---|---|---|---|---|"]
    for id_, spec in IDS.items():
        for spl, suf in (("val", ""), ("test", "_test")):
            cells = []
            verdict = []
            for ds in ("d1", "d2"):
                P = pd.read_csv(ROOT / "results" / "C03" / "base" / f"metrics_{ds}{suf}.csv")
                V = pd.read_csv(ROOT / "results" / id_ / "base" / f"metrics_{ds}{suf}.csv")
                P = P[(P.split == spl) & P.scenario.astype(str).str.match(VIEWS[ds])]
                V = V[(V.split == spl) & V.scenario.astype(str).str.match(VIEWS[ds])]
                x = (V.set_index(KEY).auc_pr - P.set_index(KEY).auc_pr).dropna()
                cells.append(f"{P.auc_pr.mean():.3f} / {V.auc_pr.mean():.3f} ({x.mean():+.4f}, {(x > 0).mean():.2f})")
                verdict.append("우위" if x.mean() > TOL[ds] else ("열세" if x.mean() < -TOL[ds] else "대등"))
            L.append(f"| {id_} | {spec} | {spl} | {cells[0]} | {cells[1]} | D1 {verdict[0]}, D2 {verdict[1]} |")
    out = ROOT / "results" / "exp10" / "design"
    out.mkdir(parents=True, exist_ok=True)
    (out / "d1d2_summary.md").write_text("\n".join(L), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))
