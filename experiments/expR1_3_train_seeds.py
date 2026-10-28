"""분석 R1-3 3절: 딥 모델 학습 seed 변동 (doc/분석R1-3_계획.md, 사전 등록 2026-10-28 21:32). 서술용(판정 바꾸지 않음).

깨끗한 D1·D2, 원고 표 4 의 선택 설정(exp6_comparators_missing.selected), model seed 0–4. 테스트 분할.
PS 값은 원고 표 4 의 PS(results/C03/base/metrics_<ds>_test.csv)와 짝짓는다(같은 주입·seed).
출력: results/expR1/train_seeds/metrics.csv, summary.md
"""
import json
import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from exp1_baselines import prep, run_model  # noqa: E402
from exp6_comparators_missing import selected  # noqa: E402
from src.data import d2 as D2  # noqa: E402
from src.data.config import load_config  # noqa: E402

OUT = ROOT / "results" / "expR1" / "train_seeds"
MODELS = ["TRANAD", "GDN", "CAE2D"]
SEEDS = [0, 1, 2, 3, 4]
TOL = {"d1": 0.0165, "d2": 0.0171}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    mall = D2.load_meta(dcfg); mall = mall[mall.ch == dcfg["d2"]["channel"]]
    recs = []
    for ds in ("d1", "d2"):
        E = ecfg["datasets"][ds]
        units = list(E["spaces"].items()) if ds == "d1" else [(f"env{e}", None) for e in E["envs"]]
        for unit, sp in units:
            D = prep(ds, unit, sp, ecfg, dcfg, ["test"], mall)
            for m in MODELS:
                cfg, variant = selected(m, ds)
                for s in SEEDS:
                    for r in run_model(D, m, {**cfg, "seed": s}, variant, "test"):
                        recs.append(dict(dataset=ds, space=unit, model=m, train_seed=s, variant=variant,
                                         cfg=json.dumps(cfg, sort_keys=True), **r))
                print(f"  {ds} {unit} {m} done", flush=True)
    M = pd.DataFrame(recs)
    M.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    L = ["# 딥 모델 학습 seed 변동 (분석 R1-3 3절, 서술)", "",
         "깨끗한 테스트, 원고 표 4 설정. 'SD 학습 seed' = (단위, 시나리오, 주입 seed) 마다 학습 seed 5개 사이 SD 의 평균. "
         "'SD 주입 seed' = 학습 seed 0 에서 (단위, 시나리오) 마다 주입 seed 사이 SD 의 평균.", ""]
    key = ["space", "scenario", "seed"]
    for ds, rx in (("d1", r"^SC\d$"), ("d2", r"^SC\dm$")):
        ps = pd.read_csv(ROOT / f"results/C03/base/metrics_{ds}_test.csv")
        ps = ps[ps.scenario.astype(str).str.match(rx)].set_index(key).auc_pr
        sub = M[(M.dataset == ds) & M.scenario.astype(str).str.match(rx)]
        L += [f"## {ds}", "", "| 모델 | 평균 AUC-PR | SD 학습 seed | SD 주입 seed | PS − 모델 (seed 0–4) | 판정 (seed 0–4) |", "|---|---|---|---|---|---|"]
        for m in MODELS:
            t = sub[sub.model == m]
            sd_tr = t.groupby(key).auc_pr.std(ddof=1).mean()
            sd_inj = t[t.train_seed == 0].groupby(["space", "scenario"]).auc_pr.std(ddof=1).mean()
            deltas, verdicts = [], []
            for s in SEEDS:
                x = (ps - t[t.train_seed == s].set_index(key).auc_pr).dropna()
                deltas.append(x.mean())
                verdicts.append("이김" if x.mean() > TOL[ds] else ("짐" if x.mean() < -TOL[ds] else "대등"))
            L.append(f"| {m} | {t.auc_pr.mean():.3f} | {sd_tr:.4f} | {sd_inj:.4f} | " + ", ".join(f"{d:+.3f}" for d in deltas) +
                     f" | {', '.join(verdicts)} |")
        L.append("")
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
