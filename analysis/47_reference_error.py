"""실험 6 1부 보충: 정상 기준 오차 측정 (doc/실험6_계획.md '1부 보충'). 서술 분석, 검증 세트.

조건 (실험 6 1부와 같음, T = 12): 기준, A (노드당 학습 윈도우 비율 f), C (학습 셀 결측률 r), 기저 cell·PS.
측정: (1) 빈칸 비율 = 원래 학습 관측이 있던 (노드, 행, 시점) 칸 중 조건 적용 후 관측 0 이 된 비율. 예측 = 평균 r^c (C)
      (2) σ_R 증가율 = 보정 잔차 표준편차(검증)의 기준 조건 대비 비율, 행별 기하평균
      (3) 실험 6 1부 AUC-PR (검증) 을 나란히 붙인다
출력: analysis/out/47_reference_error.txt, 47_reference_error.csv
"""
import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from exp6_conditions import CONDS, apply_cond, spec_for  # noqa: E402
from node_model_eval import build, full_spec  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.load import load_windows  # noqa: E402

dcfg = load_config()
ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
code = ecfg["ids"]["F01"]
conds = [c for c in CONDS if c[0] in ("base", "A", "C")]


def counts(Ws, nodes):
    """(노드, 행, 시점) 별 관측 수 -> (n_node, S, T)."""
    return np.stack([(Ws[nodes == nd] > 0).sum(0) for nd in sorted(set(nodes))])


recs = []
for ds in ("d1", "d2"):
    E = ecfg["datasets"][ds]
    units = E["spaces"].items() if ds == "d1" else [(f"env{e}", None) for e in E["envs"]]
    for unit, sp in units:
        full = load_windows("d1", sp["normal"], dcfg) if ds == "d1" else \
            load_windows("d2", f"{unit}_{E['channel_name']}", dcfg)
        X, rows, meta = full["X"], full["rows"], full["meta"]
        split = meta["split"].to_numpy()
        if ds == "d1":
            nodes = np.array([f"{unit}/{n.split('/')[-1]}" for n in meta["node"]])
            W = np.isfinite(X).astype(float)
        else:
            nodes = np.array([f"{unit}/{int(t)}" for t in meta["tidx0"]])
            W = full["cells"]["mask"]
        tr, va = split == "train", split == "val"
        base_sd, base_cnt = {}, None
        for fam, f, cseed, T, r in conds:
            Xtr, Wtr, ntr = apply_cond(X[tr], W[tr], nodes[tr], rows, fam, f, cseed or 0, r)
            for bid in ("cell", "PS"):
                comp, bank, K = build(code, full_spec(spec_for(bid, T), ecfg), ecfg, ds, Xtr, Wtr, rows, ntr,
                                      X[va], W[va], nodes[va])
                _, Ws = comp.transform(Xtr, Wtr, rows)
                cnt = counts(Ws, ntr)
                if fam == "base" and bid == "cell":
                    base_cnt = cnt
                if fam == "base":
                    base_sd[bid] = bank.sd.copy()
                had = base_cnt > 0 if cnt.shape == base_cnt.shape else None
                empty = float((cnt[had] == 0).mean()) if had is not None else np.nan
                pred = float(np.mean(r ** base_cnt[had])) if (fam == "C" and had is not None) else np.nan
                ratio = float(np.exp(np.mean(np.log(np.maximum(bank.sd, 1e-12) / np.maximum(base_sd[bid], 1e-12)))))
                recs.append(dict(dataset=ds, space=unit, family=fam, level={"base": "기준", "A": f, "C": r}[fam],
                                 cond_seed=-1 if cseed is None else cseed, basis=bid, empty_frac=empty,
                                 empty_pred=pred, sigma_ratio=ratio,
                                 median_windows_per_node=float(np.median([np.sum(ntr == nd) for nd in set(ntr)]))))
        print(f"  {ds} {unit} done", flush=True)
R = pd.DataFrame(recs)
R["level"] = R["level"].astype(str)
# 실험 6 1부 AUC-PR (검증) 을 붙인다
M = pd.concat([pd.read_csv(ROOT / f"results/exp6/conditions/metrics_{d}_val.csv") for d in ("d1", "d2")])
M = M[((M.dataset == "d1") & M.scenario.str.match(r"^SC\d$")) | ((M.dataset == "d2") & M.scenario.str.match(r"^SC\dm$"))]
M["level"] = M["level"].astype(str)
auc = M.groupby(["dataset", "space", "family", "level", "cond_seed", "basis"]).auc_pr.mean().rename("auc_pr")
R = R.merge(auc.reset_index(), on=["dataset", "space", "family", "level", "cond_seed", "basis"], how="left")
R.to_csv(ROOT / "analysis/out/47_reference_error.csv", index=False, encoding="utf-8-sig")
g = R.groupby(["dataset", "space", "family", "level", "basis"]).agg(
    windows=("median_windows_per_node", "mean"), empty=("empty_frac", "mean"), pred=("empty_pred", "mean"),
    sigma_ratio=("sigma_ratio", "mean"), auc=("auc_pr", "mean")).reset_index()
idx = ["dataset", "space", "family", "level"]
P = g.pivot_table(index=idx, columns="basis", values=["sigma_ratio", "auc"]).round(3)
P.columns = [f"{a}_{b}" for a, b in P.columns]
info = g[g.basis == "cell"].set_index(idx)[["windows", "empty", "pred"]]
P = info.join(P).reset_index().round(3)
txt = (__doc__.strip().splitlines()[0] + "\n\n" + P.to_string(index=False) +
       "\n\nempty = 관측 빈칸 비율 (실측), pred = 평균 r^c (예측, 계열 C), sigma_ratio = σ_R 기준 대비, auc = 실험 6 검증 AUC-PR")
(ROOT / "analysis/out/47_reference_error.txt").write_text(txt, encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print(txt)
