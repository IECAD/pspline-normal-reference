"""실험 6 1부 추가 3: D3 실제 결측 (doc/실험6_계획.md '1부 추가 3', 사전 등록 2026-10-10). 선택 없음.

녹화: TDoA_Random (주, 실제 결측), TDoA_Tour (대조). 행 TDOA·rx_power·fp_power 를 RANGE·RSS·RSS_FP 이름으로 바꿔
F01 합성 (V5: 거리 + RX−FP, S1, O1, T4) 에 넣는다. 관측 마스크 = obs 이고 값이 유한한 셀.
노드 = 윈도우 동안 GT 위치 평균의 격자 (1 m 주, 0.5 m 보조; oracle).
방향 1: 보정 = val, 평가 = test / 방향 2: 보정 = test, 평가 = val. 학습 노드에 속한 평가 윈도우만 평가.
주입: D2 거리형 시나리오 SC1–SC3 (configs/exp3.yaml, 몫마다 그 시드), 관측 없는 셀 라벨은 뺀다.
출력: results/exp6/d3_real_missing/metrics.csv, coverage.csv
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
from node_model_eval import build, full_spec  # noqa: E402
from src.data import d3 as D3  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.inject import inject_windows  # noqa: E402
from src.data.load import load_windows  # noqa: E402
from src.metrics.anomaly import evaluate  # noqa: E402
from src.model.bank import anchor_scores  # noqa: E402

OUT = ROOT / "results" / "exp6" / "d3_real_missing"
BASES = {"cell": {"time_basis": "cell"}, "cellfill": {"time_basis": "cellfill"},
         "cellinterp": {"time_basis": "cellinterp"}, "PS": {"K_t": 12}, "const": {"time_basis": "const"}}
RECS = ["TDoA_Random", "TDoA_Tour"]
GRIDS = [1.0, 0.5]
NAME = {"TDOA": "RANGE", "rx_power": "RSS", "fp_power": "RSS_FP"}


def load(rec, cfg):
    w = load_windows("d3", rec, cfg)
    hexes = sorted({r.split("_", 1)[1] if r.startswith("TDOA") else r.rsplit("_", 1)[1] for r in w["rows"]})
    aid = {h: f"A{i + 1}" for i, h in enumerate(hexes)}
    rows = []
    for r in w["rows"]:
        m, h = ("TDOA", r.split("_", 1)[1]) if r.startswith("TDOA") else (r.rsplit("_", 1)[0], r.rsplit("_", 1)[1])
        rows.append(f"{NAME[m]}_{aid[h]}")
    X = w["X"].astype(float)
    W = ((w["cells"]["obs"] > 0) & np.isfinite(X)).astype(float)
    b = D3.blinks(cfg, rec)
    g = b.groupby("tidx")[["gx", "gy"]].first()
    t0 = w["meta"]["tidx0"].to_numpy().astype(int)
    T = X.shape[2]
    cen = np.array([g.reindex(np.arange(t, t + T)).mean().to_numpy() for t in t0])
    return X, W, rows, w["meta"]["split"].to_numpy(), cen, len(hexes)


if __name__ == "__main__":
    cfg = load_config()
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    code = ecfg["ids"]["F01"]
    E = ecfg["datasets"]["d2"]
    recs, cov = [], []
    for rec in RECS:
        X, W, rows, split, cen, nA = load(rec, cfg)
        anchors = [f"A{i + 1}" for i in range(nA)]
        for grid in GRIDS:
            ok = np.all(np.isfinite(cen), axis=1)
            nodes = np.array([f"{int(np.floor(x / grid))},{int(np.floor(y / grid))}" if o else "nan"
                              for (x, y), o in zip(cen, ok)])
            tr = (split == "train") & ok
            for d, (cal, ev) in enumerate((("val", "test"), ("test", "val")), start=1):
                ic, ie = (split == cal) & ok, np.flatnonzero((split == ev) & ok)
                covered = np.isin(nodes[ie], np.unique(nodes[tr]))
                ie = ie[covered]
                cov.append(dict(rec=rec, grid=grid, direction=d, eval_split=ev, eval_windows=int(covered.size),
                                covered_windows=int(covered.sum()), coverage=float(covered.mean())))
                rix = [rows.index(f"RANGE_{a}") for a in anchors]
                elig = W[ie][:, rix].any(2)
                for bid, sp in BASES.items():
                    comp, bank, K = build(code, full_spec(sp, ecfg), ecfg, "d2", X[tr], W[tr], rows, nodes[tr],
                                          X[ic], W[ic], nodes[ic])
                    for scn, sc in E["scenarios"].items():
                        for s in E["injection_seeds"][ev]:
                            Xi, lab, _, _ = inject_windows(X[ie], rows, sc, s, dist_metric="RANGE", eligible=elig)
                            lab = np.where(W[ie][:, rix] > 0, lab, np.nan)
                            Xs, Ws = comp.transform(Xi, W[ie], rows)
                            A = anchor_scores(bank.score(Xs, nodes[ie], Ws), comp.anchor_of_row, anchors)
                            seqs = [(A[n], lab[n]) for n in range(len(A))]
                            recs.append(dict(rec=rec, grid=grid, direction=d, basis=bid, scenario=scn, seed=s,
                                             **evaluate(seqs, bank.thr, "cell")))
                print(f"  {rec} grid {grid} dir {d}: 평가 윈도우 {len(ie)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(recs)
    R.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cov).to_csv(OUT / "coverage.csv", index=False, encoding="utf-8-sig")
    sys.stdout.reconfigure(encoding="utf-8")
    print(pd.DataFrame(cov).to_string(index=False))
    key = ["direction", "scenario", "seed"]
    lines = []
    for (rec, grid), g in R.groupby(["rec", "grid"]):
        lines.append(f"[{rec} grid {grid} m] 평균 AUC-PR: " + ", ".join(
            f"{b} {g[g.basis == b].auc_pr.mean():.3f}" for b in BASES))
        for ref in ("cell", "cellfill", "cellinterp", "const"):
            x = (g[g.basis == "PS"].set_index(key).auc_pr - g[g.basis == ref].set_index(key).auc_pr).dropna()
            v = "이김" if x.mean() > 0.017 else ("열세" if x.mean() < -0.017 else "대등")
            lines.append(f"    PS − {ref}: {x.mean():+.4f} (양수 {np.mean(x > 0):.2f}, n {len(x)}) → {v}")
    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
