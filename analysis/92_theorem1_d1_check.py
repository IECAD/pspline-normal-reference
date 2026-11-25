"""정리 1 회고 검증 — D1 실데이터 (2026-11-13).

47_reference_error.py 의 σ_R 측정을 네 참조(cell = unit fill, cellfill, cellinterp, PS)와 실험 6 보간 실험의
조건(기준 + 무작위 r 0.3/0.5/0.7 + 버스트 4수준, 조건 시드 0–2)으로 넓힌다. 학습 데이터에만 조건을 적용하고,
보정은 깨끗한 검증 창에서 한다(원고 3.4절). 참값 m 을 모르므로 V_e 는 명제 4′ 로 간접 측정한다:
  ratio = σ_R(조건) / σ_R(기준), 행별 기하평균  →  x = ratio² − 1 ∝ V_e − V_e0   (단위마다 상수배)
검증: (a) (방, 조건, 시드) 안에서 네 참조의 AUC-PR 순서가 x 의 역순과 같은가 (Spearman).
      (b) 방마다 (조건, 시드, 참조) 점을 모아 AUC-PR = Φ0 + κ x 를 적합하고 참조 더미가 설명력을 더하는지.
AUC-PR 은 results/exp6/imputation/metrics_d1_{val,test}.csv 의 저장값을 쓴다(검증 = 선택용, 테스트 = 보고용).
출력: analysis/out/92_theorem1_d1_check.md, 92_theorem1_d1_points.csv
사용법: python analysis/92_theorem1_d1_check.py
"""
import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from exp6_conditions import spec_for  # noqa: E402
from exp6_imputation import BASES, CONDS, apply_any, level_of  # noqa: E402
from node_model_eval import build, full_spec  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.load import load_windows  # noqa: E402

OUT_MD = ROOT / "analysis/out/92_theorem1_d1_check.md"
OUT_CSV = ROOT / "analysis/out/92_theorem1_d1_points.csv"
REFS = ["cell", "cellfill", "cellinterp", "PS"]


def ols(y, X):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    return coef, float(r @ r), len(y) - X.shape[1]


def f_test(rss0, dof0, rss1, dof1):
    F = ((rss0 - rss1) / (dof0 - dof1)) / (rss1 / dof1)
    return F, 1.0 - stats.f.cdf(F, dof0 - dof1, dof1)


def measure():
    dcfg = load_config()
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    code = ecfg["ids"]["F01"]
    E = ecfg["datasets"]["d1"]
    recs = []
    for unit, sp in E["spaces"].items():
        full = load_windows("d1", sp["normal"], dcfg)
        X, rows, meta = full["X"], full["rows"], full["meta"]
        split = meta["split"].to_numpy()
        nodes = np.array([f"{unit}/{n.split('/')[-1]}" for n in meta["node"]])
        W = np.isfinite(X).astype(float)
        tr, va = split == "train", split == "val"
        base_sd = {}
        for c in CONDS:
            fam, f, cseed, T, prm = c
            Xtr, Wtr, ntr = apply_any(X[tr], W[tr], nodes[tr], rows, c)
            for bid in REFS:
                comp, bank, K = build(code, full_spec(spec_for(bid, T, BASES), ecfg), ecfg, "d1", Xtr, Wtr, rows, ntr,
                                      X[va], W[va], nodes[va])
                sd = np.maximum(bank.sd, 1e-12)
                if fam == "base":
                    base_sd[bid] = sd.copy()
                ratio = float(np.exp(np.mean(np.log(sd / base_sd[bid]))))
                recs.append(dict(space=unit, family=fam, level=level_of(c), cond_seed=-1 if cseed is None else cseed,
                                 basis=bid, sigma_ratio=ratio, sd_mean=float(np.mean(sd))))
            print(f"  {unit} {level_of(c)} seed {cseed} done", flush=True)
    return pd.DataFrame(recs)


def main():
    R = measure()
    R["x"] = R["sigma_ratio"] ** 2 - 1.0
    key = ["space", "family", "level", "cond_seed", "basis"]
    for split in ("val", "test"):
        M = pd.read_csv(ROOT / f"results/exp6/imputation/metrics_d1_{split}.csv")
        M = M[M.scenario.str.match(r"^SC\d$")]
        M["cond_seed"] = M["cond_seed"].fillna(-1).astype(int)
        M["level"] = M["level"].astype(str)
        a = M.groupby(key).auc_pr.mean().rename(f"auc_{split}").reset_index()
        R = R.merge(a, on=key, how="left")
    R.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    L = ["# 정리 1 회고 검증 — D1 (네 참조, 실험 6 보간 조건)", "",
         "x = σ_R 비율² − 1 ∝ V_e − V_e0 (명제 4′; 방마다 상수배). AUC-PR 은 실험 6 보간 실험의 저장값(장면 SC1–SC3, 주입 시드 평균).", ""]
    # (a) 조건 안 순서
    L += ["## (a) (방, 조건, 시드) 안에서 AUC-PR 순서 대 x 순서 (참조 cell·cellfill·cellinterp·PS)", "",
          "| 방 | 조건 | 시드 | Spearman val | Spearman test | AUC(test) 순서 | x 순서(작은 것부터) | x: cell / fill / interp / PS |",
          "|---|---|---|---|---|---|---|---|"]
    rhos_v, rhos_t = [], []
    for (sp_, fam, lv, cs), sub in R[R.family != "base"].groupby(["space", "family", "level", "cond_seed"]):
        if sub["basis"].nunique() < 4 or sub["auc_test"].isna().any():
            continue
        rv = stats.spearmanr(sub["auc_val"], -sub["x"])[0]
        rt = stats.spearmanr(sub["auc_test"], -sub["x"])[0]
        rhos_v.append(rv)
        rhos_t.append(rt)
        oa = " > ".join(sub.sort_values("auc_test", ascending=False)["basis"])
        ox = " < ".join(sub.sort_values("x")["basis"])
        xs = " / ".join(f"{sub[sub.basis == b]['x'].iloc[0]:.2f}" for b in REFS)
        L.append(f"| {sp_} | {lv} | {cs} | {rv:+.2f} | {rt:+.2f} | {oa} | {ox} | {xs} |")
    rv, rt = np.array(rhos_v), np.array(rhos_t)
    L += ["", f"조건 {len(rv)}개. Spearman 중앙값 val {np.median(rv):+.2f} / test {np.median(rt):+.2f}; "
              f"ρ ≥ 0.8 인 조건 val {int((rv >= 0.8).sum())}, test {int((rt >= 0.8).sum())}.", ""]
    # (b) 방별 한 곡선
    L += ["## (b) 방마다 (조건, 시드, 참조) 점: AUC-PR(test) = Φ0 + κ x, 참조 더미 F 검정", "",
          "| 방 | 점 | x 범위 | Φ0 | κ | R² | 잔차 SD | 더미 F | p | 더미 계수 (fill, interp, PS; cell 기준) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for sp_, sub in R.dropna(subset=["auc_test"]).groupby("space"):
        sub = sub[sub["x"] < 2.0]                                       # 2차 전개 영역(큰 x 는 unit fill 붕괴)
        y = sub["auc_test"].to_numpy()
        X0 = np.column_stack([np.ones(len(sub)), sub["x"]])
        c0, rss0, dof0 = ols(y, X0)
        D = pd.get_dummies(sub["basis"]).reindex(columns=["cellfill", "cellinterp", "PS"], fill_value=0).to_numpy(float)
        c1, rss1, dof1 = ols(y, np.column_stack([X0, D]))
        F, p = f_test(rss0, dof0, rss1, dof1)
        r2 = 1 - rss0 / float(((y - y.mean()) ** 2).sum())
        L.append(f"| {sp_} | {len(sub)} | {sub['x'].min():.2f}–{sub['x'].max():.2f} | {c0[0]:.3f} | {c0[1]:+.3f} | {r2:.3f} | "
                 f"{np.sqrt(rss0 / dof0):.4f} | {F:.2f} | {p:.3g} | {c1[2]:+.4f}, {c1[3]:+.4f}, {c1[4]:+.4f} |")
    L.append("")
    # 전체 표
    L += ["## 측정값", "", "| 방 | 조건 | 시드 | 참조 | σ_R 비율 | x | AUC val | AUC test |", "|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "—" if pd.isna(v) else f"{v:.3f}"  # noqa: E731
    for _, d in R.sort_values(["space", "family", "level", "cond_seed", "basis"]).iterrows():
        L.append(f"| {d['space']} | {d['level']} | {d['cond_seed']} | {d['basis']} | {d['sigma_ratio']:.3f} | {d['x']:+.3f} | "
                 f"{fmt(d['auc_val'])} | {fmt(d['auc_test'])} |")
    OUT_MD.write_text("\n".join(L), encoding="utf-8", newline="\n")
    print("written", OUT_MD.relative_to(ROOT))


if __name__ == "__main__":
    main()
