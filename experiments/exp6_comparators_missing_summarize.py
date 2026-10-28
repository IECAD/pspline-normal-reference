"""실험 6 1부 추가 5 집계: 결측 조건별 PS − 모델 짝지은 Δ(허용오차·양수 비율·부트스트랩 CI), 모델별 하락폭(조건 − 기준),
PS·셀 평균의 실험 6 1부 수치와의 재현 확인. 출력: results/exp6/comparators_missing/summary_<split>.md, table_<split>.csv
사용법: python experiments/exp6_comparators_missing_summarize.py --split val|test
"""
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exp6" / "comparators_missing"
TOL = {"d1": 0.0165, "d2": 0.0171}
KEY = ["space", "scenario", "seed", "cond_seed"]
NAMES = {"PS": "PS (proposed)", "cell": "cell mean", "OLS": "OLS", "RRR": "RRR", "TRANS": "Transformer (masked)", "RBCPHI": "RBC^φ",
         "GDN": "GDN", "OMNI": "OmniAnomaly", "LAPRPCA": "Laplacian-RPCA", "CAE2D": "2D Conv-AE", "TRANAD": "TranAD"}
VIEW = {"d1": r"^SC\d$", "d2": r"^SC\dm$"}


def boot_ci(d, label, B=5000):
    rng = np.random.default_rng(zlib.crc32(label.encode("utf-8")))
    b = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(B)])
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="val"); args = ap.parse_args()
    lines = [f"# 실험 6 1부 추가 5 집계 ({args.split})", "", "자동 생성: `experiments/exp6_comparators_missing_summarize.py`. 계획: `doc/실험6_계획.md` 1부 추가 5.", ""]
    rows_out = []
    for ds in ("d1", "d2"):
        p = OUT / f"metrics_{ds}_{args.split}.csv"
        if not p.exists():
            continue
        M = pd.read_csv(p); M["level"] = M.level.astype(str); M = M[M.scenario.astype(str).str.match(VIEW[ds])]
        tol = TOL[ds]
        base = M[M.family == "base"]
        base_mean = base.groupby("model").auc_pr.mean()
        lines += [f"## {ds} (허용오차 {tol}; 짝 = 공간·시나리오·주입 시드·조건 시드)", ""]
        conds = [c for c in M[["family", "level"]].drop_duplicates().itertuples(index=False) if c[0] != "base"]
        head = "| 조건 | " + " | ".join(NAMES.get(m, m) for m in ["PS", "cell"] + [m for m in NAMES if m not in ("PS", "cell") and m in M.model.unique()]) + " |"
        models = ["PS", "cell"] + [m for m in NAMES if m not in ("PS", "cell") and m in M.model.unique()]
        lines += ["### 절대 AUC-PR (평균)", "", head, "|" + "---|" * (len(models) + 1)]
        lines.append("| 기준 | " + " | ".join(f"{base_mean.get(m, np.nan):.3f}" for m in models) + " |")
        for fam, lev in conds:
            sub = M[(M.family == fam) & (M.level == lev)]
            lines.append(f"| {fam} {lev} | " + " | ".join(f"{sub[sub.model == m].auc_pr.mean():.3f}" for m in models) + " |")
        lines += ["", "### PS − 모델 (짝지은 Δ, 95 % CI, 양수 비율, 판정) 와 모델 하락폭 (조건 − 기준, 같은 공간·시나리오·주입 시드)", "",
                  "| 조건 | 모델 | PS − 모델 | 95 % CI | 양수 | 판정 | 모델 하락폭 | PS 하락폭 |", "|---|---|---|---|---|---|---|---|"]
        for fam, lev in conds:
            sub = M[(M.family == fam) & (M.level == lev)]
            piv = sub.pivot_table(index=KEY, columns="model", values="auc_pr")
            base_piv = base.pivot_table(index=["space", "scenario", "seed"], columns="model", values="auc_pr")
            for m in models:
                if m == "PS" or m not in piv.columns:
                    continue
                d = (piv["PS"] - piv[m]).dropna().to_numpy()
                lo, hi = boot_ci(d, f"{ds}|{fam}|{lev}|{m}")
                verdict = "PS 우위" if d.mean() > tol else ("PS 열세" if d.mean() < -tol else "대등")
                # 하락폭: 조건 값(조건 시드 평균) − 기준 값, 같은 (공간, 시나리오, 시드)
                cm = sub[sub.model == m].groupby(["space", "scenario", "seed"]).auc_pr.mean()
                drop_m = (cm - base_piv[m]).dropna().mean()
                cp = sub[sub.model == "PS"].groupby(["space", "scenario", "seed"]).auc_pr.mean()
                drop_ps = (cp - base_piv["PS"]).dropna().mean()
                lines.append(f"| {fam} {lev} | {NAMES.get(m, m)} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {(d > 0).mean():.2f} | {verdict} | {drop_m:+.3f} | {drop_ps:+.3f} |")
                rows_out.append(dict(dataset=ds, split=args.split, family=fam, level=lev, model=m, n=len(d), delta=d.mean(), ci_lo=lo, ci_hi=hi,
                                     frac_pos=(d > 0).mean(), verdict=verdict, drop_model=drop_m, drop_ps=drop_ps,
                                     auc_model=sub[sub.model == m].auc_pr.mean(), auc_ps=sub[sub.model == "PS"].auc_pr.mean()))
        # 재현 확인: PS·cell 의 실험 6 1부 수치와 대조
        ref = {("C", "0.5"): ("conditions", "C", "0.5"), ("C", "0.7"): ("conditions", "C", "0.7"),
               ("D", "pi=0.7,L=4"): ("missing_patterns", "D", "pi=0.7,L=4"), ("P", "P3"): ("phy_masks", "P", "P3")}
        lines += ["", "### 재현 확인 (PS·cell vs 실험 6 1부 표, 같은 분할)", "", "| 조건 | 이번 PS / cell | 1부 PS / cell |", "|---|---|---|"]
        for (fam, lev), (sub_dir, f2, l2) in ref.items():
            t = ROOT / "results" / "exp6" / sub_dir / f"table_{args.split}.csv"
            if not t.exists():
                continue
            T = pd.read_csv(t); T["level"] = T.level.astype(str).str.replace("r=", "")
            r = T[(T.family == f2) & (T.level == l2)]
            sub = M[(M.family == fam) & (M.level == lev)]
            if len(r) and len(sub):
                lines.append(f"| {fam} {lev} | {sub[sub.model == 'PS'].auc_pr.mean():.4f} / {sub[sub.model == 'cell'].auc_pr.mean():.4f} | "
                             f"{r[f'{ds}_PS'].iloc[0]:.4f} / {r[f'{ds}_cell'].iloc[0]:.4f} |")
        lines.append("")
    pd.DataFrame(rows_out).to_csv(OUT / f"table_{args.split}.csv", index=False, encoding="utf-8-sig")
    (OUT / f"summary_{args.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8"); print("\n".join(lines))


if __name__ == "__main__":
    main()
