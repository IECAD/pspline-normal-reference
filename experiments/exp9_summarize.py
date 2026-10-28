"""실험 9 집계와 판정 (doc/실험9_계획.md 2절).

데이터(SMD, ASD)마다 PS − 모델 엔터티 짝지은 AUC-PR Δ, 허용오차 0.0171 (우위 / 대등 / 열세).
RQ7: 두 데이터 모두 PS 가 비교 모델 8개(셀 평균·노드 평균 제외)의 과반(5개 이상)에 대해 대등 이상 -> "경쟁력 있게 적용",
     한 데이터만 -> "제한적 적용", 둘 다 아님 -> "적용 범위 밖".
사용법: python experiments/exp9_summarize.py
출력: results/exp9/summary.md, paired.csv, decision.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exp9"
TOL = 0.0171
COMPARATORS = ["PCA", "OLS", "IF", "LSTMAE", "USADFIX2", "CAE1D", "GDN", "TRANS"]  # USAD·USADFIX 는 판정에서 뺌 (기록)
sys.path.insert(0, str(ROOT / "experiments"))
from exp1_baselines_summarize import NAMES  # noqa: E402

if __name__ == "__main__":
    L = ["# 실험 9: 범용성 — D4 (테스트)", "", "자동 생성: `experiments/exp9_summarize.py`. 계획: `doc/실험9_계획.md`.",
         f"PS − 모델 엔터티 짝지은 AUC-PR, 허용오차 {TOL}.", ""]
    rows, ok = [], {}
    for ds in ("smd", "asd"):
        M = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob(f"metrics_{ds}*.csv"))], ignore_index=True)
        M = M.drop_duplicates(["entity", "model"], keep="last")
        t = M.groupby("model")[["auc_pr", "hitrate", "cell_f1", "smearing", "train_s"]].mean()
        ps = M[M.model == "PS"].set_index("entity").auc_pr
        L += [f"## {ds.upper()} (엔터티 {M.entity.nunique()}개)", "",
              "| 모델 | AUC-PR | HitRate | 셀 F1 | Smearing | 학습 s | PS − 모델 (양수 비율) | 판정 |", "|---|---|---|---|---|---|---|---|"]
        order = ["PS"] + sorted([m for m in t.index if m != "PS"], key=lambda m: -t.loc[m, "auc_pr"])
        n_ok = 0
        for m in order:
            r = t.loc[m]
            if m == "PS":
                L.append(f"| **PS (제안)** | {r.auc_pr:.3f} | {r.hitrate:.3f} | {r.cell_f1:.3f} | {r.smearing:.3f} | "
                         f"{r.train_s:.1f} | — | — |")
                continue
            x = (ps - M[M.model == m].set_index("entity").auc_pr).dropna()
            v = "PS 우위" if x.mean() > TOL else ("PS 열세" if x.mean() < -TOL else "대등")
            if m in COMPARATORS and v != "PS 열세":
                n_ok += 1
            rows.append(dict(dataset=ds, model=m, auc_pr=r.auc_pr, hitrate=r.hitrate, delta=x.mean(),
                             frac_pos=(x > 0).mean(), n=len(x), verdict=v))
            L.append(f"| {NAMES.get(m, m)} | {r.auc_pr:.3f} | {r.hitrate:.3f} | {r.cell_f1:.3f} | {r.smearing:.3f} | "
                     f"{r.train_s:.1f} | {x.mean():+.4f} ({(x > 0).mean():.2f}) | {v} |")
        n_cmp = sum(m in t.index for m in COMPARATORS)
        ok[ds] = dict(n_not_worse=n_ok, n_comparators=n_cmp, majority=n_ok >= 5)
        L += ["", f"PS가 열세가 아닌 비교 모델: {n_ok}/{n_cmp}", ""]
    k = sum(v["majority"] for v in ok.values())
    rq7 = {2: "경쟁력 있게 적용", 1: "제한적 적용", 0: "적용 범위 밖"}[k]
    L.append(f"**RQ7 판정: {rq7}** ({json.dumps(ok, ensure_ascii=False)})")
    pd.DataFrame(rows).round(4).to_csv(OUT / "paired.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(dict(rule="doc/실험9_계획.md 2절", tol=TOL, by_dataset=ok, rq7=rq7),
                                                  indent=1, ensure_ascii=False), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))
