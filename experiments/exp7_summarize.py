"""실험 7 집계와 판정 (doc/실험7_계획.md 5절).

검증: PS-hard 가 P1 SC1·SC2 모두에서 해법 4개(LS, WLS, Huber, EKF) 중 3개 이상에서 PS-soft 보다 RMSE 가 5 % 넘게 낮으면
      기본 반영 = hard, 아니면 soft.
테스트: 해법·시나리오(SC1, SC2)마다 PS-기본 개선율 = 1 − RMSE / RMSE_none (단위 합친 RMSE) >= 5 % 이고
        짝 단위(공간, 시드)의 80 % 이상에서 RMSE 가 낮으면 충족. LS·WLS 모두 두 시나리오 충족 -> RQ6 "예".
        PS vs cell: RMSE 상대 차이 2 % 안이면 대등. 정상 스텝 RMSE 가 none 보다 2 % 넘게 나쁘면 오경보 비용.
사용법: python experiments/exp7_summarize.py --split val|test
출력: results/exp7/summary_<분할>.md, decision.json (검증), claims_test.json (테스트)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exp7"
SOLVERS = ["LS", "WLS", "Huber", "EKF"]
CONFIGS = ["none", "PS-soft", "PS-hard", "cell-soft", "cell-hard", "oracle-hard"]
if "--ps1h" in sys.argv:                                            # 추가 비교 2 (doc/실험7_계획.md): results/exp7/ps1h
    OUT = OUT / "ps1h"
    CONFIGS = ["none", "PS-soft", "PS1H-soft", "PS1H-hard", "cell-soft", "oracle-hard"]


def pooled_table(P, st, scn, sub="all"):
    d = P[(P.set == st) & (P.scenario == scn) & (P.subset == sub)]
    return d.pivot_table(index="solver", columns="config", values="rmse").reindex(index=SOLVERS + ["EKFseq"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--ps1h", action="store_true")
    args = ap.parse_args()
    sp = args.split
    P = pd.read_csv(OUT / f"pooled_{sp}.csv")
    M = pd.read_csv(OUT / f"metrics_{sp}.csv")
    L = [f"# 실험 7 집계 ({sp})", "", "자동 생성: `experiments/exp7_summarize.py`. 계획: `doc/실험7_계획.md`.", ""]
    dec_path = OUT / "decision.json"
    if sp == "val":
        votes = {}
        for scn in ("SC1", "SC2"):
            t = pooled_table(P, "P1", scn)
            votes[scn] = {sv: bool(t.loc[sv, "PS-hard"] < 0.95 * t.loc[sv, "PS-soft"]) for sv in SOLVERS}
        hard_wins = all(sum(v.values()) >= 3 for v in votes.values())
        default = "hard" if hard_wins else "soft"
        dec_path.write_text(json.dumps(dict(rule="doc/실험7_계획.md 5절", hard_better_by_5pct=votes, default=default),
                                       indent=1, ensure_ascii=False), encoding="utf-8")
        L += [f"**검증 판정: 기본 반영 = {default}** (hard 가 5 % 넘게 나은 해법: " +
              ", ".join(f"{k} {sum(v.values())}/4" for k, v in votes.items()) + ")", ""]
    default = json.loads(dec_path.read_text(encoding="utf-8"))["default"]
    ps = f"PS-{default}"
    claims = {}
    for st, scns in (("P1", ["SC1", "SC2", "SC3"]), ("P3", ["SC1m", "SC2m", "SC3m"]), ("P2", None)):
        if scns is None:
            scns = sorted(P[P.set == st].scenario.unique())
        for scn in scns:
            if P[(P.set == st) & (P.scenario == scn)].empty:
                continue
            L += [f"## {st} {scn} (RMSE m, 단위 합침)", "",
                  "| 해법 | " + " | ".join(CONFIGS) + f" | 개선율 {ps} | 단위 개선 비율 | {ps} / cell | 정상 스텝 비용 | 이상 스텝 개선율 |",
                  "|" + "---|" * (len(CONFIGS) + 6)]
            t, ta, tn = (pooled_table(P, st, scn, s) for s in ("all", "anom", "norm"))
            u = M[(M.set == st) & (M.scenario == scn) & (M.subset == "all")]
            for sv in SOLVERS + ["EKFseq"]:
                if sv not in t.index or not np.isfinite(t.loc[sv, "none"]):
                    continue
                imp = 1 - t.loc[sv, ps] / t.loc[sv, "none"]
                uu = u[u.solver == sv].pivot_table(index=["space", "seed"], columns="config", values="rmse")
                frac = float((uu[ps] < uu["none"]).mean())
                rel = t.loc[sv, ps] / t.loc[sv, f"cell-{default}"] - 1
                cost = tn.loc[sv, ps] / tn.loc[sv, "none"] - 1 if sv in tn.index else np.nan
                imp_a = 1 - ta.loc[sv, ps] / ta.loc[sv, "none"] if sv in ta.index else np.nan
                ok = bool(imp >= 0.05 and frac >= 0.8)
                claims[f"{st}/{scn}/{sv}"] = dict(improvement=imp, frac_units=frac, ps_vs_cell=rel, normal_cost=cost,
                                                  anom_improvement=imp_a, meets=ok,
                                                  oracle_gap=t.loc[sv, ps] / t.loc[sv, "oracle-hard"] - 1)
                L.append(f"| {sv} | " + " | ".join(f"{t.loc[sv, c]:.3f}" for c in CONFIGS) +
                         f" | {imp:+.1%}{' ✓' if ok else ''} | {frac:.2f} | {rel:+.1%} | {cost:+.1%} | {imp_a:+.1%} |")
            L.append("")
    rq6 = all(claims.get(f"P1/{s}/{sv}", {}).get("meets", False) for s in ("SC1", "SC2") for sv in ("LS", "WLS"))
    L += [f"**기본 반영 = {default}. RQ6 (P1 SC1·SC2, LS·WLS 모두 충족): {rq6}** "
          f"({'테스트 판정' if sp == 'test' else '검증 값, 판정은 테스트에서'})", ""]
    (OUT / f"summary_{sp}.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / f"claims_{sp}.json").write_text(json.dumps(dict(default=default, rq6=rq6, claims=claims), indent=1,
                                                      ensure_ascii=False, default=float), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))
