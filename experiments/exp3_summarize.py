"""실험 3 결과 집계 + F01/F02 선택 (doc/합성방식_실험계획_최종.md 6절, 8절).

입력: results/<ID>/<run>/metrics_*.csv
출력: results/exp3/table_<dataset>_<split>.csv  (계획 8.1 표: ID, 축, 설정, AUC-PR, ΔAUC-PR, HitRate, ΔHitRate, Smearing ...)
      results/exp3/selection.json            (축별 선택, 동률 판정 근거, F01/F02 구성)
선택 규칙 (결과를 보기 전에 고정된 규칙을 그대로 구현):
  1. 검증 AUC-PR (primary_dataset = D1: 공간 3 × 시나리오 3 × 시드 5 평균)
  2. 동률: |차이| <= tie_tol. tie_tol = max(B0 의 시드 간 표준편차, 0.01)
     (계획 문구 'B0 반복 실행 표준편차(또는 0.01)' 를 둘 중 큰 값으로 해석)
  3. 동률이면 HitRate@100% 높은 쪽 -> Smearing 낮은 쪽 -> 단순한 쪽
  4. 테스트 세트는 쓰지 않는다. O3(무작위)와 S0 는 후보에서 제외
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "exp3"
METRICS = ["auc_pr", "vus_pr", "range_f1", "hitrate", "cell_f1", "smearing", "fit_rmse"]
SIMPLE = {"V": ["V1", "V5", "V2", "V3", "V4"], "S": ["S1", "S2", "S3", "S4"], "O": ["O1", "O2", "O4"],
          "T": ["T4", "T2", "T1", "T3"]}
AXIS_OF = {"①": "V", "②": "S", "③": "O", "④": "T"}


def load_all(subset=False):
    """subset=False: 본 실행 (앵커 8개). True: D2 앵커 부분추출 실행 (run 이름에 '_a<k>s<seed>')."""
    parts = [pd.read_csv(p) for p in RES.glob("[BEF]*/*/metrics_*.csv") if p.stat().st_size > 5]
    if not parts:
        return pd.DataFrame()
    M = pd.concat(parts, ignore_index=True)
    is_sub = M["run"].astype(str).str.contains(r"_a\d+s\d+$")
    return M[is_sub] if subset else M[~is_sub]


def per_run(M, dataset, split, scen_prefix=None, exclude_real=True):
    """scen_prefix 가 '^' 로 시작하면 정규식으로 본다 (예: '^SC\\d$' 거리만 주입, '^SC\\dm$' 다지표 주입)."""
    d = M[(M.dataset == dataset) & (M.split == split)]
    if exclude_real:
        d = d[~d.scenario.astype(str).str.startswith("real")]
    if scen_prefix:
        sc = d.scenario.astype(str)
        d = d[sc.str.match(scen_prefix) if scen_prefix.startswith("^") else sc.str.startswith(scen_prefix)]
    return d


def table(M, dataset, split, scen_prefix=None, exclude_real=True):
    d = per_run(M, dataset, split, scen_prefix, exclude_real)
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(["id", "run", "V", "S", "O", "T"])[METRICS].mean().reset_index()
    agg = g.groupby(["id", "V", "S", "O", "T"])[METRICS].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["n_runs"] = g.groupby("id")["run"].nunique().reindex(agg["id"]).to_numpy()
    if "B0" in set(agg.id):
        b = agg.set_index("id").loc["B0"]
        for m in ("auc_pr", "hitrate", "smearing"):
            agg[f"d_{m}"] = agg[f"{m}_mean"] - b[f"{m}_mean"]
    return agg.sort_values("id")


def seed_std(M, id_, dataset, split, metric="auc_pr", scen=None):
    d = per_run(M, dataset, split, scen)
    d = d[d.id == id_]
    if d.empty:
        return np.nan
    return float(d.groupby("seed")[metric].mean().std())


def select(tab, ecfg, tie_tol, hit_tol=None, smear_tol=None, axes=("V", "S", "O", "T")):
    """hit_tol/smear_tol 이 None 이면 엄격 비교(값이 조금이라도 크면 우선), 값이 있으면 그 안의 차이는 동률로 보고
    다음 기준으로 넘어간다 (마지막 = 단순함)."""
    base = ecfg["ids"]["B0"]
    res = {}
    for axis in axes:
        cand = {base[axis]: "B0"}
        for id_, c in ecfg["ids"].items():
            if id_.startswith("E") and int(id_[1:]) <= 14:
                # O 가 '-' (T4) 이면 순서 축은 바뀐 것이 아님
                diff = [k for k in ("V", "S", "O", "T") if c[k] != base[k] and not (k == "O" and c[k] == "-")]
                if diff == [axis] and c[axis] in SIMPLE[axis]:
                    cand[c[axis]] = id_
        rows = []
        for opt, id_ in cand.items():
            r = tab[tab.id == id_]
            if r.empty or not np.isfinite(r["auc_pr_mean"].iloc[0]):
                continue
            rows.append(dict(option=opt, id=id_, auc=r["auc_pr_mean"].iloc[0], hit=r["hitrate_mean"].iloc[0],
                             smear=r["smearing_mean"].iloc[0], simple=SIMPLE[axis].index(opt)))
        if not rows:
            continue
        R = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
        top = R.auc.max()
        tied = R[R.auc >= top - tie_tol].copy()
        if hit_tol is None:
            tied = tied.sort_values(["hit", "smear", "simple"], ascending=[False, True, True])
        else:
            tied = tied[tied.hit >= tied.hit.max() - hit_tol]
            tied = tied[tied.smear <= tied.smear.min() + smear_tol]
            tied = tied.sort_values("simple")
        best = tied.iloc[0]
        rest = R[R.option != best.option]
        second = rest.iloc[0] if len(rest) else best
        res[axis] = dict(best=best.option, best_id=best.id, second=second.option, second_id=second.id,
                         tied=tied.option.tolist(), candidates=R.round(4).to_dict("records"))
    return res


if __name__ == "__main__":
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    M = load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    views = [("d1", "val", None, True, "inject"), ("d1", "test", None, True, "inject"),
             ("d1", "test", "real", False, "real"),
             ("d2", "val", r"^SC\d$", True, "inject_dist"), ("d2", "test", r"^SC\d$", True, "inject_dist"),
             ("d2", "val", r"^SC\dm$", True, "inject_multi"), ("d2", "test", r"^SC\dm$", True, "inject_multi"),
             ("d4_smd", "test", "real", False, "real"), ("d4_asd", "test", "real", False, "real")]
    for ds, sp, pre, ex, tag in views:
        t = table(M, ds, sp, pre, ex)
        if len(t):
            t.to_csv(OUT / f"table_{ds}_{sp}_{tag}.csv", index=False, encoding="utf-8-sig")
    sel_cfg = ecfg["selection"]
    prim = sel_cfg["primary_dataset"]
    src = {a: dict(dataset=prim, scen=None) for a in ("V", "S", "O", "T")}
    for a, s in (sel_cfg.get("axis_source") or {}).items():
        src[a] = dict(dataset=s["dataset"], scen=rf"^SC\d{s.get('scenario_suffix', '')}$")
    out = dict(metric="val AUC-PR", tie_chain=sel_cfg.get("tie_chain", "tolerance"),
               tol_rule="max(B0 의 주입 시드 간 표준편차, 0.01)", axes_source=src, by_rule={})
    for rule in ("tolerance_chain", "strict"):
        sel_all = {}
        for a, s in src.items():
            tv = table(M, s["dataset"], "val", s["scen"])
            tol = lambda m: float(max(np.nan_to_num(seed_std(M, "B0", s["dataset"], "val", m, s["scen"])), 0.01))
            kw = dict(hit_tol=tol("hitrate"), smear_tol=tol("smearing")) if rule == "tolerance_chain" else {}
            r = select(tv, ecfg, tol("auc_pr"), axes=(a,), **kw)
            if a in r:
                r[a]["tolerances"] = dict(auc=tol("auc_pr"), hitrate=tol("hitrate"), smearing=tol("smearing"))
                r[a]["source"] = s
                sel_all.update(r)
        out["by_rule"][rule] = dict(axes=sel_all, F01={a: v["best"] for a, v in sel_all.items()},
                                    F02={a: v["second"] for a, v in sel_all.items()})
    use = "tolerance_chain" if out["tie_chain"] == "tolerance" else "strict"
    out["F01"], out["F02"] = out["by_rule"][use]["F01"], out["by_rule"][use]["F02"]
    (OUT / "selection.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dict(F01=out["F01"], F02=out["F02"],
                          strict={k: out["by_rule"]["strict"][k] for k in ("F01", "F02")}), ensure_ascii=False))
