"""실험 4 선택 규칙 (doc/실험4_계획.md 2–3절을 그대로 구현). 검증 세트만 쓴다.

1단계 (기본): 축별 선택 -> M01, M02 사양을 results/exp4/selection.json 에 쓴다.
2단계 (--final): M01 검증 결과로 상호작용 확인 -> 최종 모델을 selection.json 의 final 에 쓴다.
규칙
  - 비교 = 같은 공간·시나리오·시드끼리 짝지은 AUC-PR 차이의 평균 Δ (D1 주입, D2 다지표 주입)
  - o 가 b 를 '이긴다' = (Δ_D1 > tol_D1 또는 Δ_D2 > tol_D2) 그리고 Δ_D1 >= −tol_D1 그리고 Δ_D2 >= −tol_D2
  - 축: 이기는 선택지가 없으면 현재값. 여럿이면 평균 Δ 최대 -> 평균 Δ 가 tie_mean 이내면 HitRate -> Smearing -> 단순함
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "exp4"


def load(ids):
    parts = []
    for i in ids:
        for ds in ("d1", "d2"):
            p = RES / i / "base" / f"metrics_{ds}.csv"
            if p.exists():
                parts.append(pd.read_csv(p))
    M = pd.concat(parts, ignore_index=True)
    return M[M.split == "val"]


class Judge:
    def __init__(self, M, J):
        self.J = J
        self.g = {}
        for ds, rx in J["views"].items():
            d = M[(M.dataset == ds) & M.scenario.astype(str).str.match(rx)]
            self.g[ds] = d.groupby(["id", "space", "scenario", "seed"])[["auc_pr", "hitrate", "smearing"]].mean()

    def delta(self, a, b, ds, col="auc_pr"):
        g = self.g[ds]
        return float((g.loc[a, col] - g.loc[b, col]).dropna().mean())

    def level(self, a, col):
        return float(np.mean([self.g[ds].loc[a, col].mean() for ds in self.g]))

    def row(self, a, b):
        d = {ds: self.delta(a, b, ds) for ds in self.g}
        tol = self.J["tol"]
        beats = any(d[ds] > tol[ds] for ds in d) and all(d[ds] >= -tol[ds] for ds in d)
        noninf = all(d[ds] >= -tol[ds] for ds in d)
        return dict(id=a, **{f"delta_{ds}": v for ds, v in d.items()}, delta_mean=float(np.mean(list(d.values()))),
                    beats=beats, noninferior=noninf, auc=self.level(a, "auc_pr"), hit=self.level(a, "hitrate"),
                    smear=self.level(a, "smearing"))

    def chain(self, rows, order):
        """이기는 후보들 중 평균 Δ -> HitRate -> Smearing -> 단순함."""
        R = pd.DataFrame(rows)
        R = R[R.delta_mean >= R.delta_mean.max() - self.J["tie_mean"]]
        R = R[R.hit >= R.hit.max() - self.J["tol_hit"]]
        R = R[R.smear <= R.smear.min() + self.J["tol_smear"]]
        return sorted(R.id, key=order.index)[0]


def select_axes(J, cfg, judge):
    out = {}
    for axis, a in cfg["axes"].items():
        cur, opts = a["current"], a["options"]
        rows = [judge.row(o, cur) for o in opts]
        win = [r for r in rows if r["beats"]]
        best = judge.chain(win, opts) if win else cur
        rest = [r for r in win if r["id"] != best]
        if rest:
            second = judge.chain(rest, opts)
        else:
            ni = [r for r in rows if r["noninferior"] and r["id"] != best]
            second = max(ni, key=lambda r: r["delta_mean"])["id"] if ni else cur
        out[axis] = dict(current=cur, best=best, second=second, candidates=rows)
    return out


def combine(cfg, axes, key):
    spec, used = {}, {}
    basis = axes["basis"][key]
    for axis, r in axes.items():
        cid = r[key]
        if axis in ("diff", "lam") and basis in cfg["moot_if_basis"]:
            continue                                          # 시간 penalty 가 없는 기저: 차분·λ 축 무관
        if cid != r["current"]:
            spec.update(cfg["variants"][cid])
            used[axis] = cid
    return spec, used


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/exp4.yaml").read_text(encoding="utf-8"))
    J = cfg["judge"]
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    if not args.final:
        judge = Judge(load(list(cfg["variants"])), J)
        axes = select_axes(J, cfg, judge)
        sel = dict(rule="doc/실험4_계획.md 2–3절", judge=J, axes=axes)
        for key, name in (("best", "M01"), ("second", "M02")):
            spec, used = combine(cfg, axes, key)
            sel[name] = dict(spec=spec, from_axes=used)
        (OUT / "selection.json").write_text(json.dumps(sel, indent=1, ensure_ascii=False), encoding="utf-8")
        for axis, r in axes.items():
            print(f"[{axis}] 현재 {r['current']} -> 선택 {r['best']}, 차선 {r['second']}")
            for c in r["candidates"]:
                print(f"    {c['id']}: ΔD1 {c['delta_d1']:+.4f} ΔD2 {c['delta_d2']:+.4f} 평균 {c['delta_mean']:+.4f} "
                      f"이김 {c['beats']} 비열등 {c['noninferior']} | AUC {c['auc']:.4f} Hit {c['hit']:.4f} "
                      f"Smear {c['smear']:.4f}")
        print("M01", sel["M01"], "\nM02", sel["M02"])
    else:
        sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
        singles = sorted({v for v in sel["M01"]["from_axes"].values()})
        judge = Judge(load(["C00", "M01", "M02"] + singles), J)
        m01 = judge.row("M01", "C00")
        checks = [judge.row(c, "M01") for c in singles]
        beat = [c for c in checks if c["beats"]]
        final = max(beat, key=lambda c: c["auc"])["id"] if beat else "M01"
        if not sel["M01"]["from_axes"]:
            final = "C00"                                     # 바뀐 축이 없으면 F01 유지
        spec = sel["M01"]["spec"] if final == "M01" else (cfg["variants"][final] if final != "C00" else {})
        sel["final"] = dict(id=final, spec=spec, m01_vs_f01=m01, singles_vs_m01=checks,
                            m02_vs_f01=judge.row("M02", "C00"))
        (OUT / "selection.json").write_text(json.dumps(sel, indent=1, ensure_ascii=False), encoding="utf-8")
        print("M01 vs F01:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m01.items()})
        for c in checks:
            print(f"  단독 {c['id']} vs M01: ΔD1 {c['delta_d1']:+.4f} ΔD2 {c['delta_d2']:+.4f} 이김 {c['beats']}")
        print("최종:", final, spec)
