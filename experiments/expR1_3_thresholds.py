"""분석 R1-3 2절: 임계값 보정과 오경보율 (doc/분석R1-3_계획.md, 사전 등록 2026-10-28 21:32). 서술용(판정 없음).

깨끗한 D1·D2 (exp1_baselines.prep 경로, 원고 표 4 와 같은 주입), 참조 PS(K = T) 와 CG(셀 평균).
보정 방식: (a) 현행 — 검증 칸 점수 97.5 백분위 하나를 칸·그룹(앵커 최댓값)에 적용 / (b) 행별 — 행마다 검증 칸 97.5 백분위,
그룹 점수 = max_행 (A/thr_행) 를 1 과 비교 / (c) 그룹 — 검증 그룹 점수의 97.5 백분위.
오경보율 = 라벨 0 인 그룹(앵커·시점) 중 임계를 넘은 비율. 검증은 이상을 넣지 않은 검증 창, 테스트는 주입 세트의 라벨 0 그룹.
재현율 = 라벨 1 그룹 중 넘은 비율.
출력: results/expR1/thresholds/far.csv, summary.md
"""
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

from exp1_baselines import prep  # noqa: E402
from src.data import d2 as D2  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.model.bank import NormalSurfaceBank, anchor_scores  # noqa: E402

OUT = ROOT / "results" / "expR1" / "thresholds"
PCT = 97.5


def rates(Aa_by_rule, lab):
    """Aa_by_rule: {규칙: 그룹 초과 여부 (N, A, T) bool}. lab: (N, A, T) 0/1/NaN."""
    out = {}
    for k, ex in Aa_by_rule.items():
        neg, pos = lab == 0, lab == 1
        out[f"far_{k}"] = float(ex[neg].mean()) if neg.any() else np.nan
        out[f"rec_{k}"] = float(ex[pos].mean()) if pos.any() else np.nan
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    mall = D2.load_meta(dcfg); mall = mall[mall.ch == dcfg["d2"]["channel"]]
    rows = []
    for ds in ("d1", "d2"):
        E = ecfg["datasets"][ds]
        units = list(E["spaces"].items()) if ds == "d1" else [(f"env{e}", None) for e in E["envs"]]
        for unit, sp in units:
            D = prep(ds, unit, sp, ecfg, dcfg, ["test"], mall)
            T = D["tr"]["Xs"].shape[2]
            aor = D["comp"].anchor_of_row
            for ref in ("PS", "CG"):
                kw = dict(time_basis="pspline") if ref == "PS" else dict(time_basis="cell")
                bank = NormalSurfaceBank(T, T if ref == "PS" else 8, {}, **kw).fit(D["tr"]["Xs"], D["tr"]["Ws"], D["tr"]["n"])
                bank.calibrate(D["va"]["Xs"], D["va"]["Ws"], D["va"]["n"], PCT, stats="val")
                Ava = bank.score(D["va"]["Xs"], D["va"]["n"], D["va"]["Ws"])
                Ava = np.where(D["va"]["Ws"] > 0, Ava, np.nan)
                thr_a = bank.thr
                flat = np.moveaxis(Ava, 1, 0).reshape(Ava.shape[1], -1)
                thr_b = np.nanpercentile(flat, PCT, axis=1)                                 # 행별
                Ga = anchor_scores(Ava, aor, D["anchors"])
                thr_c = float(np.nanpercentile(Ga, PCT))                                   # 그룹
                Gb_va = anchor_scores(Ava / thr_b[None, :, None], aor, D["anchors"])
                va_rates = {"a": float(np.nanmean(Ga[np.isfinite(Ga)] > thr_a)),
                            "b": float(np.nanmean(Gb_va[np.isfinite(Gb_va)] > 1.0)),
                            "c": float(np.nanmean(Ga[np.isfinite(Ga)] > thr_c)),
                            "cell": float(np.nanmean(Ava[np.isfinite(Ava)] > thr_a))}
                rows.append(dict(dataset=ds, space=unit, ref=ref, split="val", scenario="clean", seed=-1,
                                 **{f"far_{k}": v for k, v in va_rates.items()}))
                Ev = D["evals"]["test"]
                for scn, s, Xi, lab in Ev["sets"]:
                    if ds == "d2" and not str(scn).endswith("m"):
                        continue
                    Xs, Ws = D["comp"].transform(Xi, Ev["W"], D["rows"])
                    A = np.where(Ws > 0, bank.score(Xs, Ev["n"], Ws), np.nan)
                    G = anchor_scores(A, aor, D["anchors"])
                    Gb = anchor_scores(A / thr_b[None, :, None], aor, D["anchors"])
                    fin = np.isfinite(G)
                    ex = {"a": np.where(fin, G > thr_a, False), "b": np.where(fin, Gb > 1.0, False), "c": np.where(fin, G > thr_c, False)}
                    lab_f = np.where(fin, lab, np.nan)
                    r = rates(ex, lab_f)
                    # 칸 수준(행): 라벨 0 앵커의 행 칸
                    lab_rows = np.stack([lab_f[:, D["anchors"].index(a)] for a in aor], axis=1)
                    cellneg = (lab_rows == 0) & np.isfinite(A)
                    r["far_cell"] = float((A[cellneg] > thr_a).mean()) if cellneg.any() else np.nan
                    rows.append(dict(dataset=ds, space=unit, ref=ref, split="test", scenario=scn, seed=s, **r))
            print(f"  {ds} {unit} done", flush=True)
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "far.csv", index=False, encoding="utf-8-sig")
    L = ["# 임계값 보정과 실현 오경보율 (분석 R1-3 2절, 서술)", "",
         f"명목 {100 - PCT:.1f} %. (a) 칸 임계를 그룹에 그대로 · (b) 행별 임계 · (c) 그룹 점수 임계. 값은 단위·시나리오·seed 평균.", ""]
    g = F.groupby(["dataset", "ref", "split"])[[c for c in F.columns if c.startswith(("far_", "rec_"))]].mean()
    cols = [c for c in ["far_cell", "far_a", "far_b", "far_c", "rec_a", "rec_b", "rec_c"] if c in g.columns]
    L += ["| 데이터 | 참조 | 분할 | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for (ds, ref, sp), r in g.iterrows():
        L.append(f"| {ds} | {ref} | {sp} | " + " | ".join("—" if not np.isfinite(r[c]) else f"{r[c]:.4f}" for c in cols) + " |")
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
