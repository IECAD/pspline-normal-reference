"""실험 6 1부 추가 5: 결측 조건에서의 비교 모델 (계획: doc/실험6_계획.md "1부 추가 5", 사전 등록 2026-10-12).
깨끗한 D1에서 PS와 대등했던 재구현 9개를 Table 4 선택 구성 그대로, 실험 6과 같은 학습 마스크(iid·버스트·실제 손실)로 학습해 PS·셀 평균과 짝지어 비교한다.
사용법:
  python experiments/exp6_comparators_missing.py --splits val            # 검증 (판정)
  python experiments/exp6_comparators_missing.py --splits test           # 테스트 (한 번)
  옵션: --datasets d1 d2, --models OLS RRR ..., --quick (연기 테스트: 공간 1, 모델 1, 조건 2)
출력: results/exp6/comparators_missing/metrics_<ds>_<split>.csv  (비교 모델 행 + 같은 마스크의 PS·cell 행)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from exp1_baselines import prep, run_model  # noqa: E402
from exp6_conditions import apply_cond, run as run_bases  # noqa: E402
from exp6_missing_patterns import apply_pattern  # noqa: E402
from exp6_phy_masks import apply_phy  # noqa: E402
from src.data import d2 as D2  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.model.bank import NormalSurfaceBank  # noqa: E402
from src.model.compose import Composer  # noqa: E402

OUT = ROOT / "results" / "exp6" / "comparators_missing"
RB = ROOT / "results" / "exp1_baselines"
MODELS9 = ["OLS", "RRR", "TRANS", "RBCPHI", "GDN", "OMNI", "LAPRPCA", "CAE2D", "TRANAD"]
CONDS_D1 = [("base", 1.0, None, 12, None)] + [("C", 1.0, s, 12, r) for r in (0.5, 0.7) for s in (0, 1, 2)] + \
           [("D", 1.0, s, 12, (0.7, 4)) for s in (0, 1, 2)] + [("P", 1.0, s, 12, "P3") for s in (0, 1, 2)]
CONDS_D2 = [("base", 1.0, None, 12, None)] + [("C", 1.0, s, 12, 0.7) for s in (0, 1, 2)]


def apply(Xtr, Wtr, nodes, rows, c):
    """실험 6 과 같은 생성기·시드: C = exp6_conditions.apply_cond, D = exp6_missing_patterns, P = exp6_phy_masks."""
    fam = c[0]
    if fam == "C":
        return apply_cond(Xtr, Wtr, nodes, rows, "C", 1.0, c[2], c[4])
    if fam == "D":
        return apply_pattern(Xtr, Wtr, nodes, rows, c)
    if fam == "P":
        return apply_phy(Xtr, Wtr, nodes, rows, c)
    return Xtr, Wtr, nodes


def level_of(c):
    fam, _, _, _, prm = c
    return {"base": "기준", "C": prm, "D": f"pi={prm[0]},L={prm[1]}" if fam == "D" else None, "P": prm}[fam]


def selected(mdl, ds):
    """Table 4 의 선택 구성 (selection.json 또는 수정판 폴더의 유일 구성)."""
    for sub in ("", "batch2/", "batch3/", "batch2_fix2/"):
        p = RB / f"{sub}selection.json"
        if p.exists():
            sel = json.loads(p.read_text(encoding="utf-8"))["selection"].get(mdl, {}).get(ds)
            if sel:
                return sel["cfg"], sel["variant"]
    for sub in ("batch2_fix/",):
        m = pd.read_csv(RB / f"{sub}metrics_test_{ds}.csv"); m = m[m.model == mdl]
        if len(m):
            return json.loads(m.cfg.iloc[0]), m.variant.iloc[0]
    raise KeyError(mdl)


def masked_copy(D, ds, c, ecfg):
    """조건 마스크를 학습 데이터에 적용한 D 사본: 합성(Composer)·셀 평균 은행을 마스크된 학습으로 다시 적합."""
    code = ecfg["ids"]["F01"]; V = code.get("V_d1", code["V"]) if ds == "d1" else code["V"]
    tr, va = D["tr"]["w"], D["va"]["w"]; rows = D["rows"]
    Wraw = np.isfinite(tr["X"]).astype(float) if ds == "d1" else tr["cells"]["mask"]
    _, Wm, _ = apply(tr["X"], Wraw, D["tr"]["n"], rows, c)
    comp = Composer(ds, V, code["S"], code["O"], code["T"]).fit(tr["X"], Wm, rows, D["tr"]["n"])
    Xs_tr, Ws_tr = comp.transform(tr["X"], Wm, rows)
    Wva = np.isfinite(va["X"]).astype(float) if ds == "d1" else va["cells"]["mask"]
    Xs_va, Ws_va = comp.transform(va["X"], Wva, rows)
    bank = NormalSurfaceBank(Xs_tr.shape[2], 8, {}, time_basis="cell").fit(Xs_tr, Ws_tr, D["tr"]["n"])
    D2_ = dict(D); D2_["comp"], D2_["bank"] = comp, bank
    D2_["tr"] = dict(X=tr["X"], Xs=Xs_tr, Ws=Ws_tr, n=D["tr"]["n"], w=tr)
    D2_["va"] = dict(X=va["X"], Xs=Xs_va, Ws=Ws_va, n=D["va"]["n"], w=va)
    return D2_, float(1 - Wm.mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--datasets", nargs="*", default=["d1", "d2"])
    ap.add_argument("--models", nargs="*", default=MODELS9)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force-variant", default=None, help="추가 5b: 입력 변형 강제 (raw)")
    ap.add_argument("--conds", default="full", choices=["full", "5b"])
    args = ap.parse_args()
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    if args.force_variant:
        OUT = OUT / args.force_variant
    OUT.mkdir(parents=True, exist_ok=True)
    mall = D2.load_meta(dcfg); mall = mall[mall.ch == dcfg["d2"]["channel"]]
    for ds in args.datasets:
        conds = CONDS_D1 if ds == "d1" else CONDS_D2
        if args.conds == "5b":
            conds = [c for c in CONDS_D1 if c[0] == "base" or (c[0] == "C" and c[4] == 0.7) or c[0] == "P"]
        E = ecfg["datasets"][ds]
        units = list(E["spaces"].items()) if ds == "d1" else [(f"env{e}", None) for e in E["envs"]]
        models = args.models
        if args.quick:
            units, conds, models = units[:1], [conds[0], conds[1]], models[:1]
        for split in args.splits:
            recs, t_all = [], time.perf_counter()
            for unit, sp in units:
                D = prep(ds, unit, sp, ecfg, dcfg, [split], mall)
                for c in conds:
                    Dm, miss = masked_copy(D, ds, c, ecfg)
                    for m in models:
                        cfg, variant = selected(m, ds)
                        if args.force_variant:
                            variant = args.force_variant
                        t0 = time.perf_counter()
                        for r in run_model(Dm, m, cfg, variant, split):
                            recs.append(dict(dataset=ds, space=unit, split=split, family=c[0], level=level_of(c),
                                             cond_seed=-1 if c[2] is None else c[2], missing_rate=miss, model=m, variant=variant,
                                             cfg=json.dumps(cfg, sort_keys=True), **r))
                        print(f"  {ds} {unit} {c[0]} {level_of(c)} s={c[2]} {m}: {time.perf_counter() - t0:.1f}s", flush=True)
            # 같은 마스크의 PS·셀 평균 (실험 6 코드 경로)
            base_rows = run_bases(ds, ecfg, dcfg, [split], conds=conds, apply=apply, level_of=level_of,
                                  bases={"cell": {"time_basis": "cell"}, "PS": {"K_t": "T"}}) if not args.quick else []
            for r in base_rows:
                r = dict(r); r["model"] = r.pop("basis"); r["variant"] = "-"; r["cfg"] = "{}"; recs.append(r)
            M = pd.DataFrame(recs)
            suf = f"{ds}_{split}" + ("_quick" if args.quick else "")
            M.to_csv(OUT / f"metrics_{suf}.csv", index=False, encoding="utf-8-sig")
            print(f"{ds} {split}: {len(M)} rows, {time.perf_counter() - t_all:.0f}s", flush=True)
