"""실험 R1-4 1부: 참조 × 탐지기 요인 실험 (doc/실험R1-4_계획.md, 사전 등록 2026-10-28 21:32, 지문 doc/R1_등록_지문.md).

실험 6 비교 모델 실험(exp6_comparators_missing.py)과 같은 데이터·마스크·주입·보정 경로에서,
탐지기 입력의 참조를 셀 평균(CG, 현행) · CF(노드 행 평균 채움) · CI(보간) · PS 로 바꾸고,
결측 조건마다 검증으로 설정(하이퍼파라미터 × 입력 {R 잔차, raw})을 다시 고른다.

사용법:
  python experiments/expR1_4_factorial.py --splits val     # 검증: 모든 설정 실행 → selection.json
  python experiments/expR1_4_factorial.py --splits test    # 테스트: 선택 설정만, 한 번
  옵션: --datasets d1 d2, --quick (공간 1, 조건 2, 탐지기 OLS)
출력: results/expR1/factorial/metrics_<ds>_<split>.csv, selection.json, self_<ds>_<split>.csv
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from exp1_baselines import prep, run_model  # noqa: E402
from exp6_comparators_missing import CONDS_D1, CONDS_D2, apply, level_of  # noqa: E402
from exp6_conditions import run as run_bases  # noqa: E402
from src.baselines.models import GRIDS, GRIDS2, GRIDS3, GRIDS5  # noqa: E402
from src.data import d2 as D2  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.model.bank import NormalSurfaceBank  # noqa: E402
from src.model.compose import Composer  # noqa: E402

OUT = ROOT / "results" / "expR1" / "factorial"
REFS = {"CG": dict(time_basis="cell"), "CF": dict(time_basis="cellfill"),
        "CI": dict(time_basis="cellinterp"), "PS": dict(time_basis="pspline", K_t="T")}
DETS = {"OLS": GRIDS["OLS"], "GDN": GRIDS3["GDN"], "CAE2D": GRIDS2["CAE2D"], "TRANAD": GRIDS5["TRANAD"]}
VARIANTS = ("nc", "raw")


def masked_unit(D, ds, c, ecfg):
    """조건 마스크를 학습에 적용하고 Composer 를 다시 적합(실험 6 과 같음). 참조 뱅크는 따로 만든다."""
    code = ecfg["ids"]["F01"]; V = code.get("V_d1", code["V"]) if ds == "d1" else code["V"]
    tr, va = D["tr"]["w"], D["va"]["w"]; rows = D["rows"]
    Wraw = np.isfinite(tr["X"]).astype(float) if ds == "d1" else tr["cells"]["mask"]
    _, Wm, _ = apply(tr["X"], Wraw, D["tr"]["n"], rows, c)
    comp = Composer(ds, V, code["S"], code["O"], code["T"]).fit(tr["X"], Wm, rows, D["tr"]["n"])
    Xs_tr, Ws_tr = comp.transform(tr["X"], Wm, rows)
    Wva = np.isfinite(va["X"]).astype(float) if ds == "d1" else va["cells"]["mask"]
    Xs_va, Ws_va = comp.transform(va["X"], Wva, rows)
    Dm = dict(D); Dm["comp"] = comp
    Dm["tr"] = dict(X=tr["X"], Xs=Xs_tr, Ws=Ws_tr, n=D["tr"]["n"], w=tr)
    Dm["va"] = dict(X=va["X"], Xs=Xs_va, Ws=Ws_va, n=D["va"]["n"], w=va)
    return Dm, float(1 - Wm.mean())


def ref_bank(Dm, ref):
    kw = dict(REFS[ref])
    T = Dm["tr"]["Xs"].shape[2]
    K = T if kw.pop("K_t", None) == "T" else 8
    return NormalSurfaceBank(T, K, {}, **kw).fit(Dm["tr"]["Xs"], Dm["tr"]["Ws"], Dm["tr"]["n"])


def cond_key(c):
    return f"{c[0]}|{level_of(c)}"


def select(val_csv_paths):
    """(데이터, 조건 계열·수준, R, 탐지기) 마다 검증 셀 AUC-PR 평균 최대 설정. 동점이면 작은 모델(격자 앞쪽)."""
    M = pd.concat([pd.read_csv(p) for p in val_csv_paths], ignore_index=True)
    M["ck"] = M.family.astype(str) + "|" + M.level.astype(str)
    sel = {}
    for (ds, ck, ref, det), g in M.groupby(["dataset", "ck", "ref", "detector"]):
        s = g.groupby(["cfg", "variant", "grid_pos"]).auc_pr.mean().reset_index()
        s = s.sort_values(["auc_pr", "grid_pos"], ascending=[False, True])
        best = s.iloc[0]
        sel.setdefault(ds, {}).setdefault(ck, {}).setdefault(ref, {})[det] = dict(
            cfg=json.loads(best.cfg), variant=best.variant, val_auc_pr=float(best.auc_pr),
            n_cfg=int(len(s)))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--datasets", nargs="*", default=["d1", "d2"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--units", nargs="*", default=None, help="이 단위만 실행(병렬 실행용). 출력 파일에 단위 이름을 붙이고, self 와 selection 은 만들지 않음")
    ap.add_argument("--self-only", action="store_true", help="참조 자체 점수(self)만 계산")
    ap.add_argument("--select-only", action="store_true", help="검증 결과 파일로 selection.json 만 만듦")
    args = ap.parse_args()
    if args.select_only:
        paths = sorted(OUT.glob("metrics_*_val.csv"))
        (OUT / "selection.json").write_text(json.dumps(select(paths), ensure_ascii=False, indent=1), encoding="utf-8")
        print("selection from", [p.name for p in paths]); return
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    mall = D2.load_meta(dcfg); mall = mall[mall.ch == dcfg["d2"]["channel"]]
    sel = None
    if "test" in args.splits:
        sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    for ds in args.datasets:
        conds = CONDS_D1 if ds == "d1" else CONDS_D2
        E = ecfg["datasets"][ds]
        units = list(E["spaces"].items()) if ds == "d1" else [(f"env{e}", None) for e in E["envs"]]
        dets = dict(DETS)
        if args.quick:
            units, conds, dets = units[:1], conds[:2], {"OLS": DETS["OLS"][:1]}
        if args.units:
            units = [u for u in units if u[0] in args.units]
            if not units:
                continue
        for split in args.splits:
            if args.self_only:
                bases = {"CG": {"time_basis": "cell"}, "CF": {"time_basis": "cellfill"},
                         "CI": {"time_basis": "cellinterp"}, "PS": {"K_t": "T"}}
                base_rows = run_bases(ds, ecfg, dcfg, [split], conds=conds, apply=apply, level_of=level_of, bases=bases)
                pd.DataFrame(base_rows).to_csv(OUT / f"self_{ds}_{split}.csv", index=False, encoding="utf-8-sig")
                print(f"self {ds} {split}: {len(base_rows)} rows", flush=True)
                continue
            recs, t_all = [], time.perf_counter()
            usuf = ("_" + "-".join(u[0] for u in units)) if args.units else ""
            for unit, sp in units:
                D = prep(ds, unit, sp, ecfg, dcfg, [split], mall)
                for c in conds:
                    Dm, miss = masked_unit(D, ds, c, ecfg)
                    banks = {r: ref_bank(Dm, r) for r in REFS}
                    for det, grid in dets.items():
                        for gp, cfg in enumerate(grid):
                            for variant in VARIANTS:
                                refs = list(REFS) if variant == "nc" else ["(raw)"]
                                if split == "test":            # 선택된 (R, 탐지기) 의 설정만
                                    keep = [r for r in REFS
                                            if sel[ds][cond_key(c)][r][det]["cfg"] == cfg
                                            and sel[ds][cond_key(c)][r][det]["variant"] == variant]
                                    if not keep:
                                        continue
                                    refs = keep if variant == "nc" else ["(raw)"]
                                    raw_for = keep
                                for r in refs:
                                    Dr = dict(Dm); Dr["bank"] = banks[r] if r != "(raw)" else banks["CG"]
                                    t0 = time.perf_counter()
                                    res = run_model(Dr, det, cfg, variant, split)
                                    # raw 입력은 참조와 무관 → 검증에서는 모든 R 에, 테스트에서는 그 설정을 고른 R 에 같은 행을 붙인다
                                    targets = ([r] if r != "(raw)" else (list(REFS) if split == "val" else raw_for))
                                    for tr_ in targets:
                                        for row in res:
                                            recs.append(dict(dataset=ds, space=unit, split=split, family=c[0],
                                                             level=level_of(c), cond_seed=-1 if c[2] is None else c[2],
                                                             missing_rate=miss, ref=tr_, detector=det, variant=variant,
                                                             cfg=json.dumps(cfg, sort_keys=True), grid_pos=gp, **row))
                                    print(f"  {ds} {unit} {cond_key(c)} s={c[2]} {r} {det} {cfg} {variant}: "
                                          f"{time.perf_counter() - t0:.1f}s", flush=True)
                # 단위마다 중간 저장 (실행이 끊겨도 끝난 단위는 남음)
                pd.DataFrame(recs).to_csv(OUT / f"metrics_{ds}{usuf}_{split}.partial.csv", index=False, encoding="utf-8-sig")
            M = pd.DataFrame(recs)
            suf = f"{ds}{usuf}_{split}" + ("_quick" if args.quick else "")
            M.to_csv(OUT / f"metrics_{suf}.csv", index=False, encoding="utf-8-sig")
            # 참조 자체의 점수(self) — 실험 6 코드 경로, 같은 마스크 (단위 지정 실행에서는 --self-only 로 따로)
            if not args.quick and not args.units:
                bases = {"CG": {"time_basis": "cell"}, "CF": {"time_basis": "cellfill"},
                         "CI": {"time_basis": "cellinterp"}, "PS": {"K_t": "T"}}
                base_rows = run_bases(ds, ecfg, dcfg, [split], conds=conds, apply=apply, level_of=level_of, bases=bases)
                pd.DataFrame(base_rows).to_csv(OUT / f"self_{suf}.csv", index=False, encoding="utf-8-sig")
            print(f"{ds} {split}: {len(M)} rows, {time.perf_counter() - t_all:.0f}s", flush=True)
        if "val" in args.splits and not args.quick and not args.units and not args.self_only:
            paths = sorted(OUT.glob("metrics_*_val.csv"))
            (OUT / "selection.json").write_text(json.dumps(select(paths), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
