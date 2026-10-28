"""실험 9: 범용성 — D4 서버 데이터 (doc/실험9_계획.md, 사전 등록 2026-10-11).

엔터티(서버)마다 노드 1개. 합성 S1·O1·T4 (실험 3 D4 처리와 같음), T = 12.
모델: PS (K_t = T, SAP), 셀 평균, 노드 평균 + PCA, OLS, IF, LSTM-AE, USAD, 1D-CAE, GDN, Transformer
      (실험 1 의 D2 선택 구성, 입력 raw 고정. D4 는 검증 라벨이 없음)
평가: 테스트 윈도우를 시간순으로 이어 붙여 시점 AUC-PR (lab_P), 위치화 HitRate·셀 F1·Smearing (lab_I)
사용법: python experiments/exp9_d4.py --datasets smd asd
출력: results/exp9/metrics_<데이터>.csv
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from node_model_eval import _bank, full_spec  # noqa: E402
from src.baselines.models import MODELS  # noqa: E402
from src.data import d4 as D4  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.load import load_windows  # noqa: E402
from src.metrics.anomaly import evaluate  # noqa: E402
from src.model.bank import anchor_scores  # noqa: E402
from src.model.compose import Composer  # noqa: E402

OUT = ROOT / "results" / "exp9"
BANKS = {"PS": {"K_t": 12}, "cell": {"time_basis": "cell"}, "Z1": {"time_basis": "const"}}
COMPARATORS = ["PCA", "OLS", "IF", "LSTMAE", "USADFIX2", "CAE1D", "GDN", "TRANS"]  # USAD → USADFIX2 (2차 수정 추가)


def d2_configs(models):
    sel = {}
    for b in ("", "batch2/", "batch2_fix/", "batch2_fix2/"):         # 뒤가 앞을 덮음 (GDN·LAPRPCA 는 수정판)
        p = ROOT / "results" / "exp1_baselines" / b / "selection.json"
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            sel.update({m: d["d2"]["cfg"] for m, d in s["selection"].items() if "d2" in d})
    return {m: sel[m] for m in models}


def run_entity(ds, ent, ecfg, dcfg, cfgs, banks=True):
    name = f"{ds}_{ent}"
    tr, va, te = (load_windows("d4", name, dcfg, split=s) for s in ("train", "val", "test"))
    rows = tr["rows"]
    W1 = lambda w: np.isfinite(w["X"]).astype(float)
    node = lambda w: np.zeros(len(w["X"]), dtype=int)
    comp = Composer("d4", "V3", "S1", "O1", "T4").fit(tr["X"], W1(tr), rows, node(tr))
    (Xtr, Wtr), (Xva, Wva), (Xte, _) = (comp.transform(w["X"], W1(w), rows) for w in (tr, va, te))
    ix = te["meta"].sort_values("tidx0").index.to_numpy()
    lab = np.concatenate(te["cells"]["lab_I"][ix], 1)
    tl = [np.concatenate(te["cells"]["lab_P"][ix][:, 0], 0)]

    def ev(A, thr):
        seqs = [(np.concatenate(A[ix], 1), lab)]
        return evaluate(seqs, thr, "time", time_labels=tl)

    out = []
    for bid, sp in (BANKS.items() if banks else []):
        spec = full_spec(sp, ecfg)
        t0 = time.perf_counter()
        bank = _bank(spec, ecfg, Xtr.shape[2], spec["K_t"], {}).fit(Xtr, Wtr, node(tr))
        bank.calibrate(Xva, Wva, node(va), ecfg["model"]["threshold_pct"])
        A = anchor_scores(bank.score(Xte, node(te)), comp.anchor_of_row, rows)
        out.append(dict(dataset=ds, entity=ent, model=bid, train_s=time.perf_counter() - t0, **ev(A, bank.thr)))
    for m, cfg in cfgs.items():
        t0 = time.perf_counter()
        mod = MODELS[m](**cfg).fit(np.where(Wtr > 0, np.nan_to_num(Xtr), 0.0))
        e = np.where(Wva > 0, mod.errors(np.nan_to_num(Xva)), np.nan)
        sd = np.nanstd(np.moveaxis(e, 1, 0).reshape(e.shape[1], -1), 1)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        thr = float(np.nanpercentile(e / sd[None, :, None], 97.5))
        A = anchor_scores(mod.errors(np.nan_to_num(Xte)) / sd[None, :, None], comp.anchor_of_row, rows)
        out.append(dict(dataset=ds, entity=ent, model=m, cfg=json.dumps(cfg), train_s=time.perf_counter() - t0,
                        **ev(A, thr)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["smd", "asd"])
    ap.add_argument("--models", nargs="*", default=COMPARATORS, help="비교 모델 부분집합 (나눠 돌릴 때)")
    ap.add_argument("--no-banks", action="store_true", help="PS·셀 평균·노드 평균을 건너뜀")
    ap.add_argument("--tag", default="", help="출력 파일 꼬리표 (나눠 돌린 결과는 집계에서 합침)")
    args = ap.parse_args()
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    cfgs = d2_configs(args.models)
    OUT.mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""
    for ds in args.datasets:
        rows = []
        for ent in D4.entities(ds):
            t0 = time.perf_counter()
            rows += run_entity(ds, ent, ecfg, dcfg, cfgs, banks=not args.no_banks)
            pd.DataFrame(rows).to_csv(OUT / f"metrics_{ds}{sfx}.csv", index=False, encoding="utf-8-sig")
            print(f"  {ds} {ent}: {time.perf_counter() - t0:.1f}s", flush=True)
        (OUT / f"config_{ds}{sfx}.json").write_text(json.dumps(dict(plan="doc/실험9_계획.md", configs=cfgs,
                                                                    banks=None if args.no_banks else BANKS),
                                                               indent=1, ensure_ascii=False), encoding="utf-8")
