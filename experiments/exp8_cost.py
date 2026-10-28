"""실험 8: 계산 비용 (doc/실험8_계획.md, 사전 등록 2026-10-11). PC 기준.

대상: PS, 셀 평균(M01), 노드 평균(Z1), 실험 1 비교 모델 1·2차의 데이터별 선택 구성 (IF 포함, UWB 규칙·ST-SSD 제외)
단위: D1 강의실, D2 env1 (대표). PS·셀 평균의 노드 수·파라미터 수는 모든 단위.
지표: 파라미터 수 (nc 기준 포함/제외), 메모리 (float32 KB), 학습 시간 (CPU 단일 스레드; 딥러닝은 GPU 도),
      추론 시간 (배치 1 중앙값, 배치 전체 윈도우당; CPU 단일 스레드, 딥러닝은 GPU 도)
사용법: python experiments/exp8_cost.py
출력: results/exp8/cost.csv, params_all_units.csv, hardware.json
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import torch
import yaml

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from exp1_baselines import model_input, prep, units  # noqa: E402
from node_model_eval import _bank, full_spec  # noqa: E402
from src.baselines.models import MODELS, _DeepAE  # noqa: E402
from src.data import d2 as D2  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.model.bank import anchor_scores  # noqa: E402

OUT = ROOT / "results" / "exp8"
REP = {"d1": "classroom", "d2": "env1"}
BANKS = {"PS": {"K_t": 12}, "cell": {"time_basis": "cell"}, "Z1": {"time_basis": "const"}}
N_INFER = 200


def med_ms(fn, idx):
    ts = []
    for i in idx:
        t0 = time.perf_counter()
        fn(i)
        ts.append(1000 * (time.perf_counter() - t0))
    return float(np.median(ts))


def bank_params(bank):
    S = len(bank.mu)
    n = sum(m.size for m in bank.M.values())
    lam = len(bank.M) if bank.time_basis == "pspline" else 0
    if bank.time_basis == "const":                       # 노드×행 평균: 시간 방향 값이 같아 행 수만 저장
        n = sum(m.shape[0] for m in bank.M.values())
    return n + lam + 2 * S + 1


def model_params(name, m, S):
    if isinstance(m, _DeepAE):                                       # USADFIX2 는 입력 정규화 상수 (행별 최소·범위) 포함
        return sum(p.numel() for p in m.net.parameters()) + (2 * S if hasattr(m, "lo") else 0)
    if name == "RBCPHI":
        return m.Phi.size + m.cm.size + m.cs.size
    if name in ("PCA", "RPCA", "LAPRPCA"):                           # LapRPCA 수정판은 열 표준화 상수 포함
        return m.mu.size + m.P.size + (2 * S if getattr(m, "std", False) else 0)
    if name == "OLS":
        return sum(b.size for _, b in m.coef)
    if name == "RRR":
        return m.B.size + 3 * S
    if name == "IF":
        return int(sum(2 * e.tree_.node_count for e in m.m.estimators_))
    raise ValueError(name)


def run_bank(D, bid, ecfg, dev_label="cpu"):
    tr, va = D["tr"], D["va"]
    spec = full_spec(BANKS[bid], ecfg)
    T = tr["Xs"].shape[2]
    t0 = time.perf_counter()
    bank = _bank(spec, ecfg, T, spec["K_t"], D["comp"].sensor_penalties()).fit(tr["Xs"], tr["Ws"], tr["n"])
    bank.calibrate(va["Xs"], va["Ws"], va["n"], ecfg["model"]["threshold_pct"])
    train_s = time.perf_counter() - t0
    E = D["evals"]["test"]
    Xs, Ws = D["comp"].transform(E["w"]["X"], E["W"], D["rows"])
    nd = E["n"]
    idx = np.arange(min(N_INFER, len(Xs)))
    f1 = lambda i: anchor_scores(bank.score(Xs[i:i + 1], nd[i:i + 1], Ws[i:i + 1]), D["comp"].anchor_of_row, D["anchors"])
    t0 = time.perf_counter()
    anchor_scores(bank.score(Xs, nd, Ws), D["comp"].anchor_of_row, D["anchors"])
    batch_ms = 1000 * (time.perf_counter() - t0) / len(Xs)
    p = bank_params(bank)
    return dict(model=bid, variant="-", device=dev_label, n_nodes=len(bank.M), params_model=p, params_total=p,
                train_s=train_s, train_s_per_node=bank.fit_time_per_node, infer_ms_b1=med_ms(f1, idx),
                infer_ms_batch=batch_ms), bank


def run_comp(D, name, cfg, variant, device, nc_cost):
    tr, va = D["tr"], D["va"]
    S = tr["Xs"].shape[1]
    kw = dict(cfg)
    if issubclass(MODELS[name], _DeepAE):
        kw["device"] = device
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    m = MODELS[name](**kw).fit(model_input(D, variant, tr["Xs"], tr["n"], tr["Ws"], train=True))
    ev = np.where(va["Ws"] > 0, m.errors(model_input(D, variant, va["Xs"], va["n"])), np.nan)
    sd = np.nanstd(np.moveaxis(ev, 1, 0).reshape(ev.shape[1], -1), 1)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    float(np.nanpercentile(ev / sd[None, :, None], 97.5))
    if device == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0 + (nc_cost["train_s"] if variant == "nc" else 0.0)
    E = D["evals"]["test"]
    Xs, _ = D["comp"].transform(E["w"]["X"], E["W"], D["rows"])
    nd = E["n"]
    score = lambda X, n: anchor_scores(m.errors(model_input(D, variant, X, n)) / sd[None, :, None],
                                       D["comp"].anchor_of_row, D["anchors"])
    idx = np.arange(min(N_INFER, len(Xs)))
    score(Xs[:2], nd[:2])                                                   # 준비 실행 (GPU 초기화)
    b1 = med_ms(lambda i: score(Xs[i:i + 1], nd[i:i + 1]), idx)
    t0 = time.perf_counter()
    score(Xs, nd)
    if device == "cuda":
        torch.cuda.synchronize()
    batch_ms = 1000 * (time.perf_counter() - t0) / len(Xs)
    pm = model_params(name, m, S) + S + 1                                   # + 행별 SD, 임계값
    return dict(model=name, variant=variant, cfg=json.dumps(cfg), device=device, params_model=pm,
                params_total=pm + (nc_cost["params"] if variant == "nc" else 0), train_s=train_s,
                infer_ms_b1=b1, infer_ms_batch=batch_ms)


if __name__ == "__main__":
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    mall = D2.load_meta(dcfg)
    mall = mall[mall.ch == dcfg["d2"]["channel"]]
    sel = {}
    for b in ("", "batch2/", "batch2_fix/", "batch2_fix2/"):         # 뒤가 앞을 덮음 (GDN·LAPRPCA 수정판)
        p = ROOT / "results" / "exp1_baselines" / b / "selection.json"
        if p.exists():
            sel.update(json.loads(p.read_text(encoding="utf-8"))["selection"])
    for m in ("USAD", "USADFIX"):                                    # 학습 오류판·발산판 (USADFIX2 로 대체)
        sel.pop(m, None)
    hw = dict(platform=platform.platform(), cpu="AMD Ryzen 7 7800X3D 8-Core", ram_gb=63.1,
              gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, python=platform.python_version(),
              torch=torch.__version__, numpy=np.__version__, blas_threads=1, torch_threads=torch.get_num_threads())
    (OUT / "hardware.json").write_text(json.dumps(hw, indent=1), encoding="utf-8")
    rows, allu = [], []
    for ds in ("d1", "d2"):
        for unit, sp in units(ds, ecfg):
            D = prep(ds, unit, sp, ecfg, dcfg, ["test"], mall)
            t0 = time.perf_counter()
            D["comp"].transform(D["evals"]["test"]["w"]["X"], D["evals"]["test"]["W"], D["rows"])
            comp_ms = 1000 * (time.perf_counter() - t0) / len(D["evals"]["test"]["w"]["X"])
            for bid in BANKS:
                r, bank = run_bank(D, bid, ecfg)
                allu.append(dict(dataset=ds, unit=unit, model=bid, n_nodes=r["n_nodes"], params=r["params_model"],
                                 train_s=r["train_s"], train_s_per_node=r["train_s_per_node"]))
                if unit == REP[ds]:
                    rows.append(dict(dataset=ds, unit=unit, compose_ms_per_window=comp_ms, **r))
                    if bid == "cell":
                        nc_cost = dict(params=r["params_model"] - (2 * len(bank.mu) + 1), train_s=r["train_s"])
            if unit != REP[ds]:
                continue
            for name, d in sel.items():
                if ds not in d or d[ds]["variant"] == "-":
                    continue
                s = d[ds]
                devs = ["cpu", "cuda"] if issubclass(MODELS[name], _DeepAE) and torch.cuda.is_available() else ["cpu"]
                for dev in devs:
                    t0 = time.perf_counter()
                    rows.append(dict(dataset=ds, unit=unit, compose_ms_per_window=comp_ms,
                                     **run_comp(D, name, s["cfg"], s["variant"], dev, nc_cost)))
                    print(f"  {ds} {unit} {name} {s['variant']} {dev}: {time.perf_counter() - t0:.1f}s", flush=True)
    C = pd.DataFrame(rows)
    C["memory_kb_total"] = C.params_total * 4 / 1024
    C["memory_kb_model"] = C.params_model * 4 / 1024
    C.to_csv(OUT / "cost.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(allu).to_csv(OUT / "params_all_units.csv", index=False, encoding="utf-8-sig")
    sys.stdout.reconfigure(encoding="utf-8")
    print(C[["dataset", "model", "variant", "device", "params_model", "params_total", "train_s", "infer_ms_b1",
             "infer_ms_batch"]].round(4).to_string(index=False))
