"""실험 R1-5: 시뮬레이션 연구 (doc/실험R1-5_계획.md, 사전 등록 2026-10-28 21:32, 지문 doc/R1_등록_지문.md).

한 반복 = 단위 하나(노드 12, 행 4, T 12). 참조 10개(CG CF CI CMI NM NMed PS PS1 PSH PS1H)를
원고와 같은 표준화(S1, 단위의 관측 학습 칸) · 보정(깨끗한 검증 창의 행별 잔차 평균·SD, 97.5 백분위) · 점수 경로로 비교한다.
M5(예측 분산 식 확인)는 별도 함수 predvar_study.

사용법: python experiments/expR1_5_sim.py [--reps 100] [--workers 8] [--quick] [--only-m5]
출력: results/expR1/sim/metrics.csv (반복 × 참조 행), m5_predvar.csv, design.json, run_info.json
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model.bank import NormalSurfaceBank  # noqa: E402
from src.model.pspline import SurfaceFit, bspline_basis, diff_matrix, kron_penalties, PenaltySpectrum  # noqa: E402

OUT = ROOT / "results/expR1/sim"
M_NODES, S, T, NVAL, NTEST = 12, 4, 12, 5, 5
AMP, NODE_SD, NODE_VAR = 2.0, 2.0, 0.3
P_EVENT = 0.5
REFS = ["CG", "CF", "CI", "CMI", "NM", "NMed", "PS", "PS1", "PSH", "PS1H"]
BASE = dict(N=10, L="L1", noise="G", miss="MCAR", r=0.0, p=0.0, atype="mix", mag=2.0)


# ---------------------------------------------------------------- 설계
def design():
    cells = []

    def add(tag, **kw):
        c = dict(BASE)
        c.update(kw)
        key = json.dumps(c, sort_keys=True)
        for e in cells:
            if e["key"] == key:
                if tag not in e["tags"]:
                    e["tags"].append(tag)
                return
        cells.append({"key": key, "cfg": c, "tags": [tag]})

    add("base")
    for N in (3, 5, 10, 20):
        add("F1", N=N)
    for L in ("L0", "L1", "L2"):
        add("F2", L=L)
    for r in (0.3, 0.5, 0.7):
        add("F3", r=r)
    add("F3", miss="BURST", r=0.5)            # r = 평균 손실 π
    add("F3", miss="MNAR", r=0.0)             # r 쓰지 않음
    for a in ("spike", "shift", "drift"):
        for mg in (1.0, 2.0, 3.0):
            add("F4", atype=a, mag=mg)
    for p in (0.0, 0.1, 0.3):
        add("F5", p=p)
    for nz in ("G", "T3", "CN"):
        add("F6", noise=nz)
    for N in (3, 5, 10, 20):
        for r in (0.0, 0.3, 0.5, 0.7):
            add("G1", N=N, r=r)
    for L in ("L0", "L1", "L2"):
        for r in (0.0, 0.3, 0.5, 0.7):
            add("G2", L=L, r=r)
    for nz in ("G", "T3", "CN"):
        for p in (0.0, 0.1, 0.3):
            add("G3", noise=nz, p=p)
    # 보완 칸 (실행 전 수정 2, 결과를 보기 전 추가): 예상 S3·S5 를 판정할 칸이 위 설계에 없었다. 맨 뒤에 붙여 기존 칸 번호·seed 는 그대로.
    add("S3x", L="L2", N=3)
    for nz in ("T3", "CN"):
        add("S5x", L="L0", noise=nz)
    for i, e in enumerate(cells):
        e["id"] = i
    return cells


# ---------------------------------------------------------------- 생성
def shapes(rng, L, n):
    """(n, T) 모양. L1: 한 창에 사인 1주기. L2: 1–4 주기 사인 합(진폭 1/k), L1 과 같은 SD 로 맞춤."""
    t = np.arange(T)
    if L == "L0":
        return np.zeros((n, T))
    if L == "L1":
        ph = rng.uniform(0, 2 * np.pi, n)[:, None]
        return np.sin(2 * np.pi * t / T + ph)
    ks = np.arange(1, 5)
    ph = rng.uniform(0, 2 * np.pi, (n, len(ks)))
    f = (np.sin(2 * np.pi * ks[None, :, None] * t[None, None, :] / T + ph[..., None]) / ks[None, :, None]).sum(1)
    return f / f.std(1, keepdims=True) / np.sqrt(2)


def noise(rng, kind, shape):
    if kind == "G":
        return rng.standard_normal(shape)
    if kind == "T3":                                          # MAD 를 가우스와 맞춤: 0.6745 / t3 의 75 % 분위 0.7649
        return rng.standard_t(3, shape) * (0.6745 / 0.7649)
    z = rng.standard_normal(shape)                            # CN: 0.95 N(0,1) + 0.05 N(0, 10²)
    return np.where(rng.random(shape) < 0.05, 10.0 * z, z)


def inject(rng, Y, atype, mag, p_event):
    """Y (n, S, T). 창마다 확률 p_event 로 행 하나에 이상. 라벨 = 더한 크기 ≥ Δ/2 인 칸(drift 는 뒤쪽 절반)."""
    Y = Y.copy()
    lab = np.zeros(Y.shape, bool)
    for i in range(Y.shape[0]):
        if rng.random() >= p_event:
            continue
        s = int(rng.integers(S))
        typ = atype if atype != "mix" else ("spike", "shift", "drift")[int(rng.integers(3))]
        sign = 1.0 if rng.random() < 0.5 else -1.0
        if typ == "spike":
            ln = int(rng.integers(1, 3))
            t0 = int(rng.integers(0, T - ln + 1))
            add = np.zeros(T)
            add[t0:t0 + ln] = mag
        elif typ == "shift":
            add = np.full(T, mag)
        else:
            add = np.linspace(0.0, mag, T)
        Y[i, s] += sign * add
        lab[i, s] = add >= mag / 2
    return Y, lab


def train_mask(rng, cfg, Ytr):
    """Ytr (n, S, T) 한 노드. 반환 관측 여부 (n, S, T) bool."""
    n = Ytr.shape[0]
    if cfg["miss"] == "MCAR":
        drop = rng.random(Ytr.shape) < cfg["r"]
    elif cfg["miss"] == "BURST":                              # Gilbert–Elliott (exp6_missing_patterns 와 같은 식)
        pi, Lb = cfg["r"], 4.0
        q = 1.0 / Lb
        p = pi * q / (1.0 - pi)
        st = rng.random((n, S)) < pi
        drop = np.zeros(Ytr.shape, bool)
        for t in range(T):
            drop[:, :, t] = st
            u = rng.random((n, S))
            st = np.where(st, u >= q, u < p)
    else:                                                     # MNAR: 노드·행 30 % 분위 아래 0.7, 나머지 0.2
        q30 = np.quantile(Ytr.transpose(1, 0, 2).reshape(S, -1), 0.3, axis=1)
        low = Ytr < q30[None, :, None]
        drop = rng.random(Ytr.shape) < np.where(low, 0.7, 0.2)
    return ~drop


# ---------------------------------------------------------------- 참조
def make_bank(ref):
    if ref in ("CG", "CMI", "NMed"):
        return NormalSurfaceBank(T, K_t=T, time_basis="cell")
    if ref == "CF":
        return NormalSurfaceBank(T, K_t=T, time_basis="cellfill")
    if ref == "CI":
        return NormalSurfaceBank(T, K_t=T, time_basis="cellinterp")
    if ref == "NM":
        return NormalSurfaceBank(T, K_t=T, time_basis="const")
    d = 1 if ref in ("PS1", "PS1H") else 2
    rob = "huber" if ref in ("PSH", "PS1H") else None
    return NormalSurfaceBank(T, K_t=T, time_basis="pspline", diff_order=d, robust=rob)


def robust_override(bank, ref, X, W, nodes):
    """CMI: 칸별 중앙값 + 시간 보간(행 전체가 비면 0 = 단위 중심). NMed: 노드·행 중앙값."""
    t = np.arange(T)
    for nd in bank.M:
        ix = nodes == nd
        x = np.where(W[ix] > 0, X[ix], np.nan)
        with np.errstate(all="ignore"):
            if ref == "CMI":
                med = np.nanmedian(x, axis=0)                 # (S, T)
                for s in range(S):
                    ok = np.isfinite(med[s])
                    if ok.all():
                        continue
                    med[s] = np.interp(t, t[ok], med[s, ok]) if ok.any() else 0.0
                bank.M[nd] = med
            else:
                v = np.nanmedian(x.transpose(1, 0, 2).reshape(S, -1), axis=1)
                bank.M[nd] = np.repeat(np.where(np.isfinite(v), v, 0.0)[:, None], T, axis=1)


def scale_params(X, W):
    x = np.where(W > 0, X, np.nan)
    flat = x.transpose(1, 0, 2).reshape(S, -1)
    c, d = np.nanmean(flat, 1), np.nanstd(flat, 1)
    d = np.where(np.isfinite(d) & (d > 0), d, 1.0)
    c = np.where(np.isfinite(c), c, 0.0)
    return c, d


def fit_ref(ref, Xtr, Wtr, ntr, Xva, nva):
    """표준화 → 적합 → 깨끗한 검증으로 보정. 반환 (bank, c, d)."""
    c, d = scale_params(Xtr, Wtr)
    z = lambda X: (X - c[None, :, None]) / d[None, :, None]  # noqa: E731
    Ztr = np.where(Wtr > 0, z(Xtr), 0.0)
    bank = make_bank(ref).fit(Ztr, Wtr, ntr)
    if ref in ("CMI", "NMed"):
        robust_override(bank, ref, Ztr, Wtr, ntr)
    Zva = z(Xva)
    bank.calibrate(Zva, np.ones_like(Zva), nva)
    return bank, c, d, z


# ---------------------------------------------------------------- 한 반복
def replicate(task):
    cell, rep = task
    cfg = cell["cfg"]
    seed = 10000 + cell["id"] * 1000 + rep
    rng = np.random.default_rng(seed)
    N = cfg["N"]
    a = rng.normal(0.0, NODE_SD, (M_NODES, S))
    f = shapes(rng, cfg["L"], S)
    b = shapes(rng, cfg["L"], M_NODES * S).reshape(M_NODES, S, T)
    m = a[..., None] + AMP * f[None] + NODE_VAR * AMP * b                       # (M, S, T) 참 정상 곡면

    def draw(nw):
        return m[:, None] + noise(rng, cfg["noise"], (M_NODES, nw, S, T))

    Ytr, Yva, Yte = draw(N), draw(NVAL), draw(NTEST)
    Lte = np.zeros(Yte.shape, bool)
    for j in range(M_NODES):
        if cfg["p"] > 0:
            Ytr[j], _ = inject(rng, Ytr[j], cfg["atype"], cfg["mag"], cfg["p"])   # 라벨 없는 학습 오염
        Yte[j], Lte[j] = inject(rng, Yte[j], cfg["atype"], cfg["mag"], P_EVENT)
    masked = not (cfg["miss"] == "MCAR" and cfg["r"] == 0.0)
    Otr = np.ones(Ytr.shape, bool)
    if masked:
        for j in range(M_NODES):
            Otr[j] = train_mask(rng, cfg, Ytr[j])

    flat = lambda A: A.reshape(-1, S, T)  # noqa: E731
    ntr = np.repeat(np.arange(M_NODES), N)
    nva = np.repeat(np.arange(M_NODES), NVAL)
    nte = np.repeat(np.arange(M_NODES), NTEST)
    Xtr, Wtr = flat(Ytr), flat(Otr).astype(float)
    Xva, Xte, Lt = flat(Yva), flat(Yte), flat(Lte)
    Wfull = np.ones_like(Wtr)
    cnt = Otr.sum(1)                                                             # (M, S, T) 학습 관측 수

    rows = []
    for ref in REFS:
        t0 = time.perf_counter()
        bank, c, d, z = fit_ref(ref, Xtr, Wtr, ntr, Xva, nva)
        A = bank.score(z(Xte), nte)
        auc = float(average_precision_score(Lt.ravel(), A.ravel())) if 0 < Lt.sum() < Lt.size else np.nan
        Morig = np.stack([bank.M[j] * d[:, None] + c[:, None] for j in range(M_NODES)])
        err = Morig - m
        obs, emp = cnt > 0, cnt == 0
        sdR = float(np.mean(bank.sd * d))
        # 오경보율: 칸 / 그룹(행 최댓값) — (a) 칸 임계, (c) 검증 그룹 점수 97.5 %
        Ava = bank.score(z(Xva), nva)
        Gva, Gte = np.nanmax(Ava, axis=1), np.nanmax(A, axis=1)
        thr_g = float(np.nanpercentile(Gva, 97.5))
        norm_cell = ~Lt
        norm_grp = ~Lt.any(axis=1)
        row = dict(cell=cell["id"], tags="+".join(cell["tags"]), rep=rep, seed=seed, ref=ref, **cfg,
                   auc_pr=auc, pos_rate=float(Lt.mean()),
                   rmse_obs=float(np.sqrt(np.mean(err[obs] ** 2))),
                   rmse_empty=float(np.sqrt(np.mean(err[emp] ** 2))) if emp.any() else np.nan,
                   empty_frac=float(emp.mean()), sdR=sdR,
                   far_cell=float((A[norm_cell] > bank.thr).mean()),
                   far_grp_a=float((Gte[norm_grp] > bank.thr).mean()),
                   far_grp_c=float((Gte[norm_grp] > thr_g).mean()),
                   val_far_cell=float((Ava > bank.thr).mean()),
                   val_far_grp_a=float((Gva > bank.thr).mean()),
                   lam_med=float(np.median([f.lam.get("t", np.nan) for f in bank.fits.values()]))
                   if ref.startswith("PS") else np.nan)
        if masked:                                                               # σ_R 비: 같은 반복, 결측 없는 학습
            b0, _, d0, _ = fit_ref(ref, Xtr, Wfull, ntr, Xva, nva)
            row["sdR_ratio"] = sdR / float(np.mean(b0.sd * d0))
        else:
            row["sdR_ratio"] = 1.0
        row["fit_s"] = time.perf_counter() - t0
        rows.append(row)
    return rows


# ---------------------------------------------------------------- M5: 예측 분산 식
def predvar_study(n_masks=20, n_draw=500, N=5, r=0.5, lams=(0.1, 1.0, 10.0), seed=777):
    """행 하나(S = 1), L1 모양, 가우스 σ = 1. 마스크를 고정하고 잡음만 다시 뽑아 빈 칸·관측 칸의 경험 분산을 식과 비교."""
    rng = np.random.default_rng(seed)
    B = bspline_basis(T, T)
    Dt = diff_matrix(T, 2)
    P = kron_penalties(1, T, Dt, {})
    spec = PenaltySpectrum(1, T, Dt, {})
    Pt = Dt.T @ Dt
    out = []
    t = np.arange(T)
    for k in range(n_masks):
        mtrue = AMP * np.sin(2 * np.pi * t / T + rng.uniform(0, 2 * np.pi))
        while True:
            O = rng.random((N, 1, T)) >= r
            cnt = O.sum(0)[0]
            if (cnt == 0).any() and (cnt > 0).sum() >= 2:
                break
        C = np.diag(cnt.astype(float))
        Acore = B.T @ C @ B
        ridge = 1e-6 * max(np.trace(Acore) / T, 1e-12)
        for lam in list(lams) + ["sap"]:
            preds, formula = [], []
            for _ in range(n_draw):
                Y = mtrue[None, None, :] + rng.standard_normal((N, 1, T))
                fit = SurfaceFit(B, P, spectrum=spec)
                if lam == "sap":
                    fit.fit(Y, O.astype(float))
                    lv = fit.lam["t"]
                else:
                    fit.fit(Y, O.astype(float), lam={"t": float(lam)}, sap=False)
                    lv = float(lam)
                preds.append(fit.predict()[0])
                H = Acore + lv * Pt + ridge * np.eye(T)
                Hi = np.linalg.inv(H)
                formula.append(np.diag(B @ Hi @ Acore @ Hi @ B.T))
                if lam != "sap":
                    formula = formula[:1]                     # λ 고정이면 식은 한 번이면 된다
            preds = np.array(preds)
            v_emp = preds.var(axis=0, ddof=1)
            v_for = np.mean(formula, axis=0)
            for tt in range(T):
                out.append(dict(mask=k, lam=str(lam), t=tt, count=int(cnt[tt]), empty=bool(cnt[tt] == 0),
                                var_emp=float(v_emp[tt]), var_formula=float(v_for[tt]),
                                ratio=float(v_emp[tt] / v_for[tt]),
                                bias=float(preds.mean(0)[tt] - mtrue[tt])))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 실행
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--quick", action="store_true", help="설계 칸 3개 × 반복 2 (시간 추정용, 결과 파일은 quick_ 접두)")
    ap.add_argument("--only-m5", action="store_true")
    ap.add_argument("--min-cell", type=int, default=0, help="이 번호 이상 칸만 실행(보완 칸), 결과 파일은 cells<번호>_ 접두, M5 생략")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cells = design()
    pre = "quick_" if args.quick else (f"cells{args.min_cell}_" if args.min_cell else "")
    if args.min_cell:
        cells_run = [c for c in cells if c["id"] >= args.min_cell]
    (OUT / "design.json").write_text(json.dumps([{"id": c["id"], "tags": c["tags"], "cfg": c["cfg"]} for c in cells],
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    t0 = time.time()
    if not args.only_m5:
        use = cells[:3] if args.quick else (cells_run if args.min_cell else cells)
        reps = 2 if args.quick else args.reps
        tasks = [(c, r) for c in use for r in range(reps)]
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, res in enumerate(ex.map(replicate, tasks, chunksize=4)):
                rows.extend(res)
                if (i + 1) % 200 == 0:
                    print(f"{i + 1}/{len(tasks)} 반복, {time.time() - t0:.0f} s", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"{pre}metrics.csv", index=False, encoding="utf-8-sig")
        print(f"반복 {len(tasks)}개, {time.time() - t0:.0f} s")
    if args.min_cell:
        return
    t1 = time.time()
    m5 = predvar_study(n_masks=3 if args.quick else 20, n_draw=50 if args.quick else 500)
    m5.to_csv(OUT / f"{pre}m5_predvar.csv", index=False, encoding="utf-8-sig")
    print(f"M5 {time.time() - t1:.0f} s")
    (OUT / f"{pre}run_info.json").write_text(json.dumps(dict(cells=len(cells), reps=args.reps, workers=args.workers,
                                                            quick=args.quick, seconds=time.time() - t0),
                                                       indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
