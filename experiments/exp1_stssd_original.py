"""실험 1 ST-SSD 원형 변형 R·D (doc/실험1_ST-SSD_계획.md "추가 비교", 사전 등록 2026-10-12).

R: 원형 감시 통계량 (식 12). 배경 G1 (전역, T1·O1, 시간 1차 차분) · N (노드별, PS 와 같은 배경).
   z = 잔차 / σ_R (검증), â_γ = S_{γ/2}(z) (항등 이상 기저), T_γ = (âᵀz)²/(âᵀâ) (â = 0 이면 0),
   γ ∈ γ_max·{0..20}/20, γ_max = 학습 창에서 â 가 모두 0 이 되는 최소 γ = 2·max|z_train|.
   검증 정상 창의 평균·SD 로 표준화 (SD = 0 인 γ 제외), T̃ = max, 셀 점수 = |â_{γ*}|, 윈도우 점수 = T̃.
D: 동적 평균 (4.2 거칠기 최소화 재귀, 부록 C). μ_t = (1 − λ̃)μ_{t−1} + λ̃ H_s(y_t − a_t), μ_0 = G1 배경.
   H_s = G1 평활 (λ / 학습 윈도우 수), 이상 = σ ⊙ S_{γ/2}((y − μ)/σ), 표본마다 수렴까지 번갈아 갱신.
   순서 = 같은 구간(seg) 안 윈도우 시간순. 셀 점수 = 수렴한 μ 기준 |z|. λ̃ ∈ {.1,.3,.5} × γ ∈ {1,2,4} 를 검증으로 선택.
D′ (사후 탐색, 검증 결과를 본 뒤 추가): D 와 같고 배경·H_s 를 행 독립 전역 P-spline (F01 T4, 전역, K_t = 12, 1차 차분) 으로 둠.
PS 도 같은 평가 세트로 함께 돈다 (짝지은 비교).
사용법: python experiments/exp1_stssd_original.py search | search_dt | test   (search_dt = D′, 사후 탐색)
출력: results/exp1_stssd/original/metrics_<단계>_<데이터>.csv, selection.json
"""
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
from exp1_stssd import variants  # noqa: E402
from exp6_part2 import all_inj, load_unit, view  # noqa: E402
from node_model_eval import build, full_spec  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.metrics.anomaly import auc_pr, evaluate  # noqa: E402
from src.model.bank import anchor_scores  # noqa: E402
from src.model.pspline import SurfaceFit  # noqa: E402

OUT = ROOT / "results" / "exp1_stssd" / "original"
D_GRID = [(lt, g) for lt in (0.1, 0.3, 0.5) for g in (1.0, 2.0, 4.0)]
VIEWS = {"d1": r"^SC\d$", "d2": r"^SC\dm$"}
EPS = 1e-6


def soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def resid_sd(bank, Xs, Ws, nodes):
    r = np.where(Ws > 0, bank.residual(Xs, nodes), np.nan)
    sd = np.nanstd(np.moveaxis(r, 1, 0).reshape(r.shape[1], -1), 1)
    return np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)


def zmap(bank, Xs, Ws, nodes, sd):
    R = bank.residual(Xs, nodes)
    m = (Ws > 0) & np.isfinite(R)
    return np.where(m, R / (sd[None, :, None] + EPS), 0.0), m


class LRT:
    """변형 R."""

    def __init__(self, bank, sd):
        self.bank, self.sd = bank, sd

    def _stats(self, z):
        A = soft(z[None], self.grid[:, None, None, None] / 2)                  # (G, N, S, T)
        az, aa = (A * z[None]).sum((2, 3)), (A * A).sum((2, 3))
        return A, np.where(aa > 0, az ** 2 / np.maximum(aa, 1e-300), 0.0)      # (G, N)

    def calibrate(self, tr, va):
        zt, _ = zmap(self.bank, *tr, self.sd)
        self.grid = 2 * np.max(np.abs(zt)) * np.arange(21) / 20
        zv, mv = zmap(self.bank, *va, self.sd)
        _, Tv = self._stats(zv)
        self.E, self.SD = Tv.mean(1), Tv.std(1)
        self.keep = self.SD > 0
        A, W = self.score(*va)
        self.thr = float(np.percentile(A[mv], 97.5))
        return self

    def score(self, Xs, Ws, nodes):
        z, m = zmap(self.bank, Xs, Ws, nodes, self.sd)
        A, T = self._stats(z)
        st = np.where(self.keep[:, None], (T - self.E[:, None]) / np.where(self.keep, self.SD, 1.0)[:, None], -np.inf)
        j = np.argmax(st, 0)
        cell = np.abs(A[j, np.arange(len(j))])
        return np.where(m, cell, np.nan), st.max(0)


class Dynamic:
    """변형 D (배경 = G1 전역 은행)."""

    def __init__(self, bank, sd, node, n_train, lt, gamma):
        self.bank, self.sd, self.node, self.lt, self.gamma = bank, sd, node, lt, gamma
        f = bank.fits[node]
        self.mu0 = bank.M[node]
        self.Bt, self.P = f.Bt, f.P
        self.lam = {k: v / n_train for k, v in f.lam.items()}

    def _hs(self, y, w):
        return SurfaceFit(self.Bt, self.P).fit(np.nan_to_num(y)[None], w[None], lam=self.lam, sap=False).predict()

    def _step(self, y, w, mu_prev):
        sd = self.sd[:, None]
        a = np.zeros_like(y)
        for _ in range(30):
            mu = (1 - self.lt) * mu_prev + self.lt * self._hs(np.where(w, y - a, 0.0), w.astype(float))
            z = np.where(w, (y - mu) / (sd + EPS), 0.0)
            a_new = soft(z, self.gamma / 2) * sd
            done = np.max(np.abs(a_new - a)) < 1e-6
            a = a_new
            if done:
                break
        return mu, z

    def warm_mu(self, warm):
        """시퀀스별 시작 평균: 앞선 정상 윈도우 (Xw, Ww) 로 재귀를 먼저 돌린 상태 (없으면 μ_0). 온라인 운용 모사."""
        out = []
        for wq in warm:
            mu = self.mu0
            if wq is not None:
                Xw, Ww = wq
                for i in range(len(Xw)):
                    mu, _ = self._step(Xw[i], (Ww[i] > 0) & np.isfinite(Xw[i]), mu)
            out.append(mu)
        return out

    def score(self, Xs, Ws, order, mu_init=None):
        """order: 시퀀스별 윈도우 인덱스 (시간순). mu_init: 시퀀스별 시작 평균 (warm_mu). 반환: |z| (N, S, T)."""
        out = np.full(Xs.shape, np.nan)
        for q, seq in enumerate(order):
            mu_prev = self.mu0 if mu_init is None else mu_init[q]
            for i in seq:
                w = (Ws[i] > 0) & np.isfinite(Xs[i])
                mu_prev, z = self._step(Xs[i], w, mu_prev)
                out[i] = np.where(w, np.abs(z), np.nan)
        return out


def sequences(meta, ix):
    m = meta.iloc[ix].assign(_i=np.arange(len(ix)))
    return [g.sort_values("tidx0")["_i"].to_numpy() for _, g in m.groupby("seg")]


def warm_sets(meta, ix, comp, X, W, rows, spl):
    """시퀀스(seg)마다 같은 구간의 앞선 정상 윈도우 (검증: 학습 몫, 테스트: 학습+검증 몫), 합성 공간. 없으면 None."""
    prev = {"val": ("train",), "test": ("train", "val")}[spl]
    m = meta.iloc[ix]
    out = []
    for sg, g in m.assign(_i=np.arange(len(ix))).groupby("seg"):
        t0 = g.tidx0.min()
        w = np.flatnonzero((meta.seg.to_numpy() == sg) & meta.split.isin(prev).to_numpy() & (meta.tidx0.to_numpy() < t0))
        w = w[np.argsort(meta.tidx0.to_numpy()[w])]
        out.append(comp.transform(X[w], W[w], rows) if len(w) else None)
    return out


def win_auc(cells_or_win, lab):
    """윈도우 단위 탐지 AUC-PR. lab: (N, A, T). 점수: (N,) 또는 셀 (N, A, T) 의 최댓값.
    셀 점수는 라벨이 있는 셀만 쓴다 (D2 실제 NLOS 셀 = 라벨 NaN 은 뺌. R 의 T̃ 는 관측 마스크로 이미 제외)."""
    y = (np.nan_to_num(lab) == 1).any((1, 2)).astype(float)
    if cells_or_win.ndim == 1:
        s = cells_or_win
    else:
        ok = np.isfinite(cells_or_win) & np.isfinite(lab)
        s = np.max(np.where(ok, cells_or_win, -np.inf), axis=(1, 2))
    return auc_pr([(s[None], y[None])])


BG_OF = {"D": "G1", "DT": "G1T4", "PS": "PS"}                     # 변형 -> 배경 (R 은 cfg[1])


def run_unit(ds, unit, sp, E, ecfg, dcfg, splits, configs):
    X, meta, rows, cells = load_unit(ds, unit, sp, E, dcfg)
    nodes, W, split = view(ds, unit, X, meta, cells)
    tr, va = split == "train", split == "val"
    anchors = [f"A{a}" for a in range(1, 5 if ds == "d1" else 9)]
    inj = all_inj(ds, unit, E, X, rows, cells.get("lab_D"), split, splits)
    V = variants(ecfg)
    f01 = ecfg["ids"]["F01"]
    BG = {"G1": V["G1"], "N": V["N"], "PS": (f01, {"K_t": 12}),
          "G1T4": (f01, {"node": "global", "K_t": 12, "diff_order": 1})}      # D′: 행 독립 전역 (사후 탐색)
    gnodes = np.array([f"{unit}/all"] * len(X))
    built = {}
    for bg in sorted({BG_OF.get(c[0]) or c[1] for c in configs}):
        code, spec = BG[bg]
        nd = gnodes if bg in ("G1", "G1T4") else nodes
        comp, bank, _ = build(code, full_spec(spec, ecfg), ecfg, ds, X[tr], W[tr], rows, nd[tr], X[va], W[va], nd[va])
        Xt, Wt = comp.transform(X[tr], W[tr], rows)
        Xv, Wv = comp.transform(X[va], W[va], rows)
        built[bg] = dict(comp=comp, bank=bank, nodes=nd, tr=(Xt, Wt, nd[tr]), va=(Xv, Wv, nd[va]),
                         sd=resid_sd(bank, Xv, Wv, nd[va]))
    out = []
    for cfg in configs:
        kind = cfg[0]
        bgname = BG_OF.get(kind) or cfg[1]
        b = built[bgname]
        comp, bank = b["comp"], b["bank"]
        dyn = kind in ("D", "DT")
        if kind == "R":
            sc = LRT(bank, b["sd"]).calibrate(b["tr"], b["va"])
        elif dyn:
            sc = Dynamic(bank, b["sd"], f"{unit}/all", int(tr.sum()), cfg[1], cfg[2])
            ixv = np.flatnonzero(va)
            Av = sc.score(*b["va"][:2], sequences(meta, ixv), sc.warm_mu(warm_sets(meta, ixv, comp, X, W, rows, "val")))
            sc.thr = float(np.nanpercentile(Av, 97.5))
        for spl, (ix, sets) in inj.items():
            order = sequences(meta, ix)
            warm = sc.warm_mu(warm_sets(meta, ix, comp, X, W, rows, spl)) if dyn else None
            for scn, s, Xi, lab in sets:
                Xs, Ws = comp.transform(Xi, W[ix], rows)
                if kind == "R":
                    cell, win = sc.score(Xs, Ws, b["nodes"][ix])
                    thr = sc.thr
                elif dyn:
                    cell = sc.score(Xs, Ws, order, warm)
                    win, thr = None, sc.thr
                else:                                                      # PS
                    cell, win, thr = bank.score(Xs, b["nodes"][ix], Ws), None, bank.thr
                A = anchor_scores(cell, comp.anchor_of_row, anchors)
                seqs = [(A[n], lab[n]) for n in range(len(A))]
                out.append(dict(dataset=ds, space=unit, split=spl, scenario=scn, seed=s, variant=kind,
                                background=bgname if kind != "PS" else "N",
                                lam_tilde=cfg[1] if dyn else np.nan, gamma=cfg[2] if dyn else np.nan,
                                win_auc_pr=win_auc(win if win is not None else A, lab),
                                **evaluate(seqs, thr, "time" if ds == "d1" else E["level"])))
        print(f"  {ds} {unit} {cfg}", flush=True)
    return out


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "search"          # search | search_dt (D′, 사후 탐색) | test
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    split = "test" if stage == "test" else "val"
    sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8")) if stage != "search" else {}
    for ds in ("d1", "d2"):
        E = ecfg["datasets"][ds]
        units = list(E["spaces"].items()) if ds == "d1" else [(f"env{e}", None) for e in E["envs"]]
        if stage == "search":
            configs = [("R", "G1"), ("R", "N"), ("PS",)] + [("D", lt, g) for lt, g in D_GRID]
        elif stage == "search_dt":
            configs = [("DT", lt, g) for lt, g in D_GRID]
        else:
            configs = [("R", "G1"), ("R", "N"), ("PS",), ("D", *sel[ds]), ("DT", *sel[f"{ds}_DT"])]
        t0 = time.perf_counter()
        rows = []
        for u, sp in units:
            rows += run_unit(ds, u, sp, E, ecfg, dcfg, [split], configs)
        pd.DataFrame(rows).to_csv(OUT / f"metrics_{stage}_{ds}.csv", index=False, encoding="utf-8-sig")
        print(f"{ds}: {len(rows)} rows, {time.perf_counter() - t0:.1f}s", flush=True)
    if stage in ("search", "search_dt"):
        kind, sfx = ("D", "") if stage == "search" else ("DT", "_DT")
        for ds in ("d1", "d2"):
            M = pd.read_csv(OUT / f"metrics_{stage}_{ds}.csv")
            d = M[(M.variant == kind) & M.scenario.astype(str).str.match(VIEWS[ds])]
            t = d.groupby(["lam_tilde", "gamma"]).auc_pr.mean()
            sel[ds + sfx] = [float(v) for v in t.idxmax()]
        (OUT / "selection.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
        print(json.dumps(sel))
