"""실험 R1-6: 다른 센서(Wi-Fi RSS)에서 같은 구조 검증 — D7 UJI (doc/실험R1-6_계획.md, 사전 등록 2026-10-28 21:32).

노드 = (기준점, 방문), 창 = 연속 스캔 6회(T = 6), 단위 = 층, 행 = 노드 학습 스캔의 50 % 이상에서 들린 AP(노드별, 단위는 합집합).
분할: 학습 trn01–10, 검증 trn11–13, 테스트 trn14·trn15·tst01·tst05 (구조가 다른 세트는 뺌).
참조(탐지기 self): GM CG CF CI NM PS PS1 PSH PS1H. 조건: N 자연 결측 · M MCAR r 0.5/0.7 (조건 seed 0–2) · F 학습 trn08–10.
주입(검증·테스트): 창마다 확률 0.5, W1 행 1 · W2 행 2 · W3 짧음(스캔 1–2). 지속(40 %) 또는 구간 2–4 스캔(60 %).
크기 Δ = U(3, 6) × 노드·행 학습 RSS SD 를 뺀다. 미수신 칸은 라벨 NaN.

사용법:
  python experiments/expR1_6_uji.py run --splits val      # 검증 → 허용오차 계산·판정
  python experiments/expR1_6_uji.py run --splits test     # 테스트 한 번
  python experiments/expR1_6_uji.py summarize --split val|test
  python experiments/expR1_6_uji.py shift                 # 6절 구성 변화(서술)
출력: results/expR1/uji/
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.data import d7_uji as U  # noqa: E402
from src.model.bank import NormalSurfaceBank  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = ROOT / "results" / "expR1" / "uji"
T = 6
SETS = {"train": [f"trn{i:02d}" for i in range(1, 11)], "val": [f"trn{i:02d}" for i in range(11, 14)],
        "test": ["trn14", "trn15", "tst01", "tst05"]}
F_SETS = [f"trn{i:02d}" for i in range(8, 11)]
REFS = ["GM", "CG", "CF", "CI", "NM", "PS", "PS1", "PSH", "PS1H"]
SCEN = {"W1": dict(n_rows=1, short=False), "W2": dict(n_rows=2, short=False), "W3": dict(n_rows=1, short=True)}
SEEDS = {"val": [142, 143, 144, 145, 146], "test": [42, 43, 44, 45, 46]}
CONDS = [("N", None, -1), ("M", 0.5, 0), ("M", 0.5, 1), ("M", 0.5, 2), ("M", 0.7, 0), ("M", 0.7, 1), ("M", 0.7, 2),
         ("F", None, -1)]


def load_all(log):
    pidx = U.point_index_month1()
    data = {}
    for spl, names in SETS.items():
        for nm in names:
            r = U.windows(1, nm, pidx)
            if r is None:
                log.append(f"세트 {nm}: 블록 구조·좌표가 trn 과 달라 뺌")
                continue
            X, node, floor = r
            data[nm] = dict(X=X, node=node, floor=floor, split=spl)
    return data, pidx


def unit_arrays(data, floor, train_sets):
    """층 하나의 배열. 행 = 노드별 선택 AP 의 합집합(학습 세트 기준)."""
    parts = [(nm, d) for nm, d in data.items()]
    X = np.concatenate([d["X"][d["floor"] == floor] for _, d in parts])
    node = np.concatenate([d["node"][d["floor"] == floor] for _, d in parts])
    sname = np.concatenate([np.full(int((d["floor"] == floor).sum()), nm) for nm, d in parts])
    split = np.concatenate([np.full(int((d["floor"] == floor).sum()), d["split"]) for _, d in parts])
    trm = np.isin(sname, train_sets)
    sel = {}
    for nd in np.unique(node):
        ix = (node == nd) & trm
        det = np.isfinite(X[ix]).mean(axis=(0, 2)) if ix.any() else np.zeros(X.shape[1])
        sel[nd] = det >= 0.5
    rows = np.flatnonzero(np.any(np.stack(list(sel.values())), axis=0))
    keep = np.stack([sel[n][rows] for n in node])                           # (n, S)
    return dict(X=X[:, rows], node=node, sname=sname, split=split, keep=keep, rows=rows, trm=trm)


def make_bank(ref):
    if ref in ("GM", "CG"):
        return NormalSurfaceBank(T, T, {}, time_basis="cell")
    if ref == "CF":
        return NormalSurfaceBank(T, T, {}, time_basis="cellfill")
    if ref == "CI":
        return NormalSurfaceBank(T, T, {}, time_basis="cellinterp")
    if ref == "NM":
        return NormalSurfaceBank(T, T, {}, time_basis="const")
    d = 1 if ref in ("PS1", "PS1H") else 2
    rob = "huber" if ref in ("PSH", "PS1H") else None
    return NormalSurfaceBank(T, T, {}, time_basis="pspline", diff_order=d, robust=rob)


def inject(X, keep, sd_node, scen, seed, scen_idx):
    rng = np.random.default_rng([seed, scen_idx])
    Xi = X.copy()
    obs = np.isfinite(X)
    lab = np.where(keep[:, :, None] & obs, 0.0, np.nan)
    sc = SCEN[scen]
    for i in range(len(X)):
        if rng.random() >= 0.5:
            continue
        elig = np.flatnonzero(keep[i] & (obs[i].mean(1) >= 0.5))
        if not len(elig):
            continue
        rs = rng.choice(elig, min(sc["n_rows"], len(elig)), replace=False)
        if sc["short"]:
            ln = int(rng.integers(1, 3)); t0 = int(rng.integers(0, T - ln + 1))
        elif rng.random() < 0.4:
            ln, t0 = T, 0
        else:
            ln = int(rng.integers(2, 5)); t0 = int(rng.integers(0, T - ln + 1))
        for r in rs:
            delta = rng.uniform(3, 6) * sd_node[i, r]
            seg = slice(t0, t0 + ln)
            Xi[i, r, seg] = Xi[i, r, seg] - delta
            lab[i, r, seg] = np.where(obs[i, r, seg], 1.0, np.nan)
    return Xi, lab


def run_unit(ua, cond, split):
    fam, r, cs = cond
    X, node, keep = ua["X"], ua["node"], ua["keep"]
    train_sets = F_SETS if fam == "F" else SETS["train"]
    tr = np.isin(ua["sname"], train_sets)
    va = ua["split"] == "val"
    ev = ua["split"] == split
    Wtr = np.isfinite(X[tr]).astype(float)
    if fam == "M":
        rng = np.random.default_rng(2000 + cs)
        Wtr = Wtr * (rng.random(Wtr.shape) >= r)
    xt = np.where(Wtr > 0, X[tr], np.nan)
    flat = xt.transpose(1, 0, 2).reshape(X.shape[1], -1)
    c, d = np.nanmean(flat, 1), np.nanstd(flat, 1)
    d = np.where(np.isfinite(d) & (d > 0), d, 1.0); c = np.where(np.isfinite(c), c, 0.0)
    z = lambda A: (A - c[None, :, None]) / d[None, :, None]  # noqa: E731
    # 주입 크기용 노드·행 학습 RSS SD (원 단위)
    sd_nd = {}
    for nd in np.unique(node):
        ix = np.flatnonzero((node == nd) & tr)
        v = np.nanstd(np.where(Wtr[np.isin(np.flatnonzero(tr), ix)] > 0, X[ix], np.nan).transpose(1, 0, 2).reshape(X.shape[1], -1), 1) \
            if len(ix) else np.full(X.shape[1], np.nan)
        sd_nd[nd] = np.where(np.isfinite(v) & (v > 0), v, d)
    sd_node = np.stack([sd_nd[n] for n in node[ev]])
    sets = [(scn, s, *inject(X[ev], keep[ev], sd_node, scn, s, k)) for k, scn in enumerate(SCEN) for s in SEEDS[split]]
    out = []
    for ref in REFS:
        lab_tr = np.full(int(tr.sum()), "ALL") if ref == "GM" else node[tr]
        lab_va = np.full(int(va.sum()), "ALL") if ref == "GM" else node[va]
        lab_ev = np.full(int(ev.sum()), "ALL") if ref == "GM" else node[ev]
        bank = make_bank(ref).fit(np.where(Wtr > 0, z(X[tr]), 0.0), Wtr, lab_tr)
        Wva = np.isfinite(X[va]).astype(float)
        bank.calibrate(np.where(Wva > 0, z(X[va]), 0.0), Wva, lab_va)
        for scn, s, Xi, lab in sets:
            A = bank.score(z(Xi), lab_ev)
            A = np.where(np.isfinite(lab), A, np.nan)
            ok = np.isfinite(lab) & np.isfinite(A)
            y, sc_ = lab[ok], A[ok]
            auc = float(average_precision_score(y, sc_)) if 0 < y.sum() < len(y) else np.nan
            far = float((sc_[y == 0] > bank.thr).mean()) if (y == 0).any() else np.nan
            out.append(dict(split=split, cond=fam, level="-" if r is None else r, cond_seed=cs, scenario=scn, seed=s,
                            ref=ref, auc_pr=auc, pos_rate=float(y.mean()), n_eval=int(len(y)), far_cell=far,
                            threshold=bank.thr, n_rows=int(X.shape[1]), train_missing=float(1 - Wtr.mean())))
    return out


def summarize(split):
    M = pd.read_csv(OUT / f"metrics_{split}.csv")
    key = ["unit", "scenario", "seed", "cond_seed"]
    info = json.loads((OUT / "tolerance.json").read_text(encoding="utf-8")) if (OUT / "tolerance.json").exists() else {}
    tol = info.get("tol", np.nan)
    L = [f"# 실험 R1-6 집계 ({split})", "", f"허용오차 {tol:.4f} ({info.get('rule', '')})", ""]

    def pair(a, b, cond, level=None):
        m = M[(M.cond == cond) & ((M.level.astype(str) == str(level)) if level is not None else True)]
        x = (m[m.ref == a].set_index(key).auc_pr - m[m.ref == b].set_index(key).auc_pr).dropna()
        v = "이김" if x.mean() > tol else ("짐" if x.mean() < -tol else "대등")
        return f"{a} − {b} [{cond}{'' if level is None else ' ' + str(level)}]: {x.mean():+.4f} (양 {(x > 0).mean():.2f}, n {len(x)}) **{v}**"

    tab = M.groupby(["cond", "level", "ref"]).auc_pr.mean().unstack("ref")
    L += ["## 평균 셀 AUC-PR", "", "```", tab.round(4).to_string(), "```", ""]
    L += ["## U1 노드 뱅크 효과", "- " + pair("PS", "GM", "N"), ""]
    L += ["## U2 자연 결측"] + ["- " + pair("PS", b, "N") for b in ("CG", "CF", "CI", "NM")] + [""]
    L += ["## U3·U4 추가 결측 r = 0.7"] + ["- " + pair("PS", b, "M", 0.7) for b in ("CG", "CF", "CI")] + [""]
    L += ["## (보고) r = 0.5"] + ["- " + pair("PS", b, "M", 0.5) for b in ("CG", "CF", "CI")] + [""]
    L += ["## U5 창 3개"] + ["- " + pair("PS", b, "F") for b in ("CG", "CI")] + [""]
    L += ["## U6 Huber", "- " + pair("PSH", "PS", "N"), ""]
    far = M.groupby(["cond", "ref"]).far_cell.mean().unstack("ref")
    L += ["## 실현 오경보율 (칸)", "", "```", far.round(4).to_string(), "```"]
    (OUT / f"summary_{split}.md").write_text("\n".join(L), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


def tolerance_from_val():
    """실험 1 규칙: tol = max(검증에서 CG 의 주입 seed 간 SD, 0.01). seed 마다 단위·시나리오 평균을 낸 뒤 seed 간 SD."""
    M = pd.read_csv(OUT / "metrics_val.csv")
    m = M[(M.cond == "N") & (M.ref == "CG")].groupby("seed").auc_pr.mean()
    sd = float(m.std(ddof=1))
    tol = max(sd, 0.01)
    info = dict(tol=tol, seed_sd=sd, rule="max(검증 CG 의 주입 seed 간 SD, 0.01), seed 별 단위·시나리오 평균의 SD",
                computed="검증 실행 직후, 테스트 전")
    (OUT / "tolerance.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def shift_far(log):
    """6절: 1개월째 학습(trn01–10) PS 모델로 이후 달 같은 기준점 세트의 정상 창 오경보율(서술)."""
    data, pidx = load_all(log)
    out = []
    for floor in (3, 5):
        ua = unit_arrays(data, floor, SETS["train"])
        tr = np.isin(ua["sname"], SETS["train"]); va = ua["split"] == "val"
        X = ua["X"]; Wtr = np.isfinite(X[tr]).astype(float)
        xt = np.where(Wtr > 0, X[tr], np.nan).transpose(1, 0, 2).reshape(X.shape[1], -1)
        c, d = np.nanmean(xt, 1), np.nanstd(xt, 1)
        d = np.where(np.isfinite(d) & (d > 0), d, 1.0); c = np.where(np.isfinite(c), c, 0.0)
        z = lambda A: (A - c[None, :, None]) / d[None, :, None]  # noqa: E731
        bank = make_bank("PS").fit(np.where(Wtr > 0, z(X[tr]), 0.0), Wtr, ua["node"][tr])
        Wva = np.isfinite(X[va]).astype(float)
        bank.calibrate(np.where(Wva > 0, z(X[va]), 0.0), Wva, ua["node"][va])
        keep_of = dict(zip(ua["node"], ua["keep"]))
        for month in range(1, 26):
            for nm in [f"tst{i:02d}" for i in range(1, 11)] + ["trn01", "trn02"]:
                r = U.windows(month, nm, pidx)
                if r is None:
                    continue
                Xm, node, fl = r
                ix = fl == floor
                if not ix.any():
                    continue
                Xm = Xm[ix][:, ua["rows"]]
                node = node[ix]
                kp = np.stack([keep_of.get(n, np.zeros(len(ua["rows"]), bool)) for n in node])
                A = bank.score(z(Xm), node)
                A = np.where(kp[:, :, None] & np.isfinite(Xm), A, np.nan)
                a = A[np.isfinite(A)]
                device = "A5" if (month == 25 and nm in ("trn02", "tst06", "tst07", "tst08", "tst09", "tst10")) else "S3"
                out.append(dict(floor=floor, month=month, set=nm, device=device, n_cells=int(a.size),
                                far_cell=float((a > bank.thr).mean()) if a.size else np.nan))
    pd.DataFrame(out).to_csv(OUT / "shift_far.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(out).groupby(["device", "month"]).far_cell.mean().round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "summarize", "shift"])
    ap.add_argument("--splits", nargs="*", default=["val"])
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = []
    if args.stage == "summarize":
        summarize(args.split); return
    if args.stage == "shift":
        shift_far(log); print("\n".join(log)); return
    data, _ = load_all(log)
    for spl in args.splits:
        t0 = time.perf_counter()
        rows = []
        for floor in (3, 5):
            for cond in CONDS:
                ua = unit_arrays(data, floor, F_SETS if cond[0] == "F" else SETS["train"])
                for r in run_unit(ua, cond, spl):
                    rows.append(dict(unit=f"F{floor}", **r))
            print(f"  F{floor} {spl} done ({time.perf_counter() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"metrics_{spl}.csv", index=False, encoding="utf-8-sig")
        if spl == "val":
            print(tolerance_from_val())
    (OUT / "load_log.txt").write_text("\n".join(log) or "모든 세트 사용", encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
