"""실험 R1-7: 새 무필터 UWB 데이터(HYMN)에서 Huber 변형과 견고 기준선 확인 (doc/실험R1-7_계획.md, 사전 등록 2026-10-29).

노드 = 지점 48, 행 = 앵커 10개 거리, 창 = 라운드 12회, 단위 하나. 연결 앵커 = 자연 학습 관측 ≥ 12.
참조(탐지기 self): CG CI CMI NM NMed PS PS1 PSH PS1H (R1-4 2부와 같은 정의·코드).
조건: N 자연 · M 학습 칸 MCAR r 0.7 (조건 seed 0–2, rng 2000 + seed). 주입: D1 SC1–SC3 주 크기와 ×3.
평가: 전체(주) · 이상값 제외(탐색, 자연 학습 링크 중앙값에서 1 m 넘는 칸을 주입 전 원값으로 정함).

사용법:
  python experiments/expR1_7_hymn.py run --split val        # 검증 → tolerance → summarize val
  python experiments/expR1_7_hymn.py tolerance
  python experiments/expR1_7_hymn.py run --split test       # 테스트 한 번
  python experiments/expR1_7_hymn.py summarize --split val|test
  python experiments/expR1_7_hymn.py bootstrap --split test
  python experiments/expR1_7_hymn.py smoke                  # 합성 난수 데이터로 코드 점검만 (HYMN 값을 쓰지 않음)
출력: results/expR1/hymn/ (smoke 는 results/expR1/hymn_smoke/)
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

import exp10_d6 as X10  # noqa: E402
from expR1_4_robust import REFS_D6 as REFS, build_ref  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.inject import inject_windows  # noqa: E402
from src.data.load import load_windows  # noqa: E402
from src.model.bank import anchor_scores  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = ROOT / "results" / "expR1" / "hymn"
MIN_TRAIN_OBS, MIN_WIN_FRAC, OUTLIER_M, MAG_SCALE = 12, 0.5, 1.0, 3.0
OUTLIER_LINK_FRAC = 0.10
R_MISS = 0.7
CONDS = [("N", -1), ("M", 0), ("M", 1), ("M", 2)]
MAIN = ["SC1", "SC2", "SC3"]
B_BOOT, BOOT_SEED = 2000, 20261029
CMP_N = [("PSH", "PS"), ("CMI", "CI"), ("PSH", "CMI"), ("PS1H", "NMed"),
         ("PS", "CG"), ("PS", "CI"), ("PS", "NM"), ("PS1", "PS")]
CMP_M = [("PS", "CG"), ("PS", "CI"), ("PSH", "CMI"), ("PSH", "PS"), ("CMI", "CI")]
JUDGED = {("N", "PSH", "PS"): "H-H1 이김", ("N", "CMI", "CI"): "H-H2 이김", ("N", "PSH", "CMI"): "H-H3 대등",
          ("N", "PS1H", "NMed"): "H-H4 대등", ("M", "PS", "CG"): "H-M1 이김",
          ("M", "PS", "CI"): "H-M2 대등 또는 작은 이김(|Δ| ≤ 0.03)", ("M", "PSH", "CMI"): "H-M3 짐 아님"}


# ---------------------------------------------------------------- 데이터
def load_data(smoke=False):
    dcfg = load_config()
    if not smoke:
        w = load_windows("d8", "hymn", dcfg)
        return w["X"].astype(float), w["rows"], w["meta"], w["cells"]["mask"].astype(float)
    # 합성 난수: HYMN 의 창 구조(meta)만 쓰고 값은 쓰지 않는다
    meta = pd.read_parquet(dcfg["paths"]["cache"] / "windows" / "d8" / "hymn_meta.parquet")
    rng = np.random.default_rng(1)
    N, S, T = len(meta), 10, 12
    nodes = meta.node.to_numpy()
    base = {nd: rng.uniform(2, 30, S) for nd in np.unique(nodes)}
    X = np.stack([base[n] for n in nodes])[:, :, None] + rng.normal(0, 0.1, (N, S, T))
    X += (rng.random((N, S, T)) < 0.05) * np.abs(rng.standard_t(3, (N, S, T)))
    M = (rng.random((N, S, T)) > 0.07).astype(float)
    return np.where(M > 0, X, 0.0), [f"DIST_A{k}" for k in range(1, S + 1)], meta, M


def structure(X, M, meta):
    """연결 앵커 (N, A), 이상값 제외 칸 (N, S, T), 이상값 링크 (N, A) — 모두 자연 학습 칸으로 정한다."""
    nodes = meta.node.to_numpy()
    tr = meta.split.to_numpy() == "train"
    keep = np.zeros(M.shape[:2], bool)
    clean = np.ones(X.shape, bool)
    out_link = np.zeros(M.shape[:2], bool)
    for nd in np.unique(nodes):
        ixn = np.flatnonzero(nodes == nd)
        itr = np.flatnonzero((nodes == nd) & tr)
        keep[ixn] = M[itr].sum((0, 2)) >= MIN_TRAIN_OBS
        x = np.where(M[itr] > 0, X[itr], np.nan)
        med = np.nanmedian(x, axis=(0, 2))
        clean[ixn] = ~((M[ixn] > 0) & (np.abs(X[ixn] - med[None, :, None]) > OUTLIER_M))
        dev = np.abs(x - med[None, :, None]) > OUTLIER_M
        nobs = np.isfinite(x).sum((0, 2))
        frac = np.where(nobs > 0, (dev & np.isfinite(x)).sum((0, 2)) / np.maximum(nobs, 1), 0.0)
        out_link[ixn] = frac >= OUTLIER_LINK_FRAC
    return keep, clean, out_link


def auc(A, lab, m=None):
    ok = np.isfinite(A) & np.isfinite(lab) & (True if m is None else m)
    y, s = lab[ok], A[ok]
    return (float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan), int(len(y)), int(y.sum())


# ---------------------------------------------------------------- 실행
def run(split, smoke=False):
    out_dir = OUT.parent / "hymn_smoke" if smoke else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    X, rows, meta, M = load_data(smoke)
    nodes, spl = meta.node.to_numpy(), meta.split.to_numpy()
    kind = meta.kind.to_numpy()
    tr, va = spl == "train", spl == "val"
    ix = np.flatnonzero(spl == split)
    anchors = [r.split("_", 1)[1] for r in rows]
    keep, clean, out_link = structure(X, M, meta)
    E = ecfg["datasets"]["d1"]
    scen = [(k, "x1", sc) for k, sc in E["scenarios"].items()] + \
           [(k, f"x{MAG_SCALE:g}", X10.scaled(sc, MAG_SCALE)) for k, sc in E["scenarios"].items()]
    Wa = M[ix]
    elig = keep[ix] & (Wa.mean(2) >= MIN_WIN_FRAC)
    inj = {}
    for scn, size, sc in scen:
        for s in E["injection_seeds"][split]:
            Xi, lab, _, _ = inject_windows(X[ix], rows, sc, s, dist_metric="DIST", eligible=elig)
            inj[(scn, size, s)] = (Xi, np.where(keep[ix][:, :, None] & (Wa > 0), lab, np.nan))
    groups = {"link_outlier": np.repeat(out_link[ix][:, :, None], 12, 2),
              "link_regular": np.repeat(~out_link[ix][:, :, None], 12, 2),
              "kind_grid": np.broadcast_to((kind[ix] == "grid")[:, None, None], Wa.shape),
              "kind_transition": np.broadcast_to((kind[ix] == "transition")[:, None, None], Wa.shape)}
    rec, sub, store = [], [], {}
    for (scn, size, s), (_, lab) in inj.items():
        store[f"lab|{scn}|{size}|{s}"] = lab.astype(np.float32)
    t_all = time.perf_counter()
    for cond, cs in CONDS:
        Wtr = M[tr].copy()
        if cond == "M":
            rng = np.random.default_rng(2000 + cs)
            Wtr = Wtr * (rng.random(Wtr.shape) >= R_MISS)
        for ref in REFS:
            t0 = time.perf_counter()
            comp, bank = build_ref(ref, ecfg, "d8", X[tr], Wtr, rows, nodes[tr], X[va], M[va], nodes[va])
            for (scn, size, s), (Xi, lab) in inj.items():
                Xs, Ws = comp.transform(Xi, Wa, rows)
                A = X10.mask_scores(anchor_scores(bank.score(Xs, nodes[ix], Ws), comp.anchor_of_row, anchors), keep[ix], Wa)
                store[f"{cond}|{cs}|{scn}|{size}|{s}|{ref}"] = A.astype(np.float32)
                for ev_name, cm in (("full", np.ones_like(clean[ix])), ("excl", clean[ix])):
                    lk, Ak = np.where(cm, lab, np.nan), np.where(cm, A, np.nan)
                    ev = X10.evaluate_light([(Ak[n], lk[n]) for n in range(len(Ak))], bank.thr)
                    ok = np.isfinite(Ak) & np.isfinite(lk)
                    far = float((Ak[ok & (lk == 0)] > bank.thr).mean()) if (ok & (lk == 0)).any() else np.nan
                    rec.append(dict(split=split, cond=cond, cond_seed=cs, scenario=scn, size=size, seed=s, ref=ref,
                                    eval=ev_name, far_cell=far, threshold=bank.thr, train_missing=float(1 - Wtr.mean()),
                                    pos_before=int(np.nansum(lab == 1)), neg_before=int(np.nansum(lab == 0)),
                                    pos_after=int(np.nansum(lk == 1)), neg_after=int(np.nansum(lk == 0)), **ev))
                for g, gm in groups.items():
                    a, n, p = auc(A, lab, gm)
                    sub.append(dict(split=split, cond=cond, cond_seed=cs, scenario=scn, size=size, seed=s, ref=ref,
                                    group=g, auc_pr=a, n_eval=n, n_pos=p))
            print(f"  {cond}{cs if cs >= 0 else ''} {ref}: {time.perf_counter() - t0:.1f}s", flush=True)
        pd.DataFrame(rec).to_csv(out_dir / f"metrics_{split}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(sub).to_csv(out_dir / f"groups_{split}.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(out_dir / f"scores_{split}.npz", nodes=nodes[ix].astype(str), **store)
    link = pd.DataFrame(dict(node=np.unique(nodes)))
    link["n_keep"] = [int(keep[nodes == n][0].sum()) for n in link.node]
    link["n_outlier_links"] = [int(out_link[nodes == n][0].sum()) for n in link.node]
    link.to_csv(out_dir / "links.csv", index=False)
    print(f"done {time.perf_counter() - t_all:.0f}s", flush=True)
    return out_dir


# ---------------------------------------------------------------- 허용오차·집계
def tolerance(out_dir=OUT):
    m = pd.read_csv(out_dir / "metrics_val.csv")
    m = m[(m.cond == "N") & (m["size"] == "x1") & (m["eval"] == "full") & (m.ref == "CG")]
    sd = float(m.groupby("seed").auc_pr.mean().std(ddof=1))
    info = dict(tol=max(sd, 0.01), seed_sd=sd, rule="max(검증·조건 N·주 크기·전체 평가 CG 의 주입 seed 간 SD, 0.01); seed 별 시나리오 평균의 SD",
                computed="검증 실행 직후, 테스트 전")
    (out_dir / "tolerance.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def md(df, fmt="{:.4f}"):
    cols = [str(c) for c in df.columns]
    L = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        L.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]) + " |")
    return "\n".join(L)


def pair_diff(m, a, b):
    key = ["scenario", "seed", "cond_seed"]
    return (m[m.ref == a].set_index(key).auc_pr - m[m.ref == b].set_index(key).auc_pr).dropna()


def verdict(x, tol):
    return "이김" if x > tol else ("짐" if x < -tol else "대등")


def summarize(split, out_dir=OUT):
    M = pd.read_csv(out_dir / f"metrics_{split}.csv")
    tinfo = json.loads((out_dir / "tolerance.json").read_text(encoding="utf-8")) if (out_dir / "tolerance.json").exists() else {}
    tol = tinfo.get("tol", np.nan)
    boot = pd.read_csv(out_dir / f"bootstrap_{split}.csv") if (out_dir / f"bootstrap_{split}.csv").exists() else None
    L = [f"# 실험 R1-7 집계 ({split})", "", f"허용오차 {tol:.4f} — {tinfo.get('rule', '(아직 없음)')}", ""]
    main = M[M.scenario.isin(MAIN)]
    for size in ("x1", f"x{MAG_SCALE:g}"):
        for ev in ("full", "excl"):
            m = main[(main["size"] == size) & (main["eval"] == ev)]
            tab = m.groupby(["cond", "ref"]).auc_pr.mean().unstack("ref")[REFS]
            tab.index.name = "cond"
            L += [f"## 평균 셀 AUC-PR — 크기 {size}, {'전체 평가' if ev == 'full' else '이상값 제외(탐색)'}", "", md(tab), ""]
            L += [f"### 비교 (크기 {size}, {ev})", ""]
            for cond, cmps in (("N", CMP_N), ("M", CMP_M)):
                mc = m[m.cond == cond]
                for a, b in cmps:
                    x = pair_diff(mc, a, b)
                    tag = JUDGED.get((cond, a, b)) if (size == "x1" and ev == "full") else None
                    ci = ""
                    if boot is not None and size == "x1" and ev == "full":
                        bb = boot[(boot.cond == cond) & (boot.a == a) & (boot.b == b)]
                        if len(bb):
                            ci = f", 지점 군집 95 % [{bb.lo.iloc[0]:+.4f}, {bb.hi.iloc[0]:+.4f}]"
                    L.append(f"- [{cond}] {a} − {b}: {x.mean():+.4f} (양 {(x > 0).mean():.2f}, n {len(x)}{ci}) "
                             f"**{verdict(x.mean(), tol)}**" + (f" — 예상 {tag}" if tag else ""))
            L.append("")
    ex = M[(M["eval"] == "excl") & (M.cond == "N") & (M.ref == "PS")]
    for size in ("x1", f"x{MAG_SCALE:g}"):
        e = ex[(ex["size"] == size) & ex.scenario.isin(MAIN)]
        pb, pa, nb, na = e.pos_before.sum(), e.pos_after.sum(), e.neg_before.sum(), e.neg_after.sum()
        L.append(f"- 이상값 제외 칸 (크기 {size}, 시나리오·seed 합): 양성 {pb:,} → {pa:,} ({1 - pa / pb:.1%} 제외), "
                 f"음성 {nb:,} → {na:,} ({1 - na / nb:.1%} 제외)")
    far = main[(main["size"] == "x1") & (main["eval"] == "full")].groupby(["cond", "ref"]).far_cell.mean().unstack("ref")[REFS]
    far.index.name = "cond"
    L += ["", "## 실현 셀 오경보율 (운영 임계, 크기 x1, 전체 평가)", "", md(far), ""]
    gpath = out_dir / f"groups_{split}.csv"
    if gpath.exists():
        G = pd.read_csv(gpath)
        G = G[G.scenario.isin(MAIN) & (G["size"] == "x1")]
        L += ["## 링크 종류·노드 종류별 (보고, 크기 x1, 전체 평가)", ""]
        for g in ("link_outlier", "link_regular", "kind_grid", "kind_transition"):
            gg = G[G.group == g]
            for cond, cmps in (("N", [("PSH", "PS"), ("CMI", "CI"), ("PS", "CG")]), ("M", [("PS", "CG"), ("PSH", "PS")])):
                for a, b in cmps:
                    x = pair_diff(gg[gg.cond == cond], a, b)
                    L.append(f"- {g} [{cond}] {a} − {b}: {x.mean():+.4f} (양 {(x > 0).mean():.2f}, n {len(x)}; "
                             f"칸 {int(gg[(gg.cond == cond) & (gg.ref == a)].n_eval.mean()):,})")
        L.append("")
    (out_dir / f"summary_{split}.md").write_text("\n".join(L), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


# ---------------------------------------------------------------- 군집 부트스트랩 (지점)
def _prep_ap(A, lab, pidx):
    ok = np.isfinite(A) & np.isfinite(lab)
    s, y, p = A[ok], lab[ok], pidx[ok]
    o = np.argsort(-s, kind="mergesort")
    s, y, p = s[o], y[o], p[o]
    thr_idx = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    return y, p, thr_idx


def _ap_w(y, p, thr_idx, cnt):
    w = cnt[p]
    tps = np.cumsum(w * y)[thr_idx]
    fps = np.cumsum(w * (1 - y))[thr_idx]
    if tps[-1] <= 0:
        return np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tps + fps > 0, tps / (tps + fps), 0.0)
    rec = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def bootstrap(split, out_dir=OUT, B=B_BOOT):
    z = np.load(out_dir / f"scores_{split}.npz", allow_pickle=False)
    nodes = z["nodes"]
    pts, pinv = np.unique(nodes, return_inverse=True)
    npnt = len(pts)
    seeds = sorted({int(k.split("|")[3]) for k in z.files if k.startswith("lab|")})
    rng = np.random.default_rng(BOOT_SEED)
    cnts = np.stack([np.bincount(rng.integers(0, npnt, npnt), minlength=npnt) for _ in range(B)]).astype(float)
    out, chk = [], []
    for cond, cmps in (("N", CMP_N), ("M", CMP_M)):
        css = [-1] if cond == "N" else [0, 1, 2]
        refs = sorted({r for ab in cmps for r in ab})
        prep, base = {}, {}
        for cs in css:
            for scn in MAIN:
                for s in seeds:
                    lab = z[f"lab|{scn}|x1|{s}"]
                    pidx = np.broadcast_to(pinv[:, None, None], lab.shape)
                    for r in refs:
                        A = z[f"{cond}|{cs}|{scn}|x1|{s}|{r}"]
                        prep[(cs, scn, s, r)] = _prep_ap(A, lab, pidx)
                        base[(cs, scn, s, r)] = auc(A, lab)[0]
        for k, v in prep.items():                                  # 가중치 1 이면 sklearn 값과 같아야 한다
            chk.append(abs(_ap_w(*v, np.ones(npnt)) - base[k]))
        for a, b in cmps:
            est = np.mean([base[(cs, scn, s, a)] - base[(cs, scn, s, b)] for cs in css for scn in MAIN for s in seeds])
            bs = np.empty(B)
            for i in range(B):
                bs[i] = np.nanmean([_ap_w(*prep[(cs, scn, s, a)], cnts[i]) - _ap_w(*prep[(cs, scn, s, b)], cnts[i])
                                    for cs in css for scn in MAIN for s in seeds])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            out.append(dict(split=split, cond=cond, a=a, b=b, est=est, lo=lo, hi=hi, B=B, clusters=npnt))
            print(f"  [{cond}] {a} − {b}: {est:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
    pd.DataFrame(out).to_csv(out_dir / f"bootstrap_{split}.csv", index=False)
    print(f"가중 AP 점검: 최대 차이 {max(chk):.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "tolerance", "summarize", "bootstrap", "smoke"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--B", type=int, default=B_BOOT)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.stage == "run":
        run(args.split)
        if args.split == "val" and not (OUT / "tolerance.json").exists():
            print(tolerance())
        summarize(args.split)
    elif args.stage == "tolerance":
        print(tolerance())
    elif args.stage == "summarize":
        summarize(args.split)
    elif args.stage == "bootstrap":
        bootstrap(args.split, B=args.B)
        summarize(args.split)
    else:
        d = run("val", smoke=True)
        print(tolerance(d))
        bootstrap("val", d, B=50)
        summarize("val", d)


if __name__ == "__main__":
    main()
