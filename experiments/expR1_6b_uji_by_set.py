"""실험 R1-6 사후 분석(탐색, 2026-10-29): 테스트 결과를 세트별로 나눈다.

등록 뒤에 읽은 데이터 논문(Mendoza-Silva et al., Data 2018, 3, 3, §2)에 따르면 Test-05 는 학습 세트와 같은 기준점이지만
다른 방향(좌·우)을 보고 측정했고, 학습·Test-01 은 위·아래 방향이다. 등록 문서가 '방문 a/b' 로 부른 두 블록은 이 두 방향이다.
그래서 등록대로 한 번 실행한 테스트(trn14, trn15, tst01, tst05)를 같은 주입으로 다시 계산해 세트별로 나눠 집계한다.
판정에 쓰지 않는다. 먼저 네 세트를 합친 값이 등록 실행(metrics_test.csv)과 같은지 확인한다.
사용법: python experiments/expR1_6b_uji_by_set.py
출력: results/expR1/uji/by_set_test.csv, by_set_test.md
"""
import os
import sys
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

import expR1_6_uji as U6  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = U6.OUT
GROUPS = {"same directions (trn14, trn15, tst01)": ["trn14", "trn15", "tst01"], "tst05 (other directions)": ["tst05"],
          "all four (registered)": ["trn14", "trn15", "tst01", "tst05"]}
CONDS = [("N", None, -1), ("M", 0.7, 0), ("M", 0.7, 1), ("M", 0.7, 2)]


def run_by_set(ua, cond, split="test"):
    """U6.run_unit 과 같은 계산(같은 표준화·주입·적합)을 하고 평가만 세트 묶음별로 나눈다."""
    fam, r, cs = cond
    X, node, keep = ua["X"], ua["node"], ua["keep"]
    tr = np.isin(ua["sname"], U6.SETS["train"])
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
    sd_nd = {}
    for nd in np.unique(node):
        ix = np.flatnonzero((node == nd) & tr)
        v = np.nanstd(np.where(Wtr[np.isin(np.flatnonzero(tr), ix)] > 0, X[ix], np.nan).transpose(1, 0, 2).reshape(X.shape[1], -1), 1) \
            if len(ix) else np.full(X.shape[1], np.nan)
        sd_nd[nd] = np.where(np.isfinite(v) & (v > 0), v, d)
    sd_node = np.stack([sd_nd[n] for n in node[ev]])
    sets = [(scn, s, *U6.inject(X[ev], keep[ev], sd_node, scn, s, k)) for k, scn in enumerate(U6.SCEN) for s in U6.SEEDS[split]]
    sname_ev = ua["sname"][ev]
    out = []
    for ref in U6.REFS:
        lab_tr = np.full(int(tr.sum()), "ALL") if ref == "GM" else node[tr]
        lab_va = np.full(int(va.sum()), "ALL") if ref == "GM" else node[va]
        lab_ev = np.full(int(ev.sum()), "ALL") if ref == "GM" else node[ev]
        bank = U6.make_bank(ref).fit(np.where(Wtr > 0, z(X[tr]), 0.0), Wtr, lab_tr)
        Wva = np.isfinite(X[va]).astype(float)
        bank.calibrate(np.where(Wva > 0, z(X[va]), 0.0), Wva, lab_va)
        for scn, s, Xi, lab in sets:
            A = bank.score(z(Xi), lab_ev)
            A = np.where(np.isfinite(lab), A, np.nan)
            for g, members in GROUPS.items():
                m = np.isin(sname_ev, members)
                ok = np.isfinite(lab[m]) & np.isfinite(A[m])
                y, sc = lab[m][ok], A[m][ok]
                auc = float(average_precision_score(y, sc)) if 0 < y.sum() < len(y) else np.nan
                far = float((sc[y == 0] > bank.thr).mean()) if (y == 0).any() else np.nan
                out.append(dict(cond=fam, level="-" if r is None else r, cond_seed=cs, scenario=scn, seed=s, ref=ref, group=g,
                                auc_pr=auc, far_cell=far, n_eval=int(len(y))))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    log = []
    data, _ = U6.load_all(log)
    rows = []
    for floor in (3, 5):
        for cond in CONDS:
            ua = U6.unit_arrays(data, floor, U6.SETS["train"])
            rows += [dict(unit=f"F{floor}", **r) for r in run_by_set(ua, cond)]
        print(f"F{floor} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "by_set_test.csv", index=False, encoding="utf-8-sig")
    # 등록 실행과 같은지
    reg = pd.read_csv(OUT / "metrics_test.csv")
    key = ["unit", "cond", "cond_seed", "scenario", "seed", "ref"]
    a = df[df.group == "all four (registered)"].copy()
    a["level"] = a.level.astype(str); reg["level"] = reg.level.astype(str)
    reg = reg[reg.cond.isin(["N", "M"]) & ((reg.cond == "N") | (reg.level == "0.7"))]
    mm = a.merge(reg[key + ["auc_pr", "far_cell"]], on=key, suffixes=("", "_reg"))
    d_auc = float((mm.auc_pr - mm.auc_pr_reg).abs().max()); d_far = float((mm.far_cell - mm.far_cell_reg).abs().max())
    L = ["# 실험 R1-6 사후 분석: 테스트 세트별 (탐색, 판정에 쓰지 않음)", "",
         f"재현 확인: 네 세트를 합친 값과 등록 실행의 차이 — AUC-PR 최대 {d_auc:.2e}, 오경보율 최대 {d_far:.2e} (짝 {len(mm)}개)", ""]
    for cond in ("N", "M"):
        sub = df[df.cond == cond]
        t = sub.groupby(["group", "ref"]).agg(auc=("auc_pr", "mean"), far=("far_cell", "mean")).unstack("ref")
        L += [f"## 조건 {cond}" + (" (+70 % 학습 칸 손실)" if cond == "M" else " (자연 결측)"), "", "평균 셀 AUC-PR", "",
              t["auc"].round(3).to_string(), "", "실현 셀 오경보율", "", t["far"].round(3).to_string(), ""]
        k2 = ["unit", "cond_seed", "scenario", "seed"]
        for g in GROUPS:
            gg = sub[sub.group == g]
            for a_, b_ in (("PS", "GM"), ("PS", "CG"), ("PS", "CI"), ("PS", "NM")):
                x = (gg[gg.ref == a_].set_index(k2).auc_pr - gg[gg.ref == b_].set_index(k2).auc_pr).dropna()
                L.append(f"- {g}: {a_} − {b_} {x.mean():+.4f} (양 {(x > 0).mean():.2f}, n {len(x)})")
        L.append("")
    (OUT / "by_set_test.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
