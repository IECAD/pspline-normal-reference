"""Mathematics 원고 표의 짝지은 ΔAUC-PR 에 대한 통계 (template V1 §2.4 '통계 검증'): n, 평균, 부트스트랩 95 % CI, Wilcoxon 부호순위 p, 양수 비율.
판정 규칙은 바꾸지 않는다 (사전 등록 = 허용오차 + 양수 비율). 이 수치는 보고용 열이다.
출력: paper/mathematics/paired_stats.csv, paired_stats.md
사용법: python analysis/63_math_paired_stats.py
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
rng = np.random.default_rng(0)
B = 5000
rows = []
PAIRS = []          # 짝 값 (최소 데이터셋용)


def stat(group, contrast, dataset, condition, a, b, key, df_a, df_b, col="auc_pr", note=""):
    """df_a, df_b: 같은 key 로 짝지을 수 있는 두 표. Δ = a − b."""
    ka = df_a.set_index(key)[col]; kb = df_b.set_index(key)[col]
    ka = ka[~ka.index.duplicated()]; kb = kb[~kb.index.duplicated()]
    common = ka.index.intersection(kb.index)
    d = (ka.loc[common] - kb.loc[common]).dropna().to_numpy()
    # 최소 데이터셋(MDPI 'Minimal dataset', 2026-10-14): 통계에 들어간 짝 값을 그대로 기록 (계산은 바꾸지 않음)
    pv = pd.DataFrame({"value_a": ka.loc[common].to_numpy(), "value_b": kb.loc[common].to_numpy()})
    pv.insert(0, "pair_key", ["|".join(f"{k}={v}" for k, v in zip(key, (ix if isinstance(ix, tuple) else (ix,)))) for ix in common])
    for c_, v_ in (("metric", col), ("model_b", b), ("model_a", a), ("condition", condition), ("dataset", dataset), ("contrast", contrast), ("group", group)):
        pv.insert(0, c_, v_)
    PAIRS.append(pv.dropna(subset=["value_a", "value_b"]))
    if len(d) < 3:
        rows.append(dict(group=group, contrast=contrast, dataset=dataset, condition=condition, n=len(d), mean=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_wilcoxon=np.nan, frac_pos=np.nan, note="n<3 " + note))
        return
    rng_c = np.random.default_rng(zlib.crc32(f"{group}|{contrast}|{dataset}|{condition}".encode("utf-8")))   # 대비별 고정 시드 (행 추가에 영향 없음)
    boot = np.array([rng_c.choice(d, len(d), replace=True).mean() for _ in range(B)])
    try:
        p = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = np.nan
    rows.append(dict(group=group, contrast=contrast, dataset=dataset, condition=condition, n=len(d), mean=d.mean(),
                     ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5), p_wilcoxon=p, frac_pos=(d > 0).mean(), note=note))


def d2multi(df):
    return df[df.scenario.astype(str).str.endswith("m")] if (df.dataset.iloc[0] == "d2") else df


# ---------------- A. 실험 6 결측·조건 (basis 끼리) ----------------------------------------------
KEY6 = ["space", "scenario", "seed", "family", "level", "cond_seed"]
for sub, fams in (("conditions", None), ("missing_patterns", None), ("imputation", None), ("phy_masks", None)):
    for ds in ("d1", "d2"):
        m = pd.read_csv(R / f"exp6/{sub}/metrics_{ds}_test.csv"); m["level"] = m.level.astype(str)
        m = d2multi(m)
        bases = [b for b in ("cell", "cellfill", "cellinterp") if b in m.basis.unique()]
        for fam, lev in m[["family", "level"]].drop_duplicates().itertuples(index=False):
            sel = m[(m.family == fam) & (m.level == lev)]
            ps = sel[sel.basis == "PS"]
            for b in bases:
                stat(f"exp6/{sub}", f"PS − {b}", ds, f"{fam} {lev}", "PS", b, KEY6, ps, sel[sel.basis == b])
# 2부 G (잡음), J (오염)
for fam_file in ("G", "J"):
    for ds in ("d1", "d2"):
        p = R / f"exp6/part2/metrics_{fam_file}_{ds}_test.csv"
        if not p.exists():
            continue
        m = pd.read_csv(p); m["level"] = m.level.astype(str); m = d2multi(m)
        for fam, lev in m[["family", "level"]].drop_duplicates().itertuples(index=False):
            sel = m[(m.family == fam) & (m.level == lev)]
            stat("exp6/part2", "PS − cell", ds, f"{fam} {lev}", "PS", "cell", KEY6, sel[sel.basis == "PS"], sel[sel.basis == "cell"])

# ---------------- B. 실험 1 비교 모델 (PS = C03) ------------------------------------------------
KEY1 = ["space", "scenario", "seed"]
for ds in ("d1", "d2"):
    ps = d2multi(pd.read_csv(R / f"C03/base/metrics_{ds}_test.csv"))
    for sub, selfile in (("", "selection.json"), ("batch2/", "batch2/selection.json"), ("batch2_fix/", None), ("batch2_fix2/", None), ("batch3/", None), ("batch3_fix/", None)):
        if not (R / f"exp1_baselines/{sub}metrics_test_{ds}.csv").exists():
            continue
        m = pd.read_csv(R / f"exp1_baselines/{sub}metrics_test_{ds}.csv"); m = d2multi(m)
        sel = json.load(open(R / f"exp1_baselines/{selfile}", encoding="utf-8"))["selection"] if selfile else {}
        for mdl in m.model.unique():
            t = m[m.model == mdl]
            s = sel.get(mdl, {}).get(ds)
            if s is not None:
                t = t[(t.variant == s["variant"]) & (t.cfg.apply(lambda x: json.loads(x) == s["cfg"]))]
            if len(t) == 0:
                continue
            tag = {"USAD": "학습 오류판(제외)", "USADFIX": "발산판(기록용)", "TRANAD": "3차", "ANOMTRANS": "3차 반대 방향판(제외)", "ANOMTRANSFIX": "3차 정정", "TRANADS": "3차 단일손실형(발산 대비)"}.get(mdl, "")
            stat("exp1_baselines", f"PS − {mdl}", ds, sub.strip("/") or "batch1", "PS", mdl, KEY1, ps, t, note=tag)
# 단순 기준선 사다리는 results/exp1_simple/paired_test.csv 에 Wilcoxon p 가 이미 있음 → 그대로 복사
pt = pd.read_csv(R / "exp1_simple/paired_test.csv")
for _, r in pt.iterrows():
    rows.append(dict(group="exp1_simple", contrast=r.contrast, dataset=r["view"], condition="", n=int(r.n), mean=r["mean"], ci_lo=np.nan, ci_hi=np.nan,
                     p_wilcoxon=r.wilcoxon_p, frac_pos=r.frac_pos, note="원 집계 (CI 없음)"))

# ---------------- C. ST-SSD 원형 변형 ------------------------------------------------------------
for ds in ("d1", "d2"):
    m = pd.read_csv(R / f"exp1_stssd/original/metrics_test_{ds}.csv"); m = d2multi(m)
    ps = m[(m.variant == "PS")]
    for v, bg in m[["variant", "background"]].drop_duplicates().itertuples(index=False):
        if v == "PS":
            continue
        t = m[(m.variant == v) & (m.background == bg)]
        stat("exp1_stssd/original", f"PS − {v}({bg}) cell", ds, "", "PS", v, KEY1, ps, t, col="auc_pr")
        stat("exp1_stssd/original", f"{v}({bg}) − PS window", ds, "", v, "PS", KEY1, t, ps, col="win_auc_pr")

# ---------------- D. 실험 10 D6 -------------------------------------------------------------------
KEY10 = ["scan", "scenario", "seed", "cond_seed"]
d6 = pd.read_csv(R / "exp10/metrics_test.csv")
for scale, lab in ((False, "main"), (True, "x3")):
    m = d6[d6.scenario.astype(str).str.contains("x") == scale]
    full = m[m.cond == "full"]
    for b in ("cell", "Z1", "cellfill", "OLS", "PCA"):
        stat("exp10", f"PS − {b}", "d6", f"{lab}, full", "PS", b, KEY10, full[full.model == "PS"], full[full.model == b])
    stat("exp10", "R − PS window", "d6", f"{lab}, full", "R", "PS", KEY10, full[full.model == "R"], full[full.model == "PS"], col="win_auc_pr")
    f01 = m[m.cond == "f0.1"]
    stat("exp10", "PS − cell", "d6", f"{lab}, f=0.1", "PS", "cell", KEY10, f01[f01.model == "PS"], f01[f01.model == "cell"])
for fn, ev in (("metrics_design_test.csv", "SD"), ("metrics_excl_design_test.csv", "outlier-excluded")):
    dd = pd.read_csv(R / f"exp10/design/{fn}")
    for scale, lab in ((False, "main"), (True, "x3")):
        m = dd[(dd.scenario.astype(str).str.contains("x") == scale) & (dd.cond == "full")]
        for a, b in (("PSH", "PS"), ("PS1", "PS"), ("PS1H", "PS"), ("PS1H", "Z1"), ("PS", "cell"), ("PS", "Z1")):
            if a in m.model.unique() and b in m.model.unique():
                stat("exp10/design", f"{a} − {b}", "d6", f"{ev}, {lab}", a, b, KEY10, m[m.model == a], m[m.model == b])

# ---------------- E. 결측 조건의 비교 모델 (실험 6 1부 추가 5; draft_v8 Table 6 = PS − 모델) ---------------
KEYC = ["space", "scenario", "seed", "cond_seed"]
for ds, rx in (("d1", r"^SC\d$"), ("d2", r"^SC\dm$")):
    p = R / f"exp6/comparators_missing/metrics_{ds}_test.csv"
    if not p.exists():
        continue
    m = pd.read_csv(p, encoding="utf-8-sig"); m = m[m.scenario.astype(str).str.match(rx)]; m["level"] = m.level.astype(str)
    m["cond"] = np.where(m.family == "base", "base", m.family + " " + m.level)
    for cond, sel in m.groupby("cond", sort=False):
        ps = sel[sel.model == "PS"]
        for mdl in sel.model.unique():
            if mdl != "PS":
                stat("exp6/comparators_missing", f"PS − {mdl}", ds, cond, "PS", mdl, KEYC, ps, sel[sel.model == mdl])

# ---------------- F. 노드별 bank 효과 Z1 − Z0 (draft_v8 Table 3(a); 이전 원고의 CI 는 추적 불가 값이라 재계산) ---------
for ds, rx in (("d1", r"^SC\d$"), ("d2", r"^SC\dm$")):
    z1 = pd.read_csv(R / f"Z1/base/metrics_{ds}_test.csv"); z0 = pd.read_csv(R / f"Z0/base/metrics_{ds}_test.csv")
    z1 = z1[z1.scenario.astype(str).str.match(rx)]; z0 = z0[z0.scenario.astype(str).str.match(rx)]
    stat("exp1_simple/boot", "Z1 − Z0 (node bank)", ds, "", "Z1", "Z0", KEY1, z1, z0)

# ---------------- 출력 ---------------------------------------------------------------------------
df = pd.DataFrame(rows)
df.to_csv(ROOT / "paper/mathematics/paired_stats.csv", index=False, encoding="utf-8-sig")
L = ["# 짝지은 ΔAUC-PR 통계 (자동 생성: `analysis/63_math_paired_stats.py`)", "",
     f"부트스트랩 95 % CI (B={B}, seed 0, 백분위법), Wilcoxon 부호순위 양측 p. 판정은 사전 등록 규칙(허용오차·양수 비율) 그대로이며 이 열은 보고용.", ""]
for g, sub in df.groupby("group", sort=False):
    L += [f"## {g}", "", "| 대비 | 데이터 | 조건 | n | 평균 Δ | 95 % CI | p | 양수 비율 | 비고 |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in sub.iterrows():
        ci = "—" if np.isnan(r.ci_lo) else f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"
        pv = "—" if np.isnan(r.p_wilcoxon) else (f"{r.p_wilcoxon:.1e}" if r.p_wilcoxon < 1e-3 else f"{r.p_wilcoxon:.3f}")
        L.append(f"| {r.contrast} | {r.dataset} | {r.condition} | {r.n} | {r['mean']:+.4f} | {ci} | {pv} | {r.frac_pos:.2f} | {r.note} |")
    L.append("")
(ROOT / "paper/mathematics/paired_stats.md").write_text("\n".join(L) + "\n", encoding="utf-8")
pd.concat(PAIRS, ignore_index=True).to_csv(ROOT / "paper/mathematics/paired_values.csv", index=False, encoding="utf-8-sig")   # 75_minimal_dataset.py 가 영문화해 묶음
sys.stdout.reconfigure(encoding="utf-8")
print(f"rows {len(df)}; written paired_stats.csv/.md, paired_values.csv ({sum(len(p) for p in PAIRS)} pairs)")
print(df[df.group.isin(["exp10", "exp10/design"])].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
