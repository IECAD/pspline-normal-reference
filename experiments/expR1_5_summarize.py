"""실험 R1-5 요약·판정 (doc/실험R1-5_계획.md 5·6절 규칙).

짝 차이 = 같은 반복에서 두 참조의 테스트 셀 AUC-PR 차이. tol = max(0.01, 2·SE). 이김 / 대등 / 짐.
등록 문서가 수치로 정하지 않은 문구는 아래 OPS 에 조작적 정의를 적고 요약에도 그대로 밝힌다.
출력: results/expR1/sim/summary.md, contrasts.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/expR1/sim"
OPS = {
    "S2_recover": "CF·CI 가 '격차 대부분을 메움' = (X − CG) ≥ 0.5 · (PS − CG), X ∈ {CF, CI}",
    "S2_trend": "'N 이 작고 r 이 클수록 커짐' = G1 격자에서 PS − CI 평균이 N 감소·r 증가 방향으로 단조인지 서술",
}


def md(df):
    """DataFrame -> 마크다운 표 (tabulate 없이)."""
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return chr(10).join(out)


def load():
    d = pd.read_csv(OUT / "metrics.csv")
    extra = sorted(OUT.glob("cells*_metrics.csv"))
    if extra:
        d = pd.concat([d] + [pd.read_csv(p) for p in extra], ignore_index=True)
    cells = json.loads((OUT / "design.json").read_text(encoding="utf-8"))
    return d, {c["id"]: c for c in cells}


def contrast(d, cell, a, b, col="auc_pr"):
    x = d[d.cell == cell].pivot(index="rep", columns="ref", values=col)
    diff = (x[a] - x[b]).dropna()
    n = len(diff)
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    tol = max(0.01, 2 * se) if np.isfinite(se) else 0.01
    verdict = "이김" if mean > tol else ("짐" if mean < -tol else "대등")
    return dict(cell=cell, a=a, b=b, n=n, mean=mean, se=se, tol=tol, verdict=verdict,
                frac_pos=float((diff > 0).mean()))


def find(cells, **kw):
    out = [cid for cid, c in cells.items() if all(c["cfg"].get(k) == v for k, v in kw.items())]
    if len(out) != 1:
        raise KeyError((kw, out))
    return out[0]


def main():
    d, cells = load()
    rows, lines = [], []
    B = dict(N=10, L="L1", noise="G", miss="MCAR", r=0.0, p=0.0, atype="mix", mag=2.0)

    def C(a, b, **kw):
        cfg = dict(B)
        cfg.update(kw)
        cid = find(cells, **cfg)
        r = contrast(d, cid, a, b)
        r.update({k: cfg[k] for k in ("N", "L", "noise", "miss", "r", "p", "atype", "mag")})
        rows.append(r)
        return r

    def fmt(r):
        return f"{r['a']} − {r['b']} = {r['mean']:+.4f} (SE {r['se']:.4f}, tol {r['tol']:.4f}, 양 {r['frac_pos']:.2f}) **{r['verdict']}**"

    lines.append("# 실험 R1-5 요약 (시뮬레이션)\n")
    lines.append(f"반복 수: 설계 칸당 {d.groupby('cell').rep.nunique().min()}–{d.groupby('cell').rep.nunique().max()}, 설계 칸 {d.cell.nunique()}개. 판정: 짝 차이 평균 대 tol = max(0.01, 2 SE).\n")
    lines.append("조작적 정의(등록 문서가 수치로 정하지 않은 문구, 실행 뒤 요약에서 정함 — 판정에 쓰는 곳마다 밝힘):")
    for k, v in OPS.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    # S1
    lines.append("## S1 기본 조건, 결측 없음: PS ≈ CG ≈ CF ≈ CI, L1·L2 에서 NM 은 짐")
    s1 = [C("PS", x) for x in ("CG", "CF", "CI")]
    s1n = [C("PS", "NM", L=L) for L in ("L1", "L2")]
    for r in s1 + s1n:
        lines.append(f"- [{r['L']}] {fmt(r)}")
    ok1 = all(r["verdict"] == "대등" for r in s1) and all(r["verdict"] == "이김" for r in s1n)
    lines.append(f"- 판정: {'예상과 같음' if ok1 else '예상과 다름'}\n")

    # S2
    lines.append("## S2 MCAR r ≥ 0.5, N ≤ 5")
    ok2 = True
    for N in (3, 5):
        for r_ in (0.5, 0.7):
            a = C("PS", "CG", N=N, r=r_)
            ci = C("PS", "CI", N=N, r=r_)
            cf = C("PS", "CF", N=N, r=r_)
            gap = a["mean"]
            rec_ci = (gap - ci["mean"]) / gap if gap > 0 else np.nan
            rec_cf = (gap - cf["mean"]) / gap if gap > 0 else np.nan
            lines.append(f"- N {N}, r {r_}: {fmt(a)}; {fmt(cf)}; {fmt(ci)}; 회복 비율 CF {rec_cf:.2f}, CI {rec_ci:.2f}")
            ok2 &= a["verdict"] == "이김" and rec_ci >= 0.5 and rec_cf >= 0.5 and ci["mean"] > 0
    lines.append(f"- 판정(PS − CG 이김, CF·CI 회복 ≥ 0.5, PS − CI > 0): {'예상과 같음' if ok2 else '예상과 다름'}")
    lines.append("- G1 격자 PS − CI 평균 (행 N, 열 r):")
    g = {}
    for N in (3, 5, 10, 20):
        g[N] = {r_: C("PS", "CI", N=N, r=r_)["mean"] for r_ in (0.0, 0.3, 0.5, 0.7)}
    lines.append("\n| N \\ r | 0 | 0.3 | 0.5 | 0.7 |\n|---|---|---|---|---|")
    for N in (3, 5, 10, 20):
        lines.append(f"| {N} | " + " | ".join(f"{g[N][r_]:+.4f}" for r_ in (0.0, 0.3, 0.5, 0.7)) + " |")
    lines.append("")

    # S3
    lines.append("## S3 L2, N = 3, r = 0: PS − CI < 0 가능(평활 바이어스)")
    r3 = C("PS", "CI", L="L2", N=3) if any(c["cfg"] == dict(B, L="L2", N=3) for c in cells.values()) else None
    if r3 is None:
        lines.append("- 설계에 (L2, N = 3) 칸이 없다 → G2 의 (L2, N = 10, r = 0) 로 대신 보고하고 평가 불가로 적는다.")
        r3 = C("PS", "CI", L="L2")
        lines.append(f"- [L2, N 10] {fmt(r3)}")
    else:
        lines.append(f"- {fmt(r3)}")
    lines.append("")

    # S4
    lines.append("## S4 L0 평탄: NM ≥ PS(d = 2), PS1 ≈ NM")
    a = C("NM", "PS", L="L0")
    b = C("PS1", "NM", L="L0")
    lines.append(f"- {fmt(a)}\n- {fmt(b)}")
    ok4 = a["verdict"] in ("이김", "대등") and b["verdict"] == "대등"
    lines.append(f"- 판정: {'예상과 같음' if ok4 else '예상과 다름'}\n")

    # S5
    lines.append("## S5 두꺼운 꼬리 (T3, CN): 견고 참조가 낫다, L1 에서 PSH ≥ CMI")
    ok5 = True
    for nz in ("T3", "CN"):
        x1 = C("CMI", "CI", noise=nz)
        x2 = C("PSH", "PS", noise=nz)
        x3 = C("PSH", "CMI", noise=nz)
        x4 = C("NMed", "NM", noise=nz)
        for r in (x1, x2, x3, x4):
            lines.append(f"- [{nz}] {fmt(r)}")
        ok5 &= x1["verdict"] == "이김" and x2["verdict"] == "이김" and x3["verdict"] in ("이김", "대등")
    lines.append(f"- 판정(CMI − CI 이김, PSH − PS 이김, PSH − CMI 이김 또는 대등): {'예상과 같음' if ok5 else '예상과 다름'}")
    ok5b = True
    for nz in ("T3", "CN"):
        try:
            r = C("PS1H", "NMed", L="L0", noise=nz)
        except KeyError:
            lines.append(f"- [L0, {nz}] 보완 칸 결과 없음")
            ok5b = False
            continue
        lines.append(f"- [L0, {nz}] (보완 칸) {fmt(r)}")
        ok5b &= r["verdict"] == "대등"
    lines.append(f"- 판정(L0 에서 PS1H ≈ NMed): {'예상과 같음' if ok5b else '예상과 다름'}\n")

    # S6
    lines.append("## S6 학습 오염: 제곱 손실 참조가 같이 나빠지고 Huber 는 덜 나빠진다")
    base = d[d.cell == find(cells, **B)].groupby("ref").auc_pr.mean()
    for p in (0.1, 0.3):
        cur = d[d.cell == find(cells, **dict(B, p=p))].groupby("ref").auc_pr.mean()
        drop = (base - cur)
        lines.append(f"- p {p}: AUC-PR 하락 " + ", ".join(f"{k} {drop[k]:+.4f}" for k in ("CG", "CF", "CI", "PS", "PSH", "PS1H", "CMI")))
    lines.append("")

    # S7
    lines.append("## S7 MNAR: 참조 편향·σ_R 증가, PS 와 CF·CI 대등")
    cid = find(cells, **dict(B, miss="MNAR"))
    sub = d[d.cell == cid].groupby("ref")[["rmse_obs", "sdR_ratio", "auc_pr"]].mean()
    lines.append(md(sub.round(4)))
    for x in ("CF", "CI"):
        r = C("PS", x, miss="MNAR")
        lines.append(f"- {fmt(r)}")
    lines.append("")

    # S8
    lines.append("## S8 M5 예측 분산 식 (빈 칸·관측 칸)")
    m5 = pd.read_csv(OUT / "m5_predvar.csv")
    t5 = m5.groupby(["lam", "empty"]).ratio.describe()[["count", "mean", "25%", "50%", "75%", "min", "max"]]
    lines.append(md(t5.round(3)))
    fixed = m5[m5.lam != "sap"]
    lo, hi = 1 - 3 * np.sqrt(2 / 499), 1 + 3 * np.sqrt(2 / 499)
    frac_in = float(((fixed.ratio > lo) & (fixed.ratio < hi)).mean())
    sap = m5[m5.lam == "sap"]
    lines.append(f"- λ 고정: 비가 몬테카를로 ±3 SE 구간 [{lo:.3f}, {hi:.3f}] 안인 비율 {frac_in:.3f}")
    sap_e, sap_o = sap[sap["empty"]].ratio.median(), sap[~sap["empty"]].ratio.median()
    lines.append(f"- λ̂(SAP): 비 중앙값 빈 칸 {sap_e:.3f}, 관측 칸 {sap_o:.3f}")
    emp = m5[(m5.lam == "1.0") & m5["empty"]]
    obs = m5[(m5.lam == "1.0") & ~m5["empty"]]
    lines.append(f"- λ = 1 에서 식의 분산 중앙값: 빈 칸 {emp.var_formula.median():.3f}σ², 관측 칸 {obs.var_formula.median():.3f}σ²\n")

    # S9
    lines.append("## S9 오경보율 (서술)")
    far = d[d.cell == find(cells, **B)].groupby("ref")[["val_far_cell", "far_cell", "val_far_grp_a", "far_grp_a", "far_grp_c"]].mean()
    lines.append(md(far.round(4)))
    lines.append("")

    pd.DataFrame(rows).drop_duplicates(subset=["cell", "a", "b"]).to_csv(OUT / "contrasts.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
