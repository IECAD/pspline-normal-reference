"""실험 1 비교 모델 집계 (doc/실험1_비교모델_계획.md 3절 판정). 테스트.

PS = results/C03 (제안 모델). 비교: 이번 1차 모델들 + 앞선 단순 기준선(Z1, Z2) + ST-SSD형(G1, N).
판정: PS − 모델 짝지은 AUC-PR 평균 (D1 주입, D2 다지표 주입), 허용오차 D1 0.0165 · D2 0.0171.
출력: results/exp1_baselines/summary.md, paired.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "exp1_baselines"
TOL = {"d1": 0.0165, "d2": 0.0171}
VIEWS = {"d1": r"^SC\d$", "d2": r"^SC\dm$"}
KEY = ["space", "scenario", "seed"]
NAMES = {"PCA": "PCA-SPE", "RPCA": "RPCA 부분공간", "OLS": "OLS 센서 간 회귀", "IF": "Isolation Forest",
         "LSTMAE": "LSTM-AE", "CAE1D": "1D-CAE", "RXFP": "RX−FP 원값 규칙", "TIMEHIST": "거리 시간이력 검정",
         "ORACLE": "참 거리 잔차 z-점수 (oracle 좌표)", "USAD": "USAD (학습 오류판, 주장에서 제외)",
         "USADFIX": "USAD 수정판 (출력 무제한, 발산 추정, 기록용)", "USADFIX2": "USAD 원형 수정판 (Sigmoid 출력)",
         "RBCPHI": "RBC^φ (Alcala & Qin, 2차 수정)", "Z1": "노드 평균 (Z1)", "Z2": "셀 평균 (Z2 = M01)",
         "STSSD_G1": "ST-SSD형 전역 배경", "STSSD_N": "ST-SSD형 노드별 배경",
         "LAPRPCA": "Laplacian-RPCA 센서 그래프 변형 (2차 수정판)", "RRR": "RRR 시차 감소 랭크 (2차)", "TRANS": "Transformer 시점 마스킹 (2차)",
         "CAE2D": "2D Conv-AE (2차)", "GDN": "GDN 재구현 (2차, 누수 수정판)", "OMNI": "OmniAnomaly 재구현 (2차)"}
BATCH2 = OUT / "batch2"


def load(ds):
    rx = VIEWS[ds]
    B = pd.read_csv(RES / "exp1_baselines" / f"metrics_test_{ds}.csv")
    parts = [B]
    for b in (BATCH2, BATCH2 / ".." / "batch2_fix", BATCH2 / ".." / "batch2_fix2"):   # 2차, 2차 수정, 추가 수정
        if (b / f"metrics_test_{ds}.csv").exists():
            parts.append(pd.read_csv(b / f"metrics_test_{ds}.csv"))
    for z in ("Z1", "Z2"):
        f = pd.read_csv(RES / z / "base" / f"metrics_{ds}_test.csv").assign(model=z, variant="-", cfg="{}")
        parts.append(f)
    S = pd.read_csv(RES / "exp1_stssd" / f"metrics_test_{ds}.csv")
    for v in ("G1", "N"):
        parts.append(S[S.variant == v].assign(model=f"STSSD_{v}", variant="-", cfg="{}"))
    P = pd.read_csv(RES / "C03" / "base" / f"metrics_{ds}_test.csv").assign(model="PS", variant="-", cfg="{}")
    parts.append(P)
    M = pd.concat(parts, ignore_index=True)
    M = M[(M.split == "test") & M.scenario.astype(str).str.match(rx)]
    if "infer_ms" not in M:
        M["infer_ms"] = float("nan")
    M["infer_ms"] = M["infer_ms"].fillna(M.get("infer_ms_per_window"))
    return M


if __name__ == "__main__":
    sel = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))["selection"]
    has2 = (BATCH2 / "metrics_test_d1.csv").exists()
    if has2:
        sel.update(json.loads((BATCH2 / "selection.json").read_text(encoding="utf-8"))["selection"])
        for fx in ("batch2_fix", "batch2_fix2"):
            fix = BATCH2 / ".." / fx / "selection.json"
            if fix.exists():
                sel.update(json.loads(fix.read_text(encoding="utf-8"))["selection"])
        OUT = BATCH2                                                  # 1·2차를 합친 표는 batch2/ 에 쓴다 (1차 표는 그대로 둠)
    lines = ["# 실험 1 비교 모델 — 테스트 (제안 모델 PS 대비)", "",
             "자동 생성: `experiments/exp1_baselines_summarize.py`. 계획: `doc/실험1_비교모델_계획.md`.",
             "PS − 모델 > 허용오차면 PS 우위, < −허용오차면 PS 열세.", ""]
    rows = []
    for ds in ("d1", "d2"):
        M = load(ds)
        ps = M[M.model == "PS"].set_index(KEY).auc_pr
        t = M.groupby("model")[["auc_pr", "hitrate", "cell_f1", "smearing", "infer_ms"]].mean()
        lines += [f"## {ds} ({'주입 SC1–SC3' if ds == 'd1' else '다지표 주입 SC1m–SC3m'}), 허용오차 {TOL[ds]}", "",
                  "| 모델 | 입력·구성 | AUC-PR | HitRate | 셀 F1 | Smearing | 추론 ms/윈도우 | PS − 모델 (양수 비율) | 판정 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        order = ["PS"] + sorted([m for m in t.index if m != "PS"], key=lambda m: -t.loc[m, "auc_pr"])
        for m in order:
            r = t.loc[m]
            conf = "—"
            if m in sel and ds in sel[m]:
                s = sel[m][ds]
                conf = f"{s['variant']} {json.dumps(s['cfg'], ensure_ascii=False)}" if s["variant"] != "-" else "튜닝 없음"
            if m == "PS":
                lines.append(f"| **PS (제안)** | 튜닝 없음 (SAP) | {r.auc_pr:.3f} | {r.hitrate:.3f} | {r.cell_f1:.3f} | "
                             f"{r.smearing:.3f} | {r.infer_ms:.4f} | — | — |")
                continue
            x = (ps - M[M.model == m].set_index(KEY).auc_pr).dropna()
            v = "PS 우위" if x.mean() > TOL[ds] else ("PS 열세" if x.mean() < -TOL[ds] else "대등")
            rows.append(dict(dataset=ds, model=m, auc_pr=r.auc_pr, delta=x.mean(), frac_pos=(x > 0).mean(),
                             n=len(x), verdict=v, config=conf))
            lines.append(f"| {NAMES.get(m, m)} | {conf} | {r.auc_pr:.3f} | {r.hitrate:.3f} | {r.cell_f1:.3f} | "
                         f"{r.smearing:.3f} | {r.infer_ms:.4f} | {x.mean():+.4f} ({(x > 0).mean():.2f}) | {v} |")
        lines.append("")
    lines.append("Isolation Forest 는 시점 점수를 모든 셀에 같게 주므로 HitRate·셀 F1·Smearing 이 위치화 지표로 의미 없다.")
    pd.DataFrame(rows).to_csv(OUT / "paired.csv", index=False, encoding="utf-8-sig")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))
