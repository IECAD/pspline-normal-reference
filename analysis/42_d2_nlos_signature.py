"""D2 다지표 주입용 NLOS 서명 풀 (실험 3 ①축 재판정, 사용자 결정 2026-10-09).

학습 몫(반복 k < 19, ch5)만 쓴다. 환경별로
  - 거리: 오차 e = RANGE − 실제 3D 거리.  Δrange = e(NLOS) − median e(LOS)
  - RSS, RSS_FP: LOS 셀에서 y = a + b·log10(실제 거리) 회귀.  Δ = y(NLOS) − 회귀 예측
NLOS 셀마다 (Δrange, ΔRSS, ΔRSS_FP) 묶음을 저장 -> 주입 때 이벤트마다 한 묶음을 뽑아 세 지표에 함께 더한다.
출력: cache/d2_nlos_signature.parquet, analysis/out/42_d2_nlos_signature.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import d2 as D2  # noqa: E402
from src.data.config import load_config  # noqa: E402

cfg = load_config()
m = D2.load_meta(cfg)
m = m[m.ch == cfg["d2"]["channel"]].copy()
kmax = m.k.max() + 1
m = m[m.k < np.cumsum(cfg["split"]["ratios"])[0] * kmax]          # 학습 몫만
m["true_d"] = np.sqrt((m.X_TAG - m.X_ANCHOR) ** 2 + (m.Y_TAG - m.Y_ANCHOR) ** 2 + (m.Z_TAG - m.Z_ANCHOR) ** 2)
m["err"] = m.RANGE - m.true_d
rows, lines = [], []
for env, g in m.groupby("env"):
    los, nl = g[~g.NLOS], g[g.NLOS]
    x = np.log10(los.true_d)
    out = pd.DataFrame({"env": env, "anchor": nl.anchor.to_numpy(),
                        "d_range": (nl.err - los.err.median()).to_numpy()})
    for c, name in (("RSS", "d_rss"), ("RSS_FP", "d_rssfp")):
        b, a = np.polyfit(x, los[c], 1)
        out[name] = (nl[c] - (a + b * np.log10(nl.true_d))).to_numpy()
        los_res = los[c] - (a + b * x)
        lines.append(f"env{env} {c}: LOS fit a={a:.2f} b={b:.2f}, LOS 잔차 sd={los_res.std():.2f} dB")
    rows.append(out)
    q = out[["d_range", "d_rss", "d_rssfp"]].quantile([.1, .5, .9]).round(2)
    lines.append(f"env{env}: NLOS 셀 {len(out)}개, 분위 (10/50/90%):\n{q.to_string()}")
    lines.append(f"  상관: {out[['d_range', 'd_rss', 'd_rssfp']].corr().round(2).to_dict()}")
P = pd.concat(rows, ignore_index=True)
P.to_parquet(cfg["paths"]["cache"] / "d2_nlos_signature.parquet", index=False)
txt = "\n".join(lines)
print(txt)
(ROOT / "analysis/out/42_d2_nlos_signature.txt").write_text(txt, encoding="utf-8")
