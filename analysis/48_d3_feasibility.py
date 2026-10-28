"""D3 를 '실제 결측' 시험에 쓸 수 있는지 점검 (모델 없이 데이터 구조만). 2026-10-10.

질문: GT 좌표로 위치 격자 노드를 만들면, 노드별 정상 곡면 방법(학습 → 같은 노드의 테스트)을 적용할 수 있는가.
  - 윈도우(12 superframe) 안 태그 이동 거리
  - 시간 분할(60/20/20)에서 테스트 윈도우의 노드가 학습에 있는 비율 (커버리지)
  - 노드당 학습 윈도우 수, 앵커별 관측률, 셀 평균이 비는 칸 비율 (실제 결측)
  - 정상 녹화(Random)와 NLOS 녹화(Racks, Racks2)의 이동 범위 겹침
출력: analysis/out/48_d3_feasibility.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import d3 as D3  # noqa: E402
from src.data.config import load_config  # noqa: E402

cfg = load_config()
T = cfg["d3"]["T"]
lines = [__doc__.strip().splitlines()[0], ""]
xy_range = {}
for rec in ("TDoA_Random", "TDoA_Grid", "TDoA_Tour", "TDoA_Racks", "TDoA_Racks2"):
    b = D3.blinks(cfg, rec)
    sf = b.groupby("tidx").agg(gx=("gx", "first"), gy=("gy", "first"), n=("anchor", "size")).sort_index()
    anchors = sorted(b.anchor.unique())
    obs = b.pivot_table(index="tidx", columns="anchor", values="p", aggfunc="first").reindex(
        index=np.arange(sf.index.min(), sf.index.max() + 1))
    xy_range[rec] = (np.nanmin(sf.gx), np.nanmax(sf.gx), np.nanmin(sf.gy), np.nanmax(sf.gy))
    n_w = len(obs) // T
    O = np.isfinite(obs.to_numpy()[:n_w * T]).reshape(n_w, T, len(anchors))          # (윈도우, 시점, 앵커)
    G = sf.reindex(obs.index)[["gx", "gy"]].to_numpy()[:n_w * T].reshape(n_w, T, 2)
    disp = np.linalg.norm(G[:, -1] - G[:, 0], axis=1)
    cen = np.nanmean(G, axis=1)
    split = np.where(np.arange(n_w) < 0.6 * n_w, "train", np.where(np.arange(n_w) < 0.8 * n_w, "val", "test"))
    lines.append(f"[{rec}] 윈도우 {n_w}, 앵커 {len(anchors)}, 앵커 관측률 중앙값 {np.median(O.mean(axis=(0, 1))):.2f}, "
                 f"윈도우 안 이동 거리 중앙값 {np.nanmedian(disp):.2f} m (90% {np.nanpercentile(disp, 90):.2f} m), "
                 f"범위 x {xy_range[rec][0]:.1f}..{xy_range[rec][1]:.1f}, y {xy_range[rec][2]:.1f}..{xy_range[rec][3]:.1f}")
    if rec not in ("TDoA_Random", "TDoA_Grid", "TDoA_Tour"):
        continue
    for cell in (0.5, 1.0):
        ok = np.all(np.isfinite(cen), axis=1)
        key = np.full(n_w, "", dtype=object)
        key[ok] = [f"{int(np.floor(x / cell))},{int(np.floor(y / cell))}" for x, y in cen[ok]]
        tr = (split == "train") & ok
        te = (split == "test") & ok
        trk = pd.Series(key[tr]).value_counts()
        cov1 = np.mean([k in trk.index for k in key[te]])
        cov2 = np.mean([trk.get(k, 0) >= 2 for k in key[te]])
        # 셀 평균이 비는 칸: 학습 윈도우가 있는 노드의 (노드, 앵커, 시점) 중 관측 0
        empt = []
        for k in trk.index:
            c = O[tr][key[tr] == k].sum(0)                                             # (T, 앵커)
            empt.append((c == 0).mean())
        lines.append(f"    격자 {cell} m: 학습 노드 {len(trk)}, 노드당 학습 윈도우 중앙값 {trk.median():.0f}, "
                     f"테스트 커버리지 (학습 ≥1) {cov1:.2f}, (학습 ≥2) {cov2:.2f}, "
                     f"셀 평균 빈칸 비율 (학습 노드 평균) {np.mean(empt):.2f}")
    lines.append("")


def overlap(a, b):
    ox = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    oy = max(0, min(a[3], b[3]) - max(a[2], b[2]))
    return ox * oy


for nl in ("TDoA_Racks", "TDoA_Racks2"):
    ar = (xy_range[nl][1] - xy_range[nl][0]) * (xy_range[nl][3] - xy_range[nl][2])
    lines.append(f"{nl} 이동 범위 중 Random 범위와 겹치는 면적 비율 (사각형 근사): "
                 f"{overlap(xy_range['TDoA_Random'], xy_range[nl]) / ar:.2f}")
txt = "\n".join(lines)
(ROOT / "analysis/out/48_d3_feasibility.txt").write_text(txt, encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print(txt)
