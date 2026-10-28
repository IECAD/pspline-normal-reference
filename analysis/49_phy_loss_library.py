"""실제 UWB 패킷 손실 패턴 라이브러리 (D5: IDLab UWB PHY Settings, Apache-2.0, data_ext/D5_idlab_phy/).

원본: imec IDLab 사무실 9층, 앵커 15개 사이 ADS-TWR. 설정(채널 3·송신 이득 2·데이터율 2·PRF 2·프리앰블 3)마다
앵커 쌍당 최대 505회 시도(성공+실패), 오류 20회면 다음 설정으로 넘어감 (Experiment_description.md).
처리 CSV 의 slot_ID_tag 는 시도마다 1씩 증가한다 -> 블록(태그 노드, 앵커, 설정, 감쇠) 안에서 번호가 건너뛴 곳 = 실패.
여기서는 각 노드의 축약 파일(processed_short, 약 0.24 GB)만 쓴다.
출력: cache/phy_loss_blocks.npz (블록별 성공 0/1 시퀀스), analysis/out/49_phy_loss_library.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data_ext" / "D5_idlab_phy" / "processed_short"
KEY = ["anchor_ID_tag", "channel", "bitrate", "preamble", "prf", "txpower_tag", "Attenuation"]
MIN_SPAN = 100

seqs, meta = [], []
for f in sorted(SRC.glob("office_lab_node_*_short.csv")):
    node = int(f.stem.split("_")[3])
    d = pd.read_csv(f, usecols=["slot_ID_tag"] + KEY)
    for k, g in d.groupby(KEY):
        s = np.sort(g.slot_ID_tag.unique())
        span = int(s[-1] - s[0] + 1)
        if span < MIN_SPAN:
            continue
        ok = np.zeros(span, np.int8)
        ok[s - s[0]] = 1
        seqs.append(ok)
        meta.append(dict(node=node, **dict(zip(KEY, k)), span=span, loss=1 - ok.mean()))
M = pd.DataFrame(meta)
lens = np.array([len(x) for x in seqs])
np.savez_compressed(ROOT / "cache" / "phy_loss_blocks.npz", seq=np.concatenate(seqs), offset=np.r_[0, np.cumsum(lens)],
                    loss=M.loss.to_numpy(), node=M.node.to_numpy())


def bursts(ok):
    d = np.diff(np.r_[1, ok, 1])
    st, en = np.flatnonzero(d == -1), np.flatnonzero(d == 1)
    return en - st


lines = [__doc__.strip().splitlines()[0], "", f"블록 {len(M)} 개 (시도 {MIN_SPAN} 회 이상), 노드 {M.node.nunique()}, 시도 합계 {lens.sum()}",
         "블록 손실률 분포: " + ", ".join(f"{q:.0%} {M.loss.quantile(q):.3f}" for q in (0.1, 0.25, 0.5, 0.75, 0.9))]
for lo, hi in ((0.05, 0.2), (0.2, 0.5), (0.5, 1.0)):
    sel = np.flatnonzero((M.loss >= lo) & (M.loss < hi))
    b = np.concatenate([bursts(seqs[i]) for i in sel]) if len(sel) else np.array([0])
    lines.append(f"  손실률 [{lo}, {hi}): 블록 {len(sel)}, 평균 손실 {M.loss.iloc[sel].mean():.3f}, "
                 f"버스트 길이 평균 {b.mean():.2f}, 중앙 {np.median(b):.0f}, 90% {np.percentile(b, 90):.0f}")
lines.append("송신 이득 × 데이터율별 평균 손실:\n" + M.groupby(["txpower_tag", "bitrate"]).loss.mean().round(3).to_string())
txt = "\n".join(lines)
(ROOT / "analysis/out/49_phy_loss_library.txt").write_text(txt, encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
print(txt)
