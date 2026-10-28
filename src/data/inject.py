"""방법 C: 정상 데이터에 NLOS형 양(+) 거리 편향과 잡음을 주입한다 (앵커×시간 정답이 정확).

파라미터 근거: analysis/out/data_report.md 5.4절 (관측된 양(+) 편차의 크기·지속 시간).
  - 구간(seg)의 target_split 블록마다 이벤트 n 개. 앵커 1개 또는 2개 동시 (BRIEF 4.2절 시나리오)
  - spike: 짧은 스파이크, persistent: 지속형 (블록보다 길면 블록 길이로 자름)
  - 주입 셀: DIST += magnitude + N(0, extra_noise_sd). signal_drop 이 있으면 RSSI/FP2 도 바꾼다
라벨 열 lab_C_A{a}: target_split 안에서 0/1, 그 밖은 NaN.
"""
import numpy as np
import pandas as pd


def inject_windows(X, rows, sc: dict, seed: int, dist_metric="DIST", eligible=None, delta_pool=None,
                   delta_metrics=None):
    """윈도우 단위 주입 (실험 3 시나리오 SC1-SC3, doc/합성방식_실험계획_최종.md 5.2절).

    X: (N, S, T) 원 단위 윈도우, rows: 행 이름 ('{metric}_A{a}').
    sc: {p_event, n_anchors, types: {name: {prob, magnitude_m, duration_steps}}, extra_noise_sd_m}
    eligible: (N, A) bool — 주입할 수 있는 앵커 (D2: 윈도우 전체가 LOS 인 앵커). 부족하면 그 윈도우는 건너뜀
    delta_pool: (M, len(delta_metrics)) 실제 NLOS 서명 (D2 다지표 주입). 주면 magnitude_m 대신 이벤트·앵커마다
                한 묶음을 뽑아 delta_metrics 의 각 행에 함께 더한다 (지표 간 상관 유지)
    반환: X_inj, labels (N, A, T) — A = 앵커 (rows 에 나온 순서, 'A1'..), anchors 목록, 이벤트 기록
    """
    rng = np.random.default_rng(seed)
    X = X.copy()
    N, _, T = X.shape
    anchors = sorted({r.rsplit("_A", 1)[1] for r in rows}, key=int)
    lab = np.zeros((N, len(anchors), T))
    types = list(sc["types"])
    tprob = np.array([sc["types"][t]["prob"] for t in types], float)
    tprob /= tprob.sum()
    events = []
    for n in range(N):
        if rng.random() >= sc["p_event"]:
            continue
        typ = types[rng.choice(len(types), p=tprob)]
        p = sc["types"][typ]
        dur = int(rng.integers(p["duration_steps"][0], p["duration_steps"][1] + 1))
        dur = max(1, min(dur, T))
        start = int(rng.integers(0, T - dur + 1))
        pool = np.arange(len(anchors)) if eligible is None else np.flatnonzero(eligible[n])
        if len(pool) < sc["n_anchors"]:
            continue
        chosen = rng.choice(pool, size=sc["n_anchors"], replace=False)
        for j in chosen:
            r = rows.index(f"{dist_metric}_A{anchors[j]}")
            if delta_pool is None:
                mag = rng.uniform(*p["magnitude_m"])
                X[n, r, start:start + dur] += mag + rng.normal(0, sc["extra_noise_sd_m"], dur)
            else:
                dv = delta_pool[rng.integers(len(delta_pool))]
                for met, dval in zip(delta_metrics, dv):
                    rr = rows.index(f"{met}_A{anchors[j]}")
                    X[n, rr, start:start + dur] += dval
                X[n, r, start:start + dur] += rng.normal(0, sc["extra_noise_sd_m"], dur)
                mag = float(dv[0])
            lab[n, j, start:start + dur] = 1
            events.append(dict(window=n, type=typ, anchor=anchors[j], start=start, dur=dur, mag=mag))
    return X, lab, [f"A{a}" for a in anchors], pd.DataFrame(events)


def inject(cyc: pd.DataFrame, ccfg: dict, rate_hz: float, seed: int, anchor_ids, dist_col="DIST",
           drop_cols=None):
    rng = np.random.default_rng(seed)
    cyc = cyc.copy()
    tgt = cyc["split"] == ccfg["target_split"]
    for a in anchor_ids:
        cyc[f"lab_C_A{a}"] = np.where(tgt, 0.0, np.nan)
    types = list(ccfg["types"])
    tprob = np.array([ccfg["types"][t]["prob"] for t in types], float)
    tprob /= tprob.sum()
    na = [int(k) for k in ccfg["anchors_per_event"]]
    nprob = np.array([ccfg["anchors_per_event"][k] for k in ccfg["anchors_per_event"]], float)
    nprob /= nprob.sum()
    events = []
    for seg, g in cyc[tgt].groupby("seg", sort=False):
        idx = g.sort_values("tidx").index.to_numpy()
        L = len(idx)
        for _ in range(ccfg["n_events_per_segment"]):
            typ = types[rng.choice(len(types), p=tprob)]
            p = ccfg["types"][typ]
            dur = int(round(rng.uniform(*p["duration_s"]) * rate_hz))
            dur = max(1, min(dur, L))
            start = int(rng.integers(0, L - dur + 1))
            k = na[rng.choice(len(na), p=nprob)]
            anchors = sorted(rng.choice(anchor_ids, size=min(k, len(anchor_ids)), replace=False).tolist())
            rows = idx[start:start + dur]
            for a in anchors:
                mag = rng.uniform(*p["magnitude_m"])
                col = f"{dist_col}_A{a}"
                cyc.loc[rows, col] = cyc.loc[rows, col] + mag + rng.normal(0, ccfg["extra_noise_sd_m"], len(rows))
                if ccfg.get("signal_drop") and drop_cols:
                    for m, delta in ccfg["signal_drop"].items():
                        if f"{m}_A{a}" in cyc:
                            cyc.loc[rows, f"{m}_A{a}"] = cyc.loc[rows, f"{m}_A{a}"] + delta
                cyc.loc[rows, f"lab_C_A{a}"] = 1.0
                events.append(dict(seed=seed, seg=seg, node=g["node"].iloc[0], type=typ, anchor=a,
                                   start_tidx=int(cyc.loc[rows[0], "tidx"]), n_steps=len(rows),
                                   duration_s=len(rows) / rate_hz, magnitude_m=mag))
    return cyc, pd.DataFrame(events)
