"""실험 7 P4: D2 실제 NLOS 측위 (doc/실험7_계획.md "P4", 사전 등록 2026-10-12). 테스트 한 번, 선택 없음.

측위 시점 = (환경, 경로 위치, 반복 k), 평가 반복 k 25–30. 앵커 8개 거리, GT 좌표, 2D 해 (높이 차 = Z_TAG − Z_ANCHOR).
구성 (soft 가중 w = (임계/A)², A > 임계):
  none / B1-soft (5-3 3-B: 경로 절반 분할, 위치 무관 RX−FP 앵커별 정상 통계, 누수 없음, 주)
  PS-soft(마스크 없음: 라벨 없이 학습) / PS-soft(마스크 학습: 누수 참고) / 정답 제외 (≥3 규칙)
해법: LS (주), WLS, Huber, EKF (경로 순서, q = 0.3 m). 편향 b_a·σ_a = 방향별 학습 절반 위치·학습 반복·LOS 셀.
평가 집합: PS 윈도우가 덮는 위치와 B1 평가 위치의 공통 시점 (주). B1·none·정답은 전체 시점도.
사용법: python experiments/exp7_d2real.py
출력: results/exp7/d2_real/metrics.csv (단위별), pooled.csv, summary.md
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from exp7_positioning import w_hard, w_soft  # noqa: E402
from node_model_eval import build, full_spec  # noqa: E402
from src.data import d2 as D2  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.load import load_windows  # noqa: E402
from src.model.bank import anchor_scores  # noqa: E402
from src.positioning.solvers import ekf, gauss_newton  # noqa: E402

OUT = ROOT / "results" / "exp7" / "d2_real"
TRAIN_K, CAL_K, EVAL_K = np.arange(0, 19), np.arange(19, 25), np.arange(25, 31)
Q = 0.3
POOL = defaultdict(list)


def env_arrays(m):
    """한 환경 → 위치·앵커·반복 배열 (위치 = 정렬된 path_idx)."""
    pos = np.sort(m.path_idx.unique())
    pi = {p: i for i, p in enumerate(pos)}
    K = int(m.k.max()) + 1
    out = {}
    for f in ("RANGE", "RXFP"):
        a = np.full((len(pos), 8, K), np.nan)
        a[m.path_idx.map(pi).to_numpy(), m.anchor.to_numpy() - 1, m.k.to_numpy()] = m[f].to_numpy()
        out[f] = a
    out["NL"] = m.groupby(["path_idx", "anchor"]).NLOS.first().unstack().reindex(index=pos, columns=range(1, 9)) \
        .to_numpy().astype(float)
    out["XYZ"] = m.groupby("path_idx")[["X_TAG", "Y_TAG", "Z_TAG"]].first().reindex(pos).to_numpy()
    out["AXYZ"] = m.groupby("anchor")[["X_ANCHOR", "Y_ANCHOR", "Z_ANCHOR"]].first().reindex(range(1, 9)).to_numpy()
    out["pos"], out["pi"] = pos, pi
    return out


def ps_scores(env, E, ecfg, dcfg, code, d, use_mask):
    """제안 모델 PS 를 D2 윈도우로 학습 (use_mask: LOS 셀만 = 원래 방식, 아니면 라벨 없이) → (위치, 앵커, k) 점수, 임계."""
    w = load_windows("d2", f"env{env}_{E['channel_name']}", dcfg)
    X, rows, meta = w["X"], w["rows"], w["meta"]
    W = w["cells"]["mask"] if use_mask else np.isfinite(X).astype(float)
    nodes = np.array([f"env{env}/{int(t)}" for t in meta["tidx0"]])
    split = meta["split"].to_numpy()
    tr, va, te = split == "train", split == "val", split == "test"
    comp, bank, _ = build(code, full_spec({"K_t": 12}, ecfg), ecfg, "d2", X[tr], W[tr], rows, nodes[tr],
                          X[va], W[va], nodes[va])
    Xs, Ws = comp.transform(X[te], W[te], rows)
    A = anchor_scores(bank.score(Xs, nodes[te], Ws), comp.anchor_of_row, [f"A{a}" for a in range(1, 9)])
    S = np.full(d["RANGE"].shape, np.nan)
    for i, (sg, t0) in enumerate(zip(meta["seg"].to_numpy()[te], meta["tidx0"].to_numpy()[te])):
        k = int(sg) - env * 100
        for t in range(A.shape[2]):
            p = int(t0) + t
            if p in d["pi"]:
                S[d["pi"][p], :, k] = A[i, :, t]
    return S, float(bank.thr)


def summarize(e, valid, key, tag, rows):
    x = e[valid & np.isfinite(e)]
    POOL[key].append(x)
    rows.append(dict(**tag, n=int(x.size), rmse=float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan,
                     p90=float(np.percentile(x, 90)) if x.size else np.nan))


if __name__ == "__main__":
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    E = ecfg["datasets"]["d2"]
    code = ecfg["ids"]["F01"]
    mall = D2.load_meta(dcfg)
    mall = mall[mall.ch == dcfg["d2"]["channel"]].copy()
    mall["RXFP"] = mall.RSS - mall.RSS_FP
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cover = [], []
    for env in E["envs"]:
        d = env_arrays(mall[mall.env == env])
        n = len(d["pos"])
        anc = d["AXYZ"][:, :2]
        h = d["XYZ"][:, 2][:, None] - d["AXYZ"][None, :, 2]                             # (n, 8)
        td = np.sqrt(((d["XYZ"][:, None, :] - d["AXYZ"][None]) ** 2).sum(-1))          # (n, 8)
        rres = d["RANGE"] - td[..., None]
        ps = {"PS-soft(마스크 없음)": ps_scores(env, E, ecfg, dcfg, code, d, False),
              "PS-soft(마스크 학습, 누수 참고)": ps_scores(env, E, ecfg, dcfg, code, d, True)}
        half = n // 2
        for direction, itr, ite in (("앞→뒤", np.arange(half), np.arange(half, n)),
                                    ("뒤→앞", np.arange(half, n), np.arange(half))):
            los_tr = (d["NL"][itr] == 0)[..., None]
            rr = np.where(los_tr, rres[itr][:, :, TRAIN_K], np.nan)
            b = np.nanmedian(rr, axis=(0, 2))
            sig = np.nanstd(rr - b[None, :, None], axis=(0, 2))
            # 학습 절반에서 LOS 셀이 하나도 없는 앵커 (첫 실행에서 b = NaN 이 해 전체를 NaN 으로 만든 버그, 2026-10-12 수정):
            # 편향은 다른 앵커 편향의 중앙값, σ 는 다른 앵커 σ 의 중앙값, B1 통계는 학습 절반의 앵커 공통 통계로 둔다
            nolos = ~np.isfinite(b)
            b = np.where(nolos, np.nanmedian(b), b)
            sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nanmedian(sig[np.isfinite(sig) & (sig > 0)]))
            rx = np.where(los_tr, d["RXFP"][itr][:, :, TRAIN_K], np.nan)
            mu_b1, sd_b1 = np.nanmean(rx, axis=(0, 2)), np.nanstd(rx, axis=(0, 2))
            mu_b1 = np.where(np.isfinite(mu_b1), mu_b1, np.nanmean(rx))
            sd_b1 = np.where(np.isfinite(sd_b1) & (sd_b1 > 0), sd_b1, np.nanstd(rx))
            zb = lambda x: np.abs(x - mu_b1[None, :, None]) / (sd_b1[None, :, None] + 1e-9)
            cal = np.where((d["NL"][itr] == 0)[..., None], zb(d["RXFP"][itr][:, :, CAL_K]), np.nan)
            thr_b1 = float(np.nanpercentile(cal, 97.5))
            # 평가: (위치 ite, 앵커, k ∈ EVAL_K) -> 해법 입력은 (k, 위치, 앵커) 시퀀스 (EKF 는 경로 순서)
            r = np.moveaxis(d["RANGE"][ite][:, :, EVAL_K], 2, 0)                     # (nk, m, 8)
            avail = np.isfinite(r)
            nl = np.broadcast_to(d["NL"][ite][None], r.shape)
            tres = r - td[ite][None] - b[None, None]
            sc = {"B1-soft": (np.moveaxis(zb(d["RXFP"][ite][:, :, EVAL_K]), 2, 0), thr_b1)}
            for name, (S, thr) in ps.items():
                sc[name] = (np.moveaxis(S[ite][:, :, EVAL_K], 2, 0), thr)
            cov = np.isfinite(sc["PS-soft(마스크 없음)"][0]).any(-1)                   # (nk, m) PS 가 덮는 시점
            cover.append(dict(env=env, direction=direction, coverage=float(cov.mean()), anchors_no_los_train=int(nolos.sum())))
            Wset = {"none": avail.astype(float)}
            for name, (A, thr) in sc.items():
                Wset[name] = w_soft(np.nan_to_num(A, nan=0.0), thr, avail)
            tr_ = lambda a: np.swapaxes(a, 1, 2)                                      # w_hard 는 (N, 앵커, T)
            Wset["정답 제외"] = tr_(w_hard(tr_(nl == 1), tr_(np.abs(np.nan_to_num(tres, nan=0.0))), tr_(avail)))
            pos = np.broadcast_to(d["XYZ"][ite][None, :, :2], r.shape[:2] + (2,))
            hh = np.broadcast_to(h[ite][None], r.shape)
            valid = avail.sum(-1) >= 3
            for cfg, w in Wset.items():
                Pls = gauss_newton(r.reshape(-1, 8), w.reshape(-1, 8), anc, hh.reshape(-1, 8), b).reshape(r.shape[:2] + (2,))
                Pw = gauss_newton(r.reshape(-1, 8), w.reshape(-1, 8), anc, hh.reshape(-1, 8), b, sig=sig) \
                    .reshape(r.shape[:2] + (2,))
                Ph = gauss_newton(r.reshape(-1, 8), w.reshape(-1, 8), anc, hh.reshape(-1, 8), b, sig=sig, huber=1.345) \
                    .reshape(r.shape[:2] + (2,))
                Pe = ekf(r, w, anc, hh, b, sig, Q, Pw[:, 0])
                for sv, P in (("LS", Pls), ("WLS", Pw), ("Huber", Ph), ("EKF", Pe)):
                    e = np.sqrt(((P - pos) ** 2).sum(-1))                             # (nk, m)
                    for ki, k in enumerate(EVAL_K):
                        for subset, m_ in (("공통", valid[ki] & cov[ki]), ("전체", valid[ki])):
                            if subset == "전체" and cfg.startswith("PS"):
                                continue
                            summarize(e[ki], m_, (subset, sv, cfg),
                                      dict(env=env, direction=direction, k=int(k), subset=subset, solver=sv, config=cfg),
                                      rows)
        print(f"  env{env} done", flush=True)
    M = pd.DataFrame(rows)
    M.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cover).to_csv(OUT / "coverage.csv", index=False, encoding="utf-8-sig")
    P = pd.DataFrame([dict(subset=k[0], solver=k[1], config=k[2], n=int(np.concatenate(v).size),
                           rmse=float(np.sqrt(np.mean(np.concatenate(v) ** 2))),
                           p90=float(np.percentile(np.concatenate(v), 90))) for k, v in POOL.items()])
    P.to_csv(OUT / "pooled.csv", index=False, encoding="utf-8-sig")
    # 판정 (사전 등록): 공통 시점, LS, B1-soft 개선율 ≥ 5 % 이고 (환경, k) 단위 80 % 이상에서 개선
    L = ["# 실험 7 P4: D2 실제 NLOS 측위 (테스트)", "", "자동 생성: `experiments/exp7_d2real.py`. 계획: `doc/실험7_계획.md` \"P4\".", ""]
    dec = {}
    for subset in ("공통", "전체"):
        t = P[P.subset == subset].pivot_table(index="solver", columns="config", values="rmse").reindex(
            ["LS", "WLS", "Huber", "EKF"])
        L += [f"## {subset} 시점 (RMSE m)", "", "```", t.round(3).to_string(), "```", ""]
        Ms = M[M.subset == subset].assign(sse=lambda x: x.rmse ** 2 * x.n)
        g = Ms.groupby(["solver", "config", "env", "k"])[["sse", "n"]].sum()
        u = np.sqrt(g.sse / g.n.clip(lower=1)).unstack("config")                       # (환경, k) 단위 RMSE (두 방향 합침)
        for cfg in [c for c in t.columns if c != "none"]:
            for sv in ("LS", "WLS", "Huber", "EKF"):
                imp = 1 - t.loc[sv, cfg] / t.loc[sv, "none"]
                uu = u.loc[sv]
                frac = float((uu[cfg] < uu["none"]).mean())
                dec[f"{subset}/{sv}/{cfg}"] = dict(improvement=float(imp), frac_units=frac,
                                                   meets=bool(imp >= 0.05 and frac >= 0.8))
    main = dec.get("공통/LS/B1-soft", {})
    L += ["## 판정 값 (개선율 = 1 − RMSE/RMSE_none, 단위 = (환경, k))", "", "```",
          json.dumps(dec, indent=1, ensure_ascii=False), "```", "",
          f"**주 판정 (공통 시점, LS, B1-soft): {main}**"]
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(dict(rule="doc/실험7_계획.md P4", main=main, all=dec), indent=1,
                                                  ensure_ascii=False), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))
