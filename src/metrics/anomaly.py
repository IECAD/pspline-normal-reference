"""탐지·위치화 지표 (doc/합성방식_실험계획_최종.md 5.3절). point-adjust F1 은 쓰지 않는다.

입력은 '시퀀스' 목록: 각 시퀀스는 (score[A, L], label[A, L]) — A = 앵커(또는 변수), L = 시간.
label 의 NaN 셀은 평가에서 뺀다.

탐지 단위
  - time: 시점 점수 = 앵커 점수의 최댓값, 시점 라벨 = 이상 앵커가 하나라도 있으면 1 (D1, D4)
  - cell: 앵커×시점 셀 그대로 (D2: 거의 모든 시점에 NLOS 앵커가 있어 time 단위가 퇴화)

정의 (이 저장소 구현)
  - AUC-PR: sklearn average_precision_score
  - VUS-PR: Paparrizos et al. (2022, PVLDB 15(11)) 의 Range-AUC-PR 을 버퍼 ℓ = 0..L_max 에 대해 평균.
            버퍼 영역 소프트 라벨 sqrt(1 − d/(ℓ+1)), 재현율 = 소프트 TP / 소프트 P × 존재 보상(검출된 구간 비율).
            임계값은 점수 분위 100개. (간이 구현: 공식 패키지와 수치가 다를 수 있음)
  - Range-F1: Tatbul et al. (2018, NeurIPS). 재현율 α=0.5 (존재 0.5 + 겹침 0.5), 정밀도 α=0, flat 위치 편향,
              cardinality = 1/(겹친 구간 수). 임계값 = 정상 검증 A_R 97.5 백분위
  - HitRate@100%: 이상 앵커가 있는 시점마다 상위 |G| 앵커 중 정답 비율의 평균 (Su et al. 2019 방식, 임계값 무관)
  - 셀 F1: 앵커×시점 셀 단위 (임계값 초과 vs 라벨)
  - Smearing: 이상 앵커가 있는 시점에서, 임계 초과 셀 중 이상 없는 앵커 셀의 비율
"""
import numpy as np
from sklearn.metrics import average_precision_score


def _runs(b):
    b = np.asarray(b, bool)
    if not b.any():
        return []
    d = np.diff(np.r_[0, b.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))   # [s, e)


def to_time(score, label):
    """(A, L) -> (L,) 시점 점수·라벨. 라벨: 1 이 하나라도 있으면 1, 전부 0 이면 0, 아니면 NaN."""
    s = np.nanmax(score, 0) if np.isfinite(score).any() else np.full(score.shape[1], np.nan)
    any1 = np.nansum(label == 1, 0) > 0
    all_known = np.isfinite(label).all(0)
    lab = np.where(any1, 1.0, np.where(all_known, 0.0, np.nan))
    return s, lab


def _flatten(seqs):
    s = np.concatenate([x.ravel() for x, _ in seqs])
    y = np.concatenate([l.ravel() for _, l in seqs])
    ok = np.isfinite(s) & np.isfinite(y)
    return s[ok], y[ok]


def auc_pr(seqs):
    s, y = _flatten(seqs)
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def _soft_labels(y, ell):
    soft = y.astype(float).copy()
    if ell <= 0:
        return soft
    for s, e in _runs(y > 0):
        for d in range(1, ell + 1):
            w = np.sqrt(1 - d / (ell + 1))
            if s - d >= 0:
                soft[s - d] = max(soft[s - d], w)
            if e - 1 + d < len(y):
                soft[e - 1 + d] = max(soft[e - 1 + d], w)
    return soft


def vus_pr(seq1d, L_max=6, n_thr=100):
    """seq1d: [(score[L], label[L])] 1차원 시퀀스 목록."""
    allsc = np.concatenate([s[np.isfinite(s)] for s, _ in seq1d])
    if allsc.size == 0:
        return np.nan
    thrs = np.unique(np.quantile(allsc, np.linspace(0, 1, n_thr)))[::-1]
    aucs = []
    for ell in range(L_max + 1):
        sc_parts, soft_parts = [], []
        for s, y in seq1d:                           # 시퀀스 사이에 구분자(점수 -inf, 라벨 0)를 넣어 이어 붙임
            ok = np.isfinite(s) & np.isfinite(y)
            soft_parts += [_soft_labels(np.where(ok, y, 0) > 0, ell) * ok, [0.0]]
            sc_parts += [np.where(ok, s, -np.inf), [-np.inf]]
        sc, soft = np.concatenate(sc_parts), np.concatenate(soft_parts)
        rng = np.array(_runs(soft > 0)).reshape(-1, 2)
        if soft.sum() == 0:
            return np.nan
        pred = sc[None, :] >= thrs[:, None]                     # (n_thr, L)
        tp = pred @ soft
        npred = pred.sum(1)
        cs = np.concatenate([np.zeros((len(thrs), 1)), np.cumsum(pred, 1)], 1)
        hit = ((cs[:, rng[:, 1]] - cs[:, rng[:, 0]]) > 0).sum(1) if len(rng) else np.zeros(len(thrs))
        P = np.where(npred > 0, tp / np.maximum(npred, 1), 1.0)
        R = tp / soft.sum() * (hit / max(len(rng), 1))
        R, P = np.r_[0, R], np.r_[1, P]
        aucs.append(float(np.sum(np.diff(R) * P[1:])))
    return float(np.mean(aucs))


def _soft_labels_paper(y, ell):
    """Paparrizos 외 (2022) 식 13: 구간 양쪽에 ℓ/2 씩, 거리 d 에서 √(1 − d/ℓ)."""
    soft = y.astype(float).copy()
    half = int(ell // 2)
    if ell <= 0 or half <= 0:
        return soft
    for s, e in _runs(y > 0):
        for d in range(1, half + 1):
            w = np.sqrt(1 - d / ell)
            if s - d >= 0:
                soft[s - d] = max(soft[s - d], w)
            if e - 1 + d < len(y):
                soft[e - 1 + d] = max(soft[e - 1 + d], w)
    return soft


def vus_pr_paper(seq1d, L_max=6, n_thr=100):
    """VUS-PR 논문식 (Paparrizos 외 2022 식 13–16, 18). 기본 평가(vus_pr, 변형)에는 쓰지 않는다.
    문헌 대조(doc/문헌대조_결과.md #9)로 추가: 소프트 라벨 양쪽 ℓ/2·√(1 − d/ℓ), 양성 수 P_ℓ = (Σy + Σy_ℓ)/2,
    재현율 = min(1, TP/P_ℓ) × 검출 구간 비율, 임계마다 사다리꼴 적분 후 ℓ = 0..L_max 평균."""
    allsc = np.concatenate([s[np.isfinite(s)] for s, _ in seq1d])
    if allsc.size == 0:
        return np.nan
    thrs = np.unique(np.quantile(allsc, np.linspace(0, 1, n_thr)))[::-1]
    aucs = []
    for ell in range(L_max + 1):
        sc_parts, soft_parts, hard_parts = [], [], []
        for s, y in seq1d:
            ok = np.isfinite(s) & np.isfinite(y)
            yy = np.where(ok, y, 0) > 0
            soft_parts += [_soft_labels_paper(yy, ell) * ok, [0.0]]
            hard_parts += [yy.astype(float), [0.0]]
            sc_parts += [np.where(ok, s, -np.inf), [-np.inf]]
        sc, soft, hard = np.concatenate(sc_parts), np.concatenate(soft_parts), np.concatenate(hard_parts)
        if hard.sum() == 0:
            return np.nan
        rng = np.array(_runs(soft > 0)).reshape(-1, 2)
        P_ell = (hard.sum() + soft.sum()) / 2
        pred = sc[None, :] >= thrs[:, None]
        tp = pred @ soft
        npred = pred.sum(1)
        cs = np.concatenate([np.zeros((len(thrs), 1)), np.cumsum(pred, 1)], 1)
        hit = ((cs[:, rng[:, 1]] - cs[:, rng[:, 0]]) > 0).sum(1) if len(rng) else np.zeros(len(thrs))
        P = np.where(npred > 0, tp / np.maximum(npred, 1), 1.0)
        R = np.minimum(1.0, tp / P_ell) * (hit / max(len(rng), 1))
        R, P = np.r_[0, R], np.r_[1, P]
        aucs.append(float(np.trapezoid(P, R)) if hasattr(np, "trapezoid") else float(np.trapz(P, R)))
    return float(np.mean(aucs))


def range_f1(seq1d, thr, alpha=0.5):
    recs, precs = [], []
    for s, y in seq1d:
        ok = np.isfinite(s) & np.isfinite(y)
        pred = (np.where(ok, s, -np.inf) > thr)
        true = np.where(ok, y, 0) > 0
        R_true, R_pred = _runs(true), _runs(pred)
        for a, b in R_true:
            ov = pred[a:b].sum()
            n_over = sum(1 for c, d in R_pred if c < b and d > a)
            card = 1.0 / n_over if n_over else 1.0
            recs.append(alpha * (ov > 0) + (1 - alpha) * ov / (b - a) * card)
        for c, d in R_pred:
            ov = true[c:d].sum()
            n_over = sum(1 for a, b in R_true if a < d and b > c)
            card = 1.0 / n_over if n_over else 1.0
            precs.append(ov / (d - c) * card)
    if not recs:
        return np.nan, np.nan, np.nan
    rec = float(np.mean(recs))
    prec = float(np.mean(precs)) if precs else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return f1, prec, rec


def hitrate(seqs):
    """라벨과 점수가 있는 앵커만 순위에 넣는다 (D2 주입 평가에서 NLOS 앵커는 라벨 NaN)."""
    hits = []
    for s, y in seqs:
        for t in range(s.shape[1]):
            ok = np.isfinite(y[:, t]) & np.isfinite(s[:, t])
            yt, st = y[ok, t], s[ok, t]
            if (yt == 1).sum() == 0:
                continue
            g = int((yt == 1).sum())
            top = np.argsort(-st)[:g]
            hits.append((yt[top] == 1).sum() / g)
    return float(np.mean(hits)) if hits else np.nan


def cell_f1(seqs, thr):
    s, y = _flatten(seqs)
    pred = s > thr
    tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else np.nan


def smearing(seqs, thr):
    flagged = smear = 0
    for s, y in seqs:
        an = np.nansum(y == 1, 0) > 0
        if not an.any():
            continue
        st, yt = s[:, an], y[:, an]
        f = (st > thr) & np.isfinite(yt)
        flagged += f.sum()
        smear += (f & (yt == 0)).sum()
    return float(smear / flagged) if flagged else np.nan


def evaluate(seqs, thr, level="time", vus_lmax=6, time_labels=None):
    """seqs: [(score[A, L], label[A, L])]. time_labels: 시점 라벨 [L] 목록 (주면 탐지 지표에 사용, D4 의 test_label).
    반환: 지표 dict."""
    if level == "time":
        s1 = [to_time(s, y) for s, y in seqs]
        if time_labels is not None:
            s1 = [(s, np.asarray(tl, float)) for (s, _), tl in zip(s1, time_labels)]
        det = [(s[None], y[None]) for s, y in s1]
    else:
        s1 = [(s[a], y[a]) for s, y in seqs for a in range(s.shape[0])]
        det = seqs
    f1, p, r = range_f1(s1, thr)
    _, y = _flatten(det)
    return dict(auc_pr=auc_pr(det), vus_pr=vus_pr(s1, vus_lmax), range_f1=f1, range_p=p, range_r=r,
                hitrate=hitrate(seqs), cell_f1=cell_f1(seqs, thr), smearing=smearing(seqs, thr),
                pos_rate=float(y.mean()) if len(y) else np.nan, n_eval=int(len(y)))
