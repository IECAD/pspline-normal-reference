"""D4 (D4a = SMD, D4b = ASD) 구조 조사. 원본은 읽기만 한다.

입력 (2026-10-08 다운로드, 출처는 각 폴더 SOURCE.txt):
  data_ext/D4a_SMD/{train,test,test_label,interpretation_label}/machine-*.txt   (OmniAnomaly, CSV 텍스트)
  data_ext/D4b_ASD/processed/omi-*_{train,test,test_label}.pkl                   (InterFusion, numpy pickle)
  data_ext/D4b_ASD/interpretation_label/{omi-*,machine-*,anomaly_type}.txt
  data_ext/D4b_ASD/processed/machine-*.pkl  (InterFusion 이 쓴 SMD 12대 부분집합 -> OmniAnomaly txt 와 대조만)

보안: .pkl 은 pickle.load 하지 않는다. pickletools.genops 로 opcode 만 읽고(실행 없음),
      numpy ndarray 재구성 템플릿(전역 3개: numpy.core.multiarray._reconstruct, numpy.ndarray, numpy.dtype)과
      정확히 일치할 때만 BINBYTES 원시 바이트를 np.frombuffer 로 해석한다. 다르면 중단.

사용법: python -I analysis/21_d4_structure.py      (-I: 현재/스크립트 폴더의 모듈을 import 하지 않음)
출력: analysis/out/d4/
  21_d4_structure.txt            로그(요약 수치)
  d4_files.csv                   파일 목록(크기, sha256)
  d4a_smd_entities.csv / d4b_asd_entities.csv        엔터티별 요약
  d4a_smd_variables.csv / d4b_asd_variables.csv      엔터티 x 변수 통계
  d4a_smd_segments.csv / d4b_asd_segments.csv        test_label 의 이상 구간 (+ 해석 라벨 매칭)
  d4a_smd_interp.csv / d4b_asd_interp.csv            해석 라벨 한 줄씩 (+ test_label 과의 정합)
  d4_interfusion_smd_crosscheck.csv                  InterFusion machine-*.pkl vs OmniAnomaly txt
"""
import hashlib
import pickletools
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D4A = ROOT / "data_ext" / "D4a_SMD"
D4B = ROOT / "data_ext" / "D4b_ASD"
OUT = ROOT / "analysis" / "out" / "d4"
OUT.mkdir(parents=True, exist_ok=True)

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


# ----------------------------------------------------------------------------- static numpy-pickle reader
_SKIP = {"BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE", "FRAME"}
_ALLOWED_OPS = {"PROTO", "GLOBAL", "BININT1", "BININT2", "BININT", "TUPLE1", "TUPLE2", "TUPLE3", "TUPLE",
                "SHORT_BINBYTES", "BINBYTES", "BINBYTES8", "REDUCE", "MARK", "BINUNICODE", "SHORT_BINUNICODE",
                "NONE", "BUILD", "NEWFALSE", "NEWTRUE", "STOP"}
_GLOBALS = ["numpy.core.multiarray _reconstruct", "numpy ndarray", "numpy dtype"]
_INT = {"BININT1", "BININT2", "BININT"}
_ANY = object()


def read_numpy_pickle_static(path: Path):
    """numpy ndarray pickle(protocol 2~4) 을 실행 없이 해석. 반환: (array, info dict)."""
    data = path.read_bytes()
    ops = [(op.name, arg) for op, arg, _ in pickletools.genops(data) if op.name not in _SKIP]
    names = {n for n, _ in ops}
    if names - _ALLOWED_OPS:
        raise ValueError(f"{path.name}: unexpected opcodes {names - _ALLOWED_OPS}")
    if [a for n, a in ops if n == "GLOBAL"] != _GLOBALS:
        raise ValueError(f"{path.name}: unexpected globals")
    i = 0

    def take(name, value=_ANY):
        nonlocal i
        n, a = ops[i]
        ok = (n in name) if isinstance(name, set) else (n == name)
        if not ok or (value is not _ANY and a != value):
            raise ValueError(f"{path.name}: op#{i} {n}={a!r}, expected {name}={value!r}")
        i += 1
        return a

    proto = take("PROTO")
    take("GLOBAL", _GLOBALS[0]); take("GLOBAL", _GLOBALS[1])
    take("BININT1", 0); take("TUPLE1"); take("SHORT_BINBYTES", b"b"); take("TUPLE3"); take("REDUCE")
    take("MARK"); take("BININT1", 1)
    shape = []
    while ops[i][0] in _INT:
        shape.append(take(_INT))
    take({"TUPLE1", "TUPLE2", "TUPLE3", "TUPLE"})
    take("GLOBAL", _GLOBALS[2])
    dstr = take({"BINUNICODE", "SHORT_BINUNICODE"})
    take("BININT1", 0); take("BININT1", 1); take("TUPLE3"); take("REDUCE")
    take("MARK"); take("BININT1", 3)
    border = take({"BINUNICODE", "SHORT_BINUNICODE"})
    take("NONE"); take("NONE"); take("NONE"); take("BININT", -1); take("BININT", -1); take("BININT1", 0)
    take("TUPLE"); take("BUILD")
    take({"NEWFALSE", "NEWTRUE"})
    fortran = ops[i - 1][0] == "NEWTRUE"
    payload = take({"BINBYTES", "SHORT_BINBYTES", "BINBYTES8"})
    take("TUPLE"); take("BUILD"); take("STOP")
    if i != len(ops):
        raise ValueError(f"{path.name}: trailing ops")
    dt = np.dtype(border + dstr)
    if dt.kind not in "fiub":
        raise ValueError(f"{path.name}: non-numeric dtype {dt}")
    if len(payload) != int(np.prod(shape)) * dt.itemsize:
        raise ValueError(f"{path.name}: payload size mismatch")
    arr = np.frombuffer(payload, dtype=dt).reshape(shape, order="F" if fortran else "C").copy()
    return arr, {"protocol": proto, "dtype": str(dt), "shape": tuple(shape), "fortran": fortran}


# ----------------------------------------------------------------------------- helpers
def sha256(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_smd_txt(p: Path):
    raw = p.read_bytes()
    bad = re.sub(rb"[0-9.,\n\r-]", b"", raw)  # 숫자/구분자 이외 문자 (nan, 공백 등)
    decs = np.unique([len(m) for m in re.findall(rb"\.(\d+)", raw)])
    arr = pd.read_csv(p, header=None, dtype=np.float64).to_numpy()
    return arr, {"non_numeric_chars": len(bad), "decimals": ",".join(map(str, decs)),
                 "crlf": b"\r" in raw}


def segments(y):
    """0/1 배열 -> [(start, end_inclusive)] (0-based)."""
    y = np.asarray(y).astype(np.int8)
    d = np.diff(np.concatenate([[0], y, [0]]))
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1) - 1
    return list(zip(s.tolist(), e.tolist()))


RE_INTERP = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*:\s*([\d,\s]+?)\s*$")


def read_interp(p: Path):
    rows, bad = [], []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        m = RE_INTERP.match(ln)
        if not m:
            bad.append(ln)
            continue
        vs = [int(v) for v in m.group(3).split(",") if v.strip()]
        rows.append((int(m.group(1)), int(m.group(2)), vs))
    return rows, bad


# 해석 라벨 (s, e) 를 0-based inclusive 라벨 구간 [a, b] 로 바꾸는 후보 규약
CONV = {"0based_incl": lambda s, e: (s, e), "1based_incl": lambda s, e: (s - 1, e - 1),
        "0based_excl": lambda s, e: (s, e - 1)}


def adjacent_corr(x):
    """train 에서 비상수 변수만: 인접 행(i,i+1) |r| 평균 vs 전체 쌍 |r| 평균. 원래 순서에서 비상수만 남긴 순서 기준."""
    sd = x.std(0)
    keep = np.flatnonzero(sd > 0)
    if len(keep) < 3:
        return np.nan, np.nan
    c = np.abs(np.corrcoef(x[:, keep].T))
    adj = np.mean([c[k, k + 1] for k in range(len(keep) - 1)])
    iu = np.triu_indices(len(keep), 1)
    return float(adj), float(np.nanmean(c[iu]))


def survey(name, entities, loader, interp_dir, extra_seg_info=None):
    """entities: 이름 목록, loader(ent) -> (train, test, label, info). 결과 DataFrame 4개 반환."""
    ent_rows, var_rows, seg_rows, int_rows = [], [], [], []
    conv_hits = {k: 0 for k in CONV}
    n_interp_total = 0
    keep = {}
    for ent in entities:
        tr, te, y, info = loader(ent)
        D = tr.shape[1]
        yv = np.unique(y)
        yb = (y > 0.5).astype(np.int8)
        segs = segments(yb)
        lens = np.array([b - a + 1 for a, b in segs]) if segs else np.array([0])
        tr_min, tr_max, te_min, te_max = tr.min(0), tr.max(0), te.min(0), te.max(0)
        tr_sd, te_sd = tr.std(0), te.std(0)
        adj, allp = adjacent_corr(tr)
        ip = interp_dir / f"{ent}.txt"
        irows, ibad = read_interp(ip) if ip.exists() else ([], [])
        n_interp_total += len(irows)
        for k, f in CONV.items():
            conv_hits[k] += sum(f(s, e) in set(segs) for s, e, _ in irows)
        keep[ent] = (yb, segs, irows, D)
        allx = np.concatenate([tr, te])
        grid01 = float(np.abs(allx * 100 - np.round(allx * 100)).max())  # 0.01 격자에서 최대 편차
        for v in range(D):
            var_rows.append({"entity": ent, "var": v + 1, "train_min": tr_min[v], "train_max": tr_max[v],
                             "train_mean": tr[:, v].mean(), "train_std": tr_sd[v], "test_min": te_min[v],
                             "test_max": te_max[v], "test_std": te_sd[v], "const_train": tr_sd[v] == 0,
                             "const_test": te_sd[v] == 0, "n_unique_all": len(np.unique(allx[:, v]))})
        ent_rows.append({
            "entity": ent, "train_len": tr.shape[0], "test_len": te.shape[0], "label_len": len(y),
            "n_vars": D, "train_dtype": info.get("train_dtype"), "label_values": " ".join(f"{u:g}" for u in yv),
            "nan_train": int(np.isnan(tr).sum()), "nan_test": int(np.isnan(te).sum()),
            "inf_any": int(np.isinf(tr).sum() + np.isinf(te).sum()),
            "train_min": float(tr.min()), "train_max": float(tr.max()),
            "test_min": float(te.min()), "test_max": float(te.max()),
            "n_var_train_max_eq1": int((tr_max == 1).sum()), "n_var_train_min_eq0": int((tr_min == 0).sum()),
            "n_var_test_gt1": int((te_max > 1).sum()), "n_var_test_lt0": int((te_min < 0).sum()),
            "n_var_union_max_eq1": int((np.maximum(tr_max, te_max) == 1).sum()),
            "n_const_train": int((tr_sd == 0).sum()), "n_const_test": int((te_sd == 0).sum()),
            "n_const_both": int(((tr_sd == 0) & (te_sd == 0)).sum()),
            "const_train_vars": " ".join(str(v + 1) for v in np.flatnonzero(tr_sd == 0)),
            "anomaly_points": int(yb.sum()), "anomaly_ratio": float(yb.mean()), "n_segments": len(segs),
            "seg_len_min": int(lens.min()), "seg_len_median": float(np.median(lens)),
            "seg_len_mean": float(lens.mean()), "seg_len_max": int(lens.max()),
            "n_interp_lines": len(irows), "interp_bad_lines": len(ibad),
            "adj_abs_corr_train": adj, "allpair_abs_corr_train": allp, "grid001_maxdev": grid01,
            **{k: v for k, v in info.items() if k != "train_dtype"},
        })
    log(f"[{name}] 해석 라벨 구간 규약 검증: 해석 라벨 {n_interp_total} 줄 중 test_label 구간과 정확히 일치하는 수")
    for k, v in conv_hits.items():
        log(f"    {k:12s}: {v}")
    conv = max(conv_hits, key=conv_hits.get)  # 가장 많이 일치하는 규약을 채택
    log(f"    -> 채택: {conv}  (start_raw, end_raw) -> 0-based inclusive 라벨 구간")
    flags = {}
    for ent, (yb, segs, irows, D) in keep.items():
        seg_set = set(segs)
        flag = np.zeros(D + 2, int)
        ent_int = []
        for s, e, vs in irows:
            a, b = CONV[conv](s, e)
            a2, b2 = max(a, 0), min(b, len(yb) - 1)
            frac = float(yb[a2:b2 + 1].mean()) if b2 >= a2 else np.nan
            ov = [j for j, (sa, sb) in enumerate(segs) if sa <= b and sb >= a]
            for v in set(vs):
                flag[min(max(v, 0), D + 1)] += 1
            r = {"entity": ent, "start_raw": s, "end_raw": e, "start0": a, "end0_incl": b,
                 "n_vars": len(set(vs)), "vars": " ".join(map(str, vs)),
                 "dup_vars": len(vs) - len(set(vs)), "min_var": min(vs), "max_var": max(vs),
                 "exact_label_seg": (a, b) in seg_set, "label_frac_inside": frac,
                 "n_label_segs_overlapped": len(ov)}
            ent_int.append(r)
            int_rows.append(r)
        flags[ent] = flag
        for j, (a, b) in enumerate(segs):
            m = [r for r in ent_int if r["start0"] <= b and r["end0_incl"] >= a]
            ex = [r for r in m if (r["start0"], r["end0_incl"]) == (a, b)]
            row = {"entity": ent, "seg_id": j, "start0": a, "end0_incl": b, "length": b - a + 1,
                   "has_interp_overlap": bool(m), "has_interp_exact": bool(ex), "n_interp_overlapping": len(m),
                   "n_vars": (ex or m)[0]["n_vars"] if m else np.nan,
                   "vars": (ex or m)[0]["vars"] if m else ""}
            if extra_seg_info:
                row.update(extra_seg_info(ent, ex[0] if ex else (m[0] if m else None)))
            seg_rows.append(row)
    for r in var_rows:
        r["n_interp_segs_flagging"] = int(flags[r["entity"]][r["var"]])
    E, V, S, I = (pd.DataFrame(r) for r in (ent_rows, var_rows, seg_rows, int_rows))
    E["interp_convention"] = conv
    return E, V, S, I


def summarize(name, E, V, S, I):
    log(f"\n===== {name} =====")
    log(f"entities={len(E)}  n_vars={sorted(E.n_vars.unique())}  "
        f"train_len 합={E.train_len.sum()}  test_len 합={E.test_len.sum()}  label_len==test_len: {(E.label_len == E.test_len).all()}")
    log(f"train_len min/median/max = {E.train_len.min()}/{E.train_len.median():.0f}/{E.train_len.max()}; "
        f"test_len min/median/max = {E.test_len.min()}/{E.test_len.median():.0f}/{E.test_len.max()}")
    log(f"label values: {sorted(set(' '.join(E.label_values).split()))}")
    log(f"NaN train/test 합 = {E.nan_train.sum()}/{E.nan_test.sum()}, inf = {E.inf_any.sum()}")
    log(f"값 범위: train [{E.train_min.min():g}, {E.train_max.max():g}], test [{E.test_min.min():g}, {E.test_max.max():g}]")
    log(f"변수별 train max==1 인 변수 비율 = {E.n_var_train_max_eq1.sum() / E.n_vars.sum():.3f}, "
        f"train min==0 비율 = {E.n_var_train_min_eq0.sum() / E.n_vars.sum():.3f}")
    log(f"test 에서 >1 인 (엔터티,변수) 수 = {E.n_var_test_gt1.sum()}, <0 인 수 = {E.n_var_test_lt0.sum()} "
        f"(엔터티 {int((E.n_var_test_gt1 > 0).sum())}/{len(E)} 곳에서 >1 존재)")
    nc = V[V.train_std > 0]
    umax, umin = np.maximum(nc.train_max, nc.test_max), np.minimum(nc.train_min, nc.test_min)
    log(f"비상수(train) 변수 {len(nc)}개 중: train max==1 {(nc.train_max == 1).mean():.3f}, train min==0 {(nc.train_min == 0).mean():.3f}, "
        f"train∪test max==1 {(umax == 1).mean():.3f}, train∪test min==0 {(umin == 0).mean():.3f}, "
        f"test max>train max {(nc.test_max > nc.train_max).mean():.3f}, test min<train min {(nc.test_min < nc.train_min).mean():.3f}")
    log(f"0.01 격자 최대 편차 (엔터티 최대) = {E.grid001_maxdev.max():.3g}; 변수별 고유값 수 median={V.n_unique_all.median():.0f}, max={V.n_unique_all.max()}")
    log(f"상수 변수(train std=0) 엔터티당: min/median/max = {E.n_const_train.min()}/{E.n_const_train.median():.0f}/{E.n_const_train.max()}, "
        f"test 상수 median={E.n_const_test.median():.0f}, 둘 다 상수 median={E.n_const_both.median():.0f}")
    cc = V.groupby("var").const_train.sum()
    log(f"변수 index 별 train 상수인 엔터티 수 (>0만): {dict(cc[cc > 0])}")
    tot = E.anomaly_points.sum() / E.test_len.sum()
    log(f"이상 비율: 전체(점 합 기준) {tot:.4f}; 엔터티별 min/median/max = {E.anomaly_ratio.min():.4f}/{E.anomaly_ratio.median():.4f}/{E.anomaly_ratio.max():.4f}")
    L = S.length
    log(f"이상 구간: 총 {len(S)}개 (엔터티당 min/median/max {E.n_segments.min()}/{E.n_segments.median():.0f}/{E.n_segments.max()}); "
        f"길이 min/p25/median/p75/max = {L.min()}/{L.quantile(.25):.0f}/{L.median():.0f}/{L.quantile(.75):.0f}/{L.max()}, mean {L.mean():.1f}")
    log(f"길이 1 인 구간 {int((L == 1).sum())}개, <=5: {int((L <= 5).sum())}, >=100: {int((L >= 100).sum())}")
    log(f"인접행 |r| 평균 (train, 비상수) median={E.adj_abs_corr_train.median():.3f} vs 전체쌍 |r| median={E.allpair_abs_corr_train.median():.3f}")
    if len(I):
        log(f"-- 해석 라벨: 줄 {len(I)}개, 파싱 실패 줄 {E.interp_bad_lines.sum()}, 해석 라벨 있는 엔터티 {int((E.n_interp_lines > 0).sum())}/{len(E)}")
        log(f"   변수 index 범위 {I.min_var.min()}..{I.max_var.max()} (변수 수 {sorted(E.n_vars.unique())}), 중복 변수 표기 줄 {int((I.dup_vars > 0).sum())}")
        log(f"   규약 {E.interp_convention.iloc[0]} 으로 test_label 구간과 정확 일치 {int(I.exact_label_seg.sum())}/{len(I)}; "
            f"구간 안 라벨=1 비율 min/median = {I.label_frac_inside.min():.3f}/{I.label_frac_inside.median():.3f}; "
            f"라벨 구간 0개와 겹침 {int((I.n_label_segs_overlapped == 0).sum())}, 2개 이상과 겹침 {int((I.n_label_segs_overlapped >= 2).sum())}")
        log(f"   라벨 구간 하나에 해석 줄 2개 이상 겹침: {int((S.n_interp_overlapping >= 2).sum())} "
            f"{list(zip(S[S.n_interp_overlapping >= 2].entity, S[S.n_interp_overlapping >= 2].start0))}")
        no = S[~S.has_interp_overlap]
        log(f"   해석 라벨 없는 라벨 구간: {list(zip(no.entity, no.start0, no.end0_incl, no.length))}")
        cst = {(r.entity, r.var): (r.const_train, r.const_test) for r in V.itertuples()}
        pairs = [(r.entity, int(v)) for r in I.itertuples() for v in str(r.vars).split()]
        log(f"   (구간,변수) 쌍 {len(pairs)}개 중 그 엔터티 train 상수 변수 {sum(cst[p][0] for p in pairs)}, "
            f"test 상수 변수 {sum(cst[p][1] for p in pairs)}")
        log(f"   라벨 구간 커버리지: 해석 라벨과 겹치는 구간 {int(S.has_interp_overlap.sum())}/{len(S)} ({S.has_interp_overlap.mean():.3f}), "
            f"정확 일치 {int(S.has_interp_exact.sum())}/{len(S)}; 점 기준 커버리지 "
            f"{(S.length * S.has_interp_overlap).sum() / S.length.sum():.3f}")
        nv = I.n_vars
        D = int(E.n_vars.iloc[0])
        log(f"   구간당 변수 수 min/p25/median/p75/max = {nv.min()}/{nv.quantile(.25):.0f}/{nv.median():.0f}/{nv.quantile(.75):.0f}/{nv.max()}, "
            f"mean {nv.mean():.2f} (= D 의 {nv.mean() / D:.2f})")
        fr = V.groupby("var").n_interp_segs_flagging.sum().sort_values(ascending=False)
        log(f"   변수별 해석 라벨 등장 횟수 상위 10: {dict(fr.head(10))}")
        log(f"   해석 라벨에 한 번도 안 나온 변수 index: {list(fr[fr == 0].index)}")
        bad = I[~I.exact_label_seg]
        if len(bad):
            log(f"   정확 일치 안 하는 줄 {len(bad)}개 (최대 30개; 겹치는 라벨 구간 표시):")
            for _, r in bad.head(30).iterrows():
                ov = S[(S.entity == r.entity) & (S.start0 <= r.end0_incl) & (S.end0_incl >= r.start0)]
                log(f"     {r.entity} raw {r.start_raw}-{r.end_raw} -> [{r.start0},{r.end0_incl}] "
                    f"frac_inside={r.label_frac_inside:.3f} label_segs={list(zip(ov.start0, ov.end0_incl))}")


# ----------------------------------------------------------------------------- D4a SMD
def smd_entities():
    return sorted((p.stem for p in (D4A / "train").glob("machine-*.txt")),
                  key=lambda s: tuple(int(x) for x in s.split("-")[1:]))


def smd_load(ent):
    tr, i1 = read_smd_txt(D4A / "train" / f"{ent}.txt")
    te, i2 = read_smd_txt(D4A / "test" / f"{ent}.txt")
    y = pd.read_csv(D4A / "test_label" / f"{ent}.txt", header=None, dtype=np.float64)[0].to_numpy()
    jump = np.abs(te[0] - tr[-1]).mean()
    step = np.abs(np.diff(tr, axis=0)).mean()
    return tr, te, y, {"train_dtype": "text float (parsed float64)", "decimals": f"{i1['decimals']}|{i2['decimals']}",
                       "non_numeric_chars": i1["non_numeric_chars"] + i2["non_numeric_chars"],
                       "crlf": i1["crlf"] or i2["crlf"], "boundary_jump_over_mean_step": jump / step}


# ----------------------------------------------------------------------------- D4b ASD
def asd_entities():
    return sorted((p.name[:-len("_train.pkl")] for p in (D4B / "processed").glob("omi-*_train.pkl")),
                  key=lambda s: int(s.split("-")[1]))


def asd_load(ent):
    P = D4B / "processed"
    tr, a = read_numpy_pickle_static(P / f"{ent}_train.pkl")
    te, b = read_numpy_pickle_static(P / f"{ent}_test.pkl")
    y, c = read_numpy_pickle_static(P / f"{ent}_test_label.pkl")
    tr64, te64 = tr.astype(np.float64), te.astype(np.float64)
    jump = np.abs(te64[0] - tr64[-1]).mean()
    step = np.abs(np.diff(tr64, axis=0)).mean()
    return tr64, te64, y.astype(np.float64), {
        "train_dtype": a["dtype"], "test_dtype": b["dtype"], "label_dtype": c["dtype"],
        "pickle_protocol": a["protocol"], "label_shape": str(c["shape"]),
        "boundary_jump_over_mean_step": jump / step}


ATYPE = {}
_atp = D4B / "interpretation_label" / "anomaly_type.txt"
if _atp.exists():
    _at = pd.read_csv(_atp)
    ATYPE = {(r.dataset, r.anomaly_segment): r.anomaly_type for r in _at.itertuples()}


def asd_seg_info(ent, irow):
    if irow is None:
        return {"anomaly_type": ""}
    return {"anomaly_type": ATYPE.get((ent, f"{irow['start_raw']}-{irow['end_raw']}"), "")}


# ----------------------------------------------------------------------------- main
def main():
    # 파일 목록
    rows = []
    for ds, base in (("D4a_SMD", D4A), ("D4b_ASD", D4B)):
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rows.append({"dataset": ds, "path": p.relative_to(base).as_posix(), "bytes": p.stat().st_size,
                             "sha256": sha256(p)})
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "d4_files.csv", index=False)
    F["top"] = F.path.str.split("/").str[0]
    log("파일 목록 (폴더별 개수, 바이트):")
    log(F.groupby(["dataset", "top"]).bytes.agg(["count", "sum"]).to_string())
    log(f"합계: {F.groupby('dataset').bytes.agg(['count', 'sum']).to_dict()}")

    # D4a
    E, V, S, I = survey("D4a SMD", smd_entities(), smd_load, D4A / "interpretation_label")
    summarize("D4a SMD (OmniAnomaly)", E, V, S, I)
    log(f"텍스트 형식: 소수 자릿수 {sorted(set(E.decimals))}, 숫자 외 문자 {E.non_numeric_chars.sum()}, CRLF {E.crlf.any()}")
    log(f"train 끝 -> test 처음 점프 / train 평균 step: median {E.boundary_jump_over_mean_step.median():.2f}, "
        f"max {E.boundary_jump_over_mean_step.max():.2f}")
    log(f"train_len - test_len: {dict(zip(E.entity, E.train_len - E.test_len))}")
    for df, fn in ((E, "d4a_smd_entities"), (V, "d4a_smd_variables"), (S, "d4a_smd_segments"), (I, "d4a_smd_interp")):
        df.to_csv(OUT / f"{fn}.csv", index=False)

    # D4b
    E2, V2, S2, I2 = survey("D4b ASD", asd_entities(), asd_load, D4B / "interpretation_label", asd_seg_info)
    summarize("D4b ASD (InterFusion)", E2, V2, S2, I2)
    log(f"pickle: protocol {sorted(E2.pickle_protocol.unique())}, dtype train {sorted(E2.train_dtype.unique())}, "
        f"test {sorted(E2.test_dtype.unique())}, label {sorted(E2.label_dtype.unique())}, label shape 예 {E2.label_shape.iloc[0]}")
    log(f"train 끝 -> test 처음 점프 / train 평균 step: median {E2.boundary_jump_over_mean_step.median():.2f}, "
        f"max {E2.boundary_jump_over_mean_step.max():.2f}")
    if len(S2):
        log(f"anomaly_type.txt: {len(ATYPE)} 줄; 라벨 구간별 유형: {S2.anomaly_type.replace('', '(없음)').value_counts().to_dict()}")
        log(f"   유형별 해석 변수 수 median: {S2[S2.anomaly_type != ''].groupby('anomaly_type').n_vars.median().to_dict()}")
    for df, fn in ((E2, "d4b_asd_entities"), (V2, "d4b_asd_variables"), (S2, "d4b_asd_segments"), (I2, "d4b_asd_interp")):
        df.to_csv(OUT / f"{fn}.csv", index=False)

    # InterFusion 의 SMD 부분집합 vs OmniAnomaly 원본
    rows = []
    for p in sorted((D4B / "processed").glob("machine-*_train.pkl")):
        ent = p.name[:-len("_train.pkl")]
        tr_o, te_o, y_o, _ = smd_load(ent)
        tr_i, _ = read_numpy_pickle_static(p)
        te_i, _ = read_numpy_pickle_static(D4B / "processed" / f"{ent}_test.pkl")
        y_i, _ = read_numpy_pickle_static(D4B / "processed" / f"{ent}_test_label.pkl")
        r = {"entity": ent, "shape_train_omni": str(tr_o.shape), "shape_train_if": str(tr_i.shape),
             "shape_test_omni": str(te_o.shape), "shape_test_if": str(te_i.shape)}
        if tr_o.shape == tr_i.shape and te_o.shape == te_i.shape:
            r["max_abs_diff_train"] = float(np.abs(tr_o - tr_i).max())
            r["max_abs_diff_test"] = float(np.abs(te_o - te_i).max())
        if len(y_o) == len(y_i):
            r["label_points_diff"] = int(((y_o > .5) != (y_i > .5)).sum())
            r["anomaly_ratio_omni"] = float((y_o > .5).mean())
            r["anomaly_ratio_if"] = float((y_i > .5).mean())
        a, _ = read_interp(D4A / "interpretation_label" / f"{ent}.txt")
        b, _ = read_interp(D4B / "interpretation_label" / f"{ent}.txt")
        sa = {(s, e, tuple(sorted(v))) for s, e, v in a}
        sb = {(s, e, tuple(sorted(v))) for s, e, v in b}
        r.update({"interp_lines_omni": len(a), "interp_lines_if": len(b), "interp_only_omni": len(sa - sb),
                  "interp_only_if": len(sb - sa)})
        rows.append(r)
    X = pd.DataFrame(rows)
    X.to_csv(OUT / "d4_interfusion_smd_crosscheck.csv", index=False)
    log("\n===== InterFusion processed/machine-*.pkl vs OmniAnomaly SMD txt =====")
    log(X.to_string(index=False))

    (OUT / "21_d4_structure.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
