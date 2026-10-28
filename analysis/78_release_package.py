"""공개 저장소용 패키지 만들기 (1차 수정 결정 D2 (가), 2026-10-29).

원본 프로젝트는 읽기만 하고 release/pspline-normal-reference/ 에 복사본을 만든다.
- 넣는 것: 공개 데이터 경로(D2·D3·D4·D5·D6·D7·D8)와 시뮬레이션의 코드·설정, 원고에 쓰인 1차 출력, 등록 문서.
- 빼는 것: D1 원자료·D1 로더(대신 안내만 하는 stub)·D1 설정 세부(앵커 좌표·파일 목록), Sensors 원고 전용(측위: 실험 2·5·7,
  src/positioning), 내려받은 원자료, 캐시, 점수 배열(npz, 다시 만들 수 있음), 중간 저장(.partial.csv), 원고 초안, 저작권 PDF.
- 텍스트 파일 안의 이 PC 절대 경로는 '<repo>' 로 바꾼다. 그 밖의 바이트는 그대로 둔다(등록 문서 지문 검증을 위해).
사용법: python -I analysis/78_release_package.py   → release/pspline-normal-reference/, 끝에 SHA256SUMS.txt
"""
import hashlib
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "release" / "pspline-normal-reference"

SRC_FILES = [
    "src/__init__.py", "src/baselines/models.py",
    "src/data/__init__.py", "src/data/build.py", "src/data/config.py", "src/data/d2.py", "src/data/d3.py", "src/data/d4.py",
    "src/data/d6.py", "src/data/d7_uji.py", "src/data/d8.py", "src/data/inject.py", "src/data/labels.py", "src/data/load.py",
    "src/data/safe_pickle.py", "src/data/windows.py",
    "src/metrics/__init__.py", "src/metrics/anomaly.py",
    "src/model/__init__.py", "src/model/bank.py", "src/model/compose.py", "src/model/interp.py", "src/model/pspline.py",
    "src/model/stssd.py",
]
EXPERIMENTS = [
    "node_model_eval.py",
    "exp1_baselines.py", "exp1_baselines_summarize.py", "exp1_simple_baselines.py", "exp1_simple_summarize.py",
    "exp1_stssd.py", "exp1_stssd_original.py", "exp1_stssd_summarize.py",
    "exp3_composition.py", "exp3_summarize.py", "exp3_figures.py",
    "exp4_components.py", "exp4_select.py", "exp4_summarize.py",
    "exp6_conditions.py", "exp6_conditions_summarize.py", "exp6_missing_patterns.py", "exp6_imputation.py", "exp6_d3.py",
    "exp6_phy_masks.py", "exp6_part2.py", "exp6_part2_summarize.py",
    "exp6_comparators_missing.py", "exp6_comparators_missing_summarize.py",
    "exp8_cost.py", "exp9_d4.py", "exp9_summarize.py", "exp10_d6.py", "exp10_design_d1d2.py",
    "expR1_3_thresholds.py", "expR1_3_train_seeds.py", "expR1_4_factorial.py", "expR1_4_factorial_summarize.py",
    "expR1_4_robust.py", "expR1_4_robust_summarize.py", "expR1_5_sim.py", "expR1_5_summarize.py",
    "expR1_6_uji.py", "expR1_7_hymn.py",
]
ANALYSIS = [
    "common.py", "fig_style.py", "00_fetch_d2.py", "12_d2_structure.py", "20_d3_structure.py", "21_d4_structure.py",
    "42_d2_nlos_signature.py", "47_reference_error.py", "48_d3_feasibility.py", "49_phy_loss_library.py",
    "60_math_figures.py", "61_huber_synthetic_check.py", "63_math_paired_stats.py", "64_math_fig1.py", "67_math_tables.py",
    "69_math_fig4_comparators_missing.py", "76_cluster_bootstrap.py", "77_pred_variance.py", "78_release_package.py", "81_supp_revision.py",
]
ANALYSIS_OUT = [
    "42_d2_nlos_signature.txt", "43_exp3_paired.csv", "43_exp3_paired.txt", "47_reference_error.csv", "47_reference_error.txt",
    "48_d3_feasibility.txt", "49_phy_loss_library.txt", "61_huber_synthetic_check.txt",
]
CONFIGS = ["exp1_simple.yaml", "exp3.yaml", "exp4.yaml"]
RESULT_DIRS = (["B0", "D1CFG", "F01", "F02", "H01", "H03", "M01", "M02", "P01", "Z0", "Z0s", "Z1", "Z1t", "Z2", "ZF01"]
               + [f"C{i:02d}" for i in range(16)] + [f"E{i:02d}" for i in range(1, 19)]
               + ["exp1_baselines", "exp1_simple", "exp1_stssd", "exp3", "exp4", "exp6", "exp8", "exp9", "exp10", "expR1"])
RESULT_SKIP_DIRS = {"expR1/hymn_smoke"}
PAPER_FILES = {"paper/mathematics/paired_values.csv": "results/paper/paired_values.csv",
               "paper/mathematics/paired_stats.csv": "results/paper/paired_stats.csv"}
DOCS = [
    "합성방식_실험계획_최종.md", "결정기록_실험3이후.md",
    "실험1_ST-SSD_계획.md", "실험1_비교모델_계획.md", "실험4_계획.md", "실험6_계획.md", "실험8_계획.md", "실험9_계획.md",
    "실험10_계획.md", "분석R1-3_계획.md", "실험R1-4_계획.md", "실험R1-5_계획.md", "실험R1-6_계획.md", "실험R1-7_계획.md",
    "R1_등록_지문.md",
]
DATA_NOTES = ["data_ext/D7_uji/STRUCTURE.md", "data_ext/D8_hymn/STRUCTURE.md", "data_ext/D8_hymn/README.md",
              "data_ext/D4a_SMD/SOURCE.txt", "data_ext/D4b_ASD/SOURCE.txt"]
SKIP_SUFFIX = (".npz", ".pyc", ".pkl", ".partial.csv")
TEXT_SUFFIX = {".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".log", ".tex", ".bib"}

BS = chr(92)
ROOT_STR = str(ROOT)                                     # C:\Users\...\uwb_bspline
PATH_FORMS = [ROOT_STR.replace(BS, BS + BS), ROOT_STR, ROOT_STR.replace(BS, "/")]


def sanitize(b: bytes) -> bytes:
    for form in PATH_FORMS:
        b = b.replace(form.encode("utf-8"), b"<repo>")
    return b


def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    if src.suffix.lower() in TEXT_SUFFIX:
        data = sanitize(data)
    dst.write_bytes(data)


def copy_tree(rel: str):
    base = ROOT / rel
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        r = p.relative_to(ROOT).as_posix()
        if any(r.startswith("results/" + s + "/") for s in RESULT_SKIP_DIRS) or "__pycache__" in r:
            continue
        if p.name.endswith(SKIP_SUFFIX):
            continue
        copy(p, DST / r)


D1_STUB = '''"""D1 (in-house UWB, DWM1001, three rooms) loader — withheld.

The D1 measurements are not public (available from the authors on reasonable request; see the paper's Data
Availability Statement). The original loader read the raw D1 workbooks and is not included. Every analysis on
D1 in this repository therefore cannot be re-run; its primary outputs are included under results/.
"""


def _unavailable(*_, **__):
    raise RuntimeError("D1 data and loader are not public; see src/data/d1.py")


load_raw = prep = cycles = files = _unavailable
'''


def data_yaml():
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text(encoding="utf-8"))
    for k in ("anchors", "anchor_height", "files", "grid"):
        cfg["d1"].pop(k, None)
    cfg["d1"]["note"] = "D1 is in-house and not public; its anchor coordinates and file list are withheld (src/data/d1.py)."
    head = "# Data configuration (release copy). D1 details are withheld; paths are relative to the repository root.\n"
    return head + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)


def main():
    if DST.exists():                                          # .git(공개 저장소 기록)은 남기고 나머지만 지운다
        for c in DST.iterdir():
            if c.name != ".git":
                shutil.rmtree(c) if c.is_dir() else c.unlink()
    for r in SRC_FILES:
        copy(ROOT / r, DST / r)
    (DST / "src/data/d1.py").write_text(D1_STUB, encoding="utf-8", newline="\n")
    for r in EXPERIMENTS:
        copy(ROOT / "experiments" / r, DST / "experiments" / r)
    for r in ANALYSIS:
        copy(ROOT / "analysis" / r, DST / "analysis" / r)
    for r in ANALYSIS_OUT:
        copy(ROOT / "analysis/out" / r, DST / "analysis/out" / r)
    for r in CONFIGS:
        copy(ROOT / "configs" / r, DST / "configs" / r)
    (DST / "configs/data.yaml").write_text(data_yaml(), encoding="utf-8", newline="\n")
    for d in RESULT_DIRS:
        copy_tree("results/" + d)
    for s, t in PAPER_FILES.items():
        copy(ROOT / s, DST / t)
    for r in DOCS:
        copy(ROOT / "doc" / r, DST / "doc" / r)
    copy_tree("doc/registered")
    for r in DATA_NOTES:
        copy(ROOT / r, DST / r)
    for p in sorted((ROOT / "release" / "_meta").iterdir()):    # README, REGISTRATIONS, LICENSE, requirements, git 설정
        if p.is_file():
            shutil.copyfile(p, DST / p.name)
    # 남은 개인 경로 확인
    leaks = []
    for p in DST.rglob("*"):
        if ".git" in p.parts:
            continue
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIX:
            b = p.read_bytes()
            if Path.home().name.encode() in b or b"Desktop" + BS.encode() in b:
                leaks.append(p.relative_to(DST).as_posix())
    files = sorted(p for p in DST.rglob("*") if p.is_file() and ".git" not in p.parts)
    size = sum(p.stat().st_size for p in files)
    print(f"files {len(files)}, {size / 1e6:.1f} MB; personal-path leaks: {leaks or 'none'}")
    big = [(p.stat().st_size, p.relative_to(DST).as_posix()) for p in files if p.stat().st_size > 20e6]
    print("files > 20 MB:", big or "none")
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(DST).as_posix()}" for p in files]
    (DST / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
