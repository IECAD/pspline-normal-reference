"""Mathematics 논문 그림 공통 스타일 (paper/mathematics/figure_table_policy.md §1 의 규칙을 코드로 고정).
- 원寸 작성: MDPI submit 판형의 본문 폭 13.86 cm(TEXT_W) / 전폭 18.46 cm(FULL_W) 그대로 그리고, LaTeX 에서 같은 폭으로 넣는다(축소 없음).
- 글자: Arial 8 pt(눈금·범례 7.5 pt, 최소 7 pt), 패널 라벨 (a)(b)(c) 굵게 9 pt 왼쪽 위. 그림 안 제목 없음(설명은 캡션).
- 두 판: 컬러판(기본, Okabe–Ito, 색 + 마커 + 선 종류) → paper/mathematics/figures/
         흑백판(환경 변수 FIG_MODE=bw 또는 각 스크립트의 --bw) → paper/mathematics/figures_bw/ (파일 이름 같음; 검정·회색 단계 + 마커 + 선 종류,
         히트맵은 회색 단계, 강조 테두리는 검정 바탕선 + 흰 안쪽선)
- 출력: 벡터 PDF(TrueType 임베드, pdf.fonttype 42) + 확인용 PNG 300 dpi. 크기 고정(bbox tight 사용 안 함, constrained layout).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

if "--bw" in sys.argv:
    os.environ["FIG_MODE"] = "bw"
MODE = os.environ.get("FIG_MODE", "color")
OUT_DIR_NAME = "figures_bw" if MODE == "bw" else "figures"

CM = 1 / 2.54
TEXT_W = 13.86 * CM      # in
FULL_W = 18.46 * CM      # in

# Okabe–Ito
OI = {"blue": "#0072B2", "vermillion": "#D55E00", "orange": "#E69F00", "purple": "#CC79A7", "green": "#009E73", "sky": "#56B4E9", "black": "#000000", "grey": "#8C8C8C"}
if MODE == "bw":
    STYLE = {   # 흑백판: 밝기 단계 + 마커 채움/비움 + 선 종류. 본문·표와 같은 이름
        "PS": dict(color="#000000", marker="s", ls="-", lw=2.0, ms=4.5, label="P-spline (PS)"),
        "cell": dict(color="#000000", marker="o", ls="-", lw=1.1, ms=4.8, mfc="white", label="cell mean"),
        "cellfill": dict(color="#555555", marker="^", ls="--", lw=1.2, ms=4.8, mfc="white", label="cell mean + row-mean fill"),
        "cellinterp": dict(color="#666666", marker="D", ls=":", lw=1.6, ms=3.8, label="cell mean + interpolation"),
        "Z1": dict(color="#000000", marker="v", ls="--", lw=1.1, ms=4.5, mfc="white", label="node mean"),
        "comp": dict(color="#9A9A9A", marker=".", ls="-", lw=0.9, ms=3.5, label="nine comparators"),
    }
    # 값 히트맵: 흰색(=결측)과 겹치지 않도록 밝은 회색~진한 회색 구간만 사용
    FIELD_CMAP = LinearSegmentedColormap.from_list("grey_field", ["#D9D9D9", "#1A1A1A"])
    SCORE_CMAP = "Greys"                      # 점수: 낮음 흰색 → 높음 검정
    BOX = ("#000000", "#FFFFFF")              # 강조 테두리: 검정 바탕선 위에 흰 안쪽선(밝은·어두운 칸 모두에서 보임)
    GROUP_MARK_COLOR = "#000000"              # Fig 2(a) 방·환경 마커
    D2_MARK_COLOR = "#777777"
else:
    # 컬러판 팔레트(모두 색각 배려 팔레트). 환경 변수 FIG_PALETTE 로 고른다. 기본 = emphasis(D안)
    PALETTES = {
        "okabe_ito": dict(PS=OI["blue"], cell=OI["vermillion"], cellfill=OI["orange"], cellinterp=OI["purple"], Z1=OI["black"], comp=OI["grey"]),
        "tol_bright": dict(PS="#4477AA", cell="#EE6677", cellfill="#228833", cellinterp="#AA3377", Z1="#000000", comp="#BBBBBB"),     # Paul Tol 'bright'
        "tol_vibrant": dict(PS="#0077BB", cell="#CC3311", cellfill="#EE7733", cellinterp="#009988", Z1="#000000", comp="#BBBBBB"),   # Paul Tol 'vibrant'
        "emphasis": dict(PS="#0072B2", cell="#000000", cellfill="#555555", cellinterp="#888888", Z1="#000000", comp="#C8C8C8"),      # 제안 모델만 색, 나머지 회색 단계
    }
    PALETTE = os.environ.get("FIG_PALETTE", "emphasis")    # 저자 선택 2026-10-13: D안(제안 모델만 색). 이전 기본 okabe_ito
    P_ = PALETTES[PALETTE]
    STYLE = {   # 컬러판: 모델별 고정 표기
        "PS": dict(color=P_["PS"], marker="s", ls="-", lw=1.8, ms=4.5, label="P-spline (PS)"),
        "cell": dict(color=P_["cell"], marker="o", ls="-", lw=1.3, ms=4.8, mfc="white", label="cell mean"),   # 속 빈 원: 회색조에서 PS(■)와 구분
        "cellfill": dict(color=P_["cellfill"], marker="^", ls="--", lw=1.3, ms=4.5, label="cell mean + row-mean fill"),
        "cellinterp": dict(color=P_["cellinterp"], marker="D", ls=":", lw=1.5, ms=3.8, label="cell mean + interpolation"),
        "Z1": dict(color=P_["Z1"], marker="v", ls="--", lw=1.2, ms=4.5, label="node mean"),
        "comp": dict(color=P_["comp"], marker=".", ls="-", lw=0.9, ms=3.5, label="nine comparators"),
    }
    FIELD_CMAP = "viridis"
    SCORE_CMAP = "magma"
    BOX = ("#00e5ff", None)
    GROUP_MARK_COLOR = P_["cell"]             # Fig 2(a) 는 셀 평균 표의 빈 칸 → 셀 평균 색
    D2_MARK_COLOR = P_["cell"]


def setup():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100,
    })


def out_dir(root):
    d = root / "paper" / "mathematics" / OUT_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def panel(ax, letter, x=-0.02, y=1.02):
    """패널 라벨 (a)(b)(c): 축 왼쪽 위, 굵게 9 pt."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


def line(ax, x, y, key, **kw):
    st = {k: v for k, v in STYLE[key].items()}
    st.update(kw)
    lab = st.pop("label"); mfc = st.pop("mfc", st["color"])
    return ax.plot(x, y, label=kw.get("label", lab) if "label" not in kw else kw["label"], **{k: v for k, v in st.items() if k != "label"},
                   markerfacecolor=mfc, markeredgecolor=st["color"], markeredgewidth=1.1)


def box(ax, x, y, w=1, h=1):
    """강조 테두리(임계 초과 칸). 흑백판은 검정 바탕선 + 흰 안쪽선."""
    outer, inner = BOX
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=outer, lw=1.8 if inner else 1.2, zorder=5))
    if inner:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=inner, lw=0.6, zorder=6))


def save(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png")
    w, h = fig.get_size_inches()
    print(f"saved {MODE} {name}: {w / CM:.2f} × {h / CM:.2f} cm")
