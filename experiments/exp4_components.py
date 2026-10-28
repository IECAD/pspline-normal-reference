"""실험 4: 구성요소 ablation -> 최종 모델 (doc/실험4_계획.md). 선택 규칙은 exp4_select.py.

사용법:
  python experiments/exp4_components.py                     # C00-C15, D1·D2, 검증
  python experiments/exp4_components.py --ids M01 M02       # results/exp4/selection.json 의 결합 구성 (검증)
  python experiments/exp4_components.py --all --splits test # 선택 확정 후 테스트 한 번 (C00-C15, M01, M02)
출력: results/<ID>/base/metrics_<데이터>[_test].csv 등 (experiments/node_model_eval.py)
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from node_model_eval import run_and_save  # noqa: E402
from src.data.config import load_config  # noqa: E402

SEL = ROOT / "results" / "exp4" / "selection.json"


def specs(cfg):
    out = dict(cfg["variants"])
    if SEL.exists():
        sel = json.loads(SEL.read_text(encoding="utf-8"))
        for k in ("M01", "M02"):
            if k in sel:
                out[k] = sel[k]["spec"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--all", action="store_true", help="C00-C15 와 (있으면) M01, M02")
    ap.add_argument("--datasets", nargs="*", default=["d1", "d2"])
    ap.add_argument("--splits", nargs="*", default=["val"])
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/exp4.yaml").read_text(encoding="utf-8"))
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    code = ecfg["ids"][cfg["base"]]
    S = specs(cfg)
    ids = list(S) if args.all else (args.ids or list(cfg["variants"]))
    for id_ in ids:
        run_and_save(id_, code, S[id_], ecfg, dcfg, args.datasets, args.splits,
                     extra_meta=dict(base=cfg["base"], exp4_config="configs/exp4.yaml"))
    print("done. 선택: python experiments/exp4_select.py")
