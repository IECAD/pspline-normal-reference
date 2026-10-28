"""단순 기준선 사다리 (doc/결정기록_실험3이후.md 2절). F01 과 시간 기저·노드·표준화만 다르다.

사용법:
  python experiments/exp1_simple_baselines.py                    # 설정의 변형 전부, D1·D2, 설정의 분할(기본 검증만)
  python experiments/exp1_simple_baselines.py --ids Z1 Z2 --datasets d2
  python experiments/exp1_simple_baselines.py --splits test      # 테스트 (doc/실험4_계획.md 4절: 최종 모델 확정 후 한 번)
출력: results/<ID>/base/metrics_<데이터>[_<분할>].csv, lambdas_*, config_* (검증만이면 접미사 없음)
평가 코드: experiments/node_model_eval.py (exp3_composition.py 의 run_d1 / run_d2 와 같은 절차)
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from node_model_eval import run_and_save  # noqa: E402
from src.data.config import load_config  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--datasets", nargs="*", default=["d1", "d2"])
    ap.add_argument("--splits", nargs="*")
    ap.add_argument("--config", default="configs/exp1_simple.yaml")
    args = ap.parse_args()
    bcfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    ecfg = yaml.safe_load((ROOT / "configs/exp3.yaml").read_text(encoding="utf-8"))
    dcfg = load_config()
    code = ecfg["ids"][bcfg["base"]]
    for id_ in args.ids or list(bcfg["variants"]):
        run_and_save(id_, code, bcfg["variants"][id_], ecfg, dcfg, args.datasets, args.splits or bcfg["splits"],
                     extra_meta=dict(base=bcfg["base"], baseline_config=args.config))
    print("done. 집계: python experiments/exp1_simple_summarize.py")
