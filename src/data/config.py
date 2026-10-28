"""설정 로드. 경로는 저장소 루트 기준으로 해석한다."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path="configs/data.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["_path"] = str(p)
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = ROOT / v
    cfg["paths"]["cache"].mkdir(parents=True, exist_ok=True)
    return cfg
