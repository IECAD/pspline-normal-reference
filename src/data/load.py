"""build.py 가 만든 윈도우 데이터셋 불러오기.

예: w = load_windows("d1", "los"); w["X"] (N,S,T), w["rows"], w["cells"]["lab_A"], w["meta"]
    w = load_windows("d1", "los_C_seed42"); w = load_windows("d4", "smd_machine-1-1")
"""
import numpy as np
import pandas as pd

from .config import load_config


def load_windows(dataset: str, name: str, cfg=None, split=None) -> dict:
    cfg = cfg or load_config()
    d = cfg["paths"]["cache"] / "windows" / dataset
    z = np.load(d / f"{name}.npz", allow_pickle=False)
    meta = pd.read_parquet(d / f"{name}_meta.parquet")
    out = dict(X=z["X"], rows=[str(r) for r in z["rows"]], meta=meta,
               cells={k[len("cell_"):]: z[k] for k in z.files if k.startswith("cell_")})
    if split is not None:
        ix = np.flatnonzero(meta["split"].to_numpy() == split)
        out["X"] = out["X"][ix]
        out["cells"] = {k: v[ix] for k, v in out["cells"].items()}
        out["meta"] = meta.iloc[ix].reset_index(drop=True)
    return out


def list_windows(dataset: str, cfg=None):
    cfg = cfg or load_config()
    d = cfg["paths"]["cache"] / "windows" / dataset
    return sorted(p.stem for p in d.glob("*.npz"))
