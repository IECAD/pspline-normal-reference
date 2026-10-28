"""numpy ndarray pickle 을 실행 없이 읽는다 (D4b ASD 의 .pkl 용).

pickle.load 는 쓰지 않는다. pickletools.genops 로 opcode 만 읽고, numpy ndarray 재구성 템플릿
(전역 3개: numpy.core.multiarray._reconstruct, numpy.ndarray, numpy.dtype)과 정확히 일치할 때만
원시 바이트를 np.frombuffer 로 해석한다. 다르면 ValueError. (analysis/21_d4_structure.py 에서 가져옴)
"""
import pickletools
from pathlib import Path

import numpy as np

_SKIP = {"BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE", "FRAME"}
_ALLOWED_OPS = {"PROTO", "GLOBAL", "BININT1", "BININT2", "BININT", "TUPLE1", "TUPLE2", "TUPLE3", "TUPLE",
                "SHORT_BINBYTES", "BINBYTES", "BINBYTES8", "REDUCE", "MARK", "BINUNICODE", "SHORT_BINUNICODE",
                "NONE", "BUILD", "NEWFALSE", "NEWTRUE", "STOP"}
_GLOBALS = ["numpy.core.multiarray _reconstruct", "numpy ndarray", "numpy dtype"]
_INT = {"BININT1", "BININT2", "BININT"}
_ANY = object()


def read_numpy_pickle_static(path: Path) -> np.ndarray:
    data = Path(path).read_bytes()
    ops = [(op.name, arg) for op, arg, _ in pickletools.genops(data) if op.name not in _SKIP]
    names = {n for n, _ in ops}
    if names - _ALLOWED_OPS:
        raise ValueError(f"{path}: unexpected opcodes {names - _ALLOWED_OPS}")
    if [a for n, a in ops if n == "GLOBAL"] != _GLOBALS:
        raise ValueError(f"{path}: unexpected globals")
    i = 0

    def take(name, value=_ANY):
        nonlocal i
        n, a = ops[i]
        ok = (n in name) if isinstance(name, set) else (n == name)
        if not ok or (value is not _ANY and a != value):
            raise ValueError(f"{path}: op#{i} {n}={a!r}, expected {name}={value!r}")
        i += 1
        return a

    take("PROTO")
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
        raise ValueError(f"{path}: trailing ops")
    dt = np.dtype(border + dstr)
    if dt.kind not in "fiub":
        raise ValueError(f"{path}: non-numeric dtype {dt}")
    if len(payload) != int(np.prod(shape)) * dt.itemsize:
        raise ValueError(f"{path}: payload size mismatch")
    return np.frombuffer(payload, dtype=dt).reshape(shape, order="F" if fortran else "C").copy()


# ----------------------------------------------------------------------------- pandas DataFrame pickle (D6 TWR-CLOUD-Bern)
import io  # noqa: E402
import pickle  # noqa: E402

# DataFrame 복원에 필요한 생성자만 허용한다 (명시 목록). 목록 밖 전역을 요청하면 그 전에 멈춘다 (실행 없음).
PANDAS_ALLOW = {
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.internals.blocks", "new_block"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("builtins", "slice"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, f, allow):
        super().__init__(f)
        self.allow, self.seen = allow, []

    def find_class(self, module, name):
        self.seen.append((module, name))
        if (module, name) not in self.allow:
            raise pickle.UnpicklingError(f"허용 목록 밖 전역: {module}.{name}")
        return super().find_class(module, name)


def read_pandas_pickle_restricted(path, allow=PANDAS_ALLOW):
    """허용 목록의 pandas·numpy 생성자만 쓰는 제한 unpickler 로 DataFrame 을 읽는다. 반환: (DataFrame, 사용한 전역 목록)."""
    with open(path, "rb") as f:
        u = _RestrictedUnpickler(io.BufferedReader(f), allow)
        obj = u.load()
    import pandas as pd
    if not isinstance(obj, pd.DataFrame):
        raise ValueError(f"{path}: DataFrame 이 아님 ({type(obj).__name__})")
    return obj, sorted(set(u.seen))
