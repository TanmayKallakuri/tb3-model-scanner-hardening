"""NumPy ``.npy`` scanning: an object-dtype array stores a pickle after the
header.  Plain numeric arrays carry no pickle and are clean."""
from __future__ import annotations

import ast

from modelgate.report import Finding

_MAGIC = b"\x93NUMPY"


def is_npy(data: bytes) -> bool:
    return data[:6] == _MAGIC


def scan_npy(data: bytes, location: str, depth: int) -> list[Finding]:
    if not is_npy(data):
        return []
    try:
        major = data[6]
        if major == 1:
            hlen = int.from_bytes(data[8:10], "little"); start = 10 + hlen
            header = data[10:10 + hlen]
        else:
            hlen = int.from_bytes(data[8:12], "little"); start = 12 + hlen
            header = data[12:12 + hlen]
        meta = ast.literal_eval(header.decode("latin-1").strip())
    except Exception as exc:
        return [Finding("unsupported", location, f"bad .npy header: {exc}")]
    descr = meta.get("descr") if isinstance(meta, dict) else None
    # only object arrays ('|O', 'O', 'O8', ...) embed a pickle
    if not (isinstance(descr, str) and "O" in descr):
        return []
    from modelgate import pickle_scan
    try:
        return pickle_scan.scan_pickle(data[start:], f"{location}:npy-object", depth)
    except Exception as exc:
        return [Finding("unsupported", location, f"could not scan .npy pickle: {exc}")]
