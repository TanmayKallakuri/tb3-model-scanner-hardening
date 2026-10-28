"""NumPy artifact scanning.

``.npy`` with an object dtype stores a pickle after the header; ``.npz`` is a
zip of ``.npy`` members (handled by the container dispatcher).  The header is
parsed by hand -- numpy is never asked to load the array, so nothing runs.
"""
from __future__ import annotations

import ast

from modelgate import pklvm
from modelgate.report import Finding

_MAGIC = b"\x93NUMPY"


def scan_npy_bytes(data, location, depth=0, fail_closed=False):
    if data[:6] != _MAGIC:
        return [Finding("unsupported", location, "not a .npy file (bad magic)")] if fail_closed else []
    try:
        major = data[6]
        if major == 1:
            hlen = int.from_bytes(data[8:10], "little")
            hstart = 10
        else:
            hlen = int.from_bytes(data[8:12], "little")
            hstart = 12
        header = data[hstart:hstart + hlen].decode("latin-1")
        body = data[hstart + hlen:]
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"could not parse .npy header: {exc}")] if fail_closed else []

    if not _has_object_dtype(header):
        return []  # plain numeric array: no pickle, nothing executes
    # object array: numpy.lib.format writes the array as one pickle here.
    return pklvm.analyse_streams(body, f"{location}:npy-pickle",
                                 persid_safe=False, depth=depth + 1, fail_closed=fail_closed)


def _has_object_dtype(header: str) -> bool:
    try:
        meta = ast.literal_eval(header)
    except Exception:  # noqa: BLE001
        # be conservative: if 'O' typecode appears, treat as object
        return "'O'" in header or '"O"' in header or "|O" in header or "O8" in header
    descr = meta.get("descr") if isinstance(meta, dict) else None
    return _descr_has_object(descr)


def _descr_has_object(descr) -> bool:
    if isinstance(descr, str):
        return "O" in descr.replace("O8", "O")  # 'O', '|O', 'O8'
    if isinstance(descr, (list, tuple)):
        for item in descr:
            if isinstance(item, (list, tuple)):
                # structured field: (name, dtype[, shape])
                for sub in item[1:]:
                    if _descr_has_object(sub):
                        return True
            elif _descr_has_object(item):
                return True
    return False
