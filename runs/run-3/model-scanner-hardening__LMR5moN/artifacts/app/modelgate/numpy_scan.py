"""NumPy header parsing; object arrays contain a pickle, numeric arrays do not."""
from __future__ import annotations

import ast
import struct

from modelgate.limits import Budget

MAGIC = b"\x93NUMPY"


def header(data):
    if not data.startswith(MAGIC) or len(data) < 10:
        raise ValueError("invalid NumPy array header")
    major, minor = data[6:8]
    if (major, minor) == (1, 0):
        size, start, encoding = struct.unpack_from("<H", data, 8)[0], 10, "latin1"
    elif (major, minor) in ((2, 0), (3, 0)):
        size, start = struct.unpack_from("<I", data, 8)[0], 12
        encoding = "utf-8" if major == 3 else "latin1"
    else:
        raise ValueError("unsupported NumPy header version")
    if start + size > len(data):
        raise ValueError("truncated NumPy header")
    value = ast.literal_eval(data[start:start + size].decode(encoding).strip())
    if not isinstance(value, dict) or not all(k in value for k in ("descr", "shape", "fortran_order")):
        raise ValueError("invalid NumPy header fields")
    return value, start + size


def scan_array(data, location, budget=None, depth=0, allow_pickle=True):
    from modelgate.pickle_scan import analyze_pickle, Instance, Ref

    budget = budget or Budget()
    budget.depth(depth)
    hdr, offset = header(data)
    # np.dtype constructs only a dtype from literal header data. It never
    # deserializes Python objects or reads array payloads.
    import numpy as np
    dtype = np.dtype(hdr["descr"])
    if dtype.hasobject and allow_pickle:
        findings, value, consumed = analyze_pickle(data[offset:], location + ":object-array", budget, depth + 1, multiple=False)
        return findings, value
    return [], Instance(Ref("numpy.ndarray"))
