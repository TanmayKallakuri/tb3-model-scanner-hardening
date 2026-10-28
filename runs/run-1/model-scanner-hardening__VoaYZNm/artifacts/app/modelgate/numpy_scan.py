"""Minimal NPY header reader. Numeric array data is never pickle-scanned."""
from __future__ import annotations

import ast
import math
import re
import struct
from modelgate.context import AnalysisError


def dtype_info(desc):
    """Return (itemsize, contains_objects) without constructing a dtype."""
    if isinstance(desc, str):
        value = desc.lstrip("<>=|")
        if value in ("object", "object_", "O", "O8", "O4"): return 8, True
        named = {"bool": 1, "bool_": 1, "byte": 1, "ubyte": 1, "int8": 1, "uint8": 1,
                 "int16": 2, "uint16": 2, "int32": 4, "uint32": 4, "int64": 8, "uint64": 8,
                 "float16": 2, "float32": 4, "float64": 8, "complex64": 8, "complex128": 16}
        if value in named: return named[value], False
        match = re.fullmatch(r"([?bBhHiIuUlLqQefdcgFDGOSVmM])(\d*)(?:\[.*\])?", value)
        if match:
            kind, size = match.groups()
            default = {"?": 1, "b": 1, "B": 1, "h": 2, "H": 2, "i": 4, "I": 4, "l": 8, "L": 8,
                       "q": 8, "Q": 8, "u": 4, "c": 16, "e": 2, "f": 4, "d": 8, "g": 16, "F": 8, "D": 16,
                       "G": 32, "m": 8, "M": 8, "O": 8, "S": 0, "U": 0, "V": 0}
            return int(size) * (4 if kind == "U" else 1) if size else default[kind], kind == "O"
        raise AnalysisError("unknown numpy dtype " + repr(desc))
    if isinstance(desc, tuple) and len(desc) == 2:
        size, obj = dtype_info(desc[0])
        shape = desc[1] if isinstance(desc[1], (tuple, list)) else (desc[1],)
        return size * math.prod(shape), obj
    if isinstance(desc, list):
        total, hasobject = 0, False
        for field in desc:
            if not isinstance(field, (tuple, list)) or len(field) < 2: raise AnalysisError("invalid structured dtype")
            size, obj = dtype_info(field[1])
            if len(field) > 2:
                shape = field[2] if isinstance(field[2], (tuple, list)) else (field[2],)
                size *= math.prod(shape)
            total += size
            hasobject |= obj
        return total, hasobject
    if isinstance(desc, dict) and "formats" in desc:
        infos = [dtype_info(d) for d in desc["formats"]]
        offsets = desc.get("offsets")
        size = max((offset + info[0] for offset, info in zip(offsets, infos)), default=0) if offsets else sum(i[0] for i in infos)
        return desc.get("itemsize", size), any(i[1] for i in infos)
    raise AnalysisError("invalid numpy dtype description")


def read_header(data):
    if not data.startswith(b"\x93NUMPY") or len(data) < 10:
        raise AnalysisError("invalid NPY header")
    version = tuple(data[6:8])
    if version == (1, 0):
        size = struct.unpack_from("<H", data, 8)[0]
        start, encoding = 10, "latin1"
    elif version in ((2, 0), (3, 0)):
        size = struct.unpack_from("<I", data, 8)[0]
        start, encoding = 12, "utf-8" if version == (3, 0) else "latin1"
    else:
        raise AnalysisError("unknown NPY version")
    end = start + size
    if end > len(data): raise AnalysisError("truncated NPY header")
    header = ast.literal_eval(data[start:end].decode(encoding))
    if not isinstance(header, dict) or not all(k in header for k in ("descr", "shape", "fortran_order")):
        raise AnalysisError("invalid NPY header mapping")
    return header, end


def scan_npy(data, location, context, depth=0):
    header, offset = read_header(data)
    size, hasobject = dtype_info(header["descr"])
    if hasobject:
        from modelgate.pickle_scan import scan_pickle
        return scan_pickle(data[offset:], location + ":object-array", context, depth + 1, sequence=False)
    shape = header["shape"]
    if not isinstance(shape, tuple) or not all(isinstance(n, int) and n >= 0 for n in shape):
        raise AnalysisError("invalid NPY shape")
    if offset + math.prod(shape) * size > len(data):
        raise AnalysisError("truncated NPY numeric data")
    return []
