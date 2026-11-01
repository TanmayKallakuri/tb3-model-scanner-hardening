"""Content-based format detection and the per-format loaders' semantics.

``analyze_blob`` decides the format of a blob from its bytes (never from a
file name), unpacks containers, and runs the pickle emulator, the Keras
configuration check or the TensorFlow graph check as that format's
standard loader would load it.  Findings are recorded on the ``Interp``.
"""
from __future__ import annotations

import ast
import posixpath
import struct

from modelgate import containers
from modelgate.pickle_vm import TORCH_MAGIC, TORCH_PROTOCOL_VERSION, load_one
from modelgate.symbolic import Exhausted, LoadError, Opaque

MAX_DEPTH = 8
UNSUPPORTED = "unsupported"


def analyze_blob(I, data: bytes, location: str, depth: int = 0):
    if depth > MAX_DEPTH:
        I.finding(UNSUPPORTED, "containers nested too deeply to analyse", location)
        return
    kind = containers.sniff(data)
    if kind == "gzip":
        out, ok = containers.gunzip(data)
        if out or ok:
            analyze_blob(I, out, f"{location}!gunzip", depth + 1)
        return
    if kind == "bz2":
        out, ok = containers.bunzip(data)
        if out or ok:
            analyze_blob(I, out, f"{location}!bunzip2", depth + 1)
        return
    if kind == "xz":
        out, ok = containers.unxz(data)
        if out or ok:
            analyze_blob(I, out, f"{location}!unxz", depth + 1)
        return
    if kind == "zlib":
        out, ok = containers.unzlib(data)
        if out:
            analyze_blob(I, out, f"{location}!zlib", depth + 1)
            return
    if kind == "lzma-alone":
        out, ok = containers.unlzma_alone(data)
        if out:
            analyze_blob(I, out, f"{location}!unlzma", depth + 1)
            return
    if kind == "zip":
        analyze_zip(I, data, location, depth)
        return
    if kind == "tar":
        analyze_tar(I, data, location, depth)
        return
    if kind == "npy":
        try:
            analyze_npy(I, data, location, allow_pickle=True)
        except LoadError:
            pass
        return
    if kind == "hdf5":
        from modelgate import keras_scan

        keras_scan.scan_h5_bytes(I, data, location)
        return
    # anything else is read as pickle stream(s); a zip archive preceded by
    # other bytes is also unpacked (zipfile accepts it)
    if containers.maybe_zip_with_prefix(data):
        analyze_zip(I, data, location, depth)
    analyze_raw_pickle(I, data, location)


# ------------------------------------------------------------------ pickle
def analyze_raw_pickle(I, data: bytes, location: str):
    """``pickle.load`` repeatedly until end of file; a legacy torch
    checkpoint (first pickle is torch's magic number) is loaded the way
    ``torch.load`` does: five pickles, then raw storage bytes."""
    try:
        v0, pos = load_one(I, data, 0, location=location, joblib=True)
    except LoadError:
        return
    if isinstance(v0, int) and not isinstance(v0, bool) and v0 == TORCH_MAGIC:
        try:
            v, pos = load_one(I, data, pos, location=f"{location}#protocol", encoding="utf-8")
            if not (isinstance(v, int) and v == TORCH_PROTOCOL_VERSION):
                return
            _, pos = load_one(I, data, pos, location=f"{location}#sys_info", encoding="utf-8")
            _, pos = load_one(I, data, pos, location=f"{location}#object", encoding="utf-8",
                              persistent="torch-legacy")
            load_one(I, data, pos, location=f"{location}#storage_keys", encoding="utf-8")
        except LoadError:
            pass
        return
    n = 1
    while pos < len(data):
        try:
            _, pos = load_one(I, data, pos, location=f"{location}#stream{n}", joblib=True)
        except LoadError:
            break
        n += 1


# --------------------------------------------------------------------- zip
def _torch_prefix(infos):
    """torch's PyTorchStreamReader takes the archive name from the first
    entry of the central directory."""
    if not infos:
        return None
    first = infos[0].filename
    if "/" not in first:
        return None
    return first.split("/", 1)[0]


def analyze_zip(I, data: bytes, location: str, depth: int):
    try:
        zf, members, infos = containers.zip_members(data)
    except Exception:  # noqa: BLE001 - not a readable zip after all
        return
    names = {n for n, _ in members}
    at_zero = data[:4] == b"PK\x03\x04"
    prefix = _torch_prefix(infos)
    if at_zero and prefix is not None and f"{prefix}/data.pkl" in names:
        info = dict(members)[f"{prefix}/data.pkl"]
        content = containers.zip_read(zf, info, data)
        if content is not None:
            try:
                load_one(I, content, 0, location=f"{location}!{info.filename}", persistent="torch-zip",
                         encoding="utf-8")
            except LoadError:
                pass
        return
    if "config.json" in names and ("model.weights.h5" in names or "metadata.json" in names):
        from modelgate import keras_scan

        content = containers.zip_read(zf, dict(members)["config.json"], data)
        if content is not None:
            keras_scan.scan_config_bytes(I, content, f"{location}!config.json")
        return
    files = [(n, i) for n, i in members if not n.endswith("/")]
    if files and all(n.endswith(".npy") for n, _ in files):
        # numpy.load on an .npz: every member is read with read_array
        for name, info in files:
            content = containers.zip_read(zf, info, data)
            if content is not None and content[:6] == b"\x93NUMPY":
                try:
                    analyze_npy(I, content, f"{location}!{name}", allow_pickle=True)
                except LoadError:
                    pass
        return
    _analyze_members(I, [(n, (lambda i=i: containers.zip_read(zf, i, data))) for n, i in files],
                     location, depth)


def analyze_tar(I, data: bytes, location: str, depth: int):
    try:
        members = containers.tar_members(data)
    except Exception:  # noqa: BLE001
        return
    _analyze_members(I, [(n, (lambda c=c: c)) for n, c in members], location, depth)


def _analyze_members(I, members, location, depth):
    """Load every member of an unpacked archive by its own format.  A
    directory holding saved_model.pb / saved_model.pbtxt is a SavedModel."""
    names = [n for n, _ in members]
    sm_dirs = {}
    for n in names:
        base = posixpath.basename(n)
        if base in ("saved_model.pb", "saved_model.pbtxt"):
            sm_dirs.setdefault(posixpath.dirname(n), set()).add(base)
    for name, get in members:
        d = posixpath.dirname(name)
        in_sm = any(d == s or d.startswith(s + "/") if s else True for s in sm_dirs)
        if in_sm:
            base = posixpath.basename(name)
            sdir = d
            if base in ("saved_model.pb", "saved_model.pbtxt") and sdir in sm_dirs:
                present = sm_dirs[sdir]
                if base == "saved_model.pbtxt" and "saved_model.pb" in present:
                    continue  # TensorFlow reads saved_model.pb first
                content = get()
                if content is not None:
                    from modelgate import savedmodel_scan

                    savedmodel_scan.scan_proto_bytes(I, content, f"{location}!{name}",
                                                     text=base.endswith(".pbtxt"))
            continue
        content = get()
        if content is None:
            continue
        analyze_blob(I, content, f"{location}!{name}", depth + 1)


# --------------------------------------------------------------------- npy
def parse_npy_header(data: bytes):
    """(header dict, offset of the array data) or raise ValueError, as
    numpy.lib.format._read_array_header would."""
    if data[:6] != b"\x93NUMPY" or len(data) < 10:
        raise ValueError("not an npy file")
    major, minor = data[6], data[7]
    if (major, minor) == (1, 0):
        hlen = struct.unpack("<H", data[8:10])[0]
        start = 10
        enc = "latin1"
    elif (major, minor) in ((2, 0), (3, 0)):
        if len(data) < 12:
            raise ValueError("truncated")
        hlen = struct.unpack("<I", data[8:12])[0]
        start = 12
        enc = "latin1" if major == 2 else "utf8"
    else:
        raise ValueError("unsupported npy version")
    raw = data[start:start + hlen]
    if len(raw) < hlen:
        raise ValueError("truncated header")
    header = raw.decode(enc)
    d = ast.literal_eval(header)
    if not isinstance(d, dict):
        raise ValueError("header is not a dict")
    if set(d.keys()) != {"descr", "fortran_order", "shape"}:
        raise ValueError("header keys")
    shape = d["shape"]
    if not isinstance(shape, tuple) or not all(isinstance(x, int) for x in shape):
        raise ValueError("shape")
    if not isinstance(d["fortran_order"], bool):
        raise ValueError("fortran_order")
    return d, start + hlen


_SIMPLE_DESCR = None


def descr_has_object(descr) -> bool:
    global _SIMPLE_DESCR
    if _SIMPLE_DESCR is None:
        import re

        _SIMPLE_DESCR = re.compile(r"^[<>|=]?(?:[?bBhHiIlLqQpPefdgFDGSUVaucmMO]\d*|[mM]8\[\w+\])$")
    if isinstance(descr, str) and _SIMPLE_DESCR.match(descr):
        return "O" in descr  # simple type codes: only 'O' holds Python objects
    try:
        import numpy as np
        from numpy.lib import format as npformat

        dt = npformat.descr_to_dtype(descr)
        return bool(dt.hasobject)
    except Exception:  # noqa: BLE001
        pass
    if isinstance(descr, str):
        return "O" in descr
    if isinstance(descr, list):
        return any(descr_has_object(f[1]) for f in descr if isinstance(f, (tuple, list)) and len(f) >= 2)
    raise ValueError("bad descr")


def analyze_npy(I, data: bytes, location: str, allow_pickle=True, encoding="ASCII"):
    """numpy.lib.format.read_array: an object array is one pickle."""
    try:
        d, off = parse_npy_header(data)
        has_obj = descr_has_object(d["descr"])
    except Exception:  # noqa: BLE001 - numpy raises; nothing is unpickled
        return None
    if not has_obj:
        return Opaque("ndarray")
    if not allow_pickle:
        raise LoadError("Object arrays cannot be loaded when allow_pickle=False")
    v, _ = load_one(I, data, off, location=f"{location}@pickle", encoding=encoding)
    return v


# ---------------------------------------------------------- nested loading
def torch_load_bytes(I, data: bytes, location: str):
    if data[:4] == b"PK\x03\x04":
        zf, members, infos = containers.zip_members(data)
        prefix = _torch_prefix(infos)
        names = dict(members)
        if prefix is None or f"{prefix}/data.pkl" not in names:
            raise LoadError("not a torch archive")
        content = containers.zip_read(zf, names[f"{prefix}/data.pkl"], data)
        if content is None:
            raise LoadError("cannot read data.pkl")
        v, _ = load_one(I, content, 0, location=f"{location}!data.pkl", persistent="torch-zip",
                        encoding="utf-8")
        return v
    v0, pos = load_one(I, data, 0, location=f"{location}#magic", encoding="utf-8")
    if not (isinstance(v0, int) and v0 == TORCH_MAGIC):
        raise LoadError("Invalid magic number; corrupt file?")
    v, pos = load_one(I, data, pos, location=f"{location}#protocol", encoding="utf-8")
    if not (isinstance(v, int) and v == TORCH_PROTOCOL_VERSION):
        raise LoadError("Invalid protocol version")
    _, pos = load_one(I, data, pos, location=f"{location}#sys_info", encoding="utf-8")
    result, pos = load_one(I, data, pos, location=f"{location}#object", encoding="utf-8",
                           persistent="torch-legacy")
    load_one(I, data, pos, location=f"{location}#storage_keys", encoding="utf-8")
    return result


def numpy_load_bytes(I, data: bytes, location: str, allow_pickle: bool, encoding="ASCII"):
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return Opaque("NpzFile (members are loaded lazily)")
    if data[:6] == b"\x93NUMPY":
        try:
            d, off = parse_npy_header(data)
            has_obj = descr_has_object(d["descr"])
        except Exception:  # noqa: BLE001
            raise LoadError("bad npy header") from None
        if not has_obj:
            return Opaque("ndarray")
        if not allow_pickle:
            raise LoadError("Object arrays cannot be loaded when allow_pickle=False")
        v, _ = load_one(I, data, off, location=f"{location}@pickle", encoding=encoding)
        return v
    if not allow_pickle:
        raise LoadError("Cannot load file containing pickled data when allow_pickle=False")
    v, _ = load_one(I, data, 0, location=location, encoding=encoding)
    return v


def joblib_load_bytes(I, data: bytes, location: str):
    kind = containers.sniff(data)
    if kind == "gzip":
        data, _ = containers.gunzip(data)
    elif kind == "bz2":
        data, _ = containers.bunzip(data)
    elif kind in ("xz",):
        data, _ = containers.unxz(data)
    elif kind == "zlib":
        data, _ = containers.unzlib(data)
    elif kind == "lzma-alone":
        data, _ = containers.unlzma_alone(data)
    v, _ = load_one(I, data, 0, location=location, joblib=True)
    return v
