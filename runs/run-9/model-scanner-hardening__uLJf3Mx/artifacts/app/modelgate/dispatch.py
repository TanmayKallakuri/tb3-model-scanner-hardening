"""Content-based, in-memory dispatch. Archives are never extracted to disk."""
from __future__ import annotations

import ast
import io
import json
import os
import struct
import tarfile
import zipfile

from modelgate import keras_scan, pickle_scan, savedmodel_scan
from modelgate.limits import LimitError, decompress, read_limited, MAX_BYTES
from modelgate.report import Result, Finding
from modelgate.symbolic import Context, UNKNOWN, Symbol, AnalysisError

HDF_MAGIC = b"\x89HDF\r\n\x1a\n"


def _compression(data):
    if data.startswith(b"\x1f\x8b"): return "gzip"
    if data.startswith(b"BZh"): return "bz2"
    if data.startswith(b"\xfd7zXZ\x00"): return "xz"
    # joblib's zlib wrapper has no filename or extension requirement.
    if len(data) > 2 and data[0] == 0x78 and (data[0] * 256 + data[1]) % 31 == 0:
        return "zlib"
    return None


def _hdf(data):
    # HDF5 permits a user block before its superblock.
    return any(data[p:p + 8] == HDF_MAGIC for p in (0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536))


def _npy(data, location, ctx, allow_pickle=True):
    if len(data) < 10: raise AnalysisError("truncated NumPy header")
    version = tuple(data[6:8])
    if version == (1, 0): size, begin = struct.unpack_from("<H", data, 8)[0], 10
    elif version in ((2, 0), (3, 0)): size, begin = struct.unpack_from("<I", data, 8)[0], 12
    else: raise AnalysisError("unsupported NumPy version")
    if begin + size > len(data): raise AnalysisError("truncated NumPy header")
    header = ast.literal_eval(data[begin:begin + size].decode("utf-8" if version == (3, 0) else "latin1"))
    hasobject, width = pickle_scan.dtype_info(header["descr"])
    if hasobject:
        if not allow_pickle: return UNKNOWN
        result, _, _ = pickle_scan.analyze_pickle(data[begin + size:], location + "/array", ctx, streams=1)
        return result
    return Symbol("instance", "numpy.ndarray", (), {"shape": header.get("shape"), "dtype": header["descr"]})


def _zip(data, location, ctx, depth, loader):
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # Name lookup in zipfile uses the last central-directory entry, even if
        # an earlier local header (or a duplicate name) suggests another file.
        members = {entry.filename: entry for entry in zf.infolist() if not entry.is_dir()}
        def read(name):
            info = members[name]
            if info.file_size > MAX_BYTES: raise LimitError("zip member size limit")
            ctx.budget.tick()
            with zf.open(info, "r") as fh: return read_limited(fh)
        torch_names = [n for n in members if n == "data.pkl" or n.endswith("/data.pkl")]
        is_torch = data.startswith(b"PK\x03\x04") and torch_names and any(n == "version" or n.endswith("/version") for n in members)
        if is_torch:
            value = UNKNOWN
            for name in torch_names:
                value, _, _ = pickle_scan.analyze_pickle(read(name), f"{location}!{name}", ctx, streams=1, persistent=True)
            return value
        if "config.json" in members and ("metadata.json" in members or "model.weights.h5" in members):
            ctx.findings.extend(keras_scan.scan_config(read("config.json"), location + "!config.json"))
            return UNKNOWN
        saved = {n for n in members if n.rsplit("/", 1)[-1] in ("saved_model.pb", "saved_model.pbtxt")}
        # Prefer the binary SavedModel when both representations exist, just as
        # TensorFlow does. Other models in the archive are still independent.
        for name in sorted(saved):
            if name.endswith(".pbtxt") and name[:-3] in saved: continue
            ctx.findings.extend(savedmodel_scan.scan_saved_model(read(name), f"{location}!{name}", text=name.endswith("pbtxt")))
        roots = [n.rsplit("/", 1)[0] + "/" if "/" in n else "" for n in saved]
        for name in members:
            if name in saved: continue
            if any(name.startswith(root + "variables/") or name.startswith(root + "assets/") for root in roots): continue
            if name.endswith("/"): continue
            analyze_bytes(read(name), f"{location}!{name}", ctx, depth=depth + 1, auxiliary=True)
        return UNKNOWN


def _tar(data, location, ctx, depth):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
        members = {m.name: m for m in tf if m.isfile()}
        def read(name):
            if members[name].size > MAX_BYTES: raise LimitError("tar member size limit")
            fh = tf.extractfile(members[name])
            if fh is None: raise AnalysisError("unreadable tar member")
            return read_limited(fh)
        saved = {n for n in members if n.rsplit("/", 1)[-1] in ("saved_model.pb", "saved_model.pbtxt")}
        for name in sorted(saved):
            if name.endswith(".pbtxt") and name[:-3] in saved: continue
            ctx.findings.extend(savedmodel_scan.scan_saved_model(read(name), f"{location}!{name}", text=name.endswith("pbtxt")))
        roots = [n.rsplit("/", 1)[0] + "/" if "/" in n else "" for n in saved]
        for name in members:
            if name in saved: continue
            if any(name.startswith(root + "variables/") or name.startswith(root + "assets/") for root in roots): continue
            analyze_bytes(read(name), f"{location}!{name}", ctx, depth=depth + 1, auxiliary=True)
    return UNKNOWN


def analyze_bytes(data: bytes, location: str, ctx: Context, *, depth=0, loader=None, auxiliary=False, allow_pickle=True):
    ctx.budget.charge(len(data))
    if depth > 12: raise AnalysisError("container nesting limit")
    kind = _compression(data)
    if kind:
        return analyze_bytes(decompress(data, kind), location + "/" + kind, ctx, depth=depth + 1, loader=loader, auxiliary=auxiliary, allow_pickle=allow_pickle)
    if data.startswith(b"\x93NUMPY"):
        return _npy(data, location, ctx, allow_pickle)
    if _hdf(data):
        ctx.findings.extend(keras_scan.scan_h5(data, location))
        return UNKNOWN
    # Tar has a checksum rather than a unique initial signature. The stdlib
    # header parser checks it without interpreting any member data.
    if len(data) >= 512:
        try: header = tarfile.TarInfo.frombuf(data[:512], "utf-8", "surrogateescape")
        except tarfile.HeaderError: header = None
        if header is not None: return _tar(data, location, ctx, depth)
    if zipfile.is_zipfile(io.BytesIO(data)):
        if loader == "torch" and not data.startswith(b"PK\x03\x04"):
            # torch.load does not use zipfile.is_zipfile: it checks the first
            # four bytes. A prefixed ZIP instead enters the legacy unpickler.
            value, _, _ = pickle_scan.analyze_pickle(data, location, ctx, streams=1)
            return value
        return _zip(data, location, ctx, depth, loader)
    basename = location.rsplit("!", 1)[-1].rsplit("/", 1)[-1]
    if basename in ("saved_model.pb", "saved_model.pbtxt"):
        ctx.findings.extend(savedmodel_scan.scan_saved_model(data, location, text=basename.endswith("pbtxt")))
        return UNKNOWN
    if auxiliary:
        # Model cards and archive metadata are inert. Detection of model and
        # container bytes above always takes priority over the member's name.
        try:
            json.loads(data)
            return UNKNOWN
        except (ValueError, UnicodeError): pass
    if loader == "numpy" and not allow_pickle: return UNKNOWN
    result, _, _ = pickle_scan.analyze_pickle(data, location, ctx,
        streams=1 if loader in ("numpy", "joblib") else None,
        persistent=loader == "torch", joblib=loader != "numpy")
    return result


def scan_path(path: str) -> Result:
    ctx = Context()
    try:
        if os.path.isdir(path):
            ctx.findings.extend(savedmodel_scan.scan_saved_model_dir(path))
        else:
            with open(path, "rb") as fh: data = read_limited(fh)
            analyze_bytes(data, os.path.basename(path), ctx)
    except Exception as exc:
        ctx.findings.append(Finding("unsupported", path, f"cannot analyse artifact: {type(exc).__name__}: {exc}"))
    return Result(path, ctx.findings)
