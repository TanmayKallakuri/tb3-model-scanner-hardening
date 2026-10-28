"""Content-based container traversal. All unpacking happens in memory."""
from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os
import pickletools
import tarfile
import zipfile
import zlib

from modelgate import keras_scan, savedmodel_scan
from modelgate.limits import Budget
from modelgate.report import Finding, Result

MODEL_EXTS = {".pkl", ".pickle", ".joblib", ".bin", ".pt", ".pth", ".ckpt",
              ".npy", ".npz", ".keras", ".h5", ".hdf5", ".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz"}
PICKLE_STARTS = {ord(op.code) for op in pickletools.opcodes}
HDF_MAGIC = b"\x89HDF\r\n\x1a\n"


def scan_path(path: str) -> Result:
    budget = Budget()
    if os.path.isdir(path):
        return Result(path, savedmodel_scan.scan_saved_model_dir(path, budget))
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        # Seekable containers let us inspect metadata without copying large
        # tensor stores into memory (or charging them as expanded pickles).
        if zipfile.is_zipfile(fh):
            fh.seek(0)
            findings, _ = scan_zip(fh, os.path.basename(path), budget, 0, prefix)
            return Result(path, findings)
        size = os.fstat(fh.fileno()).st_size
        offset = 0
        while offset + 8 <= size:
            fh.seek(offset)
            if fh.read(8) == HDF_MAGIC:
                return Result(path, keras_scan.scan_h5(path, os.path.basename(path), budget))
            offset = 512 if offset == 0 else offset * 2
        if prefix.startswith(b"\x93NUMPY"):
            import struct
            from modelgate.numpy_scan import header
            import numpy as np
            fh.seek(8)
            width = 2 if prefix[6] == 1 else 4
            encoded_size = fh.read(width)
            header_size = struct.unpack("<H" if width == 2 else "<I", encoded_size)[0]
            if header_size > budget.max_blob:
                raise ValueError("oversized NumPy header")
            hdr, _ = header(prefix + encoded_size + fh.read(header_size))
            if not np.dtype(hdr["descr"]).hasobject:
                return Result(path, [])
        fh.seek(0)
        data = budget.read(fh)
    return Result(path, scan_bytes(data, os.path.basename(path), budget))


def _is_tar(data):
    if len(data) < 512:
        return False
    if data[257:263] in (b"ustar\x00", b"ustar "):
        return True
    try:
        tarfile.TarInfo.frombuf(data[:512], "utf-8", "surrogateescape")
        return True
    except (tarfile.TarError, ValueError):
        return False


def _is_hdf(data):
    # HDF5 permits a power-of-two user block before its signature.
    offsets = [0]
    offset = 512
    while offset + 8 <= len(data):
        offsets.append(offset)
        offset *= 2
    return any(data[i:i + 8] == HDF_MAGIC for i in offsets)


def decompress(data, budget):
    """Return the logical decompressed byte stream (including all members)."""
    if data.startswith(b"\x1f\x8b"):
        stream = gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb")
    elif data.startswith(b"BZh"):
        stream = bz2.BZ2File(io.BytesIO(data), mode="rb")
    elif data.startswith(b"\xfd7zXZ\x00"):
        stream = lzma.LZMAFile(io.BytesIO(data), mode="rb")
    elif len(data) >= 2 and data[0] & 15 == 8 and int.from_bytes(data[:2], "big") % 31 == 0:
        # joblib's default compression is a zlib stream.
        obj = zlib.decompressobj()
        out = obj.decompress(data, min(budget.max_blob, budget.remaining_bytes) + 1)
        budget.bytes(len(out))
        if obj.unconsumed_tail or not obj.eof:
            raise ValueError("oversized or incomplete zlib stream")
        return out
    else:
        return None
    with stream:
        return budget.read(stream)


def scan_bytes(data: bytes, location: str, budget=None, depth=0, member=False):
    budget = budget or Budget()
    budget.depth(depth)
    try:
        return analyze_bytes(data, location, budget, depth, member)[0]
    except Exception as exc:
        return [Finding("unsupported", location, f"cannot statically parse artifact: {type(exc).__name__}: {exc}")]


def analyze_bytes(data, location, budget, depth=0, member=False):
    from modelgate.pickle_scan import analyze_pickle, UNKNOWN

    budget.depth(depth)
    # zipfile uses the central directory, including concatenated prefixes and
    # duplicate names. Do not search local headers or use the first name match.
    if zipfile.is_zipfile(io.BytesIO(data)):
        return scan_zip(data, location, budget, depth)
    expanded = decompress(data, budget)
    if expanded is not None:
        return analyze_bytes(expanded, location + "!decompressed", budget, depth + 1)
    if _is_tar(data):
        findings = []
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
            members = {m.name: m for m in tf.getmembers()}
            for name, entry in members.items():
                budget.check()
                if not entry.isfile():
                    continue
                source = tf.extractfile(entry)
                if source is not None:
                    with source:
                        payload = budget.read(source)
                    findings.extend(scan_bytes(payload, f"{location}!{name}", budget, depth + 1, member=True))
        return findings, UNKNOWN
    if data.startswith(b"\x93NUMPY"):
        from modelgate.numpy_scan import scan_array
        return scan_array(data, location, budget, depth)
    if _is_hdf(data):
        return keras_scan.scan_h5(data, location, budget), UNKNOWN
    basename = location.replace("\\", "/").rsplit("/", 1)[-1].rsplit("!", 1)[-1]
    if basename in ("saved_model.pb", "saved_model.pbtxt"):
        return savedmodel_scan.scan_saved_model(data, location, basename.endswith("pbtxt"), budget), UNKNOWN
    if data and data[0] in PICKLE_STARTS:
        # Ordinary archive metadata is not itself a pickle. Require a full
        # opcode stream when sniffing an otherwise unrecognised member.
        ext = os.path.splitext(basename)[1].lower()
        if member and ext not in MODEL_EXTS and data[0] != 0x80:
            try:
                next(op for op, _, _ in pickletools.genops(data) if op.name == "STOP")
            except Exception:
                return [], UNKNOWN
        findings, value, _ = analyze_pickle(data, location, budget, depth + 1)
        return findings, value
    if member and os.path.splitext(basename)[1].lower() not in MODEL_EXTS:
        return [], UNKNOWN
    raise ValueError("unrecognised artifact format")


def scan_zip(data, location, budget, depth, prefix=None):
    from modelgate.pickle_scan import analyze_pickle, UNKNOWN

    findings, value = [], UNKNOWN
    if isinstance(data, bytes):
        prefix, source = data[:4], io.BytesIO(data)
    else:
        source = data
    with zipfile.ZipFile(source) as zf:
        entries = {info.filename: info for info in zf.infolist()}
        # A native torch checkpoint is a zip only with a local header at zero.
        pickles = [name for name in entries if name == "data.pkl" or name.endswith("/data.pkl")]
        torch_pickles = [name for name in pickles if name[:-8] + "version" in entries]
        if prefix.startswith(b"PK\x03\x04") and torch_pickles:
            for name in torch_pickles:
                with zf.open(entries[name]) as fh:
                    payload = budget.read(fh)
                fs, value, _ = analyze_pickle(payload, f"{location}!{name}", budget, depth + 1, multiple=False)
                findings.extend(fs)
            return findings, value
        # Configuration is interpreted only in a Keras archive; weights and
        # metadata are data, not an independent collection of model inputs.
        if all(name in entries for name in ("config.json", "metadata.json", "model.weights.h5")):
            with zf.open(entries["config.json"]) as fh:
                config = budget.read(fh)
            return keras_scan.scan_config(config, f"{location}!config.json", budget), UNKNOWN
        for name, info in entries.items():
            budget.check()
            if info.is_dir():
                continue
            with zf.open(info) as fh:
                payload = budget.read(fh)
            findings.extend(scan_bytes(payload, f"{location}!{name}", budget, depth + 1, member=True))
    return findings, value
