"""Content-based dispatch and in-memory container traversal.

Never extracts to disk, imports model frameworks, or invokes an artifact's
loader. ZIP membership is determined by the central directory; duplicate
names have the same last-entry-wins semantics as ZipFile.read(name).
"""
from __future__ import annotations

import bz2
import gzip
import io
import json
import lzma
import os
import pickletools
import tarfile
import zipfile
import zlib

from modelgate import keras_scan, numpy_scan, pickle_scan, savedmodel_scan
from modelgate.context import AnalysisError, Context
from modelgate.report import Finding, Result

HDF_MAGIC = b"\x89HDF\r\n\x1a\n"
MODEL_EXTS = {".pkl", ".pickle", ".bin", ".joblib", ".pt", ".pth", ".ckpt", ".npy", ".npz", ".h5", ".hdf5", ".keras", ".gz", ".gzip", ".bz2", ".xz", ".zip", ".tar", ".tgz"}
PICKLE_START = {ord(op.code) for op in pickletools.opcodes}


def scan_path(path: str) -> Result:
    context = Context(path)
    if os.path.isdir(path):
        return Result(path, savedmodel_scan.scan_saved_model_dir(path))
    with open(path, "rb") as fh:
        data = fh.read(context.max_bytes + 1)
    return Result(path, scan_data(data, os.path.basename(path), context))


def compression(data):
    if data.startswith(b"\x1f\x8b"): return "gzip"
    if data.startswith(b"BZh"): return "bz2"
    if data.startswith(b"\xfd7zXZ\x00"): return "lzma"
    if len(data) >= 2 and data[0] & 15 == 8 and data[0] >> 4 <= 7 and (data[0] * 256 + data[1]) % 31 == 0:
        return "zlib"
    # joblib also uses the legacy LZMA-alone envelope.
    if data.startswith(b"\x5d\x00\x00"): return "lzma"
    return None


def decompress(data, context, codec=None):
    codec = codec or compression(data)
    context.tick()
    if codec == "gzip": reader = gzip.GzipFile(fileobj=io.BytesIO(data))
    elif codec == "bz2": reader = bz2.BZ2File(io.BytesIO(data))
    elif codec in ("lzma", "xz"): reader = lzma.LZMAFile(io.BytesIO(data))
    elif codec == "zlib":
        obj = zlib.decompressobj()
        value = obj.decompress(data, context.max_bytes + 1)
        if len(value) > context.max_bytes or obj.unconsumed_tail: raise AnalysisError("compressed stream exceeds analysis limit")
        value += obj.flush()
        if not obj.eof: raise AnalysisError("truncated zlib stream")
        context.account(len(value))
        return value
    else: raise AnalysisError("unknown compressed stream")
    with reader:
        value = reader.read(context.max_bytes + 1)
    context.account(len(value))
    return value


def is_hdf(data):
    if data.startswith(HDF_MAGIC): return True
    # An HDF5 user block can precede the superblock, at power-of-two offsets.
    offset = 512
    while offset + 8 <= len(data):
        if data[offset:offset + 8] == HDF_MAGIC: return True
        offset *= 2
    return False


def is_tar(data):
    if len(data) < 512: return False
    try:
        tarfile.TarInfo.frombuf(data[:512], "utf-8", "surrogateescape")
        return True
    except (tarfile.HeaderError, ValueError):
        return len(data) >= 1024 and not any(data[:1024])


def scan_data(data: bytes, location: str, context=None, depth=0, member=False):
    context = context or Context()
    try:
        context.tick()
        if len(data) > context.max_bytes: raise AnalysisError("artifact exceeds analysis size limit")
        if depth > 32: raise AnalysisError("container nesting limit exceeded")
        # Strong format headers come first: raw array/compressed data can
        # itself contain ZIP signatures and an entire central directory.
        codec = compression(data)
        if codec:
            try:
                unpacked = decompress(data, context, codec)
            except (OSError, EOFError, zlib.error, lzma.LZMAError):
                # A generic ZIP may have an arbitrary prefix that happens to
                # resemble a compression header. Its central directory wins.
                if not zipfile.is_zipfile(io.BytesIO(data)): raise
            else:
                return scan_data(unpacked, location + "!" + codec, context, depth + 1, member=member)
        if data.startswith(b"\x93NUMPY"):
            return numpy_scan.scan_npy(data, location, context, depth)
        if is_hdf(data):
            return keras_scan.scan_h5(data, location)
        # ZIP wins over a prepended pickle, but a ZIP inside a pickle operand
        # is just data unless an executing opcode hands it to another loader.
        if zipfile.is_zipfile(io.BytesIO(data)) and not zip_in_pickle_literal(data, context):
            return scan_zip(data, location, context, depth)
        if is_tar(data):
            return scan_tar(data, location, context, depth)
        leaf = location.split("!")[-1].rsplit("/", 1)[-1]
        if leaf in ("saved_model.pb", "saved_model.pbtxt"):
            return savedmodel_scan.scan_saved_model(data, location, text=leaf.endswith("pbtxt"))
        if data and data[0] in PICKLE_START:
            # Only attempt a pickle if its first opcode is parseable. Text
            # cards and numeric metadata can happen to start with an opcode.
            if member and os.path.splitext(leaf)[1].lower() not in MODEL_EXTS and not looks_pickle(data):
                return []
            return pickle_scan.scan_pickle(data, location, context, depth)
        if member and os.path.splitext(leaf)[1].lower() not in MODEL_EXTS:
            # Passive card/metadata/storage members have no model loader.
            return []
        raise AnalysisError("unrecognized artifact content")
    except Exception as exc:
        return [Finding("unsupported", location, f"cannot analyse artifact: {type(exc).__name__}: {exc}")]


def zip_in_pickle_literal(data, context):
    """Do not mistake a serialized byte buffer for an outer ZIP container.

    ZIP tolerates both a prefix and a suffix, so is_zipfile alone also accepts
    a pickle containing ZIP bytes. The operand boundaries distinguish that
    case from a genuine prefixed archive, including a pickle used as a prefix.
    """
    if not data or data[0] not in PICKLE_START: return False
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            central = archive.start_dir
        reader = io.BytesIO(data)
        while reader.tell() <= central:
            contained = False
            for op, arg, pos in pickletools.genops(reader):
                context.tick()
                if isinstance(arg, (bytes, bytearray, str)) and pos < central < reader.tell():
                    contained = True
                if op.name == "STOP" and contained: return True
    except (ValueError, EOFError, zipfile.BadZipFile, UnicodeError):
        pass
    return False


def looks_pickle(data):
    import pickletools
    if data.startswith(b"\x80"): return True
    try:
        # Parsing only, without resolving GLOBAL or executing REDUCE.
        for index, (op, arg, pos) in enumerate(pickletools.genops(data)):
            if op.name == "STOP": return True
            if index > 100000: return True
    except Exception:
        pass
    return False


def scan_zip(data, location, context, depth):
    findings = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = {entry.filename: entry for entry in zf.infolist() if not entry.is_dir()}
        def read(name):
            entry = members[name]
            context.account(entry.file_size)
            return zf.read(entry)
        # Keras's config, not arbitrary data in metadata.json, is executable.
        if all(name in members for name in ("config.json", "metadata.json", "model.weights.h5")):
            return keras_scan.scan_config(read("config.json"), location + "!config.json")
        torch_pickles = [name for name in members if name == "data.pkl" or name.endswith("/data.pkl")]
        torch_prefixes = {name[:-8] for name in torch_pickles if name[:-8] + "version" in members}
        is_torch = bool(torch_prefixes) and data.startswith(b"PK\x03\x04")
        for name, entry in members.items():
            context.tick()
            if is_torch:
                # PyTorch storages and version metadata are not loaded with
                # independent pickle loaders, even if bytes resemble opcodes.
                if not any(name == prefix + "data.pkl" for prefix in torch_prefixes): continue
            findings.extend(scan_data(read(name), location + "!" + name, context, depth + 1, member=True))
    return findings


def scan_tar(data, location, context, depth):
    findings = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        members = {entry.name: entry for entry in archive if entry.isfile()}
        for name, entry in members.items():
            context.account(entry.size)
            reader = archive.extractfile(entry)
            if reader is None: raise AnalysisError("unreadable archive member")
            with reader: payload = reader.read(context.max_bytes + 1)
            findings.extend(scan_data(payload, location + "!" + name, context, depth + 1, member=True))
    return findings


def inspect_loaded(data, location, context, depth, mode="artifact", kwargs=None):
    """Value/effects of one *nested* loader invocation.

    Unlike archive ingestion, a call to a loader observes that loader's format
    and stopping rules. This also preserves a returned callable for an outer
    REDUCE without ever obtaining a real Python callable from the artifact.
    """
    kwargs = kwargs or {}
    context.tick()
    if depth > 32: raise AnalysisError("nested loader limit exceeded")
    if len(data) > context.max_bytes: raise AnalysisError("embedded data exceeds analysis limit")
    if mode in ("torch", "torch-bytes") and kwargs.get("weights_only") is True:
        return [], pickle_scan.UNKNOWN, len(data)
    if mode in ("numpy", "numpy-array"):
        # allow_pickle=False prevents deserialization, rather than making an
        # otherwise unsafe pickle into an executed call.
        if not kwargs.get("allow_pickle", False):
            return [], pickle_scan.UNKNOWN, len(data)
        if data.startswith(b"\x93NUMPY"):
            header, offset = numpy_scan.read_header(data)
            if not numpy_scan.dtype_info(header["descr"])[1]:
                return [], pickle_scan.Obj(pickle_scan.Ref("numpy.ndarray")), len(data)
            child = pickle_scan.Interpreter(location + ":object-array", context, depth + 1)
            reader = io.BytesIO(data)
            reader.seek(offset)
            value = child.read_one(reader)
            return child.findings, value, reader.tell()
        if mode == "numpy" and zipfile.is_zipfile(io.BytesIO(data)):
            # numpy.load on NPZ returns a lazy object. Its members are read on
            # __getitem__; top-level NPZ ingestion is handled by scan_zip.
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                entries = {x.filename: x for x in zf.infolist() if not x.is_dir()}
                values = {}
                for name, entry in entries.items():
                    context.account(entry.file_size)
                    values[name] = zf.read(entry)
            obj = pickle_scan.Obj(pickle_scan.Ref("numpy.lib.npyio.NpzFile"), value=values, state=kwargs)
            return [], obj, len(data)
    if mode in ("artifact", "joblib"):
        codec = compression(data)
        if codec:
            findings, value, _ = inspect_loaded(decompress(data, context, codec), location + "!" + codec, context, depth + 1, "joblib", kwargs)
            return findings, value, len(data)
    if mode == "pandas" and kwargs.get("compression") not in (None, "infer"):
        codec = kwargs["compression"]
        if isinstance(codec, dict): codec = codec.get("method")
        findings, value, _ = inspect_loaded(decompress(data, context, codec), location + "!" + str(codec), context, depth + 1, "pickle", kwargs)
        return findings, value, len(data)
    if mode in ("torch", "torch-bytes") and data.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            entries = {x.filename: x for x in zf.infolist() if not x.is_dir()}
            names = [n for n in entries if n == "data.pkl" or n.endswith("/data.pkl")]
            if not names: raise AnalysisError("torch ZIP has no data.pkl")
            findings, value = [], pickle_scan.UNKNOWN
            for name in names:
                context.account(entries[name].file_size)
                child_findings, value, _ = inspect_loaded(zf.read(entries[name]), location + "!" + name, context, depth + 1, "pickle", kwargs)
                findings.extend(child_findings)
            return findings, value, len(data)
    child = pickle_scan.Interpreter(location, context, depth + 1)
    reader = io.BytesIO(data)
    first = child.read_one(reader)
    value = first
    if mode in ("torch", "torch-bytes") and isinstance(first, int) and first == pickle_scan.TORCH_MAGIC:
        for index in range(1, 5):
            child.memo = {}
            current = child.read_one(reader, index)
            if index == 3: value = current
    return child.findings, value, reader.tell()
