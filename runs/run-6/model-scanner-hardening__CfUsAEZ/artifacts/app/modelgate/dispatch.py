"""Content-based, in-memory parsing of model files and nested containers."""
from __future__ import annotations

import ast
import bz2
import io
import lzma
import os
import struct
import tarfile
import zipfile
import zlib

from modelgate import keras_scan, pickle_scan, savedmodel_scan
from modelgate.abstract import World, Detected
from modelgate.limits import AnalysisLimit
from modelgate.report import Result, Finding

MAX_EXPANSION = 256 * 1024 * 1024


def decompress_one(data, fmt, budget, args=(), kwargs=None):
    kwargs = kwargs or {}
    if fmt in ("zlib", "gzip"):
        wbits = args[0] if args else kwargs.get("wbits", 31 if fmt == "gzip" else 15)
        dec = zlib.decompressobj(wbits)
        out = dec.decompress(data, MAX_EXPANSION + 1)
        if len(out) > MAX_EXPANSION or dec.unconsumed_tail:
            raise AnalysisLimit("decompressed data exceeds limit")
        if not dec.eof: raise ValueError("truncated compressed stream")
        rest = dec.unused_data
    else:
        dec = bz2.BZ2Decompressor() if fmt == "bz2" else lzma.LZMADecompressor(memlimit=256*1024*1024)
        out = dec.decompress(data, max_length=MAX_EXPANSION + 1)
        if len(out) > MAX_EXPANSION: raise AnalysisLimit("decompressed data exceeds limit")
        if not dec.eof: raise ValueError("truncated compressed stream")
        rest = dec.unused_data
    budget.expand(len(out))
    if rest and fmt != "zlib":
        if fmt == "gzip": rest = rest.lstrip(b"\0")
        if rest: out += decompress_one(rest, fmt, budget)
    return out


def compression(data):
    if data.startswith(b"\x1f\x8b"): return "gzip"
    if data.startswith(b"BZh"): return "bz2"
    if data.startswith(b"\xfd7zXZ\x00"): return "lzma"
    if data.startswith(b"\x5d\x00\x00") and len(data) >= 13: return "lzma"
    # joblib's default compressed form is a bare zlib stream.
    if len(data) > 2 and data[0] & 15 == 8 and data[0] >> 4 <= 7 and (data[0]*256 + data[1]) % 31 == 0:
        return "zlib"
    return None


def decompress(data, budget):
    for _ in range(8):
        fmt = compression(data)
        if not fmt: return data
        data = decompress_one(data, fmt, budget)
    raise AnalysisLimit("compression nesting limit exceeded")


def _is_tar(data):
    if len(data) < 512: return False
    block = data[:512]
    if block == b"\0" * 512: return True
    try:
        expected = int(block[148:156].strip(b"\0 ") or b"0", 8)
        return expected == sum(block[:148]) + 8*32 + sum(block[156:])
    except ValueError:
        return False


def _is_h5(data):
    # HDF5 permits a user block before its superblock.
    offset = 0
    while offset + 8 <= len(data):
        if data[offset:offset+8] == b"\x89HDF\r\n\x1a\n": return True
        offset = 512 if offset == 0 else offset * 2
    return False


def _dtype_object(descr):
    if isinstance(descr, str):
        plain = descr.lstrip("<>=|")
        if plain.startswith("O") or plain in ("object", "object_"): return True
        if any(char in descr for char in (",", "(", "[")) and not plain.startswith(("M8", "m8", "datetime64", "timedelta64")):
            # NumPy also accepts compact structured descriptors such as 'i4,O'
            # and subarray descriptors. Interpret only the dtype, never its data.
            from modelgate.parser_env import prepare
            prepare()
            import numpy as np
            return np.dtype(descr).hasobject
        return False
    if isinstance(descr, list):
        return any(isinstance(field, (list, tuple)) and len(field) >= 2 and _dtype_object(field[1]) for field in descr)
    if isinstance(descr, tuple): return _dtype_object(descr[0])
    return False


def read_npy_header(data):
    if len(data) < 10: raise ValueError("truncated NumPy header")
    version = data[6:8]
    if version == b"\x01\x00":
        size, start, encoding = struct.unpack_from("<H", data, 8)[0], 10, "latin1"
    elif version in (b"\x02\x00", b"\x03\x00"):
        size, start, encoding = struct.unpack_from("<I", data, 8)[0], 12, "utf-8" if version[0] == 3 else "latin1"
    else: raise ValueError("unrecognised NumPy format version")
    if size > 4 * 1024 * 1024 or start + size > len(data): raise ValueError("invalid NumPy header size")
    header = ast.literal_eval(data[start:start+size].decode(encoding).strip())
    if not isinstance(header, dict) or not {"descr", "shape", "fortran_order"} <= header.keys():
        raise ValueError("invalid NumPy header")
    return header, start + size


def scan_npy(data, location, world):
    header, offset = read_npy_header(data)
    if _dtype_object(header["descr"]):
        return pickle_scan.scan_pickle(data[offset:], location + ":array", world=world, one=True)
    return []  # Numeric array bytes are never a pickle, whatever they resemble.


def _members(files, location, world, depth):
    findings = []
    # TensorFlow model directories can occur inside any archive. Their variable
    # files contain tensor bytes; never treat those as standalone pickle data.
    tf_dirs = {name.rsplit("/", 1)[0] if "/" in name else "" for name in files
               if name.rsplit("/", 1)[-1] in ("saved_model.pb", "saved_model.pbtxt")}
    for name, read in files.items():
        world.budget.tick()
        if name.endswith("/"): continue
        loc = f"{location}!{name}"
        base = name.rsplit("/", 1)[-1]
        parent = name.rsplit("/", 1)[0] if "/" in name else ""
        if base in ("saved_model.pb", "saved_model.pbtxt"):
            if base.endswith("pbtxt") and (parent + "/" if parent else "") + "saved_model.pb" in files:
                continue
            data = read()
            findings.extend(savedmodel_scan.scan_saved_model(data, loc, text=base.endswith("pbtxt")))
            continue
        if any(name.startswith((p + "/" if p else "") + "variables/") for p in tf_dirs): continue
        findings.extend(scan_bytes(read(), loc, world=world, depth=depth+1, member=True))
    return findings


def scan_zip(source, location, world, depth=0, mode=None, local_header=True):
    with zipfile.ZipFile(source) as archive:
        entries = {info.filename: info for info in archive.infolist()}
        def read(name):
            info = entries[name]
            world.budget.expand(info.file_size)
            if info.file_size > MAX_EXPANSION: raise AnalysisLimit("zip member exceeds size limit")
            return archive.read(info)
        if "config.json" in entries and ("metadata.json" in entries or "model.weights.h5" in entries):
            return keras_scan.scan_config(read("config.json"), location + "!config.json")
        torch_members = [name for name in entries if name == "data.pkl" or name.endswith("/data.pkl")]
        torch_zip = local_header and any(
            (name[:-8] + "version") in entries for name in torch_members)
        if mode == "torch" and not local_header:
            return []  # torch's loader attempts legacy pickle and stops here
        if torch_zip or mode == "torch":
            findings = []
            for name in torch_members:
                findings.extend(pickle_scan.scan_pickle(read(name), f"{location}!{name}", world=world, one=True, persistent=True))
            return findings
        return _members({name: (lambda name=name: read(name)) for name in entries}, location, world, depth)


def scan_bytes(data: bytes, location: str, *, world=None, depth=0, mode=None, member=False):
    world = world or World()
    world.budget.tick()
    if depth > 12: raise AnalysisLimit("container nesting limit exceeded")
    fmt = compression(data)
    if fmt:
        return scan_bytes(decompress_one(data, fmt, world.budget), location + "!" + fmt,
                          world=world, depth=depth+1, mode=mode, member=member)
    # ZipFile follows the central directory, supports prefixed archives and the
    # zip64 EOCD written by torch.save. Never search local headers by hand.
    if (not _is_tar(data) and not data.startswith(b"\x93NUMPY") and not _is_h5(data)
            and zipfile.is_zipfile(io.BytesIO(data))):
        return scan_zip(io.BytesIO(data), location, world, depth, mode, data.startswith(b"PK\x03\x04"))
    if _is_tar(data):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            entries = {info.name: info for info in archive.getmembers() if not info.isdir()}
            files = {}
            for name, info in entries.items():
                if not info.isfile():
                    # Do not follow links or create any filesystem object.
                    raise ValueError("archive has non-regular member")
                def read(info=info):
                    world.budget.expand(info.size)
                    if info.size > MAX_EXPANSION: raise AnalysisLimit("tar member exceeds size limit")
                    return archive.extractfile(info).read()
                files[name] = read
            return _members(files, location, world, depth)
    if data.startswith(b"\x93NUMPY"):
        return scan_npy(data, location, world)
    if _is_h5(data):
        return keras_scan.scan_h5(io.BytesIO(data), location)
    base = location.rsplit("!", 1)[-1].rsplit("/", 1)[-1]
    if base in ("saved_model.pb", "saved_model.pbtxt"):
        return savedmodel_scan.scan_saved_model(data, location, text=base.endswith("pbtxt"))
    if mode == "torch":
        return pickle_scan.scan_pickle(data, location, world=world, torch=True)
    if mode == "numpy":
        return pickle_scan.scan_pickle(data, location, world=world, one=True)
    # Binary and protocol-0 pickle prefixes. The parser, not a string search,
    # decides whether these are executable opcodes.
    pickle_start = b"\x80(}])cIJKLMNFGSVXBCTU\x8a\x8b\x8c\x8d\x8e\x95\x96\x88\x89\x8f\x91\x82\x83\x84P"
    if data and data[0] in pickle_start:
        findings = pickle_scan.scan_pickle(data, location, world=world)
        if (member and not data.startswith(b"\x80") and findings
                and findings[0].detail.startswith("cannot parse pickle:")
                and os.path.splitext(base)[1].lower() not in (".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".bin")):
            return []
        return findings
    if member:
        # Archives also contain README files, JSON metadata and weight buffers.
        # Unknown model-shaped members still fail closed.
        suffix = os.path.splitext(base)[1].lower()
        if suffix not in (".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".bin", ".npy", ".npz", ".keras", ".h5", ".hdf5"):
            return []
    return [Finding("unsupported", location, "unrecognised artifact format")]


def scan_path(path: str) -> Result:
    try:
        if os.path.isdir(path):
            findings = savedmodel_scan.scan_saved_model_dir(path)
        else:
            world = World()
            location = os.path.basename(path.rstrip("/"))
            with open(path, "rb") as fh:
                prefix = fh.read(4096)
                if _is_h5(prefix):
                    return Result(path, keras_scan.scan_h5(path, location))
                if prefix.startswith(b"\x93NUMPY"):
                    if prefix[6:8] == b"\x01\x00":
                        size, start = struct.unpack_from("<H", prefix, 8)[0], 10
                    else:
                        size, start = struct.unpack_from("<I", prefix, 8)[0], 12
                    if size > 4*1024*1024: raise AnalysisLimit("NumPy header size limit")
                    fh.seek(0)
                    header, _ = read_npy_header(fh.read(start+size))
                    if not _dtype_object(header["descr"]): return Result(path, [])
                elif not _is_tar(prefix) and not compression(prefix) and zipfile.is_zipfile(fh):
                    return Result(path, scan_zip(fh, location, world, local_header=prefix.startswith(b"PK\x03\x04")))
                fh.seek(0)
                data = fh.read(512*1024*1024 + 1)
            world.budget.expand(len(data))
            findings = scan_bytes(data, location, world=world)
    except Detected as exc:
        findings = [exc.finding]
    except Exception as exc:
        findings = [Finding("unsupported", path, f"cannot analyse artifact: {type(exc).__name__}: {exc}")]
    return Result(path, findings)
