"""Content-based, in-memory artifact dispatch. Archives are never extracted."""
from __future__ import annotations

import ast
import bz2
import gzip
import io
import lzma
import os
import pickletools
import struct
import tarfile
import zipfile
import zlib

from modelgate import keras_scan, pickle_scan, savedmodel_scan
from modelgate.report import Finding, Result

MAX_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 10000


def _bounded(data):
    if len(data) > MAX_BYTES:
        raise ValueError("expanded artifact exceeds analysis size limit")
    return data


def decompress(data):
    """Return decompressed bytes, or None if this is not a compressed stream."""
    if data.startswith(b"\x1f\x8b"):
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            return _bounded(stream.read(MAX_BYTES + 1))
    if data.startswith(b"BZh"):
        cls = bz2.BZ2Decompressor
    elif data.startswith(b"\xfd7zXZ\x00"):
        cls = lzma.LZMADecompressor
    elif len(data) > 1 and data[0] == 0x78 and int.from_bytes(data[:2], "big") % 31 == 0:
        obj = zlib.decompressobj()
        return _bounded(obj.decompress(data, MAX_BYTES + 1))
    elif data.startswith(b"]\x00"):
        cls = lzma.LZMADecompressor  # joblib's legacy FORMAT_ALONE
    else:
        return None
    # Bzip2 and xz, like gzip, allow concatenated members.
    result = bytearray()
    rest = data
    while rest:
        obj = cls()
        result.extend(obj.decompress(rest, max_length=MAX_BYTES + 1 - len(result)))
        _bounded(result)
        if not obj.eof:
            raise ValueError("truncated or oversized compressed stream")
        rest = obj.unused_data
        if not rest or not rest.strip(b"\0"):
            break
    return bytes(result)


def npy_payload(data):
    """Return (object dtype, data offset), without loading NumPy objects."""
    if data[:6] != b"\x93NUMPY":
        raise ValueError("not a NumPy array")
    version = tuple(data[6:8])
    size = 2 if version == (1, 0) else 4
    if version not in ((1, 0), (2, 0), (3, 0)):
        raise ValueError("unsupported NumPy header version")
    length = int.from_bytes(data[8:8 + size], "little")
    offset = 8 + size + length
    header = ast.literal_eval(data[8 + size:offset].decode("utf-8" if version == (3, 0) else "latin1").strip())
    def objects(dtype):
        if isinstance(dtype, str):
            return dtype.lstrip("<>=|").startswith("O")
        if isinstance(dtype, list):
            return any(objects(field[1]) for field in dtype)
        if isinstance(dtype, tuple):
            return objects(dtype[0])
        return False
    return objects(header["descr"]), offset


def zip_members(data):
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        if len(zf.infolist()) > MAX_MEMBERS:
            raise ValueError("too many archive members")
        # A zip's central directory is authoritative; later duplicate names win.
        entries = {info.filename: info for info in zf.infolist() if not info.is_dir()}
        total = sum(i.file_size for i in entries.values())
        if total > MAX_BYTES:
            raise ValueError("expanded archive exceeds analysis size limit")
        return {name: zf.read(info) for name, info in entries.items()}


def scan_path(path: str) -> Result:
    if os.path.isdir(path):
        return Result(path, savedmodel_scan.scan_saved_model_dir(path))
    with open(path, "rb") as fh:
        data = fh.read(MAX_BYTES + 1)
    _bounded(data)
    vm = pickle_scan.Interpreter(artifact_path=path)
    scan_bytes(data, os.path.basename(path), vm=vm)
    return Result(path, vm.findings)


def scan_bytes(data: bytes, location: str, *, depth=0, vm=None, member=False):
    if vm is None:
        vm = pickle_scan.Interpreter()
    vm.check_budget()
    _bounded(data)
    expanded = decompress(data)
    if expanded is not None:
        if depth > 6:
            raise ValueError("container nesting limit")
        return scan_bytes(expanded, location + "!decompressed", depth=depth + 1, vm=vm, member=member)
    if zipfile.is_zipfile(io.BytesIO(data)):
        if depth > 6:
            raise ValueError("container nesting limit")
        members = zip_members(data)
        if "config.json" in members and ("metadata.json" in members or "model.weights.h5" in members or location.lower().endswith(".keras")):
            vm.findings.extend(keras_scan.scan_config(members["config.json"], location + "!config.json"))
            return None
        pickles = [name for name in members if name == "data.pkl" or name.endswith("/data.pkl")]
        # Torch's zip loader only accepts local-header magic at byte zero.
        if data.startswith(b"PK\x03\x04") and pickles and any(n == "version" or n.endswith("/version") for n in members):
            result = None
            for name in pickles:
                result = vm.load(members[name], f"{location}!{name}", all_streams=False, persistent=True)
            return result
        return scan_members(members, location, depth, vm)
    if data.startswith(b"\x93NUMPY"):
        objects, offset = npy_payload(data)
        if objects:
            return vm.load(data[offset:], location + "!array", all_streams=False)
        return None
    # HDF5 permits a user block before its signature at power-of-two offsets.
    if data.startswith(b"\x89HDF\r\n\x1a\n") or any(data[1 << i:(1 << i) + 8] == b"\x89HDF\r\n\x1a\n" for i in range(9, len(data).bit_length())):
        vm.findings.extend(keras_scan.scan_h5(data, location))
        return None
    try:
        archive = tarfile.open(fileobj=io.BytesIO(data), mode="r:")
    except tarfile.ReadError:
        archive = None
    if archive is not None:
        if depth > 6:
            raise ValueError("container nesting limit")
        members = {}
        with archive:
            total = 0
            for info in archive:
                if len(members) > MAX_MEMBERS:
                    raise ValueError("too many archive members")
                if info.isfile():
                    total += info.size
                    if total > MAX_BYTES:
                        raise ValueError("expanded archive exceeds analysis size limit")
                    members[info.name] = archive.extractfile(info).read()
        return scan_members(members, location, depth, vm)
    base = location.rsplit("!", 1)[-1].replace("\\", "/").rsplit("/", 1)[-1]
    if base in ("saved_model.pb", "saved_model.pbtxt"):
        vm.findings.extend(savedmodel_scan.scan_saved_model(data, location, text=base.endswith("pbtxt")))
        return None
    if data and data[:1] in pickle_scan.START_BYTES:
        if member:
            try:
                # Storage blobs and text metadata may start with an opcode
                # byte by coincidence. Require a syntactically complete
                # pickle before treating an otherwise untyped member as one.
                for op, _, _ in pickletools.genops(data):
                    if op.name == "STOP":
                        break
            except Exception:
                if not base.lower().endswith((".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".bin")):
                    return None
        return vm.load(data, location, all_streams=True)
    if member:
        # Archive metadata, weights, storage data, README files, etc. are inert.
        return None
    vm.findings.append(Finding("unsupported", location, "unrecognized artifact format"))
    return None


def scan_members(members, location, depth, vm):
    for name, data in members.items():
        if name.replace("\\", "/").rsplit("/", 1)[-1] in ("saved_model.pb", "saved_model.pbtxt"):
            # Only one protobuf representation is used by the standard loader.
            if name.endswith("pbtxt") and name[:-3] in members:
                continue
            vm.findings.extend(savedmodel_scan.scan_saved_model(data, f"{location}!{name}", text=name.endswith("pbtxt")))
        elif "/variables/" not in "/" + name:
            scan_bytes(data, f"{location}!{name}", depth=depth + 1, vm=vm, member=True)
