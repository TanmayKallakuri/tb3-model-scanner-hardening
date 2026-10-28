"""Format detection and container recursion.

Content, not extension, decides the format at every level (threat model
section 2).  ``scan_bytes`` sniffs a magic, routes to the right analyser, and
for compressed or archived artifacts unpacks in memory -- never to disk -- and
recurses on each member by that member's own format, up to the nesting the
policy allows.  Nothing here loads or runs an artifact; archives are read with
``zipfile``/``tarfile`` and decompressors with a hard output cap so a bomb
cannot exhaust memory or time.
"""
from __future__ import annotations

import ast
import io
import struct
import tarfile
import zipfile

import posixpath

from modelgate import keras_scan, pickle_scan, savedmodel_scan
from modelgate.report import Finding

_MAX_BYTES = 512 * 1024 * 1024      # cap on any single decompressed/read blob
_MAX_DEPTH = 6                      # policy nests to 3; a margin, then stop
_MAX_MEMBERS = 4096
_TOTAL_BYTES = 2 * 1024 * 1024 * 1024   # cumulative read cap per artifact


def _over_budget(budget):
    return budget.get("bytes", 0) > _TOTAL_BYTES


# --------------------------------------------------------------------------
def detect(data):
    if data[:2] == b"\x1f\x8b":
        return "gzip"
    if data[:3] == b"BZh":
        return "bz2"
    if data[:6] == b"\xfd7zXZ\x00":
        return "xz"
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08") or _is_zip(data):
        return "zip"
    if data[:8] == b"\x89HDF\r\n\x1a\n":
        return "hdf5"
    if data[:6] == b"\x93NUMPY":
        return "npy"
    if len(data) >= 262 and data[257:262] == b"ustar":
        return "tar"
    if pickle_scan.looks_like_pickle(data):
        return "pickle"
    if _is_tar(data):
        return "tar"
    return "unknown"


def _is_zip(data):
    try:
        return zipfile.is_zipfile(io.BytesIO(data))
    except Exception:
        return False


def _is_tar(data):
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            return tf.next() is not None
    except Exception:
        return False


# --------------------------------------------------------------------------
def scan_bytes(data, location, depth=0, budget=None):
    if budget is None:
        budget = {"ops": 0}
    if depth > _MAX_DEPTH:
        return []
    fmt = detect(data)
    try:
        if fmt in ("gzip", "bz2", "xz"):
            out = _decompress(data, fmt)
            budget["bytes"] = budget.get("bytes", 0) + len(out)
            return scan_bytes(out, f"{location}!{fmt}", depth + 1, budget)
        if fmt == "zip":
            return _scan_zip(data, location, depth, budget)
        if fmt == "tar":
            return _scan_tar(data, location, depth, budget)
        if fmt == "hdf5":
            return keras_scan.scan_h5_bytes(data, location)
        if fmt == "npy":
            return scan_npy_bytes(data, location, depth, budget)
        if fmt == "pickle":
            return pickle_scan.scan_pickle(data, location, depth, budget)
    except Exception as exc:
        # A member that declares a real format but will not parse is treated
        # fail-closed; an unrecognised blob at the top level likewise.
        return [Finding("unsupported", location, f"could not analyse {fmt}: {type(exc).__name__}: {exc}")]
    # Unknown content: at the top level the policy is fail-closed; inside an
    # archive it is just a non-model file (a README, a text card) -> clean.
    if depth == 0:
        return [Finding("unsupported", location, "unrecognised artifact format")]
    return []


# --------------------------------------------------------------------------
def _decompress(data, kind):
    if kind == "gzip":
        import gzip
        reader = gzip.GzipFile(fileobj=io.BytesIO(data))
    elif kind == "bz2":
        import bz2
        reader = bz2.BZ2File(io.BytesIO(data))
    else:
        import lzma
        reader = lzma.LZMAFile(io.BytesIO(data))
    with reader:
        out = reader.read(_MAX_BYTES + 1)
    if len(out) > _MAX_BYTES:
        raise ValueError("decompressed output exceeds cap")
    return out


# --------------------------------------------------------------------------
def _effective_members(zf):
    """Last entry with a given name wins, as zipfile's own lookup does."""
    effective = {}
    for info in zf.infolist():
        effective[info.filename] = info
    return effective


def _scan_zip(data, location, depth, budget):
    findings = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = _effective_members(zf)
        names = list(members)
        is_torch = any(n == "data.pkl" or n.endswith("/data.pkl") for n in names)
        is_keras = "config.json" in names and "metadata.json" in names
        npy_names = [n for n in names if n.endswith(".npy")]
        is_npz = bool(npy_names) and all(
            n.endswith(".npy") or n.endswith("/") for n in names
        )
        if is_torch:
            for n in names:
                if n == "data.pkl" or n.endswith("/data.pkl"):
                    findings.extend(pickle_scan.scan_pickle(
                        _read_member(zf, members[n], budget), f"{location}!{n}", depth + 1, budget))
            return findings
        if is_keras:
            raw = _read_member(zf, members["config.json"], budget)
            return keras_scan.scan_model_json(raw, f"{location}!config.json")
        if is_npz:
            for n in npy_names:
                if _over_budget(budget):
                    break
                findings.extend(scan_npy_bytes(
                    _read_member(zf, members[n], budget), f"{location}!{n}", depth + 1, budget))
            return findings
        # Generic archive: load each member by its own format.
        count = 0
        for n, info in members.items():
            if n.endswith("/") or getattr(info, "is_dir", lambda: False)():
                continue
            count += 1
            if count > _MAX_MEMBERS or _over_budget(budget):
                break
            findings.extend(_scan_member(
                _read_member(zf, info, budget), n, f"{location}!{n}", depth + 1, budget))
    return findings


def _scan_member(data, name, location, depth, budget):
    """Scan one archive member, routing by name where the member is part of a
    multi-file format (a SavedModel's protobuf, a .keras config) and otherwise
    by content."""
    base = posixpath.basename(name)
    try:
        if base == "saved_model.pb":
            return savedmodel_scan.scan_saved_model_bytes(data, location, text=False)
        if base == "saved_model.pbtxt":
            return savedmodel_scan.scan_saved_model_bytes(data, location, text=True)
        if base == "config.json":
            return keras_scan.scan_model_json(data, location)
    except Exception as exc:
        return [Finding("unsupported", location, f"could not analyse member: {type(exc).__name__}: {exc}")]
    return scan_bytes(data, location, depth, budget)


def _read_member(zf, info, budget):
    if info.file_size > _MAX_BYTES:
        raise ValueError("zip member exceeds cap")
    with zf.open(info) as fh:
        blob = fh.read(_MAX_BYTES + 1)
    budget["bytes"] = budget.get("bytes", 0) + len(blob)
    return blob


def _scan_tar(data, location, depth, budget):
    findings = []
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        count = 0
        for member in tf:
            if not member.isfile():
                continue
            count += 1
            if count > _MAX_MEMBERS or _over_budget(budget):
                break
            if member.size > _MAX_BYTES:
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            blob = extracted.read(_MAX_BYTES + 1)
            budget["bytes"] = budget.get("bytes", 0) + len(blob)
            findings.extend(_scan_member(
                blob, member.name, f"{location}!{member.name}", depth + 1, budget))
    return findings


# --------------------------------------------------------------------------
def scan_npy_bytes(data, location, depth=0, budget=None):
    """A .npy with an object dtype carries a pickle after its header; any other
    dtype is raw numbers and loads nothing."""
    if budget is None:
        budget = {"ops": 0}
    if data[:6] != b"\x93NUMPY":
        return [Finding("unsupported", location, "not a .npy file")]
    major = data[6]
    if major == 1:
        hlen = struct.unpack("<H", data[8:10])[0]
        start = 10 + hlen
    else:
        hlen = struct.unpack("<I", data[8:12])[0]
        start = 12 + hlen
    header = data[(10 if major == 1 else 12):start].decode("latin1")
    if not _is_object_dtype(header):
        return []
    return pickle_scan.scan_pickle(data[start:], f"{location}:npy-data", depth, budget)


def _is_object_dtype(header):
    try:
        meta = ast.literal_eval(header.strip())
        descr = meta.get("descr")
    except Exception:
        return "'|O'" in header or '"|O"' in header or "O8" in header
    return _descr_has_object(descr)


def _descr_has_object(descr):
    if isinstance(descr, str):
        return "O" in descr
    if isinstance(descr, (list, tuple)):
        return any(_descr_has_object(item) for item in descr)
    return False
