"""Content-based format detection, archive unpacking and numpy handling.

The threat model says content, not extension, decides the format at every
level, and that compressed/archived artefacts are unpacked and each member
loaded by its own format, nested up to three levels.  ``scan_blob`` detects a
blob's format from its magic bytes and dispatches; archive members and nested
loader payloads are scanned the same way.
"""
from __future__ import annotations

import ast
import bz2
import gzip
import io
import lzma
import tarfile
import zipfile
import zlib

from modelgate import pickle_scan
from modelgate.report import Finding

_MAX_DEPTH = 6                       # containers nest up to 3; a nested loader inside adds a level
_MAX_DECOMP = 400 * 1024 * 1024     # decompression bomb guard
_MAX_MEMBERS = 2048
_MEMBER_SCAN_CAP = 150 * 1024 * 1024  # read members up to this fully; larger ones
#   are scanned only if their header is a nested container (a big member with no
#   container/model magic is bulk data -- e.g. a torch tensor storage -- and is
#   skipped rather than read into memory or fail-closed).


def _is_container_header(head: bytes) -> bool:
    return (head[:2] == b"\x1f\x8b" or head[:3] == b"BZh"
            or head[:6] == b"\xfd7zXZ\x00" or head[:6] == b"\x93NUMPY"
            or head[:8] == b"\x89HDF\r\n\x1a\n"
            or head[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
            or (len(head) >= 262 and head[257:262] == b"ustar"))


def _decompress_stream(decomp, data: bytes):
    out = bytearray()
    for chunk_start in range(0, len(data), 1 << 20):
        try:
            out += decomp.decompress(data[chunk_start:chunk_start + (1 << 20)])
        except Exception:
            break
        if len(out) > _MAX_DECOMP:
            raise ValueError("decompressed size exceeds limit")
    try:
        out += decomp.flush()
    except Exception:
        pass
    return bytes(out)


def _gunzip_all(data: bytes) -> bytes:
    """Decompress every concatenated gzip member (threat model: a .gz may hold
    several members)."""
    out = bytearray()
    off = 0
    n = len(data)
    while off < n and data[off:off + 2] == b"\x1f\x8b":
        d = zlib.decompressobj(31)
        try:
            out += d.decompress(data[off:])
            out += d.flush()
        except Exception:
            if not out:
                raise
            break
        if len(out) > _MAX_DECOMP:
            raise ValueError("decompressed size exceeds limit")
        unused = d.unused_data
        if not unused or len(unused) >= (n - off):
            break
        off = n - len(unused)
    return bytes(out)


def _is_tar(data: bytes) -> bool:
    if len(data) < 265:
        return False
    return data[257:262] == b"ustar"


def scan_blob(data: bytes, location: str, depth: int = 0, top: bool = False) -> list:
    """Scan a blob whose format is decided by content.

    ``top`` marks a top-level artefact: an unrecognised/unparsable blob then
    fails closed (reported unsupported).  Inside a container it is treated as a
    benign non-model member and skipped.
    """
    if depth > _MAX_DEPTH:
        return [Finding("unsupported", location, "container nested too deep")]
    if len(data) < 2:
        return _unknown(location, top, "empty or tiny blob")

    sig = data[:8]
    try:
        # Prefix-magic formats first (unambiguous leading bytes).
        if sig[:2] == b"\x1f\x8b":
            return scan_blob(_gunzip_all(data), location + "|gunzip", depth + 1)
        if sig[:3] == b"BZh":
            return scan_blob(_decompress_stream(bz2.BZ2Decompressor(), data),
                             location + "|bunzip2", depth + 1)
        if sig[:6] == b"\xfd7zXZ\x00":
            return scan_blob(_decompress_stream(lzma.LZMADecompressor(), data),
                             location + "|unxz", depth + 1)
        if sig[:6] == b"\x93NUMPY":
            return _scan_npy(data, location)
        if sig[:8] == b"\x89HDF\r\n\x1a\n":
            from modelgate import keras_scan
            return keras_scan.scan_h5_bytes(data, location)
        # zip: detected via the central directory, which zipfile locates even
        # when bytes precede the first local header (threat model section 2).
        if _looks_zip(data):
            zf = _open_zip(data)
            if zf is not None:
                with zf:
                    return _scan_zip(zf, location, depth)
        if _is_tar(data):
            return _scan_tar(data, location, depth)
    except Exception as exc:
        return _unknown(location, top, f"{type(exc).__name__}: {exc}")

    # fall back to pickle: a pickle stream has no fixed magic but genops will
    # reject non-pickle data quickly.
    pf = pickle_scan.scan_pickle(data, location, depth)
    if pf and all(f.kind == "unsupported" for f in pf):
        return _unknown(location, top, pf[0].detail)
    return pf


def _unknown(location, top, why):
    if top:
        return [Finding("unsupported", location, f"unrecognised artefact: {why}")]
    return []


def _looks_zip(data: bytes) -> bool:
    # A zip (even one with bytes before the first local header, or a torch/keras
    # archive) carries the end-of-central-directory signature near the end.
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return True
    return b"PK\x05\x06" in data[-(1 << 16):]


def _open_zip(data: bytes):
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception:
        return None
    if not zf.infolist():
        zf.close()
        return None
    return zf


# --------------------------------------------------------------------------
# zip
# --------------------------------------------------------------------------
def _scan_zip(zf, location: str, depth: int) -> list:
    findings = []
    names = []
    seen = set()
    for info in zf.infolist():          # last occurrence wins via open(name)
        if info.filename in seen:
            continue
        seen.add(info.filename)
        names.append(info.filename)
    base = {n.rsplit("/", 1)[-1] for n in names}

    # A .keras v3 archive carries the model JSON in config.json; scan it as a
    # Keras config.  Other members are still scanned by content below, so a
    # pickle smuggled into the archive is not missed.
    is_keras_v3 = "config.json" in base and (
        "metadata.json" in base or "model.weights.h5" in base)
    if is_keras_v3:
        from modelgate import keras_scan
        for n in names:
            if n.rsplit("/", 1)[-1] == "config.json":
                findings.extend(keras_scan.scan_config_json(
                    _read_member(zf, n), f"{location}!{n}"))

    count = 0
    for n in names:
        if n.endswith("/"):
            continue
        leaf = n.rsplit("/", 1)[-1]
        if is_keras_v3 and leaf in ("config.json", "metadata.json"):
            continue
        count += 1
        if count > _MAX_MEMBERS:
            break
        try:
            size = zf.getinfo(n).file_size
        except Exception:
            size = 0
        if size > _MEMBER_SCAN_CAP:
            with zf.open(n) as fh:
                head = fh.read(262)
            if not _is_container_header(head):
                continue                      # bulk data (e.g. tensor storage)
        try:
            member = _read_member(zf, n)
        except Exception as exc:
            findings.append(Finding("unsupported", f"{location}!{n}",
                                    f"unreadable zip member: {exc}"))
            continue
        findings.extend(scan_blob(member, f"{location}!{n}", depth + 1))
    return _dedup(findings)


def _read_member(zf, name):
    with zf.open(name) as fh:
        out = bytearray()
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            out += chunk
            if len(out) > _MAX_DECOMP:
                break                         # scan the capped prefix; never fail-closed here
        return bytes(out)


# --------------------------------------------------------------------------
# tar
# --------------------------------------------------------------------------
def _scan_tar(data: bytes, location: str, depth: int) -> list:
    findings = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        count = 0
        for member in tf.getmembers():
            if not member.isfile():
                continue
            count += 1
            if count > _MAX_MEMBERS:
                break
            fh = tf.extractfile(member)
            if fh is None:
                continue
            if member.size > _MEMBER_SCAN_CAP:
                head = fh.read(262)
                if not _is_container_header(head):
                    continue                  # bulk data; skip
                data = head + fh.read(_MAX_DECOMP)
            else:
                data = fh.read(_MAX_DECOMP + 1)[:_MAX_DECOMP]
            findings.extend(scan_blob(data, f"{location}!{member.name}", depth + 1))
    return _dedup(findings)


# --------------------------------------------------------------------------
# numpy .npy
# --------------------------------------------------------------------------
def _scan_npy(data: bytes, location: str) -> list:
    try:
        major = data[6]
        if major == 1:
            hlen = int.from_bytes(data[8:10], "little")
            start = 10 + hlen
        else:
            hlen = int.from_bytes(data[8:12], "little")
            start = 12 + hlen
        header = data[(10 if major == 1 else 12):start].decode("latin1")
        meta = ast.literal_eval(header)
        descr = meta.get("descr")
    except Exception as exc:
        return [Finding("unsupported", location, f"bad .npy header: {exc}")]
    if _is_object_descr(descr):
        return pickle_scan.scan_pickle(data[start:], location + "|npy-pickle")
    return []


def _is_object_descr(descr) -> bool:
    if isinstance(descr, str):
        return "O" in descr
    if isinstance(descr, list):
        return any(_is_object_descr(f[1]) for f in descr if len(f) >= 2)
    return False


def _dedup(findings):
    out = []
    seen = set()
    for f in findings:
        key = (f.kind, f.detail)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out
