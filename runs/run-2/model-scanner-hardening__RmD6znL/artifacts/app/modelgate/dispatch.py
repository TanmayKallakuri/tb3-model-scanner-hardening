"""Route an artifact to the right scanner by *content*, and unpack containers.

Extension is only a hint; the magic bytes decide the format at every level,
and compressed / archived artifacts are unpacked and each member scanned by
its own format, nested up to the policy's three levels (with margin).
Scanning never runs the artifact: containers are read into memory, never
extracted to disk, and no loader is invoked.
"""
from __future__ import annotations

import io
import os
import tarfile
import zipfile

from modelgate import keras_scan, pickle_scan, savedmodel_scan
from modelgate.report import Finding, Result

_MAX_DEPTH = 5
_MAX_BYTES = 256 * 1024 * 1024  # decompression / member-size guard


def scan_path(path: str) -> Result:
    if os.path.isdir(path):
        return Result(path, savedmodel_scan.scan_saved_model_dir(path))
    try:
        with open(path, "rb") as fh:
            data = fh.read(_MAX_BYTES + 1)
    except OSError as exc:
        return Result(path, [Finding("unsupported", path, f"cannot read: {exc}")])
    if len(data) > _MAX_BYTES:
        return Result(path, [Finding("unsupported", path, "artifact too large to scan")])
    return Result(path, scan_bytes(data, os.path.basename(path.rstrip("/")), 0))


def scan_bytes(data: bytes, location: str, depth: int) -> list[Finding]:
    if depth > _MAX_DEPTH:
        return [Finding("unsupported", location, "container nesting too deep")]
    if not data:
        return []

    # --- HDF5 (Keras) ---
    if data[:8] == b"\x89HDF\r\n\x1a\n":
        return keras_scan.scan_h5_bytes(data, location)

    # --- compressed single streams ---
    if data[:2] == b"\x1f\x8b":
        return _scan_decompressed(_gunzip(data), location, depth, "gzip")
    if data[:3] == b"BZh":
        import bz2
        return _scan_decompressed(_bounded(bz2.BZ2Decompressor(), data), location, depth, "bz2")
    if data[:6] == b"\xfd7zXZ\x00":
        import lzma
        return _scan_decompressed(_bounded(lzma.LZMADecompressor(), data), location, depth, "xz")

    # --- zip family (zip / npz / keras v3 / torch) --- (also prefixed zips) ---
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08") or _is_zip(data):
        return _scan_zip(data, location, depth)

    # --- numpy .npy ---
    if data[:6] == b"\x93NUMPY":
        return _scan_npy(data, location, depth)

    # --- tar / tar.* ---
    if _is_tar(data):
        return _scan_tar(data, location, depth)

    # --- otherwise: treat as pickle stream(s) ---
    return pickle_scan.scan_pickle(data, location, depth)


# ---------------------------------------------------------------- compressed
def _safe(fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None


def _bounded(decompressor, data):
    """Decompress with a hard output cap so a bomb cannot exhaust memory."""
    try:
        out = decompressor.decompress(data, _MAX_BYTES + 1)
        if len(out) > _MAX_BYTES:
            return None
        return out
    except Exception:  # noqa: BLE001
        return None


def _gunzip(data):
    # gzip may hold several concatenated members; concatenate their output.
    try:
        import gzip
        out = bytearray()
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
            while True:
                chunk = gz.read(1024 * 1024)
                if not chunk:
                    break
                out += chunk
                if len(out) > _MAX_BYTES:
                    break
        return bytes(out)
    except Exception:  # noqa: BLE001
        return None


def _scan_decompressed(out, location, depth, label):
    if out is None:
        return [Finding("unsupported", location, f"corrupt {label} stream")]
    if len(out) > _MAX_BYTES:
        return [Finding("unsupported", location, f"{label} stream too large")]
    return scan_bytes(out, f"{location}|{label}", depth + 1)


# ---------------------------------------------------------------------- zip
def _is_zip(data):
    try:
        return zipfile.is_zipfile(io.BytesIO(data))
    except Exception:  # noqa: BLE001
        return False


def _zip_members(zf):
    """name -> ZipInfo, keeping the LAST entry for a duplicated name
    (zipfile / unzip let a later entry with the same name win)."""
    members = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        members[info.filename] = info
    return members


def _read_member(zf, info):
    try:
        with zf.open(info) as fh:
            return fh.read(_MAX_BYTES + 1)
    except Exception:  # noqa: BLE001
        return None


def _scan_zip(data, location, depth):
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"corrupt zip: {type(exc).__name__}")]
    with zf:
        members = _zip_members(zf)
        names = set(members)
        prefix0 = data[:4] == b"PK\x03\x04"  # torch needs the magic at offset 0

        pkl_names = [n for n in names if n == "data.pkl" or n.endswith("/data.pkl")]
        torchy = pkl_names and any(
            n.endswith("/version") or n == "version" or "/data/" in n or n.endswith("/byteorder")
            for n in names
        )
        if torchy and prefix0:
            findings = []
            for n in sorted(pkl_names):
                blob = _read_member(zf, members[n])
                if blob is not None:
                    findings += pickle_scan.scan_pickle(blob, f"{location}!{n}", depth + 1)
            return findings

        if "config.json" in names:
            blob = _read_member(zf, members["config.json"])
            if blob is not None:
                return keras_scan.scan_keras_config(blob, f"{location}!config.json")

        npy_names = [n for n in names if n.endswith(".npy")]
        if names and npy_names and len(npy_names) == len(names):
            findings = []
            for n in sorted(npy_names):
                blob = _read_member(zf, members[n])
                if blob is not None:
                    findings += scan_bytes(blob, f"{location}!{n}", depth + 1)
            return findings

        # generic archive: load each member by its own format
        findings = []
        for n in sorted(names):
            blob = _read_member(zf, members[n])
            if blob is None:
                continue
            if len(blob) > _MAX_BYTES:
                findings.append(Finding("unsupported", f"{location}!{n}", "member too large"))
                continue
            findings += scan_bytes(blob, f"{location}!{n}", depth + 1)
        return findings


# ---------------------------------------------------------------------- tar
def _is_tar(data):
    if len(data) >= 265 and data[257:262] == b"ustar":
        return True
    if len(data) < 512:
        return False
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            return tf.next() is not None
    except Exception:  # noqa: BLE001
        return False


def _scan_tar(data, location, depth):
    findings = []
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            for member in tf:
                if not member.isfile():
                    continue
                if member.size > _MAX_BYTES:
                    findings.append(Finding("unsupported", f"{location}!{member.name}",
                                            "member too large"))
                    continue
                try:
                    fh = tf.extractfile(member)
                    blob = fh.read() if fh is not None else None
                except Exception:  # noqa: BLE001
                    blob = None
                if blob is not None:
                    findings += scan_bytes(blob, f"{location}!{member.name}", depth + 1)
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"corrupt tar: {type(exc).__name__}")]
    return findings


# ---------------------------------------------------------------------- npy
def _scan_npy(data, location, depth):
    # Parse the header without compiling it (it is attacker-controlled): we only
    # need the descr, and whether it names the object dtype -- numpy.load pickles
    # only for object arrays.
    try:
        if data[6:8] == b"\x01\x00":
            hlen = int.from_bytes(data[8:10], "little")
            start = 10 + hlen
            header = data[10:10 + hlen]
        else:
            hlen = int.from_bytes(data[8:12], "little")
            start = 12 + hlen
            header = data[12:12 + hlen]
        text = header.decode("latin1")
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"bad npy header: {type(exc).__name__}")]
    if _header_has_object(text):
        # object arrays are stored as a pickle of the array object
        return pickle_scan.scan_pickle(data[start:], f"{location}:npy-pickle", depth)
    return []


def _header_has_object(text):
    """True if the npy descr names the object dtype (so numpy.load unpickles)."""
    i = text.find("descr")
    j = text.find("fortran_order")
    seg = text[i:j] if 0 <= i < j else text
    return any(tok in seg for tok in ("|O", "'O'", '"O"', "|O8", "'O8'", '"O8"'))
