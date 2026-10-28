"""Content-based dispatch.

The format of every artifact and of every container member is decided by its
content, not its extension (threat-model section 2).  Containers (gzip, bzip2,
xz, zip, tar) are unpacked and each member is loaded by its own format, nested
up to a few levels.  The registry policy is fail-closed: a top-level artifact
the scanner cannot analyse is reported ``unsupported`` (malicious); inert,
non-model members inside a container are simply clean.
"""
from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os
import tarfile
import zipfile

from modelgate import keras_scan, numpy_scan, pickle_scan, savedmodel_scan
from modelgate.report import Finding, Result

MAX_DEPTH = 6
MAX_BYTES = 1 << 30          # 1 GiB decompression cap per member
MAX_MEMBERS = 20000

_HDF5 = b"\x89HDF\r\n\x1a\n"
_GZIP = b"\x1f\x8b"
_BZIP2 = b"BZh"
_XZ = b"\xfd7zXZ\x00"


def scan_path(path: str) -> Result:
    if os.path.isdir(path):
        return Result(path, _guard(savedmodel_scan.scan_saved_model_dir, path))
    if not os.path.isfile(path):
        return Result(path, [Finding("unsupported", path, "path is neither a file nor a directory")])
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        return Result(path, [Finding("unsupported", path, f"cannot read artifact: {exc}")])
    base = os.path.basename(path.rstrip("/")) or "artifact"
    findings = _detect_and_scan(data, base, 0, top_level=True, name_hint=base)
    return Result(path, findings)


def scan_embedded(data: bytes, location: str, depth: int, name_hint: str = "") -> list[Finding]:
    return _detect_and_scan(data, location, depth, top_level=False, name_hint=name_hint)


def _guard(fn, *a):
    try:
        return fn(*a)
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", a[-1] if a else "?", f"scanner error: {type(exc).__name__}: {exc}")]


def _unknown(location, top_level, why):
    if top_level:
        return [Finding("unsupported", location, why)]
    return []  # inert member inside a container


def _detect_and_scan(data, location, depth, top_level, name_hint=""):
    if depth > MAX_DEPTH:
        return _unknown(location, top_level, "nesting too deep")
    hint = (name_hint or "").lower()

    # name-hinted TensorFlow SavedModel protobuf (e.g. inside an archive)
    if hint.endswith("saved_model.pb"):
        return _guard(savedmodel_scan.scan_saved_model_bytes, data, location)
    if hint.endswith("saved_model.pbtxt"):
        return savedmodel_scan.scan_saved_model_bytes(data, location, True)

    # reliable, distinct magics commit immediately
    if data[:8] == _HDF5:
        return _guard(_scan_h5_bytes, data, location)
    if data[:2] == _GZIP:
        return _scan_compressed(data, location, depth, top_level, "gzip")
    if data[:3] == _BZIP2:
        return _scan_compressed(data, location, depth, top_level, "bz2")
    if data[:6] == _XZ:
        return _scan_compressed(data, location, depth, top_level, "xz")
    if numpy_scan.is_npy(data):
        return numpy_scan.scan_npy(data, location, depth)

    # zip and tar are recognised structurally; if the structure does not in
    # fact parse (a stray end-of-central-directory signature inside a pickle,
    # say), fall through rather than committing to the format.
    if _is_zip(data):
        res = _scan_zip(data, location, depth)
        if res is not None:
            return res
    if _is_tar(data):
        res = _scan_tar(data, location, depth)
        if res is not None:
            return res

    # otherwise: a pickle (one or more streams), else unknown
    try:
        return pickle_scan.scan_pickle(data, location, depth)
    except Exception:
        return _unknown(location, top_level, "unrecognised format")


# ----- HDF5 -----------------------------------------------------------------

def _scan_h5_bytes(data, location):
    import h5py
    with h5py.File(io.BytesIO(data), "r") as fh:
        raw = fh.attrs.get("model_config")
    if raw is None:
        return []
    return keras_scan._scan_config(raw, location)


# ----- zip ------------------------------------------------------------------

def _is_zip(data):
    try:
        return zipfile.is_zipfile(io.BytesIO(data))
    except Exception:
        return False


def _last_wins(zf):
    """Unique members, later entry with the same name wins (as extraction and
    the central directory resolve duplicates)."""
    out = {}
    for zi in zf.infolist():
        if zi.is_dir():
            continue
        out[zi.filename] = zi
    return list(out.values())


def _scan_zip(data, location, depth):
    """Scan a zip, or return None if it does not actually parse as one."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        members = _last_wins(zf)
        names = {m.filename for m in members}
    except Exception:
        return None
    if not members:
        return None

    base_names = {n.rsplit("/", 1)[-1] for n in names}
    # Keras v3 archive
    if "config.json" in base_names and ("model.weights.h5" in base_names or "metadata.json" in base_names):
        for m in members:
            if m.filename.rsplit("/", 1)[-1] == "config.json":
                try:
                    cfg = _read_zip_member(zf, m)
                except Exception as exc:
                    return [Finding("unsupported", location, f"unreadable keras config: {exc}")]
                return keras_scan.scan_keras_v3(cfg, f"{location}!config.json")
        return []

    # generic archive (covers torch zip checkpoints and .npz)
    findings = []
    count = 0
    for m in members:
        count += 1
        if count > MAX_MEMBERS:
            break
        if m.file_size > MAX_BYTES:
            continue  # oversized blob: weights, not an attack surface
        try:
            member = _read_zip_member(zf, m)
        except Exception:
            continue
        findings.extend(scan_embedded(member, f"{location}!{m.filename}", depth + 1, m.filename))
    return findings


def _read_zip_member(zf, m):
    with zf.open(m) as fh:
        return fh.read(MAX_BYTES + 1)


# ----- tar ------------------------------------------------------------------

def _is_tar(data):
    if len(data) < 512:
        return False
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:  # noqa: SIM117
            return True
    except Exception:
        return False


def _scan_tar(data, location, depth):
    """Scan a tar, or return None if it does not actually parse as one."""
    findings = []
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data))
    except Exception:
        return None
    count = 0
    for m in tf.getmembers():
        if not m.isfile():
            continue
        count += 1
        if count > MAX_MEMBERS:
            break
        if m.size > MAX_BYTES:
            continue  # oversized blob: weights, not an attack surface
        try:
            ex = tf.extractfile(m)
            member = ex.read(MAX_BYTES + 1) if ex else b""
        except Exception:
            continue
        findings.extend(scan_embedded(member, f"{location}!{m.name}", depth + 1, m.name))
    return findings


# ----- gzip / bz2 / xz ------------------------------------------------------

def _scan_compressed(data, location, depth, top_level, kind):
    try:
        raw = _decompress(data, kind)
    except Exception as exc:
        return _unknown(location, top_level, f"cannot decompress {kind}: {exc}")
    if raw is None:
        return _unknown(location, top_level, f"{kind} stream too large")
    return scan_embedded(raw, f"{location}!{kind}", depth + 1)


def _decompress(data, kind):
    if kind == "gzip":
        fh = gzip.GzipFile(fileobj=io.BytesIO(data))
    elif kind == "bz2":
        fh = bz2.BZ2File(io.BytesIO(data))
    else:
        fh = lzma.LZMAFile(io.BytesIO(data))
    with fh:
        out = fh.read(MAX_BYTES + 1)
    if len(out) > MAX_BYTES:
        return None
    return out
