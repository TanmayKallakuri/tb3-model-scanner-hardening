"""Content-based artifact dispatch and container unwrapping.

Format is decided by content, not extension, at every level.  Compressed and
archived artifacts are unwrapped in memory (we never extract to disk) and each
member is analysed by its own content, nested up to the policy's three levels.
"""
from __future__ import annotations

import ast
import io
import tarfile
import zipfile

from modelgate import keras_scan, pickle_scan, savedmodel_scan
from modelgate.report import Finding

MAX_DECOMPRESS = 200 * 1024 * 1024
MAX_MEMBERS = 4096
MAX_NEST = 6


def analyze_artifact(data, location, nest=0, top_level=False):
    """Return findings for an artifact given as raw bytes."""
    if nest > MAX_NEST:
        return [Finding("unsupported", location, "nesting too deep")]
    try:
        return _analyze(data, location, nest, top_level)
    except Exception as exc:  # noqa: BLE001 - any failure still yields a verdict
        if top_level:
            return [Finding("unsupported", location,
                            f"scanner error: {type(exc).__name__}: {exc}")]
        return []


def _analyze(data, location, nest, top_level):
    if len(data) >= 2 and data[:2] == b"\x1f\x8b":
        return _wrap(_gunzip(data), location, nest, top_level, "gzip")
    if len(data) >= 3 and data[:3] == b"BZh":
        return _wrap(_bunzip(data), location, nest, top_level, "bz2")
    if len(data) >= 6 and data[:6] == b"\xfd7zXZ\x00":
        return _wrap(_unxz(data), location, nest, top_level, "xz")
    # A zip whose local-file-header magic is at offset 0.
    if data[:4] == b"PK\x03\x04":
        return _analyze_zip(data, location, nest, top_level)
    if data[:6] == b"\x93NUMPY":
        return analyze_npy_bytes(data, location, nest)
    if data[:8] == b"\x89HDF\r\n\x1a\n":
        return keras_scan.scan_h5_bytes(data, location)
    # A real tar must be recognised before the loose is_zipfile fallback, since
    # a tar that embeds a zip would otherwise be read as that inner zip.
    tar_findings = _try_tar(data, location, nest, top_level)
    if tar_findings is not None:
        return tar_findings
    # A zip with bytes before the first local header (central directory still
    # found by zipfile); the registry loads it as a generic archive.
    if zipfile.is_zipfile(io.BytesIO(data)):
        return _analyze_zip(data, location, nest, top_level)
    # Try a (possibly multi-stream) pickle.  A legacy torch checkpoint is a
    # sequence of pickles whose first stream is torch's magic number and which
    # torch.load unpickles with its own persistent-id (storage) handler.
    if _looks_pickle(data):
        torch_ctx = _is_torch_legacy(data)
        return pickle_scan.scan_pickle(data, location, torch_context=torch_ctx,
                                       nest=nest)
    if top_level:
        return [Finding("unsupported", location, "unrecognised artifact format")]
    return []


_TORCH_MAGIC = 0x1950A86A20F9469CFC6C


def _is_torch_legacy(data):
    """True if ``data`` begins with torch's legacy-checkpoint magic pickle."""
    import pickletools
    try:
        value = None
        n = 0
        for opcode, arg, _pos in pickletools.genops(io.BytesIO(data)):
            name = opcode.name
            if name in ("PROTO", "FRAME"):
                continue
            if name == "STOP":
                break
            if name in ("LONG", "LONG1", "LONG4", "INT", "BININT", "BININT1",
                        "BININT2"):
                value = arg
                n += 1
            else:
                return False
            if n > 1:
                return False
        return value == _TORCH_MAGIC
    except Exception:  # noqa: BLE001
        return False


def _wrap(result, location, nest, top_level, label):
    if isinstance(result, Finding):
        return [result]
    return analyze_artifact(result, f"{location}|{label}", nest=nest + 1,
                            top_level=top_level)


def _gunzip(data):
    import gzip
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as fh:
            out = fh.read(MAX_DECOMPRESS + 1)
    except Exception as exc:  # noqa: BLE001
        return Finding("unsupported", "gzip", f"bad gzip: {exc}")
    if len(out) > MAX_DECOMPRESS:
        return Finding("unsupported", "gzip", "decompressed data too large")
    return out


def _bunzip(data):
    import bz2
    try:
        with bz2.BZ2File(io.BytesIO(data)) as fh:
            out = fh.read(MAX_DECOMPRESS + 1)
    except Exception as exc:  # noqa: BLE001
        return Finding("unsupported", "bz2", f"bad bz2: {exc}")
    if len(out) > MAX_DECOMPRESS:
        return Finding("unsupported", "bz2", "decompressed data too large")
    return out


def _unxz(data):
    import lzma
    try:
        with lzma.LZMAFile(io.BytesIO(data)) as fh:
            out = fh.read(MAX_DECOMPRESS + 1)
    except Exception as exc:  # noqa: BLE001
        return Finding("unsupported", "xz", f"bad xz: {exc}")
    if len(out) > MAX_DECOMPRESS:
        return Finding("unsupported", "xz", "decompressed data too large")
    return out


def _looks_zip(data):
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return True
    return zipfile.is_zipfile(io.BytesIO(data))


def _zip_members_lastwins(zf):
    """Distinct member infos, last entry winning for a repeated name."""
    out = {}
    for info in zf.infolist():
        out[info.filename] = info
    return out


def _zip_read(zf, info):
    with zf.open(info) as fh:
        data = fh.read(MAX_DECOMPRESS + 1)
    if len(data) > MAX_DECOMPRESS:
        raise ValueError("zip member too large")
    return data


def _analyze_zip(data, location, nest, top_level):
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        if top_level:
            return [Finding("unsupported", location, f"bad zip: {exc}")]
        return []
    members = _zip_members_lastwins(zf)
    names = list(members)
    files = [n for n in names if not n.endswith("/")]

    # Keras v3 archive.
    if "config.json" in members and (
            "model.weights.h5" in members or "metadata.json" in members):
        try:
            cfg = _zip_read(zf, members["config.json"])
        except Exception as exc:  # noqa: BLE001
            return [Finding("unsupported", f"{location}!config.json",
                            f"bad keras config: {exc}")]
        return keras_scan.scan_keras_config_bytes(cfg, f"{location}!config.json")

    # PyTorch zip checkpoint: torch.load reads only <prefix>/data.pkl (the
    # object graph); the sibling data/<n> storages and version file are not
    # independently loaded.  torch.load requires the local-header magic at
    # offset 0, so a zip with bytes before the first header is NOT a torch
    # checkpoint but a generic archive whose every member is loaded.  Also
    # require the telltale version file beside data.pkl.
    magic_at_zero = data[:4] == b"PK\x03\x04"
    torch_pkls = []
    if magic_at_zero:
        for n in files:
            if n == "data.pkl" or n.endswith("/data.pkl"):
                prefix = n[:-len("data.pkl")]
                if (prefix + "version") in members:
                    torch_pkls.append(n)
    if torch_pkls:
        findings = []
        for n in torch_pkls:
            try:
                raw = _zip_read(zf, members[n])
            except Exception as exc:  # noqa: BLE001
                findings.append(Finding("unsupported", f"{location}!{n}",
                                        f"bad member: {exc}"))
                continue
            findings.extend(pickle_scan.scan_pickle(
                raw, f"{location}!{n}", torch_context=True, nest=nest + 1))
        return findings

    # NumPy .npz: a zip of .npy members.
    if files and all(n.endswith(".npy") for n in files):
        findings = []
        for n in files:
            try:
                raw = _zip_read(zf, members[n])
            except Exception as exc:  # noqa: BLE001
                findings.append(Finding("unsupported", f"{location}!{n}",
                                        f"bad member: {exc}"))
                continue
            findings.extend(analyze_npy_bytes(raw, f"{location}!{n}", nest + 1))
        return findings

    # Generic zip archive: analyse each member by its own content.
    return _analyze_members(
        [(n, lambda n=n: _zip_read(zf, members[n])) for n in files],
        location, nest)


def _try_tar(data, location, nest, top_level):
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data))
    except Exception:  # noqa: BLE001 - not a tar
        return None
    try:
        members = {}
        for m in tf.getmembers():
            if m.isfile():
                members[m.name] = m  # last wins
        loaders = []
        for name, m in members.items():
            def _load(m=m):
                fh = tf.extractfile(m)
                if fh is None:
                    return b""
                raw = fh.read(MAX_DECOMPRESS + 1)
                if len(raw) > MAX_DECOMPRESS:
                    raise ValueError("tar member too large")
                return raw
            loaders.append((name, _load))
        return _analyze_members(loaders, location, nest)
    except Exception as exc:  # noqa: BLE001
        if top_level:
            return [Finding("unsupported", location, f"bad tar: {exc}")]
        return []


def _analyze_members(loaders, location, nest):
    findings = []
    count = 0
    for name, load in loaders:
        count += 1
        if count > MAX_MEMBERS:
            break
        try:
            raw = load()
        except Exception as exc:  # noqa: BLE001
            findings.append(Finding("unsupported", f"{location}!{name}",
                                    f"bad member: {exc}"))
            continue
        findings.extend(analyze_artifact(raw, f"{location}!{name}",
                                         nest=nest + 1, top_level=False))
    return findings


def analyze_npy_bytes(data, location, nest=0):
    """A .npy with object dtype carries a pickle in its data section."""
    if data[:6] != b"\x93NUMPY":
        # numpy.load on a .npz (zip) or raw pickle
        if _looks_zip(data):
            return _analyze_zip(data, location, nest, top_level=False)
        if _looks_pickle(data):
            return pickle_scan.scan_pickle(data, location, nest=nest)
        return [Finding("unsupported", location, "not a valid .npy")]
    try:
        major = data[6]
        if major == 1:
            hlen = int.from_bytes(data[8:10], "little")
            hstart = 10
        else:
            hlen = int.from_bytes(data[8:12], "little")
            hstart = 12
        header = data[hstart:hstart + hlen].decode("latin-1")
        hdr = ast.literal_eval(header.strip())
        descr = hdr.get("descr")
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"bad .npy header: {exc}")]
    if not _descr_has_object(descr):
        return []  # plain numeric array: no pickle, nothing executes
    body = data[hstart + hlen:]
    return pickle_scan.scan_pickle(body, f"{location}|npy", nest=nest + 1)


def _descr_has_object(descr):
    """Whether a numpy dtype descriptor contains an object field (which pickles).

    A descriptor is a typestring (``'|O'``, ``'<f4'``) or a list of fields, each
    ``(name, type[, shape])``.  Only the type position may carry the object
    kind ``'O'``; a field *name* containing an 'O' must not count.
    """
    if isinstance(descr, str):
        # Typestring: object kind is 'O'/'|O'; byteorder/size chars are never O.
        return "O" in descr
    if isinstance(descr, list):
        return any(_field_has_object(f) for f in descr)
    if isinstance(descr, tuple):
        # A subarray descriptor (base_descr, shape); the object kind, if any,
        # is in the base at position 0 (never in the shape).
        return bool(descr) and _descr_has_object(descr[0])
    return False


def _field_has_object(field):
    # A field is (name, type) or (name, type, shape); only the type carries the
    # object kind, never the name.
    if isinstance(field, (list, tuple)) and len(field) >= 2:
        return _descr_has_object(field[1])
    return False


def _looks_pickle(data):
    """True only for a well-formed pickle: its first stream parses to a STOP.

    Requiring a STOP avoids mistaking a raw binary member (e.g. a tensor-storage
    blob that happens to start with a pickle-looking byte) for a pickle.
    """
    if not data:
        return False
    import pickletools
    try:
        count = 0
        for opcode, _arg, _pos in pickletools.genops(io.BytesIO(data)):
            count += 1
            if opcode.name == "STOP":
                return True
            if count > 200000:
                return False
    except Exception:  # noqa: BLE001
        return False
    return False
