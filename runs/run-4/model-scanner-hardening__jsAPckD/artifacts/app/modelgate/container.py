"""Content-based format detection and recursive dispatch.

Content, not extension, decides the format at every level.  Compressed and
archived artifacts are unpacked in memory (never to disk) and each member is
analysed by its own format, nested up to the policy's depth.  Reading bytes
with ``gzip``/``bz2``/``lzma``/``zipfile``/``tarfile`` executes nothing.
"""
from __future__ import annotations

import io
import tarfile
import zipfile

from modelgate import keras_scan, numpy_scan, pklvm
from modelgate.report import Finding

MAX_DEPTH = 6
_MEMBER_CAP = 128 * 1024 * 1024       # skip absurd archive members (bomb guard)
_DECOMPRESS_CAP = 256 * 1024 * 1024   # cap a single decompression
_MAX_MEMBERS = 20000                  # cap archive entries processed (bomb guard)
_MAX_GZIP_MEMBERS = 1000

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_NPY_MAGIC = b"\x93NUMPY"
_GZIP_MAGIC = b"\x1f\x8b"
_BZ2_MAGIC = b"BZh"
_XZ_MAGIC = b"\xfd7zXZ\x00"
_ZIP_MAGIC = b"PK\x03\x04"


def analyse_blob(data, location, depth=0, torch=False, toplevel=False):
    """Analyse an in-memory artifact blob; return a list of Findings.

    ``toplevel`` makes an unrecognised blob fail closed (unsupported);
    nested members that are not a code-bearing format are simply clean.
    """
    if depth > MAX_DEPTH:
        return []
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    if len(data) < 2 and not toplevel:
        return []

    try:
        # Strong, unambiguous magic first.
        if data[:2] == _GZIP_MAGIC:
            return _scan_gzip(data, location, depth)
        if data[:3] == _BZ2_MAGIC:
            return _scan_stream_compress(data, location, depth, "bz2")
        if data[:6] == _XZ_MAGIC:
            return _scan_stream_compress(data, location, depth, "xz")
        if data[:8] == _HDF5_MAGIC:
            return keras_scan.scan_h5_bytes(data, location, fail_closed=toplevel)
        if data[:4] == _ZIP_MAGIC:
            return _scan_zip(data, location, depth, toplevel=toplevel)
        if data[:6] == _NPY_MAGIC:
            return numpy_scan.scan_npy_bytes(data, location, depth, fail_closed=toplevel)
        if len(data) >= 262 and data[257:262] == b"ustar":
            return _scan_tar(data, location, depth)
        # Then pickle, before the weak container trials -- a pickle that merely
        # embeds a zip/tar signature in its data must not be read as an archive.
        if _looks_pickle(data):
            return pklvm.analyse_streams(data, location, persid_safe=torch, depth=depth,
                                         fail_closed=toplevel)
        # Weak trials: a zip with bytes before its first local header (the
        # central directory still decides), or a tar without the ustar magic.
        if zipfile.is_zipfile(io.BytesIO(data)):
            return _scan_zip(data, location, depth, toplevel=toplevel)
        if _is_tar(data):
            return _scan_tar(data, location, depth)
    except pklvm._Stop:
        return []
    except Exception as exc:  # noqa: BLE001
        if toplevel:
            return [Finding("unsupported", location, f"could not analyse: {type(exc).__name__}: {pklvm._trim(str(exc))}")]
        return []

    if toplevel:
        return [Finding("unsupported", location, "unrecognised artifact format")]
    return []


# --------------------------------------------------------------------------
# Compressed single-stream / multi-member formats
# --------------------------------------------------------------------------
def _scan_gzip(data, location, depth):
    findings = []
    # gzip may hold several concatenated members
    members = _gunzip_members(data)
    if not members:
        return [Finding("unsupported", location, "could not decompress gzip")] if depth == 0 else []
    for i, blob in enumerate(members):
        tag = f"{location}|gunzip" if len(members) == 1 else f"{location}|gunzip[{i}]"
        findings.extend(analyse_blob(blob, tag, depth + 1))
    return findings


def _gunzip_members(data):
    import zlib
    members = []
    pos = 0
    n = len(data)
    while pos < n and len(members) < _MAX_GZIP_MEMBERS:
        if data[pos:pos + 2] != _GZIP_MAGIC:
            break
        dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            out = dec.decompress(data[pos:], _DECOMPRESS_CAP)
        except Exception:  # noqa: BLE001
            break
        members.append(out)
        if dec.unused_data:
            consumed = n - len(dec.unused_data)
            if consumed <= pos:
                break
            pos = consumed
        else:
            break
    return members


def _scan_stream_compress(data, location, depth, which):
    try:
        if which == "bz2":
            import bz2
            out = bz2.BZ2Decompressor().decompress(data, _DECOMPRESS_CAP)
            tag = f"{location}|bunzip2"
        else:
            import lzma
            out = lzma.LZMADecompressor().decompress(data, _DECOMPRESS_CAP)
            tag = f"{location}|unxz"
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"could not decompress: {exc}")] if depth == 0 else []
    return analyse_blob(out, tag, depth + 1)


# --------------------------------------------------------------------------
# Zip (torch checkpoint / keras v3 / npz / generic archive)
# --------------------------------------------------------------------------
def _scan_zip(data, location, depth, toplevel=False):
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"could not open zip: {exc}")] if depth == 0 else []
    with zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]

        data_pkls = [n for n in names if n == "data.pkl" or n.endswith("/data.pkl")]
        if data_pkls:
            findings = []
            for n in data_pkls:
                try:
                    blob = zf.read(n)
                except Exception:  # noqa: BLE001
                    continue
                findings.extend(pklvm.analyse_streams(blob, f"{location}!{n}",
                                                      persid_safe=True, depth=depth + 1))
            return findings

        base = {n.rsplit("/", 1)[-1] for n in names}
        if "config.json" in base and "metadata.json" in base:
            return keras_scan.scan_keras_v3(zf, location, fail_closed=toplevel)

        if names and all(n.lower().endswith(".npy") for n in names):
            findings = []
            for n in names:
                blob = _safe_read(zf, n)
                if blob is None:
                    continue
                findings.extend(numpy_scan.scan_npy_bytes(blob, f"{location}!{n}", depth + 1))
            return findings

        # generic archive: analyse every member by its own format
        findings = []
        for count, info in enumerate(zf.infolist()):
            if count >= _MAX_MEMBERS:
                break
            if info.filename.endswith("/"):
                continue
            if info.file_size > _MEMBER_CAP:
                continue
            blob = _safe_read(zf, info.filename)
            if blob is None:
                continue
            findings.extend(analyse_blob(blob, f"{location}!{info.filename}", depth + 1))
        return findings


def _safe_read(zf, name):
    try:
        with zf.open(name) as fh:
            return fh.read(_MEMBER_CAP + 1)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# Tar
# --------------------------------------------------------------------------
def _is_tar(data):
    if len(data) < 512:
        return False
    if data[257:262] == b"ustar":
        return True
    try:
        with tarfile.open(fileobj=io.BytesIO(data)):
            return True
    except Exception:  # noqa: BLE001
        return False


def _scan_tar(data, location, depth):
    findings = []
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        return [Finding("unsupported", location, f"could not open tar: {exc}")] if depth == 0 else []
    with tf:
        for count, member in enumerate(tf):
            if count >= _MAX_MEMBERS:
                break
            if not member.isfile():
                continue
            if member.size > _MEMBER_CAP:
                continue
            try:
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                blob = fh.read(_MEMBER_CAP + 1)
            except Exception:  # noqa: BLE001
                continue
            findings.extend(analyse_blob(blob, f"{location}!{member.name}", depth + 1))
    return findings


# --------------------------------------------------------------------------
# Pickle sniffing
# --------------------------------------------------------------------------
_PICKLE_PROTO0_OPCODES = set(b"(.0123456789cdFgGhIjlLNpqQRsStuV}])ab")


def _looks_pickle(data):
    """Heuristic: does this blob start like a pickle stream?"""
    if not data:
        return False
    first = data[0]
    if first == 0x80:  # PROTO (protocol 2+)
        return len(data) >= 2 and data[1] <= 5
    # protocol 0/1 start with a printable opcode; require that the first
    # opcode parses to avoid treating raw tensor bytes as a pickle.
    if first in _PICKLE_PROTO0_OPCODES:
        import pickletools
        try:
            gen = pickletools.genops(io.BytesIO(data))
            next(gen)
            return True
        except Exception:  # noqa: BLE001
            return False
    return False
