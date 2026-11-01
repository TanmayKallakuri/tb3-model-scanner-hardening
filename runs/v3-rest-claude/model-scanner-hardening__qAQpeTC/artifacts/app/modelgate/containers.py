"""In-memory unpacking of compressed and archived artifacts.

Nothing is written to disk.  Decompression is bounded; a stream that is
truncated or corrupt yields whatever a streaming reader would have produced
before failing (a loader reading the stream would already have run that
part), together with a flag saying whether it was complete.
"""
from __future__ import annotations

import bz2
import io
import lzma
import struct
import tarfile
import zipfile
import zlib

MAX_OUT = 128 * 1024 * 1024  # bytes produced by one decompression
TOTAL_OUT = [512 * 1024 * 1024]  # bytes all decompressions of one scan may still produce


def _budget(n):
    """Account for ``n`` decompressed bytes; the remaining allowance."""
    TOTAL_OUT[0] -= n
    return max(0, TOTAL_OUT[0])


def sniff(data: bytes) -> str:
    """Format of a blob, decided by content."""
    if data[:2] == b"\x1f\x8b":
        return "gzip"
    if data[:3] == b"BZh" and len(data) >= 10 and (data[4:10] == b"1AY&SY" or data[4:10] == b"\x17rE8P\x90"):
        return "bz2"
    if data[:6] == b"\xfd7zXZ\x00":
        return "xz"
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return "zip"
    if data[:6] == b"\x93NUMPY":
        return "npy"
    if data[:8] == b"\x89HDF\r\n\x1a\n":
        return "hdf5"
    for off in (512, 1024, 2048, 4096, 8192):
        if data[off:off + 8] == b"\x89HDF\r\n\x1a\n":
            return "hdf5"
    if len(data) >= 512 and data[257:262] == b"ustar":
        return "tar"
    if is_zlib(data):
        return "zlib"
    if len(data) > 13 and data[0] <= 0xE0 and data[1] == 0 and data[0] == 0x5D:
        # lzma "alone" (joblib's lzma compressor); "]\x00" never starts a
        # valid pickle, so nothing is lost by trying
        return "lzma-alone"
    if len(data) >= 512 and _is_tar(data):
        return "tar"
    return "other"


def is_zlib(data: bytes) -> bool:
    if len(data) < 2:
        return False
    cmf, flg = data[0], data[1]
    return (cmf & 0x0F) == 8 and (cmf >> 4) <= 7 and ((cmf << 8) | flg) % 31 == 0 and cmf == 0x78


def _is_tar(data: bytes) -> bool:
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
            return tf.next() is not None
    except Exception:  # noqa: BLE001
        return False


def maybe_zip_with_prefix(data: bytes) -> bool:
    """True if ``data`` is a zip archive preceded by other bytes (zipfile
    accepts it): its end-of-central-directory record, with its comment, ends
    the file.  A zip merely embedded inside other data (bytes inside a
    pickle) is not one."""
    tail = data[-(1 << 16) - 22:]
    i = tail.rfind(b"PK\x05\x06")
    if i < 0 or len(tail) - i < 22:
        return False
    comment_len = struct.unpack("<H", tail[i + 20:i + 22])[0]
    if i + 22 + comment_len != len(tail):
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            return len(zf.infolist()) > 0
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------ decompressors
def gunzip(data: bytes):
    """All members of a (possibly multi-member) gzip stream."""
    out = []
    total = MAX_OUT - min(MAX_OUT, TOTAL_OUT[0])
    pos = 0
    ok = True
    n = 0
    while pos < len(data):
        if n > 0:
            # padding between/after members is tolerated by gzip
            while pos < len(data) and data[pos] == 0:
                pos += 1
            if pos >= len(data):
                break
        if data[pos:pos + 2] != b"\x1f\x8b":
            ok = n > 0 and False
            break
        try:
            hdr_end = _gzip_header_end(data, pos)
        except ValueError:
            ok = False
            break
        d = zlib.decompressobj(-15)
        try:
            chunk = d.decompress(data[hdr_end:], MAX_OUT - total)
        except zlib.error:
            ok = False
            break
        out.append(chunk)
        total += len(chunk)
        if total >= MAX_OUT:
            ok = False
            break
        if not d.eof:
            ok = False
            break
        rest = d.unused_data
        if len(rest) < 8:
            ok = False
            break
        pos = len(data) - len(rest) + 8
        n += 1
    res = b"".join(out)
    _budget(len(res))
    return res, ok


def _gzip_header_end(data, pos):
    if len(data) < pos + 10:
        raise ValueError("short header")
    method, flags = data[pos + 2], data[pos + 3]
    if method != 8:
        raise ValueError("unknown compression method")
    p = pos + 10
    if flags & 4:  # FEXTRA
        if len(data) < p + 2:
            raise ValueError("short")
        xlen = struct.unpack("<H", data[p:p + 2])[0]
        p += 2 + xlen
    if flags & 8:  # FNAME
        i = data.find(b"\x00", p)
        if i < 0:
            raise ValueError("short")
        p = i + 1
    if flags & 16:  # FCOMMENT
        i = data.find(b"\x00", p)
        if i < 0:
            raise ValueError("short")
        p = i + 1
    if flags & 2:  # FHCRC
        p += 2
    if p > len(data):
        raise ValueError("short")
    return p


def bunzip(data: bytes):
    out = []
    total = MAX_OUT - min(MAX_OUT, TOTAL_OUT[0])
    ok = True
    rest = data
    while rest:
        d = bz2.BZ2Decompressor()
        try:
            chunk = d.decompress(rest, MAX_OUT - total)
        except (OSError, EOFError, ValueError):
            ok = False
            break
        out.append(chunk)
        total += len(chunk)
        if total >= MAX_OUT or not d.eof:
            ok = d.eof
            break
        rest = d.unused_data
        if rest and not rest.startswith(b"BZh"):
            break
    return b"".join(out), ok


def unxz(data: bytes, auto=False):
    out = []
    total = MAX_OUT - min(MAX_OUT, TOTAL_OUT[0])
    ok = True
    rest = data
    fmt = lzma.FORMAT_AUTO if auto else lzma.FORMAT_XZ
    while rest:
        d = lzma.LZMADecompressor(format=fmt)
        try:
            chunk = d.decompress(rest, MAX_OUT - total)
        except (lzma.LZMAError, EOFError, ValueError):
            ok = False
            break
        out.append(chunk)
        total += len(chunk)
        if total >= MAX_OUT or not d.eof:
            ok = d.eof
            break
        rest = d.unused_data.lstrip(b"\x00")
        if rest and not (rest.startswith(b"\xfd7zXZ\x00") or auto):
            break
    res = b"".join(out)
    _budget(len(res))
    return res, ok


def unlzma_alone(data: bytes):
    d = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE)
    try:
        out = d.decompress(data, min(MAX_OUT, max(1, TOTAL_OUT[0])))
    except (lzma.LZMAError, EOFError, ValueError):
        return b"", False
    _budget(len(out))
    return out, d.eof


def unzlib(data: bytes):
    d = zlib.decompressobj()
    try:
        out = d.decompress(data, min(MAX_OUT, max(1, TOTAL_OUT[0])))
    except zlib.error:
        return b"", False
    _budget(len(out))
    return out, d.eof


# --------------------------------------------------------------------- zip
def zip_members(data: bytes):
    """[(name, ZipInfo)] in central-directory order, one entry per name: the
    last entry with that name wins, as ``zipfile`` and extraction do."""
    zf = zipfile.ZipFile(io.BytesIO(data))
    infos = zf.infolist()
    last = {}
    order = []
    for info in infos:
        if info.filename not in last:
            order.append(info.filename)
        last[info.filename] = info
    return zf, [(name, last[name]) for name in order], infos


def zip_read(zf: zipfile.ZipFile, info: zipfile.ZipInfo, data: bytes):
    """Content of a member; tolerant of a bad CRC (a streaming reader would
    have consumed the content before noticing)."""
    if info.flag_bits & 0x1:
        return None  # encrypted
    try:
        with zf.open(info) as fh:
            out = fh.read(min(MAX_OUT, max(1, TOTAL_OUT[0])))
            _budget(len(out))
            return out
    except (zipfile.BadZipFile, EOFError, zlib.error, OSError, NotImplementedError, ValueError,
            lzma.LZMAError) as exc:
        if isinstance(exc, NotImplementedError):
            return None
    # manual extraction from the local header
    try:
        base = info.header_offset + getattr(zf, "_start_disk", 0)
        # zipfile adjusts header_offset for bytes before the archive
        fp = zf.fp
        fp.seek(info.header_offset)
        hdr = fp.read(30)
        if hdr[:4] != b"PK\x03\x04":
            return None
        nlen, xlen = struct.unpack("<HH", hdr[26:30])
        start = info.header_offset + 30 + nlen + xlen
        fp.seek(start)
        raw = fp.read(info.compress_size)
        if info.compress_type == zipfile.ZIP_STORED:
            return raw
        if info.compress_type == zipfile.ZIP_DEFLATED:
            return zlib.decompressobj(-15).decompress(raw, MAX_OUT)
        if info.compress_type == zipfile.ZIP_BZIP2:
            return bz2.BZ2Decompressor().decompress(raw, MAX_OUT)
        if info.compress_type == zipfile.ZIP_LZMA:
            return None
    except Exception:  # noqa: BLE001
        return None
    return None


# --------------------------------------------------------------------- tar
def tar_members(data: bytes):
    """[(name, content)] of the regular files of an uncompressed tar; the
    last member with a name wins (extraction overwrites)."""
    out = {}
    order = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
        while True:
            try:
                m = tf.next()
            except tarfile.ReadError:
                break
            except Exception:  # noqa: BLE001
                break
            if m is None:
                break
            if not (m.isreg() or m.islnk()):
                continue
            try:
                fh = tf.extractfile(m)
                content = fh.read(min(MAX_OUT, max(1, TOTAL_OUT[0]))) if fh is not None else None
                if content is not None:
                    _budget(len(content))
            except Exception:  # noqa: BLE001
                content = None
            if content is None:
                continue
            name = m.name
            if name not in out:
                order.append(name)
            out[name] = content
    return [(n, out[n]) for n in order]
