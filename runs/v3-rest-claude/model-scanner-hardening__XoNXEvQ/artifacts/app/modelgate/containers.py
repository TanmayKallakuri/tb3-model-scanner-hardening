"""Compressed and archived containers, read entirely in memory.

Formats are recognised by content, never by extension.  Archives are read the
way the registry unpacks them: a zip's central directory decides what it
holds and a later entry with the same name wins; tar members are read in
order and a later member with the same name wins as well.
"""
from __future__ import annotations

import bz2
import io
import lzma
import struct
import tarfile
import zipfile
import zlib

MAX_MEMBER = 512 << 20
MAX_TOTAL = 1536 << 20

GZIP_MAGIC = b"\x1f\x8b"
BZ2_MAGIC = b"BZh"
XZ_MAGIC = b"\xfd7zXZ\x00"
ZIP_LOCAL = b"PK\x03\x04"
ZIP_EOCD = b"PK\x05\x06"
NPY_MAGIC = b"\x93NUMPY"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


class ContainerError(Exception):
    pass


class TooLarge(ContainerError):
    pass


def gunzip(data: bytes, limit: int = MAX_MEMBER):
    """Decompress one or several concatenated gzip members (like
    gzip.decompress, but bounded).  Returns None if the data is not gzip."""
    out = []
    total = 0
    pos = 0
    first = True
    while pos < len(data):
        if data[pos:pos + 2] != GZIP_MAGIC:
            if first:
                return None
            if data[pos:].strip(b"\x00") == b"":
                break
            break  # trailing garbage after valid members: keep what was read
        hdr_end = _gzip_header_end(data, pos)
        if hdr_end is None:
            if first:
                return None
            break
        d = zlib.decompressobj(-zlib.MAX_WBITS)
        try:
            chunk = d.decompress(data[hdr_end:], max(1, limit - total))
        except zlib.error:
            if first:
                raise ContainerError("corrupt gzip data") from None
            break
        if d.unconsumed_tail:
            raise TooLarge("gzip member too large")
        total += len(chunk)
        out.append(chunk)
        if not d.eof:
            if first and not chunk:
                return None
            break  # truncated member: keep what was decompressed
        rest = d.unused_data
        pos = len(data) - len(rest) + 8  # skip CRC32 + ISIZE
        first = False
    return b"".join(out)


def _gzip_header_end(data, pos):
    if len(data) < pos + 10:
        return None
    method, flags = data[pos + 2], data[pos + 3]
    if method != 8:
        return None
    p = pos + 10
    if flags & 4:  # FEXTRA
        if len(data) < p + 2:
            return None
        xlen = struct.unpack("<H", data[p:p + 2])[0]
        p += 2 + xlen
    if flags & 8:  # FNAME
        z = data.find(b"\x00", p)
        if z < 0:
            return None
        p = z + 1
    if flags & 16:  # FCOMMENT
        z = data.find(b"\x00", p)
        if z < 0:
            return None
        p = z + 1
    if flags & 2:  # FHCRC
        p += 2
    return p if p <= len(data) else None


def _multi_stream(data, make, limit):
    out = []
    total = 0
    while data:
        d = make()
        try:
            chunk = d.decompress(data, max(1, limit - total))
        except (OSError, EOFError, lzma.LZMAError, ValueError):
            if not out:
                return None
            break
        total += len(chunk)
        out.append(chunk)
        if not d.eof:
            if total >= limit:
                raise TooLarge("decompressed data too large")
            break
        data = d.unused_data
        if data.strip(b"\x00") == b"":
            break
    return b"".join(out)


def bunzip2(data: bytes, limit: int = MAX_MEMBER):
    return _multi_stream(data, bz2.BZ2Decompressor, limit)


def unxz(data: bytes, limit: int = MAX_MEMBER):
    return _multi_stream(data, lambda: lzma.LZMADecompressor(format=lzma.FORMAT_AUTO), limit)


def unzlib(data: bytes, limit: int = MAX_MEMBER):
    try:
        d = zlib.decompressobj()
        out = d.decompress(data, limit)
    except zlib.error:
        return None
    if d.unconsumed_tail:
        raise TooLarge("zlib stream too large")
    if not d.eof:
        return None
    return out


def looks_like_zlib(data: bytes) -> bool:
    if len(data) < 2:
        return False
    cmf, flg = data[0], data[1]
    return cmf & 0x0F == 8 and (cmf >> 4) <= 7 and ((cmf << 8) | flg) % 31 == 0


def is_zip(data: bytes) -> bool:
    if data[:4] == ZIP_LOCAL:
        return True
    if data[:4] == ZIP_EOCD:
        return True
    tail = data[-(1 << 16) - 22:]
    if ZIP_EOCD not in tail:
        return False
    try:
        return zipfile.is_zipfile(io.BytesIO(data))
    except Exception:
        return False


def zip_members(data: bytes):
    """[(name, bytes)] in central-directory order, a later duplicate name
    replacing the earlier one (as zipfile and extraction do)."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError, ValueError, EOFError, NotImplementedError) as exc:
        raise ContainerError(f"unreadable zip: {exc}") from None
    order = []
    last = {}
    for info in zf.infolist():
        if info.filename not in last:
            order.append(info.filename)
        last[info.filename] = info
    out = []
    total = 0
    for name in order:
        info = last[name]
        if info.is_dir():
            continue
        if info.file_size > MAX_MEMBER:
            raise TooLarge(f"zip member {name} too large")
        total += info.file_size
        if total > MAX_TOTAL:
            raise TooLarge("zip contents too large")
        try:
            with zf.open(info) as fh:
                content = fh.read(MAX_MEMBER + 1)
        except RuntimeError as exc:  # encrypted
            raise ContainerError(f"zip member {name} cannot be read: {exc}") from None
        except (zipfile.BadZipFile, OSError, EOFError, zlib.error, lzma.LZMAError, ValueError,
                NotImplementedError) as exc:
            raise ContainerError(f"zip member {name} is corrupt: {exc}") from None
        if len(content) > MAX_MEMBER:
            raise TooLarge(f"zip member {name} too large")
        out.append((name, content))
    first = zf.infolist()[0].filename if zf.infolist() else None
    return out, first


def is_tar(data: bytes) -> bool:
    if len(data) < 512:
        return False
    if data[257:262] == b"ustar":
        return True
    # old V7 headers carry no magic: check the header checksum
    try:
        chk = int(data[148:156].split(b"\x00")[0].strip() or b"-1", 8)
    except ValueError:
        return False
    if chk < 0:
        return False
    hdr = bytearray(data[:512])
    hdr[148:156] = b" " * 8
    return sum(hdr) == chk


def tar_members(data: bytes):
    """[(name, bytes)] for the regular files of a tar archive, in order,
    a later member with the same name replacing the earlier one."""
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:")
        members = tf.getmembers()
    except (tarfile.TarError, OSError, EOFError, ValueError) as exc:
        raise ContainerError(f"unreadable tar: {exc}") from None
    order = []
    last = {}
    total = 0
    for m in members:
        name = m.name
        if m.isfile() or m.islnk():
            try:
                fh = tf.extractfile(m)
                content = fh.read(MAX_MEMBER + 1) if fh is not None else b""
            except (tarfile.TarError, OSError, EOFError, KeyError) as exc:
                raise ContainerError(f"tar member {name} unreadable: {exc}") from None
            if len(content) > MAX_MEMBER:
                raise TooLarge(f"tar member {name} too large")
            total += len(content)
            if total > MAX_TOTAL:
                raise TooLarge("tar contents too large")
            if name not in last:
                order.append(name)
            last[name] = content
        elif m.isdir():
            continue
        else:
            last.pop(name, None)
            if name in order:
                order.remove(name)
    return [(n, last[n]) for n in order]


class LazyZip:
    """Central-directory view of a zip; members are read on demand."""

    def __init__(self, data: bytes):
        try:
            self.zf = zipfile.ZipFile(io.BytesIO(data))
        except (zipfile.BadZipFile, OSError, ValueError, EOFError, NotImplementedError) as exc:
            raise ContainerError(f"unreadable zip: {exc}") from None
        infos = self.zf.infolist()
        self.first = infos[0].filename if infos else None
        self.order = []
        self.last = {}
        for info in infos:
            if info.filename not in self.last:
                self.order.append(info.filename)
            self.last[info.filename] = info
        self.total = 0

    def names(self):
        return [n for n in self.order if not self.last[n].is_dir()]

    def __contains__(self, name):
        return name in self.last and not self.last[name].is_dir()

    def read(self, name):
        info = self.last[name]
        if info.file_size > MAX_MEMBER:
            raise TooLarge(f"zip member {name} too large")
        self.total += info.file_size
        if self.total > MAX_TOTAL:
            raise TooLarge("zip contents too large")
        try:
            with self.zf.open(info) as fh:
                content = fh.read(MAX_MEMBER + 1)
        except RuntimeError as exc:  # encrypted
            raise ContainerError(f"zip member {name} cannot be read: {exc}") from None
        except (zipfile.BadZipFile, OSError, EOFError, zlib.error, lzma.LZMAError, ValueError,
                NotImplementedError) as exc:
            raise ContainerError(f"zip member {name} is corrupt: {exc}") from None
        if len(content) > MAX_MEMBER:
            raise TooLarge(f"zip member {name} too large")
        return content

    def members(self):
        return [(n, self.read(n)) for n in self.names()]
