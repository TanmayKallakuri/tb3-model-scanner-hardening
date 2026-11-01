"""Read the string attributes of an HDF5 file's root group, in pure Python.

Keras keeps the model configuration in root attributes (``model_config``,
``training_config``).  Parsing untrusted HDF5 metadata with the C library can
hang on a malformed file, so this reader handles the structures h5py writes
for such attributes -- superblock v0-v3, object header v1/v2 (with
continuation blocks), attribute messages v1-v3, fixed-length and
variable-length strings (global heap) -- with every read bounds-checked and
every loop bounded.  Structures it does not implement (dense attribute
storage, shared datatypes) raise ``H5Unsupported``; malformed ones raise
``H5Corrupt``.
"""
from __future__ import annotations

SIG = b"\x89HDF\r\n\x1a\n"
MAX_MESSAGES = 20000
MAX_BLOCKS = 256


class H5Corrupt(Exception):
    pass


class H5Unsupported(Exception):
    pass


class _R:
    def __init__(self, data: bytes, base: int):
        self.data = data
        self.base = base

    def bytes(self, off: int, n: int) -> bytes:
        if off < 0 or n < 0 or off + n > len(self.data):
            raise H5Corrupt("read outside the file")
        return self.data[off:off + n]

    def uint(self, off: int, n: int) -> int:
        return int.from_bytes(self.bytes(off, n), "little")


def _undef(v, size):
    return v == (1 << (8 * size)) - 1


def find_superblock(data: bytes) -> int:
    for off in (0, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
        if data[off:off + 8] == SIG:
            return off
    raise H5Corrupt("no HDF5 signature")


def root_attributes(data: bytes) -> dict:
    """{name: value} for the root group's string attributes (value is str or
    bytes); other attribute types are skipped."""
    sb = find_superblock(data)
    r = _R(data, 0)
    version = r.uint(sb + 8, 1)
    if version in (0, 1):
        so, sl = r.uint(sb + 13, 1), r.uint(sb + 14, 1)
        if so not in (2, 4, 8) or sl not in (2, 4, 8):
            raise H5Corrupt("bad offset/length sizes")
        p = sb + 24 + (4 if version == 1 else 0)
        base = r.uint(p, so)
        entry = p + 4 * so
        root = r.uint(entry + so, so)
    elif version in (2, 3):
        so, sl = r.uint(sb + 9, 1), r.uint(sb + 10, 1)
        if so not in (2, 4, 8) or sl not in (2, 4, 8):
            raise H5Corrupt("bad offset/length sizes")
        base = r.uint(sb + 12, so)
        root = r.uint(sb + 12 + 3 * so, so)
    else:
        raise H5Unsupported(f"superblock version {version}")
    if base == 0:
        base = sb  # addresses are relative to the superblock when no base is recorded
    r.base = base
    reader = _Reader(r, so, sl)
    return reader.attributes(root)


class _Reader:
    def __init__(self, r: _R, so: int, sl: int):
        self.r = r
        self.so = so
        self.sl = sl
        self.heaps = {}

    def addr(self, a: int) -> int:
        return self.r.base + a

    # ------------------------------------------------------------ object headers
    def messages(self, oh_addr: int):
        r = self.r
        start = self.addr(oh_addr)
        out = []
        if r.bytes(start, 4) == b"OHDR":
            self._v2_messages(start, out)
        else:
            if r.uint(start, 1) != 1:
                raise H5Corrupt("unknown object header version")
            nmsgs = r.uint(start + 2, 2)
            size = r.uint(start + 8, 4)
            blocks = [(start + 16, size)]
            seen = set()
            count = 0
            while blocks:
                if len(seen) > MAX_BLOCKS:
                    raise H5Corrupt("too many header blocks")
                pos, size = blocks.pop(0)
                if (pos, size) in seen:
                    continue
                seen.add((pos, size))
                end = pos + size
                if end > len(r.data):
                    raise H5Corrupt("header block outside the file")
                while pos + 8 <= end:
                    mtype = r.uint(pos, 2)
                    msize = r.uint(pos + 2, 2)
                    mflags = r.uint(pos + 4, 1)
                    body = pos + 8
                    if body + msize > end:
                        raise H5Corrupt("header message overruns its block")
                    count += 1
                    if count > MAX_MESSAGES:
                        raise H5Corrupt("too many header messages")
                    if mtype == 0x10:
                        caddr = r.uint(body, self.so)
                        clen = r.uint(body + self.so, self.sl)
                        blocks.append((self.addr(caddr), clen))
                    else:
                        out.append((mtype, mflags, body, msize))
                    pos = body + msize
                    if msize % 8:
                        pos += 8 - msize % 8
        return out

    def _v2_messages(self, start, out):
        r = self.r
        if r.uint(start + 4, 1) != 2:
            raise H5Corrupt("unknown OHDR version")
        flags = r.uint(start + 5, 1)
        pos = start + 6
        if flags & 0x20:
            pos += 16
        if flags & 0x10:
            pos += 4
        csize_len = 1 << (flags & 3)
        size = r.uint(pos, csize_len)
        pos += csize_len
        creation = bool(flags & 0x04)
        blocks = [(pos, pos + size)]
        seen = set()
        count = 0
        while blocks:
            if len(seen) > MAX_BLOCKS:
                raise H5Corrupt("too many header blocks")
            pos, end = blocks.pop(0)
            if (pos, end) in seen:
                continue
            seen.add((pos, end))
            if end > len(r.data):
                raise H5Corrupt("header chunk outside the file")
            hdr = 4 + (2 if creation else 0)
            while pos + hdr <= end:
                mtype = r.uint(pos, 1)
                msize = r.uint(pos + 1, 2)
                mflags = r.uint(pos + 3, 1)
                body = pos + hdr
                if body + msize > end:
                    break  # gap at the end of a chunk
                count += 1
                if count > MAX_MESSAGES:
                    raise H5Corrupt("too many header messages")
                if mtype == 0x10:
                    caddr = self.addr(r.uint(body, self.so))
                    clen = r.uint(body + self.so, self.sl)
                    if r.bytes(caddr, 4) != b"OCHK":
                        raise H5Corrupt("bad continuation block")
                    blocks.append((caddr + 4, caddr + clen - 4))
                else:
                    out.append((mtype, mflags, body, msize))
                pos = body + msize

    # ------------------------------------------------------------ attributes
    def attributes(self, oh_addr: int) -> dict:
        out = {}
        for mtype, mflags, body, msize in self.messages(oh_addr):
            if mtype == 0x15:
                self._check_attr_info(body)
            elif mtype == 0x0C:
                if mflags & 0x02:
                    raise H5Unsupported("shared attribute message")
                name, value = self._attribute(body, msize)
                if name is not None and value is not None:
                    out[name] = value
        return out

    def _check_attr_info(self, body):
        r = self.r
        flags = r.uint(body + 1, 1)
        pos = body + 2 + (2 if flags & 1 else 0)
        heap = r.uint(pos, self.so)
        if not _undef(heap, self.so):
            raise H5Unsupported("dense attribute storage")

    def _attribute(self, body, msize):
        r = self.r
        version = r.uint(body, 1)
        flags = r.uint(body + 1, 1)
        nlen = r.uint(body + 2, 2)
        tlen = r.uint(body + 4, 2)
        slen = r.uint(body + 6, 2)
        if version == 1:
            pad = lambda n: n + (-n % 8)  # noqa: E731
            p = body + 8
            name = r.bytes(p, nlen)
            p += pad(nlen)
            tpos = p
            p += pad(tlen)
            spos = p
            p += pad(slen)
        elif version in (2, 3):
            if flags & 0x03:
                raise H5Unsupported("shared datatype or dataspace in attribute")
            p = body + 8 + (1 if version == 3 else 0)
            name = r.bytes(p, nlen)
            p += nlen
            tpos = p
            p += tlen
            spos = p
            p += slen
        else:
            raise H5Corrupt(f"attribute message version {version}")
        dpos = p
        if dpos > body + msize:
            raise H5Corrupt("attribute message overruns")
        name = name.split(b"\x00", 1)[0].decode("utf-8", "replace")
        count = self._dataspace_count(spos)
        if count != 1:
            return name, None
        return name, self._string_value(tpos, dpos, body + msize)

    def _dataspace_count(self, pos):
        r = self.r
        version = r.uint(pos, 1)
        rank = r.uint(pos + 1, 1)
        if version == 1:
            dims_at = pos + 8
            dtype = 1 if rank else 0
        elif version == 2:
            dtype = r.uint(pos + 3, 1)
            dims_at = pos + 4
        else:
            raise H5Corrupt("dataspace version")
        if dtype == 2:
            return 0
        if rank > 32:
            raise H5Corrupt("dataspace rank")
        n = 1
        for i in range(rank):
            n *= r.uint(dims_at + i * self.sl, self.sl)
        return n

    def _string_value(self, tpos, dpos, limit):
        r = self.r
        cv = r.uint(tpos, 1)
        cls = cv & 0x0F
        size = r.uint(tpos + 4, 4)
        if cls == 3:  # fixed-length string
            raw = r.bytes(dpos, size)
            return raw.split(b"\x00", 1)[0]
        if cls == 9:  # variable-length
            bits = r.uint(tpos + 1, 3)
            if bits & 0x0F != 1:
                return None  # a vlen sequence, not a string
            length = r.uint(dpos, 4)
            gaddr = r.uint(dpos + 4, self.so)
            gindex = r.uint(dpos + 4 + self.so, 4)
            obj = self._heap_object(gaddr, gindex)
            return obj[:length]
        return None

    def _heap_object(self, gaddr, index):
        r = self.r
        heap = self.heaps.get(gaddr)
        if heap is None:
            start = self.addr(gaddr)
            if r.bytes(start, 4) != b"GCOL":
                raise H5Corrupt("bad global heap")
            csize = r.uint(start + 8, self.sl)
            end = min(start + csize, len(r.data))
            pos = start + 8 + self.sl
            heap = {}
            count = 0
            while pos + 8 + self.sl <= end:
                idx = r.uint(pos, 2)
                osize = r.uint(pos + 8, self.sl)
                if idx == 0:
                    break
                body = pos + 8 + self.sl
                heap[idx] = (body, osize)
                pos = body + osize + (-osize % 8)
                count += 1
                if count > 1_000_000:
                    raise H5Corrupt("global heap too large")
            self.heaps[gaddr] = heap
        if index not in heap:
            raise H5Corrupt("global heap object missing")
        body, osize = heap[index]
        return r.bytes(body, osize)
