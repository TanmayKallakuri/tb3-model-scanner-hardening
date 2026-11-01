"""Read the root group's attributes of an HDF5 file without the HDF5 library.

Keras keeps its model configuration in root attributes (``model_config``,
``training_config``).  This reader covers the layouts h5py writes for them:
superblock versions 0-3, object header versions 1 and 2 (with continuation
blocks), attribute message versions 1-3 stored compactly in the header, and
string values that are fixed-length or variable-length (global heap).  Every
read is bounds-checked and every loop is bounded, so a malformed file raises
``H5Unsupported`` instead of hanging; callers fall back to h5py only then.
"""
from __future__ import annotations


SIGNATURE = b"\x89HDF\r\n\x1a\n"


class H5Unsupported(Exception):
    """A valid layout this reader does not implement (use the HDF5 library)."""


class H5Corrupt(H5Unsupported):
    """The file is structurally broken."""


class _Reader:
    def __init__(self, data, base):
        self.data = data
        self.base = base
        self.so = 8   # size of offsets
        self.sl = 8   # size of lengths

    def need(self, pos, n):
        if pos < 0 or n < 0 or pos + n > len(self.data):
            raise H5Corrupt("read outside the file")

    def u(self, pos, n):
        self.need(pos, n)
        return int.from_bytes(self.data[pos:pos + n], "little")

    def addr(self, pos):
        return self.u(pos, self.so)

    def length(self, pos):
        return self.u(pos, self.sl)

    def undefined(self, a):
        return a == (1 << (8 * self.so)) - 1


def _find_superblock(data):
    off = 0
    while off <= len(data) - 8 and off <= 1 << 30:
        if data[off:off + 8] == SIGNATURE:
            return off
        off = 512 if off == 0 else off * 2
    raise H5Corrupt("no HDF5 signature")


def root_attributes(data):
    """Return {name: str-or-bytes} for the root group's compact attributes."""
    sb = _find_superblock(data)
    r = _Reader(data, sb)
    version = r.u(sb + 8, 1)
    if version in (0, 1):
        r.so = r.u(sb + 13, 1)
        r.sl = r.u(sb + 14, 1)
        if r.so not in (2, 4, 8) or r.sl not in (2, 4, 8):
            raise H5Corrupt("bad offset size")
        pos = sb + 24 + (4 if version == 1 else 0)
        base = r.addr(pos)
        pos += 4 * r.so            # base, free-space, end-of-file, driver addresses
        pos += r.so                # root entry: link name offset
        root = r.addr(pos)
    elif version in (2, 3):
        r.so = r.u(sb + 9, 1)
        r.sl = r.u(sb + 10, 1)
        if r.so not in (2, 4, 8) or r.sl not in (2, 4, 8):
            raise H5Corrupt("bad offset size")
        pos = sb + 12
        base = r.addr(pos)
        root = r.addr(pos + 3 * r.so)
    else:
        raise H5Unsupported(f"superblock version {version}")
    r.base = base
    messages = _object_header_messages(r, base + root)
    attrs = {}
    for mtype, body_pos, body_len in messages:
        if mtype == 0x000C:
            name, value = _attribute(r, body_pos, body_len)
            attrs[name] = value
        elif mtype == 0x0015:
            # attribute info: dense storage in a fractal heap when the address is defined
            ver = r.u(body_pos, 1)
            flags = r.u(body_pos + 1, 1)
            p = body_pos + 2 + (2 if flags & 1 else 0)
            if ver == 0 and not r.undefined(r.addr(p)):
                raise H5Unsupported("dense attribute storage")
    return attrs


def _object_header_messages(r, start):
    data = r.data
    out = []
    blocks = []
    if data[start:start + 4] == b"OHDR":
        version = r.u(start + 4, 1)
        if version != 2:
            raise H5Unsupported("object header version")
        flags = r.u(start + 5, 1)
        p = start + 6
        if flags & 0x20:
            p += 16
        if flags & 0x10:
            p += 4
        size_len = 1 << (flags & 3)
        chunk = r.u(p, size_len)
        p += size_len
        blocks.append((2, p, chunk, flags))  # chunk size excludes the checksum
    else:
        version = r.u(start, 1)
        if version != 1:
            raise H5Unsupported("object header version")
        size = r.u(start + 8, 4)
        blocks.append((1, start + 16, size, 0))
    seen = set()
    count = 0
    while blocks:
        ver, p, size, flags = blocks.pop(0)
        if (p, size) in seen or len(seen) > 256:
            raise H5Corrupt("object header continuation loop")
        seen.add((p, size))
        r.need(p, size)
        end = p + size
        while p < end:
            count += 1
            if count > 100000:
                raise H5Corrupt("too many header messages")
            if ver == 1:
                if p + 8 > end:
                    break
                mtype = r.u(p, 2)
                msize = r.u(p + 2, 2)
                body = p + 8
                p = body + msize
            else:
                if p + 4 > end:
                    break
                mtype = r.u(p, 1)
                msize = r.u(p + 1, 2)
                mflags = r.u(p + 3, 1)
                body = p + 4 + (2 if flags & 0x04 else 0)
                p = body + msize
            if p > end:
                raise H5Corrupt("header message overruns its block")
            if mtype == 0x0010:  # continuation
                caddr = r.addr(body)
                clen = r.length(body + r.so)
                cpos = r.base + caddr
                if ver == 2:
                    if data[cpos:cpos + 4] != b"OCHK":
                        raise H5Corrupt("bad continuation block")
                    blocks.append((2, cpos + 4, clen - 8, flags))  # minus signature and checksum
                else:
                    blocks.append((1, cpos, clen, 0))
            else:
                out.append((mtype, body, msize))
    return out


def _attribute(r, pos, length):
    end = pos + length
    version = r.u(pos, 1)
    if version not in (1, 2, 3):
        raise H5Unsupported("attribute message version")
    if version >= 2 and r.u(pos + 1, 1) & 0x03:
        raise H5Unsupported("shared datatype or dataspace")
    name_size = r.u(pos + 2, 2)
    dt_size = r.u(pos + 4, 2)
    ds_size = r.u(pos + 6, 2)
    p = pos + 8
    if version == 3:
        p += 1
    pad = (lambda n: (n + 7) & ~7) if version == 1 else (lambda n: n)
    r.need(p, name_size)
    name = r.data[p:p + name_size].split(b"\x00", 1)[0].decode("utf-8", "replace")
    p += pad(name_size)
    dt_pos = p
    p += pad(dt_size)
    ds_pos = p
    p += pad(ds_size)
    if p > end:
        raise H5Corrupt("attribute message overruns")
    count = _dataspace_count(r, ds_pos)
    value = _read_value(r, dt_pos, p, end, count)
    return name, value


def _dataspace_count(r, pos):
    version = r.u(pos, 1)
    rank = r.u(pos + 1, 1)
    if version == 1:
        p = pos + 8
    elif version == 2:
        if r.u(pos + 3, 1) == 2:  # null dataspace
            return 0
        p = pos + 4
    else:
        raise H5Unsupported("dataspace version")
    n = 1
    for i in range(rank):
        n *= r.length(p + i * r.sl)
        if n > 1 << 20:
            raise H5Unsupported("attribute too large")
    return n


def _read_value(r, dt_pos, p, end, count):
    cls = r.u(dt_pos, 1) & 0x0F
    size = r.u(dt_pos + 4, 4)
    if count != 1:
        if count == 0:
            return b""
        # arrays of strings (e.g. layer_names): first element is enough here
    if cls == 3:  # fixed-length string
        r.need(p, size)
        return r.data[p:p + size].rstrip(b"\x00 ")
    if cls == 9:  # variable-length
        bits = r.u(dt_pos + 1, 1)
        if bits & 0x0F not in (1, 0):
            raise H5Unsupported("variable-length type")
        n = r.u(p, 4)
        coll = r.addr(p + 4)
        idx = r.u(p + 4 + r.so, 4)
        return _global_heap_object(r, r.base + coll, idx, n)
    raise H5Unsupported(f"datatype class {cls}")


def _global_heap_object(r, pos, index, expected):
    if r.data[pos:pos + 4] != b"GCOL":
        raise H5Corrupt("bad global heap collection")
    size = r.length(pos + 8)
    end = pos + size
    r.need(pos, size)
    p = pos + 8 + r.sl
    steps = 0
    while p + 8 + r.sl <= end:
        steps += 1
        if steps > 1_000_000:
            raise H5Corrupt("global heap too large")
        idx = r.u(p, 2)
        osize = r.length(p + 8)
        body = p + 8 + r.sl
        if idx == 0:
            break
        if idx == index:
            r.need(body, osize)
            return bytes(r.data[body:body + min(osize, expected if expected else osize)])
        p = body + ((osize + 7) & ~7)
    raise H5Corrupt("global heap object not found")
