"""Minimal, bounded HDF5 reader for the root group's string attributes.

Keras stores the model configuration in the root attribute ``model_config``.
Reading it with libhdf5 risks hanging the process on a malformed file (the
library holds the GIL), so the common layouts written by h5py are parsed here
directly: superblock v0-v3, version 1 and 2 object headers with continuation
chunks, attribute messages v1-v3, fixed-length and variable-length (global
heap) strings.  ``read_root_attrs`` returns None for anything it does not
understand, and the caller then falls back to h5py.
"""
from __future__ import annotations

import struct

SIG = b"\x89HDF\r\n\x1a\n"
UNDEF = 0xFFFFFFFFFFFFFFFF


class H5Error(Exception):
    """Malformed or inconsistent file."""


class H5Unsupported(Exception):
    """A valid layout this reader does not implement."""


CORRUPT = "corrupt"
UNSUPPORTED = "unsupported"


class _Reader:
    def __init__(self, data: bytes, base: int, so: int, sl: int):
        self.d = data
        self.base = base
        self.so = so
        self.sl = sl

    def u(self, off, n):
        if off < 0 or off + n > len(self.d):
            raise H5Error("read past end of file")
        return int.from_bytes(self.d[off:off + n], "little")

    def b(self, off, n):
        if off < 0 or n < 0 or off + n > len(self.d):
            raise H5Error("read past end of file")
        return self.d[off:off + n]

    def addr(self, a):
        if a == UNDEF or a == (1 << (8 * self.so)) - 1:
            raise H5Error("undefined address")
        return self.base + a


def _find_superblock(data):
    off = 0
    while off < len(data) and off <= 1 << 20:
        if data[off:off + 8] == SIG:
            return off
        off = 512 if off == 0 else off * 2
    raise H5Error("no superblock")


def read_root_attrs(data: bytes, max_messages=4096):
    """{name: value} of the root group's attributes whose values are strings;
    UNSUPPORTED for a layout this reader does not implement; CORRUPT for a
    file that is malformed (which libhdf5 may not survive)."""
    try:
        return _read_root_attrs(data, max_messages)
    except H5Unsupported:
        return UNSUPPORTED
    except (H5Error, struct.error, IndexError, ValueError, OverflowError, RecursionError):
        return CORRUPT


def _read_root_attrs(data, max_messages):
    sb = _find_superblock(data)
    ver = data[sb + 8]
    if ver in (0, 1):
        so, sl = data[sb + 13], data[sb + 14]
        if so not in (2, 4, 8) or sl not in (2, 4, 8):
            raise H5Error("sizes")
        p = sb + 24 + (4 if ver == 1 else 0)
        r = _Reader(data, 0, so, sl)
        base = r.u(p, so)
        r.base = base if base != UNDEF else sb
        p += 4 * so  # base, free-space, end-of-file, driver info
        # root group symbol table entry: link name offset, object header address
        ohdr = r.u(p + so, so)
    elif ver in (2, 3):
        so, sl = data[sb + 9], data[sb + 10]
        if so not in (2, 4, 8) or sl not in (2, 4, 8):
            raise H5Error("sizes")
        r = _Reader(data, 0, so, sl)
        p = sb + 12
        base = r.u(p, so)
        r.base = base if base != UNDEF else sb
        ohdr = r.u(p + 3 * so, so)
    else:
        raise H5Error("superblock version")  # only versions 0-3 exist
    attrs = {}
    for mtype, mdata_off, msize in _messages(r, r.addr(ohdr), max_messages):
        if mtype == 0x0C:
            name, value = _attribute(r, mdata_off, msize)
            if name is not None:
                attrs[name] = value
        elif mtype == 0x15:
            # attribute info: attributes may live in dense storage
            attrs.setdefault("\0dense", True)
    if attrs.pop("\0dense", False) and "model_config" not in attrs:
        raise H5Unsupported("dense attribute storage")
    return attrs


def _messages(r, off, max_messages):
    """Yield (type, data offset, size) for all messages of an object header."""
    out = []
    if r.b(off, 4) == b"OHDR":
        _ohdr_v2(r, off, out, max_messages)
    else:
        if r.u(off, 1) != 1:
            raise H5Error("object header version")
        nmsgs = r.u(off + 2, 2)
        size = r.u(off + 8, 4)
        chunks = [(off + 16, size)]
        seen = 0
        ci = 0
        while ci < len(chunks):
            start, length = chunks[ci]
            ci += 1
            p = start
            end = start + length
            if end > len(r.d):
                raise H5Error("chunk past end")
            while p + 8 <= end:
                mtype = r.u(p, 2)
                msize = r.u(p + 2, 2)
                dp = p + 8
                if dp + msize > end:
                    raise H5Error("message past chunk")
                if mtype == 0x10:
                    caddr = r.u(dp, r.so)
                    clen = r.u(dp + r.so, r.sl)
                    if len(chunks) > 256:
                        raise H5Error("too many chunks")
                    chunks.append((r.addr(caddr), clen))
                else:
                    out.append((mtype, dp, msize))
                seen += 1
                if seen > max_messages:
                    raise H5Error("too many messages")
                p = dp + msize
                p = start + (((p - start) + 7) & ~7)
        _ = nmsgs
    return out


def _ohdr_v2(r, off, out, max_messages):
    if r.u(off + 4, 1) != 2:
        raise H5Error("object header v2 version")
    flags = r.u(off + 5, 1)
    p = off + 6
    if flags & 0x20:
        p += 16
    if flags & 0x10:
        p += 4
    nsz = 1 << (flags & 3)
    size0 = r.u(p, nsz)
    p += nsz
    chunks = [(p, size0)]
    ci = 0
    seen = 0
    while ci < len(chunks):
        start, length = chunks[ci]
        ci += 1
        q = start
        end = start + length
        if end > len(r.d):
            raise H5Error("chunk past end")
        hdr = 4 + (2 if flags & 0x04 else 0)
        while q + hdr <= end:
            mtype = r.u(q, 1)
            msize = r.u(q + 1, 2)
            dp = q + hdr
            if dp + msize > end:
                break  # gap
            if mtype == 0x10:
                caddr = r.u(dp, r.so)
                clen = r.u(dp + r.so, r.sl)
                a = r.addr(caddr)
                if r.b(a, 4) != b"OCHK":
                    raise H5Error("continuation signature")
                if len(chunks) > 256:
                    raise H5Error("too many chunks")
                chunks.append((a + 4, clen - 8))
            elif mtype != 0:
                out.append((mtype, dp, msize))
            seen += 1
            if seen > max_messages:
                raise H5Error("too many messages")
            q = dp + msize


def _attribute(r, off, size):
    ver = r.u(off, 1)
    if ver == 1:
        nlen, tlen, slen = r.u(off + 2, 2), r.u(off + 4, 2), r.u(off + 6, 2)
        p = off + 8
        name = r.b(p, nlen)
        p += (nlen + 7) & ~7
        tp = p
        p += (tlen + 7) & ~7
        sp = p
        p += (slen + 7) & ~7
    elif ver in (2, 3):
        nlen, tlen, slen = r.u(off + 2, 2), r.u(off + 4, 2), r.u(off + 6, 2)
        p = off + 8 + (1 if ver == 3 else 0)
        name = r.b(p, nlen)
        p += nlen
        tp = p
        p += tlen
        sp = p
        p += slen
    else:
        raise H5Error("attribute version")
    if ver >= 2 and r.u(off + 1, 1) & 0x03:
        raise H5Unsupported("shared datatype or dataspace in attribute")
    name = name.split(b"\x00", 1)[0].decode("utf-8", "replace")
    npoints = _dataspace_points(r, sp)
    if npoints != 1:
        return name, None
    value = _string_value(r, tp, p)
    return name, value


def _dataspace_points(r, off):
    ver = r.u(off, 1)
    ndims = r.u(off + 1, 1)
    if ver == 1:
        p = off + 8
    elif ver == 2:
        if r.u(off + 3, 1) == 2:
            return 0  # null dataspace
        p = off + 4
    else:
        raise H5Error("dataspace version")
    n = 1
    for k in range(ndims):
        n *= r.u(p + k * r.sl, r.sl)
    return n


def _string_value(r, tp, dp):
    cls = r.u(tp, 1) & 0x0F
    size = r.u(tp + 4, 4)
    if cls == 3:  # fixed-length string
        return r.b(dp, size).split(b"\x00", 1)[0]
    if cls == 9:  # variable-length
        bits = r.u(tp + 1, 3)
        if bits & 0x0F not in (0, 1):
            return None
        length = r.u(dp, 4)
        coll = r.u(dp + 4, r.so)
        idx = r.u(dp + 4 + r.so, 4)
        return _global_heap_object(r, r.addr(coll), idx, length)
    return None


def _global_heap_object(r, addr, idx, length):
    if r.b(addr, 4) != b"GCOL":
        raise H5Error("global heap signature")
    csize = r.u(addr + 8, r.sl)
    p = addr + 8 + r.sl
    end = min(addr + csize, len(r.d))
    n = 0
    while p + 8 + r.sl <= end:
        oid = r.u(p, 2)
        osize = r.u(p + 8, r.sl)
        dp = p + 8 + r.sl
        if oid == 0:
            break
        if oid == idx:
            return r.b(dp, min(osize, length) if length else osize)
        p = dp + ((osize + 7) & ~7)
        n += 1
        if n > 1 << 20:
            raise H5Error("global heap too long")
    raise H5Error("global heap object not found")
