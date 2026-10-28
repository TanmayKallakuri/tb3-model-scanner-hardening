"""Shared resource accounting for recursive, entirely in-memory parsing."""
from __future__ import annotations

import bz2
import io
import lzma
import time
import zlib

MAX_BYTES = 256 * 1024 * 1024
MAX_ITEMS = 1_000_000


class LimitError(Exception):
    pass


class Budget:
    def __init__(self):
        self.deadline = time.monotonic() + 45
        self.steps = 0
        self.bytes = 0

    def tick(self, n=1):
        self.steps += n
        if self.steps > 2_000_000 or time.monotonic() > self.deadline:
            raise LimitError("static analysis work limit exceeded")

    def charge(self, size):
        self.bytes += size
        if size > MAX_BYTES or self.bytes > 2 * MAX_BYTES:
            raise LimitError("expanded artifact exceeds analysis size limit")
        self.tick()


def decompress(data, kind):
    """Bound expansion, and preserve concatenated members (not just the first)."""
    out = bytearray()
    while data:
        if kind in ("gzip", "zlib"):
            dec = zlib.decompressobj(31 if kind == "gzip" else zlib.MAX_WBITS)
            block = dec.decompress(data, MAX_BYTES + 1 - len(out))
        elif kind == "bz2":
            dec = bz2.BZ2Decompressor()
            block = dec.decompress(data, MAX_BYTES + 1 - len(out))
        elif kind == "xz":
            dec = lzma.LZMADecompressor()
            block = dec.decompress(data, max_length=MAX_BYTES + 1 - len(out))
        else:
            raise ValueError("unknown compression")
        out.extend(block)
        if len(out) > MAX_BYTES or not dec.eof:
            raise LimitError("compressed data exceeds limit or is truncated")
        data = dec.unused_data
        if kind == "gzip":
            data = data.lstrip(b"\0")  # gzip permits zero padding
        if kind == "zlib":
            break  # zlib's standard decoder consumes one stream
    return bytes(out)


def read_limited(fh):
    data = fh.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise LimitError("artifact exceeds analysis size limit")
    return data
