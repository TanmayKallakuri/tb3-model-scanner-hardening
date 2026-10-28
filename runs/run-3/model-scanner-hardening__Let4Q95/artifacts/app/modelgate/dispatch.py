"""Content-based dispatch and in-memory container traversal.

No container member is extracted to disk. Only metadata and serialization
syntax are parsed; framework loaders are never called.
"""
from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os
import struct
import tarfile
import zipfile
import zlib

from modelgate import keras_scan, numpy_scan, pickle_scan, savedmodel_scan
from modelgate.analysis import Budget, MAX_BYTES, Unsupported
from modelgate.pickle_scan import UNKNOWN
from modelgate.report import Finding, Result


def compression(data):
    if data.startswith(b'\x1f\x8b'):
        return 'gzip'
    if data.startswith(b'BZh'):
        return 'bz2'
    if data.startswith(b'\xfd7zXZ\x00'):
        return 'lzma'
    if len(data) > 2 and data[0] & 15 == 8 and (data[0] * 256 + data[1]) % 31 == 0:
        return 'zlib'
    return None


def decompress(data, budget, kind=None):
    kind = kind or compression(data)
    if kind == 'gzip':
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode='rb') as fh:
            out = fh.read(MAX_BYTES + 1)
        if len(out) > MAX_BYTES:
            raise Unsupported('compressed artifact exceeds analysis limit')
    elif kind in ('bz2', 'lzma'):
        pieces = []
        total = 0
        while data:
            budget.tick()
            decoder = bz2.BZ2Decompressor() if kind == 'bz2' else lzma.LZMADecompressor(memlimit=MAX_BYTES)
            chunk = decoder.decompress(data, max_length=MAX_BYTES - total + 1)
            total += len(chunk)
            if total > MAX_BYTES or not decoder.eof:
                raise Unsupported('truncated or oversized compressed artifact')
            pieces.append(chunk)
            data = decoder.unused_data
        out = b''.join(pieces)
    elif kind == 'zlib':
        decoder = zlib.decompressobj()
        out = decoder.decompress(data, MAX_BYTES + 1)
        if len(out) > MAX_BYTES or not decoder.eof:
            raise Unsupported('truncated or oversized zlib artifact')
    else:
        raise Unsupported('unknown compression format')
    budget.expand(len(out))
    return out


def is_tar(data):
    if len(data) < 512:
        return False
    header = data[:512]
    if header == bytes(512):
        return len(data) >= 1024 and data[:1024] == bytes(1024)
    try:
        claimed = int(header[148:156].strip(b' \0'), 8)
    except ValueError:
        return False
    return sum(header[:148]) + 8 * 32 + sum(header[156:]) == claimed


def looks_pickle(data):
    # Binary streams have a protocol opcode; older streams begin with an
    # opcode capable of pushing a value or MARK (not arbitrary text matches).
    if not data:
        return False
    if data[0] == 0x80:
        return len(data) >= 2 and data[1] <= 5
    if data[:1] not in b'(cVNISFJKTUXBLP]})\x88\x89':
        return False
    try:
        # A text file may start with N, I, etc. Recognise an entire first
        # stream, not just a byte that happens to be an opcode.
        for _ in pickle_scan.pickletools.genops(data):
            pass
        return True
    except Exception:
        return False


def hdf5_offset(data):
    # HDF5 permits a user block at zero or a power-of-two offset >= 512.
    offsets = [0]
    n = 512
    while n + 8 <= len(data):
        offsets.append(n)
        n *= 2
    return next((n for n in offsets if data[n:n + 8] == b'\x89HDF\r\n\x1a\n'), None)


def analyze_bytes(data, location, budget=None, depth=0, *, auxiliary=False):
    budget = budget or Budget()
    budget.nest(depth)
    if len(data) > MAX_BYTES:
        raise Unsupported('artifact exceeds analysis limit')
    stream = io.BytesIO(data)
    # zipfile's central-directory interpretation handles preambles, duplicate
    # names and the ZIP64 end record emitted by torch.save.
    if zipfile.is_zipfile(stream):
        return analyze_zip(data, location, budget, depth)
    kind = compression(data)
    if kind:
        return analyze_bytes(decompress(data, budget, kind), location + '!' + kind, budget, depth + 1)
    if is_tar(data):
        return analyze_tar(data, location, budget, depth)
    if data.startswith(numpy_scan.MAGIC):
        return numpy_scan.analyze_npy(data, location, budget, depth)
    if hdf5_offset(data) is not None:
        return keras_scan.scan_h5(stream, location), UNKNOWN
    base = location.rsplit('!', 1)[-1].replace('\\', '/').rsplit('/', 1)[-1]
    if base in ('saved_model.pb', 'saved_model.pbtxt'):
        return savedmodel_scan.scan_saved_model(data, location, text=base.endswith('pbtxt')), UNKNOWN
    if looks_pickle(data):
        findings, value, _ = pickle_scan.read_pickle(data, location, budget, depth, multiple=True)
        return findings, value
    if auxiliary:
        # Archive metadata, README files and raw tensor storage are not
        # themselves model formats. Their contents carry no capability.
        return [], UNKNOWN
    raise Unsupported('unrecognised artifact format')


def analyze_zip(data, location, budget, depth):
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # The unpacked view has one entry per name: the LAST entry wins.
        entries = {info.filename: info for info in zf.infolist() if not info.is_dir()}

        def read(name):
            info = entries[name]
            if info.file_size > MAX_BYTES:
                raise Unsupported('ZIP member exceeds analysis limit')
            budget.expand(info.file_size)
            with zf.open(info) as fh:
                content = fh.read(MAX_BYTES + 1)
            if len(content) > MAX_BYTES:
                raise Unsupported('ZIP member exceeds analysis limit')
            return content

        if {'config.json', 'metadata.json', 'model.weights.h5'} <= entries.keys():
            return keras_scan.scan_config(read('config.json'), location + '!config.json'), UNKNOWN
        pickles = [n for n in entries if n == 'data.pkl' or n.endswith('/data.pkl')]
        # A PyTorch loader only consumes its object pickle, not unrelated ZIP
        # metadata. A preamble makes this a generic registry-unpacked ZIP.
        torch = data.startswith(b'PK\x03\x04') and any(
            (n[:-8] + 'version') in entries for n in pickles)
        if torch:
            findings = []
            value = UNKNOWN
            for name in pickles:
                child, value, _ = pickle_scan.read_pickle(read(name), location + '!' + name, budget, depth + 1)
                findings.extend(child)
            return findings, value
        findings = []
        for name in entries:
            child, _ = analyze_bytes(read(name), location + '!' + name, budget, depth + 1, auxiliary=True)
            findings.extend(child)
        return findings, UNKNOWN


def analyze_tar(data, location, budget, depth):
    findings = []
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as tf:
        entries = {info.name: info for info in tf if info.isfile()}
        for name, info in entries.items():
            if info.size > MAX_BYTES:
                raise Unsupported('TAR member exceeds analysis limit')
            budget.expand(info.size)
            fh = tf.extractfile(info)
            if fh is None:
                continue
            with fh:
                content = fh.read(MAX_BYTES + 1)
            child, _ = analyze_bytes(content, location + '!' + name, budget, depth + 1, auxiliary=True)
            findings.extend(child)
    return findings, UNKNOWN


def scan_path(path: str) -> Result:
    budget = Budget()
    try:
        if os.path.isdir(path):
            return Result(path, savedmodel_scan.scan_saved_model_dir(path))
        with open(path, 'rb') as fh:
            data = fh.read(MAX_BYTES + 1)
        findings, _ = analyze_bytes(data, os.path.basename(path), budget)
        return Result(path, findings)
    except Exception as exc:
        return Result(path, [Finding('unsupported', path, f'cannot analyse artifact: {type(exc).__name__}: {exc}')])
