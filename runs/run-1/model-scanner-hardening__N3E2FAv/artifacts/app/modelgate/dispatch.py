"""Content-based, in-memory dispatch for model artifacts and their containers."""
from __future__ import annotations

import ast
import bz2
import gzip
import io
import lzma
import os
import pickletools
import struct
import tarfile
import zipfile
import zlib

from modelgate import keras_scan, pickle_scan, savedmodel_scan
from modelgate.report import Finding, Result

MAX_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 10000
MAX_DEPTH = 24
HDF_MAGIC = b'\x89HDF\r\n\x1a\n'
MODEL_EXTS = {'.pkl', '.pickle', '.joblib', '.bin', '.pt', '.pth', '.ckpt', '.npy',
              '.npz', '.keras', '.h5', '.hdf5', '.gz', '.bz2', '.xz', '.zip', '.tar', '.tgz'}


def unsupported(location, reason):
    return [Finding('unsupported', location, reason)]


def read_limited(fh):
    data = fh.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError('artifact exceeds analysis byte budget')
    return data


def decompress(data, kind):
    if kind == 'zlib':
        obj = zlib.decompressobj()
        value = obj.decompress(data, MAX_BYTES + 1)
        if len(value) > MAX_BYTES or obj.unconsumed_tail:
            raise ValueError('decompression byte budget exceeded')
        if not obj.eof:
            raise ValueError('truncated zlib stream')
        return value
    cls = {'bz2': bz2.BZ2File, 'lzma': lzma.LZMAFile}
    source = io.BytesIO(data)
    fh = gzip.GzipFile(fileobj=source, mode='rb') if kind == 'gzip' else cls[kind](source, 'rb')
    with fh:
        return read_limited(fh)


def scan_path(path: str) -> Result:
    if os.path.isdir(path):
        return Result(path, savedmodel_scan.scan_saved_model_dir(path))
    with open(path, 'rb') as fh:
        data = read_limited(fh)
    return Result(path, scan_bytes(data, os.path.basename(path)))


def _hdf(data):
    # HDF5 permits a user block before its signature.
    if data.startswith(HDF_MAGIC):
        return True
    pos = 512
    while pos + 8 <= len(data):
        if data[pos:pos + 8] == HDF_MAGIC:
            return True
        pos *= 2
    return False


def _tar(data):
    if len(data) < 512:
        return False
    if data[257:262] == b'ustar':
        return True
    try:
        tarfile.TarInfo.frombuf(data[:512], 'utf-8', 'surrogateescape')
        return True
    except (tarfile.HeaderError, ValueError):
        return False


def _pickle(data):
    if not data:
        return False
    if data[0] == 0x80:
        return True
    # A complete protocol 0/1 stream has no magic. This does not interpret
    # strings as globals; it merely establishes the opcode framing.
    try:
        for op, arg, _ in pickletools.genops(data):
            if op.name == 'GLOBAL' and arg in ('joblib.numpy_pickle NumpyArrayWrapper', 'joblib.numpy_pickle NDArrayWrapper'):
                return True
            if op.name == 'STOP':
                return True
    except (ValueError, UnicodeError, OverflowError):
        pass
    return False


def scan_bytes(data: bytes, location: str, *, depth=0, member=False) -> list[Finding]:
    try:
        return _scan_bytes(data, location, depth=depth, member=member)
    except Exception as exc:
        return unsupported(location, f'cannot analyse artifact: {type(exc).__name__}: {str(exc)[:500]}')


def _scan_bytes(data, location, *, depth, member):
    if depth > MAX_DEPTH:
        return unsupported(location, 'container/loader nesting limit exceeded')
    if len(data) > MAX_BYTES:
        return unsupported(location, 'artifact exceeds analysis byte budget')
    if data.startswith(b'\x1f\x8b'):
        return scan_bytes(decompress(data, 'gzip'), location + '!gzip', depth=depth + 1)
    if data.startswith(b'BZh'):
        return scan_bytes(decompress(data, 'bz2'), location + '!bzip2', depth=depth + 1)
    if data.startswith(b'\xfd7zXZ\x00'):
        return scan_bytes(decompress(data, 'lzma'), location + '!xz', depth=depth + 1)
    # joblib also commonly uses zlib without a filename extension.
    if len(data) >= 2 and data[0] & 15 == 8 and (data[0] * 256 + data[1]) % 31 == 0:
        return scan_bytes(decompress(data, 'zlib'), location + '!zlib', depth=depth + 1)
    if data.startswith(b'\x93NUMPY'):
        return scan_npy(data, location, depth=depth)
    if _hdf(data):
        return keras_scan.scan_h5(io.BytesIO(data), location)
    if zipfile.is_zipfile(io.BytesIO(data)):
        # A zip embedded as a pickle byte string is data until its reducer
        # loads it. genops establishes whether the outer pickle encompasses
        # the zip, rather than merely being a self-extractor prefix.
        is_outer_pickle = False
        if data[:1] == b'\x80' or data[:1] in (b'c', b'(', b'd', b'}', b'l', b']'):
            try:
                for op, _, pos in pickletools.genops(data):
                    if op.name == 'STOP':
                        start = data.find(b'PK\x03\x04')
                        is_outer_pickle = start < 0 or pos > start
            except (ValueError, UnicodeError):
                pass
        if not is_outer_pickle:
            return scan_zip(data, location, depth=depth)
    if _tar(data):
        return scan_tar(data, location, depth=depth)
    basename = location.replace('\\', '/').rsplit('/', 1)[-1].rsplit('!', 1)[-1]
    if basename in ('saved_model.pb', 'saved_model.pbtxt'):
        return savedmodel_scan.scan_saved_model(data, location, text=basename.endswith('pbtxt'))
    if _pickle(data):
        return pickle_scan.scan_pickle(data, location, depth=depth)
    # Non-model archive members (cards, metadata, tensor storages) are data.
    # A claimed model that cannot be parsed is not silently accepted.
    ext = os.path.splitext(basename)[1].lower()
    if member and ext not in MODEL_EXTS:
        return []
    return unsupported(location, 'unrecognized or malformed artifact format')


def scan_npy(data, location, *, depth=0):
    if len(data) < 10:
        raise ValueError('truncated NumPy header')
    version = tuple(data[6:8])
    if version == (1, 0):
        n = struct.unpack_from('<H', data, 8)[0]
        offset = 10
    elif version in ((2, 0), (3, 0)):
        n = struct.unpack_from('<I', data, 8)[0]
        offset = 12
    else:
        raise ValueError(f'unsupported NumPy version {version}')
    if n > 1024 * 1024 or offset + n > len(data):
        raise ValueError('invalid NumPy header length')
    header = ast.literal_eval(data[offset:offset + n].decode('utf-8' if version == (3, 0) else 'latin1'))
    if not isinstance(header, dict) or set(header) != {'descr', 'fortran_order', 'shape'}:
        raise ValueError('invalid NumPy header')
    import numpy as np
    dtype = np.dtype(header['descr'])
    if dtype.hasobject:
        return pickle_scan.scan_pickle(data[offset + n:], location + '!array.pkl', depth=depth + 1, first_only=True)
    # Numeric/structured numeric payloads are never deserialized as Python.
    return []


def scan_zip(data, location, *, depth=0):
    findings = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ValueError('archive member limit exceeded')
        # zipfile resolves duplicate names using the last central-directory
        # entry. Do not scan a shadowed member or trust local-header names.
        entries = {info.filename: info for info in infos if not info.is_dir()}
        if sum(info.file_size for info in entries.values()) > MAX_BYTES:
            raise ValueError('archive expansion byte budget exceeded')
        if 'config.json' in entries and ('metadata.json' in entries or 'model.weights.h5' in entries):
            return keras_scan.scan_config(zf.read(entries['config.json']), location + '!config.json')
        torch_pickles = [n for n in entries if n == 'data.pkl' or n.endswith('/data.pkl')]
        torch = data.startswith(b'PK\x03\x04') and any(
            n[:-len('data.pkl')] + 'version' in entries for n in torch_pickles)
        if torch:
            for name in torch_pickles:
                findings.extend(pickle_scan.scan_pickle(zf.read(entries[name]), f'{location}!{name}', depth=depth + 1, first_only=True))
            return findings
        for name, info in entries.items():
            # TensorFlow binary wins over its optional text counterpart.
            if name.endswith('saved_model.pbtxt') and name[:-5] + 'pb' in entries:
                continue
            findings.extend(scan_bytes(zf.read(info), f'{location}!{name}', depth=depth + 1, member=True))
    return findings


def scan_tar(data, location, *, depth=0):
    findings = []
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as tf:
        entries = {}
        total = 0
        for i, entry in enumerate(tf):
            if i >= MAX_MEMBERS:
                raise ValueError('archive member limit exceeded')
            if entry.isfile():
                entries[entry.name] = entry
                total += entry.size
                if total > MAX_BYTES:
                    raise ValueError('archive expansion byte budget exceeded')
        for name, entry in entries.items():
            if name.endswith('saved_model.pbtxt') and name[:-5] + 'pb' in entries:
                continue
            fh = tf.extractfile(entry)
            if fh is not None:
                findings.extend(scan_bytes(read_limited(fh), f'{location}!{name}', depth=depth + 1, member=True))
    return findings
