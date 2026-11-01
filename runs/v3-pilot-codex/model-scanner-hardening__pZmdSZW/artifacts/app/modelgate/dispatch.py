"""Content-based, in-memory artifact and container dispatch.

Nothing is extracted to the filesystem, and no framework loader is called.
"""
from __future__ import annotations
import ast
import bz2
import io
import lzma
import os
import re
import struct
import tarfile
import zipfile
import zlib

from modelgate import keras_scan, pickle_scan, savedmodel_scan
from modelgate.report import Finding, Result

MAX_BYTES = 256 * 1024 * 1024
MAX_MEMBERS = 10000
MAX_DEPTH = 12  # Three nested archives, with compression and format wrappers.


def scan_path(path: str) -> Result:
    if os.path.isdir(path):
        return Result(path, savedmodel_scan.scan_saved_model_dir(path))
    with open(path, 'rb') as fh:
        data = fh.read(MAX_BYTES + 1)
    return Result(path, scan_bytes(data, os.path.basename(path)))


def _limited(data):
    if len(data) > MAX_BYTES:
        raise ValueError('artifact exceeds static analysis size limit')
    return data


def _decompress(data, kind):
    if kind == 'gzip':
        # decompressobj lets us bound output without allocating a zip bomb.
        parts = []
        while data:
            if not data.strip(b'\0'):
                break
            obj = zlib.decompressobj(16 + zlib.MAX_WBITS)
            part = obj.decompress(data, MAX_BYTES + 1)
            _limited(part)
            if not obj.eof:
                raise ValueError('truncated or oversized gzip member')
            parts.append(part)
            data = obj.unused_data
            if sum(map(len, parts)) > MAX_BYTES:
                raise ValueError('gzip exceeds static analysis size limit')
        return parts
    obj = bz2.BZ2Decompressor() if kind == 'bz2' else lzma.LZMADecompressor()
    result = obj.decompress(data, max_length=MAX_BYTES + 1)
    _limited(result)
    if not obj.eof:
        raise ValueError('truncated or oversized compressed artifact')
    parts = [result]
    if obj.unused_data:
        parts.extend(_decompress(obj.unused_data, kind))
    _limited(b''.join(parts))
    return parts


def npy_info(data: bytes):
    if len(data) < 10:
        raise ValueError('truncated NumPy header')
    version = data[6:8]
    if version == b'\x01\x00':
        hlen, start = struct.unpack_from('<H', data, 8)[0], 10
    elif version in (b'\x02\x00', b'\x03\x00'):
        hlen, start = struct.unpack_from('<I', data, 8)[0], 12
    else:
        raise ValueError('unknown NumPy version')
    if start + hlen > len(data):
        raise ValueError('truncated NumPy header')
    header = ast.literal_eval(data[start:start + hlen].decode('utf-8' if version[0] == 3 else 'latin1'))

    def has_object(dtype):
        if isinstance(dtype, str):
            return re.fullmatch(r'[|<>=]?O[0-9]*', dtype) is not None or dtype == 'object'
        if isinstance(dtype, list):
            return any(has_object(field[1]) for field in dtype if isinstance(field, (list, tuple)) and len(field) > 1)
        if isinstance(dtype, tuple) and dtype:
            return has_object(dtype[0])
        return False

    return start + hlen, has_object(header['descr'])


def scan_npy(data: bytes, location: str, *, context=None):
    start, hasobject = npy_info(data)
    if hasobject:
        return pickle_scan.scan_pickle(data[start:], location + ':array',
                                       mode='one', context=context)
    return []


def scan_bytes(data: bytes, location: str, depth: int = 0, *, member=False, context=None):
    _limited(data)
    if depth > MAX_DEPTH:
        return [Finding('unsupported', location, 'container nesting exceeds analysis limit')]
    if data.startswith(b'\x1f\x8b'):
        kind = 'gzip'
    elif data.startswith(b'BZh'):
        kind = 'bz2'
    elif data.startswith(b'\xfd7zXZ\x00'):
        kind = 'xz'
    else:
        kind = None
    if kind:
        parts = _decompress(data, kind)
        # Multiple gzip members are one uncompressed byte stream, including
        # concatenated pickles. If each member is a complete artifact, inspect
        # them separately as well (e.g. two NPY or zip members).
        findings = scan_bytes(b''.join(parts), location + '!' + kind, depth + 1,
                              member=member, context=context)
        if len(parts) > 1 and not findings:
            for i, part in enumerate(parts):
                findings.extend(scan_bytes(part, f'{location}!{kind}[{i}]', depth + 1,
                                           member=True, context=context))
        return findings
    # joblib also writes RFC 1950 zlib streams without a filename marker.
    if (len(data) >= 2 and data[0] & 15 == 8 and data[0] >> 4 <= 7 and
            (data[0] * 256 + data[1]) % 31 == 0):
        return scan_bytes(pickle_scan.VM.inflate(data), location + '!zlib', depth + 1,
                          member=member, context=context)
    if data.startswith(b'\x93NUMPY'):
        return scan_npy(data, location, context=context)
    if data.startswith(b'\x89HDF\r\n\x1a\n'):
        return keras_scan.scan_h5(io.BytesIO(data), location)
    # A valid leading format takes precedence over zipfile's tolerant search:
    # a tar or pickle can itself contain a complete zip byte string.
    is_tar = False
    if len(data) >= 512:
        try:
            tarfile.TarInfo.frombuf(data[:512], 'utf-8', 'surrogateescape')
            is_tar = True
        except (tarfile.HeaderError, ValueError):
            pass
    is_pickle = pickle_scan.looks_like_pickle(data)
    # A zip can have *any* prefix, including one that happens to be a complete
    # pickle. An EOCD ending at EOF disambiguates that from a zip embedded in
    # a pickle value or tar member.
    end = data.rfind(b'PK\x05\x06')
    zip_at_eof = (end >= 0 and end + 22 <= len(data) and
                  end + 22 + struct.unpack_from('<H', data, end + 20)[0] == len(data))
    if (zip_at_eof or not is_tar and not is_pickle) and zipfile.is_zipfile(io.BytesIO(data)):
        findings = []
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if len(archive.infolist()) > MAX_MEMBERS:
                raise ValueError('too many zip members')
            entries = {info.filename: info for info in archive.infolist()}
            total = sum(info.file_size for info in entries.values())
            if total > MAX_BYTES:
                raise ValueError('expanded zip exceeds analysis size limit')
            def read(name):
                return _limited(archive.read(entries[name]))
            pickles = [name for name in entries if name == 'data.pkl' or name.endswith('/data.pkl')]
            torch = data.startswith(b'PK\x03\x04') and any(
                name[:-8] + 'version' in entries for name in pickles)
            if torch:
                for name in pickles:
                    findings.extend(pickle_scan.scan_pickle(read(name), f'{location}!{name}',
                                                           mode='one', persistent=True, context=context))
                return findings
            if {'config.json', 'metadata.json', 'model.weights.h5'} <= entries.keys():
                return keras_scan.scan_json(read('config.json'), location + '!config.json')
            # SavedModels can be packaged in an archive too. Protobuf file
            # names disambiguate the otherwise untagged wire format.
            for name, info in entries.items():
                if info.is_dir():
                    continue
                payload = read(name)
                findings.extend(scan_bytes(payload, f'{location}!{name}', depth + 1,
                                           member=True, context=context))
        return findings
    # A tar's checksum, rather than its extension, identifies it. tarfile never
    # extracts: file payloads are read into memory and links are not followed.
    if is_tar:
        findings = []
        total = 0
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
            entries = {}
            for i, info in enumerate(archive):
                if i >= MAX_MEMBERS:
                    raise ValueError('too many tar members')
                entries[info.name] = info
            for name, info in entries.items():
                if info.isfile():
                    total += info.size
                    if total > MAX_BYTES:
                        raise ValueError('expanded tar exceeds analysis size limit')
                    findings.extend(scan_bytes(archive.extractfile(info).read(MAX_BYTES + 1),
                                               f'{location}!{name}', depth + 1,
                                               member=True, context=context))
        return findings
    base = location.rsplit('!', 1)[-1].rsplit('/', 1)[-1]
    if base in ('saved_model.pb', 'saved_model.pbtxt'):
        return savedmodel_scan.scan_saved_model(data, location, text=base.endswith('pbtxt'))
    # Protocol zero has no magic. All pickle opcode prefixes are accepted;
    # inert text/weights in a container are left as data if they are not pickles.
    if is_pickle:
        return pickle_scan.scan_pickle(data, location, context=context)
    if member:
        return []
    return [Finding('unsupported', location, 'unrecognized artifact format')]
