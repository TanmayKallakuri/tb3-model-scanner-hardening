"""Content-based dispatch and bounded, in-memory container traversal."""
from __future__ import annotations
import ast
import io
import os
import struct
import tarfile
import zipfile
import pickletools

from modelgate import keras_scan, pickle_scan, savedmodel_scan
from modelgate.symbolic import Runtime, Symbol, UNKNOWN, Stopped, Unsupported, MAX_BYTES
from modelgate.report import Finding, Result

HDF5 = b'\x89HDF\r\n\x1a\n'


def scan_path(path):
    runtime = Runtime()
    try:
        if os.path.isdir(path):
            runtime.findings.extend(savedmodel_scan.scan_saved_model_dir(path))
        else:
            with open(path, 'rb') as fh:
                data = fh.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise Unsupported('artifact exceeds static analysis size limit')
            runtime.artifact_files[os.path.abspath(path)] = data
            analyze_data(data, os.path.basename(path), runtime)
    except Stopped:
        pass
    except Exception as exc:
        runtime.findings.append(Finding('unsupported', runtime.location or path,
                                        f'could not analyze artifact: {type(exc).__name__}: {exc}'))
    return Result(path, runtime.findings)


def _npy(data, location, runtime, allow_pickle=True):
    if len(data) < 10:
        raise Unsupported('truncated NumPy header')
    major, minor = data[6:8]
    if major == 1:
        hlen = struct.unpack_from('<H', data, 8)[0]
        start = 10
    elif major in (2, 3):
        hlen = struct.unpack_from('<I', data, 8)[0]
        start = 12
    else:
        raise Unsupported('unknown NumPy version')
    if start + hlen > len(data):
        raise Unsupported('truncated NumPy header')
    header = ast.literal_eval(data[start:start + hlen].decode('utf-8' if major == 3 else 'latin1'))
    # dtype parses only a data description, never a Python object. Structured
    # and subarray dtypes can contain objects even when their top level is V.
    import numpy as np
    dtype = np.dtype(header['descr'])
    if dtype.hasobject:
        if not allow_pickle:
            raise Stopped('NumPy object array requires allow_pickle=True')
        value, _ = pickle_scan.run_pickle(data[start + hlen:], location + '!array', runtime, mode='one')
        return value
    count = 1
    shape = header['shape']
    if not isinstance(shape, tuple) or not all(isinstance(n, int) and n >= 0 for n in shape):
        raise Stopped('invalid NumPy array shape')
    for n in shape:
        count *= n
    if count * dtype.itemsize > len(data) - start - hlen:
        raise Stopped('truncated NumPy array data')
    return Symbol('array', 'numpy.ndarray')


def _member(data, location, runtime, depth):
    try:
        return analyze_data(data, location, runtime, depth=depth)
    except (ValueError, Unsupported) as exc:
        # Non-model metadata can share the first byte of a pickle opcode.
        # A model-looking member that cannot be analyzed remains fail-closed.
        suffix = os.path.splitext(location.rsplit('!', 1)[-1])[1].lower()
        if suffix in ('.pkl', '.pickle', '.bin', '.joblib', '.pt', '.pth', '.ckpt', '.npy', '.npz', '.keras', '.h5', '.hdf5'):
            runtime.findings.append(Finding('unsupported', location, str(exc)))
        return UNKNOWN
    except Stopped:
        # Each unpacked member has its own loader. A failing member cannot
        # hide capabilities in a different member.
        return UNKNOWN


def _zip(data, location, runtime, depth, loader):
    result = UNKNOWN
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # ZipFile's central directory is authoritative, including duplicate
        # names and prepended data. The last entry with a given name wins.
        entries = {info.filename: info for info in zf.infolist()}
        total = sum(info.file_size for info in entries.values())
        if total > MAX_BYTES or len(entries) > 100000:
            raise Unsupported('archive expansion limit exceeded')
        runtime.decoded_bytes += total
        if runtime.decoded_bytes > 2 * MAX_BYTES:
            raise Unsupported('total archive expansion limit exceeded')
        names = set(entries)
        torch_data = [n for n in names if n == 'data.pkl' or n.endswith('/data.pkl')]
        torch = False
        if len(torch_data) == 1:
            prefix = torch_data[0][:-len('data.pkl')]
            torch = prefix + 'version' in names and all(n.startswith(prefix) for n in names)
        if loader == 'torch' and not torch:
            raise Stopped('invalid torch ZIP layout')
        if loader == 'torch' and not data.startswith(b'PK\x03\x04'):
            raise Stopped('torch requires local zip header at offset zero')
        if torch and data.startswith(b'PK\x03\x04'):
            # Storage byte blobs and version metadata are not deserialized.
            for name in sorted(torch_data):
                result, _ = pickle_scan.run_pickle(zf.read(entries[name]), location + '!' + name,
                                                   runtime, mode='one', persistent=True)
            return result
        if 'config.json' in names and ('metadata.json' in names or 'model.weights.h5' in names):
            runtime.findings.extend(keras_scan.scan_json(zf.read(entries['config.json']), location + '!config.json'))
            return UNKNOWN
        saved = {n for n in names if n.rsplit('/', 1)[-1] in ('saved_model.pb', 'saved_model.pbtxt')}
        saved_roots = {n.rpartition('/')[0] for n in saved}
        for name, info in entries.items():
            runtime.tick()
            if info.is_dir():
                continue
            loc = location + '!' + name
            if name in saved:
                # Standard loader prefers binary protobuf when both exist.
                binary = name[:-3] if name.endswith('pbtxt') else name
                if name.endswith('pbtxt') and binary in names:
                    continue
                runtime.findings.extend(savedmodel_scan.scan_saved_model(zf.read(info), loc, name.endswith('pbtxt')))
                continue
            if any(name.startswith((root + '/' if root else '') + 'variables/') or
                   name.startswith((root + '/' if root else '') + 'assets/') for root in saved_roots):
                continue
            result = _member(zf.read(info), loc, runtime, depth + 1)
    return result


def _tar(data, location, runtime, depth):
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as tf:
        members = {m.name: m for m in tf.getmembers()}
        if sum(m.size for m in members.values()) > MAX_BYTES:
            raise Unsupported('tar expansion limit exceeded')
        saved = {n for n in members if n.rsplit('/', 1)[-1] in ('saved_model.pb', 'saved_model.pbtxt')}
        roots = {n.rpartition('/')[0] for n in saved}
        result = UNKNOWN
        for name, member in members.items():
            runtime.tick()
            if member.isdir():
                continue
            if not member.isfile():
                raise Unsupported('archive contains non-regular member')
            if any(name.startswith((root + '/' if root else '') + 'variables/') or
                   name.startswith((root + '/' if root else '') + 'assets/') for root in roots):
                continue
            raw = tf.extractfile(member).read(MAX_BYTES + 1)
            loc = location + '!' + name
            if name in saved:
                binary = name[:-3] if name.endswith('pbtxt') else name
                if name.endswith('pbtxt') and binary in members:
                    continue
                runtime.findings.extend(savedmodel_scan.scan_saved_model(raw, loc, name.endswith('pbtxt')))
            else:
                result = _member(raw, loc, runtime, depth + 1)
        return result


def _is_tar(data):
    if len(data) < 512:
        return False
    if data[257:262] == b'ustar':
        return True
    # Older tar headers need no magic. Validate the checksum before choosing
    # tar so an incidental string inside a pickle cannot change its format.
    try:
        tarfile.TarInfo.frombuf(data[:512], 'utf-8', 'surrogateescape')
        return True
    except tarfile.HeaderError:
        return False


def _is_zip(data, runtime):
    if not zipfile.is_zipfile(io.BytesIO(data)):
        return False
    # is_zipfile only checks for an end record. A literal containing a ZIP
    # inside a pickle is data, and its embedded end record is not a container
    # boundary. A ZIP appended after a pickle/stub is still a ZIP.
    if not data.startswith(b'PK\x03\x04'):
        fh = io.BytesIO(data)
        try:
            while fh.tell() < len(data):
                for op, arg, pos in pickletools.genops(fh):
                    runtime.tick()
            return False
        except (ValueError, IndexError, UnicodeError):
            pass
    try:
        with zipfile.ZipFile(io.BytesIO(data)):
            return True
    except (zipfile.BadZipFile, ValueError):
        return False


def analyze_data(data, location, runtime, depth=0, loader='auto'):
    runtime.tick()
    if depth > 8:
        raise Unsupported('container nesting limit exceeded')
    if not data:
        if loader != 'auto':
            raise Stopped('nested loader reached end of file')
        return UNKNOWN
    if loader == 'pandas':
        return pickle_scan.run_pickle(data, location, runtime, mode='one')[0]
    if loader == 'torch' and not data.startswith(b'PK\x03\x04'):
        # Unlike the registry's archive unpacker, torch.load does not unwrap
        # compression or recognize a zip after a prepended stub.
        return pickle_scan.run_pickle(data, location, runtime, mode='torch')[0]
    if _is_tar(data):
        return _tar(data, location, runtime, depth)
    if data.startswith(b'\x1f\x8b'):
        return analyze_data(runtime.decompress(data, 'gzip'), location + '!gzip', runtime, depth + 1, loader)
    if data.startswith(b'BZh'):
        return analyze_data(runtime.decompress(data, 'bz2'), location + '!bzip2', runtime, depth + 1, loader)
    if data.startswith(b'\xfd7zXZ\x00'):
        return analyze_data(runtime.decompress(data, 'xz'), location + '!xz', runtime, depth + 1, loader)
    # Joblib also uses a zlib stream, recognized by its RFC 1950 header.
    if len(data) > 2 and data[0] & 15 == 8 and data[0] >> 4 <= 7 and (data[0] * 256 + data[1]) % 31 == 0:
        return analyze_data(runtime.decompress(data, 'zlib'), location + '!zlib', runtime, depth + 1, loader)
    if _is_zip(data, runtime):
        return _zip(data, location, runtime, depth, loader)
    if data.startswith(b'\x93NUMPY'):
        return _npy(data, location, runtime)
    if data.startswith(HDF5) or any(data[offset:offset + 8] == HDF5 for offset in (512, 1024, 2048, 4096)):
        runtime.findings.extend(keras_scan.scan_h5(io.BytesIO(data), location))
        return UNKNOWN
    leaf = location.rsplit('!', 1)[-1].rsplit('/', 1)[-1]
    if leaf in ('saved_model.pb', 'saved_model.pbtxt'):
        runtime.findings.extend(savedmodel_scan.scan_saved_model(data, location, leaf.endswith('pbtxt')))
        return UNKNOWN
    # Archive metadata (README, JSON model cards, etc.) has no loader with
    # execution semantics. Recognize pickles by content even under odd names.
    if depth and chr(data[0]) not in pickletools.code2op:
        return UNKNOWN
    value, _ = pickle_scan.run_pickle(data, location, runtime, mode='torch' if loader == 'torch' else 'one' if loader in ('joblib', 'numpy') else 'sequence')
    return value
