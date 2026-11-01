"""Content-based, in-memory traversal of the supported model containers."""
from __future__ import annotations

import ast
import bz2
import gzip
import io
import lzma
import os
import signal
import struct
import tarfile
import zipfile
import zlib

from modelgate import keras_scan, savedmodel_scan
from modelgate.context import Context, LimitError
from modelgate.report import Result

PICKLE_EXTS = ('.pkl', '.pickle', '.bin', '.joblib', '.pt', '.pth', '.ckpt')


def _timeout(signum, frame):
    raise LimitError('artifact analysis exceeded time limit')


def scan_path(path: str) -> Result:
    ctx = Context(path)
    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, 50)
    try:
        if os.path.isdir(path):
            ctx.findings.extend(savedmodel_scan.scan_saved_model_dir(path))
        else:
            with open(path, 'rb') as fh:
                data = fh.read()
            scan_bytes(data, os.path.basename(path), ctx)
    except Exception as exc:
        if not ctx.findings:
            ctx.add('unsupported', path, f'could not analyse artifact: {type(exc).__name__}: {exc}')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    return Result(path, ctx.findings)


def _expand(data, kind, ctx):
    limit = ctx.max_bytes - ctx.expanded
    if kind == 'gzip':
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as fh:
            result = fh.read(limit + 1)
    elif kind == 'bz2':
        # BZ2File/LZMAFile, like gzip, handle concatenated members.
        with bz2.BZ2File(io.BytesIO(data)) as fh:
            result = fh.read(limit + 1)
    elif kind in ('xz', 'lzma'):
        with lzma.LZMAFile(io.BytesIO(data)) as fh:
            result = fh.read(limit + 1)
    else:
        decoder = zlib.decompressobj()
        result = decoder.decompress(data, limit + 1)
    ctx.account(len(result))
    return result


def scan_bytes(data: bytes, location: str, ctx: Context, depth=0, *, force=None):
    """Inspect bytes. Unknown archive sidecars are data, not model pickles."""
    ctx.tick()
    if depth > 20:
        raise LimitError('too many nested containers or loaders')
    if force is None and location.rsplit('!', 1)[-1].lower().endswith('.joblib'):
        force = 'joblib'
    if data.startswith(b'\x1f\x8b'):
        return scan_bytes(_expand(data, 'gzip', ctx), location + '!gzip', ctx, depth + 1, force=force)
    if data.startswith(b'BZh'):
        return scan_bytes(_expand(data, 'bz2', ctx), location + '!bz2', ctx, depth + 1, force=force)
    if data.startswith(b'\xfd7zXZ\x00') or data.startswith(b'\x5d\x00\x00'):
        return scan_bytes(_expand(data, 'xz', ctx), location + '!xz', ctx, depth + 1, force=force)
    if len(data) >= 2 and data[0] == 0x78 and (data[0] * 256 + data[1]) % 31 == 0:
        return scan_bytes(_expand(data, 'zlib', ctx), location + '!zlib', ctx, depth + 1, force=force)

    # Test tar by its header/checksum. Do not extract paths or follow links.
    if len(data) >= 512:
        try:
            tf = tarfile.open(fileobj=io.BytesIO(data), mode='r:')
        except tarfile.TarError:
            tf = None
        if tf is not None:
            with tf:
                # Match extraction semantics for duplicate paths.
                entries = {member.name: member for member in tf if member.isfile()}
                for name, member in entries.items():
                    ctx.account(member.size)
                    with tf.extractfile(member) as fh:
                        scan_bytes(fh.read(), location + '!' + name, ctx, depth + 1)
            return

    # The central directory, not the first PK magic, determines zip members.
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = zf.NameToInfo  # duplicate names: last central entry wins
            if 'config.json' in members and ('model.weights.h5' in members or 'metadata.json' in members):
                ctx.findings.extend(keras_scan.scan_config(zf.read('config.json'), location + '!config.json'))
                return
            torch_members = [n for n in members if n == 'data.pkl' or n.endswith('/data.pkl')]
            is_torch = bool(torch_members and data.startswith(b'PK\x03\x04') and
                            any(n == 'version' or n.endswith('/version') for n in members))
            if is_torch:
                from modelgate.pickle_scan import scan_pickle
                for name in torch_members:
                    info = members[name]
                    ctx.account(info.file_size)
                    scan_pickle(zf.read(info), location + '!' + name, ctx, mode='torch', depth=depth + 1,
                                streams=False)
                return
            for name, info in members.items():
                if info.is_dir():
                    continue
                ctx.account(info.file_size)
                scan_bytes(zf.read(info), location + '!' + name, ctx, depth + 1)
        return

    if data.startswith(b'\x93NUMPY'):
        return scan_npy(data, location, ctx, depth)
    if data.startswith(b'\x89HDF\r\n\x1a\n') or any(
            data[n:n+8] == b'\x89HDF\r\n\x1a\n' for n in (1 << i for i in range(9, len(data).bit_length()))):
        ctx.findings.extend(keras_scan.scan_h5(data, location))
        return
    leaf = location.rsplit('!', 1)[-1].replace('\\', '/').rsplit('/', 1)[-1]
    if leaf in ('saved_model.pb', 'saved_model.pbtxt'):
        ctx.findings.extend(savedmodel_scan.scan_saved_model(data, location, leaf.endswith('pbtxt')))
        return
    from modelgate.pickle_scan import scan_pickle
    suffix = os.path.splitext(leaf)[1].lower()
    if force or suffix in PICKLE_EXTS or _looks_pickle(data):
        scan_pickle(data, location, ctx, mode=force or ('joblib' if suffix == '.joblib' else 'pickle'),
                    depth=depth)
    elif suffix in ('.npy', '.npz', '.keras', '.h5', '.hdf5', '.gz', '.xz', '.bz2', '.zip', '.tar', '.tgz'):
        ctx.add('unsupported', location, 'unrecognized or invalid model/container format')


def _looks_pickle(data):
    if not data:
        return False
    if data.startswith(b'\x80'):
        return True
    # Protocol 0/1 has no magic. Validate the first stream without running it.
    import pickletools
    try:
        for op, arg, pos in pickletools.genops(data):
            if op.name == 'STOP':
                return True
    except Exception:
        pass
    return False


def scan_npy(data, location, ctx, depth=0):
    if len(data) < 10:
        raise ValueError('truncated npy header')
    major, minor = data[6:8]
    if major == 1:
        size = struct.unpack('<H', data[8:10])[0]
        start = 10
    elif major in (2, 3):
        size = struct.unpack('<I', data[8:12])[0]
        start = 12
    else:
        raise ValueError('unknown npy version')
    header = ast.literal_eval(data[start:start + size].decode('utf-8' if major == 3 else 'latin1').strip())
    # dtype descriptors may be structured and nested, not just the string O.
    import numpy as np
    dtype = np.dtype(header['descr'])
    if dtype.hasobject:
        from modelgate.pickle_scan import scan_pickle
        return scan_pickle(data[start + size:], location + ':array', ctx, streams=False, depth=depth + 1)
