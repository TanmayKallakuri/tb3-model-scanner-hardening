"""Content-based, in-memory format dispatch. Containers are never extracted."""
from __future__ import annotations

import ast
import bz2
import io
import json
import lzma
import os
import struct
import tarfile
import zipfile
import zlib

from modelgate import keras_scan, savedmodel_scan
from modelgate.abstract import Context, Value, UNKNOWN, LoadError, LimitError, Detected
from modelgate.report import Finding, Result

MAX_DATA = 256 * 1024 * 1024
PICKLE_INITIAL = set(b'\x80(ciltd}])NIFJMKLSVUXTGCB\x8c\x8d\x8e\x96\x88\x89\x8f')


def decompress(data, kind, ctx, args=(), kw=None):
    """Bound output size, including concatenated gzip/bzip2/xz members."""
    kw = kw or {}
    chunks, total = [], 0
    rest = data
    while rest:
        ctx.tick()
        if kind in ('zlib', 'gzip'):
            wbits = args[0] if args else kw.get('wbits', 31 if kind == 'gzip' else 15)
            decoder = zlib.decompressobj(wbits)
            out = decoder.decompress(rest, MAX_DATA - total + 1)
            if decoder.unconsumed_tail:
                raise LimitError('decompressed input exceeds size limit')
        elif kind == 'bz2':
            decoder = bz2.BZ2Decompressor()
            out = decoder.decompress(rest, max_length=MAX_DATA - total + 1)
        elif kind == 'lzma':
            decoder = lzma.LZMADecompressor(memlimit=256 * 1024 * 1024)
            out = decoder.decompress(rest, max_length=MAX_DATA - total + 1)
        else:
            raise LimitError('unknown compression method')
        total += len(out)
        if total > MAX_DATA:
            raise LimitError('compressed input is too large')
        if not decoder.eof:
            raise LoadError('compressed input is truncated')
        chunks.append(out)
        rest = decoder.unused_data
        if kind == 'zlib':
            break
        if kind == 'gzip':
            rest = rest.lstrip(b'\x00')
    ctx.allocate(total)
    return b''.join(chunks)


def compression(data):
    if data.startswith(b'\x1f\x8b'):
        return 'gzip'
    if data.startswith(b'BZh'):
        return 'bz2'
    if data.startswith(b'\xfd7zXZ\x00') or data.startswith(b']\x00\x00'):
        return 'lzma'
    if len(data) >= 2 and data[0] & 15 == 8 and ((data[0] << 8) + data[1]) % 31 == 0:
        return 'zlib'
    return None


def dtype_info(descr):
    """Item size and hasobject for NPY/joblib descriptors, without NumPy."""
    if isinstance(descr, str):
        aliases = {'object': 'O', 'bool': '?1', 'bool_': '?1', 'int8': 'i1', 'int16': 'i2',
                   'int32': 'i4', 'int64': 'i8', 'uint8': 'u1', 'uint16': 'u2', 'uint32': 'u4',
                   'uint64': 'u8', 'float16': 'f2', 'float32': 'f4', 'float64': 'f8',
                   'complex64': 'c8', 'complex128': 'c16'}
        code = aliases.get(descr, descr).lstrip('<>=|')
        if code.startswith('O'):
            return 8, True
        if not code:
            raise LoadError('empty NumPy dtype')
        count = int(code[1:].split('[', 1)[0] or (1 if code[0] in '?bB' else 0))
        return count * (4 if code[0] == 'U' else 1), False
    if isinstance(descr, (list, tuple)):
        size, objects = 0, False
        for field in descr:
            if not isinstance(field, tuple) or len(field) < 2:
                raise LoadError('invalid structured dtype')
            n, obj = dtype_info(field[1])
            if len(field) > 2:
                shape = field[2]
                for dim in ((shape,) if isinstance(shape, int) else shape):
                    n *= dim
            size += n
            objects |= obj
        return size, objects
    raise LimitError('unknown NumPy dtype')


def analyze_npy(data, location, ctx, allow_pickle=True):
    if len(data) < 10:
        raise LoadError('truncated NPY header')
    version = tuple(data[6:8])
    width = 2 if version == (1, 0) else 4
    start = 8 + width
    size = int.from_bytes(data[8:start], 'little')
    if start + size > len(data):
        raise LoadError('truncated NPY header')
    header = ast.literal_eval(data[start:start + size].decode('utf-8' if version == (3, 0) else 'latin1'))
    _, objects = dtype_info(header['descr'])
    if objects:
        if not allow_pickle:
            raise LoadError('object arrays require allow_pickle=True')
        from modelgate.pickle_scan import analyze_pickle
        return analyze_pickle(data[start + size:], location + '!object-array', ctx, single=True)
    return Value('instance', 'numpy.ndarray')


def is_tar(data):
    if len(data) < 512:
        return False
    try:
        tarfile.TarInfo.frombuf(data[:512], 'utf-8', 'surrogateescape')
        return True
    except (ValueError, tarfile.TarError):
        return False


def unsafe_member(name):
    # The registry unpacks archives. Traversal would write outside the artifact.
    import posixpath
    path = posixpath.normpath(name.replace('\\', '/'))
    return path.startswith('/') or path == '..' or path.startswith('../') or (
        len(path) > 1 and path[1] == ':')


def is_zip(data):
    # A ZIP embedded as a byte literal in a pickle is not the outer format.
    # In-scope archives end with their EOCD and its declared comment; arbitrary
    # bytes *before* the local headers are fine. ZipFile still interprets the
    # central directory (and ZIP64 end records), not our signature check.
    end = data.rfind(b'PK\x05\x06', max(0, len(data) - 65557))
    if end < 0 or end + 22 > len(data):
        return False
    comment_size = int.from_bytes(data[end + 20:end + 22], 'little')
    return end + 22 + comment_size == len(data) and zipfile.is_zipfile(io.BytesIO(data))


def analyze_zip(data, location, ctx, depth, loader):
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # ZipFile's name map gives the same last-entry-wins behavior as its
        # standard reader. Never walk local headers looking for hidden pickles.
        members = {info.filename: info for info in zf.infolist()}
        keys = set(members)
        keras = {'config.json', 'metadata.json', 'model.weights.h5'} <= keys
        torch_members = [name for name in keys if name == 'data.pkl' or name.endswith('/data.pkl')]
        torch = bool(torch_members) and data.startswith(b'PK\x03\x04') and (
            loader in ('torch.load', 'torch.storage._load_from_bytes') or
            any((n[:-8] + 'version') in keys for n in torch_members))
        def read(name):
            info = members[name]
            ctx.allocate(info.file_size)
            return zf.read(info)
        if keras:
            findings = keras_scan.scan_config(read('config.json'), location + '!config.json')
            if findings:
                raise Detected(findings[0])
            return UNKNOWN
        if torch:
            from modelgate.pickle_scan import analyze_pickle
            for name in torch_members:
                return analyze_pickle(read(name), location + '!' + name, ctx, single=True, persistent=True)
        result = UNKNOWN
        for name, info in members.items():
            ctx.tick()
            if unsafe_member(name):
                raise Detected(Finding('file-access', location + '!' + name, 'archive member escapes artifact'))
            if info.is_dir():
                continue
            # Unix symlink targets are stored as the member body.
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                target = read(name).decode('utf-8', 'surrogateescape')
                if unsafe_member(os.path.join(os.path.dirname(name), target)):
                    raise Detected(Finding('file-access', location + '!' + name, 'archive link escapes artifact'))
                continue
            try:
                result = analyze_bytes(read(name), location + '!' + name, ctx, depth + 1, auxiliary=True)
            except LoadError:
                continue
        return result


def analyze_tar(data, location, ctx, depth):
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as tf:
        members = {}
        for member in tf:
            ctx.tick()
            members[member.name] = member
        result = UNKNOWN
        for name, member in members.items():
            ctx.tick()
            if unsafe_member(name):
                raise Detected(Finding('file-access', location + '!' + name, 'archive member escapes artifact'))
            if member.issym() or member.islnk():
                target = os.path.join(os.path.dirname(name), member.linkname) if member.issym() else member.linkname
                if unsafe_member(target):
                    raise Detected(Finding('file-access', location + '!' + name, 'archive link escapes artifact'))
                continue
            if not member.isfile():
                continue
            ctx.allocate(member.size)
            stream = tf.extractfile(member)
            try:
                result = analyze_bytes(stream.read(), location + '!' + name, ctx, depth + 1, auxiliary=True)
            except LoadError:
                continue
        return result


def analyze_bytes(data, location, ctx, depth=0, *, loader=None, auxiliary=False, nested=False):
    ctx.tick()
    if depth > 12:
        raise LimitError('container nesting limit exceeded')
    if not isinstance(data, bytes):
        data = bytes(data)
    if len(data) > MAX_DATA:
        raise LimitError('artifact exceeds size limit')
    comp = compression(data)
    if comp:
        return analyze_bytes(decompress(data, comp, ctx), location + '!' + comp,
                             ctx, depth + 1, loader=loader, auxiliary=auxiliary, nested=nested)
    if data.startswith(b'\x93NUMPY'):
        return analyze_npy(data, location, ctx)
    if data.startswith(b'\x89HDF\r\n\x1a\n'):
        findings = keras_scan.scan_h5(io.BytesIO(data), location)
        if findings:
            raise Detected(findings[0])
        return UNKNOWN
    if is_tar(data):
        return analyze_tar(data, location, ctx, depth)
    # ZIP is deliberately recognized from the central directory, including SFX
    # prefixes and torch's zip64 end record with ordinary 32-bit member sizes.
    if is_zip(data):
        return analyze_zip(data, location, ctx, depth, loader)
    leaf = location.split('!')[-1].replace('\\', '/').rsplit('/', 1)[-1]
    if leaf in ('saved_model.pb', 'saved_model.pbtxt'):
        findings = savedmodel_scan.scan_saved_model(data, location, text=leaf.endswith('pbtxt'))
        if findings:
            raise Detected(findings[0])
        return UNKNOWN
    if data and data[0] in PICKLE_INITIAL:
        from modelgate.pickle_scan import analyze_pickle
        return analyze_pickle(data, location, ctx,
                              single=loader in ('numpy.load', 'numpy.lib.format.read_array', 'pandas.read_pickle', 'pandas.io.pickle.read_pickle', 'joblib.load'),
                              legacy=loader in ('torch.load', 'torch.storage._load_from_bytes'))
    if auxiliary:
        # Registries also unpack cards, metadata, weights and storage shards.
        # Opaque members do not become executable by containing suspicious text.
        return UNKNOWN
    raise LimitError('unrecognized artifact format')


def analyze_loaded(data, location, ctx, loader, args=(), kw=None):
    """Embedded data follows the selected loader, rather than archive dispatch."""
    from modelgate.pickle_scan import analyze_pickle
    kw = kw or {}
    if loader in ('torch.load', 'torch.storage._load_from_bytes'):
        if data.startswith(b'PK\x03\x04'):
            return analyze_zip(data, location, ctx, 0, loader)
        return analyze_pickle(data, location, ctx, legacy=True)
    if loader in ('numpy.load', 'numpy.lib.format.read_array'):
        option_index = 1 if loader == 'numpy.load' else 0
        allowed = kw.get('allow_pickle', args[option_index] if len(args) > option_index else False)
        if data.startswith(b'\x93NUMPY'):
            return analyze_npy(data, location, ctx, allow_pickle=allowed)
        if loader.endswith('read_array'):
            raise LoadError('read_array requires a NumPy header')
        if data.startswith((b'PK\x03\x04', b'PK\x05\x06')):
            return Value('npz', 'numpy.lib.npyio.NpzFile', (),
                         {'data': data, 'allow_pickle': allowed, 'location': location})
        if not allowed:
            raise LoadError('NumPy pickle loading is disabled')
        return analyze_pickle(data, location, ctx, single=True)
    if loader == 'joblib.load':
        comp = compression(data)
        if comp:
            data = decompress(data, comp, ctx)
        return analyze_pickle(data, location, ctx, single=True)
    if loader in ('pandas.read_pickle', 'pandas.io.pickle.read_pickle'):
        comp = kw.get('compression', args[0] if args else None)
        if isinstance(comp, dict):
            comp = comp.get('method')
        if comp and comp != 'infer':
            data = decompress(data, {'xz': 'lzma', 'gz': 'gzip'}.get(comp, comp), ctx)
        return analyze_pickle(data, location, ctx, single=True)
    return analyze_pickle(data, location, ctx, single=True)


def scan_path(path: str) -> Result:
    try:
        if os.path.isdir(path):
            return Result(path, savedmodel_scan.scan_saved_model_dir(path))
        with open(path, 'rb') as fh:
            data = fh.read(MAX_DATA + 1)
        analyze_bytes(data, os.path.basename(path), Context())
        return Result(path)
    except Detected as exc:
        return Result(path, [exc.finding])
    except LoadError:
        return Result(path)
    except Exception as exc:
        return Result(path, [Finding('unsupported', path, f'cannot analyze artifact: {type(exc).__name__}: {exc}')])
