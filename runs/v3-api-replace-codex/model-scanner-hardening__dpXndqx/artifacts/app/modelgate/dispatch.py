"""Content-based, in-memory format dispatch and recursive container traversal."""
from __future__ import annotations

import ast
import io
import os
import pickletools
import struct
import tarfile
import zipfile
import zlib

from modelgate import keras_scan, pickle_scan, savedmodel_scan
from modelgate.limits import AnalysisLimit, MAX_BYTES, MAX_ITEMS
from modelgate.report import Finding, Result
from modelgate.symbolic import UNKNOWN, CapabilityFound, Instance, LoadError, World

PICKLE_EXTS = ('.pkl', '.pickle', '.joblib', '.pt', '.pth', '.ckpt', '.bin')
H5_MAGIC = b'\x89HDF\r\n\x1a\n'


def scan_path(path):
    world = World(artifact=path, stop_on_finding=True)
    world.embedded = lambda loader, data, loc, kw: _embedded(world, loader, data, loc, kw)
    if os.path.isdir(path):
        return Result(path, savedmodel_scan.scan_saved_model_dir(path))
    location = os.path.basename(path)
    try:
        with open(path, 'rb') as fh:
            head = fh.read(512)
            # Read checkpoint metadata through the central directory instead
            # of materializing potentially gigabytes of tensor storages.
            if not _tar(head) and not _compressed(head) and _zip_source(fh):
                _scan_zip(fh, location, world, 0)
                return Result(path, world.findings)
            if pickle_scan.is_legacy_prefix(head):
                fh.seek(0)
                pickle_scan.scan_legacy_stream(fh, location, world)
                return Result(path, world.findings)
            fh.seek(0, 2)
            size = fh.tell()
            offset = 0
            while not (_compressed(head) or _tar(head) or head.startswith(b'\x93NUMPY')) and offset + 8 <= size:
                fh.seek(offset)
                if fh.read(8) == H5_MAGIC:
                    world.findings.extend(keras_scan.scan_h5(path, location))
                    return Result(path, world.findings)
                offset = 512 if offset == 0 else offset * 2
            if head.startswith(b'\x93NUMPY'):
                major = head[6]
                offset = 10 if major == 1 else 12
                length = int.from_bytes(head[8:offset], 'little')
                if length > MAX_BYTES: raise AnalysisLimit('NumPy header size limit exceeded')
                fh.seek(0)
                header, _ = _npy_header(fh.read(offset + length))
                if not _object_dtype(header['descr']): return Result(path, [])
            fh.seek(0)
            data = fh.read(MAX_BYTES + 1)
        world.budget.add_bytes(len(data))
        scan_data(data, location, world)
    except CapabilityFound:
        pass
    return Result(path, world.findings)


def _embedded(world, loader, data, location, kw):
    if loader in ('torch.load', 'torch.serialization.load', 'torch.storage._load_from_bytes'):
        if kw.get('weights_only') is True:
            # No capability-bearing Python callable is allowed by this loader.
            return UNKNOWN
        if data.startswith(b'PK\x03\x04') and _zip(data):
            return _scan_zip(data, location, world, 0, torch_only=True)
        value, _ = pickle_scan.interpret(data, location, world, legacy=True, persistent=True, strict=True)
        return value
    if loader.startswith('numpy.'):
        if loader == 'numpy.lib.format.read_array' or data.startswith(b'\x93NUMPY'):
            return _npy(data, location, world, allow_pickle=kw.get('allow_pickle', False))
        if _zip(data):
            # numpy.load(npz) creates a lazy NpzFile; members are only loaded
            # when indexed. Preserve it for later __getitem__ calls.
            return Instance('numpy.NpzFile', (data, location), {'allow_pickle': kw.get('allow_pickle', False)})
        if not kw.get('allow_pickle', False): raise LoadError('NumPy pickle loading is disabled')
        return pickle_scan.interpret(data, location, world, single=True, strict=True)[0]
    if loader.startswith('joblib.') or loader == 'pandas.read_pickle':
        if loader.startswith('joblib.'): data = _decompress_if_needed(data, world)
        else:
            compression = kw.get('compression', 'infer')
            if isinstance(compression, dict): compression = compression.get('method')
            if compression not in (None, 'infer'):
                formats = {'gzip': 'gzip', 'bz2': 'bz2', 'xz': 'lzma'}
                if compression not in formats: raise AnalysisLimit('unsupported pandas compression')
                data = world.decompress(formats[compression], data)
        return pickle_scan.interpret(data, location, world, single=True, joblib=loader.startswith('joblib.'), strict=True)[0]
    world.finding('unsupported', 'unmodelled nested loader ' + loader)
    return UNKNOWN


def _zip(data):
    return len(data) >= 22 and _zip_source(io.BytesIO(data))


def _zip_source(source):
    """Respect SFX prefixes without confusing bytes *inside* a model with it.

    A pickled BytesIO may itself hold a complete torch/npz zip. zipfile also
    recognizes that inner directory when given the entire outer pickle. A
    complete enclosing pickle literal, or an NPY/HDF5 header, identifies the
    outer model. A zip following a pickle STOP is still an ordinary SFX zip.
    """
    saved = source.tell()
    try:
        source.seek(0)
        head = source.read(16)
        if head.startswith((b'\x93NUMPY', H5_MAGIC)): return False
        if not zipfile.is_zipfile(source): return False
        if head.startswith(b'PK\x03\x04') or not _looks_pickle(head): return True
        end = zipfile._EndRecData(source)
        if end and end[zipfile._ECD_ENTRIES_TOTAL] > MAX_ITEMS:
            raise AnalysisLimit('archive entry count limit exceeded')
        with zipfile.ZipFile(source) as zf:
            first = min((i.header_offset for i in zf.infolist()), default=end[zipfile._ECD_LOCATION])
        source.seek(0)
        legacy = pickle_scan.is_legacy_prefix(source.read(512))
        source.seek(0)
        embedded = False
        streams = 0
        # Bounded reads prevent a fake binary length in an SFX prefix from
        # turning format identification into an unbounded allocation.
        reader = pickle_scan.LimitedReader(source, World().budget)
        try:
            while True:
                for op, _, pos in pickletools.genops(reader):
                    if op.name in pickle_scan.SCALARS and pos <= first < reader.tell(): embedded = True
                    if op.name == 'STOP': break
                streams += 1
                if embedded or legacy and streams == 5: return False
                if reader.tell() >= first: return True
        except (ValueError, EOFError, UnicodeError, LoadError):
            return True
    finally:
        source.seek(saved)


def _compressed(data):
    if data.startswith(b'\x1f\x8b'): return 'gzip'
    if data.startswith(b'BZh'): return 'bz2'
    if data.startswith(b'\xfd7zXZ\x00'): return 'lzma'
    if len(data) > 2 and data[0] == 0x78 and (data[0] * 256 + data[1]) % 31 == 0: return 'zlib'
    return None


def _decompress_if_needed(data, world):
    fmt = _compressed(data)
    return world.decompress(fmt, data) if fmt else data


def scan_data(data, location, world, depth=0, member=False):
    world.budget.tick()
    if depth > 8: raise AnalysisLimit('container nesting limit exceeded')
    filename = location.rsplit('!', 1)[-1].rsplit('/', 1)[-1]
    if filename in ('saved_model.pb', 'saved_model.pbtxt'):
        from google.protobuf.message import DecodeError
        from google.protobuf.text_format import ParseError
        try:
            findings = savedmodel_scan.scan_saved_model(data, location, filename.endswith('pbtxt'), require_graph=True)
        except AnalysisLimit:
            raise
        except (ValueError, UnicodeError, DecodeError, ParseError):
            pass
        else:
            world.findings.extend(findings)
            return UNKNOWN
    fmt = _compressed(data)
    if fmt:
        try:
            unpacked = world.decompress(fmt, data)
        except (LoadError, OSError, EOFError, zlib.error):
            if _zip(data): return _scan_zip(data, location, world, depth)
            raise
        return scan_data(unpacked, location + '!' + fmt, world, depth + 1, member)
    # A tar can contain a complete zip checkpoint close enough to EOF for
    # is_zipfile to see its inner directory. Its valid tar header wins.
    if _tar(data): return _scan_tar(data, location, world, depth)
    if _zip(data): return _scan_zip(data, location, world, depth)
    if data.startswith(b'\x93NUMPY'):
        return _npy(data, location, world)
    if _h5(data):
        world.findings.extend(keras_scan.scan_h5(data, location))
        return UNKNOWN
    filename = location.rsplit('!', 1)[-1].rsplit('/', 1)[-1]
    if filename in ('saved_model.pb', 'saved_model.pbtxt'):
        world.findings.extend(savedmodel_scan.scan_saved_model(data, location, filename.endswith('pbtxt')))
        return UNKNOWN
    if _looks_pickle(data):
        before = len(world.findings)
        value = pickle_scan.interpret(data, location, world, joblib=True)[0]
        if member and data[:1] != b'\x80' and not location.lower().endswith(PICKLE_EXTS):
            # Auxiliary text/storage members can begin with a pickle opcode
            # by coincidence. A decoding failure alone does not turn that data
            # into a model; actual reachable capabilities are still retained.
            world.findings[before:] = [f for f in world.findings[before:] if f.kind != 'unsupported']
        return value
    # Container cards, JSON, weights and storage bytes are inert data, not
    # standalone executable models. An unrecognised top-level upload is closed.
    if not member:
        world.findings.append(Finding('unsupported', location, 'unrecognised artifact format'))
    return UNKNOWN


def _looks_pickle(data):
    if not data: return False
    if data[0] == 0x80: return True
    # Protocol 0/1, and streams that omit PROTO. Avoid treating a JSON model
    # card as a pickle just because both can start with a container character.
    if data[:1] in (b'{', b'['):
        try:
            import json
            json.loads(data)
            return False
        except (ValueError, UnicodeError): pass
    return data[:1] in b'()]}cdilostNIFLSV012PJKMUTXBG\x95\x82\x83\x84\x8a\x8b\x88\x89\x8f\x8c\x8d\x8e\x96C'


def _h5(data):
    # HDF5 permits a user block before the superblock at power-of-two offsets.
    if data.startswith(H5_MAGIC): return True
    offset = 512
    while offset + 8 <= len(data):
        if data[offset:offset + 8] == H5_MAGIC: return True
        offset *= 2
    return False


def _tar(data):
    if len(data) < 512: return False
    if data[257:262] == b'ustar': return True
    try:
        tarfile.TarInfo.frombuf(data[:512], 'utf-8', 'surrogateescape')
        return True
    except (tarfile.HeaderError, ValueError):
        return False


def _read_zip(zf, info, world):
    if info.file_size > MAX_BYTES: raise AnalysisLimit('zip member size limit exceeded')
    world.budget.add_bytes(info.file_size)
    with zf.open(info) as fh:
        data = fh.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES: raise AnalysisLimit('zip member size limit exceeded')
    return data


def _scan_zip(data, location, world, depth, torch_only=False):
    value = UNKNOWN
    source = io.BytesIO(data) if isinstance(data, bytes) else data
    source.seek(0)
    local_magic = source.read(4) == b'PK\x03\x04'
    end = zipfile._EndRecData(source)
    if end and end[zipfile._ECD_ENTRIES_TOTAL] > MAX_ITEMS:
        raise AnalysisLimit('archive entry count limit exceeded')
    with zipfile.ZipFile(source) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_ITEMS: raise AnalysisLimit('archive entry count limit exceeded')
        effective = {info.filename: info for info in infos if not info.is_dir()}
        # These two formats have their own loaders, which do not load arbitrary
        # unrelated members. Ordinary zip/npz archives traverse every member.
        for name, info in effective.items():
            if name == 'data.pkl' or name.endswith('/data.pkl'):
                prefix = name[:-len('data.pkl')]
                if local_magic and prefix + 'version' in effective:
                    return pickle_scan.interpret(_read_zip(zf, info, world), location + '!' + name,
                                                 world, single=True, persistent=True)[0]
        if torch_only: raise LoadError('invalid PyTorch zip checkpoint')
        if {'config.json', 'metadata.json', 'model.weights.h5'} <= effective.keys():
            world.findings.extend(keras_scan.scan_json(_read_zip(zf, effective['config.json'], world),
                                                       location + '!config.json'))
            return value
        for name, info in effective.items():
            scan_data(_read_zip(zf, info, world), location + '!' + name, world, depth + 1, member=True)
    return value


def _scan_tar(data, location, world, depth):
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as tf:
        effective = {}
        for info in tf:
            world.budget.tick()
            if len(effective) > MAX_ITEMS: raise AnalysisLimit('archive entry count limit exceeded')
            if info.isfile(): effective[info.name] = info
        for name, info in effective.items():
            world.budget.add_bytes(info.size)
            if info.size > MAX_BYTES: raise AnalysisLimit('tar member size limit exceeded')
            fh = tf.extractfile(info)
            if fh is not None:
                scan_data(fh.read(MAX_BYTES + 1), location + '!' + name, world, depth + 1, member=True)
    return UNKNOWN


def _object_dtype(descr):
    if isinstance(descr, str): return descr.lstrip('<>=|').startswith('O') or descr == 'object'
    if isinstance(descr, list):
        return any(isinstance(field, (tuple, list)) and len(field) >= 2 and _object_dtype(field[1]) for field in descr)
    if isinstance(descr, tuple): return _object_dtype(descr[0]) if descr else False
    if isinstance(descr, dict): return any(_object_dtype(v) for v in descr.get('formats', []))
    return False


def _npy_header(data):
    if len(data) < 10 or not data.startswith(b'\x93NUMPY'): raise LoadError('invalid npy header')
    major, minor = data[6:8]
    if major == 1:
        size = struct.unpack_from('<H', data, 8)[0]
        offset = 10
    elif major in (2, 3):
        size = struct.unpack_from('<I', data, 8)[0]
        offset = 12
    else: raise AnalysisLimit('unknown NumPy format version')
    if size > MAX_BYTES or offset + size > len(data): raise LoadError('truncated npy header')
    header = ast.literal_eval(data[offset:offset + size].decode('utf-8' if major == 3 else 'latin1').strip())
    if not isinstance(header, dict) or 'descr' not in header: raise LoadError('invalid npy header')
    return header, offset + size


def _npy(data, location, world, allow_pickle=True):
    header, offset = _npy_header(data)
    if _object_dtype(header['descr']):
        if not allow_pickle: raise LoadError('object arrays require allow_pickle')
        return pickle_scan.interpret(data[offset:], location + ':object-array', world, single=True, strict=True)[0]
    return Instance('numpy.ndarray', (header,))
