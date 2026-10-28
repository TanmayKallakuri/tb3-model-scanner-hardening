"""Content-based, bounded container traversal without extracting any files."""
from __future__ import annotations

import ast
import bz2
import gzip
import io
import lzma
import os
import struct
import tarfile
import zipfile
import zlib

from modelgate import keras_scan, savedmodel_scan
from modelgate.limits import Budget, LimitExceeded, MAX_BYTES, MAX_ITEMS
from modelgate.report import Finding, Result

HDF5_MAGIC = b'\x89HDF\r\n\x1a\n'
PICKLE_EXTS = ('.pkl', '.pickle', '.bin', '.joblib', '.pt', '.pth', '.ckpt')


def unsupported(location, reason):
    return [Finding('unsupported', location, str(reason))]


def decompress(data, kind, budget, wbits=zlib.MAX_WBITS):
    """Use streaming interfaces with a hard output cap, including gzip members."""
    constructors = {'gzip': gzip.GzipFile, 'bz2': bz2.BZ2File, 'xz': lzma.LZMAFile}
    if kind == 'zlib':
        dec = zlib.decompressobj(wbits)
        result = dec.decompress(data, MAX_BYTES + 1)
        if len(result) > MAX_BYTES or dec.unconsumed_tail:
            raise LimitExceeded('decompressed data exceeds byte limit')
        if not dec.eof:
            raise ValueError('incomplete zlib stream')
    elif kind == 'xz':
        chunks, size, remaining = [], 0, data
        while remaining:
            dec = lzma.LZMADecompressor(memlimit=MAX_BYTES)
            chunk = dec.decompress(remaining, max_length=MAX_BYTES - size + 1)
            chunks.append(chunk)
            size += len(chunk)
            while not dec.eof and not dec.needs_input and size <= MAX_BYTES:
                budget.tick()
                chunk = dec.decompress(b'', max_length=MAX_BYTES - size + 1)
                chunks.append(chunk)
                size += len(chunk)
            if size > MAX_BYTES:
                raise LimitExceeded('decompressed data exceeds byte limit')
            if not dec.eof:
                raise ValueError('incomplete XZ stream')
            remaining = dec.unused_data.lstrip(b'\0')
        result = b''.join(chunks)
    else:
        source = (gzip.GzipFile(fileobj=io.BytesIO(data), mode='rb') if kind == 'gzip'
                  else constructors[kind](io.BytesIO(data), mode='rb'))
        with source as fh:
            result = fh.read(MAX_BYTES + 1)
    budget.add_bytes(len(result))
    return result


def gzip_members(data, budget):
    parts = []
    remaining = data
    while remaining:
        budget.tick()
        dec = zlib.decompressobj(31)
        part = dec.decompress(remaining, MAX_BYTES + 1)
        if len(part) > MAX_BYTES or dec.unconsumed_tail:
            raise LimitExceeded('decompressed data exceeds byte limit')
        if not dec.eof:
            raise ValueError('incomplete gzip member')
        budget.add_bytes(len(part))
        parts.append(part)
        remaining = dec.unused_data.lstrip(b'\0')
        if len(parts) > MAX_ITEMS:
            raise LimitExceeded('too many gzip members')
    return parts


def numpy_payload(data):
    """Return (has_objects, offset), parsing only the literal NPY header."""
    if data[:6] != b'\x93NUMPY' or len(data) < 10:
        raise ValueError('invalid NPY header')
    version = data[6:8]
    if version == b'\x01\x00':
        n, offset = struct.unpack_from('<H', data, 8)[0], 10
    elif version in (b'\x02\x00', b'\x03\x00'):
        n, offset = struct.unpack_from('<I', data, 8)[0], 12
    else:
        raise ValueError('unknown NPY version')
    if n > 4 * 1024 * 1024 or offset + n > len(data):
        raise ValueError('invalid NPY header size')
    header = ast.literal_eval(data[offset:offset+n].decode('utf-8' if version[0] == 3 else 'latin1'))
    descr = header['descr']

    def has_object(d):
        if isinstance(d, str):
            return d.lstrip('<>=|').startswith('O') or d == 'object'
        if isinstance(d, list):
            return any(isinstance(f, (list, tuple)) and len(f) >= 2 and has_object(f[1]) for f in d)
        if isinstance(d, tuple) and d:
            return has_object(d[0])
        return False

    return has_object(descr), offset + n


def is_tar(data):
    if len(data) < 512:
        return False
    try:
        expected = int(data[148:156].rstrip(b'\0 ').lstrip() or b'0', 8)
        return expected > 0 and sum(data[:148]) + 256 + sum(data[156:512]) == expected
    except ValueError:
        return False


def is_pickle(data):
    if not data:
        return False
    # Legal starting opcodes; attempting a walk here would interpret streams
    # twice and is unnecessary. Unknown members are not treated as code.
    if data[0] == 0x80:
        return len(data) > 1 and data[1] <= 5
    return data[:1] in b'(cdil}])NIFLVSXKJM\x88\x89\x8a\x8b\x8cCUBT\x8d\x8e\x96'


def member_pickle(data, budget):
    """Do not mistake numeric storage bytes or text metadata for a pickle."""
    if not is_pickle(data):
        return False
    import pickletools
    try:
        for op, arg, pos in pickletools.genops(data):
            budget.tick()
            if op.name == 'STOP':
                return True
    except (ValueError, UnicodeError, IndexError):
        pass
    return False


def zip_inside_pickle_literal(data, budget):
    """Distinguish a ZIP envelope from ZIP bytes stored inside a pickle.

    zipfile accepts arbitrary prefixes *and* suffixes. Consequently a pickle
    containing a BytesIO/bytes ZIP value also passes is_zipfile. Its length
    delimited literal owns those bytes; only a nested loader may inspect them.
    A ZIP after a pickle STOP is not a literal and remains an archive.
    """
    if not is_pickle(data):
        return False
    import pickletools
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            start = min((i.header_offset for i in zf.infolist()), default=zf.start_dir)
        end_record = data.rfind(b'PK\x05\x06')
        end = end_record + 22 + struct.unpack_from('<H', data, end_record + 20)[0]
        fh = io.BytesIO(data)
        while fh.tell() < len(data):
            for op, arg, pos in pickletools.genops(fh):
                budget.tick()
                if op.name in ('BINBYTES', 'SHORT_BINBYTES', 'BINBYTES8', 'BYTEARRAY8',
                               'BINSTRING', 'SHORT_BINSTRING', 'BINUNICODE', 'BINUNICODE8',
                               'SHORT_BINUNICODE') and pos < start and fh.tell() >= end:
                    return True
                if pos > start:
                    return False
    except (ValueError, UnicodeError, IndexError, struct.error, zipfile.BadZipFile):
        pass
    return False


def scan_data(data, location, budget=None, depth=0, member=False):
    budget = budget or Budget()
    budget.tick()
    if depth > 12:
        return unsupported(location, 'container nesting exceeds analysis limit')
    if len(data) > MAX_BYTES:
        return unsupported(location, 'artifact exceeds analysis byte limit')
    try:
        compression = None
        if data.startswith(b'\x1f\x8b'):
            compression = 'gzip'
        elif data.startswith(b'BZh'):
            compression = 'bz2'
        elif data.startswith(b'\xfd7zXZ\x00'):
            compression = 'xz'
        elif len(data) >= 2 and data[0] == 0x78 and (data[0] * 256 + data[1]) % 31 == 0:
            compression = 'zlib'  # joblib's default compression
        if compression:
            if compression == 'gzip':
                parts = gzip_members(data, budget)
                # Gzip permits members to be chunks of one logical stream,
                # or complete artifacts. Preserve concatenated pickle/torch
                # semantics and inspect independently recognizable members.
                findings = scan_data(b''.join(parts), location + '!gzip', budget, depth + 1)
                if findings or len(parts) < 2:
                    return findings
                for i, part in enumerate(parts):
                    complete = (member_pickle(part, budget) or part.startswith((b'\x93NUMPY', HDF5_MAGIC,
                                b'\x1f\x8b', b'BZh', b'\xfd7zXZ\x00')) or is_tar(part)
                                or zipfile.is_zipfile(io.BytesIO(part)))
                    if complete:
                        findings.extend(scan_data(part, f'{location}!gzip[{i}]', budget, depth + 1))
                        if findings:
                            return findings
                return findings
            return scan_data(decompress(data, compression, budget), location + '!' + compression,
                             budget, depth + 1)
        if data.startswith(b'\x93NUMPY'):
            objects, offset = numpy_payload(data)
            if not objects:
                return []
            from modelgate.pickle_scan import scan_pickle
            return scan_pickle(data[offset:], location + ':array', budget=budget, sequence=False)
        if data.startswith(HDF5_MAGIC) or any(data[i:i+8] == HDF5_MAGIC for i in (512, 1024, 2048, 4096, 8192)):
            return keras_scan.scan_h5(data, location)
        # The central directory decides membership and duplicate names. In
        # particular, a ZIP need not start with a local header (but torch does).
        if zipfile.is_zipfile(io.BytesIO(data)) and not zip_inside_pickle_literal(data, budget):
            findings = []
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                entries = {info.filename: info for info in zf.infolist()}
                if len(entries) > MAX_ITEMS:
                    raise LimitExceeded('too many ZIP members')
                torch_names = [n for n in entries if n == 'data.pkl' or n.endswith('/data.pkl')]
                torch_names = [n for n in torch_names if n[:-8] + 'version' in entries]
                if data.startswith(b'PK\x03\x04') and torch_names:
                    from modelgate.pickle_scan import scan_pickle
                    for name in torch_names:
                        info = entries[name]
                        budget.add_bytes(info.file_size)
                        findings.extend(scan_pickle(zf.read(info), location + '!' + name,
                                                    budget=budget, sequence=False, persistent=True))
                    return findings
                if 'config.json' in entries and ('metadata.json' in entries or 'model.weights.h5' in entries):
                    budget.add_bytes(entries['config.json'].file_size)
                    return keras_scan.scan_config(zf.read(entries['config.json']), location + '!config.json')
                for name, info in entries.items():
                    if info.is_dir():
                        continue
                    budget.add_bytes(info.file_size)
                    if name in torch_names:
                        from modelgate.pickle_scan import scan_pickle
                        findings.extend(scan_pickle(zf.read(info), location + '!' + name,
                                                    budget=budget, sequence=False, persistent=True))
                    elif not any(name.startswith(n[:-8] + 'data/') or name == n[:-8] + 'version'
                                 or name == n[:-8] + 'byteorder' for n in torch_names):
                        findings.extend(scan_data(zf.read(info), location + '!' + name, budget, depth + 1, True))
            return findings
        if is_tar(data):
            findings = []
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as tf:
                # Extraction semantics: the last member at a path replaces its
                # predecessors. Read member data directly; never follow links.
                entries = {}
                for info in tf:
                    budget.tick()
                    if len(entries) > MAX_ITEMS:
                        raise LimitExceeded('too many TAR members')
                    entries[info.name] = info
                torch_names = [n for n in entries if (n == 'data.pkl' or n.endswith('/data.pkl'))
                               and n[:-8] + 'version' in entries]
                for name, info in entries.items():
                    if not info.isfile():
                        continue
                    budget.add_bytes(info.size)
                    fh = tf.extractfile(info)
                    if fh is not None:
                        if name in torch_names:
                            from modelgate.pickle_scan import scan_pickle
                            findings.extend(scan_pickle(fh.read(), location + '!' + name,
                                                        budget=budget, sequence=False, persistent=True))
                        elif not any(name.startswith(n[:-8] + 'data/') or name == n[:-8] + 'version'
                                     or name == n[:-8] + 'byteorder' for n in torch_names):
                            findings.extend(scan_data(fh.read(), location + '!' + name, budget, depth + 1, True))
            return findings
        name = location.rsplit('!', 1)[-1].replace('\\', '/').rsplit('/', 1)[-1]
        if name in ('saved_model.pb', 'saved_model.pbtxt'):
            return savedmodel_scan.scan_saved_model(data, location, name.endswith('pbtxt'))
        if name.lower().endswith(PICKLE_EXTS) or (member_pickle(data, budget) if member else is_pickle(data)):
            from modelgate.pickle_scan import scan_pickle
            return scan_pickle(data, location, budget=budget)
        # Archives may contain weights, metadata, vocabularies or other data
        # that has no loader. A standalone unknown format must fail closed.
        return [] if member else unsupported(location, 'unrecognized artifact format')
    except Exception as exc:
        return unsupported(location, f'{type(exc).__name__}: {exc}')


def scan_path(path):
    if os.path.isdir(path):
        return Result(path, savedmodel_scan.scan_saved_model_dir(path))
    with open(path, 'rb') as fh:
        data = fh.read(MAX_BYTES + 1)
    budget = Budget()
    budget.add_bytes(len(data))
    return Result(path, scan_data(data, os.path.basename(path), budget))
