"""Parse NPY headers and locate object-array pickles without numpy.load."""
from __future__ import annotations

import ast
import re
import struct

from modelgate.analysis import Unsupported


MAGIC = b'\x93NUMPY'


def dtype_info(descr, state=None):
    """Return itemsize and hasobject for literal NumPy dtype descriptors."""
    if isinstance(descr, str):
        name = descr.lstrip('<>=|')
        aliases = {'object': (8, True), 'object_': (8, True), 'bool': (1, False),
                   'bool_': (1, False), 'float64': (8, False), 'float32': (4, False),
                   'int64': (8, False), 'int32': (4, False), 'uint64': (8, False),
                   'uint32': (4, False), 'complex128': (16, False), 'complex64': (8, False)}
        if name in aliases:
            size, objects = aliases[name]
        else:
            m = re.fullmatch(r'([?bBhHiIlLqQefdgFDGOSUVmM])([0-9]*)(?:\[[^\]]*\])?', name)
            if not m:
                raise Unsupported(f'unrecognised NumPy dtype {descr!r}')
            kind, length = m.groups()
            default = {'?': 1, 'b': 1, 'B': 1, 'h': 2, 'H': 2, 'i': 4, 'I': 4,
                       'l': 8, 'L': 8, 'q': 8, 'Q': 8, 'e': 2, 'f': 4, 'd': 8,
                       'g': 16, 'F': 8, 'D': 16, 'G': 32, 'O': 8, 'm': 8, 'M': 8}
            size = int(length) if length else default.get(kind, 0)
            if kind == 'U':
                size *= 4
            objects = kind == 'O'
    elif isinstance(descr, list):
        size, objects = 0, False
        for field in descr:
            subsize, subobject = dtype_info(field[1])
            if len(field) > 2:
                shape = field[2] if isinstance(field[2], (tuple, list)) else (field[2],)
                for dim in shape:
                    subsize *= dim
            size += subsize
            objects |= subobject
    elif isinstance(descr, tuple) and len(descr) == 2:
        size, objects = dtype_info(descr[0])
        shape = descr[1] if isinstance(descr[1], (tuple, list)) else (descr[1],)
        for dim in shape:
            size *= dim
    elif isinstance(descr, dict):
        size, objects = 0, False
        formats = descr.get('formats', [])
        offsets = descr.get('offsets')
        for i, sub in enumerate(formats):
            subsize, subobject = dtype_info(sub)
            size = max(size, offsets[i] + subsize) if offsets else size + subsize
            objects |= subobject
        size = descr.get('itemsize', size)
    else:
        raise Unsupported('nonliteral NumPy dtype')
    if isinstance(state, tuple) and len(state) >= 8:
        # numpy.dtype.__setstate__: version, endian, subarray, names, fields,
        # itemsize, alignment, flags. Object fields are represented in flags.
        if isinstance(state[5], int) and state[5] >= 0:
            size = state[5]
        if isinstance(state[7], int):
            objects |= bool(state[7] & 1)
    return size, objects


def analyze_npy(data, location, budget, depth):
    from modelgate.pickle_scan import Object, read_pickle
    if not data.startswith(MAGIC) or len(data) < 10:
        raise Unsupported('invalid NPY header')
    version = tuple(data[6:8])
    if version == (1, 0):
        offset = 10
        length = struct.unpack_from('<H', data, 8)[0]
    elif version in ((2, 0), (3, 0)):
        offset = 12
        length = struct.unpack_from('<I', data, 8)[0]
    else:
        raise Unsupported(f'unsupported NPY version {version}')
    if length > 1_000_000 or offset + length > len(data):
        raise Unsupported('invalid NPY header length')
    header = ast.literal_eval(data[offset:offset + length].decode('utf-8' if version == (3, 0) else 'latin1').strip())
    if not isinstance(header, dict) or not {'descr', 'shape', 'fortran_order'} <= header.keys():
        raise Unsupported('invalid NPY header fields')
    size, objects = dtype_info(header['descr'])
    offset += length
    if objects:
        findings, value, _ = read_pickle(data[offset:], location + ':object-array', budget, depth + 1)
        return findings, value
    return [], Object('numpy.ndarray', state=header)
