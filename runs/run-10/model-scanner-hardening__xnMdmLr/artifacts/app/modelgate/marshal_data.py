"""Small, non-executing reader for marshal values used as pickle arguments.

Code objects are opaque symbols. No Python code object is ever constructed.
The wire type of the outer value matters: marshal.loads of a number or a dict
is data, and cannot be used as a FunctionType code argument.
"""
import struct
from modelgate.abstract import Value, LoadError, LimitError


def read_marshaled(data, context):
    if not isinstance(data, (bytes, bytearray)):
        raise LoadError('marshal input is not bytes')
    pos, refs = 0, []

    def read(n):
        nonlocal pos
        if n < 0 or pos + n > len(data):
            raise LoadError('truncated marshal data')
        result = data[pos:pos + n]
        pos += n
        return result

    def integer():
        return int.from_bytes(read(4), 'little', signed=True)

    def value(depth=0):
        context.tick()
        if depth > 100:
            raise LimitError('marshal nesting limit')
        byte = read(1)[0]
        flag, tag = byte & 128, chr(byte & 127)
        index = len(refs)
        if flag:
            refs.append(None)
        if tag == 'c':
            start = pos - 1
            result = Value('code', 'marshal')
            if flag:
                refs[index] = result
            # 3.11+ uses localsplusnames/kinds and an exception table;
            # 3.8-3.10 stores nlocals, varnames, freevars and cellvars.
            modern = pos + 20 < len(data) and data[pos + 20] & 127 == ord('s')
            for _ in range(5 if modern else 6):
                integer()
            for _ in range(8):
                value(depth + 1)
            integer()  # firstlineno
            value(depth + 1)  # line table
            if modern:
                value(depth + 1)  # exception table
            result.args = (bytes(data[start:pos]),)
        elif tag in ('N', 'F', 'T', 'S', '.'):
            result = {'N': None, 'F': False, 'T': True, 'S': Value('stopiteration'), '.': Ellipsis}[tag]
        elif tag == 'i':
            result = integer()
        elif tag == 'I':
            result = int.from_bytes(read(8), 'little', signed=True)
        elif tag == 'l':
            n = integer()
            if abs(n) > 1_000_000:
                raise LimitError('marshal integer size')
            result = 0
            for i in range(abs(n)):
                result |= int.from_bytes(read(2), 'little') << (15 * i)
            if n < 0:
                result = -result
        elif tag == 'g':
            result = struct.unpack('<d', read(8))[0]
        elif tag == 'f':
            result = float(read(read(1)[0]))
        elif tag == 'y':
            result = complex(*struct.unpack('<dd', read(16)))
        elif tag in ('s', 't', 'u', 'a', 'A', 'z', 'Z'):
            n = read(1)[0] if tag in ('z', 'Z') else integer()
            raw = bytes(read(n))
            result = raw if tag == 's' else raw.decode('utf-8', 'surrogatepass')
        elif tag == 'r':
            index_ref = integer()
            if not 0 <= index_ref < len(refs):
                raise LoadError('invalid marshal reference')
            result = refs[index_ref]
        elif tag in ('(', ')', '[', '<', '>'):
            n = read(1)[0] if tag == ')' else integer()
            if not 0 <= n < 1_000_000:
                raise LimitError('marshal collection size')
            result = []
            if flag:
                refs[index] = result
            result.extend(value(depth + 1) for _ in range(n))
            result = {'(': tuple, ')': tuple, '[': list, '<': set, '>': frozenset}[tag](result)
        elif tag == '{':
            result = {}
            if flag:
                refs[index] = result
            while pos < len(data) and data[pos] != ord('0'):
                key = value(depth + 1)
                result[key] = value(depth + 1)
            read(1)
        else:
            raise LoadError('invalid marshal type tag')
        if flag:
            refs[index] = result
        return result

    return value()
