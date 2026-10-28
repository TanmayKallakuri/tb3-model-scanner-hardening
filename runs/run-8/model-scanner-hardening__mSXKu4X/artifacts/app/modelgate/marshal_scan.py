"""Read marshal data into inert values; never instantiate a Python code object."""
from __future__ import annotations
import io
import struct
from modelgate.symbolic import Symbol, Stopped, Unsupported, MAX_BYTES, MAX_ITEMS

NULL = object()


def read_marshaled(data, runtime):
    stream = io.BytesIO(data)
    refs = []
    strings = []

    def read(n):
        if n < 0 or n > MAX_BYTES:
            raise Stopped('invalid marshal length')
        value = stream.read(n)
        if len(value) != n:
            raise Stopped('truncated marshal data')
        return value

    def integer():
        return struct.unpack('<i', read(4))[0]

    def value(depth=0):
        runtime.tick()
        if depth > 100:
            raise Unsupported('marshal nesting limit exceeded')
        tag = read(1)[0]
        ref = bool(tag & 128)
        tag = chr(tag & 127)
        slot = len(refs)
        if ref:
            refs.append(NULL)
        if tag in ('N', 'F', 'T', '.', 'S', '0'):
            obj = {'N': None, 'F': False, 'T': True, '.': Ellipsis,
                   'S': runtime.ref('builtins.StopIteration'), '0': NULL}[tag]
        elif tag == 'i':
            obj = integer()
        elif tag == 'I':
            obj = struct.unpack('<q', read(8))[0]
        elif tag == 'l':
            length = integer()
            if abs(length) > MAX_ITEMS:
                raise Unsupported('marshal integer limit exceeded')
            obj = 0
            for i in range(abs(length)):
                digit = struct.unpack('<H', read(2))[0]
                if digit >= 32768:
                    raise Stopped('invalid marshal integer digit')
                obj |= digit << (15 * i)
            if length < 0:
                obj = -obj
        elif tag in ('f', 'g'):
            obj = float(read(read(1)[0])) if tag == 'f' else struct.unpack('<d', read(8))[0]
        elif tag in ('x', 'y'):
            if tag == 'y':
                real, imag = struct.unpack('<dd', read(16))
            else:
                real, imag = float(read(read(1)[0])), float(read(read(1)[0]))
            obj = complex(real, imag)
        elif tag in ('s', 't', 'u', 'a', 'A', 'z', 'Z'):
            raw = read(read(1)[0] if tag in ('z', 'Z') else integer())
            obj = raw if tag == 's' else raw.decode('utf-8', 'surrogatepass')
            if tag in ('t', 'A', 'Z'):
                strings.append(obj)
        elif tag == 'r':
            idx = integer()
            if idx < 0 or idx >= len(refs) or refs[idx] is NULL:
                raise Stopped('invalid marshal reference')
            obj = refs[idx]
        elif tag == 'R':
            idx = integer()
            if idx < 0 or idx >= len(strings):
                raise Stopped('invalid marshal string reference')
            obj = strings[idx]
        elif tag in ('(', ')', '[', '<', '>'):
            size = read(1)[0] if tag == ')' else integer()
            if size < 0 or size > MAX_ITEMS:
                raise Unsupported('marshal container size limit exceeded')
            obj = []
            if ref:
                refs[slot] = obj
            for _ in range(size):
                obj.append(value(depth + 1))
            if tag in ('(', ')'):
                obj = tuple(obj)
            elif tag in ('<', '>'):
                obj = set(obj) if tag == '<' else frozenset(obj)
        elif tag == '{':
            obj = {}
            if ref:
                refs[slot] = obj
            while True:
                key = value(depth + 1)
                if key is NULL:
                    break
                if len(obj) >= MAX_ITEMS:
                    raise Unsupported('marshal dictionary size limit exceeded')
                obj[key] = value(depth + 1)
        elif tag == 'c':
            obj = Symbol('code', 'marshal.code')
            if ref:
                refs[slot] = obj
            # CPython 3.11+ code format. Code and constant tuples remain inert
            # byte strings and descriptions, including nested code objects.
            counts = [integer() for _ in range(5)]
            if any(n < 0 for n in counts[:4]):
                raise Stopped('invalid marshalled code counts')
            fields = [value(depth + 1) for _ in range(8)]
            firstlineno = integer()
            tables = [value(depth + 1), value(depth + 1)]
            if not isinstance(fields[0], bytes) or not all(isinstance(fields[i], tuple) for i in (1, 2, 3)):
                raise Stopped('invalid marshalled code fields')
            if not isinstance(fields[4], bytes) or not all(isinstance(fields[i], str) for i in (5, 6, 7)):
                raise Stopped('invalid marshalled code metadata')
            if not all(isinstance(x, bytes) for x in tables):
                raise Stopped('invalid marshalled code tables')
            obj.value = fields[0]
        else:
            raise Stopped('unknown marshal type tag')
        if ref:
            refs[slot] = obj
        return obj

    return value(), stream.tell()
