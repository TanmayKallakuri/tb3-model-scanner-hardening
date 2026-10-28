"""Decode marshal's data subset without materialising Python code objects."""
import io
import struct

from modelgate.abstract import Obj, LoadStopped
from modelgate.limits import AnalysisLimit


NULL = object()


def read_marshaled(data, budget):
    stream = io.BytesIO(data)
    refs, interned = [], []

    def read(n):
        if n < 0 or n > len(data): raise LoadStopped("invalid marshal length")
        result = stream.read(n)
        if len(result) != n: raise LoadStopped("truncated marshal data")
        return result

    def integer(): return struct.unpack('<i', read(4))[0]

    def item(depth=0):
        budget.tick()
        if depth > 100: raise AnalysisLimit("marshal nesting limit")
        tag = read(1)[0]
        reference, tag = tag & 128, chr(tag & 127)
        index = len(refs)
        if reference: refs.append(None)
        if tag == '0': result = NULL
        elif tag == 'N': result = None
        elif tag == 'F': result = False
        elif tag == 'T': result = True
        elif tag == 'S': result = Obj('builtins.StopIteration')
        elif tag == '.': result = Ellipsis
        elif tag == 'i': result = integer()
        elif tag == 'I': result = struct.unpack('<q', read(8))[0]
        elif tag == 'l':
            count = integer()
            if abs(count) > 100000: raise AnalysisLimit('marshal integer limit')
            result = 0
            for i in range(abs(count)): result |= struct.unpack('<H', read(2))[0] << (15*i)
            if count < 0: result = -result
        elif tag == 'g': result = struct.unpack('<d', read(8))[0]
        elif tag == 'y': result = complex(*struct.unpack('<dd', read(16)))
        elif tag == 'f': result = float(read(read(1)[0]))
        elif tag == 'x': result = complex(float(read(read(1)[0])), float(read(read(1)[0])))
        elif tag in ('s', 't', 'u', 'a', 'A', 'z', 'Z'):
            size = read(1)[0] if tag in ('z', 'Z') else integer()
            raw = read(size)
            result = raw if tag == 's' else raw.decode('utf-8', 'surrogatepass')
            if tag == 't': interned.append(result)
        elif tag == 'R': result = interned[integer()]
        elif tag == 'r': result = refs[integer()]
        elif tag in ('(', ')', '[', '<', '>'):
            size = read(1)[0] if tag == ')' else integer()
            if size < 0 or size > 1000000: raise AnalysisLimit('marshal collection limit')
            values = []
            if reference: refs[index] = values
            for _ in range(size): values.append(item(depth+1))
            result = tuple(values) if tag in ('(', ')') else set(values) if tag == '<' else frozenset(values) if tag == '>' else values
        elif tag == '{':
            result = {}
            if reference: refs[index] = result
            while True:
                key = item(depth+1)
                if key is NULL: break
                result[key] = item(depth+1)
        elif tag == 'c':
            result = Obj('code')
            if depth == 0:
                # Only its type matters until something builds/calls a function.
                # Never construct a native CodeType from the attacker bytes.
                return result
            if reference: refs[index] = result
            for _ in range(5): integer()
            for _ in range(8): item(depth+1)
            integer()
            item(depth+1)
            item(depth+1)
        else: raise LoadStopped('unknown marshal type')
        if reference: refs[index] = result
        return result

    return item()
