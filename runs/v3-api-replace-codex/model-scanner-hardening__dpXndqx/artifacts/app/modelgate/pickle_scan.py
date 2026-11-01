"""Static stack-machine interpretation of pickle protocols 0 through 5.

pickletools only decodes opcodes. Values are built from data or inert symbols;
no pickle.Unpickler, target imports, or artifact-provided functions are used.
"""
from __future__ import annotations

import io
import math
import pickletools

from modelgate.limits import AnalysisLimit, MAX_BYTES, MAX_ITEMS
from modelgate.report import Finding
from modelgate.symbolic import (END, UNKNOWN, CapabilityFound, Instance, LoadError, Unknown, World)

TORCH_MAGIC = 0x1950a86a20f9469cfc6c
MARK = object()
SCALARS = {'INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4',
           'STRING', 'BINSTRING', 'SHORT_BINSTRING', 'BINBYTES', 'SHORT_BINBYTES',
           'BINBYTES8', 'BYTEARRAY8', 'UNICODE', 'SHORT_BINUNICODE', 'BINUNICODE',
           'BINUNICODE8', 'FLOAT', 'BINFLOAT'}


class Interpreter:
    def __init__(self, world, data, location, persistent=False, joblib=False):
        self.world = world
        self.stream = io.BytesIO(data) if isinstance(data, (bytes, bytearray, memoryview)) else data
        self.location = location
        self.persistent = persistent
        self.joblib = joblib
        self.proto = 0
        self.pos = 0
        self.stream_number = 0
        self.memo = {}
        self.stack = []
        self.stopped = False
        self.persistent_callback = None

    def pop_mark(self):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i] is MARK:
                values = self.stack[i + 1:]
                del self.stack[i:]
                return values
        raise LoadError('pickle MARK missing')

    def one(self, reuse=False):
        if not reuse: self.proto = 0
        self.stack = []
        if not reuse: self.memo = {}
        self.stream_number += 1
        oldloc = self.world.location
        try:
            for op, arg, pos in pickletools.genops(self.stream):
                self.world.budget.tick()
                self.world.location = f'{self.location}:stream[{self.stream_number}]@{pos} {op.name}'
                name = op.name
                stack = self.stack
                if name in SCALARS:
                    stack.append(arg)
                elif name == 'NONE': stack.append(None)
                elif name == 'NEWTRUE': stack.append(True)
                elif name == 'NEWFALSE': stack.append(False)
                elif name == 'MARK': stack.append(MARK)
                elif name == 'EMPTY_TUPLE': stack.append(())
                elif name == 'EMPTY_LIST': stack.append([])
                elif name == 'EMPTY_DICT': stack.append({})
                elif name == 'EMPTY_SET': stack.append(set())
                elif name == 'PROTO':
                    if arg > 5: raise AnalysisLimit('unsupported pickle protocol')
                    self.proto = arg
                elif name == 'FRAME':
                    if arg > MAX_BYTES: raise AnalysisLimit('pickle frame size limit exceeded')
                elif name == 'POP':
                    if stack and stack[-1] is MARK: self.pop_mark()
                    else: stack.pop()
                elif name == 'POP_MARK': self.pop_mark()
                elif name == 'DUP': stack.append(stack[-1])
                elif name in ('TUPLE', 'LIST', 'DICT', 'FROZENSET'):
                    vals = self.pop_mark()
                    if name == 'TUPLE': stack.append(tuple(vals))
                    elif name == 'LIST': stack.append(vals)
                    elif name == 'FROZENSET':
                        values = set()
                        for value in vals: self.world.set_add(values, value)
                        result = frozenset(values)
                        self.world.copy_hashes(values, result)
                        stack.append(result)
                    else:
                        if len(vals) % 2: raise LoadError('odd dictionary item count')
                        obj = {}
                        for i in range(0, len(vals), 2): self.world.setitem(obj, vals[i], vals[i + 1])
                        stack.append(obj)
                elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
                    n = int(name[-1])
                    vals = tuple(stack[-n:])
                    del stack[-n:]
                    stack.append(vals)
                elif name == 'APPEND':
                    val = stack.pop()
                    self.append(stack[-1], [val])
                elif name in ('APPENDS', 'ADDITEMS'):
                    vals = self.pop_mark()
                    if name == 'ADDITEMS':
                        if isinstance(stack[-1], set):
                            for value in vals: self.world.set_add(stack[-1], value)
                        else:
                            for v in vals: self.world.call(self.world.attr(stack[-1], 'add'), (v,))
                    else: self.append(stack[-1], vals)
                elif name == 'SETITEM':
                    val, key = stack.pop(), stack.pop()
                    self.world.setitem(stack[-1], key, val)
                elif name == 'SETITEMS':
                    vals = self.pop_mark()
                    if len(vals) % 2: raise LoadError('odd dictionary item count')
                    for i in range(0, len(vals), 2): self.world.setitem(stack[-1], vals[i], vals[i + 1])
                elif name in ('PUT', 'BINPUT', 'LONG_BINPUT'):
                    if arg < 0: raise LoadError('negative pickle memo index')
                    self.memo[arg] = stack[-1]
                elif name == 'MEMOIZE': self.memo[len(self.memo)] = stack[-1]
                elif name in ('GET', 'BINGET', 'LONG_BINGET'):
                    if arg not in self.memo: raise LoadError('undefined pickle memo reference')
                    stack.append(self.memo[arg])
                elif name == 'GLOBAL':
                    module, attr = arg.split(' ', 1)
                    stack.append(self.world.global_ref(module, attr, self.proto))
                elif name == 'STACK_GLOBAL':
                    attr, module = stack.pop(), stack.pop()
                    if isinstance(attr, Unknown) or isinstance(module, Unknown):
                        stack.append(Unknown('opaque STACK_GLOBAL reference'))
                        continue
                    if not isinstance(attr, str) or not isinstance(module, str):
                        raise LoadError('STACK_GLOBAL requires strings')
                    stack.append(self.world.global_ref(module, attr, self.proto))
                elif name in ('EXT1', 'EXT2', 'EXT4'):
                    cache = getattr(self.world, 'extension_cache', {})
                    value = cache.get(arg, END)
                    if value is END:
                        extension = getattr(self.world, 'extensions', {}).get(arg)
                        if extension is None: raise LoadError('unregistered pickle extension')
                        value = self.world.global_ref(*extension, self.proto)
                        cache[arg] = value
                        self.world.extension_cache = cache
                    stack.append(value)
                elif name in ('REDUCE', 'NEWOBJ', 'NEWOBJ_EX'):
                    kw = stack.pop() if name == 'NEWOBJ_EX' else {}
                    args, func = stack.pop(), stack.pop()
                    stack.append(self.world.call(func, args, kw, 'call' if name == 'REDUCE' else 'new'))
                elif name in ('INST', 'OBJ'):
                    vals = self.pop_mark()
                    if name == 'INST':
                        module, attr = arg.split(' ', 1)
                        func = self.world.global_ref(module, attr, self.proto)
                    else: func, vals = vals[0], vals[1:]
                    isclass = self.world.is_class(func)
                    stack.append(self.world.call(func, tuple(vals), {}, 'new' if isclass and not vals else 'call'))
                elif name == 'BUILD':
                    state = stack.pop()
                    self.world.build(stack[-1], state)
                    if self.joblib and isinstance(stack[-1], Instance) and stack[-1].name.endswith('.NumpyArrayWrapper'):
                        stack[-1] = self.joblib_array(stack[-1])
                elif name in ('PERSID', 'BINPERSID'):
                    pid = stack.pop() if name == 'BINPERSID' else arg
                    if self.persistent_callback is not None:
                        stack.append(self.world.call(self.persistent_callback, (pid,)))
                    elif not self.persistent: raise LoadError('plain pickle has no persistent-id handler')
                    else: stack.append(Instance('torch.storage', (pid,)))
                elif name == 'NEXT_BUFFER':
                    raise LoadError('standard loader has no out-of-band buffers')
                elif name == 'READONLY_BUFFER': pass
                elif name == 'STOP':
                    self.pos = self.stream.tell()
                    if not stack: raise LoadError('empty pickle stack')
                    return stack.pop()
                else: raise AnalysisLimit('unsupported pickle opcode ' + name)
                if len(stack) > MAX_ITEMS or len(self.memo) > MAX_ITEMS:
                    raise AnalysisLimit('pickle object count limit exceeded')
            raise AnalysisLimit('pickle has no STOP')
        finally:
            self.pos = self.stream.tell()
            self.world.location = oldloc

    def append(self, obj, values):
        if isinstance(obj, list): obj.extend(values)
        else:
            for value in values: self.world.call(self.world.attr(obj, 'append'), (value,))

    def joblib_array(self, wrapper):
        attrs = wrapper.attrs
        dtype = attrs.get('dtype')
        spec = dtype.args[0] if isinstance(dtype, Instance) and dtype.args else dtype
        shape = attrs.get('shape', ())
        if not isinstance(shape, tuple) or not all(isinstance(n, int) and n >= 0 for n in shape):
            raise AnalysisLimit('unsupported joblib array shape')
        state = dtype.state if isinstance(dtype, Instance) else None
        hasobject = isinstance(spec, str) and spec.lstrip('<>=|').startswith('O')
        hasobject |= isinstance(state, tuple) and len(state) >= 8 and isinstance(state[7], int) and bool(state[7] & 1)
        if hasobject:
            nested = Interpreter(self.world, self.stream.getbuffer()[self.stream.tell():], self.location + ':joblib-array')
            result = nested.one()
            self.stream.seek(self.stream.tell() + nested.pos)
            return result
        size = dtype_size(spec)
        if size is None and isinstance(state, tuple) and len(state) >= 6 and isinstance(state[5], int) and state[5] >= 0:
            size = state[5]
        count = math.prod(shape)
        if size is None: raise AnalysisLimit('unsupported joblib array dtype')
        alignment = attrs.get('numpy_array_alignment_bytes')
        if alignment is not None:
            raw = self.stream.read(1)
            if not raw: raise LoadError('truncated joblib alignment')
            self.stream.seek(raw[0], 1)
        nbytes = count * size
        if nbytes > MAX_BYTES: raise AnalysisLimit('joblib array size limit exceeded')
        end = self.stream.tell() + nbytes
        if end > len(self.stream.getbuffer()): raise LoadError('truncated joblib array')
        self.stream.seek(end)
        return Instance('numpy.ndarray', state=wrapper)


def dtype_size(spec):
    if not isinstance(spec, str): return None
    spec = spec.lstrip('<>=|')
    if len(spec) > 1 and spec[1:].isdigit():
        return int(spec[1:]) * (4 if spec[0] == 'U' else 1)
    return {'?': 1, 'b': 1, 'B': 1, 'h': 2, 'H': 2, 'i': 4, 'I': 4,
            'l': 8, 'L': 8, 'q': 8, 'Q': 8, 'f': 4, 'd': 8,
            'bool': 1, 'int64': 8, 'float64': 8, 'float32': 4}.get(spec)


def interpret(data, location, world=None, *, single=False, persistent=False, legacy=False, joblib=False, strict=False):
    world = world or World()
    vm = Interpreter(world, data, location, persistent=persistent, joblib=joblib)
    value = UNKNOWN
    first = True
    torch = legacy
    count = 0
    object_value = UNKNOWN
    try:
        while vm.pos < len(data):
            if torch: vm.persistent = count == 3
            value = vm.one()
            count += 1
            if first:
                first = False
                torch = torch or not single and type(value) is int and value == TORCH_MAGIC
                if legacy and value != TORCH_MAGIC: raise LoadError('invalid torch magic number')
            if torch and count == 2 and value != 1001: raise LoadError('invalid torch protocol version')
            if torch and count == 4: object_value = value
            if single and not torch: break
            if torch and count == 5: break
    except CapabilityFound:
        raise
    except LoadError:
        if strict: raise
        pass  # no later opcode would be executed by this loader
    except AnalysisLimit as exc:
        world.findings.append(Finding('unsupported', location, str(exc)))
    except Exception as exc:
        if strict and isinstance(exc, (ValueError, EOFError, UnicodeError)):
            raise LoadError('nested loader cannot decode pickle') from None
        world.findings.append(Finding('unsupported', location, f'cannot decode pickle: {type(exc).__name__}: {exc}'))
    return object_value if torch and count >= 4 else value, vm.pos


def scan_pickle(data, location):
    world = World()
    interpret(data, location, world)
    return world.findings


class LimitedReader:
    """Bound literal reads while walking metadata in a large legacy file."""
    def __init__(self, source, budget):
        self.source, self.budget = source, budget

    def read(self, n=-1):
        if n < 0: raise AnalysisLimit('unbounded pickle literal read')
        if n > MAX_BYTES:
            saved = self.source.tell()
            self.source.seek(0, 2)
            remaining = self.source.tell() - saved
            self.source.seek(saved)
            if n > remaining: raise LoadError('truncated pickle literal')
            raise AnalysisLimit('pickle literal size limit exceeded')
        self.budget.add_bytes(n)
        return self.source.read(n)

    def readline(self):
        data = self.source.readline(MAX_BYTES + 1)
        self.budget.add_bytes(len(data))
        return data

    def tell(self): return self.source.tell()


def is_legacy_prefix(data):
    try:
        ops = [(op.name, arg) for op, arg, _ in pickletools.genops(data) if op.name not in ('PROTO', 'FRAME')]
        return len(ops) == 2 and ops[0][0] in ('INT', 'LONG', 'LONG1', 'LONG4') and ops[0][1] == TORCH_MAGIC and ops[1][0] == 'STOP'
    except (ValueError, UnicodeError): return False


def scan_legacy_stream(source, location, world):
    vm = Interpreter(world, LimitedReader(source, world.budget), location)
    try:
        for i in range(5):
            vm.persistent = i == 3
            value = vm.one()
            if i == 0 and value != TORCH_MAGIC: raise LoadError('invalid torch magic')
            if i == 1 and value != 1001: raise LoadError('invalid torch protocol')
    except (LoadError, EOFError, ValueError):
        pass


class MemoryReader:
    """A live, inert file interface for nested Unpickler instances."""
    def __init__(self, world, memory):
        self.world, self.memory = world, memory

    def read(self, n=-1):
        if n < 0 or n > MAX_BYTES: raise AnalysisLimit('pickle literal size limit exceeded')
        data = self.world.bound(self.memory, 'read', (n,), {})
        if not isinstance(data, (bytes, bytearray)): raise AnalysisLimit('opaque unpickler bytes')
        return data

    def readline(self):
        data = self.world.bound(self.memory, 'readline', (), {})
        if not isinstance(data, (bytes, bytearray)): raise AnalysisLimit('opaque unpickler bytes')
        return data

    def tell(self): return self.memory.pos
