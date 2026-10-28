"""A static pickle machine.

Only literal values and symbolic references are built. No GLOBAL is imported,
no reducer is executed, and no artifact object is instantiated. Calls are
interpreted by a small, explicit model of Python's data and callable semantics.
"""
from __future__ import annotations

import _compat_pickle
import io
import pickletools
from dataclasses import dataclass, field

from modelgate.analysis import Budget, Unsupported
from modelgate.report import Finding


@dataclass(eq=False)
class Ref:
    name: str


@dataclass(eq=False)
class Module:
    name: str


@dataclass(eq=False)
class Object:
    cls: str
    args: tuple = ()
    attrs: dict = field(default_factory=dict)
    state: object = None


@dataclass(eq=False)
class Partial:
    func: object
    args: tuple = ()
    keywords: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Getter:
    kind: str
    args: tuple
    keywords: dict = field(default_factory=dict)


@dataclass(eq=False)
class Bound:
    obj: object
    name: str


@dataclass(eq=False)
class Buffer:
    data: object = b''
    pos: int = 0


@dataclass(eq=False)
class Lazy:
    kind: str
    func: object
    items: object


@dataclass(eq=False)
class Unknown:
    reason: str = 'unknown value'


UNKNOWN = Unknown()
MARK = object()
TORCH_MAGIC = 0x1950A86A20F9469CFC6C
ALIASES = {'posix': 'os', 'nt': 'os', '_pickle': 'pickle', '_io': 'io',
           '_operator': 'operator', '_functools': 'functools', '_socket': 'socket',
           'pathlib.PosixPath': 'pathlib.Path', 'pathlib.WindowsPath': 'pathlib.Path',
           'socket.SocketType': 'socket.socket', 'socket._socketobject': 'socket.socket'}


def canonical(name):
    for old, new in ALIASES.items():
        if name == old or name.startswith(old + '.'):
            name = new + name[len(old):]
    return name


def global_ref(module, name, proto=4):
    if not isinstance(module, str) or not isinstance(name, str):
        raise Unsupported('GLOBAL requires literal module and attribute names')
    if proto < 3:
        if (module, name) in _compat_pickle.NAME_MAPPING:
            module, name = _compat_pickle.NAME_MAPPING[module, name]
        else:
            module = _compat_pickle.IMPORT_MAPPING.get(module, module)
    return Ref(canonical(module + '.' + name))


def as_bytes(value):
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return None


def typename(value):
    if isinstance(value, Object):
        return value.cls
    if isinstance(value, Partial):
        return 'functools.partial'
    if isinstance(value, Buffer):
        return 'io.BytesIO' if isinstance(value.data, (bytes, bytearray)) else 'io.StringIO'
    if type(value) in (str, bytes, bytearray, int, float, bool, list, tuple, dict, set, frozenset):
        return 'builtins.' + type(value).__name__
    return None


class Machine:
    def __init__(self, location, budget=None, depth=0, findings=None):
        self.location = location
        self.budget = budget or Budget()
        self.depth = depth
        self.budget.nest(depth)
        self.findings = findings if findings is not None else []
        self.memo = {}
        self.stack = []
        self.proto = 0
        self.site = location
        self.op = 'CALL'
        self.call_depth = 0

    def finding(self, kind, detail):
        if len(self.findings) < 200:
            self.findings.append(Finding(kind, self.site, detail))

    def take_mark(self):
        stack = self.stack
        for i in range(len(stack) - 1, -1, -1):
            if stack[i] is MARK:
                values = stack[i + 1:]
                del stack[i:]
                return values
        raise Unsupported('pickle stack has no MARK')

    def read(self, fh):
        stack = self.stack
        for opcode, arg, pos in pickletools.genops(fh):
            self.budget.tick()
            op = opcode.name
            self.op = op
            self.site = f'{self.location}@{pos}'
            if op in ('PROTO', 'FRAME'):
                if op == 'PROTO':
                    self.proto = arg
                continue
            if op in ('INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4',
                      'STRING', 'BINSTRING', 'SHORT_BINSTRING', 'UNICODE', 'BINUNICODE',
                      'SHORT_BINUNICODE', 'BINUNICODE8', 'BINBYTES', 'SHORT_BINBYTES',
                      'BINBYTES8', 'BYTEARRAY8', 'FLOAT', 'BINFLOAT'):
                stack.append(arg)
            elif op == 'NONE':
                stack.append(None)
            elif op in ('NEWTRUE', 'NEWFALSE'):
                stack.append(op == 'NEWTRUE')
            elif op == 'MARK':
                stack.append(MARK)
            elif op == 'POP':
                stack.pop()
            elif op == 'POP_MARK':
                self.take_mark()
            elif op == 'DUP':
                stack.append(stack[-1])
            elif op in ('PUT', 'BINPUT', 'LONG_BINPUT', 'MEMOIZE'):
                self.memo[len(self.memo) if op == 'MEMOIZE' else arg] = stack[-1]
            elif op in ('GET', 'BINGET', 'LONG_BINGET'):
                if arg not in self.memo:
                    raise Unsupported(f'undefined pickle memo entry {arg}')
                stack.append(self.memo[arg])
            elif op == 'EMPTY_LIST':
                stack.append([])
            elif op == 'EMPTY_TUPLE':
                stack.append(())
            elif op == 'EMPTY_DICT':
                stack.append({})
            elif op == 'EMPTY_SET':
                stack.append(set())
            elif op in ('LIST', 'TUPLE', 'FROZENSET', 'DICT'):
                items = self.take_mark()
                if op == 'DICT':
                    stack.append(dict(zip(items[::2], items[1::2])))
                else:
                    stack.append({'LIST': list, 'TUPLE': tuple, 'FROZENSET': frozenset}[op](items))
            elif op in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
                n = int(op[-1])
                items = tuple(stack[-n:])
                del stack[-n:]
                stack.append(items)
            elif op in ('APPEND', 'APPENDS', 'ADDITEMS'):
                items = [stack.pop()] if op == 'APPEND' else self.take_mark()
                target = stack[-1]
                if type(target) is list:
                    target.extend(items)
                elif type(target) is set:
                    target.update(items)
                else:
                    for item in items:
                        self.call(self.attr(target, 'add' if op == 'ADDITEMS' else 'append'), (item,))
            elif op in ('SETITEM', 'SETITEMS'):
                if op == 'SETITEM':
                    val, key = stack.pop(), stack.pop()
                    items = [key, val]
                else:
                    items = self.take_mark()
                target = stack[-1]
                for key, val in zip(items[::2], items[1::2]):
                    self.setitem(target, key, val)
            elif op in ('GLOBAL', 'INST'):
                module, name = arg.split(' ', 1)
                ref = global_ref(module, name, self.proto)
                if op == 'INST':
                    args = tuple(self.take_mark())
                    stack.append(self.call(ref, args) if args else self.newobj(ref, ()))
                else:
                    stack.append(ref)
            elif op == 'STACK_GLOBAL':
                name, module = stack.pop(), stack.pop()
                stack.append(global_ref(module, name, self.proto))
            elif op == 'REDUCE':
                args, func = stack.pop(), stack.pop()
                stack.append(self.call(func, args))
            elif op == 'OBJ':
                items = self.take_mark()
                func, args = items[0], tuple(items[1:])
                stack.append(self.call(func, args) if args else self.newobj(func, ()))
            elif op in ('NEWOBJ', 'NEWOBJ_EX'):
                kwargs = stack.pop() if op == 'NEWOBJ_EX' else {}
                args, cls = stack.pop(), stack.pop()
                stack.append(self.newobj(cls, args, kwargs))
            elif op == 'BUILD':
                state = stack.pop()
                self.build(stack[-1], state)
                if isinstance(stack[-1], Object) and stack[-1].cls == 'joblib.numpy_pickle.NumpyArrayWrapper':
                    stack[-1] = self.joblib_array(stack[-1], fh)
            elif op == 'PERSID':
                stack.append(Object('persistent.storage', (arg,)))
            elif op == 'BINPERSID':
                stack.append(Object('persistent.storage', (stack.pop(),)))
            elif op in ('EXT1', 'EXT2', 'EXT4'):
                raise Unsupported('pickle extension registry is not supplied by the artifact')
            elif op == 'NEXT_BUFFER':
                raise Unsupported('pickle requires an out-of-band buffer')
            elif op == 'READONLY_BUFFER':
                pass
            elif op == 'STOP':
                if not stack:
                    raise Unsupported('pickle STOP with empty stack')
                return stack.pop()
            else:
                raise Unsupported(f'unmodelled pickle opcode {op}')
        raise Unsupported('pickle has no STOP')

    def newobj(self, cls, args, kwargs=None):
        # Registry policy explicitly excludes __new__-only construction.
        name = cls.name if isinstance(cls, Ref) else typename(cls)
        if name == 'functools.partial':
            return Partial(args[0] if args else UNKNOWN, tuple(args[1:]), dict(kwargs or {}))
        return Object(name or 'unknown.class', tuple(args) if isinstance(args, (tuple, list)) else ())

    def build(self, obj, state):
        if isinstance(obj, Partial):
            if not isinstance(state, tuple) or len(state) != 4:
                raise Unsupported('invalid functools.partial state')
            obj.func, obj.args, obj.keywords, attrs = state
            obj.attrs = dict(attrs or {})
            return
        if isinstance(obj, Object):
            obj.state = state
            if obj.cls.endswith('NDArrayWrapper') and obj.cls.startswith(('joblib.', 'sklearn.externals.joblib.')):
                self.finding('file-access', 'joblib array wrapper loads a separate file')
            setter = obj.attrs.get('__setstate__')
            if setter is not None:
                self.call(setter, (state,))
                return
            attrs, slots = state, None
            if isinstance(state, tuple) and len(state) == 2:
                attrs, slots = state
            if isinstance(attrs, dict):
                obj.attrs.update(attrs)
            if isinstance(slots, dict):
                obj.attrs.update(slots)

    def attr(self, obj, name, default=UNKNOWN):
        if not isinstance(name, str):
            return Unknown('nonliteral attribute name')
        if isinstance(obj, Module):
            if name == '__dict__':
                return Object('module.namespace', (obj,))
            return Ref(canonical(obj.name + '.' + name))
        if isinstance(obj, Ref):
            if name == '__call__':
                return obj
            if name == '__dict__':
                return Object('class.namespace', (obj,))
            return Ref(canonical(obj.name + '.' + name))
        if name == '__class__' and typename(obj):
            return Ref(typename(obj))
        if isinstance(obj, (Object, Partial)):
            if name == '__dict__':
                return obj.attrs
            if name in obj.attrs:
                return obj.attrs[name]
            if isinstance(obj, Partial):
                if name in ('func', 'args', 'keywords'):
                    return getattr(obj, name)
                if name == '__call__':
                    return obj
            if isinstance(obj, Object) and obj.cls == 'builtins.super':
                return self.attr(obj.args[-1], name, default)
        if typename(obj) or isinstance(obj, (Lazy, Getter)):
            return Bound(obj, name)
        return default

    def getitem(self, obj, key):
        if type(obj) in (dict, list, tuple, str, bytes, bytearray):
            try:
                return obj[key]
            except (TypeError, KeyError, IndexError):
                return UNKNOWN
        if isinstance(obj, Object):
            if obj.cls in ('module.namespace', 'class.namespace'):
                return self.attr(obj.args[0], key)
            if obj.cls in ('collections.OrderedDict', 'collections.defaultdict', 'collections.UserDict'):
                return obj.attrs.get('_items', {}).get(key, UNKNOWN)
        return UNKNOWN

    def setitem(self, obj, key, val):
        if type(obj) is dict:
            obj[key] = val
        elif type(obj) is list and isinstance(key, int):
            obj[key] = val
        elif isinstance(obj, Object) and obj.cls in ('collections.OrderedDict', 'collections.defaultdict', 'collections.UserDict'):
            obj.attrs.setdefault('_items', {})[key] = val
        else:
            self.call(self.attr(obj, '__setitem__'), (key, val))

    def call(self, func, args=(), kwargs=None):
        from modelgate.symbolic import invoke
        if not isinstance(args, (tuple, list)) or not isinstance(kwargs or {}, dict):
            raise Unsupported('call arguments are not a statically known tuple/dictionary')
        self.budget.tick()
        self.call_depth += 1
        if self.call_depth > 80:
            raise Unsupported('symbolic call recursion limit exceeded')
        try:
            return invoke(self, func, tuple(args), kwargs or {})
        finally:
            self.call_depth -= 1

    def joblib_array(self, obj, fh):
        from modelgate.numpy_scan import dtype_info
        dtype = obj.attrs.get('dtype')
        if not isinstance(dtype, Object) or dtype.cls != 'numpy.dtype' or not dtype.args:
            raise Unsupported('joblib array has unknown dtype')
        itemsize, hasobject = dtype_info(dtype.args[0], dtype.state)
        if hasobject:
            child = Machine(self.site + ':object-array', self.budget, self.depth + 1, self.findings)
            return child.read(fh)
        shape = obj.attrs.get('shape')
        if not isinstance(shape, (tuple, list)):
            raise Unsupported('joblib array has unknown shape')
        count = 1
        for dim in shape:
            if type(dim) is not int or dim < 0:
                raise Unsupported('invalid joblib array dimension')
            count *= dim
        if obj.attrs.get('numpy_array_alignment_bytes') is not None:
            padding = fh.read(1)
            if not padding:
                raise Unsupported('truncated joblib array alignment')
            fh.seek(padding[0], 1)
        length = count * itemsize
        start = fh.tell()
        fh.seek(length, 1)
        if fh.tell() > len(fh.getbuffer()):
            raise Unsupported('truncated joblib array data')
        return Object('numpy.ndarray', state={'shape': shape, 'dtype': dtype, 'offset': start})


def read_pickle(data, location, budget=None, depth=0, *, multiple=False):
    """Return findings, symbolic result, and consumed bytes. Never unpickle."""
    budget = budget or Budget()
    findings = []
    fh = data if isinstance(data, io.BytesIO) else io.BytesIO(data)
    end = len(fh.getbuffer())
    value = UNKNOWN
    stream = 0
    legacy = False
    try:
        while fh.tell() < end:
            machine = Machine(f'{location}:pickle[{stream}]', budget, depth, findings)
            value = machine.read(fh)
            stream += 1
            if stream == 1 and type(value) is int and value == TORCH_MAGIC and multiple:
                legacy = True
            if not multiple or (legacy and stream == 5):
                break
        if stream == 0:
            raise Unsupported('empty pickle stream')
        if legacy and stream < 5:
            raise Unsupported('truncated legacy PyTorch checkpoint')
    except Exception as exc:
        findings.append(Finding('unsupported', getattr(locals().get('machine'), 'site', location),
                                f'cannot interpret pickle: {type(exc).__name__}: {exc}'))
    return findings, value, fh.tell()


def scan_pickle(data: bytes, location: str, budget=None, depth=0, *, multiple=True) -> list[Finding]:
    return read_pickle(data, location, budget, depth, multiple=multiple)[0]
