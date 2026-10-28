"""A non-executing interpreter for the pickle virtual machine.

Only literals and scanner-owned symbolic values are constructed. GLOBAL never
imports a module; REDUCE/BUILD never call the referenced Python implementation.
Mutable values preserve identity through the memo, including retargeted partials.
"""
from __future__ import annotations

import _compat_pickle
import ast
import base64
import binascii
import io
import math
import operator
import pickletools
from dataclasses import dataclass, field

from modelgate.policy import CALLS, HOST_CLIENTS, INERT_GLOBALS, INERT_MODULES, NESTED_LOADERS
from modelgate.report import Finding

MAX_VALUE = 128 * 1024 * 1024
MAX_STEPS = 2_000_000
TORCH_MAGIC = 119547037146038801333356


class AnalysisError(ValueError):
    pass


@dataclass(eq=False)
class Symbol:
    kind: str
    name: str = ''
    args: tuple = ()
    attrs: dict = field(default_factory=dict)
    value: object = None


UNKNOWN = Symbol('unknown')
MARK = object()

MODULE_ALIASES = {
    'posix': 'os', 'nt': 'os', '_pickle': 'pickle', '_io': 'io',
    '__builtin__': 'builtins', 'copy_reg': 'copyreg', 'commands': 'subprocess',
    'urllib2': 'urllib.request', 'httplib': 'http.client', '_socket': 'socket',
    '_operator': 'operator', '_functools': 'functools', 'cPickle': 'pickle',
    'cStringIO': 'io', 'StringIO': 'io', 'posixpath': 'os.path', 'ntpath': 'os.path',
}


def canonical(path):
    head, dot, tail = path.partition('.')
    path = MODULE_ALIASES.get(head, head) + dot + tail
    if path in ('pickle._loads', 'dill._dill.loads', 'cloudpickle.cloudpickle.loads'):
        return 'pickle.loads'
    if path in ('pickle._load', 'dill._dill.load', 'cloudpickle.cloudpickle.load'):
        return 'pickle.load'
    for source, dest in (
        ('pathlib.PosixPath.', 'pathlib.Path.'),
        ('pathlib.WindowsPath.', 'pathlib.Path.'),
        ('pathlib._local.Path.', 'pathlib.Path.'),
        ('socket.SocketType', 'socket.socket'),
        ('numpy._core.', 'numpy.core.'),
        ('multiprocessing.context.Process', 'multiprocessing.Process'),
    ):
        if path.startswith(source):
            path = dest + path[len(source):]
    return path


def global_symbol(module, name, protocol=4):
    if not isinstance(module, str) or not isinstance(name, str):
        raise AnalysisError('GLOBAL module and name must be strings')
    if protocol < 3:
        if (module, name) in _compat_pickle.NAME_MAPPING:
            module, name = _compat_pickle.NAME_MAPPING[module, name]
        else:
            module = _compat_pickle.IMPORT_MAPPING.get(module, module)
    return Symbol('global', canonical(module + '.' + name))


def typename(value):
    if isinstance(value, Symbol):
        return value.name
    return 'builtins.' + type(value).__name__


def seq(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    raise AnalysisError('call arguments are not a statically known tuple')


class Interpreter:
    def __init__(self, location, depth=0):
        self.location = location
        self.depth = depth
        self.findings = []
        self.memo = {}
        self.steps = 0
        self.call_depth = 0
        self.where = location
        self.op = ''
        self.last_value = UNKNOWN
        self.consumed = 0

    def finding(self, kind, detail):
        if len(self.findings) < 1000:
            self.findings.append(Finding(kind, self.where, detail))

    def run(self, data, *, first_only=False):
        if self.depth > 24:
            raise AnalysisError('embedded loader nesting limit exceeded')
        stream = io.BytesIO(data)
        stream_no = 0
        legacy = False
        while stream.tell() < len(data):
            stack = []
            protocol = 0
            stream_no += 1
            stopped = False
            for opcode, arg, pos in pickletools.genops(stream):
                self.steps += 1
                if self.steps > MAX_STEPS:
                    raise AnalysisError('pickle opcode limit exceeded')
                name = opcode.name
                self.op = name
                self.where = f'{self.location}:stream[{stream_no}]@{pos}'
                if name == 'PROTO':
                    protocol = arg
                    if arg > 5:
                        raise AnalysisError(f'unsupported pickle protocol {arg}')
                elif name == 'FRAME':
                    pass
                elif name == 'MARK':
                    stack.append(MARK)
                elif name == 'STOP':
                    if not stack:
                        raise AnalysisError('STOP on empty stack')
                    self.last_value = stack[-1]
                    stopped = True
                    break
                elif name == 'POP':
                    stack.pop()
                elif name == 'POP_MARK':
                    self.pop_mark(stack)
                elif name == 'DUP':
                    stack.append(stack[-1])
                elif name in ('NONE', 'NEWTRUE', 'NEWFALSE'):
                    stack.append({'NONE': None, 'NEWTRUE': True, 'NEWFALSE': False}[name])
                elif name in ('INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4',
                              'FLOAT', 'BINFLOAT', 'STRING', 'BINSTRING', 'SHORT_BINSTRING',
                              'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE', 'BINUNICODE8',
                              'BINBYTES', 'SHORT_BINBYTES', 'BINBYTES8', 'BYTEARRAY8'):
                    stack.append(arg)
                elif name == 'EMPTY_TUPLE':
                    stack.append(())
                elif name == 'EMPTY_LIST':
                    stack.append([])
                elif name == 'EMPTY_DICT':
                    stack.append({})
                elif name == 'EMPTY_SET':
                    stack.append(set())
                elif name in ('TUPLE', 'LIST', 'DICT', 'FROZENSET'):
                    values = self.pop_mark(stack)
                    if name == 'TUPLE': value = tuple(values)
                    elif name == 'LIST': value = values
                    elif name == 'FROZENSET': value = frozenset(values)
                    else: value = dict(zip(values[::2], values[1::2]))
                    stack.append(value)
                elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
                    count = int(name[-1])
                    value = tuple(stack[-count:])
                    del stack[-count:]
                    stack.append(value)
                elif name == 'APPEND':
                    value = stack.pop()
                    self.append(stack[-1], [value])
                elif name in ('APPENDS', 'ADDITEMS'):
                    values = self.pop_mark(stack)
                    if name == 'ADDITEMS' and isinstance(stack[-1], set):
                        stack[-1].update(values)
                    else:
                        self.append(stack[-1], values)
                elif name == 'SETITEM':
                    value, key = stack.pop(), stack.pop()
                    self.setitem(stack[-1], key, value)
                elif name == 'SETITEMS':
                    values = self.pop_mark(stack)
                    for k, v in zip(values[::2], values[1::2]):
                        self.setitem(stack[-1], k, v)
                elif name in ('PUT', 'BINPUT', 'LONG_BINPUT'):
                    self.memo[arg] = stack[-1]
                elif name == 'MEMOIZE':
                    self.memo[len(self.memo)] = stack[-1]
                elif name in ('GET', 'BINGET', 'LONG_BINGET'):
                    if arg not in self.memo:
                        raise AnalysisError(f'missing pickle memo {arg}')
                    stack.append(self.memo[arg])
                elif name in ('GLOBAL', 'INST'):
                    module, attr = arg.split(' ', 1)
                    value = global_symbol(module, attr, protocol)
                    if name == 'INST':
                        args = tuple(self.pop_mark(stack))
                        value = self.call(value, args) if args else self.new(value, args)
                    stack.append(value)
                elif name == 'STACK_GLOBAL':
                    attr, module = stack.pop(), stack.pop()
                    stack.append(global_symbol(module, attr, protocol))
                elif name == 'REDUCE':
                    args, fn = stack.pop(), stack.pop()
                    stack.append(self.call(fn, seq(args)))
                elif name == 'OBJ':
                    values = self.pop_mark(stack)
                    fn, args = values[0], tuple(values[1:])
                    stack.append(self.call(fn, args) if args else self.new(fn, args))
                elif name in ('NEWOBJ', 'NEWOBJ_EX'):
                    kwargs = stack.pop() if name == 'NEWOBJ_EX' else {}
                    args, cls = stack.pop(), stack.pop()
                    stack.append(self.new(cls, seq(args), kwargs))
                elif name == 'BUILD':
                    state = stack.pop()
                    self.build(stack[-1], state)
                    self.joblib_payload(stack[-1], stream)
                elif name == 'PERSID':
                    stack.append(Symbol('persistent', value=arg))
                elif name == 'BINPERSID':
                    stack.append(Symbol('persistent', value=stack.pop()))
                elif name == 'NEXT_BUFFER':
                    stack.append(Symbol('buffer'))
                elif name == 'READONLY_BUFFER':
                    pass
                elif name in ('EXT1', 'EXT2', 'EXT4'):
                    # Resolving an extension requires a loader-specific registry.
                    stack.append(Symbol('extension', str(arg)))
                else:
                    raise AnalysisError(f'unsupported opcode {name}')
                if self.findings:
                    # Once a capability executes, later opcodes cannot undo
                    # it. Avoid spending the remaining budget on the payload.
                    return UNKNOWN
            if not stopped:
                raise AnalysisError('pickle has no STOP')
            self.consumed = stream.tell()
            if stream_no == 1 and type(self.last_value) is int and self.last_value == TORCH_MAGIC:
                legacy = True
            if first_only or (legacy and stream_no == 5):
                break
        if not stream_no:
            raise AnalysisError('empty pickle')
        return self.last_value

    @staticmethod
    def pop_mark(stack):
        for i in range(len(stack) - 1, -1, -1):
            if stack[i] is MARK:
                result = stack[i + 1:]
                del stack[i:]
                return result
        raise AnalysisError('pickle MARK missing')

    def append(self, obj, values):
        if isinstance(obj, list):
            obj.extend(values)
        elif isinstance(obj, Symbol):
            obj.attrs.setdefault('_items', []).extend(values)
        else:
            raise AnalysisError('APPEND target is not a sequence')

    def setitem(self, obj, key, value):
        if isinstance(obj, dict):
            obj[key] = value
        elif isinstance(obj, (list, bytearray)):
            obj[key] = value
        elif isinstance(obj, Symbol):
            obj.attrs.setdefault('_items', {})[key] = value
        else:
            raise AnalysisError('SETITEM target is not a mapping')

    def new(self, cls, args, kwargs=None):
        # Policy explicitly gives NEWOBJ/NEWOBJ_EX no capability. Preserve
        # enough __new__ state to understand later BUILD or method calls.
        if not isinstance(cls, Symbol):
            return Symbol('instance', args=args)
        if cls.name == 'functools.partial':
            return Symbol('partial', args=args[1:], value=args[0] if args else UNKNOWN,
                          attrs={'keywords': dict(kwargs or {})})
        if cls.kind == 'class':
            return Symbol('instance', cls.name, args=args, value=cls)
        return Symbol('instance', cls.name, args=args)

    def build(self, obj, state):
        if not isinstance(obj, Symbol):
            return
        if obj.kind == 'partial' or obj.name == 'functools.partial':
            if isinstance(state, tuple) and len(state) == 4:
                obj.kind = 'partial'
                obj.value, args, keywords, namespace = state
                obj.args = seq(args)
                obj.attrs = dict(namespace or {})
                obj.attrs['keywords'] = dict(keywords or {})
                return
            raise AnalysisError('invalid partial state')
        if obj.name in ('io.BytesIO', 'io.StringIO'):
            if isinstance(state, tuple) and state:
                obj.value = state[0]
                obj.attrs['position'] = state[1] if len(state) > 1 else 0
        # Some pickles install __setstate__ on an instance before BUILD.
        setter = obj.attrs.get('__setstate__')
        if setter is None and isinstance(obj.value, Symbol) and obj.value.kind == 'class':
            setter = self.attr(obj, '__setstate__') if '__setstate__' in obj.value.attrs else None
        if setter is not None:
            self.call(setter, (state,))
            return
        obj.attrs['_state'] = state
        if isinstance(state, dict):
            obj.attrs.update(state)
        elif isinstance(state, tuple) and len(state) == 2:
            for part in state:
                if isinstance(part, dict):
                    obj.attrs.update(part)

    def attr(self, obj, name, default=UNKNOWN):
        if not isinstance(name, str):
            return UNKNOWN
        if isinstance(obj, Symbol):
            if name in obj.attrs:
                return obj.attrs[name]
            if obj.kind == 'partial':
                if name == 'func': return obj.value
                if name == 'args': return obj.args
                if name == 'keywords': return obj.attrs.get('keywords', {})
            if name == '__dict__':
                if obj.kind in ('module', 'global'):
                    return Symbol('namespace', obj.name)
                return obj.attrs
            if name in ('__call__',) and obj.kind in ('global', 'partial', 'attrgetter', 'methodcaller', 'itemgetter', 'function'):
                return obj
            if obj.kind in ('global', 'module'):
                return Symbol('global', canonical(obj.name + '.' + name))
            if obj.kind == 'instance':
                if name == '__class__':
                    return obj.value if isinstance(obj.value, Symbol) and obj.value.kind == 'class' else Symbol('global', obj.name)
                if isinstance(obj.value, Symbol) and obj.value.kind == 'class' and name in obj.value.attrs:
                    attr = obj.value.attrs[name]
                    # Ordinary global builtins stored on a class do not bind;
                    # Python functions and explicit MethodTypes do.
                    if isinstance(attr, Symbol) and attr.kind == 'function':
                        return Symbol('method', value=attr, args=(obj,))
                    return attr
                return Symbol('bound', canonical(obj.name + '.' + name), value=obj)
            if obj.kind == 'partial' and name == '__setstate__':
                return Symbol('bound', 'functools.partial.__setstate__', value=obj)
            return default
        if name == '__class__':
            return Symbol('global', typename(obj))
        return Symbol('bound', typename(obj) + '.' + name, value=obj)

    def getitem(self, obj, key):
        if isinstance(obj, (dict, list, tuple, bytes, bytearray, str)):
            return obj[key]
        if isinstance(obj, Symbol):
            if obj.kind == 'namespace' and isinstance(key, str):
                return Symbol('global', canonical(obj.name + '.' + key))
            return obj.attrs.get('_items', {}).get(key, UNKNOWN)
        return UNKNOWN

    def call(self, fn, args, kwargs=None):
        self.call_depth += 1
        try:
            if self.call_depth > 80:
                raise AnalysisError('symbolic call recursion limit exceeded')
            return self._call(fn, args, kwargs or {})
        finally:
            self.call_depth -= 1

    def _call(self, fn, args, kwargs):
        if not isinstance(fn, Symbol):
            raise AnalysisError('unresolved callable')
        if fn.kind == 'partial':
            keywords = dict(fn.attrs.get('keywords', {}))
            keywords.update(kwargs)
            return self.call(fn.value, fn.args + args, keywords)
        if fn.kind == 'attrgetter':
            values = []
            for name in fn.args:
                obj = args[0]
                for part in name.split('.'):
                    obj = self.attr(obj, part)
                values.append(obj)
            return values[0] if len(values) == 1 else tuple(values)
        if fn.kind == 'itemgetter':
            values = tuple(self.getitem(args[0], key) for key in fn.args)
            return values[0] if len(values) == 1 else values
        if fn.kind == 'methodcaller':
            return self.call(self.attr(args[0], fn.name), fn.args, fn.attrs)
        if fn.kind == 'function':
            self.finding('code-execution', f'{self.op} serialized Python function')
            return UNKNOWN
        if fn.kind == 'method':
            return self.call(fn.value, fn.args + args, kwargs)
        if fn.kind == 'decorator':
            return args[0]
        if fn.kind == 'class':
            obj = self.new(fn, args, kwargs)
            if '__new__' in fn.attrs:
                self.call(fn.attrs['__new__'], (fn,) + args, kwargs)
            if '__init__' in fn.attrs:
                self.call(self.attr(obj, '__init__'), args, kwargs)
            return obj
        if fn.kind == 'instance':
            return self.call(self.attr(fn, '__call__'), args, kwargs)
        if fn.kind not in ('global', 'bound'):
            raise AnalysisError(f'unresolved callable ({fn.kind})')
        name = canonical(fn.name)
        bound = fn.value if fn.kind == 'bound' else None
        if name == 'builtins.type.__call__':
            return self.call(args[0], args[1:], kwargs)
        if name.endswith('.__new__'):
            return self.new(args[0] if args else Symbol('global', name[:-8]), args[1:], kwargs)
        if name.endswith('.__init__'):
            cls = name[:-9]
            if cls in CALLS or cls in HOST_CLIENTS or cls in NESTED_LOADERS:
                return self.call(Symbol('global', cls), args if bound is not None else args[1:], kwargs)
        if name.endswith('.__call__') and bound is None and name not in CALLS:
            return self.call(Symbol('global', name[:-9]), args, kwargs)
        if name in ('builtins.getattr', 'builtins.object.__getattribute__', 'builtins.type.__getattribute__', 'operator.getattr') and bound is None:
            return self.attr(args[0], args[1], args[2] if len(args) > 2 else UNKNOWN)
        if name in ('builtins.setattr', 'builtins.object.__setattr__'):
            if isinstance(args[0], Symbol):
                args[0].attrs[args[1]] = args[2]
            return None
        if name == 'builtins.delattr':
            if isinstance(args[0], Symbol):
                args[0].attrs.pop(args[1], None)
            return None
        if name in ('builtins.__import__', 'importlib.import_module'):
            module = args[0] if args else kwargs.get('name')
            if not isinstance(module, str):
                return UNKNOWN
            if name == 'builtins.__import__':
                fromlist = args[3] if len(args) > 3 else kwargs.get('fromlist', ())
                if not fromlist:
                    module = module.split('.')[0]
            return Symbol('module', canonical(module))
        if name in ('operator.attrgetter', 'operator.itemgetter'):
            return Symbol(name.split('.')[-1], args=args)
        if name == 'operator.methodcaller':
            return Symbol('methodcaller', args[0], args[1:], attrs=dict(kwargs))
        if name == 'functools.partial':
            return Symbol('partial', args=args[1:], value=args[0], attrs={'keywords': dict(kwargs)})
        if name in ('functools.cache', 'functools.lru_cache'):
            if args and isinstance(args[0], Symbol):
                return args[0]
            return Symbol('decorator', name)
        if name == 'functools.wraps':
            return Symbol('decorator', name)
        if name == 'functools.update_wrapper':
            return args[0]
        if name == 'functools.partial.__setstate__':
            obj, state = (bound, args[0]) if bound is not None else args[:2]
            self.build(obj, state)
            return None
        if name == 'types.MethodType':
            return Symbol('method', value=args[0], args=(args[1],))
        if name in ('builtins.staticmethod', 'builtins.classmethod'):
            return Symbol('instance', name, value=args[0])
        if name in ('copyreg.__newobj__', 'copyreg.__newobj_ex__'):
            return self.new(args[0], seq(args[1]) if name.endswith('_ex__') else args[1:], args[2] if name.endswith('_ex__') else {})
        if name == 'copyreg._reconstructor':
            return self.new(args[0], ())
        if name == 'operator.call' or name == 'builtins.apply':
            if name == 'builtins.apply':
                return self.call(args[0], seq(args[1]) if len(args) > 1 else (), args[2] if len(args) > 2 else {})
            return self.call(args[0], args[1:], kwargs)
        if name in ('bdb.Bdb.runcall', 'pdb.Pdb.runcall', 'pdb.runcall',
                    'profile.Profile.runcall', 'cProfile.Profile.runcall', 'trace.Trace.runfunc'):
            values = args if bound is not None or name == 'pdb.runcall' else args[1:]
            return self.call(values[0], values[1:], kwargs)
        if name in ('torch._tensor._rebuild_from_type_v2', 'torch._tensor._rebuild_from_type'):
            return self.call(args[0], seq(args[2]))
        if name in ('numpy.frompyfunc', 'numpy.vectorize'):
            return Symbol('partial', value=args[0], attrs={'keywords': {}})
        if name in ('_thread.start_new_thread', 'threading._start_new_thread'):
            return self.call(args[0], seq(args[1]), args[2] if len(args) > 2 else {})
        if name == 'threading.Thread.start' and bound is not None:
            target = bound.attrs.get('target', bound.args[1] if len(bound.args) > 1 else None)
            if target is not None:
                return self.call(target, seq(bound.attrs.get('args', bound.args[3] if len(bound.args) > 3 else ())),
                                 bound.attrs.get('kwargs', bound.args[4] if len(bound.args) > 4 else {}))
            return None
        if name == 'operator.getitem':
            return self.getitem(args[0], args[1])
        if name == 'operator.setitem':
            self.setitem(*args)
            return None
        if name == 'builtins.vars':
            return self.attr(args[0], '__dict__') if args else Symbol('namespace', 'builtins')
        if name == 'builtins.globals' or name == 'builtins.locals':
            return {'__builtins__': Symbol('module', 'builtins')}
        if name in NESTED_LOADERS:
            return self.nested(name, args, kwargs, bound)
        if name in ('pickle.Unpickler', 'pickle._Unpickler'):
            return Symbol('instance', name, value=args[0] if args else UNKNOWN)
        if name in ('logging.config.dictConfig', 'logging.config.DictConfigurator.configure'):
            config = args[0] if args else (bound.args[0] if bound is not None and bound.args else {})
            self.logging_config(config)
            return None
        if name in ('logging.config.fileConfig', 'logging.config.listen'):
            self.finding('file-access' if name.endswith('fileConfig') else 'network', f'{self.op} {name}')
            return UNKNOWN
        if name in HOST_CLIENTS:
            host = args[0] if args else kwargs.get('host', '')
            if host or name.startswith(('imaplib.', 'poplib.')):
                self.finding('network', f'{self.op} {name} connects to host')
            return Symbol('instance', name, args=args)
        if name == 'timeit.Timer':
            # Construction compiles attacker source, even without timeit().
            if any(isinstance(x, str) and x for x in args[:2]) or any(isinstance(kwargs.get(k), str) for k in ('stmt', 'setup')):
                self.finding('code-execution', f'{self.op} {name} compiles source')
            return Symbol('instance', name, args=args)
        if name in ('marshal.loads', 'marshal.load'):
            if name == 'marshal.load' and args and not self.is_buffer(args[0]):
                self.finding('file-access', f'{self.op} {name}')
            return Symbol('code', 'marshal result')
        if name.startswith('pandas.') and name.rsplit('.', 1)[-1].startswith('read_'):
            source = args[0] if args else next(iter(kwargs.values()), UNKNOWN)
            if not self.is_buffer(source):
                self.finding('file-access', f'{self.op} {name} reads a path')
            return Symbol('instance', 'pandas.DataFrame')
        if name in ('gzip.GzipFile', 'gzip.open', 'bz2.BZ2File', 'bz2.open', 'lzma.LZMAFile', 'lzma.open'):
            source = kwargs.get('fileobj', args[0] if args else kwargs.get('filename'))
            if self.is_buffer(source):
                from modelgate.dispatch import decompress
                mode = args[1] if len(args) > 1 else kwargs.get('mode', 'rb')
                data = decompress(source.value, name.split('.')[0]) if 'r' in mode else b''
                return Symbol('instance', 'io.BytesIO', value=data, attrs={'position': 0})
        if name == 'zipfile.ZipFile' or name in ('tarfile.open', 'tarfile.TarFile.open'):
            source = kwargs.get('fileobj', args[0] if args else kwargs.get('file'))
            if self.is_buffer(source):
                return Symbol('instance', 'zipfile.ZipFile' if name.startswith('zipfile.') else 'tarfile.TarFile', value=source)
        if name.startswith('logging.') and name.endswith('FileHandler'):
            delay_index = 5 if 'Rotating' in name else 3
            delay = kwargs.get('delay', args[delay_index] if len(args) > delay_index else False)
            if delay:
                return Symbol('instance', name, args=args, attrs=dict(kwargs))
        if name.endswith(('._create_function', '._make_function', '._make_skeleton_function')) and name.startswith(('dill.', 'cloudpickle.')):
            self.finding('code-execution', f'{self.op} {name} materializes Python function')
            return Symbol('function')
        kind = CALLS.get(name)
        if kind:
            # A BytesIO's read/write is memory only; the entries here are file
            # or network APIs, not a broad match on method names.
            self.finding(kind, f'{self.op} {name}')
            return Symbol('function') if name in ('types.FunctionType', 'types.LambdaType') else UNKNOWN
        return self.pure_call(name, args, kwargs, bound)

    @staticmethod
    def is_buffer(value):
        return isinstance(value, Symbol) and value.name in ('io.BytesIO', 'io.StringIO')

    def nested(self, name, args, kwargs, bound):
        from modelgate import dispatch

        if bound is not None and name.startswith('pickle.'):
            source = bound.value
        else:
            source = args[0] if args else next((kwargs[k] for k in ('file', 'f', 'filepath_or_buffer', 'filename', 'data') if k in kwargs), UNKNOWN)
        byte_loader = name.endswith('.loads') or name == 'torch.storage._load_from_bytes'
        if self.is_buffer(source):
            data = source.value
            position = source.attrs.get('position', 0)
            if not isinstance(data, (bytes, bytearray, str)):
                raise AnalysisError(f'unknown buffer for {name}')
            data = data[position:]
            if isinstance(data, str):
                data = data.encode('latin1')
        elif byte_loader and isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        elif isinstance(source, (str, bytes)) or (isinstance(source, Symbol) and source.name.startswith('pathlib.')):
            self.finding('file-access', f'{self.op} {name} reads a path')
            return UNKNOWN
        else:
            raise AnalysisError(f'cannot statically resolve input to {name}')
        location = self.where + '->' + name
        raw_pickle = name.startswith(('pickle.', 'dill.', 'cloudpickle.'))
        if name.startswith(('numpy.',)):
            if name.endswith('read_array'):
                allow_pickle = args[1] if len(args) > 1 else kwargs.get('allow_pickle', False)
                if not data.startswith(b'\x93NUMPY'):
                    return UNKNOWN  # read_array rejects non-NPY input before loading anything
            else:
                allow_pickle = args[2] if len(args) > 2 else kwargs.get('allow_pickle', False)
            if not allow_pickle:
                return UNKNOWN  # object pickles are not interpreted in this mode
            raw_pickle = not (data.startswith(b'\x93NUMPY') or data.startswith(b'PK\x03\x04'))
        if name.startswith(('pandas.',)):
            compression = kwargs.get('compression', args[1] if len(args) > 1 else None)
            if isinstance(compression, dict): compression = compression.get('method')
            if compression not in (None, 'infer'):
                data = dispatch.decompress(data, {'xz': 'lzma'}.get(compression, compression))
            raw_pickle = True
        if name.startswith('joblib.'):
            for magic, compression in ((b'\x1f\x8b', 'gzip'), (b'BZh', 'bz2'), (b'\xfd7zXZ\x00', 'lzma')):
                if data.startswith(magic):
                    data = dispatch.decompress(data, compression)
                    break
            else:
                if len(data) >= 2 and data[0] & 15 == 8 and (data[0] * 256 + data[1]) % 31 == 0:
                    data = dispatch.decompress(data, 'zlib')
            raw_pickle = True
        if name.startswith('torch.') and not data.startswith(b'PK\x03\x04'):
            # The legacy loader unpickles the magic before checking it. If it
            # is an ordinary object, later streams are never reached.
            probe = Interpreter(location, self.depth + 1)
            try:
                first = probe.run(data, first_only=True)
            finally:
                self.findings.extend(probe.findings)
            if type(first) is not int or first != TORCH_MAGIC:
                return UNKNOWN
        if raw_pickle:
            nested = Interpreter(location, self.depth + 1)
            try:
                result = nested.run(data, first_only=True)
            finally:
                self.findings.extend(nested.findings)
            if self.is_buffer(source):
                source.attrs['position'] = position + nested.consumed
            return result
        # Remaining NumPy and torch inputs are their native containers.
        self.findings.extend(dispatch.scan_bytes(data, location, depth=self.depth + 1))
        return UNKNOWN

    def logging_config(self, config):
        # dictConfig is harmless for ordinary logging configurations. Factories
        # under () and handler constructors are actual calls, with kwargs.
        if not isinstance(config, dict):
            raise AnalysisError('unresolved logging configuration')
        for section in ('formatters', 'filters', 'handlers'):
            entries = config.get(section, {})
            if not isinstance(entries, dict):
                continue
            for spec in entries.values():
                if not isinstance(spec, dict):
                    continue
                factory = spec.get('()', spec.get('class') if section == 'handlers' else None)
                if factory is None:
                    continue
                if isinstance(factory, str):
                    factory = Symbol('global', canonical(factory.removeprefix('ext://')))
                kwargs = {k: v for k, v in spec.items() if k not in ('()', 'class', 'level', 'formatter', 'filters', '.')}
                self.call(factory, (), kwargs)

    def pure_call(self, name, args, kwargs, bound):
        """Small, explicit transfer functions for inert standard-library APIs."""
        if name in ('io.BytesIO', 'io.StringIO'):
            return Symbol('instance', name, value=args[0] if args else (b'' if name.endswith('BytesIO') else ''), attrs={'position': 0})
        if bound is not None:
            method = name.rsplit('.', 1)[-1]
            if isinstance(bound, Symbol) and bound.name in ('zipfile.ZipFile', 'tarfile.TarFile') and self.is_buffer(bound.value):
                if method in ('read', 'open', 'extractfile'):
                    import tarfile
                    import zipfile
                    source = io.BytesIO(bound.value.value)
                    if bound.name == 'zipfile.ZipFile':
                        with zipfile.ZipFile(source) as zf:
                            result = zf.read(args[0])
                    else:
                        with tarfile.open(fileobj=source) as tf:
                            result = tf.extractfile(args[0]).read(MAX_VALUE + 1)
                    self.bounded(result)
                    return result if method == 'read' else Symbol('instance', 'io.BytesIO', value=result, attrs={'position': 0})
            if self.is_buffer(bound):
                value = bound.value
                pos = bound.attrs.get('position', 0)
                if method in ('getvalue', 'getbuffer'):
                    return value
                if method == 'tell':
                    return pos
                if method == 'seek':
                    whence = args[1] if len(args) > 1 else 0
                    bound.attrs['position'] = args[0] + (pos if whence == 1 else len(value) if whence == 2 else 0)
                    return bound.attrs['position']
                if method in ('read', 'read1', 'readline'):
                    end = min(len(value), pos + args[0]) if args and args[0] >= 0 else len(value)
                    if method == 'readline':
                        sep = b'\n' if isinstance(value, bytes) else '\n'
                        found = value.find(sep, pos, end)
                        if found >= 0: end = found + 1
                    bound.attrs['position'] = end
                    return value[pos:end]
                if method == 'write':
                    value = value[:pos] + args[0] + value[pos + len(args[0]):]
                    bound.value = value
                    bound.attrs['position'] = pos + len(args[0])
                    return len(args[0])
            if method == '__setstate__':
                self.build(bound, args[0])
                return None
            if method == '__getattribute__':
                return self.attr(bound, args[0])
            if method == '__getitem__':
                return self.getitem(bound, args[0])
            if method == '__setitem__':
                self.setitem(bound, args[0], args[1])
                return None
            if isinstance(bound, dict):
                if method == 'get': return bound.get(*args)
                if method == 'pop': return bound.pop(*args)
                if method == 'update': bound.update(*args, **kwargs); return None
                if method in ('items', 'keys', 'values'): return list(getattr(bound, method)())
            if isinstance(bound, list):
                if method == 'append': bound.append(args[0]); return None
                if method == 'extend': bound.extend(args[0]); return None
                if method == 'pop': return bound.pop(*args)
            if isinstance(bound, (str, bytes, bytearray)) and method in (
                'encode', 'decode', 'join', 'replace', 'format', 'strip', 'lstrip', 'rstrip',
                'split', 'rsplit', 'lower', 'upper', 'capitalize', 'removeprefix', 'removesuffix'):
                # No format strings may traverse symbolic objects: only exact
                # primitive literals are ever passed to Python's methods.
                if all(self.literal(x) for x in args) and all(self.literal(x) for x in kwargs.values()):
                    return self.bounded(getattr(bound, method)(*args, **kwargs))
        if name in ('bytes.fromhex', 'builtins.bytes.fromhex', 'builtins.bytearray.fromhex'):
            return bytes.fromhex(args[0])
        if bound is None and name.startswith('builtins.') and name.count('.') == 2:
            cls, method = name.split('.')[1:]
            if args and cls in ('str', 'bytes', 'bytearray', 'dict', 'list', 'tuple'):
                return self.pure_call(name, args[1:], kwargs, args[0])
        if name in ('_codecs.encode', 'codecs.encode', '_codecs.decode', 'codecs.decode'):
            if not args or not isinstance(args[0], (str, bytes, bytearray)):
                return UNKNOWN
            encoding = args[1] if len(args) > 1 else kwargs.get('encoding', 'utf-8')
            errors = args[2] if len(args) > 2 else kwargs.get('errors', 'strict')
            # Restrict codecs to data transforms, never arbitrary codec lookup.
            norm = encoding.lower().replace('_', '-').replace(' ', '')
            if norm in ('base64', 'base64-codec'):
                return base64.b64decode(args[0]) if name.endswith('decode') else base64.b64encode(args[0])
            if norm in ('hex', 'hex-codec'):
                return binascii.unhexlify(args[0]) if name.endswith('decode') else binascii.hexlify(args[0])
            if norm not in ('utf-8', 'utf8', 'latin1', 'latin-1', 'iso-8859-1', 'ascii', 'utf-16', 'utf-16-le', 'utf-16-be', 'unicode-escape', 'raw-unicode-escape'):
                raise AnalysisError('unsupported codec')
            return getattr(args[0], 'encode' if name.endswith('encode') else 'decode')(encoding, errors)
        if name.startswith('base64.') and name.rsplit('.', 1)[-1] in (
            'b64decode', 'standard_b64decode', 'urlsafe_b64decode', 'b32decode', 'b16decode', 'b85decode', 'a85decode', 'decodebytes',
            'b64encode', 'standard_b64encode', 'urlsafe_b64encode', 'b32encode', 'b16encode', 'b85encode', 'a85encode', 'encodebytes'):
            return self.bounded(getattr(base64, name.split('.')[-1])(*args, **kwargs))
        if name in ('binascii.a2b_base64', 'binascii.unhexlify', 'binascii.a2b_hex', 'binascii.b2a_base64', 'binascii.hexlify'):
            return self.bounded(getattr(binascii, name.split('.')[-1])(*args, **kwargs))
        if name in ('gzip.decompress', 'bz2.decompress', 'lzma.decompress', 'zlib.decompress'):
            from modelgate.dispatch import decompress
            return decompress(args[0], name.split('.')[0])
        if name == 'bytes.fromhex' or name == 'builtins.bytes.fromhex':
            return bytes.fromhex(args[0])
        if name in ('builtins.tuple', 'builtins.list', 'builtins.set', 'builtins.frozenset'):
            values = self.materialize(args[0]) if args else []
            factory = {'builtins.tuple': tuple, 'builtins.list': list, 'builtins.set': set, 'builtins.frozenset': frozenset}[name]
            return factory(values)
        if name in ('builtins.dict', 'collections.OrderedDict', 'collections.defaultdict', 'collections.Counter'):
            if name == 'collections.defaultdict':
                args = args[1:]
            if not args: return dict(kwargs)
            if isinstance(args[0], dict): return dict(args[0], **kwargs)
            return dict(self.materialize(args[0]), **kwargs)
        if name in ('builtins.str', 'builtins.bytes', 'builtins.bytearray', 'builtins.int', 'builtins.float', 'builtins.bool', 'builtins.complex'):
            if all(self.literal(x) for x in args):
                if args and type(args[0]) is int and name in ('builtins.bytes', 'builtins.bytearray') and args[0] > MAX_VALUE:
                    raise AnalysisError('byte allocation limit')
                factory = {'builtins.str': str, 'builtins.bytes': bytes, 'builtins.bytearray': bytearray,
                           'builtins.int': int, 'builtins.float': float, 'builtins.bool': bool, 'builtins.complex': complex}[name]
                return self.bounded(factory(*args, **kwargs))
            return UNKNOWN
        if name == 'builtins.slice':
            return slice(*args)
        if name == 'builtins.range':
            result = range(*args)
            if len(result) > MAX_STEPS: raise AnalysisError('range limit')
            return list(result)
        if name in ('builtins.len', 'operator.length_hint') and args:
            return len(args[0]) if isinstance(args[0], (str, bytes, list, tuple, dict, set)) else UNKNOWN
        if name == 'builtins.chr': return chr(args[0])
        if name == 'builtins.ord': return ord(args[0])
        if name == 'builtins.type':
            if len(args) == 1: return Symbol('global', typename(args[0]))
            return Symbol('class', 'dynamic.' + args[0], args=seq(args[1]), attrs=dict(args[2]))
        if name in ('os.fspath', 'os.fsencode', 'os.fsdecode'):
            value = args[0]
            if isinstance(value, Symbol) and value.name.startswith('pathlib.'):
                value = value.value
            if name == 'os.fsencode' and isinstance(value, str): return value.encode()
            if name == 'os.fsdecode' and isinstance(value, bytes): return value.decode()
            return value
        if name.startswith('pathlib.') and name.split('.')[-1] in ('Path', 'PosixPath', 'WindowsPath', 'PurePath', 'PurePosixPath', 'PureWindowsPath'):
            return Symbol('instance', 'pathlib.Path', args=args, value='/'.join(str(x) for x in args))
        if name == 'builtins.map':
            return Symbol('iterator', 'map', args=args)
        if name == 'builtins.filter':
            return Symbol('iterator', 'filter', args=args)
        if name in ('builtins.iter', 'builtins.enumerate', 'builtins.zip', 'itertools.starmap'):
            return Symbol('iterator', name, args=args)
        if name in ('itertools.chain', 'itertools.chain.from_iterable', 'itertools.islice', 'itertools.repeat'):
            return Symbol('iterator', name, args=args)
        if name in ('builtins.next',):
            values = self.materialize(args[0], limit=1)
            return values[0] if values else (args[1] if len(args) > 1 else UNKNOWN)
        if name in ('builtins.sorted', 'builtins.min', 'builtins.max'):
            values = self.materialize(args[0]) if len(args) == 1 else list(args)
            if 'key' in kwargs and kwargs['key'] is not None:
                for value in values: self.call(kwargs['key'], (value,))
            return values if name == 'builtins.sorted' else (values[0] if values else UNKNOWN)
        if name in ('functools.reduce',):
            values = self.materialize(args[1])
            if len(args) > 2: values.insert(0, args[2])
            result = values[0] if values else UNKNOWN
            for item in values[1:]: result = self.call(args[0], (result, item))
            return result
        if name in ('builtins.sum', 'builtins.any', 'builtins.all'):
            values = self.materialize(args[0])
            if all(self.literal(v) for v in values):
                return {'builtins.sum': sum, 'builtins.any': any, 'builtins.all': all}[name](values)
            return UNKNOWN
        if name.startswith('operator.') and name.split('.')[-1] in (
            'add', 'concat', 'mul', 'sub', 'truediv', 'floordiv', 'mod', 'neg', 'pos', 'not_', 'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'index', 'and_', 'or_', 'xor', 'contains'):
            if all(self.literal(x) for x in args):
                if name.endswith('.mul'):
                    if any(type(x) is int and abs(x) > MAX_VALUE for x in args): raise AnalysisError('multiplication limit')
                    if len(args) == 2 and isinstance(args[0], (str, bytes, tuple, list)) and type(args[1]) is int and len(args[0]) * args[1] > MAX_VALUE:
                        raise AnalysisError('multiplication limit')
                return self.bounded(getattr(operator, name.split('.')[-1])(*args))
        if name == 'ast.literal_eval':
            return ast.literal_eval(args[0])
        if name in ('json.loads', 'json.decoder.JSONDecoder.decode'):
            import json
            return json.loads(args[0])
        if name == 'numpy.dtype':
            return Symbol('instance', name, args=args, value=args[0] if args else None)
        if name in ('numpy.core.multiarray._reconstruct', 'numpy.core.multiarray.scalar', 'numpy.core.numeric._frombuffer'):
            return Symbol('instance', 'numpy.ndarray', args=args)
        # Known inert reconstructions remain symbolic, including their class
        # identity: HTTPConnection(...).request must still be recognized.
        if (name in INERT_GLOBALS or name.rsplit('.', 1)[0] in INERT_MODULES
                or name.startswith(('numpy.', 'scipy.', 'sklearn.', 'torch.', 'pandas.',
                                    'operator.', 'functools.', 'os.path.', 'pathlib.Pure'))
                or name in ('os.getenv', 'os.getpid', 'os.getppid', 'os.uname',
                            'time.time', 'time.monotonic', 'time.sleep')):
            return Symbol('instance', name, args=args, attrs=dict(kwargs))
        raise AnalysisError(f'no static call model for {name}')

    def materialize(self, value, limit=100000):
        if isinstance(value, (list, tuple, set, frozenset, dict, str, bytes, range)):
            if len(value) > MAX_STEPS: raise AnalysisError('iteration limit')
            return list(value)[:limit]
        if isinstance(value, Symbol) and value.kind == 'iterator':
            args = value.args
            if value.name in ('map', 'itertools.starmap'):
                inputs = [self.materialize(v, limit) for v in args[1:]]
                result = []
                for items in zip(*inputs):
                    callargs = seq(items[0]) if value.name == 'itertools.starmap' else items
                    result.append(self.call(args[0], callargs))
                return result
            if value.name == 'filter':
                values = self.materialize(args[1], limit)
                if args[0] is not None:
                    for item in values: self.call(args[0], (item,))
                return values
            if value.name == 'builtins.iter':
                if len(args) == 1:
                    return self.materialize(args[0], limit)
                result = []
                for _ in range(min(limit, 1000)):
                    item = self.call(args[0], ())
                    if self.findings or item == args[1]: break
                    result.append(item)
                return result
            if value.name == 'builtins.enumerate': return list(enumerate(self.materialize(args[0], limit), args[1] if len(args) > 1 else 0))
            if value.name == 'builtins.zip': return list(zip(*(self.materialize(v, limit) for v in args)))
            if value.name in ('itertools.chain', 'itertools.chain.from_iterable'):
                inputs = args if value.name == 'itertools.chain' else self.materialize(args[0], limit)
                result = []
                for item in inputs:
                    result.extend(self.materialize(item, limit - len(result)))
                    if len(result) >= limit: break
                return result
            if value.name == 'itertools.islice':
                selection = slice(*args[1:])
                return self.materialize(args[0], min(limit, selection.stop) if selection.stop is not None else limit)[selection]
            if value.name == 'itertools.repeat':
                return [args[0]] * min(limit, args[1] if len(args) > 1 else limit)
        return []

    @staticmethod
    def literal(value, depth=0):
        if depth > 20: return False
        if value is None or type(value) in (str, bytes, bytearray, bool, int, float, complex): return True
        if type(value) in (list, tuple, set, frozenset):
            return all(Interpreter.literal(x, depth + 1) for x in value)
        if type(value) is dict:
            return all(Interpreter.literal(k, depth + 1) and Interpreter.literal(v, depth + 1) for k, v in value.items())
        return False

    @staticmethod
    def bounded(value):
        if isinstance(value, (str, bytes, bytearray, list, tuple)) and len(value) > MAX_VALUE:
            raise AnalysisError('symbolic value size limit')
        return value

    def joblib_payload(self, obj, stream):
        if not isinstance(obj, Symbol) or obj.name not in ('joblib.numpy_pickle.NumpyArrayWrapper', 'joblib.numpy_pickle.NDArrayWrapper'):
            return
        if obj.name.endswith('.NDArrayWrapper'):
            self.finding('file-access', 'joblib NDArrayWrapper reads an external array file')
            return
        import numpy as np
        symbolic_dtype = obj.attrs.get('dtype')
        dtype = np.dtype(symbolic_dtype.value if isinstance(symbolic_dtype, Symbol) else symbolic_dtype)
        hasobject = dtype.hasobject
        # Structured dtype state can turn a plain void descriptor into an
        # object-containing dtype. Do not skip its following pickle as bytes.
        todo, seen = [symbolic_dtype], set()
        while todo:
            item = todo.pop()
            if id(item) in seen: continue
            seen.add(id(item))
            if isinstance(item, Symbol) and item.name == 'numpy.dtype':
                hasobject |= np.dtype(item.value).hasobject
                todo.append(item.attrs.get('_state'))
            elif isinstance(item, (tuple, list)):
                todo.extend(item)
            elif isinstance(item, dict):
                todo.extend(item.values())
        if hasobject:
            # joblib stores object arrays as an independent pickle.
            start = stream.tell()
            inner = Interpreter(self.where + '->joblib array', self.depth + 1)
            try:
                inner.run(stream.getbuffer()[start:].tobytes(), first_only=True)
            finally:
                self.findings.extend(inner.findings)
            stream.seek(start + inner.consumed)
        else:
            if obj.attrs.get('numpy_array_alignment_bytes') is not None:
                padding = stream.read(1)
                if not padding: raise AnalysisError('truncated joblib alignment')
                stream.seek(padding[0], 1)
            count = math.prod(obj.attrs.get('shape', ()))
            size = count * dtype.itemsize
            if size < 0 or stream.tell() + size > len(stream.getbuffer()):
                raise AnalysisError('truncated joblib array')
            stream.seek(size, 1)


def scan_pickle(data: bytes, location: str, *, depth=0, first_only=False) -> list[Finding]:
    interpreter = Interpreter(location, depth)
    try:
        interpreter.run(data, first_only=first_only)
    except Exception as exc:
        interpreter.findings.append(Finding('unsupported', interpreter.where,
                                            f'cannot analyse pickle: {type(exc).__name__}: {exc}'))
    return interpreter.findings
