"""Capability-aware abstract execution of calls made by an unpickler.

Only literal operations on our own inert values are performed. Globals are
symbolic: this module never imports an artifact-specified module or invokes
an artifact-specified Python callable. Call summaries preserve references and
mutations, and enter embedded serialization through the same static scanner.
"""
from __future__ import annotations

import _compat_pickle
import base64
import binascii
import codecs
import io
import itertools
import json
import math
import marshal
import operator
import os
import posixpath
import struct
import types

from modelgate.limits import Budget, MAX_BYTES, MAX_ITEMS, LimitExceeded
from modelgate.policy import CALL_EFFECTS
from modelgate.report import Finding
from modelgate.pickle_values import (
    Ref, Module, Obj, Partial, Bound, Adapter, Lazy, Buffer, Namespace,
    Unknown, UNKNOWN, LoadError, Unresolved, EndIteration, FindingRaised,
)

ALIASES = {
    'posix': 'os', 'nt': 'os', '_pickle': 'pickle', '_io': 'io',
    '_operator': 'operator', '_functools': 'functools', '_socket': 'socket',
    '_thread': '_thread', 'numpy.core': 'numpy._core',
    'numpy.core.multiarray': 'numpy._core.multiarray',
    'numpy.core.numeric': 'numpy._core.numeric',
    'pathlib.PosixPath': 'pathlib.Path', 'pathlib.WindowsPath': 'pathlib.Path',
    'socket.SocketType': 'socket.socket',
    'os.path': 'posixpath', 'builtins.function': 'types.FunctionType',
    'builtins.code': 'types.CodeType', 'builtins.module': 'types.ModuleType',
    'builtins.method': 'types.MethodType',
    'importlib._bootstrap.__import__': 'builtins.__import__',
    'pickle._Unpickler': 'pickle.Unpickler',
}

# Generated from trusted stdlib metadata, not from an artifact. This resolves
# aliases such as tempfile._os and functions re-exported by other modules
# without importing any of those modules while scanning.
with open(os.path.join(os.path.dirname(__file__), 'stdlib_symbols.json'), encoding='utf-8') as _fh:
    STDLIB_SYMBOLS = json.load(_fh)
with open(os.path.join(os.path.dirname(__file__), 'stdlib_signatures.json'), encoding='utf-8') as _fh:
    CALL_SIGNATURES = json.load(_fh)
# References exposed by other standard modules. They retain module identity,
# so rebinding through any alias affects every later resolution.
MODULE_ATTRS = {
    'os.path': 'posixpath',
    'shutil.os': 'os', 'shutil.sys': 'sys', 'shutil.stat': 'stat',
    'subprocess.os': 'os', 'subprocess.sys': 'sys', 'subprocess.io': 'io',
    'subprocess.signal': 'signal', 'subprocess.time': 'time',
    'pickle.sys': 'sys', 'pickle.io': 'io', 'pickle.codecs': 'codecs',
    'pickle.re': 're', 'pickle.struct': 'struct',
    'importlib.sys': 'sys', 'importlib.importlib': 'importlib',
    'pkgutil.os': 'os', 'pkgutil.sys': 'sys', 'pkgutil.importlib': 'importlib',
    'pathlib.os': 'os', 'pathlib.io': 'io', 'pathlib.sys': 'sys',
    'posixpath.os': 'os', 'posixpath.sys': 'sys', 'ntpath.os': 'os',
    'tempfile.os': 'os', 'tempfile._os': 'os', 'tempfile._io': 'io',
    'tempfile._shutil': 'shutil', 'tempfile._sys': 'sys',
    'pty.os': 'os', 'pty.sys': 'sys', 'pydoc.os': 'os', 'pydoc.sys': 'sys',
    'pydoc.subprocess': 'subprocess', 'pydoc.builtins': 'builtins',
    'bdb.os': 'os', 'bdb.sys': 'sys', 'pdb.os': 'os', 'pdb.sys': 'sys',
    'code.sys': 'sys', 'code.traceback': 'traceback',
    'runpy.sys': 'sys', 'runpy.os': 'os', 'runpy.io': 'io',
    'timeit.sys': 'sys', 'timeit.itertools': 'itertools',
    'logging.os': 'os', 'logging.sys': 'sys', 'logging.io': 'io',
    'logging.config.os': 'os', 'logging.config.sys': 'sys',
    'logging.config.logging': 'logging',
    'urllib.request.os': 'os', 'urllib.request.sys': 'sys',
    'urllib.request.socket': 'socket', 'urllib.request.http': 'http',
    'urllib.request.io': 'io', 'urllib.request.ssl': 'ssl',
    'http.client.socket': 'socket', 'http.client.ssl': 'ssl', 'http.client.io': 'io',
    'socket.os': 'os', 'socket.sys': 'sys', 'ssl.socket': 'socket',
    'ftplib.socket': 'socket', 'smtplib.socket': 'socket',
    'webbrowser.os': 'os', 'webbrowser.subprocess': 'subprocess',
    'threading._os': 'os', 'concurrent.futures.thread.threading': 'threading',
    'torch.serialization.pickle': 'pickle', 'torch.serialization.io': 'io',
    'torch.serialization.os': 'os', 'torch.storage.io': 'io',
    'torch.storage.torch': 'torch', 'torch.serialization.torch': 'torch',
    'numpy.lib.format.pickle': 'pickle', 'numpy.lib.format.numpy': 'numpy',
    'numpy.lib.npyio.pickle': 'pickle', 'joblib.numpy_pickle.pickle': 'pickle',
    'joblib.numpy_pickle.io': 'io', 'pandas.io.pickle.pickle': 'pickle',
}
MODULES = set('''builtins os sys posixpath ntpath shutil subprocess pickle io operator functools
    importlib importlib.util importlib.machinery pkgutil numpy numpy._core numpy._core.multiarray
    numpy._core.numeric numpy.lib numpy.lib.format numpy.lib.npyio numpy.testing
    numpy.testing._private numpy.testing._private.utils torch torch._utils torch.storage
    torch.serialization joblib joblib.numpy_pickle pandas pandas.io pandas.io.pickle
    socket urllib urllib.request urllib.parse http http.client ftplib smtplib ssl
    xmlrpc xmlrpc.client threading _thread concurrent concurrent.futures
    concurrent.futures.thread concurrent.futures.process multiprocessing multiprocessing.pool
    multiprocessing.process multiprocessing.context pathlib copy copyreg types marshal
    base64 binascii codecs _codecs zlib gzip bz2 lzma itertools collections
    collections.abc logging logging.config logging.handlers code bdb pdb timeit runpy
    pty pydoc tempfile glob time math re struct random ast json csv decimal fractions
    datetime array weakref contextlib traceback signal stat warnings enum inspect
    requests requests.api requests.sessions httpx ctypes ctypes.util sqlite3 shelve dbm
    imaplib poplib nntplib telnetlib webbrowser cloudpickle cloudpickle.cloudpickle dill dill._dill
    _posixsubprocess _winapi'''.split())

CLASS_NAMES = set('''builtins.object builtins.type builtins.str builtins.bytes builtins.bytearray
    builtins.int builtins.float builtins.bool builtins.complex builtins.list builtins.tuple
    builtins.dict builtins.set builtins.frozenset builtins.range builtins.slice builtins.map
    builtins.filter builtins.zip builtins.enumerate builtins.reversed
    functools.partial functools.partialmethod functools.cached_property functools._lru_cache_wrapper
    operator.attrgetter operator.itemgetter operator.methodcaller
    collections.defaultdict collections.deque collections.OrderedDict collections.Counter
    io.BytesIO io.StringIO io.FileIO io.TextIOWrapper io.BufferedReader io.BufferedWriter
    socket.socket pathlib.Path pathlib.PosixPath pathlib.WindowsPath pathlib.PurePath
    pathlib.PurePosixPath pathlib.PureWindowsPath numpy.dtype numpy.ndarray numpy.memmap
    types.FunctionType types.CodeType types.LambdaType types.ModuleType types.MethodType
    types.SimpleNamespace types.MappingProxyType array.array'''.split())

NESTED = {
    'pickle.loads': 'pickle-bytes', 'pickle.load': 'pickle-file',
    'pickle.Unpickler.load': 'pickle-file',
    'torch.load': 'torch', 'torch.serialization.load': 'torch',
    'torch.storage._load_from_bytes': 'torch-bytes',
    'numpy.load': 'numpy', 'numpy.lib.npyio.load': 'numpy',
    'numpy.lib._npyio_impl.load': 'numpy', 'numpy.lib.format.read_array': 'numpy',
    'joblib.load': 'joblib', 'joblib.numpy_pickle.load': 'joblib',
    'pandas.read_pickle': 'pandas', 'pandas.io.pickle.read_pickle': 'pandas',
    'cloudpickle.loads': 'pickle-bytes', 'cloudpickle.load': 'pickle-file',
    'dill.loads': 'pickle-bytes', 'dill.load': 'pickle-file',
}


def canonical(name):
    if name in ALIASES:
        return ALIASES[name]
    first, sep, tail = name.partition('.')
    if first in ALIASES:
        name = ALIASES[first] + (sep + tail if sep else '')
    for old, new in (('numpy.core.', 'numpy._core.'),
                     ('pathlib.PosixPath.', 'pathlib.Path.'),
                     ('pathlib.WindowsPath.', 'pathlib.Path.'),
                     ('socket.SocketType.', 'socket.socket.')):
        if name.startswith(old):
            return new + name[len(old):]
    for old, new in ALIASES.items():
        if '.' in old and name.startswith(old + '.'):
            return new + name[len(old):]
    return name


def literal(value):
    if type(value) in (str, bytes, bytearray, int, float, bool, complex, type(None)):
        return True
    if type(value) in (tuple, list):
        return all(literal(x) for x in value)
    return False


class VM:
    def __init__(self, budget=None):
        self.budget = budget or Budget()
        self.modules = {}
        self.class_attrs = {}
        self.extension_registry = {}
        self.inverted_registry = {}
        self.extension_cache = {}
        self.copy_reducers = {}
        self.findings = []
        self.location = ''
        self.depth = 0
        self.call_depth = 0
        self.artifact_paths = set()

    def finding(self, kind, detail):
        self.findings.append(Finding(kind, self.location[:2000], detail[:4000]))
        raise FindingRaised()

    def module(self, name):
        name = canonical(name)
        if name not in self.modules:
            self.modules[name] = Module(name)
        return self.modules[name]

    def resolve(self, module, name, protocol=4):
        if not isinstance(module, str) or not isinstance(name, str):
            raise LoadError('GLOBAL names must be strings')
        if protocol < 3:
            if (module, name) in _compat_pickle.NAME_MAPPING:
                module, name = _compat_pickle.NAME_MAPPING[module, name]
            else:
                module = _compat_pickle.IMPORT_MAPPING.get(module, module)
        if protocol < 4 and '.' in name:
            raise LoadError('dotted GLOBAL is unavailable below protocol 4')
        result = self.module(module)
        for part in name.split('.'):
            result = self.attr(result, part)
        return result

    def resolve_name(self, name):
        if not isinstance(name, str):
            raise LoadError('name is not a string')
        if ':' in name:
            module, attrs = name.split(':', 1)
            obj = self.module(module)
            for part in attrs.split('.') if attrs else ():
                obj = self.attr(obj, part)
            return obj
        parts = name.split('.')
        module_len = 1
        for i in range(1, len(parts) + 1):
            if canonical('.'.join(parts[:i])) in MODULES:
                module_len = i
        obj = self.module('.'.join(parts[:module_len]))
        for part in parts[module_len:]:
            obj = self.attr(obj, part)
        return obj

    def attr(self, obj, name, default=UNKNOWN):
        self.budget.tick()
        if not isinstance(name, str):
            if isinstance(name, (Unknown, Obj)):
                raise Unresolved('cannot resolve attribute name')
            raise LoadError('attribute name is not a string')
        if name in ('__getattribute__', '__setattr__'):
            return Bound(obj, name)
        if isinstance(obj, Module):
            if name in obj.attrs:
                return obj.attrs[name]
            if name == '__dict__':
                return Namespace(obj)
            if name == '__name__':
                return obj.name
            if name == '__builtins__':
                return Namespace(self.module('builtins'))
            if obj.name == 'sys' and name == 'modules':
                return Namespace(self)
            if obj.name in ('copyreg', 'pickle'):
                registries = {'_extension_registry': self.extension_registry,
                              '_inverted_registry': self.inverted_registry,
                              '_extension_cache': self.extension_cache}
                if name in registries:
                    return registries[name]
            full = obj.name + '.' + name
            if full in MODULE_ATTRS:
                return self.module(MODULE_ATTRS[full])
            if full in MODULES:
                return self.module(full)
            if full in CALL_EFFECTS or full in NESTED:
                return Ref(full)
            entry = STDLIB_SYMBOLS.get(obj.name, {}).get(name)
            if entry:
                kind, value = entry
                if kind == 'module':
                    return self.module(value)
                if kind == 'ref':
                    return Ref(canonical(value))
                if kind == 'literal':
                    return value
            if full in ('os._exit', 'sys.exit', 'builtins.exit', 'builtins.quit'):
                return Ref(full)
            return Ref(canonical(full))
        if isinstance(obj, Ref):
            if name in self.class_attrs.get(obj.name, {}):
                return self.class_attrs[obj.name][name]
            if name == '__call__':
                return obj
            if name == '__module__':
                return obj.name.rsplit('.', 1)[0]
            if name in ('__name__', '__qualname__'):
                return obj.name.rsplit('.', 1)[-1]
            if name == '__globals__':
                return Namespace(self.module(obj.name.rsplit('.', 1)[0]))
            if name == '__dict__':
                return Namespace(obj)
            return Ref(canonical(obj.name + '.' + name))
        if isinstance(obj, Partial):
            if name in obj.attrs:
                return obj.attrs[name]
            if name in ('func', 'args', 'keywords'):
                return {'func': obj.fn, 'args': obj.args, 'keywords': obj.kw}[name]
            if name == '__dict__':
                return obj.attrs
            if name in ('__call__', '__wrapped__'):
                return obj
            return Bound(obj, name)
        if isinstance(obj, Obj):
            if name in obj.attrs:
                return obj.attrs[name]
            if name in self.class_attrs.get(obj.cls, {}):
                return self.class_attrs[obj.cls][name]
            if name == '__dict__':
                return obj.attrs
            if name == '__class__':
                return Ref(obj.cls)
            return Bound(obj, name)
        if isinstance(obj, (Buffer, Lazy, Adapter, Namespace)):
            return Bound(obj, name)
        if name == '__class__':
            return Ref('builtins.' + type(obj).__name__)
        if isinstance(obj, (str, bytes, bytearray, list, tuple, dict, set, frozenset, int, float, range)):
            return Bound(obj, name)
        return default

    def item(self, obj, key, default=UNKNOWN):
        if isinstance(obj, Obj) and obj.cls == 'numpy.lib.npyio.NpzFile':
            import zipfile
            with zipfile.ZipFile(io.BytesIO(obj.args[0])) as zf:
                name = key if key in zf.namelist() else str(key) + '.npy'
                if name not in zf.namelist():
                    raise LoadError('NPZ member not found')
                self.budget.add_bytes(zf.getinfo(name).file_size)
                data = zf.read(name)
            if data.startswith(b'\x93NUMPY'):
                return self.nested('numpy.load', (Buffer(data),),
                                   {'allow_pickle': obj.attrs.get('allow_pickle', False)})
            return data
        if isinstance(obj, Namespace):
            if obj.owner is self:
                return self.module(key) if isinstance(key, str) else default
            return self.attr(obj.owner, key, default)
        if isinstance(obj, (dict, list, tuple, str, bytes, bytearray, range)):
            try:
                return obj[key]
            except (KeyError, IndexError, TypeError):
                if default is not UNKNOWN:
                    return default
                raise LoadError('invalid subscript')
        return UNKNOWN

    def setitem(self, obj, key, value):
        if isinstance(obj, Namespace):
            if obj.owner is self:
                self.modules[key] = value
            elif hasattr(obj.owner, 'attrs'):
                obj.owner.attrs[key] = value
            elif isinstance(obj.owner, Ref):
                self.class_attrs.setdefault(obj.owner.name, {})[key] = value
            else:
                raise Unresolved('cannot assign to this namespace')
        elif isinstance(obj, (dict, list)):
            obj[key] = value
        elif isinstance(obj, Obj):
            obj.attrs.setdefault('_items', {})[key] = value
        else:
            raise LoadError('object does not support item assignment')

    def is_class(self, obj):
        return isinstance(obj, Ref) and (obj.name in CLASS_NAMES
            or obj.name.rsplit('.', 1)[-1][:1].isupper()
            or obj.name.startswith('itertools.'))

    def new(self, cls, args, kw=None):
        if isinstance(cls, Unknown) or isinstance(args, Unknown) or isinstance(kw, Unknown):
            raise Unresolved('cannot resolve NEWOBJ constructor or arguments')
        if not self.is_class(cls) or not isinstance(args, tuple) or (kw is not None and not isinstance(kw, dict)):
            raise LoadError('NEWOBJ requires a class, tuple and keyword dict')
        name = cls.name
        if name in ('functools.partial', 'types.FunctionType', 'types.LambdaType', 'types.CodeType',
                    'builtins.bytes', 'builtins.str', 'builtins.bytearray', 'builtins.tuple',
                    'operator.attrgetter', 'operator.itemgetter', 'operator.methodcaller'):
            return self.call(cls, args, kw or {})
        if name == 'numpy.memmap':
            return self.call(cls, args, kw or {})
        return Obj(name, args)

    def build(self, obj, state):
        if isinstance(obj, Partial):
            if not isinstance(state, tuple) or len(state) != 4:
                raise LoadError('invalid partial state')
            fn, args, kw, attrs = state
            if not isinstance(args, tuple) or not isinstance(kw, (dict, type(None))):
                raise LoadError('invalid partial arguments')
            obj.fn, obj.args, obj.kw = fn, args, kw or {}
            obj.attrs = attrs or {}
            return
        if isinstance(obj, (Obj, Module)):
            setter = obj.attrs.get('__setstate__')
            if setter is None and isinstance(obj, Obj):
                setter = self.class_attrs.get(obj.cls, {}).get('__setstate__')
            if setter is not None:
                self.call(setter, (state,))
                return
        if isinstance(obj, Obj):
            if obj.cls == 'numpy.dtype':
                obj.state = state
                return
            if obj.cls in ('io.BytesIO', 'io.StringIO') and isinstance(state, tuple):
                obj.attrs['_buffer'] = Buffer(state[0], state[1])
                return
            obj.state = state
        states = state if isinstance(state, tuple) and len(state) == 2 else (state,)
        for part in states:
            if isinstance(part, dict):
                if isinstance(obj, (Module, Obj, Partial)):
                    obj.attrs.update(part)
                # OrderedDict's instance dictionary is irrelevant to its items.
                elif isinstance(obj, dict):
                    pass

    def call(self, fn, args, kw=None):
        self.budget.tick()
        if not isinstance(args, tuple):
            if isinstance(args, (Unknown, Obj)):
                raise Unresolved('cannot resolve REDUCE argument tuple')
            raise LoadError('REDUCE arguments must be a tuple')
        kw = kw or {}
        self.call_depth += 1
        if self.call_depth > 100:
            self.call_depth -= 1
            raise Unresolved('call indirection exceeds analysis limit')
        try:
            return self._call(fn, args, kw)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, UnicodeError, struct.error) as exc:
            raise LoadError(str(exc)) from None
        finally:
            self.call_depth -= 1

    def _call(self, fn, args, kw):
        if isinstance(fn, Partial):
            return self.call(fn.fn, fn.args + args, {**fn.kw, **kw})
        if isinstance(fn, Bound):
            return self.bound_call(fn.owner, fn.name, args, kw)
        if isinstance(fn, Adapter):
            if fn.kind == 'attrgetter':
                values = []
                for key in fn.args:
                    value = args[0]
                    for part in key.split('.'):
                        value = self.attr(value, part)
                    values.append(value)
                return values[0] if len(values) == 1 else tuple(values)
            if fn.kind == 'itemgetter':
                values = [self.item(args[0], key) for key in fn.args]
                return values[0] if len(values) == 1 else tuple(values)
            if fn.kind == 'methodcaller':
                return self.call(self.attr(args[0], fn.args[0]), fn.args[1:], fn.kw)
            if fn.kind == 'decorator':
                return Partial(args[0])
        if isinstance(fn, Obj):
            method = self.class_attrs.get(fn.cls, {}).get('__call__', fn.attrs.get('__call__'))
            if method is not None:
                return self.call(method, args, kw)
            if fn.cls == 'xmlrpc.client.ServerProxy.method':
                self.finding('network', 'XML-RPC method call')
                return UNKNOWN
            if fn.cls == 'function':
                self.finding('code-execution', 'call serialized Python function')
                return UNKNOWN
            raise Unresolved('unresolved callable instance ' + fn.cls)
        if isinstance(fn, Unknown):
            raise Unresolved('unresolved callable')
        if not isinstance(fn, Ref):
            raise LoadError('object is not callable')
        name = canonical(fn.name)
        self.validate_call(name, args, kw)
        args, kw = self.positional_keywords(name, args, kw)
        if name in ('os._exit', 'sys.exit', 'builtins.exit', 'builtins.quit'):
            raise LoadError('loader exits')
        if name in ('builtins.getattr', 'builtins.hasattr'):
            result = self.attr(args[0], args[1], args[2] if len(args) > 2 else UNKNOWN)
            return result if name.endswith('getattr') else result is not UNKNOWN
        if name in ('builtins.setattr', 'builtins.delattr'):
            obj, key = args[:2]
            attrs = self.class_attrs.setdefault(obj.name, {}) if isinstance(obj, Ref) else getattr(obj, 'attrs', None)
            if attrs is not None:
                if name.endswith('setattr'):
                    attrs[key] = args[2]
                else:
                    attrs.pop(key, None)
            return None
        if name in ('builtins.__import__', 'importlib.import_module'):
            module = args[0]
            if not isinstance(module, str):
                if isinstance(module, (Unknown, Obj)):
                    raise Unresolved('cannot resolve imported module name')
                raise LoadError('module name must be a string')
            if name == 'builtins.__import__':
                fromlist = args[3] if len(args) > 3 else kw.get('fromlist')
                return self.module(module if fromlist else module.split('.')[0])
            return self.module(module)
        if name == 'pkgutil.resolve_name':
            return self.resolve_name(args[0])
        if name in ('builtins.vars', 'builtins.globals', 'builtins.locals'):
            return self.attr(args[0], '__dict__') if args else Namespace(self.module('builtins'))
        if name in ('operator.attrgetter', 'operator.itemgetter', 'operator.methodcaller'):
            return Adapter(name.split('.')[1], args, kw)
        if name in ('functools.partial', 'functools.partialmethod'):
            return Partial(args[0], args[1:], kw)
        if name in ('copy.copy', 'copy.deepcopy', 'functools.wraps'):
            if name == 'functools.wraps':
                return Adapter('decorator')
            value = args[0]
            if isinstance(value, Obj):
                reducer = self.copy_reducers.get(value.cls)
                if reducer is None:
                    reducer = self.class_attrs.get(value.cls, {}).get('__reduce__', value.attrs.get('__reduce__'))
                    reduction_args = ()
                else:
                    reduction_args = (value,)
                if reducer is not None:
                    reduction = self.call(reducer, reduction_args)
                    if isinstance(reduction, str):
                        return value
                    if isinstance(reduction, tuple) and len(reduction) >= 2:
                        restored = self.call(reduction[0], reduction[1])
                        if len(reduction) > 2 and reduction[2] is not None:
                            self.build(restored, reduction[2])
                        return restored
                    raise LoadError('invalid copy reduction')
            if isinstance(value, Partial):
                return Partial(value.fn, value.args, value.kw.copy(), value.attrs.copy())
            if isinstance(value, (list, dict, set, bytearray)):
                return value.copy() if not isinstance(value, bytearray) else bytearray(value)
            return value
        if name in ('functools.lru_cache', 'functools.cache'):
            if args and isinstance(args[0], (Ref, Partial, Bound, Adapter, Obj)):
                return Partial(args[0])
            return Adapter('decorator')
        if name == 'functools._lru_cache_wrapper':
            return Partial(args[0])
        if name == 'functools.update_wrapper':
            if hasattr(args[0], 'attrs'):
                args[0].attrs['__wrapped__'] = args[1]
            return args[0]
        if name in ('types.MethodType',):
            return Partial(args[0], (args[1],))
        if name in ('operator.call', 'builtins.apply'):
            return self.call(args[0], args[1:], kw)
        if name in ('copyreg._reconstructor',):
            cls, base, state = args
            if isinstance(base, Ref) and base.name != 'builtins.object':
                result = self.new(cls, (state,))
                self.call(base, (state,))
                return result
            return self.new(cls, ())
        if name == 'copyreg.add_extension':
            module, attr, code = args
            if not isinstance(code, int) or not 0 < code <= 0x7fffffff:
                raise LoadError('invalid extension code')
            key = (module, attr)
            if key in self.extension_registry or code in self.inverted_registry:
                if self.extension_registry.get(key) != code or self.inverted_registry.get(code) != key:
                    raise LoadError('extension code is already registered')
            self.extension_registry[key] = code
            self.inverted_registry[code] = key
            return None
        if name == 'copyreg.remove_extension':
            module, attr, code = args
            if self.extension_registry.get((module, attr)) != code:
                raise LoadError('extension is not registered')
            del self.extension_registry[module, attr]
            del self.inverted_registry[code]
            self.extension_cache.pop(code, None)
            return None
        if name == 'copyreg.clear_extension_cache':
            self.extension_cache.clear()
            return None
        if name == 'copyreg.pickle':
            cls, reducer = args[:2]
            if isinstance(cls, Ref):
                self.copy_reducers[cls.name] = reducer
            return None
        if name in ('copyreg.__newobj__', 'copyreg.__newobj_ex__'):
            return self.new(args[0], args[1] if name.endswith('_ex__') else args[1:],
                            args[2] if name.endswith('_ex__') else {})
        if name.endswith('.__new__'):
            return self.new(args[0], args[1:], kw)
        if name.endswith('.__call__'):
            return self.call(Ref(name[:-9]), args, kw)
        if name.endswith('.__setstate__') and args:
            self.build(args[0], args[1])
            return None
        if name in ('builtins.bytes.fromhex', 'builtins.bytearray.fromhex'):
            result = bytes.fromhex(args[0])
            self.budget.add_bytes(len(result))
            return bytearray(result) if 'bytearray' in name else result
        # Unbound descriptors can also arrive as globals.
        parts = name.rsplit('.', 1)
        primitive_types = ('builtins.str', 'builtins.bytes', 'builtins.bytearray',
                           'builtins.dict', 'builtins.list', 'builtins.tuple',
                           'builtins.object', 'builtins.int', 'builtins.float')
        if len(parts) == 2 and parts[0] in primitive_types and args:
            return self.bound_call(args[0], parts[1], args[1:], kw)
        if args and isinstance(args[0], Obj) and name.startswith(args[0].cls + '.'):
            return self.bound_call(args[0], name[len(args[0].cls) + 1:], args[1:], kw)
        if name in NESTED:
            return self.nested(name, args, kw)
        if name in ('pickle.Unpickler', 'pickle._Unpickler'):
            return Obj('pickle.Unpickler', args)
        if name in ('marshal.loads', 'marshal.load'):
            source = args[0]
            if name.endswith('.load'):
                if not isinstance(source, Buffer):
                    raise LoadError('marshal.load requires a binary file object')
                data = source.data[source.pos:]
            else:
                data = source
            if not isinstance(data, (bytes, bytearray)):
                raise LoadError('marshal data must be bytes')
            try:
                # Marshal is a data parser: unlike pickle it has no imports,
                # constructors or state hooks. Code records are immediately
                # replaced with inert markers, never made into functions.
                fh = io.BytesIO(data)
                value = self.marshal_value(marshal.load(fh), {})
                if name.endswith('.load'):
                    source.pos += fh.tell()
                return value
            except (EOFError, ValueError, TypeError):
                raise LoadError('invalid marshal data')
        if name in ('types.CodeType', 'dill._dill._create_code'):
            return Obj('code', args)
        if name in ('logging.config.dictConfig', 'logging.config.DictConfigurator.configure'):
            return self.logging_config(args[0])
        if name in ('logging.config.fileConfig',):
            self.finding('file-access' if args and isinstance(args[0], (str, bytes)) else 'code-execution', name)
            return UNKNOWN
        if name in ('ctypes.CDLL', 'ctypes.PyDLL', 'ctypes.WinDLL', 'ctypes.OleDLL'):
            self.finding('code-execution', name + ' loads native code')
            return Obj(name, args)
        if name in ('ftplib.FTP', 'ftplib.FTP_TLS', 'smtplib.SMTP', 'smtplib.SMTP_SSL', 'telnetlib.Telnet'):
            host = args[0] if args else kw.get('host')
            if host:
                self.finding('network', name + ' connects to host')
            return Obj(name, args)
        if name in ('socket.socket', 'http.client.HTTPConnection', 'http.client.HTTPSConnection',
                    'xmlrpc.client.ServerProxy', 'requests.Session', 'requests.sessions.Session'):
            return Obj('requests.Session' if name.endswith('.Session') else name, args)
        if name in ('io.BytesIO', 'io.StringIO'):
            data = args[0] if args else kw.get('initial_bytes', b'') if name == 'io.BytesIO' else kw.get('initial_value', '')
            if isinstance(data, Obj) and data.cls in ('builtins.memoryview', 'pickle.PickleBuffer'):
                data = data.args[0]
            if isinstance(data, (Unknown, Obj)):
                raise Unresolved('cannot resolve in-memory buffer contents')
            if name == 'io.BytesIO' and isinstance(data, bytearray):
                data = bytes(data)
            if not isinstance(data, bytes if name == 'io.BytesIO' else str):
                raise LoadError('invalid in-memory buffer input')
            return Buffer(data)
        if name in ('builtins.memoryview', 'pickle.PickleBuffer'):
            return Obj(name, args)
        if name in ('io.BufferedReader', 'io.BufferedWriter', 'io.BufferedRandom', 'io.TextIOWrapper'):
            return args[0] if args else Obj(name)
        if name in ('pathlib.Path', 'pathlib.PurePath', 'pathlib.PurePosixPath', 'pathlib.PureWindowsPath'):
            return Obj(name, args)
        if name in ('zipfile.ZipFile', 'tarfile.open', 'tarfile.TarFile.open', 'tarfile.TarFile',
                    'gzip.open', 'gzip.GzipFile', 'bz2.open', 'bz2.BZ2File',
                    'lzma.open', 'lzma.LZMAFile', 'codecs.open'):
            source = kw.get('fileobj', args[0] if args else kw.get('filename', kw.get('name')))
            if name == 'gzip.GzipFile' and len(args) > 3:
                source = args[3]
            if isinstance(source, Buffer):
                return Obj(name, (source,))
            if source is not None:
                self.finding('file-access', name + ' opens an external file')
            return Obj(name, args)
        if name in ('threading.Thread', 'threading.Timer', 'multiprocessing.Process',
                    'multiprocessing.context.Process', 'multiprocessing.context.SpawnProcess',
                    'multiprocessing.context.ForkProcess'):
            obj = Obj('multiprocessing.Process' if 'Process' in name else name, args)
            if name == 'threading.Timer':
                obj.attrs.update(_target=args[1], _args=args[2] if len(args) > 2 else kw.get('args', ()),
                                 _kwargs=args[3] if len(args) > 3 else kw.get('kwargs', {}))
            else:
                obj.attrs.update(_target=args[1] if len(args) > 1 else kw.get('target'),
                                 _args=args[3] if len(args) > 3 else kw.get('args', ()),
                                 _kwargs=args[4] if len(args) > 4 else kw.get('kwargs', {}))
            return obj
        if name in ('_thread.start_new_thread', '_thread.start_new', 'threading._start_new_thread'):
            return self.call(args[0], args[1], args[2] if len(args) > 2 else {})
        if name in ('concurrent.futures.ThreadPoolExecutor', 'concurrent.futures.ProcessPoolExecutor',
                    'multiprocessing.pool.ThreadPool'):
            obj = Obj(name, args)
            obj.attrs.update(kw)
            return obj
        if name in CALL_EFFECTS:
            return self.effect(name, args, kw)
        if name.startswith(('base64.', 'binascii.', 'codecs.', '_codecs.')):
            return self.decode(name, args, kw)
        if name in ('zlib.decompress', 'gzip.decompress', 'bz2.decompress', 'lzma.decompress'):
            from modelgate.dispatch import decompress
            kind = 'xz' if name.startswith('lzma.') else name.split('.')[0]
            wbits = args[1] if len(args) > 1 else kw.get('wbits', 15)
            return decompress(bytes(args[0]), kind, self.budget, wbits=wbits)
        if name in ('operator.getitem', 'operator.__getitem__'):
            return self.item(args[0], args[1])
        if name in ('operator.setitem', 'operator.__setitem__'):
            self.setitem(*args)
            return None
        if name.startswith('operator.'):
            return self.operator(name.split('.', 1)[1], args)
        if name in ('builtins.map', 'builtins.filter', 'itertools.starmap', 'itertools.filterfalse',
                    'itertools.takewhile', 'itertools.dropwhile', 'itertools.accumulate'):
            return Lazy(name, args)
        if name in ('builtins.iter',):
            if len(args) == 2:
                return Lazy('call-sentinel', args)
            return self.iterator(args[0])
        if name in ('builtins.next',):
            try:
                return self.next(args[0])
            except EndIteration:
                if len(args) > 1:
                    return args[1]
                raise LoadError('iterator exhausted')
        if name in ('builtins.zip', 'builtins.enumerate', 'builtins.reversed',
                    'itertools.chain', 'itertools.chain.from_iterable', 'itertools.islice',
                    'itertools.repeat', 'itertools.count', 'itertools.cycle', 'itertools.zip_longest',
                    'itertools.product', 'itertools.permutations', 'itertools.combinations'):
            return Lazy(name, args)
        if name == 'functools.reduce':
            it = self.iterator(args[1])
            try:
                acc = args[2] if len(args) > 2 else self.next(it)
            except EndIteration:
                raise LoadError('reduce of empty iterable without initializer')
            while True:
                try:
                    value = self.next(it)
                except EndIteration:
                    return acc
                acc = self.call(args[0], (acc, value))
        if name in ('builtins.sorted', 'builtins.min', 'builtins.max'):
            values = self.consume(args[0]) if len(args) == 1 or name.endswith('sorted') else list(args)
            key = kw.get('key')
            keys = [self.call(key, (v,)) for v in values] if key is not None else values
            if not values:
                if name.endswith('sorted'):
                    return []
                if 'default' in kw:
                    return kw['default']
                raise LoadError('empty min/max')
            if all(literal(x) for x in keys):
                try:
                    indices = sorted(range(len(values)), key=lambda i: keys[i], reverse=bool(kw.get('reverse')))
                    if name.endswith('min'):
                        return values[indices[0]]
                    if name.endswith('max'):
                        return values[indices[-1]]
                    return [values[i] for i in indices]
                except TypeError:
                    pass
            return values if name.endswith('sorted') else UNKNOWN
        if name in ('builtins.any', 'builtins.all'):
            it = self.iterator(args[0])
            while True:
                try:
                    value = self.next(it)
                except EndIteration:
                    return name.endswith('all')
                if bool(value) == name.endswith('any'):
                    return name.endswith('any')
        if name in ('builtins.list', 'builtins.tuple', 'builtins.set', 'builtins.frozenset',
                    'collections.deque'):
            values = self.consume(args[0]) if args else []
            return {'builtins.tuple': tuple, 'builtins.set': set, 'builtins.frozenset': frozenset}.get(name, list)(values)
        if name in ('builtins.dict', 'collections.OrderedDict', 'collections.Counter', 'collections.defaultdict'):
            if name == 'collections.defaultdict':
                obj = Obj(name, args)
                obj.attrs['_items'] = dict(args[1]) if len(args) > 1 else {}
                return obj
            value = args[0] if args else {}
            return {**dict(value if isinstance(value, dict) else self.consume(value)), **kw}
        if name in ('builtins.bytes', 'builtins.bytearray', 'builtins.str', 'builtins.int',
                    'builtins.float', 'builtins.bool', 'builtins.complex'):
            if name == 'builtins.str' and args and not literal(args[0]):
                return UNKNOWN
            if args and isinstance(args[0], int) and name in ('builtins.bytes', 'builtins.bytearray'):
                self.budget.add_bytes(max(0, args[0]))
            if args and isinstance(args[0], Lazy):
                args = (self.consume(args[0]),) + args[1:]
            if args and isinstance(args[0], Obj) and args[0].cls in ('builtins.memoryview', 'pickle.PickleBuffer'):
                args = (args[0].args[0],) + args[1:]
            constructors = {'bytes': bytes, 'bytearray': bytearray, 'str': str, 'int': int,
                            'float': float, 'bool': bool, 'complex': complex}
            if all(literal(x) for x in args):
                return constructors[name.split('.')[1]](*args, **kw)
            return UNKNOWN
        if name in ('builtins.len', 'operator.length_hint'):
            return len(args[0]) if isinstance(args[0], (list, tuple, dict, str, bytes, bytearray, set, range)) else UNKNOWN
        if name == 'builtins.range':
            return range(*args)
        if name == 'builtins.slice':
            return slice(*args)
        if name == 'builtins.sum':
            values = self.consume(args[0])
            return sum(values, args[1] if len(args) > 1 else 0) if all(literal(x) for x in values) else UNKNOWN
        if name in ('builtins.abs', 'builtins.round', 'builtins.chr', 'builtins.ord', 'builtins.hex',
                    'builtins.oct', 'builtins.bin', 'builtins.divmod', 'builtins.pow'):
            funcs = {'abs': abs, 'round': round, 'chr': chr, 'ord': ord, 'hex': hex, 'oct': oct,
                     'bin': bin, 'divmod': divmod, 'pow': pow}
            if name.endswith('pow') and len(args) == 2 and isinstance(args[1], int) and abs(args[1]) > 10000:
                raise LimitExceeded('oversized power')
            return funcs[name.split('.')[1]](*args) if all(literal(x) for x in args) else UNKNOWN
        if name in ('builtins.type',):
            if len(args) == 1:
                return self.attr(args[0], '__class__')
            self.finding('code-execution', 'dynamically constructed class')
            return Ref('dynamic.class')
        if name == 'types.SimpleNamespace':
            return Obj(name, (), kw.copy())
        if name == 'types.ModuleType':
            return Module(args[0])
        if name == 'types.MappingProxyType':
            return args[0]
        if name in ('os.fspath', 'os.fsencode', 'os.fsdecode'):
            value = args[0]
            if isinstance(value, Obj) and value.cls.startswith('pathlib.'):
                value = posixpath.join(*value.args) if value.args else '.'
            if name == 'os.fsencode' and isinstance(value, str):
                return value.encode()
            if name == 'os.fsdecode' and isinstance(value, bytes):
                return value.decode()
            return value
        if name in ('os.getpid', 'os.getppid', 'os.getuid', 'os.getgid', 'os.geteuid', 'os.getegid'):
            return 1
        if name in ('os.getcwd', 'os.getcwdb', 'os.getenv', 'os.getenvb', 'sys.getdefaultencoding',
                    'sys.getfilesystemencoding', 'time.time', 'time.monotonic', 'time.perf_counter'):
            return UNKNOWN
        if name.startswith(('posixpath.', 'ntpath.')):
            method = name.split('.')[1]
            if method in ('join', 'normpath', 'basename', 'dirname', 'split', 'splitext', 'isabs', 'commonprefix', 'relpath'):
                return getattr(posixpath, method)(*args)
        if name in ('struct.pack', 'struct.unpack', 'struct.calcsize'):
            if args and isinstance(args[0], str) and len(args[0]) < 10000:
                size = struct.calcsize(args[0])
                if size > MAX_BYTES:
                    raise LimitExceeded('oversized struct')
                return getattr(struct, name.split('.')[1])(*args)
        if name in ('json.loads', 'ast.literal_eval'):
            import ast
            return self.json_loads(args[0], kw) if name == 'json.loads' else ast.literal_eval(args[0])
        if name in ('json.JSONDecoder', 'json.decoder.JSONDecoder'):
            return Obj('json.JSONDecoder', (), kw.copy())
        # Pure scientific constructors and reconstruction helpers. Their
        # arguments/state remain symbolic and can still carry callable refs.
        if name.startswith(('numpy.', 'torch.', 'sklearn.', 'scipy.', 'pandas.')):
            if name.endswith(('.load', '.load_library', '.load_op_library', '.read', '.write', '.dump')):
                raise Unresolved('unmodeled scientific I/O callable ' + name)
            if name.endswith('._reconstruct') and args and isinstance(args[0], Ref):
                return Obj(args[0].name, args)
            if name.endswith('._rebuild_from_type_v2') and args:
                result = self.call(args[0], args[2])
                self.build(result, args[3])
                return result
            return Obj(name, args)
        # Constructors of ordinary standard-library data objects are inert.
        if self.is_class(fn) or name.startswith(('math.', 'statistics.', 'datetime.', 'decimal.',
                   'fractions.', 're.', 'uuid.', 'array.', 'collections.', 'enum.')):
            return Obj(name, args)
        raise Unresolved('no static call summary for ' + name)

    def validate_call(self, name, args, kw):
        signature = CALL_SIGNATURES.get(name)
        if signature is None:
            return
        positional = [p for p in signature if p[0] in (0, 1)]
        if len(args) > len(positional) and not any(p[0] == 2 for p in signature):
            raise LoadError('too many arguments for ' + name)
        consumed = set()
        for i, (kind, key, required) in enumerate(positional):
            if i < len(args):
                if kind == 1 and key in kw:
                    raise LoadError('multiple values for ' + key)
            elif kind == 1 and key in kw:
                consumed.add(key)
            elif required:
                raise LoadError('missing argument ' + key)
        for kind, key, required in signature:
            if kind == 3:
                if key in kw:
                    consumed.add(key)
                elif required:
                    raise LoadError('missing keyword argument ' + key)
        if set(kw) - consumed and not any(p[0] == 4 for p in signature):
            raise LoadError('unexpected keyword arguments for ' + name)

    def positional_keywords(self, name, args, kw):
        signature = CALL_SIGNATURES.get(name, ())
        positional = [p for p in signature if p[0] in (0, 1)]
        args, kw = list(args), kw.copy()
        for kind, key, required in positional[len(args):]:
            if kind == 1 and key in kw:
                args.append(kw.pop(key))
            else:
                break
        return tuple(args), kw

    def effect(self, name, args, kw):
        self.validate_call(name, args, kw)
        # Functions with required positional arguments cannot execute on an
        # empty argument tuple. Constructors reached via NEWOBJ never enter
        # this function: they run only __new__, not __init__.
        no_arg_effects = {'os.fork', 'os.forkpty', 'pty.fork', 'os.getcwd',
                          'tempfile.mkstemp', 'tempfile.mkdtemp', 'tempfile.NamedTemporaryFile',
                          'tempfile.TemporaryFile', 'tempfile.TemporaryDirectory',
                          'multiprocessing.Pool', 'multiprocessing.pool.Pool'}
        if not args and not kw and name not in no_arg_effects:
            # timeit supplies a harmless default statement.
            if name in ('timeit.timeit', 'timeit.repeat', 'timeit.Timer'):
                return Obj(name)
            raise LoadError('missing call arguments for ' + name)
        if name in ('builtins.eval', 'builtins.exec', 'builtins.compile'):
            source = args[0] if args else kw.get('source')
            if not isinstance(source, (str, bytes, Obj, Unknown)):
                raise LoadError('source is not executable')
            if isinstance(source, Obj) and source.cls != 'code' and not source.cls.startswith('ast.'):
                raise Unresolved('cannot resolve executable source')
        if name in ('os.system', 'os.popen') and args:
            if not isinstance(args[0], (str, bytes)):
                if isinstance(args[0], (Unknown, Obj)):
                    raise Unresolved('cannot resolve process command')
                raise LoadError('command must be a string')
        if name in ('subprocess.Popen', 'subprocess.run', 'subprocess.call',
                    'subprocess.check_call', 'subprocess.check_output'):
            command = args[0] if args else kw.get('args')
            if not isinstance(command, (str, bytes, list, tuple, Obj, Unknown)):
                raise LoadError('invalid subprocess arguments')
            if isinstance(command, (list, tuple)) and not command and not kw.get('shell'):
                raise LoadError('empty subprocess arguments')
        if name in ('types.FunctionType', 'types.LambdaType'):
            if args and isinstance(args[0], Unknown):
                raise Unresolved('cannot resolve function code object')
            if len(args) < 2 or not isinstance(args[0], Obj) or args[0].cls != 'code' or not isinstance(args[1], dict):
                raise LoadError('FunctionType requires a code object and globals dictionary')
        if name in ('timeit.timeit', 'timeit.repeat', 'timeit.Timer'):
            stmt = args[0] if args else kw.get('stmt', 'pass')
            setup = args[1] if len(args) > 1 else kw.get('setup', 'pass')
            # Literal source is compiled, even in a Timer constructor. Callable
            # statements are deferred until timeit/repeat is invoked.
            if isinstance(stmt, str) or isinstance(setup, str) and setup != 'pass':
                self.finding('code-execution', name + ' compiles source')
            if name != 'timeit.Timer':
                for value in (setup, stmt):
                    if not isinstance(value, str):
                        self.call(value, ())
            return Obj('timeit.Timer', args, {'stmt': stmt, 'setup': setup})
        # In-memory destinations/sources cannot touch an external file.
        if name in ('numpy.save', 'numpy.savez', 'numpy.savez_compressed', 'torch.save', 'joblib.dump'):
            destination = args[1] if name in ('torch.save', 'joblib.dump') and len(args) > 1 else args[0]
            if isinstance(destination, Buffer):
                return None
        if name == 'sqlite3.connect' and args and args[0] == ':memory:':
            return Obj(name, args)
        if name == 'logging.FileHandler' and kw.get('delay'):
            return Obj('logging.FileHandler', args)
        self.finding(CALL_EFFECTS[name], 'call ' + name)
        return Obj(name, args)

    def bound_call(self, obj, method, args, kw):
        if method == '__call__':
            return self.call(obj, args, kw)
        if method == '__setstate__':
            self.build(obj, args[0])
            return None
        if method == '__getattribute__':
            return self.attr(obj, args[0])
        if method == '__setattr__':
            return self.call(Ref('builtins.setattr'), (obj,) + args)
        if method == '__getitem__':
            return self.item(obj, args[0])
        if method == '__setitem__':
            self.setitem(obj, args[0], args[1])
            return None
        if method == '__iter__':
            return self.iterator(obj)
        if method == '__next__':
            return self.next(obj)
        if isinstance(obj, Namespace):
            if method in ('get', '__getitem__'):
                return self.item(obj, args[0], args[1] if len(args) > 1 else None)
            if method in ('update', '__ior__'):
                for key, val in (args[0].items() if isinstance(args[0], dict) else args[0]):
                    self.setitem(obj, key, val)
                for key, val in kw.items():
                    self.setitem(obj, key, val)
                return None
            if method in ('copy', 'items', 'keys', 'values') and hasattr(obj.owner, 'attrs'):
                return self.bound_call(obj.owner.attrs, method, args, kw)
        if isinstance(obj, Buffer):
            if method in ('getvalue', 'getbuffer'):
                return obj.data
            if method == 'tell':
                return obj.pos
            if method == 'seek':
                offset = args[0]
                whence = args[1] if len(args) > 1 else 0
                obj.pos = offset + (0 if whence == 0 else obj.pos if whence == 1 else len(obj.data))
                return obj.pos
            if method in ('read', 'read1', 'readline'):
                size = args[0] if args else -1
                end = len(obj.data) if size < 0 else min(len(obj.data), obj.pos + size)
                if method == 'readline':
                    newline = b'\n' if isinstance(obj.data, bytes) else '\n'
                    index = obj.data.find(newline, obj.pos, end)
                    if index >= 0:
                        end = index + 1
                value = obj.data[obj.pos:end]
                obj.pos = end
                return value
            if method == 'write':
                value = args[0]
                obj.data = obj.data[:obj.pos] + value + obj.data[obj.pos + len(value):]
                obj.pos += len(value)
                self.budget.add_bytes(len(value))
                return len(value)
            if method in ('close', 'flush'):
                return None
        if isinstance(obj, Obj):
            name = canonical(obj.cls + '.' + method)
            self.validate_call(name, (obj,) + args, kw)
            if obj.cls in ('builtins.memoryview', 'pickle.PickleBuffer') and method in ('raw', 'tobytes', '__bytes__'):
                return obj.args[0]
            if method == '__init__':
                return self.call(Ref(obj.cls), args, kw)
            if obj.cls == 'pickle.Unpickler' and method == 'load':
                return self.nested('pickle.load', obj.args, {})
            if obj.cls in ('json.JSONDecoder', 'json.decoder.JSONDecoder') and method in ('decode', 'raw_decode'):
                return self.json_loads(args[0], obj.attrs)
            if obj.cls == 'zipfile.ZipFile' and obj.args and isinstance(obj.args[0], Buffer):
                import zipfile
                with zipfile.ZipFile(io.BytesIO(obj.args[0].data)) as zf:
                    if method == 'namelist':
                        return zf.namelist()
                    if method in ('read', 'open'):
                        self.budget.add_bytes(zf.getinfo(args[0]).file_size)
                        data = zf.read(args[0])
                        return Buffer(data) if method == 'open' else data
                    if method in ('extract', 'extractall'):
                        self.finding('file-access', name + ' extracts files')
                        return UNKNOWN
            if obj.cls in ('gzip.open', 'gzip.GzipFile', 'bz2.open', 'bz2.BZ2File',
                           'lzma.open', 'lzma.LZMAFile') and obj.args and isinstance(obj.args[0], Buffer):
                if '_buffer' not in obj.attrs:
                    from modelgate.dispatch import decompress
                    kind = {'gzip': 'gzip', 'bz2': 'bz2', 'lzma': 'xz'}[obj.cls.split('.')[0]]
                    obj.attrs['_buffer'] = Buffer(decompress(obj.args[0].data, kind, self.budget))
                return self.bound_call(obj.attrs['_buffer'], method, args, kw)
            if obj.cls == 'xmlrpc.client.ServerProxy':
                self.finding('network', 'XML-RPC request ' + method)
                return UNKNOWN
            if obj.cls in ('threading.Thread', 'threading.Timer', 'multiprocessing.Process') and method in ('start', 'run'):
                if obj.cls == 'multiprocessing.Process' and method == 'start':
                    self.finding('process', 'start multiprocessing.Process')
                target = obj.attrs.get('_target', obj.attrs.get('target'))
                if target is not None:
                    return self.call(target, tuple(obj.attrs.get('_args', ())), obj.attrs.get('_kwargs', {}))
                return None
            if obj.cls in ('concurrent.futures.ThreadPoolExecutor', 'concurrent.futures.ProcessPoolExecutor',
                           'multiprocessing.pool.ThreadPool') and method in ('submit', 'map', 'apply', 'apply_async', 'imap'):
                if 'ProcessPool' in obj.cls:
                    self.finding('process', obj.cls + '.' + method)
                initializer = obj.attrs.get('initializer')
                if initializer:
                    self.call(initializer, tuple(obj.attrs.get('initargs', ())))
                if method in ('submit',):
                    result = self.call(args[0], args[1:], kw)
                elif method in ('apply', 'apply_async'):
                    result = self.call(args[0], tuple(args[1]) if len(args) > 1 else (),
                                       args[2] if len(args) > 2 else {})
                else:
                    result = self.consume(Lazy('builtins.map', args))
                return Obj('concurrent.futures.Future', (), {'result_value': result})
            if obj.cls == 'concurrent.futures.Future' and method == 'result':
                return obj.attrs.get('result_value', UNKNOWN)
            if obj.cls == 'timeit.Timer' and method in ('timeit', 'repeat'):
                for value in (obj.attrs.get('setup'), obj.attrs.get('stmt')):
                    if value is not None and not isinstance(value, str):
                        self.call(value, ())
                return UNKNOWN
            if obj.cls == 'logging.config.DictConfigurator' and method == 'configure':
                return self.logging_config(obj.args[0])
            if obj.cls.startswith('pathlib.'):
                if method in ('__fspath__', '__str__', 'as_posix'):
                    return posixpath.join(*obj.args) if obj.args else '.'
                if method in ('joinpath', '__truediv__'):
                    return Obj(obj.cls, obj.args + args)
                if method in ('with_name', 'with_suffix', 'relative_to', 'absolute'):
                    return Obj(obj.cls, obj.args)
            if obj.cls == 'collections.defaultdict' and method in ('get', '__getitem__'):
                values = obj.attrs['_items']
                if args[0] in values:
                    return values[args[0]]
                if method == '__getitem__' and obj.args and obj.args[0] is not None:
                    value = self.call(obj.args[0], ())
                    values[args[0]] = value
                    return value
                return args[1] if len(args) > 1 else None
            if name in CALL_EFFECTS:
                return self.effect(name, (obj,) + args, kw)
            if name in NESTED:
                return self.nested(name, args, kw)
            if method in obj.attrs:
                return self.call(obj.attrs[method], args, kw)
            if obj.cls.startswith(('numpy.', 'torch.', 'sklearn.', 'scipy.', 'pandas.')):
                if method in ('dump', 'tofile', 'to_pickle', 'to_csv', 'to_json', 'save'):
                    self.finding('file-access', name)
                return UNKNOWN
            if method in ('close', 'join', 'shutdown', 'acquire', 'release'):
                return None
            raise Unresolved('no static method summary for ' + name)
        if isinstance(obj, dict):
            if method in ('get', 'pop', 'setdefault', 'update', 'copy', 'clear'):
                return getattr(obj, method)(*args, **kw)
            if method in ('keys', 'values', 'items'):
                return list(getattr(obj, method)())
        if isinstance(obj, list):
            if method == 'extend':
                obj.extend(self.consume(args[0]))
                return None
            if method in ('append', 'pop', 'insert', 'reverse', 'copy', 'clear', 'remove', 'count', 'index'):
                return getattr(obj, method)(*args)
            if method == 'sort':
                obj[:] = self.call(Ref('builtins.sorted'), (obj,), kw)
                return None
        if isinstance(obj, (str, bytes, bytearray)):
            if method == 'join':
                return obj.join(self.consume(args[0]))
            safe = {'encode', 'decode', 'replace', 'split', 'rsplit', 'splitlines', 'strip', 'lstrip', 'rstrip',
                    'upper', 'lower', 'casefold', 'capitalize', 'title', 'swapcase', 'zfill',
                    'removeprefix', 'removesuffix', 'partition', 'rpartition', 'startswith', 'endswith',
                    'find', 'rfind', 'index', 'rindex', 'count', 'hex', 'translate', '__add__', '__mul__'}
            if method in safe:
                if method == '__mul__' and args and isinstance(args[0], int) and len(obj) * args[0] > MAX_BYTES:
                    raise LimitExceeded('oversized string repeat')
                result = getattr(obj, method)(*args, **kw)
                if isinstance(result, (str, bytes, bytearray)):
                    self.budget.add_bytes(len(result))
                return result
            if method in ('format', 'format_map'):
                # Formatting literals is safe. Attribute/subscript traversal
                # on symbolic values is interpreted instead of using Python.
                import string
                formatter = string.Formatter()
                output, auto = [], 0
                for text, field, spec, conversion in formatter.parse(obj):
                    output.append(text)
                    if field is None:
                        continue
                    if not field:
                        field, auto = str(auto), auto + 1
                    import _string
                    root, rest = _string.formatter_field_name_split(field)
                    value = args[root] if isinstance(root, int) else kw[root]
                    for is_attr, key in rest:
                        value = self.attr(value, key) if is_attr else self.item(value, key)
                    if not literal(value):
                        return UNKNOWN
                    output.append(format(value, spec))
                return ''.join(output)
        if isinstance(obj, (int, float)) and method in ('__add__', '__sub__', '__mul__', '__neg__', '__int__', '__float__'):
            return getattr(obj, method)(*args)
        raise Unresolved('no static method summary for ' + type(obj).__name__ + '.' + method)

    def operator(self, name, args):
        aliases = {'concat': 'add', 'iconcat': 'add', 'iadd': 'add', 'imul': 'mul', 'isub': 'sub'}
        name = aliases.get(name.strip('_'), name.strip('_'))
        funcs = {key: getattr(operator, key) for key in (
            'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'neg', 'pos', 'not_',
            'eq', 'ne', 'lt', 'le', 'ge', 'gt', 'and_', 'or_', 'xor', 'inv', 'index',
            'is_', 'is_not', 'truth', 'contains', 'countOf', 'indexOf')}
        if name in ('mul', 'add'):
            if name == 'add' and len(args) == 2 and type(args[0]) in (bytes, str, list, tuple, bytearray) and type(args[1]) is type(args[0]):
                self.budget.add_bytes(len(args[0]) + len(args[1]))
                return args[0] + args[1]
            if name == 'mul' and len(args) == 2:
                a, b = args
                if isinstance(a, int):
                    a, b = b, a
                if isinstance(a, (bytes, str, list, tuple, bytearray)) and isinstance(b, int):
                    self.budget.add_bytes(max(0, len(a) * b))
                    return a * b
            if all(literal(x) for x in args):
                return funcs[name](*args)
        if name in funcs and all(literal(x) for x in args):
            return funcs[name](*args)
        return UNKNOWN

    def marshal_value(self, value, memo):
        self.budget.tick()
        if id(value) in memo:
            return memo[id(value)]
        if isinstance(value, types.CodeType):
            return Obj('code')
        if isinstance(value, list):
            result = []
            memo[id(value)] = result
            result.extend(self.marshal_value(x, memo) for x in value)
            return result
        if isinstance(value, dict):
            result = {}
            memo[id(value)] = result
            for key, item in value.items():
                result[self.marshal_value(key, memo)] = self.marshal_value(item, memo)
            return result
        if isinstance(value, (tuple, set, frozenset)):
            result = type(value)(self.marshal_value(x, memo) for x in value)
            memo[id(value)] = result
            return result
        return value

    def json_loads(self, data, options):
        hooks = {}
        for key in ('object_hook', 'object_pairs_hook', 'parse_int', 'parse_float', 'parse_constant'):
            callback = options.get(key)
            if callback is not None:
                # Only scanner-owned closures are handed to the JSON parser.
                # The referenced callable is interpreted, never invoked.
                hooks[key] = lambda value, fn=callback: self.call(fn, (value,))
        cls = options.get('cls')
        if cls is not None and not (isinstance(cls, Ref) and cls.name in ('json.JSONDecoder', 'json.decoder.JSONDecoder')):
            raise Unresolved('custom JSON decoder')
        if 'strict' in options:
            hooks['strict'] = options['strict']
        return json.loads(data, **hooks)

    def decode(self, name, args, kw):
        module, method = name.split('.', 1)
        if not args:
            raise LoadError('missing codec input')
        value = args[0]
        if not isinstance(value, (bytes, bytearray, str)):
            return UNKNOWN
        if module == 'base64':
            methods = {'b64decode', 'standard_b64decode', 'urlsafe_b64decode', 'b32decode',
                       'b16decode', 'b85decode', 'a85decode', 'decodebytes',
                       'b64encode', 'standard_b64encode', 'urlsafe_b64encode', 'encodebytes'}
            if method in methods:
                result = getattr(base64, method)(*args, **kw)
            else:
                raise Unresolved('unknown base64 operation')
        elif module == 'binascii':
            if method not in ('a2b_base64', 'unhexlify', 'a2b_hex', 'a2b_qp', 'a2b_uu',
                              'hexlify', 'b2a_hex', 'b2a_base64'):
                raise Unresolved('unknown binascii operation')
            result = getattr(binascii, method)(*args, **kw)
        else:
            encoding = args[1] if len(args) > 1 else kw.get('encoding', 'utf-8')
            # Only trusted builtin codecs. No codec can import artifact code.
            safe = {'utf8', 'utf-8', 'utf_8', 'ascii', 'latin1', 'latin-1', 'latin_1',
                    'iso-8859-1', 'utf-16', 'utf-16le', 'utf-16-le', 'utf-32',
                    'raw_unicode_escape', 'unicode_escape', 'base64', 'base64_codec',
                    'hex', 'hex_codec', 'rot13', 'rot_13', 'zlib', 'zlib_codec', 'bz2', 'bz2_codec'}
            if encoding not in safe:
                raise Unresolved('unmodeled codec ' + str(encoding))
            if encoding in ('zlib', 'zlib_codec', 'bz2', 'bz2_codec') and method == 'decode':
                from modelgate.dispatch import decompress
                result = decompress(bytes(value), encoding.split('_')[0], self.budget)
            elif method in ('encode', 'decode'):
                result = getattr(codecs, method)(*args, **kw)
            else:
                raise Unresolved('unmodeled codec operation ' + name)
        self.budget.add_bytes(len(result))
        return result

    def iterator(self, value):
        if isinstance(value, Lazy):
            return value
        if isinstance(value, (list, tuple, str, bytes, bytearray, dict, set, frozenset, range)):
            return Lazy('sequence', (list(value) if isinstance(value, (dict, set, frozenset)) else value,))
        if isinstance(value, Obj) and value.cls.startswith(('numpy.', 'torch.')):
            if isinstance(value.state, tuple) and value.state and isinstance(value.state[-1], list):
                return Lazy('sequence', (value.state[-1],))
            return Lazy('sequence', ([],))
        raise Unresolved('cannot resolve iterable')

    def next(self, value):
        self.budget.tick()
        if not isinstance(value, Lazy):
            raise LoadError('next requires an iterator')
        if value.done:
            raise EndIteration()
        kind, args = value.kind, value.args
        try:
            if kind == 'sequence':
                if value.index >= len(args[0]):
                    raise EndIteration()
                result = args[0][value.index]
                value.index += 1
                return result
            if kind == 'call-sentinel':
                result = self.call(args[0], ())
                if result == args[1]:
                    raise EndIteration()
                return result
            if kind in ('builtins.map', 'builtins.zip', 'itertools.zip_longest'):
                offset = 1 if kind == 'builtins.map' else 0
                if value.state is None:
                    value.state = [self.iterator(x) for x in args[offset:]]
                if not value.state:
                    raise EndIteration()
                values, active = [], False
                for it in value.state:
                    try:
                        values.append(self.next(it))
                        active = True
                    except EndIteration:
                        if kind != 'itertools.zip_longest':
                            raise
                        values.append(None)
                if not active:
                    raise EndIteration()
                return self.call(args[0], tuple(values)) if offset else tuple(values)
            if kind in ('builtins.filter', 'itertools.filterfalse', 'itertools.takewhile',
                        'itertools.dropwhile', 'itertools.starmap'):
                if value.state is None:
                    value.state = self.iterator(args[1])
                while True:
                    item = self.next(value.state)
                    if kind == 'itertools.starmap':
                        return self.call(args[0], tuple(self.consume(item)))
                    if kind == 'itertools.dropwhile' and value.index:
                        return item
                    predicate = self.call(args[0], (item,)) if args[0] is not None else item
                    if kind == 'itertools.takewhile':
                        if not predicate:
                            raise EndIteration()
                        return item
                    if kind == 'itertools.dropwhile':
                        if not predicate:
                            value.index = 1
                            return item
                    elif bool(predicate) != (kind == 'itertools.filterfalse'):
                        return item
            if kind == 'builtins.enumerate':
                if value.state is None:
                    value.state = self.iterator(args[0])
                    value.index = args[1] if len(args) > 1 else 0
                result = (value.index, self.next(value.state))
                value.index += 1
                return result
            if kind == 'builtins.reversed':
                if value.state is None:
                    value.state = self.iterator(list(reversed(self.consume(args[0]))))
                return self.next(value.state)
            if kind in ('itertools.chain', 'itertools.chain.from_iterable'):
                if value.state is None:
                    outer = self.iterator(args[0]) if kind.endswith('from_iterable') else self.iterator(args)
                    value.state = [outer, None]
                while True:
                    if value.state[1] is None:
                        value.state[1] = self.iterator(self.next(value.state[0]))
                    try:
                        return self.next(value.state[1])
                    except EndIteration:
                        value.state[1] = None
            if kind == 'itertools.islice':
                if value.state is None:
                    value.state = self.iterator(args[0])
                start = 0 if len(args) == 2 else (args[1] or 0)
                stop = args[1] if len(args) == 2 else args[2]
                step = args[3] if len(args) > 3 else 1
                while stop is None or value.index < stop:
                    item = self.next(value.state)
                    index = value.index
                    value.index += 1
                    if index >= start and (index - start) % step == 0:
                        return item
                raise EndIteration()
            if kind == 'itertools.repeat':
                if len(args) > 1 and value.index >= args[1]:
                    raise EndIteration()
                value.index += 1
                return args[0]
            if kind == 'itertools.count':
                result = (args[0] if args else 0) + value.index * (args[1] if len(args) > 1 else 1)
                value.index += 1
                return result
            if kind == 'itertools.cycle':
                if value.state is None:
                    value.state = self.consume(args[0])
                if not value.state:
                    raise EndIteration()
                result = value.state[value.index % len(value.state)]
                value.index += 1
                return result
            if kind == 'itertools.accumulate':
                if value.state is None:
                    value.state = [self.iterator(args[0]), UNKNOWN]
                item = self.next(value.state[0])
                if value.index:
                    fn = args[1] if len(args) > 1 and args[1] is not None else Ref('operator.add')
                    item = self.call(fn, (value.state[1], item))
                value.state[1] = item
                value.index += 1
                return item
            if kind in ('itertools.product', 'itertools.permutations', 'itertools.combinations'):
                if value.state is None:
                    method = kind.split('.')[1]
                    lists = [self.consume(x) for x in args] if method == 'product' else [self.consume(args[0])] + list(args[1:])
                    if method == 'product' and math.prod(len(x) for x in lists) > MAX_ITEMS:
                        raise LimitExceeded('iterator product too large')
                    value.state = getattr(itertools, method)(*lists)
                try:
                    return next(value.state)
                except StopIteration:
                    raise EndIteration()
            raise Unresolved('unknown iterator ' + kind)
        except EndIteration:
            value.done = True
            raise

    def consume(self, value):
        it = self.iterator(value)
        result = []
        while True:
            try:
                result.append(self.next(it))
            except EndIteration:
                return result
            if len(result) > MAX_ITEMS:
                raise LimitExceeded('iterator consumption limit exceeded')

    def logging_config(self, config):
        if not isinstance(config, dict):
            raise LoadError('logging configuration is not a dict')
        # Only the configuration fields that logging actually interprets can
        # call factories. Text in format strings and logger names is data.
        if config.get('version') != 1:
            raise LoadError('invalid logging configuration version')
        for section in ('formatters', 'filters', 'handlers'):
            for settings in config.get(section, {}).values():
                if not isinstance(settings, dict):
                    continue
                factory = settings.get('()')
                if factory is None and section == 'handlers':
                    factory = settings.get('class')
                if isinstance(factory, str):
                    factory = self.resolve_name(factory)
                if factory is not None:
                    kwargs = {k: v for k, v in settings.items() if k not in ('()', 'class', '.', 'level', 'formatter', 'filters')}
                    for key, value in kwargs.items():
                        if isinstance(value, str) and value.startswith('ext://'):
                            kwargs[key] = self.resolve_name(value[6:])
                    self.call(factory, (), kwargs)
        return None

    def nested(self, name, args, kw):
        from modelgate.pickle_scan import interpret
        from modelgate.dispatch import numpy_payload, decompress, scan_data
        import zipfile

        kind = NESTED[name]
        source = args[0] if args else kw.get('file', kw.get('f', kw.get('filepath_or_buffer', UNKNOWN)))
        if isinstance(source, Obj) and source.cls == 'pickle.Unpickler':
            source = source.args[0]
        if kind in ('pickle-bytes', 'torch-bytes'):
            if isinstance(source, Obj) and source.cls in ('builtins.memoryview', 'pickle.PickleBuffer'):
                source = source.args[0]
            if not isinstance(source, (bytes, bytearray)):
                if isinstance(source, (Unknown, Obj)):
                    raise Unresolved('cannot resolve embedded loader bytes')
                raise LoadError('embedded loader needs bytes')
            data = bytes(source)
        elif isinstance(source, Buffer):
            data = source.data[source.pos:]
            if not isinstance(data, bytes):
                raise LoadError('binary loader needs a binary stream')
        elif isinstance(source, (str, bytes)) or isinstance(source, Obj) and source.cls.startswith('pathlib.'):
            self.finding('file-access', name + ' reads an external path')
            return UNKNOWN
        else:
            raise Unresolved('cannot resolve input to ' + name)
        self.depth += 1
        if self.depth > 16:
            self.depth -= 1
            raise Unresolved('nested loader depth limit')
        outer_location = self.location
        self.location += ':' + name
        try:
            if kind.startswith('pickle'):
                value, end, halted = interpret(data, self, sequence=False, location=self.location)
                if isinstance(source, Buffer):
                    source.pos += end
                if halted:
                    raise LoadError('nested pickle load failed')
                return value
            if kind.startswith('torch'):
                if data.startswith(b'PK\x03\x04'):
                    with zipfile.ZipFile(io.BytesIO(data)) as zf:
                        names = [x for x in zf.namelist() if x.endswith('/data.pkl') or x == 'data.pkl']
                        if not names:
                            raise LoadError('torch archive has no data.pkl')
                        raw = zf.read(names[-1])
                        value, _, halted = interpret(raw, self, sequence=False, persistent=True, location=self.location + '!' + names[-1])
                else:
                    value, _, halted = interpret(data, self, sequence=True, torch=True, location=self.location)
                if halted:
                    raise LoadError('nested torch load failed')
                return value
            if kind in ('joblib', 'pandas'):
                compression = ('gzip' if data.startswith(b'\x1f\x8b') else 'bz2' if data.startswith(b'BZh') else
                               'xz' if data.startswith(b'\xfd7zXZ\x00') else 'zlib' if data[:1] == b'x' else None)
                if compression:
                    data = decompress(data, compression, self.budget)
                value, _, halted = interpret(data, self, sequence=False, location=self.location)
                if halted:
                    raise LoadError('nested pickle load failed')
                return value
            if kind == 'numpy':
                allow_pickle_index = 1 if name.endswith('read_array') else 2
                allowed = args[allow_pickle_index] if len(args) > allow_pickle_index else kw.get('allow_pickle', False)
                if data.startswith(b'\x93NUMPY'):
                    objects, offset = numpy_payload(data)
                    if objects and allowed:
                        value, _, halted = interpret(data[offset:], self, sequence=False, location=self.location + ':array')
                        if halted:
                            raise LoadError('nested NumPy load failed')
                        return value
                    if objects:
                        raise LoadError('NumPy disallows object arrays')
                    return Obj('numpy.ndarray')
                if zipfile.is_zipfile(io.BytesIO(data)):
                    # numpy.load returns an NpzFile; members are loaded lazily
                    # on __getitem__, unlike registry unpacking a top-level NPZ.
                    return Obj('numpy.lib.npyio.NpzFile', (data,), {'allow_pickle': allowed})
                if allowed:
                    value, _, halted = interpret(data, self, sequence=False, location=self.location)
                    if halted:
                        raise LoadError('nested NumPy pickle failed')
                    return value
                raise LoadError('NumPy disallows pickle data')
            self.findings.extend(scan_data(data, self.location, self.budget))
            return UNKNOWN
        finally:
            self.location = outer_location
            self.depth -= 1
