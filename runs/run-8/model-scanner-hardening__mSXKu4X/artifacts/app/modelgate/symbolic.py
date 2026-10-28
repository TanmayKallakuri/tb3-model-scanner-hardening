"""A small abstract Python runtime for pickle reducers.

Objects here are descriptions, not objects imported from an artifact. Only
explicitly implemented operations on inert values run in the scanner. In
particular no find_class, unpickler, eval, marshal code object, or arbitrary
Python callable is used. Mutable descriptions preserve pickle memo aliases.
"""
from __future__ import annotations

import _compat_pickle
import base64
import binascii
import bz2
import codecs
import io
import lzma
import math
import operator
import os
import posixpath
import time
import zlib
from dataclasses import dataclass, field

from modelgate.policy import CALL_CAPABILITIES, NESTED_LOADERS, PYTHON2_MODULE_ALIASES
from modelgate.report import Finding

MAX_ITEMS = 100000
MAX_BYTES = 256 * 1024 * 1024


class Stopped(Exception):
    """The real loader would raise before executing any further opcode."""


class Unsupported(Exception):
    """Analysis cannot safely determine this operation's effects."""


@dataclass(eq=False)
class Symbol:
    kind: str
    name: str = ''
    args: tuple = ()
    attrs: dict = field(default_factory=dict)
    value: object = None
    meta: dict = field(default_factory=dict)


UNKNOWN = Symbol('unknown')
END = Symbol('end')


def canon_module(module):
    if module == 'os.path':
        return 'posixpath'
    return PYTHON2_MODULE_ALIASES.get(module, module)


def canon_name(name):
    parts = name.split('.')
    parts[0] = canon_module(parts[0])
    name = '.'.join(parts)
    for source in ('PosixPath', 'WindowsPath'):
        name = name.replace('pathlib.' + source, 'pathlib.Path')
    return name


# Imported modules exposed by stdlib modules. These are descriptions only;
# importing the artifact's requested module would itself be unsafe.
MODULE_ATTRS = {
    'shutil': {'os': 'os', 'sys': 'sys', 'stat': 'stat', 'collections': 'collections'},
    'subprocess': {'os': 'os', 'sys': 'sys', 'io': 'io', 'builtins': 'builtins'},
    'pathlib': {'os': 'os', 'sys': 'sys', 'io': 'io'},
    'os': {'path': 'posixpath', 'sys': 'sys'},
    'posixpath': {'os': 'os', 'sys': 'sys'},
    'ntpath': {'os': 'os', 'sys': 'sys'},
    'pickle': {'sys': 'sys', 'io': 'io', 'codecs': 'codecs'},
    'functools': {'types': 'types'},
    'operator': {'_operator': 'operator'},
    'tempfile': {'_os': 'os', '_io': 'io', '_sys': 'sys'},
    'io': {'_io': 'io'},
    'importlib': {'sys': 'sys'},
    'pkgutil': {'sys': 'sys', 'os': 'os', 'importlib': 'importlib'},
    'pydoc': {'os': 'os', 'sys': 'sys', 'subprocess': 'subprocess', 'inspect': 'inspect'},
    'platform': {'os': 'os', 'sys': 'sys', 'subprocess': 'subprocess'},
    'webbrowser': {'os': 'os', 'sys': 'sys', 'subprocess': 'subprocess'},
    'http.client': {'socket': 'socket', 'ssl': 'ssl'},
    'urllib.request': {'os': 'os', 'sys': 'sys', 'socket': 'socket', 'http': 'http'},
    'socket': {'os': 'os', 'sys': 'sys', '_socket': 'socket'},
    'logging': {'os': 'os', 'sys': 'sys', 'io': 'io'},
    'logging.config': {'os': 'os', 'sys': 'sys', 'logging': 'logging'},
    'numpy': {'lib': 'numpy.lib', 'core': 'numpy.core', '_core': 'numpy._core'},
    'torch': {'storage': 'torch.storage', '_utils': 'torch._utils'},
    'http': {'client': 'http.client'},
    'urllib': {'request': 'urllib.request', 'parse': 'urllib.parse'},
    'xmlrpc': {'client': 'xmlrpc.client'},
    'concurrent': {'futures': 'concurrent.futures'},
}

CLASS_NAMES = set('''builtins.type builtins.object builtins.str builtins.bytes builtins.bytearray
    builtins.int builtins.float builtins.bool builtins.complex builtins.list builtins.tuple
    builtins.dict builtins.set builtins.frozenset builtins.slice builtins.range
    builtins.memoryview builtins.map builtins.filter builtins.zip builtins.enumerate
    builtins.reversed functools.partial functools.partialmethod functools.cached_property
    operator.attrgetter operator.itemgetter operator.methodcaller
    io.BytesIO io.StringIO io.FileIO io.TextIOWrapper io.BufferedReader io.BufferedWriter
    io.BufferedRandom io.BufferedRWPair socket.socket socket.SocketType
    subprocess.Popen pathlib.Path pathlib.PosixPath pathlib.WindowsPath pathlib.PurePath
    pathlib.PurePosixPath pathlib.PureWindowsPath http.client.HTTPConnection
    http.client.HTTPSConnection ftplib.FTP ftplib.FTP_TLS smtplib.SMTP smtplib.SMTP_SSL
    smtplib.LMTP xmlrpc.client.ServerProxy threading.Thread threading.Timer
    concurrent.futures.ThreadPoolExecutor concurrent.futures.ProcessPoolExecutor
    multiprocessing.Process multiprocessing.pool.Pool multiprocessing.pool.ThreadPool
    timeit.Timer code.InteractiveInterpreter code.InteractiveConsole bdb.Bdb pdb.Pdb
    types.FunctionType types.LambdaType types.CodeType types.ModuleType types.SimpleNamespace
    collections.OrderedDict collections.defaultdict collections.deque collections.Counter
    pickle.Unpickler pickle._Unpickler numpy.dtype numpy.ndarray
    logging.FileHandler logging.handlers.RotatingFileHandler logging.handlers.TimedRotatingFileHandler
    logging.handlers.WatchedFileHandler logging.handlers.SocketHandler logging.handlers.DatagramHandler
    logging.handlers.HTTPHandler logging.handlers.SMTPHandler gzip.GzipFile bz2.BZ2File lzma.LZMAFile
    zipfile.ZipFile tarfile.TarFile tempfile.TemporaryDirectory tempfile.SpooledTemporaryFile
    importlib.machinery.SourceFileLoader importlib.machinery.SourcelessFileLoader
    importlib.machinery.ExtensionFileLoader ctypes.CDLL ctypes.PyDLL ctypes.WinDLL ctypes.OleDLL
    sqlite3.Connection numpy.memmap'''.split())

# Pure construction and numeric summaries used by ordinary model pickles.
# This is separate from capability summaries: an unmodelled callable is not
# silently assumed to be harmless.
DATA_MODULES = {
    'math', 'cmath', 'decimal', '_decimal', 'fractions', 'datetime', '_datetime',
    'time', 'calendar', 'numbers', 'uuid', 'ipaddress', 'enum', 'random', '_random',
    'collections', 'collections.abc', '_collections', 'array', 'copyreg',
    'types', 'builtins', 'operator', 'functools', 'weakref', '_weakref',
}
PURE_GLOBALS = set('''os.getenv os.getenvb os.getpid os.getppid os.getuid os.getgid
    os.geteuid os.getegid os.getgroups os.getcwd os.getcwdb os.cpu_count os.uname
    os.urandom sys.intern sys.getrecursionlimit sys.getsizeof
    re.compile re.escape re.match re.search re.fullmatch re.findall re.split
    logging.StreamHandler logging.Formatter logging.Filter logging.NullHandler
    logging.Logger logging.RootLogger logging.Manager logging.getLogger
    logging.handlers.SocketHandler logging.handlers.DatagramHandler
    logging.handlers.HTTPHandler logging.handlers.SMTPHandler
    code.InteractiveInterpreter code.InteractiveConsole bdb.Bdb pdb.Pdb
    contextlib.nullcontext contextlib.suppress contextlib.ExitStack
    threading.Lock threading.RLock threading.Event threading.Condition
    threading.Semaphore threading.BoundedSemaphore threading.Barrier threading.local
    _thread.allocate_lock _thread.RLock'''.split())


def is_class(obj):
    if not isinstance(obj, Symbol):
        return False
    if obj.kind == 'class':
        return True
    if obj.kind != 'global':
        return False
    # ML classes and stdlib exception/types not explicitly enumerated.
    leaf = obj.name.rsplit('.', 1)[-1]
    return obj.name in CLASS_NAMES or (leaf[:1].isupper() and not leaf.isupper())


class Runtime:
    def __init__(self):
        self.findings = []
        self.modules = {}
        self.globals = {}
        self.steps = 0
        self.deadline = time.monotonic() + 45
        self.depth = 0
        self.location = ''
        self.opcode = ''
        self.decoded_bytes = 0
        self.extensions = {}
        self.extension_cache = {}
        self.artifact_files = {}

    def tick(self, n=1):
        self.steps += n
        if self.steps > 1500000 or time.monotonic() > self.deadline:
            raise Unsupported('static analysis work limit exceeded')

    def note(self, kind, detail):
        if len(self.findings) < 64:
            self.findings.append(Finding(kind, self.location, (self.opcode + ' ' + detail).strip()))

    def module(self, name):
        name = canon_module(name)
        if name not in self.modules:
            self.modules[name] = Symbol('module', name)
        return self.modules[name]

    def ref(self, name):
        name = canon_name(name)
        if name not in self.globals:
            self.globals[name] = Symbol('global', name)
        return self.globals[name]

    def class_of(self, obj):
        return obj.meta.get('class') or self.ref(obj.name)

    def bind(self, value, obj):
        if isinstance(value, Symbol):
            if value.kind == 'staticmethod':
                return value.value
            if value.kind == 'classmethod':
                return Symbol('bound', args=(self.class_of(obj),), value=value.value)
            # Python functions are descriptors. C builtins and partials do
            # not acquire an implicit self when stored in a class dictionary.
            if value.kind == 'global' and value.name.startswith(('subprocess.', 'shutil.', 'runpy.', 'timeit.', 'pdb.', 'bdb.', 'code.')):
                return Symbol('bound', args=(obj,), value=value)
        return value

    def global_ref(self, module, name, protocol=4):
        if not isinstance(module, str) or not isinstance(name, str):
            raise Stopped('GLOBAL requires string names')
        if protocol >= 3 and module in _compat_pickle.IMPORT_MAPPING and module not in ('dbm', '_elementtree'):
            raise Stopped('Python 2 module name without fix_imports')
        if protocol < 3:
            pair = _compat_pickle.NAME_MAPPING.get((module, name))
            if pair:
                module, name = pair
            else:
                module = _compat_pickle.IMPORT_MAPPING.get(module, module)
        if '.' in name and protocol < 4:
            raise Stopped('dotted GLOBAL cannot resolve below protocol 4')
        obj = self.module(module)
        for part in name.split('.'):
            obj = self.getattr(obj, part)
        return obj

    def getattr(self, obj, name, default=END):
        self.tick()
        if not isinstance(name, str):
            raise Stopped('attribute name is not a string')
        if isinstance(obj, Symbol):
            if obj.kind == 'partial':
                if name == 'func':
                    return obj.value
                if name == 'args':
                    return obj.args
                if name == 'keywords':
                    return obj.meta.setdefault('keywords', {})
            if name in ('__getattribute__', '__setattr__', '__delattr__'):
                return Symbol('method', name, (obj,))
            if name == '__call__' and obj.kind in ('partial', 'wrapper', 'bound', 'method', 'attrgetter', 'itemgetter', 'methodcaller', 'decorator'):
                return Symbol('method', name, (obj,))
            if name in obj.attrs:
                return obj.attrs[name]
            if name == '__dict__':
                if obj.kind in ('module', 'global', 'class'):
                    return Symbol('namespace', value=obj)
                return obj.attrs
            if name == '__class__':
                if obj.kind == 'instance':
                    return self.ref(obj.name)
                return self.ref('builtins.type' if obj.kind in ('global', 'class') else 'types.ModuleType')
            if name == '__builtins__':
                return self.module('builtins')
            if obj.kind == 'module':
                if obj.name == 'sys' and name == 'modules':
                    return Symbol('modules')
                target = MODULE_ATTRS.get(obj.name, {}).get(name)
                if target:
                    return self.module(target)
                # Standard package submodules, as exposed by import_module.
                full = obj.name + '.' + name
                if full in MODULE_ATTRS or full in ('logging.config', 'numpy.lib.format', 'numpy.lib.npyio'):
                    return self.module(full)
                return self.ref(full)
            if obj.kind in ('global', 'class'):
                if name in ('__globals__', 'func_globals'):
                    owner = obj.name.rsplit('.', 1)[0]
                    return Symbol('namespace', value=self.module(owner))
                if name in ('__func__', '__wrapped__'):
                    return obj
                return self.ref(obj.name + '.' + name)
            if obj.kind == 'wrapper' and name in ('__wrapped__', 'func'):
                return obj.value
            if obj.kind in ('instance', 'path', 'file', 'partial', 'iterator', 'wrapper', 'array', 'unpickler', 'namespace', 'modules', 'decompressor', 'npz', 'archive'):
                if obj.kind == 'instance':
                    cls = self.class_of(obj)
                    custom = cls.attrs.get(name, END)
                    if custom is not END:
                        return self.bind(custom, obj)
                return Symbol('method', name, (obj,))
            if obj.kind == 'unknown':
                return UNKNOWN
        if isinstance(obj, (str, bytes, bytearray, list, tuple, dict, set, frozenset, int, float, complex)):
            if name == '__class__':
                return self.ref('builtins.' + type(obj).__name__)
            # Only the explicit method summaries below are ever evaluated.
            if hasattr(type(obj), name):
                return Symbol('method', name, (obj,))
        if default is not END:
            return default
        raise Stopped('attribute does not exist')

    def getitem(self, obj, key):
        if isinstance(obj, Symbol):
            if obj.kind == 'npz':
                import zipfile
                from modelgate.dispatch import _npy
                with zipfile.ZipFile(io.BytesIO(obj.value)) as zf:
                    target = key if key in zf.namelist() else key + '.npy'
                    raw = zf.read(target)
                if raw.startswith(b'\x93NUMPY'):
                    return _npy(raw, obj.attrs['location'] + '!' + target, self, bool(obj.attrs['allow_pickle']))
                return raw
            if obj.kind == 'modules':
                return self.module(key) if isinstance(key, str) else UNKNOWN
            if obj.kind == 'namespace':
                return self.getattr(obj.value, key)
            if obj.kind == 'instance':
                fn = self.class_of(obj).attrs.get('__getitem__')
                if fn is not None:
                    return self.call(self.bind(fn, obj), (key,))
            if obj.kind == 'array' and isinstance(obj.value, (list, tuple)):
                return self.getitem(obj.value, key)
            return UNKNOWN
        try:
            return obj[key]
        except (TypeError, IndexError, KeyError):
            raise Stopped('invalid item lookup') from None

    def setitem(self, obj, key, value):
        if isinstance(obj, Symbol):
            if obj.kind == 'namespace':
                obj.value.attrs[key] = value
            elif obj.kind == 'modules':
                self.modules[canon_module(key)] = value
            elif obj.kind == 'instance':
                fn = self.class_of(obj).attrs.get('__setitem__')
                if fn is not None:
                    self.call(self.bind(fn, obj), (key, value))
                else:
                    obj.attrs.setdefault('_items', {})[key] = value
            else:
                obj.attrs.setdefault('_items', {})[key] = value
        else:
            try:
                obj[key] = value
            except (TypeError, IndexError):
                raise Stopped('invalid item assignment') from None

    def build(self, obj, state):
        if isinstance(obj, Symbol) and obj.kind == 'partial':
            if not isinstance(state, tuple) or len(state) != 4 or not isinstance(state[1], tuple):
                raise Stopped('invalid functools.partial state')
            if not isinstance(state[0], Symbol) or not isinstance(state[2], (dict, type(None))) or not isinstance(state[3], (dict, type(None))):
                raise Stopped('invalid functools.partial state types')
            obj.value, obj.args, keywords, namespace = state
            obj.attrs = namespace if namespace is not None else {}
            obj.meta['keywords'] = keywords if keywords is not None else {}
            return
        if isinstance(obj, Symbol):
            custom = obj.attrs.get('__setstate__')
            if custom is None and obj.kind == 'instance':
                custom = self.class_of(obj).attrs.get('__setstate__')
                if custom is not None:
                    self.call(self.bind(custom, obj), (state,))
                    return
            if custom is not None:
                self.call(custom, (state,))
                return
            if obj.kind == 'array':
                if isinstance(state, tuple) and len(state) >= 5:
                    obj.value = state[-1]
                return
            if obj.kind == 'file':
                if isinstance(state, tuple) and state:
                    obj.value = state[0]
                    obj.attrs['position'] = state[1] if len(state) > 1 else 0
                return
            if obj.name in ('numpy.dtype', 'numpy.dtypes.ObjectDType'):
                obj.attrs['_state'] = state
                return
        slotstate = None
        if isinstance(state, tuple) and len(state) == 2:
            state, slotstate = state
        if isinstance(obj, Symbol):
            if isinstance(state, dict):
                obj.attrs.update(state)
            elif isinstance(state, Symbol) and state.kind == 'namespace':
                obj.attrs.update(state.value.attrs)
            elif state is not None:
                # Most native reconstruction types have their own inert state
                # layouts (random, datetime, numpy, sklearn trees, etc.).
                obj.attrs['_state'] = state
            if isinstance(slotstate, dict):
                obj.attrs.update(slotstate)
        # OrderedDict and other concrete safe containers can have attributes
        # in CPython even though our inert dict/list representation cannot.

    def newobj(self, cls, args, kwargs=END):
        if kwargs is END:
            kwargs = {}
        if not isinstance(args, tuple) or not isinstance(kwargs, dict):
            raise Stopped('NEWOBJ arguments have the wrong type')
        if not is_class(cls):
            raise Stopped('NEWOBJ requires a type')
        name = cls.name
        if name in ('functools.partial', 'operator.attrgetter', 'operator.itemgetter',
                    'operator.methodcaller', 'types.FunctionType', 'types.LambdaType',
                    'builtins.str', 'builtins.bytes', 'builtins.tuple',
                    'builtins.int', 'builtins.float', 'builtins.frozenset', 'builtins.complex'):
            return self.call(cls, args, kwargs)
        if name.startswith('pathlib.'):
            return Symbol('path', name, args, value=self.path_string(args))
        if name in ('builtins.list', 'collections.deque'):
            return []
        if name == 'builtins.bytearray':
            return bytearray()
        if name == 'builtins.set':
            return set()
        if name == 'builtins.dict' or name == 'collections.OrderedDict':
            return {}
        if name in ('io.BytesIO', 'io.StringIO'):
            return Symbol('file', name, value=b'' if name.endswith('BytesIO') else '', attrs={'position': 0})
        custom = cls.attrs.get('__new__')
        if custom is not None:
            return self.call(custom, (cls,) + args, kwargs)
        return Symbol('instance', name, args, meta={'class': cls})

    def construct(self, cls, args):
        if not args and is_class(cls):
            return self.newobj(cls, ())
        return self.call(cls, args)

    def path_string(self, args):
        parts = [x.value if isinstance(x, Symbol) and x.kind == 'path' else x for x in args]
        if all(isinstance(x, str) for x in parts):
            return posixpath.join(*parts) if parts else '.'
        return UNKNOWN

    def artifact_content(self, path):
        if isinstance(path, Symbol) and path.kind == 'path':
            path = path.value
        if isinstance(path, (str, bytes)):
            return self.artifact_files.get(os.path.abspath(os.fsdecode(path)), END)
        return END

    def call(self, fn, args=(), kwargs=None):
        self.tick()
        if not isinstance(args, tuple):
            raise Stopped('REDUCE requires a tuple')
        kwargs = kwargs or {}
        if not isinstance(kwargs, dict):
            raise Stopped('keyword arguments require a dict')
        self.depth += 1
        if self.depth > 100:
            self.depth -= 1
            raise Unsupported('call analysis recursion limit exceeded')
        try:
            return self._call(fn, args, kwargs)
        except (ValueError, TypeError, IndexError, KeyError, AttributeError, OverflowError, ZeroDivisionError, UnicodeError, binascii.Error):
            # These can arise only from the explicitly listed pure operations
            # on concrete inert values, and correspond to a loader exception.
            raise Stopped('invalid arguments to ' + getattr(fn, 'name', 'callable')) from None
        finally:
            self.depth -= 1

    def _call(self, fn, args, kw):
        if not isinstance(fn, Symbol):
            raise Stopped('object is not callable')
        if fn.kind == 'partial':
            keywords = dict(fn.meta.get('keywords', {}))
            keywords.update(kw)
            return self.call(fn.value, fn.args + args, keywords)
        if fn.kind == 'wrapper':
            return self.call(fn.value, args, kw)
        if fn.kind == 'bound':
            return self.call(fn.value, fn.args + args, kw)
        if fn.kind == 'method':
            return self.method(fn.args[0], fn.name, args, kw)
        if fn.kind == 'attrgetter':
            out = []
            for path in fn.args:
                obj = args[0]
                for part in path.split('.'):
                    obj = self.getattr(obj, part)
                out.append(obj)
            return out[0] if len(out) == 1 else tuple(out)
        if fn.kind == 'itemgetter':
            out = [self.getitem(args[0], k) for k in fn.args]
            return out[0] if len(out) == 1 else tuple(out)
        if fn.kind == 'methodcaller':
            return self.call(self.getattr(args[0], fn.name), fn.args, fn.attrs.get('_keywords'))
        if fn.kind == 'decorator':
            return Symbol('wrapper', fn.name, value=args[0])
        if fn.kind == 'instance':
            custom = self.class_of(fn).attrs.get('__call__')
            if custom is not None:
                return self.call(self.bind(custom, fn), args, kw)
            if fn.name == 'xmlrpc.client.ServerProxy':
                self.note('network', 'XML-RPC request')
            return UNKNOWN
        if fn.kind == 'unknown':
            raise Unsupported('unresolved callable')
        if fn.kind not in ('global', 'class'):
            raise Stopped('object is not callable')
        n = fn.name
        self.validate_call(n, args, kw)
        if n.endswith('.__call__'):
            base = n[:-9]
            if base in ('builtins.type', 'types.FunctionType', 'types.BuiltinFunctionType',
                        'functools.partial', 'operator.attrgetter', 'operator.itemgetter', 'operator.methodcaller'):
                return self.call(args[0], args[1:], kw)
            return self.call(self.ref(base), args, kw)
        if n in ('builtins.bytes.fromhex', 'builtins.bytearray.fromhex'):
            return bytes.fromhex(args[0]) if n.startswith('builtins.bytes.') else bytearray.fromhex(args[0])

        if n in ('builtins.getattr', 'builtins.hasattr'):
            result = self.getattr(args[0], args[1], args[2] if len(args) > 2 else END)
            return True if n.endswith('hasattr') else result
        if n in ('builtins.setattr', 'builtins.delattr'):
            if isinstance(args[0], Symbol):
                if n.endswith('setattr'):
                    args[0].attrs[args[1]] = args[2]
                else:
                    args[0].attrs.pop(args[1], None)
            return None
        if n == 'builtins.__import__':
            name = args[0]
            fromlist = args[3] if len(args) > 3 else kw.get('fromlist', ())
            return self.module(name if fromlist else name.split('.')[0])
        if n in ('importlib.import_module', 'importlib.__import__'):
            name = args[0]
            if isinstance(name, str) and name.startswith('.'):
                package = args[1] if len(args) > 1 else kw.get('package', '')
                name = package + name
            return self.module(name)
        if n == 'pkgutil.resolve_name':
            return self.resolve_name(args[0])
        if n == 'builtins.vars':
            return self.getattr(args[0], '__dict__') if args else Symbol('namespace', value=self.module('builtins'))
        if n in ('builtins.globals', 'builtins.locals'):
            return Symbol('namespace', value=self.module('builtins'))
        if n == 'operator.attrgetter':
            return Symbol('attrgetter', args=args)
        if n == 'operator.itemgetter':
            return Symbol('itemgetter', args=args)
        if n == 'operator.methodcaller':
            return Symbol('methodcaller', args[0], args[1:], {'_keywords': kw})
        if n in ('functools.partial', 'functools.partialmethod'):
            if not args or not isinstance(args[0], Symbol):
                raise Stopped('partial requires a callable')
            return Symbol('partial', args=args[1:], meta={'keywords': dict(kw)}, value=args[0])
        if n in ('functools.lru_cache', 'functools.cache'):
            if args and isinstance(args[0], Symbol):
                return Symbol('wrapper', n, value=args[0])
            return Symbol('decorator', n)
        if n == 'functools._lru_cache_wrapper':
            return Symbol('wrapper', n, value=args[0])
        if n in ('functools.update_wrapper', 'functools.wraps'):
            if n.endswith('wraps'):
                return Symbol('decorator', n)
            args[0].attrs['__wrapped__'] = args[1]
            return args[0]
        if n in ('copy.copy', 'copy.deepcopy'):
            obj = args[0]
            if isinstance(obj, Symbol) and obj.kind not in ('global', 'module', 'class'):
                return Symbol(obj.kind, obj.name, obj.args, dict(obj.attrs), obj.value, dict(obj.meta))
            if isinstance(obj, (list, dict, set, bytearray)):
                return obj.copy() if hasattr(obj, 'copy') else bytearray(obj)
            return obj
        if n in ('operator.call', 'builtins.apply'):
            return self.call(args[0], args[1:] if n == 'operator.call' else tuple(args[1]), kw)
        if n == 'copyreg.add_extension':
            if len(args) != 3 or not isinstance(args[2], int) or args[2] <= 0:
                raise Stopped('invalid extension registration')
            self.extensions[args[2]] = args[:2]
            return None
        if n == 'copyreg.remove_extension':
            self.extensions.pop(args[2], None)
            self.extension_cache.pop(args[2], None)
            return None
        if n == 'copyreg.clear_extension_cache':
            self.extension_cache.clear()
            return None
        if n in ('copyreg._reconstructor', 'copyreg.__newobj__', 'copyreg.__newobj_ex__'):
            if n.endswith('__newobj_ex__'):
                return self.newobj(args[0], args[1], args[2])
            if n.endswith('__newobj__'):
                return self.newobj(args[0], args[1:])
            obj = self.newobj(args[0], ())
            if len(args) > 1 and isinstance(args[1], Symbol) and args[1].name != 'builtins.object':
                self.call(args[1], (args[2],))
            return obj
        if n == 'builtins.type':
            if len(args) == 1:
                return self.getattr(args[0], '__class__')
            if len(args) == 3 and isinstance(args[2], dict):
                return Symbol('class', args[0], args=tuple(args[1]), attrs=dict(args[2]))
        if n == 'types.ModuleType':
            return Symbol('module', args[0] if args else '')
        if n in ('types.SimpleNamespace',):
            return Symbol('instance', n, attrs=dict(kw))
        if n == 'types.MethodType':
            return Symbol('bound', args=(args[1],), value=args[0])
        if n == 'collections.namedtuple':
            fields = args[1].replace(',', ' ').split() if isinstance(args[1], str) else list(args[1])
            if not isinstance(args[0], str) or not args[0].isidentifier():
                raise Stopped('invalid namedtuple type name')
            return Symbol('class', args[0], attrs={'_fields': tuple(fields)})
        if n in ('builtins.staticmethod', 'builtins.classmethod'):
            return Symbol(n.split('.')[1], value=args[0])

        # The classes below connect/open only in __init__, which NEWOBJ skips.
        if n in ('ftplib.FTP', 'ftplib.FTP_TLS', 'smtplib.SMTP', 'smtplib.SMTP_SSL', 'smtplib.LMTP'):
            host = args[0] if args else kw.get('host', '')
            if host:
                self.note('network', n + '(host)')
            return Symbol('instance', n, args, dict(kw))
        if n in ('socket.socket', 'socket.SocketType', 'http.client.HTTPConnection', 'http.client.HTTPSConnection',
                 'xmlrpc.client.ServerProxy', 'urllib.request.Request'):
            return Symbol('instance', n, args, dict(kw))
        if n in ('threading.Thread', 'threading.Timer', 'multiprocessing.Process', 'multiprocessing.context.Process',
                 'multiprocessing.context.ForkProcess', 'multiprocessing.context.SpawnProcess'):
            attrs = dict(kw)
            if n == 'threading.Timer':
                attrs['target'] = args[1] if len(args) > 1 else kw.get('function', UNKNOWN)
                attrs['args'] = args[2] if len(args) > 2 else kw.get('args', ())
                attrs['kwargs'] = args[3] if len(args) > 3 else kw.get('kwargs', {})
            else:
                for i, key in enumerate(('group', 'target', 'name', 'args', 'kwargs')):
                    if len(args) > i:
                        attrs[key] = args[i]
            return Symbol('instance', n, args, attrs)
        if n in ('concurrent.futures.ThreadPoolExecutor', 'concurrent.futures.ProcessPoolExecutor'):
            return Symbol('instance', n, args, dict(kw))
        if n in ('multiprocessing.pool.ThreadPool', 'multiprocessing.dummy.Pool'):
            attrs = dict(kw)
            for i, key in enumerate(('processes', 'initializer', 'initargs')):
                if len(args) > i:
                    attrs[key] = args[i]
            if attrs.get('initializer') is not None:
                self.call(attrs['initializer'], tuple(attrs.get('initargs', ())))
            return Symbol('instance', 'multiprocessing.pool.ThreadPool', args, attrs)
        if n in ('_thread.start_new_thread', '_thread.start_new', 'threading._start_new_thread'):
            self.call(args[0], tuple(args[1]), args[2] if len(args) > 2 else {})
            return UNKNOWN
        if n.startswith('pathlib.') and n.split('.')[-1] in ('Path', 'PosixPath', 'WindowsPath', 'PurePath', 'PurePosixPath', 'PureWindowsPath'):
            return Symbol('path', n, args, value=self.path_string(args))
        if n in ('io.BytesIO', 'io.StringIO'):
            return Symbol('file', n, value=args[0] if args else (b'' if n.endswith('BytesIO') else ''), attrs={'position': 0})
        if n in ('io.BufferedReader', 'io.BufferedRandom', 'io.BufferedWriter', 'io.TextIOWrapper'):
            source = args[0] if args else kw.get('buffer', kw.get('raw', UNKNOWN))
            if isinstance(source, Symbol) and source.kind == 'file':
                meta = {'wrapped': source}
                if n == 'io.TextIOWrapper':
                    meta['encoding'] = args[1] if len(args) > 1 else kw.get('encoding', 'utf-8')
                return Symbol('file', n, attrs={'position': 0}, meta=meta)
            return Symbol('instance', n, args, dict(kw))
        if n == 'tempfile.SpooledTemporaryFile':
            return Symbol('file', n, value=b'', attrs={'position': 0, 'max_size': args[0] if args else kw.get('max_size', 0)})
        if n in ('pickle.Unpickler', 'pickle._Unpickler'):
            return Symbol('unpickler', n, value=args[0])
        if n in ('io.FileIO', 'io.open', 'builtins.open', 'gzip.open', 'gzip.GzipFile', 'bz2.open', 'bz2.BZ2File',
                 'lzma.open', 'lzma.LZMAFile', 'zipfile.ZipFile', 'tarfile.open', 'tarfile.TarFile.open'):
            source = kw.get('fileobj')
            if source is None and n == 'gzip.GzipFile' and len(args) > 3:
                source = args[3]
            if source is None:
                source = args[0] if args else kw.get('file', kw.get('filename', kw.get('name', UNKNOWN)))
            if not isinstance(source, Symbol) or source.kind not in ('file',):
                own = self.artifact_content(source)
                if own is not END and n in ('io.FileIO', 'io.open', 'builtins.open'):
                    mode = args[1] if len(args) > 1 else kw.get('mode', 'r')
                    raw = b'' if 'w' in mode else own
                    if 'b' not in mode and n != 'io.FileIO':
                        if 'w' in mode:
                            return Symbol('file', n, value='', attrs={'position': 0})
                        binary = Symbol('file', n, value=raw, attrs={'position': 0})
                        return Symbol('file', n, attrs={'position': 0},
                                      meta={'wrapped': binary, 'encoding': kw.get('encoding') or 'utf-8'})
                    return Symbol('file', n, value=raw, attrs={'position': len(raw) if 'a' in mode else 0})
                self.note('file-access', n)
                return Symbol('instance', n, args)
            if n.startswith(('gzip.', 'bz2.', 'lzma.')):
                # These constructors do not read/decompress until requested.
                return Symbol('file', n, attrs={'position': 0},
                              meta={'wrapped': source, 'compression': n.split('.')[0]})
            if n.startswith(('zipfile.', 'tarfile.')):
                data = self.file_bytes(source)
                # Validate the header now, as the real archive constructor does.
                if n.startswith('zipfile.'):
                    import zipfile
                    with zipfile.ZipFile(io.BytesIO(data)):
                        pass
                    return Symbol('archive', 'zip', value=data)
                import tarfile
                with tarfile.open(fileobj=io.BytesIO(data)):
                    pass
                return Symbol('archive', 'tar', value=data)
            raise Stopped('open expects a path or file descriptor')
        if n in ('logging.FileHandler', 'logging.handlers.RotatingFileHandler',
                 'logging.handlers.TimedRotatingFileHandler', 'logging.handlers.WatchedFileHandler'):
            delay_index = 3 if n == 'logging.FileHandler' else (5 if 'Rotating' in n else 3)
            delay = kw.get('delay', args[delay_index] if len(args) > delay_index else False)
            if not delay:
                self.note('file-access', n)
            return Symbol('instance', n, args, dict(kw))
        if n == 'timeit.Timer':
            # Timer compiles user-provided source during construction.
            if any(isinstance(v, str) and v for v in args[:2]) or any(isinstance(kw.get(k), str) for k in ('stmt', 'setup')):
                self.note('code-execution', n)
            attrs = dict(kw)
            for i, key in enumerate(('stmt', 'setup', 'timer', 'globals')):
                if len(args) > i:
                    attrs[key] = args[i]
            return Symbol('instance', n, args, attrs)
        if n in ('timeit.timeit', 'timeit.repeat'):
            names = ('stmt', 'setup', 'timer', 'number', 'globals') if n.endswith('.timeit') else ('stmt', 'setup', 'timer', 'repeat', 'number', 'globals')
            params = dict(kw)
            params.update(zip(names, args))
            timer = self.call(self.ref('timeit.Timer'), (), {k: v for k, v in params.items() if k in ('stmt', 'setup', 'timer', 'globals')})
            return self.method(timer, n.split('.')[1], (), {k: v for k, v in params.items() if k in ('repeat', 'number')})
        if n == 'marshal.loads':
            # Treat serialized bytecode as inert until FunctionType/call.
            from modelgate.marshal_scan import read_marshaled
            if not isinstance(args[0], (bytes, bytearray)):
                raise Stopped('marshal requires a bytes-like object')
            return read_marshaled(args[0], self)[0]
        if n == 'marshal.load':
            if args and isinstance(args[0], Symbol) and args[0].kind == 'file':
                from modelgate.marshal_scan import read_marshaled
                result, used = read_marshaled(self.file_bytes(args[0]), self)
                args[0].attrs['position'] = args[0].attrs.get('position', 0) + used
                return result
            raise Stopped('marshal.load requires a readable file')
        if n == 'types.CodeType':
            return Symbol('code', n, args)
        if n in ('torch._tensor._rebuild_from_type_v2', 'torch._tensor._rebuild_from_type'):
            result = self.call(args[0], args[2])
            if len(args) > 3:
                self.build(result, args[3])
            return result
        if n in ('torch._utils._rebuild_wrapper_subclass',):
            return self.newobj(args[0], args[1:], kw)

        if n == 'logging.config.dictConfig':
            return self.logging_config(args[0])

        if n in ('numpy.save', 'numpy.savez', 'numpy.savez_compressed', 'numpy.savetxt',
                 'torch.save', 'joblib.dump', 'pickle.dump', 'marshal.dump'):
            index = 1 if n in ('torch.save', 'joblib.dump', 'pickle.dump', 'marshal.dump') else 0
            dest = args[index] if len(args) > index else kw.get('file', kw.get('f', kw.get('filename', UNKNOWN)))
            if not isinstance(dest, Symbol) or dest.kind != 'file':
                self.note('file-access', n)
            return None

        if n in NESTED_LOADERS:
            return self.nested(n, args, kw)
        if n in CALL_CAPABILITIES:
            self.note(CALL_CAPABILITIES[n], n)
            return UNKNOWN
        # Descriptor forms such as GLOBAL socket.socket.connect and Path.open.
        for cls in sorted(CLASS_NAMES, key=len, reverse=True):
            if n.startswith(cls + '.'):
                method = n[len(cls) + 1:]
                if method == '__new__':
                    return self.newobj(args[0], args[1:], kw)
                if method == '__init__':
                    return self.call(self.ref(cls), args[1:], kw)
                if args:
                    return self.method(args[0], method, args[1:], kw, cls)

        if n in ('builtins.map', 'builtins.filter', 'itertools.starmap', 'itertools.filterfalse',
                 'itertools.takewhile', 'itertools.dropwhile'):
            return Symbol('iterator', n, args)
        if n in ('builtins.zip', 'builtins.enumerate', 'builtins.reversed', 'itertools.chain',
                 'itertools.chain.from_iterable', 'itertools.islice', 'itertools.repeat',
                 'itertools.count', 'itertools.cycle', 'itertools.accumulate',
                 'itertools.compress', 'itertools.zip_longest'):
            return Symbol('iterator', n, args, dict(kw))
        if n == 'builtins.iter':
            return Symbol('iterator', 'calliter', args) if len(args) == 2 else self.iterator(args[0])
        if n == 'builtins.next':
            value = self.next(args[0])
            if value is END:
                if len(args) > 1:
                    return args[1]
                raise Stopped('iterator exhausted')
            return value
        if n in ('builtins.list', 'builtins.tuple', 'builtins.set', 'builtins.frozenset', 'collections.deque'):
            out = self.consume(args[0]) if args else []
            if n.endswith('tuple'):
                return tuple(out)
            if n in ('builtins.set', 'builtins.frozenset'):
                return set(out) if n.endswith('.set') else frozenset(out)
            return out
        if n in ('builtins.dict', 'collections.OrderedDict', 'collections.Counter'):
            result = {}
            if args:
                if isinstance(args[0], dict):
                    result.update(args[0])
                elif isinstance(args[0], Symbol) and args[0].kind == 'namespace':
                    result.update(args[0].value.attrs)
                else:
                    result.update(self.consume(args[0]))
            result.update(kw)
            return result
        if n == 'collections.defaultdict':
            return Symbol('instance', n, args, {'default_factory': args[0] if args else None, '_items': dict(args[1]) if len(args) > 1 else {}})
        if n in ('builtins.sorted', 'builtins.min', 'builtins.max'):
            values = self.consume(args[0]) if len(args) == 1 else list(args)
            key = kw.get('key')
            keys = [self.call(key, (value,)) for value in values] if key is not None else values
            if not values:
                if n == 'builtins.sorted':
                    return []
                if 'default' in kw:
                    return kw['default']
                raise Stopped('empty min/max')
            if all(isinstance(v, (str, bytes, int, float)) for v in keys):
                order = sorted(range(len(values)), key=lambda i: keys[i], reverse=bool(kw.get('reverse', False)))
                if n == 'builtins.min':
                    return values[order[0]]
                if n == 'builtins.max':
                    return values[order[-1]]
                return [values[i] for i in order]
            return values if n == 'builtins.sorted' else values[0]
        if n == 'functools.reduce':
            values = self.iterator(args[1])
            acc = args[2] if len(args) > 2 else self.next(values)
            if acc is END:
                raise Stopped('empty reduce')
            while True:
                value = self.next(values)
                if value is END:
                    return acc
                acc = self.call(args[0], (acc, value))
        if n in ('builtins.any', 'builtins.all', 'builtins.sum'):
            it = self.iterator(args[0])
            result = args[1] if len(args) > 1 else 0
            while True:
                value = self.next(it)
                if value is END:
                    return result if n.endswith('sum') else n.endswith('all')
                if n.endswith('any') and self.truth(value):
                    return True
                if n.endswith('all') and not self.truth(value):
                    return False
                if n.endswith('sum'):
                    result = self.binary('add', result, value)
        if n == 'builtins.range':
            value = range(*args)
            if len(value) > MAX_ITEMS:
                return Symbol('iterator', 'range', args)
            return value
        if n == 'builtins.len':
            return len(args[0]) if not isinstance(args[0], Symbol) else UNKNOWN
        if n == 'builtins.slice':
            return slice(*args)
        if n == 'builtins.bool':
            return self.truth(args[0]) if args else False
        if n == 'builtins.callable':
            return isinstance(args[0], Symbol) and args[0].kind in ('global', 'partial', 'method', 'wrapper')
        if n in ('builtins.str', 'builtins.repr', 'builtins.ascii', 'builtins.int', 'builtins.float', 'builtins.complex',
                 'builtins.bytes', 'builtins.bytearray', 'builtins.memoryview', 'builtins.chr', 'builtins.ord',
                 'builtins.hex', 'builtins.oct', 'builtins.bin', 'builtins.abs', 'builtins.round'):
            if any(isinstance(a, Symbol) for a in args):
                if n == 'builtins.str' and len(args) == 1 and args[0].kind == 'path':
                    return args[0].value
                return UNKNOWN
            funcs = {'str': str, 'repr': repr, 'ascii': ascii, 'int': int, 'float': float, 'complex': complex,
                     'bytes': bytes, 'bytearray': bytearray, 'memoryview': bytes, 'chr': chr, 'ord': ord,
                     'hex': hex, 'oct': oct, 'bin': bin, 'abs': abs, 'round': round}
            if n in ('builtins.bytes', 'builtins.bytearray') and args and isinstance(args[0], int) and args[0] > MAX_BYTES:
                raise Unsupported('byte allocation limit exceeded')
            return funcs[n.split('.')[1]](*args, **kw)
        if n == 'os.fspath':
            return args[0].value if isinstance(args[0], Symbol) and args[0].kind == 'path' else args[0]
        if n in ('os.fsencode', 'os.fsdecode'):
            value = args[0].value if isinstance(args[0], Symbol) and args[0].kind == 'path' else args[0]
            if n.endswith('fsencode'):
                return value.encode() if isinstance(value, str) else value
            return value.decode() if isinstance(value, bytes) else value
        if n.startswith(('posixpath.', 'ntpath.')) or n.startswith('os.path.'):
            leaf = n.rsplit('.', 1)[-1]
            if leaf in ('join', 'basename', 'dirname', 'split', 'splitext', 'normpath', 'isabs'):
                return getattr(posixpath, leaf)(*args)
            if leaf in ('exists', 'lexists', 'isfile', 'isdir', 'islink', 'getsize', 'getmtime', 'getctime', 'getatime', 'samefile'):
                self.note('file-access', n)
                return UNKNOWN
        if n.startswith('operator.'):
            leaf = n.split('.')[1].strip('_')
            if leaf in ('getitem', 'setitem'):
                if leaf == 'getitem':
                    return self.getitem(*args)
                self.setitem(*args)
                return None
            if leaf in ('add', 'concat', 'iadd', 'iconcat', 'mul', 'imul', 'sub', 'truediv', 'floordiv', 'mod', 'pow', 'or', 'and', 'xor', 'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'is', 'is_not'):
                return self.binary(leaf, *args)
        if n in ('_codecs.encode', 'codecs.encode', '_codecs.decode', 'codecs.decode'):
            if isinstance(args[0], (str, bytes, bytearray)):
                return self.codec(n.endswith('encode'), *args)
            return UNKNOWN
        if n.startswith('base64.'):
            leaf = n.split('.')[1]
            allowed = {'b64decode', 'standard_b64decode', 'urlsafe_b64decode', 'decodebytes',
                       'b32decode', 'b16decode', 'a85decode', 'b85decode', 'b64encode', 'standard_b64encode', 'urlsafe_b64encode'}
            if leaf in allowed:
                return getattr(base64, leaf)(*args, **kw)
        if n.startswith('binascii.'):
            leaf = n.split('.')[1]
            if leaf in ('unhexlify', 'a2b_hex', 'a2b_base64', 'hexlify', 'b2a_hex', 'b2a_base64'):
                return getattr(binascii, leaf)(*args, **kw)
        if n in ('struct.pack', 'struct.unpack', 'struct.calcsize'):
            import struct
            if not all(isinstance(x, (str, bytes, int, float)) for x in args):
                return UNKNOWN
            if args and struct.calcsize(args[0]) > MAX_BYTES:
                raise Unsupported('struct buffer size limit exceeded')
            return getattr(struct, n.split('.')[1])(*args)
        if n in ('zlib.decompress', 'gzip.decompress', 'bz2.decompress', 'lzma.decompress'):
            return self.decompress(args[0], n.split('.')[0], args[1:] or ())
        if n == 'zlib.decompressobj':
            return Symbol('decompressor', 'zlib', args)
        if n in ('bz2.BZ2Decompressor', 'lzma.LZMADecompressor'):
            return Symbol('decompressor', n.split('.')[0], args)
        if n in ('json.loads', 'ast.literal_eval'):
            if n.startswith('json'):
                import json
                if kw.get('object_hook') is not None:
                    data = json.loads(args[0])
                    return self.call(kw['object_hook'], (data,))
                return json.loads(args[0])
            import ast
            return ast.literal_eval(args[0])
        if n in ('numpy.core.multiarray._reconstruct', 'numpy._core.multiarray._reconstruct', 'numpy.ndarray'):
            return Symbol('array', 'numpy.ndarray', args)
        if n in ('numpy.array', 'numpy.asarray'):
            return Symbol('array', 'numpy.ndarray', args, value=args[0])
        if n in ('numpy.core.numeric._frombuffer', 'numpy._core.numeric._frombuffer'):
            return Symbol('array', 'numpy.ndarray', args)
        if n == 'numpy.dtype':
            return Symbol('instance', n, args)
        if n in ('numpy.core.multiarray.scalar', 'numpy._core.multiarray.scalar'):
            return UNKNOWN
        if n.startswith(('dill._dill.', 'cloudpickle.cloudpickle.')):
            leaf = n.rsplit('.', 1)[-1]
            if leaf in ('_create_function', '_make_function', '_make_skeleton_function'):
                self.note('code-execution', n + ' materializes serialized Python function')
            if leaf in ('_create_code', '_make_code'):
                return Symbol('code', n, args)
            if leaf in ('_import_module', 'subimport'):
                return self.module(args[0])
        if n in ('importlib.machinery.SourceFileLoader', 'importlib.machinery.SourcelessFileLoader', 'importlib.machinery.ExtensionFileLoader'):
            return Symbol('instance', n, args)
        if n.startswith('ctypes.') and n.rsplit('.', 1)[-1] in ('CDLL', 'PyDLL', 'WinDLL', 'OleDLL', 'LoadLibrary'):
            if args and args[0]:
                self.note('code-execution', n + ' loads native library')
            return Symbol('instance', n, args)
        if fn.kind == 'class':
            obj = self.newobj(fn, args, kw)
            init = fn.attrs.get('__init__')
            if init is not None:
                self.call(self.bind(init, obj), args, kw)
            elif fn.args:
                self.call(fn.args[0], args, kw)
            return obj
        # Native model rebuilding routines create data. Their arguments have
        # already been interpreted, so embedded reducers are not skipped.
        if n.startswith(('torch._utils._rebuild_', 'torch._tensor._rebuild_', 'torch._C.',
                         'torch.nn.', 'torch.optim.', 'torch.Size', 'torch.device',
                         'sklearn.', 'scipy.', 'numpy.', 'pandas.')):
            return Symbol('instance', n, args, dict(kw))
        if n in PURE_GLOBALS or n.rpartition('.')[0] in DATA_MODULES:
            return Symbol('instance', n, args, dict(kw)) if is_class(fn) else UNKNOWN
        raise Unsupported('no static call summary for ' + n)

    def validate_call(self, name, args, kw):
        """Reject definitely invalid calls before assigning a capability.

        This is deliberately independent of the host's callable signatures:
        importing a named class or asking a reducer for its signature would
        violate the scanner's static boundary.
        """
        one_arg = {
            'os.system': ('command', 1), 'os.popen': ('cmd', 3),
            'builtins.eval': ('source', 3), 'builtins.exec': ('source', 3),
            'runpy.run_path': ('path_name', 3), 'runpy.run_module': ('mod_name', 4),
            'logging.config.dictConfig': ('config', 1), 'logging.config.fileConfig': ('fname', 4),
            'pickle.loads': ('data', 1), 'pickle.load': ('file', 1),
            'marshal.loads': ('bytes', 1), 'marshal.load': ('file', 1),
            'socket.create_connection': ('address', 3),
            'urllib.request.urlopen': ('url', 3), 'urllib.request.urlretrieve': ('url', 4),
            'webbrowser.open': ('url', 3), 'webbrowser.open_new': ('url', 1),
            'webbrowser.open_new_tab': ('url', 1),
            'builtins.open': ('file', 8), 'io.open': ('file', 8), 'io.FileIO': ('file', 3),
        }
        if name in one_arg:
            first, maxargs = one_arg[name]
            if (not args and first not in kw) or len(args) > maxargs:
                raise Stopped('invalid call arity')
        if name == 'os.system':
            if len(args) != 1 or kw or not isinstance(args[0], (str, bytes)):
                raise Stopped('invalid os.system arguments')
        if name == 'os.popen' and args and not isinstance(args[0], str):
            raise Stopped('invalid os.popen command')
        if name in ('builtins.eval', 'builtins.exec'):
            source = args[0] if args else kw.get('source')
            if not isinstance(source, (str, bytes, Symbol)):
                raise Stopped('source is neither text nor code')
            if len(args) > 1 and args[1] is not None and not isinstance(args[1], dict):
                raise Stopped('globals must be a dict')
        if name == 'builtins.compile':
            if len(args) < 3 and not all(k in kw for k in ('source', 'filename', 'mode')):
                raise Stopped('compile requires source, filename and mode')
        if name in ('types.FunctionType', 'types.LambdaType'):
            if len(args) < 2 or not isinstance(args[0], Symbol) or args[0].kind != 'code' or not isinstance(args[1], dict):
                raise Stopped('FunctionType requires code and globals')
        if name in ('subprocess.Popen', 'subprocess.call', 'subprocess.run', 'subprocess.check_call', 'subprocess.check_output'):
            command = args[0] if args else kw.get('args', END)
            if command is END or isinstance(command, (list, tuple)) and not command:
                raise Stopped('missing process arguments')

    def logging_config(self, config):
        if not isinstance(config, dict) or config.get('version') != 1:
            raise Stopped('invalid logging configuration')
        if config.get('incremental'):
            return None
        for group in ('formatters', 'filters', 'handlers'):
            entries = config.get(group, {})
            if not isinstance(entries, dict):
                raise Stopped('invalid logging configuration group')
            for value in entries.values():
                if not isinstance(value, dict):
                    raise Stopped('invalid logging configuration entry')
                factory = value.get('()', value.get('class') if group == 'handlers' else None)
                if factory is None:
                    continue
                if isinstance(factory, str):
                    factory = self.resolve_name(factory.removeprefix('ext://'))
                kwargs = {k: self.logging_value(v) for k, v in value.items()
                          if k not in ('()', 'class', '.', 'level', 'formatter', 'filters')}
                self.call(factory, (), kwargs)
        return None

    def logging_value(self, value):
        if isinstance(value, str) and value.startswith('ext://'):
            return self.resolve_name(value[6:])
        if isinstance(value, tuple):
            return tuple(self.logging_value(x) for x in value)
        if isinstance(value, list):
            return [self.logging_value(x) for x in value]
        if isinstance(value, dict):
            return {k: self.logging_value(v) for k, v in value.items()}
        return value

    def resolve_name(self, name):
        if ':' in name:
            module, attr = name.split(':', 1)
        else:
            module, _, attr = name.partition('.')
        obj = self.module(module or name)
        for part in filter(None, attr.split('.')):
            obj = self.getattr(obj, part)
        return obj

    def method(self, obj, name, args, kw, class_name=None):
        self.tick()
        if isinstance(obj, Symbol):
            if name in ('__call__', 'call') and obj.kind in ('global', 'partial', 'wrapper', 'bound', 'method', 'attrgetter', 'itemgetter', 'methodcaller', 'decorator'):
                return self.call(obj, args, kw)
            if name == '__getattribute__':
                return self.getattr(obj, args[0])
            if name == '__setattr__':
                obj.attrs[args[0]] = args[1]
                return None
            if name == '__setstate__':
                self.build(obj, args[0])
                return None
            if name in ('__getitem__', 'get', 'pop'):
                try:
                    if obj.kind == 'instance' and obj.name == 'collections.defaultdict':
                        items = obj.attrs['_items']
                        if args[0] not in items:
                            factory = obj.attrs['default_factory']
                            if factory is None:
                                raise Stopped('missing defaultdict key')
                            items[args[0]] = self.call(factory)
                        return items[args[0]]
                    return self.getitem(obj, args[0])
                except Stopped:
                    if name == 'get':
                        return args[1] if len(args) > 1 else None
                    raise
            if name == '__setitem__':
                self.setitem(obj, args[0], args[1])
                return None
            if name == 'update' and obj.kind in ('namespace', 'modules'):
                for key, value in (args[0].items() if isinstance(args[0], dict) else self.consume(args[0])):
                    self.setitem(obj, key, value)
                for key, value in kw.items():
                    self.setitem(obj, key, value)
                return None
            if name in ('values', 'items', 'keys') and obj.kind == 'namespace':
                return list(getattr(obj.value.attrs, name)())
            if obj.kind == 'file':
                if obj.value is None and name in ('read', 'read1', 'readall', 'readline', 'seek', 'getvalue', 'getbuffer'):
                    self.file_bytes(obj)
                pos = obj.attrs.get('position', 0)
                if name in ('getvalue', 'getbuffer'):
                    return obj.value
                if name == 'tell':
                    return pos
                if name == 'seek':
                    offset = args[0]
                    whence = args[1] if len(args) > 1 else 0
                    obj.attrs['position'] = offset + (pos if whence == 1 else len(obj.value) if whence == 2 else 0)
                    return obj.attrs['position']
                if name in ('read', 'read1', 'readall', 'readline'):
                    size = args[0] if args else -1
                    end = len(obj.value) if size < 0 else min(len(obj.value), pos + size)
                    if name == 'readline':
                        newline = b'\n' if isinstance(obj.value, (bytes, bytearray)) else '\n'
                        found = obj.value.find(newline, pos, end)
                        if found >= 0:
                            end = found + 1
                    obj.attrs['position'] = end
                    return obj.value[pos:end]
                if name == 'write':
                    value = args[0]
                    if obj.value is None:
                        obj.value = '' if isinstance(value, str) else b''
                    if obj.name == 'tempfile.SpooledTemporaryFile' and obj.attrs.get('max_size') and pos + len(value) > obj.attrs['max_size']:
                        self.note('file-access', 'SpooledTemporaryFile rollover')
                    obj.value = obj.value[:pos] + value + obj.value[pos + len(value):]
                    obj.attrs['position'] = pos + len(value)
                    return len(value)
                if name == 'truncate':
                    end = args[0] if args else pos
                    obj.value = obj.value[:end]
                    return end
                if name in ('close', 'flush'):
                    return None
                if obj.name == 'tempfile.SpooledTemporaryFile' and name in ('fileno', 'rollover'):
                    self.note('file-access', 'SpooledTemporaryFile rollover')
                    return UNKNOWN
            if obj.kind == 'archive':
                if name in ('extract', 'extractall', 'write', 'writestr', 'add'):
                    if name not in ('writestr',):
                        self.note('file-access', obj.name + '.' + name)
                    return None
                if name == 'close':
                    return None
                if obj.name == 'zip':
                    import zipfile
                    with zipfile.ZipFile(io.BytesIO(obj.value)) as zf:
                        if name == 'namelist':
                            return zf.namelist()
                        if name in ('read', 'open'):
                            info = zf.getinfo(args[0])
                            if info.file_size > MAX_BYTES:
                                raise Unsupported('embedded ZIP member size limit exceeded')
                            data = zf.read(info)
                            return Symbol('file', 'zipfile.ZipExtFile', value=data, attrs={'position': 0}) if name == 'open' else data
                else:
                    import tarfile
                    with tarfile.open(fileobj=io.BytesIO(obj.value)) as tf:
                        if name == 'getnames':
                            return tf.getnames()
                        if name == 'extractfile':
                            info = tf.getmember(args[0])
                            if not info.isfile() or info.size > MAX_BYTES:
                                raise Unsupported('unsupported embedded TAR member')
                            return Symbol('file', 'tarfile.ExFileObject', value=tf.extractfile(info).read(), attrs={'position': 0})
                return UNKNOWN
            if obj.kind == 'decompressor' and name == 'decompress':
                return self.decompress(args[0], obj.name, obj.args)
            if obj.kind == 'unpickler' and name == 'load':
                return self.nested('pickle.load', (obj.value,), kw)
            if obj.kind == 'path':
                if name in ('__str__', '__fspath__', 'as_posix'):
                    return obj.value
                if name in ('joinpath', '__truediv__'):
                    return Symbol('path', obj.name, value=self.path_string((obj.value,) + args))
                if name == 'with_name':
                    return Symbol('path', obj.name, value=posixpath.join(posixpath.dirname(obj.value), args[0]))
                if name == 'with_suffix':
                    return Symbol('path', obj.name, value=posixpath.splitext(obj.value)[0] + args[0])
                cap = CALL_CAPABILITIES.get('pathlib.Path.' + name)
                if cap:
                    self.note(cap, 'pathlib.Path.' + name)
                    return UNKNOWN
            if obj.kind == 'iterator':
                if name == '__iter__':
                    return obj
                if name in ('__next__', 'next'):
                    return self.next(obj)
            if obj.kind == 'array' and name in ('tolist', '__iter__'):
                return obj.value if obj.value is not None else []
            if obj.kind == 'array' and name in ('dump', 'tofile'):
                dest = args[0] if args else UNKNOWN
                if not isinstance(dest, Symbol) or dest.kind != 'file':
                    self.note('file-access', 'numpy.ndarray.' + name)
                return None
            cls = class_name or obj.name
            if cls == 'timeit.Timer' and name in ('timeit', 'repeat', 'autorange'):
                repeat = (args[0] if args else kw.get('repeat', 5)) if name == 'repeat' else 1
                number = (args[1] if len(args) > 1 else kw.get('number', 1000000)) if name == 'repeat' else (args[0] if args else kw.get('number', 1000000))
                if repeat > 0:
                    setup = obj.attrs.get('setup')
                    timer = obj.attrs.get('timer')
                    statement = obj.attrs.get('stmt')
                    if isinstance(setup, Symbol):
                        self.call(setup)
                    if isinstance(timer, Symbol):
                        self.call(timer)
                    if number > 0 and isinstance(statement, Symbol):
                        self.call(statement)
                return UNKNOWN
            if cls.startswith('threading.') and name in ('start', 'run'):
                target = obj.attrs.get('target', obj.attrs.get('_target'))
                callargs = obj.attrs.get('args', obj.attrs.get('_args', ()))
                callkw = obj.attrs.get('kwargs', obj.attrs.get('_kwargs', {}))
                if target is not None:
                    self.call(target, tuple(callargs), callkw)
                return None
            if cls == 'multiprocessing.pool.ThreadPool' and name in ('apply', 'apply_async', 'map', 'map_async', 'starmap', 'starmap_async', 'imap', 'imap_unordered'):
                if name.startswith('apply'):
                    return self.call(args[0], tuple(args[1]) if len(args) > 1 else (), args[2] if len(args) > 2 else {})
                iterator = Symbol('iterator', 'itertools.starmap' if name.startswith('starmap') else 'builtins.map', args[:2])
                return self.consume(iterator)
            if cls.startswith('multiprocessing.') and name in ('start', 'apply', 'apply_async', 'map', 'map_async', 'starmap', 'imap', 'imap_unordered'):
                self.note('process', cls + '.' + name)
                return UNKNOWN
            if cls.startswith('concurrent.futures.') and name in ('submit', 'map'):
                if cls.endswith('ProcessPoolExecutor'):
                    self.note('process', cls + '.' + name)
                initializer = obj.attrs.get('initializer')
                if initializer is not None:
                    self.call(initializer, tuple(obj.attrs.get('initargs', ())))
                if name == 'submit':
                    result = self.call(args[0], args[1:], kw)
                    return Symbol('instance', 'concurrent.futures.Future', attrs={'_result': result})
                self.consume(Symbol('iterator', 'builtins.map', args))
                return Symbol('iterator', 'concrete', ([],))
            if cls == 'concurrent.futures.Future' and name == 'result':
                return obj.attrs.get('_result', UNKNOWN)
            if cls == 'xmlrpc.client.ServerProxy':
                if name in ('__close', 'close'):
                    return None
                self.note('network', cls + '.' + name)
                return UNKNOWN
            if cls.startswith('importlib.machinery.') and name in ('load_module', 'exec_module', 'get_code', 'get_source', 'get_data'):
                self.note('code-execution' if name in ('load_module', 'exec_module') else 'file-access', cls + '.' + name)
                return UNKNOWN
            if cls.startswith('logging.') and name in ('emit', 'handle'):
                if any(x in cls for x in ('SocketHandler', 'DatagramHandler', 'HTTPHandler', 'SMTPHandler')):
                    self.note('network', cls + '.' + name)
                elif any(x in cls for x in ('FileHandler', 'RotatingFileHandler', 'WatchedFileHandler')):
                    self.note('file-access', cls + '.' + name)
                return None
            cap = CALL_CAPABILITIES.get(canon_name(cls + '.' + name))
            if cap:
                self.note(cap, cls + '.' + name)
                return UNKNOWN
            if name == '__init__':
                result = self.call(self.ref(cls), (obj,) + args if cls == 'builtins.object' else args, kw)
                if isinstance(result, Symbol):
                    obj.attrs.update(result.attrs)
                    obj.value = result.value
                return None
            if name == '__reduce__' or name == '__reduce_ex__':
                return (self.ref(cls), obj.args)
            return UNKNOWN

        # Bound methods on literal values. None of these can dispatch to an
        # artifact-defined Python method: every value is a primitive or Symbol.
        if name == '__getitem__':
            return self.getitem(obj, args[0])
        if name == '__setitem__':
            self.setitem(obj, args[0], args[1])
            return None
        if name == '__iter__':
            return self.iterator(obj)
        if name in ('__add__', '__radd__', '__mul__', '__rmul__'):
            return self.binary('add' if 'add' in name else 'mul', obj, args[0])
        if isinstance(obj, dict):
            if name == 'get':
                return obj.get(*args)
            if name in ('keys', 'values', 'items'):
                return list(getattr(obj, name)())
            if name == 'update':
                other = args[0] if args else {}
                obj.update(other if isinstance(other, dict) else self.consume(other))
                obj.update(kw)
                return None
            if name in ('pop', 'setdefault', 'clear', 'copy'):
                return getattr(obj, name)(*args)
        if isinstance(obj, list):
            if name in ('append', 'insert', 'pop', 'remove', 'clear', 'copy', 'reverse', 'count', 'index'):
                return getattr(obj, name)(*args)
            if name == 'extend':
                obj.extend(self.consume(args[0]))
                return None
            if name == 'sort':
                obj[:] = self.call(self.ref('builtins.sorted'), (obj,), kw)
                return None
        if isinstance(obj, (set, frozenset)):
            if name in ('add', 'discard', 'remove', 'pop', 'clear', 'copy'):
                return getattr(obj, name)(*args)
            if name == 'update':
                for arg in args:
                    obj.update(self.consume(arg))
                return None
        if isinstance(obj, (str, bytes, bytearray)):
            if name in ('encode', 'decode'):
                return self.codec(name == 'encode', obj, *args)
            if name == 'join':
                return obj.join(self.consume(args[0]))
            if name in ('format', 'format_map'):
                # Python formatting can resolve attributes; interpret the
                # fields rather than executing native formatting on Symbols.
                if any(isinstance(x, Symbol) for x in args) or any(isinstance(x, Symbol) for x in kw.values()):
                    return UNKNOWN
            if name in ('lower', 'upper', 'strip', 'lstrip', 'rstrip', 'replace', 'split', 'rsplit', 'splitlines',
                        'partition', 'rpartition', 'removeprefix', 'removesuffix', 'capitalize', 'title', 'swapcase',
                        'startswith', 'endswith', 'find', 'rfind', 'index', 'count', 'hex', 'format', 'format_map',
                        'ljust', 'rjust', 'center', 'zfill', 'translate', 'expandtabs'):
                if any(isinstance(x, Symbol) for x in args):
                    return UNKNOWN
                return getattr(obj, name)(*args, **kw)
        return UNKNOWN

    def truth(self, obj):
        if isinstance(obj, Symbol):
            if obj.kind == 'instance':
                cls = self.class_of(obj)
                hook = cls.attrs.get('__bool__', cls.attrs.get('__len__'))
                if hook is not None:
                    return self.truth(self.call(self.bind(hook, obj)))
            return True
        return bool(obj)

    def binary(self, op, a, b):
        if op in ('is', 'is_not'):
            return (a is b) if op == 'is' else (a is not b)
        if isinstance(a, Symbol) or isinstance(b, Symbol):
            if op == 'truediv' and isinstance(a, Symbol) and a.kind == 'path':
                return self.method(a, '__truediv__', (b,), {})
            return UNKNOWN
        if op in ('mul', 'imul') and isinstance(b, int) and isinstance(a, (str, bytes, bytearray, list, tuple)):
            if len(a) * b > MAX_BYTES:
                raise Unsupported('sequence allocation limit exceeded')
        if op == 'pow' and isinstance(b, int) and b > 10000:
            raise Unsupported('integer exponent limit exceeded')
        funcs = {'add': operator.add, 'concat': operator.concat, 'iadd': operator.iadd, 'iconcat': operator.iconcat,
                 'mul': operator.mul, 'imul': operator.imul, 'sub': operator.sub, 'truediv': operator.truediv,
                 'floordiv': operator.floordiv, 'mod': operator.mod, 'pow': operator.pow, 'or': operator.or_,
                 'and': operator.and_, 'xor': operator.xor, 'eq': operator.eq, 'ne': operator.ne,
                 'lt': operator.lt, 'le': operator.le, 'gt': operator.gt, 'ge': operator.ge}
        return funcs[op](a, b)

    def codec(self, encode, value, encoding='utf-8', errors='strict'):
        # Avoid the codec registry, whose search hooks are mutable at runtime.
        # These built-in codecs operate solely on byte/text literals.
        key = encoding.lower().replace('_', '-').replace(' ', '')
        safe = {'utf8', 'utf-8', 'utf-8-sig', 'ascii', 'us-ascii', 'latin1', 'latin-1', 'latin', 'iso-8859-1',
                'utf-16', 'utf-16-le', 'utf-16-be', 'utf-32', 'utf-32-le', 'utf-32-be',
                'unicode-escape', 'raw-unicode-escape', 'cp1252', 'rot-13', 'rot13',
                'base64', 'base64-codec', 'hex', 'hex-codec', 'zlib', 'zlib-codec', 'bz2', 'bz2-codec'}
        if key not in safe or errors not in ('strict', 'ignore', 'replace', 'surrogatepass', 'surrogateescape', 'backslashreplace'):
            raise Unsupported('unsupported literal text codec')
        if key in ('zlib', 'zlib-codec', 'bz2', 'bz2-codec') and not encode:
            return self.decompress(value, key.split('-')[0])
        fn = codecs.encode if encode else codecs.decode
        return fn(value, encoding, errors)

    def decompress(self, data, kind, args=()):
        if not isinstance(data, (bytes, bytearray)):
            raise Stopped('compressed data must be bytes')
        data = bytes(data)
        out = []
        size = 0
        while data:
            self.tick()
            if kind in ('gzip', 'zlib'):
                wbits = 31 if kind == 'gzip' else (args[0] if args else 15)
                dec = zlib.decompressobj(wbits)
                part = dec.decompress(data, MAX_BYTES + 1 - size)
            elif kind == 'bz2':
                dec = bz2.BZ2Decompressor()
                part = dec.decompress(data, max_length=MAX_BYTES + 1 - size)
            else:
                dec = lzma.LZMADecompressor(memlimit=MAX_BYTES)
                part = dec.decompress(data, max_length=MAX_BYTES + 1 - size)
            out.append(part)
            size += len(part)
            if size > MAX_BYTES or not dec.eof:
                raise Unsupported('decompression is truncated or exceeds size limit')
            data = dec.unused_data
            if kind == 'zlib':
                break
            # RFC 1952 permits zero padding after gzip members.
            if kind == 'gzip':
                data = data.lstrip(b'\0')
        self.decoded_bytes += size
        if self.decoded_bytes > 2 * MAX_BYTES:
            raise Unsupported('total decompression limit exceeded')
        return b''.join(out)

    def file_bytes(self, obj):
        if isinstance(obj, Symbol) and obj.kind == 'file':
            if obj.value is None and 'wrapped' in obj.meta:
                data = self.file_bytes(obj.meta['wrapped'])
                if 'compression' in obj.meta:
                    data = self.decompress(data, obj.meta['compression'])
                if 'encoding' in obj.meta:
                    data = self.codec(False, data, obj.meta['encoding'])
                obj.value = data
            data = obj.value
            if isinstance(data, (bytes, bytearray, str)):
                return data[obj.attrs.get('position', 0):]
        raise Unsupported('unknown in-memory file content')

    def nested(self, name, args, kw):
        source = args[0] if args else kw.get('file', kw.get('f', kw.get('filepath_or_buffer', UNKNOWN)))
        if name in ('pickle.loads', 'dill.loads', 'cloudpickle.loads', 'numpy.ndarray.loads', 'torch.storage._load_from_bytes'):
            data = source
            if not isinstance(data, (bytes, bytearray)):
                if isinstance(data, str):
                    raise Stopped('loads requires a bytes-like object')
                raise Unsupported('unknown nested pickle bytes')
        elif isinstance(source, Symbol) and source.kind == 'file':
            data = self.file_bytes(source)
        elif isinstance(source, (str, bytes)) or isinstance(source, Symbol) and source.kind == 'path':
            data = self.artifact_content(source)
            if data is END:
                self.note('file-access', name + ' reads a path')
                return UNKNOWN
        else:
            raise Unsupported('unknown nested loader input')
        from modelgate.pickle_scan import run_pickle
        oldloc, oldop = self.location, self.opcode
        try:
            loc = oldloc + '!' + name
            if name.startswith(('pickle.', 'dill.', 'cloudpickle.')) or name == 'numpy.ndarray.loads':
                value, used = run_pickle(bytes(data), loc, self, mode='one')
                if isinstance(source, Symbol) and source.kind == 'file':
                    source.attrs['position'] = source.attrs.get('position', 0) + used
                return value
            from modelgate.dispatch import analyze_data, _npy
            if name.startswith('numpy.'):
                allow = kw.get('allow_pickle', args[1] if name == 'numpy.lib.format.read_array' and len(args) > 1 else args[2] if len(args) > 2 else False)
                if data.startswith(b'\x93NUMPY'):
                    return _npy(bytes(data), loc, self, allow_pickle=bool(allow))
                if data.startswith((b'PK\x03\x04', b'PK\x05\x06')) and name == 'numpy.load':
                    return Symbol('npz', 'numpy.lib.npyio.NpzFile', attrs={'allow_pickle': allow, 'location': loc}, value=bytes(data))
                if not allow or name == 'numpy.lib.format.read_array':
                    raise Stopped('NumPy loader cannot unpickle this input')
                return run_pickle(bytes(data), loc, self, mode='one')[0]
            mode = 'torch' if name.startswith('torch.') else 'pandas' if name.startswith('pandas.') else 'joblib'
            return analyze_data(bytes(data), loc, self, loader=mode)
        finally:
            self.location, self.opcode = oldloc, oldop

    def iterator(self, obj):
        if isinstance(obj, Symbol) and obj.kind == 'iterator':
            return obj
        if isinstance(obj, Symbol) and obj.kind == 'array':
            obj = obj.value or []
        if isinstance(obj, Symbol) and obj.kind == 'namespace':
            obj = list(obj.value.attrs)
        if isinstance(obj, Symbol):
            if obj.kind == 'instance':
                hook = self.class_of(obj).attrs.get('__iter__')
                if hook is not None:
                    return self.call(self.bind(hook, obj))
            return Symbol('iterator', 'unknown')
        if isinstance(obj, dict):
            obj = list(obj)
        if isinstance(obj, (list, tuple, set, frozenset, str, bytes, bytearray, range)):
            if isinstance(obj, (set, frozenset)):
                obj = list(obj)
            return Symbol('iterator', 'concrete', (obj,), {'index': 0})
        raise Stopped('object is not iterable')

    def next(self, it):
        self.tick()
        if not isinstance(it, Symbol) or it.kind != 'iterator':
            raise Stopped('not an iterator')
        n, args = it.name, it.args
        if n == 'concrete':
            idx = it.attrs.get('index', 0)
            if idx >= len(args[0]):
                return END
            it.attrs['index'] = idx + 1
            return args[0][idx]
        if n == 'unknown':
            raise Unsupported('cannot determine consumed iterator values')
        if n == 'range':
            if 'inner' not in it.attrs:
                it.attrs['inner'] = self.iterator(range(*args))
            return self.next(it.attrs['inner'])
        if n == 'calliter':
            result = self.call(args[0])
            return END if result == args[1] else result
        if n in ('itertools.repeat', 'itertools.count'):
            idx = it.attrs.get('index', 0)
            it.attrs['index'] = idx + 1
            if n == 'itertools.repeat':
                return END if len(args) > 1 and idx >= args[1] else args[0]
            return (args[0] if args else 0) + idx * (args[1] if len(args) > 1 else 1)
        if n in ('builtins.map', 'builtins.zip', 'itertools.zip_longest'):
            start = 1 if n == 'builtins.map' else 0
            if 'sources' not in it.attrs:
                it.attrs['sources'] = [self.iterator(arg) for arg in args[start:]]
            vals = []
            for source in it.attrs['sources']:
                value = self.next(source)
                if value is END and n != 'itertools.zip_longest':
                    return END
                vals.append(value)
            if not vals or (all(x is END for x in vals) if n == 'itertools.zip_longest' else any(x is END for x in vals)):
                return END
            if n == 'itertools.zip_longest':
                vals = [it.attrs.get('fillvalue') if x is END else x for x in vals]
            return self.call(args[0], tuple(vals)) if n == 'builtins.map' else tuple(vals)
        if n in ('builtins.filter', 'itertools.filterfalse', 'itertools.takewhile', 'itertools.dropwhile', 'itertools.starmap'):
            if 'source' not in it.attrs:
                it.attrs['source'] = self.iterator(args[1])
            if it.attrs.get('done'):
                return END
            while True:
                value = self.next(it.attrs['source'])
                if value is END:
                    return END
                if n == 'itertools.starmap':
                    return self.call(args[0], tuple(self.consume(value)))
                if n == 'itertools.dropwhile' and it.attrs.get('dropped'):
                    return value
                check = self.truth(self.call(args[0], (value,))) if args[0] is not None else self.truth(value)
                if n == 'itertools.takewhile':
                    if not check:
                        it.attrs['done'] = True
                        return END
                    return value
                if n == 'itertools.dropwhile':
                    if not check:
                        it.attrs['dropped'] = True
                        return value
                elif check != (n == 'itertools.filterfalse'):
                    return value
        if n in ('itertools.chain', 'itertools.chain.from_iterable'):
            if 'outer' not in it.attrs:
                it.attrs['outer'] = self.iterator(args[0] if n.endswith('from_iterable') else args)
            while True:
                if 'current' not in it.attrs:
                    value = self.next(it.attrs['outer'])
                    if value is END:
                        return END
                    it.attrs['current'] = self.iterator(value)
                value = self.next(it.attrs['current'])
                if value is not END:
                    return value
                del it.attrs['current']
        if n == 'itertools.islice':
            if 'source' not in it.attrs:
                it.attrs['source'] = self.iterator(args[0])
                it.attrs['index'] = 0
                it.attrs['next'] = args[1] or 0 if len(args) >= 3 else 0
            stop = args[2] if len(args) >= 3 else args[1]
            step = args[3] if len(args) > 3 else 1
            target = it.attrs['next']
            if stop is not None and target >= stop:
                return END
            while it.attrs['index'] <= target:
                value = self.next(it.attrs['source'])
                it.attrs['index'] += 1
                if value is END:
                    return END
            it.attrs['next'] += step
            return value
        if n in ('builtins.enumerate', 'builtins.reversed', 'itertools.cycle', 'itertools.accumulate', 'itertools.compress'):
            if 'source' not in it.attrs:
                source = args[0]
                if n == 'builtins.reversed':
                    source = list(reversed(self.consume(source)))
                it.attrs['source'] = self.iterator(source)
                it.attrs['index'] = args[1] if len(args) > 1 and n == 'builtins.enumerate' else 0
                it.attrs['cache'] = []
            value = self.next(it.attrs['source'])
            if n == 'itertools.cycle':
                if value is END:
                    if not it.attrs['cache']:
                        return END
                    it.attrs['source'] = self.iterator(it.attrs['cache'])
                    it.attrs['cycled'] = True
                    value = self.next(it.attrs['source'])
                if not it.attrs.get('cycled'):
                    it.attrs['cache'].append(value)
                return value
            if value is END:
                return END
            if n == 'builtins.enumerate':
                idx = it.attrs['index']
                it.attrs['index'] += 1
                return (idx, value)
            if n == 'itertools.accumulate':
                if 'acc' not in it.attrs:
                    it.attrs['acc'] = value
                else:
                    func = args[1] if len(args) > 1 else None
                    it.attrs['acc'] = self.call(func, (it.attrs['acc'], value)) if func else self.binary('add', it.attrs['acc'], value)
                return it.attrs['acc']
            if n == 'itertools.compress':
                if 'selectors' not in it.attrs:
                    it.attrs['selectors'] = self.iterator(args[1])
                selector = self.next(it.attrs['selectors'])
                if selector is END:
                    return END
                return value if self.truth(selector) else self.next(it)
            return value
        raise Unsupported('unsupported consumed iterator ' + n)

    def consume(self, value):
        it = self.iterator(value)
        out = []
        for _ in range(MAX_ITEMS):
            item = self.next(it)
            if item is END:
                return out
            out.append(item)
        raise Unsupported('iterator work limit exceeded')
