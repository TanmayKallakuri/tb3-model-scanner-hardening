"""Side-effect-free abstract Python values used by the pickle machine.

Artifact globals are names, never imported Python objects. Only explicitly
implemented operations on our own data values are evaluated. All callable,
module, iterator and instance state stays in this private symbolic world.
"""
from __future__ import annotations

import _compat_pickle
import base64
import binascii
import bz2
import io
import itertools
import lzma
import math
import operator
import posixpath
import re
import zlib
import zipfile
from dataclasses import dataclass, field

from modelgate.limits import AnalysisLimit, Budget
from modelgate.report import Finding


class LoadStopped(Exception):
    """The real loader raises here, before any further opcode can execute."""


class Detected(Exception):
    def __init__(self, finding):
        self.finding = finding


@dataclass(eq=False)
class Ref:
    name: str


@dataclass(eq=False)
class Obj:
    type: str
    args: tuple = ()
    attrs: dict = field(default_factory=dict)
    value: object = None
    extra: dict = field(default_factory=dict)


@dataclass(eq=False)
class Method:
    obj: object
    name: str


@dataclass(eq=False)
class Lazy:
    kind: str
    args: tuple
    index: int = 0
    state: object = None
    ended: bool = False


UNKNOWN = Obj("unknown")
MISSING = object()

ALIASES = {
    "posix": "os", "nt": "os", "_pickle": "pickle", "_io": "io",
    "_operator": "operator", "_functools": "functools", "_socket": "socket",
}
PY2_MODULES = (set(_compat_pickle.IMPORT_MAPPING) - {"dbm", "_elementtree"}) | {"exceptions"}

# These are aliases in standard library modules, not guesses from a substring.
MODULE_ATTRS = {
    "os.path": "posixpath", "shutil.os": "os", "subprocess.os": "os",
    "shutil.sys": "sys", "subprocess.sys": "sys", "pathlib.os": "os",
    "pathlib.io": "io", "pathlib.sys": "sys", "tempfile._os": "os",
    "tempfile._io": "io", "tempfile._shutil": "shutil", "tempfile._sys": "sys",
    "pickle.io": "io", "pickle.sys": "sys", "pickle.codecs": "codecs",
    "pickle._compat_pickle": "_compat_pickle", "copyreg.sys": "sys",
    "copy.types": "types", "copy.copyreg": "copyreg", "copy.weakref": "weakref",
    "copy.sys": "sys", "functools.types": "types", "functools.partial": None,
    "threading._sys": "sys", "threading._os": "os", "threading._thread": "_thread",
    "socket.os": "os", "socket.sys": "sys", "socket.io": "io",
    "socket._socket": "socket", "ssl.socket": None, "ssl.os": "os",
    "urllib.request.os": "os", "urllib.request.sys": "sys",
    "urllib.request.socket": "socket", "urllib.request.http": "http",
    "urllib.request.io": "io", "urllib.request.ftpwrapper": None,
    "http.client.socket": "socket", "http.client.ssl": "ssl", "http.client.io": "io",
    "ftplib.socket": "socket", "ftplib.os": "os", "ftplib.sys": "sys",
    "smtplib.socket": "socket", "smtplib.io": "io", "smtplib.sys": "sys",
    "pdb.os": "os", "pdb.sys": "sys", "pdb.bdb": "bdb", "pdb.code": "code",
    "bdb.os": "os", "bdb.sys": "sys", "runpy.sys": "sys", "runpy.os": "os",
    "timeit.sys": "sys", "timeit.itertools": "itertools",
    "pydoc.os": "os", "pydoc.sys": "sys", "pydoc.subprocess": "subprocess",
    "pkgutil.sys": "sys", "pkgutil.importlib": "importlib", "pkgutil.os": "os",
    "importlib.sys": "sys", "inspect.os": "os", "inspect.sys": "sys",
    "logging.os": "os", "logging.sys": "sys", "logging.config.os": "os",
    "logging.config.logging": "logging", "logging.handlers.socket": "socket",
    "multiprocessing.process.os": "os", "multiprocessing.util.os": "os",
    "numpy.ctypeslib.ctypes": "ctypes", "numpy.lib.npyio.pickle": "pickle",
    "numpy.lib._npyio_impl.pickle": "pickle", "numpy.lib.format.pickle": "pickle",
    "joblib.numpy_pickle.pickle": "pickle", "torch.serialization.pickle": "pickle",
    "pandas.io.pickle.pickle": "pickle", "xmlrpc.client.http": "http",
}
MODULE_NAMES = set(ALIASES.values()) | {
    "sys", "types", "importlib", "importlib.util", "pkgutil", "codecs", "_codecs",
    "operator", "functools", "collections", "collections.abc", "copy", "copyreg",
    "os", "posixpath", "ntpath", "pathlib", "shutil", "tempfile", "subprocess", "pty",
    "socket", "ssl", "urllib", "urllib.request", "urllib.parse", "http", "http.client",
    "ftplib", "smtplib", "webbrowser", "xmlrpc", "xmlrpc.client", "logging", "logging.config",
    "logging.handlers", "code", "bdb", "pdb", "timeit", "runpy", "pydoc", "marshal",
    "base64", "binascii", "zlib", "gzip", "bz2", "lzma", "itertools", "threading", "_thread",
    "concurrent", "concurrent.futures", "multiprocessing", "multiprocessing.pool",
    "pickle", "io", "builtins", "numpy", "numpy.core", "numpy._core", "numpy.lib",
    "numpy.lib.format", "numpy.core.multiarray", "numpy._core.multiarray", "torch",
    "torch.storage", "torch.serialization", "joblib", "pandas", "ctypes", "weakref",
}

# Calling these functions performs the capability, rather than merely referring
# to it. Constructor and argument-dependent cases are handled below this table.
CAPABILITIES = {}
def _caps(kind, module, names):
    CAPABILITIES.update({module + "." + n: kind for n in names.split()})

_caps("code-execution", "builtins", "eval exec compile")
_caps("code-execution", "runpy", "run_path run_module _run_code _run_module_code _run_module_as_main")
_caps("code-execution", "pdb", "run runeval runcall")
_caps("code-execution", "numpy.testing._private.utils", "runstring")
_caps("code-execution", "numpy.f2py", "compile")
_caps("process", "os", "system popen execv execve execvp execvpe execl execle execlp execlpe spawnl spawnle spawnlp spawnlpe spawnv spawnve spawnvp spawnvpe posix_spawn posix_spawnp fork forkpty startfile")
_caps("process", "subprocess", "Popen call run check_call check_output getoutput getstatusoutput")
_caps("process", "pty", "spawn fork")
_caps("process", "pydoc", "pipepager tempfilepager")
_caps("process", "_posixsubprocess", "fork_exec")
_caps("process", "multiprocessing", "Pool")
_caps("process", "multiprocessing.pool", "Pool")
_caps("file-access", "builtins", "open")
_caps("file-access", "io", "open open_code FileIO")
_caps("file-access", "os", "open read write pread pwrite readv writev preadv pwritev sendfile truncate ftruncate remove unlink rename replace mkdir makedirs rmdir removedirs renames link symlink mknod mkfifo chmod fchmod lchmod chown fchown lchown utime listdir scandir stat lstat statvfs access readlink chdir chroot getxattr setxattr listxattr removexattr")
_caps("file-access", "shutil", "copy copy2 copyfile copyfileobj copymode copystat copytree move rmtree chown disk_usage")
_caps("file-access", "tempfile", "TemporaryFile NamedTemporaryFile TemporaryDirectory mkstemp mkdtemp mktemp")
_caps("file-access", "glob", "glob")
_caps("file-access", "linecache", "getline getlines updatecache")
_caps("file-access", "fileinput", "input FileInput")
_caps("file-access", "numpy", "fromfile loadtxt genfromtxt savetxt tofile memmap")
_caps("file-access", "numpy.lib.npyio", "loadtxt genfromtxt savetxt")
_caps("file-access", "numpy.lib._npyio_impl", "loadtxt genfromtxt savetxt")
_caps("file-access", "numpy.core.multiarray", "fromfile")
_caps("file-access", "numpy._core.multiarray", "fromfile")
_caps("file-access", "sqlite3", "connect")
_caps("file-access", "shelve", "open")
_caps("file-access", "dbm", "open")
_caps("file-access", "dbm.gnu", "open")
_caps("file-access", "logging", "FileHandler")
_caps("file-access", "logging.handlers", "RotatingFileHandler TimedRotatingFileHandler WatchedFileHandler")
_caps("file-access", "logging.config", "fileConfig")
_caps("network", "socket", "create_connection create_server getaddrinfo gethostbyname gethostbyname_ex gethostbyaddr")
_caps("network", "urllib.request", "urlopen urlretrieve")
_caps("network", "webbrowser", "open open_new open_new_tab")
_caps("network", "ssl", "get_server_certificate")
_caps("network", "requests", "request get post put delete patch head options")
_caps("network", "httpx", "request get post put delete patch head options stream")

CLASS_NAMES = set("""builtins.object builtins.type builtins.str builtins.bytes builtins.bytearray
builtins.int builtins.float builtins.complex builtins.bool builtins.list builtins.tuple builtins.dict
builtins.set builtins.frozenset builtins.slice builtins.range builtins.map builtins.filter builtins.zip
builtins.enumerate builtins.reversed builtins.memoryview builtins.property builtins.staticmethod
builtins.classmethod functools.partial operator.attrgetter operator.itemgetter operator.methodcaller
collections.OrderedDict collections.defaultdict collections.deque collections.Counter
collections.ChainMap types.SimpleNamespace types.ModuleType types.FunctionType types.CodeType
io.BytesIO io.StringIO io.FileIO io.TextIOWrapper io.BufferedReader io.BufferedWriter
socket.socket ssl.SSLSocket subprocess.Popen threading.Thread threading.Timer
concurrent.futures.ThreadPoolExecutor concurrent.futures.ProcessPoolExecutor multiprocessing.Process
multiprocessing.context.Process multiprocessing.pool.ThreadPool bdb.Bdb pdb.Pdb code.InteractiveInterpreter
code.InteractiveConsole timeit.Timer pathlib.Path pathlib.PosixPath pathlib.WindowsPath pathlib.PurePath
pathlib.PurePosixPath pathlib.PureWindowsPath http.client.HTTPConnection http.client.HTTPSConnection
ftplib.FTP ftplib.FTP_TLS smtplib.SMTP smtplib.SMTP_SSL smtplib.LMTP xmlrpc.client.ServerProxy
numpy.ndarray numpy.dtype numpy.memmap numpy.matrix numpy.ma.core.MaskedArray
datetime.datetime datetime.date datetime.time datetime.timedelta datetime.timezone
functools._lru_cache_wrapper types.MethodType types.MappingProxyType re.Pattern
torch.device torch.dtype numpy.str_ numpy.bytes_ numpy.bool_ numpy.int8 numpy.int16 numpy.int32
numpy.int64 numpy.uint8 numpy.uint16 numpy.uint32 numpy.uint64 numpy.float16 numpy.float32 numpy.float64
numpy.complex64 numpy.complex128 numpy.longdouble numpy.clongdouble numpy.void
""".split())


def canon(module):
    return ALIASES.get(module, module)


def concrete(value, seen=None):
    """Only data made by the parser may be passed to safe standard primitives."""
    if value is None or type(value) in (str, bytes, bytearray, bool, int, float, complex, slice):
        return True
    if type(value) not in (tuple, list, dict, set, frozenset, range):
        return False
    seen = set() if seen is None else seen
    if id(value) in seen:
        return False
    seen.add(id(value))
    items = value.items() if isinstance(value, dict) else value
    return all(concrete(v, seen.copy()) for v in items)


class World:
    def __init__(self, budget=None):
        self.budget = budget or Budget()
        self.modules = {}
        self.location = "pickle"
        self.call_depth = 0
        self.nested_depth = 0

    def flag(self, kind, detail):
        raise Detected(Finding(kind, self.location, detail))

    def unsupported(self, detail):
        self.flag("unsupported", detail)

    def module(self, name):
        name = canon(name)
        if name not in self.modules:
            if name in PY2_MODULES:
                raise LoadStopped("Python 2 module name without fix_imports mapping")
            self.modules[name] = Obj("module", (name,))
        return self.modules[name]

    def global_ref(self, module, name, protocol=4):
        if protocol < 3:
            module, name = _compat_pickle.NAME_MAPPING.get((module, name), (module, name))
            module = _compat_pickle.IMPORT_MAPPING.get(module, module)
        if protocol < 4 and "." in name:
            raise LoadStopped("dotted globals require protocol 4")
        result = self.module(module)
        for part in name.split("."):
            result = self.attr(result, part)
        return result

    def is_class(self, obj):
        return isinstance(obj, Ref) and (obj.name in CLASS_NAMES or obj.name.rsplit(".", 1)[-1].lstrip("_")[:1].isupper())

    def attr(self, obj, name, default=MISSING):
        if not isinstance(name, str):
            raise LoadStopped("attribute name must be str")
        if isinstance(obj, Obj):
            if obj.type == "functools.partial":
                if name == "func": return obj.value
                if name == "args": return obj.args
                if name == "keywords": return obj.extra.setdefault("keywords", {})
            if name in obj.attrs:
                return obj.attrs[name]
            if name == "__dict__":
                if obj.type == "module":
                    # Populate ordinary module members lazily through a special
                    # mapping view, preserving writes and alias identity.
                    return Obj("module-dict", value=obj)
                return obj.attrs
            if name == "__class__":
                return Ref("types.ModuleType" if obj.type == "module" else obj.type)
            if obj.type == "module":
                full = obj.args[0] + "." + name
                if obj.args[0] == "builtins":
                    import builtins
                    if name not in vars(builtins): raise LoadStopped("missing builtin")
                if obj.args[0] == "itertools" and name in ("imap", "ifilter", "izip", "izip_longest", "ifilterfalse"):
                    raise LoadStopped("Python 2 itertools name without fix_imports mapping")
                if full == "sys.modules":
                    return Obj("sys.modules")
                if full == "builtins.__dict__":
                    return Obj("module-dict", value=obj)
                if full == "builtins.None": return None
                if full == "builtins.True": return True
                if full == "builtins.False": return False
                if full in MODULE_ATTRS and MODULE_ATTRS[full] is not None:
                    return self.module(MODULE_ATTRS[full])
                if full in MODULE_NAMES:
                    return self.module(full)
                aliases = {"io.open": "builtins.open", "ssl.socket": "socket.socket",
                           "types.LambdaType": "types.FunctionType", "types.BuiltinFunctionType": "builtins.type"}
                return Ref(aliases.get(full, full))
            if obj.type == "wrapper" and name == "__wrapped__":
                return obj.value
            if obj.type == "path" and name in ("name", "parent", "suffix", "stem"):
                path = obj.value
                if name == "parent": return Obj("path", value=posixpath.dirname(path))
                if name == "name": return posixpath.basename(path)
                if name == "suffix": return posixpath.splitext(path)[1]
                return posixpath.splitext(posixpath.basename(path))[0]
            if default is not MISSING and obj.type in ("types.SimpleNamespace", "builtins.object"):
                return default
            return Method(obj, name)
        if isinstance(obj, Ref):
            if name == "__call__": return obj
            if name == "__name__": return obj.name.rsplit(".", 1)[-1]
            if name == "__module__": return obj.name.rsplit(".", 1)[0]
            if name == "__class__": return Ref("builtins.type" if self.is_class(obj) else "types.FunctionType")
            if name in ("__globals__", "func_globals"):
                module = self.module(obj.name.rsplit(".", 1)[0])
                module.attrs.setdefault("__builtins__", Obj("module-dict", value=self.module("builtins")))
                return Obj("module-dict", value=module)
            return Ref(obj.name + "." + name)
        if isinstance(obj, Method) and name == "__call__": return obj
        if isinstance(obj, Lazy) and name in ("__next__", "__iter__", "__setstate__"):
            return Method(obj, name)
        if name == "__class__":
            return Ref("builtins." + type(obj).__name__)
        if type(obj) in (str, bytes, bytearray, tuple, list, dict, set, frozenset, int, float, complex, range):
            if not hasattr(type(obj), name):
                if default is not MISSING: return default
                raise LoadStopped("missing attribute")
            return Method(obj, name)
        if default is not MISSING: return default
        return UNKNOWN

    def getitem(self, obj, key):
        if isinstance(obj, Obj):
            if obj.type == "sys.modules": return self.module(key)
            if obj.type == "module-dict": return self.attr(obj.value, key)
            if obj.type in ("collections.OrderedDict", "collections.defaultdict", "collections.Counter"):
                if key in obj.value: return obj.value[key]
                if obj.type == "collections.defaultdict" and obj.args and obj.args[0] is not None:
                    obj.value[key] = self.call(obj.args[0], ())
                    return obj.value[key]
            return self.call(self.attr(obj, "__getitem__"), (key,))
        try:
            return obj[key]
        except (IndexError, KeyError, TypeError):
            raise LoadStopped("invalid item access")

    def setitem(self, obj, key, value):
        if isinstance(obj, Obj):
            if obj.type == "module-dict": obj.value.attrs[key] = value
            elif obj.type == "sys.modules": self.modules[canon(key)] = value
            elif isinstance(obj.value, dict): obj.value[key] = value
            else: self.call(self.attr(obj, "__setitem__"), (key, value))
        else:
            try: obj[key] = value
            except (TypeError, KeyError, IndexError): raise LoadStopped("invalid item assignment")

    def build(self, obj, state):
        if isinstance(obj, Obj) and obj.attrs.get("__setstate__") is not None:
            self.call(obj.attrs["__setstate__"], (state,))
            return
        if isinstance(obj, Obj) and obj.type == "functools.partial":
            if not isinstance(state, tuple) or len(state) != 4 or not isinstance(state[1], tuple):
                raise LoadStopped("invalid partial state")
            obj.value, obj.args, keywords, namespace = state
            if (keywords is not None and not isinstance(keywords, dict)) or (namespace is not None and not isinstance(namespace, dict)):
                raise LoadStopped("invalid partial state dictionaries")
            obj.attrs = namespace if namespace is not None else {}
            obj.extra["keywords"] = keywords if keywords is not None else {}
            return
        if isinstance(obj, Obj) and obj.type == "numpy.dtype":
            obj.attrs["state"] = state
            return
        if isinstance(obj, Obj) and obj.type in ("io.BytesIO", "io.StringIO"):
            if not isinstance(state, tuple) or len(state) < 2:
                raise LoadStopped("invalid in-memory stream state")
            obj.value = io.BytesIO(state[0]) if obj.type == "io.BytesIO" else io.StringIO(state[0])
            # BytesIO: (data, position, dict); StringIO: (data, newline,
            # position, dict).
            pos = state[1] if obj.type == "io.BytesIO" else state[2]
            obj.value.seek(pos)
            if isinstance(state[-1], dict): obj.attrs.update(state[-1])
            return
        if isinstance(obj, Obj) and obj.type in ("numpy.ndarray", "numpy.matrix"):
            obj.value = state
            return
        if isinstance(obj, Lazy):
            if isinstance(state, int): obj.index = state
            return
        # Instance attributes can override __setstate__; the unpickler's lookup
        # is an ordinary attribute lookup, not special-method lookup on type.
        if isinstance(obj, Obj) and obj.type.startswith("logging.config."):
            self.unsupported("unmodelled configuration object state")
        slotstate = None
        if isinstance(state, tuple) and len(state) == 2:
            state, slotstate = state
        if isinstance(obj, Obj):
            if isinstance(state, dict): obj.attrs.update(state)
            elif state is not None:
                obj.attrs["state"] = state  # standard numerical reducers use tuples
            if isinstance(slotstate, dict): obj.attrs.update(slotstate)
        elif isinstance(obj, (Ref, Method)):
            raise LoadStopped("object cannot accept state")
        elif state or slotstate:
            raise LoadStopped("primitive value has no instance dictionary")

    def call(self, fn, args=(), kwargs=None, *, new=False):
        self.budget.tick()
        self.call_depth += 1
        if self.call_depth > 100:
            raise AnalysisLimit("symbolic call depth exceeded")
        try:
            if not isinstance(args, tuple) or (kwargs is not None and not isinstance(kwargs, dict)):
                raise LoadStopped("call requires tuple arguments and dict keywords")
            return self._call(fn, args, kwargs or {}, new=new)
        except (TypeError, ValueError, IndexError, KeyError, AttributeError, OverflowError, ZeroDivisionError, UnicodeError, binascii.Error):
            raise LoadStopped("invalid arguments to callable") from None
        finally:
            self.call_depth -= 1

    def _call(self, fn, a, kw, *, new=False):
        if isinstance(fn, Method):
            return self.method(fn.obj, fn.name, a, kw)
        if isinstance(fn, Obj):
            if fn.type == "functools.partial":
                return self.call(fn.value, fn.args + a, {**fn.extra.get("keywords", {}), **kw})
            if fn.type == "wrapper": return self.call(fn.value, a, kw)
            if fn.type == "operator.attrgetter":
                results = []
                for path in fn.args:
                    value = a[0]
                    for part in path.split("."): value = self.attr(value, part)
                    results.append(value)
                return results[0] if len(results) == 1 else tuple(results)
            if fn.type == "operator.itemgetter":
                results = [self.getitem(a[0], k) for k in fn.args]
                return results[0] if len(results) == 1 else tuple(results)
            if fn.type == "operator.methodcaller":
                return self.call(self.attr(a[0], fn.args[0]), fn.args[1:], fn.attrs)
            if fn.type == "lru-decorator": return Obj("wrapper", value=a[0])
            if fn.type == "wraps-decorator": return Obj("wrapper", value=a[0], attrs={"__wrapped__": fn.value})
            if fn.type == "types.FunctionType": self.flag("code-execution", "call serialized Python function")
            if "__call__" in fn.attrs: return self.call(fn.attrs["__call__"], a, kw)
            self.unsupported(f"call of unmodelled {fn.type} instance")
        if not isinstance(fn, Ref):
            raise LoadStopped("object is not callable")
        n = fn.name
        if new:
            if not self.is_class(fn): raise LoadStopped("NEWOBJ requires a class")
            if n in ("builtins.list", "builtins.dict", "builtins.set", "builtins.bytearray"):
                return {"builtins.list": list, "builtins.dict": dict, "builtins.set": set, "builtins.bytearray": bytearray}[n]()
            if n in ("collections.OrderedDict", "collections.defaultdict", "collections.Counter"):
                return Obj(n, value={})
            if n == "collections.deque": return Obj(n, value=[])
            if n in ("io.BytesIO", "io.StringIO"):
                return Obj(n, value=io.BytesIO() if n == "io.BytesIO" else io.StringIO())
            # Most Python class capabilities live in __init__, which NEWOBJ does
            # not invoke. These classes implement their behavior in __new__.
            if n not in {"functools.partial", "types.FunctionType", "types.CodeType", "numpy.memmap",
                         "builtins.map", "builtins.filter", "builtins.zip", "builtins.enumerate",
                         "operator.attrgetter", "operator.itemgetter", "operator.methodcaller", "functools._lru_cache_wrapper",
                         "types.MethodType", "types.MappingProxyType"} and not n.startswith("builtins."):
                return Obj(n, a)
        if n in ("builtins.getattr", "builtins.hasattr"):
            value = self.attr(a[0], a[1], a[2] if len(a) > 2 else MISSING)
            return True if n.endswith("hasattr") else value
        if n == "builtins.setattr":
            if not isinstance(a[0], Obj): raise LoadStopped("cannot assign attribute")
            if a[0].type == "functools.partial" and a[1] in ("func", "args", "keywords"):
                raise LoadStopped("read-only partial attribute")
            a[0].attrs[a[1]] = a[2]
            return None
        if n == "builtins.delattr":
            if isinstance(a[0], Obj): a[0].attrs.pop(a[1], None)
            return None
        if n == "builtins.vars":
            return self.attr(a[0], "__dict__") if a else {}
        if n in ("builtins.globals", "builtins.locals"):
            return {"__builtins__": Obj("module-dict", value=self.module("builtins"))}
        if n in ("builtins.__import__", "importlib.import_module"):
            name = a[0]
            if not isinstance(name, str): raise LoadStopped("module name must be str")
            if n == "builtins.__import__" and not (a[3] if len(a) > 3 else kw.get("fromlist")):
                name = name.split(".")[0]
            return self.module(name)
        if n == "pkgutil.resolve_name":
            name = a[0]
            if ":" in name:
                module, attrs = name.split(":", 1)
                value = self.module(module)
                for part in filter(None, attrs.split(".")): value = self.attr(value, part)
                return value
            parts = name.split(".")
            i = len(parts)
            while i > 1 and ".".join(parts[:i]) not in MODULE_NAMES: i -= 1
            value = self.module(".".join(parts[:i]))
            for part in parts[i:]: value = self.attr(value, part)
            return value
        if n in ("operator.attrgetter", "operator.itemgetter", "operator.methodcaller"):
            return Obj(n, a, dict(kw))
        if n == "operator.call": return self.call(a[0], a[1:], kw)
        if n == "functools.partial":
            if not a: raise LoadStopped("partial requires a callable")
            target, args, keywords = a[0], a[1:], dict(kw)
            if isinstance(target, Obj) and target.type == n and not target.attrs:
                args = target.args + args
                keywords = {**target.extra.get("keywords", {}), **keywords}
                target = target.value
            return Obj(n, args, value=target, extra={"keywords": keywords})
        if n in ("functools.lru_cache", "functools.cache"):
            if a and isinstance(a[0], (Ref, Obj, Method)):
                return Obj("wrapper", value=a[0])
            return Obj("lru-decorator")
        if n == "functools._lru_cache_wrapper": return Obj("wrapper", value=a[0])
        if n == "types.MethodType":
            return Obj("functools.partial", (a[1],), attrs={"__func__": a[0], "__self__": a[1]}, value=a[0])
        if n == "types.MappingProxyType": return Obj("mapping-proxy", value=a[0])
        if n in ("functools.update_wrapper", "functools.wraps"):
            if n.endswith("wraps"): return Obj("wraps-decorator", value=a[0])
            if isinstance(a[0], Obj):
                a[0].attrs["__wrapped__"] = a[1]
                return a[0]
            return Obj("wrapper", value=a[0], attrs={"__wrapped__": a[1]})
        if n in ("copy.copy", "copy.deepcopy"):
            return self.copy(a[0])
        if n in ("copyreg.add_extension", "copyreg.remove_extension", "copyreg.clear_extension_cache"):
            if not hasattr(self, "extensions"): self.extensions = {}
            if n.endswith("add_extension"): self.extensions[a[2]] = a[:2]
            elif n.endswith("remove_extension"): self.extensions.pop(a[2], None)
            return None
        if n in ("copyreg._reconstructor", "copyreg.__newobj__", "copyreg.__newobj_ex__"):
            if n.endswith("_reconstructor"):
                base = a[1]
                if isinstance(base, Ref) and base.name == "builtins.object":
                    return Obj(a[0].name) if isinstance(a[0], Ref) else self.call(a[0], (), new=True)
                result = self.call(base, (a[2],), new=True)
                if isinstance(base, Ref) and base.name not in ("builtins.str", "builtins.bytes", "builtins.int", "builtins.float", "builtins.tuple", "builtins.frozenset"):
                    self.method(result, "__init__", (a[2],), {}, declared=base.name)
                return result
            return self.call(a[0], a[1] if n.endswith("ex__") else a[1:], a[2] if n.endswith("ex__") else None, new=True)
        if n in ("types.FunctionType", "types.LambdaType"):
            if a and isinstance(a[0], Obj) and a[0].type == "code":
                self.flag("code-execution", "materialize serialized bytecode with types.FunctionType")
            raise LoadStopped("FunctionType requires a code object")
        if n in ("marshal.loads", "marshal.load"):
            if n.endswith(".load"):
                payload = self.buffer(a[0])
            else: payload = a[0]
            # Do not invoke marshal on untrusted data. Code object's leading
            # type is 'c', possibly with the reference flag (0x80).
            if isinstance(payload, (bytes, bytearray)) and payload:
                from modelgate.marshal_scan import read_marshaled
                return read_marshaled(payload, self.budget)
            raise LoadStopped("invalid marshal data")
        if n == "types.CodeType": return Obj("code", a)
        if n in ("pickle.loads", "pickle.load", "pickle._loads", "pickle._load", "pandas.read_pickle",
                 "pandas.io.pickle.read_pickle", "torch.load", "torch.serialization.load",
                 "torch.storage._load_from_bytes", "numpy.load", "numpy.lib.npyio.load", "numpy.lib._npyio_impl.load",
                 "numpy.lib.format.read_array", "joblib.load", "joblib.numpy_pickle.load"):
            return self.nested(n, a, kw)
        if n in ("pickle.Unpickler", "pickle._Unpickler"):
            return Obj("unpickler", a)
        if n in ("io.BytesIO", "io.StringIO"):
            return Obj(n, value=io.BytesIO(bytes(a[0]) if a else b"") if n.endswith("BytesIO") else io.StringIO(a[0] if a else ""))
        if n in ("types.ModuleType", "types.SimpleNamespace"):
            return Obj("module", a) if n.endswith("ModuleType") else Obj(n, a, dict(kw))
        if n.startswith("pathlib.") and n.rsplit(".", 1)[-1] in ("Path", "PosixPath", "WindowsPath", "PurePath", "PurePosixPath", "PureWindowsPath"):
            return Obj("path", value=posixpath.join(*(self.path(v) for v in a)) if a else ".")
        if n == "logging.config.dictConfig":
            self.logging_config(a[0])
            return None
        if n in ("timeit.timeit", "timeit.repeat", "timeit.Timer"):
            stmt = a[0] if a else kw.get("stmt")
            setup = a[1] if len(a) > 1 else kw.get("setup")
            if isinstance(stmt, str) or isinstance(setup, str):
                self.flag("code-execution", f"{n} compiles Python source")
            timer = Obj("timeit.Timer", (stmt, setup))
            if n != "timeit.Timer": self.method(timer, "timeit", (), {})
            return timer
        if n in CAPABILITIES:
            self.capability(n, a, kw)
            return UNKNOWN
        result = self.builtins(n, a, kw)
        if result is not MISSING: return result
        result = self.functional(n, a, kw)
        if result is not MISSING: return result
        result = self.pure(n, a, kw)
        if result is not MISSING: return result
        result = self.construct(n, a, kw)
        if result is not MISSING: return result
        # Unbound methods have the same semantics as their bound equivalent.
        parent, sep, method = n.rpartition(".")
        if sep and (parent in CLASS_NAMES or parent.rsplit(".", 1)[-1][:1].isupper()) and a:
            return self.method(a[0], method, a[1:], kw, declared=parent)
        if self.is_class(fn):
            return Obj(n, a, dict(kw))
        if self.safe_numerical(n):
            return Obj(n, a)
        self.unsupported(f"unmodelled callable {n}")

    def copy(self, value):
        if isinstance(value, Obj):
            if value.type == "module": raise LoadStopped("modules cannot be copied")
            if value.type in ("module-dict", "sys.modules"): return self.mapping(value).copy()
            if value.type == "wrapper": return value
            if value.type == "functools.partial":
                return Obj(value.type, value.args, value.attrs, value.value, dict(value.extra))
            return Obj(value.type, value.args, dict(value.attrs), value.value, dict(value.extra))
        if isinstance(value, (dict, list, set, bytearray)):
            return value.copy() if not isinstance(value, bytearray) else bytearray(value)
        return value

    def path(self, value):
        return value.value if isinstance(value, Obj) and value.type == "path" else value

    def buffer(self, value):
        if isinstance(value, Obj) and value.type in ("io.BytesIO", "io.StringIO"):
            return value.value.read()
        if isinstance(value, Obj) and value.type == "buffer-wrapper":
            return self.buffer(value.value)
        self.unsupported("loader input is not a known in-memory stream")

    def nested(self, name, args, kw):
        if not args: raise LoadStopped("loader requires input")
        value = args[0]
        loads = name in ("pickle.loads", "pickle._loads", "torch.storage._load_from_bytes")
        if loads:
            if not isinstance(value, (bytes, bytearray)):
                if isinstance(value, Obj): self.unsupported(f"unknown bytes argument to {name}")
                raise LoadStopped("loads requires a bytes-like object")
            payload = bytes(value)
        elif isinstance(value, (str, bytes)) or (isinstance(value, Obj) and value.type == "path"):
            self.flag("file-access", f"{name} reads a filesystem path")
        elif isinstance(value, Obj) and value.type in ("io.BytesIO", "io.StringIO"):
            payload = value.value.read()
        else:
            self.unsupported(f"cannot resolve input to {name}")
        if not isinstance(payload, (bytes, bytearray)):
            raise LoadStopped("binary loader requires bytes")
        self.nested_depth += 1
        oldloc = self.location
        try:
            if self.nested_depth > 20: raise AnalysisLimit("nested deserialization limit exceeded")
            self.budget.expand(len(payload))
            from modelgate.pickle_scan import Machine
            loc = oldloc + "!" + name
            if name.startswith(("pickle.", "pandas.", "joblib.")):
                rewind = True
                if name.startswith("pandas."):
                    fmt = kw.get("compression", "infer")
                    if isinstance(fmt, dict): fmt = fmt.get("method")
                    if fmt not in (None, "infer"):
                        payload = self.decompress(payload, fmt)
                        rewind = False
                if name.startswith("joblib."):
                    from modelgate.dispatch import decompress, compression
                    rewind = compression(payload) is None
                    payload = decompress(payload, self.budget)
                machine = Machine(self, loc)
                stream = io.BytesIO(payload)
                result = machine.one(stream)
                if rewind and isinstance(value, Obj) and value.type == "io.BytesIO":
                    value.value.seek(value.value.tell() - len(payload) + stream.tell())
                return result
            if name.startswith("torch."):
                return self.torch_load(payload, loc)
            allow_index = 1 if name.endswith("read_array") else 2
            allow = args[allow_index] if len(args) > allow_index else kw.get("allow_pickle", False)
            return self.numpy_load(payload, loc, allow, array_only=name.endswith("read_array"))
        finally:
            self.location = oldloc
            self.nested_depth -= 1

    def torch_load(self, payload, loc):
        from modelgate.pickle_scan import Machine, TORCH_MAGIC
        if payload.startswith(b"PK\x03\x04"):
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                names = [n for n in archive.namelist() if n == "data.pkl" or n.endswith("/data.pkl")]
                if not names: raise LoadStopped("checkpoint has no data.pkl")
                self.budget.expand(archive.getinfo(names[-1]).file_size)
                data = archive.read(names[-1])
                return Machine(self, loc + "!" + names[-1], persistent=True).one(io.BytesIO(data))
        stream = io.BytesIO(payload)
        machine = Machine(self, loc)
        if machine.one(stream) != TORCH_MAGIC: raise LoadStopped("invalid torch magic")
        if machine.one(stream) != 1001: raise LoadStopped("invalid torch protocol")
        machine.one(stream)
        machine.persistent = True
        result = machine.one(stream)
        machine.persistent = False
        machine.one(stream)
        return result

    def numpy_load(self, payload, loc, allow, *, array_only=False):
        from modelgate.pickle_scan import Machine
        from modelgate.dispatch import read_npy_header, _dtype_object
        if payload.startswith(b"\x93NUMPY"):
            header, offset = read_npy_header(payload)
            if _dtype_object(header["descr"]):
                if not allow: raise LoadStopped("NumPy pickle support is disabled")
                return Machine(self, loc + ":array").one(io.BytesIO(payload[offset:]))
            return Obj("numpy.ndarray", attrs={"shape": header["shape"]}, value=payload[offset:])
        if array_only: raise LoadStopped("not an NPY array")
        if payload.startswith((b"PK\x03\x04", b"PK\x05\x06")):
            return Obj("numpy.npz", attrs={"allow_pickle": allow, "location": loc}, value=payload)
        if not allow: raise LoadStopped("NumPy pickle support is disabled")
        return Machine(self, loc).one(io.BytesIO(payload))

    def capability(self, name, a, kw):
        kind = CAPABILITIES[name]
        if name in ("io.BufferedReader", "io.BufferedWriter", "io.BufferedRandom"):
            return
        zero_args = {"os.fork", "os.forkpty", "pty.fork", "tempfile.TemporaryFile", "tempfile.NamedTemporaryFile",
                     "tempfile.TemporaryDirectory", "tempfile.mkstemp", "tempfile.mkdtemp", "tempfile.mktemp",
                     "os.listdir", "os.scandir", "multiprocessing.Pool", "multiprocessing.pool.Pool",
                     "fileinput.input", "fileinput.FileInput"}
        if not a and not kw and name not in zero_args:
            raise LoadStopped("missing required argument")
        if name in ("builtins.eval", "builtins.exec", "builtins.compile"):
            first = a[0] if a else kw.get("source", kw.get("object"))
            if not isinstance(first, (str, bytes, bytearray)) and not (isinstance(first, Obj) and first.type == "code"):
                if isinstance(first, Obj): self.unsupported(f"unknown source argument to {name}")
                raise LoadStopped("Python source must be text or code")
            if name == "builtins.compile" and len(a) < 3 and not {"filename", "mode"} <= set(kw):
                raise LoadStopped("compile missing required arguments")
        if name in ("os.system", "os.popen") and a and not isinstance(a[0], (str, bytes)):
            if isinstance(a[0], Obj): self.unsupported(f"unknown command argument to {name}")
            raise LoadStopped("command must be text")
        if name == "os.system" and (len(a) != 1 or kw):
            raise LoadStopped("os.system requires one positional argument")
        if name in ("builtins.eval", "builtins.exec") and len(a) > 3:
            raise LoadStopped("too many arguments to Python evaluation")
        if name in ("builtins.open", "io.open", "io.FileIO", "os.open") and a:
            if not isinstance(self.path(a[0]), (str, bytes, int)):
                raise LoadStopped("invalid file argument")
        self.flag(kind, f"call {name}")

    def builtins(self, n, a, kw):
        if n == "builtins.object": return Obj(n)
        if n in ("builtins.str", "builtins.bytes", "builtins.bytearray", "builtins.int", "builtins.float",
                 "builtins.complex", "builtins.bool", "builtins.memoryview"):
            constructors = {"str": str, "bytes": bytes, "bytearray": bytearray, "int": int,
                            "float": float, "complex": complex, "bool": bool, "memoryview": bytes}
            if all(concrete(v) for v in a) and concrete(kw):
                if a and isinstance(a[0], int) and n in ("builtins.bytes", "builtins.bytearray") and a[0] > 16 * 1024 * 1024:
                    raise AnalysisLimit("byte allocation limit")
                return constructors[n.split(".")[1]](*a, **kw)
            if n == "builtins.bool": return True
            if n == "builtins.str" and a and isinstance(a[0], Obj) and a[0].type == "path": return a[0].value
            return UNKNOWN
        if n == "builtins.type":
            if len(a) == 1: return self.attr(a[0], "__class__")
            if len(a) == 3:
                return Obj("dynamic-class", a, dict(a[2]))
            raise LoadStopped("invalid type arguments")
        if n in ("builtins.list", "builtins.tuple", "builtins.set", "builtins.frozenset"):
            values = self.consume(a[0]) if a else []
            return {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[n.split(".")[1]](values)
        if n == "builtins.dict":
            result = dict(self.mapping(a[0])) if a and self.is_mapping(a[0]) else dict(self.consume(a[0])) if a else {}
            result.update(kw)
            return result
        if n == "builtins.slice": return slice(*a)
        if n == "builtins.range": return range(*a)
        if n in ("builtins.map", "builtins.filter", "itertools.starmap"):
            if len(a) < 2: raise LoadStopped("iterator needs function and input")
            return Lazy(n.rsplit(".", 1)[-1], a)
        if n in ("builtins.iter", "builtins.reversed", "builtins.enumerate", "builtins.zip"):
            if n == "builtins.iter" and len(a) == 1 and isinstance(a[0], Lazy): return a[0]
            return Lazy(n.rsplit(".", 1)[-1], a)
        if n == "builtins.next":
            try: return self.next(a[0])
            except StopIteration:
                if len(a) > 1: return a[1]
                raise LoadStopped("iterator exhausted")
        if n in ("builtins.sorted", "builtins.min", "builtins.max"):
            if not a: raise LoadStopped("missing iterable")
            values = self.consume(a[0]) if n.endswith("sorted") or len(a) == 1 else list(a)
            key = kw.get("key")
            keyed = [(self.call(key, (v,)) if key is not None else v, v) for v in values]
            if n.endswith("sorted"):
                try: return [v for k, v in sorted(keyed, key=lambda p: p[0], reverse=kw.get("reverse", False))]
                except TypeError: return values
            if not values:
                if "default" in kw: return kw["default"]
                raise LoadStopped("empty min/max")
            if all(concrete(k) for k, v in keyed):
                return (min if n.endswith("min") else max)(keyed, key=lambda p: p[0])[1]
            return values[0]
        if n in ("builtins.all", "builtins.any"):
            it = self.iterator(a[0])
            want = n.endswith("any")
            while True:
                try: value = self.next(it)
                except StopIteration: return not want
                if bool(value) == want: return want
        if n == "builtins.sum":
            result = a[1] if len(a) > 1 else 0
            for v in self.consume(a[0]):
                result = result + v if concrete(result) and concrete(v) else UNKNOWN
            return result
        if n in ("builtins.len", "operator.length_hint"):
            return len(a[0]) if not isinstance(a[0], (Obj, Lazy)) else 0
        if n in ("builtins.repr", "builtins.ascii", "builtins.format"):
            if concrete(a): return {"repr": repr, "ascii": ascii, "format": format}[n.split(".")[1]](*a)
            return "<symbolic object>"
        if n in ("builtins.abs", "builtins.round", "builtins.divmod", "builtins.pow", "builtins.chr", "builtins.ord", "builtins.hex", "builtins.oct", "builtins.bin"):
            if n.endswith("pow") and len(a) > 1 and isinstance(a[1], int) and abs(a[1]) > 10000:
                raise AnalysisLimit("integer power limit")
            if concrete(a):
                funcs = {"abs": abs, "round": round, "divmod": divmod, "pow": pow, "chr": chr, "ord": ord, "hex": hex, "oct": oct, "bin": bin}
                return funcs[n.split(".")[1]](*a, **kw)
            return UNKNOWN
        if n == "builtins.callable": return isinstance(a[0], (Ref, Method, Obj))
        if n in ("builtins.isinstance", "builtins.issubclass"): return False
        if n in ("builtins.staticmethod", "builtins.classmethod"):
            return Obj("wrapper", value=a[0])
        return MISSING

    def functional(self, n, a, kw):
        if n == "functools.reduce":
            values = self.consume(a[1])
            if len(a) > 2: acc = a[2]
            elif values: acc = values.pop(0)
            else: raise LoadStopped("reduce of empty input")
            for v in values: acc = self.call(a[0], (acc, v))
            return acc
        if n == "functools.cmp_to_key": return Obj("cmp-wrapper", value=a[0])
        if n.startswith("itertools."):
            kind = n.split(".")[1]
            if kind in ("chain", "chain.from_iterable", "islice", "repeat", "count", "cycle", "takewhile", "dropwhile", "filterfalse", "accumulate", "compress", "tee", "zip_longest", "pairwise", "batched", "groupby", "product", "permutations", "combinations", "combinations_with_replacement"):
                if kind == "tee":
                    shared = [self.iterator(a[0]), [], False]
                    return tuple(Lazy("tee", (), state=shared) for _ in range(a[1] if len(a) > 1 else 2))
                if kind in ("product", "permutations", "combinations", "combinations_with_replacement"):
                    values = tuple(self.consume(v) for v in a) if kind == "product" else (self.consume(a[0]),) + a[1:]
                    return Lazy("native", (getattr(itertools, kind)(*values, **kw),))
                return Lazy(kind, a)
        if n == "itertools.chain.from_iterable": return Lazy("chain_from", a)
        if n in ("operator.getitem", "operator.__getitem__"): return self.getitem(*a)
        if n in ("operator.setitem", "operator.__setitem__"):
            self.setitem(*a)
            return None
        if n in ("operator.delitem", "operator.__delitem__"):
            del a[0][a[1]]
            return None
        if n.startswith("operator."):
            op = n.split(".")[1].strip("_")
            funcs = {"add": operator.add, "concat": operator.concat, "iadd": operator.iadd, "iconcat": operator.iconcat,
                     "mul": operator.mul, "imul": operator.imul, "sub": operator.sub, "mod": operator.mod,
                     "truediv": operator.truediv, "floordiv": operator.floordiv, "and": operator.and_, "or": operator.or_,
                     "xor": operator.xor, "lshift": operator.lshift, "rshift": operator.rshift, "not": operator.not_,
                     "neg": operator.neg, "pos": operator.pos, "invert": operator.invert, "eq": operator.eq,
                     "ne": operator.ne, "lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge,
                     "is": operator.is_, "is_not": operator.is_not, "truth": operator.truth, "contains": operator.contains,
                     "index": operator.index, "indexOf": operator.indexOf, "countOf": operator.countOf}
            if op in ("add", "concat", "iadd", "iconcat") and len(a) == 2 and type(a[0]) is type(a[1]) and isinstance(a[0], (list, tuple)):
                if len(a[0]) + len(a[1]) > 1000000: raise AnalysisLimit("sequence limit")
                return funcs[op](*a)
            if op in funcs and concrete(a):
                if op in ("mul", "imul") and len(a) == 2:
                    seq, num = a if isinstance(a[1], int) else a[::-1]
                    if isinstance(seq, (str, bytes, bytearray, list, tuple)) and isinstance(num, int) and len(seq) * num > 16*1024*1024:
                        raise AnalysisLimit("sequence allocation limit")
                if op in ("lshift", "rshift") and abs(a[1]) > 1000000: raise AnalysisLimit("shift limit")
                return funcs[op](*a)
        return MISSING

    def iterator(self, obj):
        if isinstance(obj, Lazy): return obj
        if isinstance(obj, Obj) and obj.type in ("collections.OrderedDict", "collections.defaultdict", "collections.Counter"):
            obj = obj.value
        if isinstance(obj, Obj) and obj.type == "collections.deque": obj = obj.value
        if isinstance(obj, Obj) and obj.type == "numpy.ndarray" and isinstance(obj.value, tuple) and len(obj.value) > 4:
            obj = obj.value[4]
        if isinstance(obj, Obj):
            self.unsupported(f"cannot iterate {obj.type}")
        if not isinstance(obj, (tuple, list, dict, set, frozenset, str, bytes, bytearray, range)):
            raise LoadStopped("not iterable")
        return Lazy("values", (tuple(obj) if isinstance(obj, (dict, set, frozenset)) else obj,))

    def next(self, it):
        if not isinstance(it, Lazy): raise LoadStopped("not an iterator")
        if it.ended: raise StopIteration
        try:
            return self._next(it)
        except StopIteration:
            it.ended = True
            raise

    def _next(self, it):
        self.budget.tick()
        if not isinstance(it, Lazy): raise LoadStopped("not an iterator")
        k, a = it.kind, it.args
        if k == "native": return next(a[0])
        if k == "tee":
            source, cache, done = it.state
            if it.index == len(cache):
                if done: raise StopIteration
                try: cache.append(self.next(source))
                except StopIteration:
                    it.state[2] = True
                    raise
            value = cache[it.index]
            it.index += 1
            return value
        if k == "values":
            if it.index >= len(a[0]): raise StopIteration
            value = a[0][it.index]
            it.index += 1
            return value
        if k in ("file-iterator", "network-iterator"):
            self.flag("file-access" if k == "file-iterator" else "network", "consume " + str(a[0]))
        if k == "iter" and len(a) == 2:
            value = self.call(a[0], ())
            if value == a[1]: raise StopIteration
            return value
        if k == "repeat":
            if len(a) > 1 and it.index >= a[1]: raise StopIteration
            it.index += 1
            return a[0]
        if k == "count":
            value = (a[0] if a else 0) + it.index * (a[1] if len(a) > 1 else 1)
            it.index += 1
            return value
        if k == "map":
            if it.state is None: it.state = [self.iterator(v) for v in a[1:]]
            values = []
            for v in it.state: values.append(self.next(v))
            return self.call(a[0], tuple(values))
        if k == "starmap":
            if it.state is None: it.state = self.iterator(a[1])
            return self.call(a[0], tuple(self.consume(self.next(it.state))))
        if k in ("filter", "filterfalse", "takewhile", "dropwhile"):
            if it.state is None: it.state = self.iterator(a[1])
            while True:
                v = self.next(it.state)
                if k == "dropwhile" and it.index: return v
                result = self.call(a[0], (v,)) if a[0] is not None else v
                yes = bool(result)
                if k == "takewhile":
                    if not yes: raise StopIteration
                    return v
                if k == "dropwhile":
                    if not yes:
                        it.index = 1
                        return v
                elif yes != (k == "filterfalse"): return v
        if k in ("iter", "enumerate", "reversed", "cycle"):
            if it.state is None:
                seq = self.consume(a[0])[::-1] if k == "reversed" else a[0]
                it.state = self.iterator(seq)
            try: value = self.next(it.state)
            except StopIteration:
                if k != "cycle" or it.index == 0: raise
                it.state = self.iterator(a[0])
                value = self.next(it.state)
            i = it.index
            it.index += 1
            return (i + (a[1] if len(a) > 1 else 0), value) if k == "enumerate" else value
        if k in ("zip", "zip_longest"):
            if not a: raise StopIteration
            if it.state is None: it.state = [self.iterator(v) for v in a]
            values, ended = [], 0
            for v in it.state:
                try: values.append(self.next(v))
                except StopIteration:
                    if k == "zip": raise
                    values.append(None)
                    ended += 1
            if ended == len(a): raise StopIteration
            return tuple(values)
        if k in ("chain", "chain_from"):
            if it.state is None: it.state = [self.iterator(a if k == "chain" else a[0]), None]
            while True:
                if it.state[1] is None: it.state[1] = self.iterator(self.next(it.state[0]))
                try: return self.next(it.state[1])
                except StopIteration: it.state[1] = None
        if k == "islice":
            start, stop, step = (0, a[1], 1) if len(a) == 2 else (a[1] or 0, a[2], a[3] if len(a) > 3 and a[3] is not None else 1)
            if start < 0 or step <= 0 or (stop is not None and stop < 0): raise LoadStopped("invalid islice indices")
            if it.state is None: it.state = [self.iterator(a[0]), 0]
            target = start + it.index * step
            # CPython consumes the initial `start` items even if start > stop.
            # Subsequent skips stop exactly at `stop`, including a final skip
            # that produces no output.
            limit = target if it.index == 0 or stop is None else min(target, stop)
            while it.state[1] < limit:
                self.next(it.state[0])
                it.state[1] += 1
            if stop is not None and it.state[1] >= stop: raise StopIteration
            value = self.next(it.state[0])
            it.state[1] += 1
            it.index += 1
            return value
        if k == "accumulate":
            if it.state is None:
                it.state = [self.iterator(a[0]), MISSING]
            v = self.next(it.state[0])
            if it.state[1] is MISSING: it.state[1] = v
            elif len(a) > 1 and a[1] is not None: it.state[1] = self.call(a[1], (it.state[1], v))
            elif concrete((it.state[1], v)): it.state[1] += v
            else: it.state[1] = UNKNOWN
            return it.state[1]
        if k == "compress":
            if it.state is None: it.state = [self.iterator(v) for v in a]
            while True:
                v, selector = self.next(it.state[0]), self.next(it.state[1])
                if selector: return v
        if k == "groupby":
            if it.state is None: it.state = self.iterator(a[0])
            v = self.next(it.state)
            return (self.call(a[1], (v,)) if len(a) > 1 and a[1] is not None else v, Lazy("values", ([v],)))
        if k == "pairwise":
            if it.state is None:
                iterator = self.iterator(a[0])
                it.state = [iterator, self.next(iterator)]
            v = self.next(it.state[0])
            pair = (it.state[1], v)
            it.state[1] = v
            return pair
        if k == "batched":
            if it.state is None: it.state = self.iterator(a[0])
            values = []
            for _ in range(a[1]):
                try: values.append(self.next(it.state))
                except StopIteration: break
            if not values: raise StopIteration
            return tuple(values)
        self.unsupported(f"cannot consume iterator {k}")

    def consume(self, value):
        it = self.iterator(value)
        result = []
        while True:
            try: result.append(self.next(it))
            except StopIteration: return result
            if len(result) > 100000: raise AnalysisLimit("iterator expansion limit")

    def is_mapping(self, obj):
        return isinstance(obj, dict) or isinstance(obj, Obj) and obj.type in ("module-dict", "sys.modules", "collections.OrderedDict", "collections.defaultdict", "collections.Counter")

    def mapping(self, obj):
        if isinstance(obj, dict): return obj
        if isinstance(obj, Obj) and isinstance(obj.value, dict): return obj.value
        if isinstance(obj, Obj) and obj.type == "module-dict":
            module = obj.value
            prefix = module.args[0] + "."
            names = {n[len(prefix):].split(".")[0] for n in set(CAPABILITIES) | set(MODULE_ATTRS) | MODULE_NAMES | CLASS_NAMES if n.startswith(prefix)}
            if module.args[0] == "builtins":
                import builtins
                names.update(vars(builtins))
            if module.args[0] == "os": names.update(("fspath", "fsencode", "fsdecode", "getcwd", "getenv", "environ"))
            if module.args[0] == "sys": names.add("modules")
            result = {name: self.attr(module, name) for name in names if name != "__dict__"}
            result.update(module.attrs)
            return result
        if isinstance(obj, Obj) and obj.type == "sys.modules":
            return {name: self.module(name) for name in MODULE_NAMES | set(self.modules)}
        raise LoadStopped("not a mapping")

    def pure(self, n, a, kw):
        if n in ("os.getpid", "os.getppid", "os.getuid", "os.geteuid", "os.getgid", "os.getegid", "os.cpu_count"):
            return 1
        if n in ("os.getenv", "os.get_exec_path", "os.getcwd", "os.getcwdb"):
            return UNKNOWN
        if n in ("os.fspath", "os.fsencode", "os.fsdecode"):
            v = self.path(a[0])
            if n.endswith("fsencode"): return v.encode() if isinstance(v, str) else v
            if n.endswith("fsdecode"): return v.decode() if isinstance(v, bytes) else v
            return v
        if n.startswith(("posixpath.", "ntpath.", "os.path.")):
            name = n.rsplit(".", 1)[-1]
            if name in ("exists", "lexists", "isfile", "isdir", "islink", "ismount", "getsize", "getmtime", "getatime", "getctime", "samefile", "realpath"):
                self.flag("file-access", f"call {n}")
            if name in ("join", "split", "splitext", "basename", "dirname", "normpath", "normcase", "commonprefix", "commonpath", "isabs", "splitdrive"):
                if concrete(a): return getattr(posixpath, name)(*a)
        if n in ("_codecs.encode", "codecs.encode", "_codecs.decode", "codecs.decode"):
            value = a[0]
            encoding = (a[1] if len(a) > 1 else kw.get("encoding", "utf-8")).lower().replace("_", "-")
            if encoding in ("base64", "base64-codec"):
                return base64.encodebytes(value) if n.endswith("encode") else base64.decodebytes(value)
            if encoding in ("hex", "hex-codec"):
                return binascii.hexlify(value) if n.endswith("encode") else binascii.unhexlify(value)
            if encoding in ("zlib", "zlib-codec") and n.endswith("decode"):
                return self.decompress(value, "zlib")
            if encoding in ("bz2", "bz2-codec") and n.endswith("decode"):
                return self.decompress(value, "bz2")
            if encoding in ("rot13", "rot-13"):
                import codecs
                return codecs.encode(value, "rot_13")
            if encoding not in {"utf-8", "utf8", "latin1", "latin-1", "iso-8859-1", "ascii", "unicode-escape", "raw-unicode-escape", "utf-16", "utf-16le", "utf-16-le", "utf-16be", "utf-16-be", "utf-32"}:
                self.unsupported("unmodelled text encoding")
            errors = a[2] if len(a) > 2 else kw.get("errors", "strict")
            return value.encode(encoding, errors) if n.endswith("encode") else value.decode(encoding, errors)
        if n.startswith("base64."):
            name = n.split(".")[1]
            if name in ("b64decode", "standard_b64decode", "urlsafe_b64decode", "decodebytes", "b32decode", "b16decode", "b85decode", "a85decode", "b64encode", "standard_b64encode", "urlsafe_b64encode", "encodebytes", "b32encode", "b16encode", "b85encode", "a85encode") and concrete(a) and concrete(kw):
                result = getattr(base64, name)(*a, **kw)
                self.budget.expand(len(result))
                return result
        if n.startswith("binascii."):
            name = n.split(".")[1]
            if name in ("unhexlify", "a2b_hex", "a2b_base64", "a2b_qp", "a2b_uu", "hexlify", "b2a_hex", "b2a_base64", "crc32", "crc_hqx") and concrete(a) and concrete(kw):
                return getattr(binascii, name)(*a, **kw)
        if n in ("zlib.decompress", "bz2.decompress", "lzma.decompress", "gzip.decompress"):
            return self.decompress(a[0], n.split(".")[0], a[1:] if len(a) > 1 else (), kw)
        if n in ("zlib.decompressobj", "bz2.BZ2Decompressor", "lzma.LZMADecompressor"):
            return Obj("decompressor", a, {"module": n.split(".")[0]})
        if n in ("bytes.fromhex", "builtins.bytes.fromhex", "builtins.bytearray.fromhex"):
            result = bytes.fromhex(a[0])
            return bytearray(result) if "bytearray" in n else result
        if n == "struct.pack":
            import struct
            if concrete(a):
                if struct.calcsize(a[0]) > 16 * 1024 * 1024: raise AnalysisLimit("packed buffer size limit")
                return struct.pack(*a)
        if n in ("json.loads", "ast.literal_eval"):
            if not isinstance(a[0], (str, bytes)): raise LoadStopped("literal must be text")
            if n == "json.loads":
                import json
                result = json.loads(a[0])
                hook = kw.get("object_pairs_hook") or kw.get("object_hook")
                if hook is not None and isinstance(result, dict):
                    result = self.call(hook, (list(result.items()) if "object_pairs_hook" in kw else result,))
                return result
            import ast
            return ast.literal_eval(a[0])
        if n.startswith("math.") and concrete(a):
            method = n.split(".")[1]
            if method in ("sqrt", "log", "log1p", "exp", "expm1", "sin", "cos", "tan", "floor", "ceil", "fabs", "isnan", "isfinite"):
                return getattr(math, method)(*a)
        return MISSING

    def decompress(self, data, fmt, a=(), kw=None):
        from modelgate.dispatch import decompress_one
        return decompress_one(data, fmt, self.budget, a, kw or {})

    def construct(self, n, a, kw):
        if n in ("numpy.str_", "numpy.bytes_"):
            return self.builtins("builtins.str" if n.endswith("str_") else "builtins.bytes", a, kw)
        if n in ("re.compile", "re._compile"):
            return Obj("re.Pattern", a)
        if n in ("collections.OrderedDict", "collections.defaultdict", "collections.Counter"):
            source = a[1:] if n.endswith("defaultdict") else a
            value = dict(self.mapping(source[0])) if source and self.is_mapping(source[0]) else dict(self.consume(source[0])) if source else {}
            value.update(kw)
            return Obj(n, a[:1] if n.endswith("defaultdict") else (), value=value)
        if n == "collections.deque":
            values = self.consume(a[0]) if a else []
            length = a[1] if len(a) > 1 else kw.get("maxlen")
            return Obj(n, value=values[-length:] if length else values if length is None else [])
        if n == "numpy.dtype": return Obj(n, a)
        if n in ("numpy.core.multiarray._reconstruct", "numpy._core.multiarray._reconstruct"):
            return Obj("numpy.ndarray", a)
        if n in ("numpy.core.numeric._frombuffer", "numpy._core.numeric._frombuffer", "numpy.frombuffer", "numpy.array", "numpy.asarray"):
            return Obj("numpy.ndarray", a)
        if n in ("threading.Thread", "threading.Timer"):
            if n.endswith("Timer"):
                target = a[1] if len(a) > 1 else kw.get("function")
                args = a[2] if len(a) > 2 else kw.get("args", ())
                kwargs = a[3] if len(a) > 3 else kw.get("kwargs", {})
                return Obj(n, attrs={"function": target, "args": tuple(args or ()), "kwargs": kwargs or {}})
            else:
                target = a[1] if len(a) > 1 else kw.get("target")
                args = a[3] if len(a) > 3 else kw.get("args", ())
                kwargs = a[4] if len(a) > 4 else kw.get("kwargs", {})
            return Obj(n, attrs={"_target": target, "_args": tuple(args or ()), "_kwargs": kwargs or {}})
        if n in ("_thread.start_new_thread", "_thread.start_new", "threading._start_new_thread"):
            return self.call(a[0], a[1], a[2] if len(a) > 2 else {})
        if n in ("concurrent.futures.ThreadPoolExecutor", "concurrent.futures.thread.ThreadPoolExecutor", "concurrent.futures.ProcessPoolExecutor", "concurrent.futures.process.ProcessPoolExecutor"):
            return Obj(n, a, dict(kw))
        if n in ("http.client.HTTPConnection", "http.client.HTTPSConnection", "socket.socket", "ssl.SSLSocket", "xmlrpc.client.ServerProxy", "urllib.request.Request", "requests.Session", "requests.sessions.Session"):
            return Obj(n, a, dict(kw))
        if n in ("ftplib.FTP", "ftplib.FTP_TLS", "smtplib.SMTP", "smtplib.SMTP_SSL", "smtplib.LMTP", "telnetlib.Telnet", "poplib.POP3", "poplib.POP3_SSL", "imaplib.IMAP4", "imaplib.IMAP4_SSL", "nntplib.NNTP"):
            host = a[0] if a else kw.get("host")
            if host or n.startswith(("poplib.", "imaplib.", "nntplib.")):
                self.flag("network", f"{n} connects during construction")
            return Obj(n, a, dict(kw))
        if n in ("io.TextIOWrapper", "io.BufferedReader", "io.BufferedWriter", "io.BufferedRandom"):
            return Obj("buffer-wrapper", value=a[0])
        if n in ("gzip.GzipFile", "bz2.BZ2File", "lzma.LZMAFile"):
            source = kw.get("fileobj", a[0] if a else None)
            if isinstance(source, Obj) and source.type == "io.BytesIO":
                data = self.decompress(source.value.read(), n.split(".")[0])
                return Obj("io.BytesIO", value=io.BytesIO(data))
            self.flag("file-access", f"{n} opens a file")
        if n in ("gzip.open", "bz2.open", "lzma.open", "zipfile.ZipFile", "tarfile.open", "tarfile.TarFile.open"):
            source = kw.get("fileobj", a[0] if a else None)
            if isinstance(source, Obj) and source.type == "io.BytesIO": return Obj(n, a, value=source)
            self.flag("file-access", f"{n} opens a file")
        if n in ("os.walk", "os.fwalk", "glob.iglob"):
            return Lazy("file-iterator", (n,))
        if n in ("ctypes.CDLL", "ctypes.PyDLL", "ctypes.WinDLL", "ctypes.OleDLL"):
            if a and a[0] is not None: self.flag("code-execution", f"{n} loads a native library")
            return Obj(n, a)
        if n == "weakref.ref": return Obj("wrapper", value=a[0])
        if n == "weakref.finalize":
            return self.call(a[1], a[2:], kw)
        return MISSING

    def method(self, obj, name, a, kw, declared=None):
        if name == "__call__": return self.call(obj, a, kw)
        if name == "__getattribute__": return self.attr(obj, a[0])
        if name == "__setattr__": return self.call(Ref("builtins.setattr"), (obj,) + a, kw)
        if name == "__getitem__" and (not isinstance(obj, Obj) or obj.type in ("sys.modules", "module-dict", "collections.OrderedDict", "collections.defaultdict", "collections.Counter")):
            return self.getitem(obj, a[0])
        if name == "__setitem__":
            if not isinstance(obj, Obj) or obj.type in ("sys.modules", "module-dict", "collections.OrderedDict", "collections.defaultdict", "collections.Counter"):
                self.setitem(obj, a[0], a[1])
                return None
        if name == "__new__" and isinstance(obj, Ref): return self.call(obj, a, kw, new=True)
        if isinstance(obj, Lazy):
            if name == "__next__":
                try: return self.next(obj)
                except StopIteration: raise LoadStopped("iterator exhausted")
            if name == "__iter__": return obj
            if name == "__setstate__":
                self.build(obj, a[0])
                return None
        if self.is_mapping(obj) and name in ("get", "pop", "setdefault", "update", "__ior__", "items", "keys", "values", "copy", "clear"):
            if name in ("get", "pop", "setdefault"):
                if isinstance(obj, Obj) and obj.type in ("module-dict", "sys.modules"):
                    try: return self.getitem(obj, a[0])
                    except LoadStopped: return a[1] if len(a) > 1 else None
                mapping = self.mapping(obj)
                return getattr(mapping, name)(*a)
            if name in ("update", "__ior__"):
                source = self.mapping(a[0]) if a and self.is_mapping(a[0]) else dict(self.consume(a[0])) if a else {}
                for k, v in {**source, **kw}.items(): self.setitem(obj, k, v)
                return obj if name == "__ior__" else None
            mapping = self.mapping(obj)
            if name == "items": return list(mapping.items())
            if name == "keys": return list(mapping.keys())
            if name == "values": return list(mapping.values())
            if name == "copy": return mapping.copy()
            mapping.clear()
            return None
        if isinstance(obj, Obj):
            typ = declared or obj.type
            if name == "__setstate__":
                self.build(obj, a[0])
                return None
            if name == "__init__":
                replacement = self.call(Ref(typ), a, kw)
                if isinstance(replacement, Obj):
                    obj.args, obj.attrs, obj.value = replacement.args, replacement.attrs, replacement.value
                return None
            if obj.type in ("io.BytesIO", "io.StringIO"):
                if name in ("read", "read1", "readline", "readlines", "getvalue", "getbuffer", "seek", "tell", "write", "writelines", "truncate", "flush", "close", "readinto", "readable", "writable", "seekable"):
                    method = "getvalue" if name == "getbuffer" else "read" if name == "read1" else name
                    if all(concrete(v) for v in a): return getattr(obj.value, method)(*a, **kw)
                if name == "__iter__": return self.iterator(obj.value.readlines())
            if obj.type == "buffer-wrapper": return self.method(obj.value, name, a, kw)
            if obj.type == "mapping-proxy": return self.method(obj.value, name, a, kw)
            if obj.type == "decompressor" and name == "decompress":
                return self.decompress(a[0], obj.attrs["module"], obj.args)
            if obj.type == "decompressor" and name == "flush": return b""
            if obj.type == "unpickler" and name == "load":
                from modelgate.pickle_scan import Machine
                source = obj.args[0]
                if not isinstance(source, Obj) or source.type != "io.BytesIO":
                    self.unsupported("Unpickler input is not an in-memory stream")
                if "machine" not in obj.extra:
                    obj.extra["machine"] = Machine(self, self.location + "!Unpickler.load")
                    obj.extra["memo"] = {}
                return obj.extra["machine"].one(source.value, obj.extra["memo"])
            if obj.type == "numpy.npz":
                if name in ("__getitem__", "get"):
                    with zipfile.ZipFile(io.BytesIO(obj.value)) as archive:
                        key = a[0] if a[0] in archive.namelist() else a[0] + ".npy"
                        if key not in archive.namelist():
                            if name == "get": return a[1] if len(a) > 1 else None
                            raise LoadStopped("missing NPZ array")
                        self.budget.expand(archive.getinfo(key).file_size)
                        payload = archive.read(key)
                    if not payload.startswith(b"\x93NUMPY"): return payload
                    return self.numpy_load(payload, obj.attrs["location"] + "!" + key, obj.attrs["allow_pickle"])
                if name == "close": return None
            if obj.type == "numpy.ndarray" and name in ("item", "__getitem__", "tolist", "flatten", "ravel"):
                values = obj.value[4] if isinstance(obj.value, tuple) and len(obj.value) == 5 else obj.value
                if name in ("flatten", "ravel"): return obj
                if isinstance(values, list):
                    if name == "tolist": return values
                    return values[a[0] if a else 0]
                return UNKNOWN
            if typ == "path" or typ.startswith("pathlib."):
                if name in ("write_text", "write_bytes", "read_text", "read_bytes", "open", "touch", "unlink", "rename", "replace", "mkdir", "rmdir", "chmod", "lchmod", "symlink_to", "hardlink_to", "link_to", "stat", "lstat", "exists", "is_file", "is_dir", "is_symlink", "is_mount", "is_socket", "is_fifo", "is_block_device", "is_char_device", "resolve", "readlink", "owner", "group", "samefile"):
                    self.flag("file-access", f"call pathlib.Path.{name}")
                if name in ("iterdir", "glob", "rglob", "walk"):
                    return Lazy("file-iterator", ("pathlib.Path." + name,))
                if name in ("__str__", "__fspath__", "as_posix"): return obj.value
                if name in ("joinpath", "__truediv__"):
                    return Obj("path", value=posixpath.join(obj.value, *(self.path(v) for v in a)))
                if name == "with_name": return Obj("path", value=posixpath.join(posixpath.dirname(obj.value), a[0]))
            if typ.startswith(("socket.", "ssl.", "http.client.", "ftplib.", "smtplib.", "telnetlib.", "poplib.", "imaplib.", "nntplib.", "requests.", "httpx.")):
                if name in ("connect", "connect_ex", "request", "send", "sendall", "sendto", "sendmsg", "putrequest", "endheaders", "login", "get", "post", "put", "delete", "patch", "head", "options", "sendmail", "send_message", "retrbinary", "retrlines", "storbinary", "storlines", "nlst", "dir", "cwd", "quit", "noop", "ehlo", "helo", "starttls"):
                    self.flag("network", f"call {typ}.{name}")
                return UNKNOWN
            if typ == "xmlrpc.client.ServerProxy":
                self.flag("network", f"call XML-RPC method {name}")
            if typ in ("threading.Thread", "threading.Timer"):
                if name in ("start", "run"):
                    target = obj.attrs.get("_target", obj.attrs.get("function"))
                    if target is not None:
                        try:
                            return self.call(target, tuple(obj.attrs.get("_args", obj.attrs.get("args", ()))), obj.attrs.get("_kwargs", obj.attrs.get("kwargs", {})))
                        except LoadStopped:
                            if name == "run": raise
                            return None
                    return None
                return UNKNOWN
            if "Process" in typ and typ.startswith("multiprocessing.") and name == "start":
                self.flag("process", f"call {typ}.start")
            if typ.endswith(("ThreadPoolExecutor", "ProcessPoolExecutor")):
                if name in ("submit", "map"):
                    if typ.endswith("ProcessPoolExecutor"): self.flag("process", "start process pool workers")
                    initializer = obj.attrs.get("initializer")
                    if initializer is not None: self.call(initializer, tuple(obj.attrs.get("initargs", ())))
                    if name == "submit":
                        try: result = self.call(a[0], a[1:], kw)
                        except LoadStopped: result = UNKNOWN
                        return Obj("future", value=result)
                    values = self.consume(Lazy("map", a))
                    return Lazy("values", (values,))
                return None
            if typ == "future":
                if name == "result": return obj.value
                if name == "add_done_callback": return self.call(a[0], (obj,))
            if typ in ("bdb.Bdb", "pdb.Pdb", "code.InteractiveInterpreter", "code.InteractiveConsole"):
                if name in ("run", "runeval", "runsource", "runcode", "push"):
                    if not a and not kw: raise LoadStopped("missing source argument")
                    self.flag("code-execution", f"call {typ}.{name}")
                if name == "runcall": return self.call(a[0], a[1:], kw)
            if typ == "timeit.Timer" and name in ("timeit", "repeat", "autorange"):
                for fn in obj.args[::-1]:
                    if fn is None: continue
                    if isinstance(fn, str): self.flag("code-execution", "timeit source execution")
                    self.call(fn, ())
                return 0.0
            if typ == "logging.config.DictConfigurator" and name == "configure":
                self.logging_config(obj.args[0])
                return None
            if typ in ("logging.config.DictConfigurator", "logging.config.BaseConfigurator") and name in ("resolve", "ext_convert"):
                return self.call(Ref("pkgutil.resolve_name"), (a[0],))
            if typ.startswith("ctypes."):
                self.flag("code-execution", f"call native function {typ}.{name}")
            if typ == "collections.deque":
                return self.method(obj.value, name, a, kw)
            if name in ("append", "extend", "add", "update"):
                if obj.value is None: obj.value = []
                if isinstance(obj.value, list):
                    obj.value.extend(self.consume(a[0])) if name in ("extend", "update") else obj.value.append(a[0])
                    return None
            self.unsupported(f"unmodelled method {typ}.{name}")
        if isinstance(obj, list) and name == "sort":
            obj[:] = self.builtins("builtins.sorted", (obj,), kw)
            return None
        if isinstance(obj, (list, set)) and name in ("extend", "update"):
            values = self.consume(a[0])
            obj.extend(values) if isinstance(obj, list) else obj.update(values)
            return None
        # Access to built-in methods is restricted to data types and named pure
        # operations. No module or artifact-provided method is ever called here.
        safe = {
            str: {"encode", "join", "replace", "format", "format_map", "lower", "upper", "strip", "lstrip", "rstrip", "split", "rsplit", "splitlines", "partition", "rpartition", "removeprefix", "removesuffix", "startswith", "endswith", "capitalize", "casefold", "swapcase", "translate", "__add__", "__mul__", "__getitem__"},
            bytes: {"decode", "join", "replace", "hex", "fromhex", "strip", "lstrip", "rstrip", "split", "__add__", "__mul__", "__getitem__"},
            bytearray: {"decode", "join", "replace", "hex", "fromhex", "append", "extend", "__add__", "__mul__", "__getitem__"},
            list: {"append", "pop", "insert", "remove", "clear", "copy", "reverse", "__add__", "__mul__", "__getitem__"},
            tuple: {"count", "index", "__add__", "__mul__", "__getitem__"},
            set: {"add", "remove", "discard", "pop", "clear", "copy", "union", "intersection", "difference"},
            dict: {"__getitem__", "__setitem__"},
        }
        if name in safe.get(type(obj), set()):
            if name in ("join",) and a:
                values = self.consume(a[0])
                if all(isinstance(v, (str, bytes, bytearray)) for v in values):
                    self.budget.expand(sum(len(v) for v in values) + len(obj) * max(0, len(values)-1))
                a = (values,) + a[1:]
            if name in ("__mul__",) and a and isinstance(a[0], int) and len(obj) * a[0] > 16*1024*1024:
                raise AnalysisLimit("sequence allocation limit")
            if type(obj) in (str, bytes, bytearray) and not concrete(a): return UNKNOWN
            if name in ("format", "format_map"):
                if any(int(n) > 1000000 for n in re.findall(r"\d{7,}", obj)):
                    raise AnalysisLimit("format field width limit")
            if name == "replace" and len(a) >= 2:
                count = obj.count(a[0])
                if len(a) > 2 and a[2] >= 0: count = min(count, a[2])
                self.budget.expand(max(len(obj), len(obj) + count * (len(a[1]) - len(a[0]))))
            if name == "encode": return self.pure("codecs.encode", (obj,) + a, kw)
            if name == "decode": return self.pure("codecs.decode", (obj,) + a, kw)
            return getattr(obj, name)(*a, **kw)
        if name == "__iter__": return self.iterator(obj)
        if name == "__len__": return len(obj)
        raise LoadStopped("unsupported attribute on primitive value")

    def safe_numerical(self, n):
        # These are reconstruction/math operations, not an allowlist of whole
        # modules. File and code-loading entry points are handled explicitly.
        return (n.startswith(("torch._utils._rebuild_", "torch._tensor._rebuild_", "torch._tensor._rebuild_from_type",
                              "numpy.core.multiarray.", "numpy._core.multiarray.", "numpy.random._pickle.",
                              "numpy.ma.core._mareconstruct", "scipy.sparse._", "sklearn."))
                or n in {"numpy.log1p", "numpy.expm1", "numpy.exp", "numpy.log", "numpy.sqrt", "numpy.sin", "numpy.cos",
                         "numpy.add", "numpy.multiply", "numpy.zeros", "numpy.ones", "numpy.empty", "numpy.arange",
                         "numpy.core.multiarray.scalar", "numpy._core.multiarray.scalar", "torch.device",
                         "numpy.random.bit_generator.__pyx_unpickle_SeedSequence",
                         "_codecs.latin_1_encode", "_codecs.latin_1_decode"})

    def logging_config(self, config):
        if not isinstance(config, dict): raise LoadStopped("invalid logging configuration")
        if config.get("version") != 1: raise LoadStopped("invalid logging configuration version")
        if config.get("incremental"): return
        def factory(value):
            if isinstance(value, str):
                if value.startswith("ext://"): value = value[6:]
                return self.call(Ref("pkgutil.resolve_name"), (value,))
            return value
        for group in ("formatters", "filters", "handlers"):
            entries = config.get(group, {})
            if not isinstance(entries, dict): continue
            for item in entries.values():
                if not isinstance(item, dict): continue
                target = item.get("()")
                if target is None and group == "handlers": target = item.get("class")
                if target is not None:
                    kwargs = {k: factory(v) if isinstance(v, str) and v.startswith("ext://") else v
                              for k, v in item.items() if k not in ("()", "class", ".", "level", "formatter", "filters")}
                    self.call(factory(target), (), kwargs)
