"""A non-executing interpreter for the pickle virtual machine.

Values here are data or symbolic references. No global named by a pickle is
imported, resolved with Python getattr, instantiated, or called. Calls are
interpreted by explicit summaries. Mutable symbolic objects preserve memo
aliases, including partials whose target changes during BUILD.
"""
from __future__ import annotations

import _compat_pickle
import ast
import base64
import binascii
import builtins
import functools
import operator
import os
import types
import codecs
import io
import math
import pickletools
from dataclasses import dataclass, field

from modelgate.context import AnalysisError, Context
from modelgate.policy import CALL_EFFECTS, NESTED_LOADERS, DATA_CALLS, PURE_MODULES
from modelgate.report import Finding


@dataclass(frozen=True)
class Ref:
    name: str


@dataclass(frozen=True)
class Module:
    name: str


@dataclass(frozen=True)
class Unknown:
    reason: str = "unknown value"


@dataclass(eq=False)
class Obj:
    cls: object
    args: tuple = ()
    attrs: dict = field(default_factory=dict)
    state: object = None
    value: object = None
    position: int = 0


@dataclass(eq=False)
class Bound:
    function: object
    receiver: object


@dataclass(frozen=True)
class Namespace:
    owner: object
    globals: bool = False


UNKNOWN = Unknown()
MARK = object()
TORCH_MAGIC = 0x1950A86A20F9469CFC6C
ALIASES = {
    "posix": "os", "nt": "os", "_pickle": "pickle", "_io": "io",
    "_operator": "operator", "_functools": "functools", "_socket": "socket",
    "posixpath": "os.path", "ntpath": "os.path", "genericpath": "os.path",
    "numpy._core": "numpy.core", "_codecs": "codecs",
}


def canonical_module(module):
    if module.startswith("numpy._core."):
        module = "numpy.core." + module[len("numpy._core."):]
    return ALIASES.get(module, module)


def canonical_name(name):
    for alias, target in ALIASES.items():
        if name.startswith(alias + "."):
            name = target + name[len(alias):]
            break
    if name == "socket.SocketType" or name.startswith("socket.SocketType."):
        name = "socket.socket" + name[len("socket.SocketType"):]
    return name


def global_ref(module, name, protocol=4):
    if not isinstance(module, str) or not isinstance(name, str):
        raise AnalysisError("GLOBAL needs literal module and name strings")
    if protocol < 3:
        module, name = _compat_pickle.NAME_MAPPING.get((module, name),
                         (_compat_pickle.IMPORT_MAPPING.get(module, module), name))
    module = canonical_module(module)
    aliases = {("socket", "SocketType"): ("socket", "socket"),
               ("builtins", "file"): ("builtins", "open")}
    module, name = aliases.get((module, name), (module, name))
    return Ref(canonical_name(module + "." + name))


def refname(value):
    return value.name if isinstance(value, Ref) else "<symbolic callable>"


def is_bytes(value):
    return isinstance(value, (bytes, bytearray))


def seq(value):
    if isinstance(value, (tuple, list)):
        return tuple(value)
    raise AnalysisError("call arguments are not a known tuple")


class Interpreter:
    def __init__(self, location, context=None, depth=0):
        self.location = location
        self.context = context or Context()
        self.depth = depth
        if depth > 32:
            raise AnalysisError("nested deserialization limit exceeded")
        self.findings = []
        self.memo = {}
        self.protocol = 0
        self.where = location
        self.operation = "call"
        self.call_depth = 0

    def finding(self, kind, detail):
        if len(self.findings) < 1000:
            self.findings.append(Finding(kind, self.where, detail))

    def effect(self, kind, name):
        self.finding(kind, f"{self.operation} {name}")

    def getitem(self, obj, key, default=UNKNOWN):
        if isinstance(obj, Namespace):
            if obj.globals and isinstance(key, str):
                if key == "__builtins__": return Namespace(Module("builtins"))
                if key in ("os", "sys", "builtins", "pickle", "subprocess", "io", "socket", "importlib", "functools", "operator"):
                    return Module(key)
            if isinstance(obj.owner, Module) and obj.owner.name == "sys.modules" and isinstance(key, str):
                return Module(canonical_module(key))
            return self.attr(obj.owner, key, default)
        if isinstance(obj, Obj):
            if refname(obj.cls) == "numpy.lib.npyio.NpzFile" and isinstance(key, str):
                name = key if key in obj.value else key + ".npy"
                if name not in obj.value: return default
                data = obj.value[name]
                if not data.startswith(b"\x93NUMPY"): return data
                from modelgate.dispatch import inspect_loaded
                findings, value, _ = inspect_loaded(data, self.where + "!" + name, self.context, self.depth + 1, "numpy-array", obj.state)
                self.findings.extend(findings)
                return value
            if isinstance(obj.value, (dict, tuple, list, bytes, str)):
                return self.getitem(obj.value, key, default)
            hook = self.special(obj, "__getitem__")
            if hook is not None:
                return self.call(hook, (key,))
            return default
        if isinstance(obj, (tuple, list, dict, str, bytes, bytearray)):
            try:
                return obj[key]
            except (KeyError, IndexError, TypeError):
                return default
        return default

    def special(self, obj, name):
        if isinstance(obj, Obj) and isinstance(obj.cls, Obj):
            fn = obj.cls.attrs.get(name)
            if fn is not None:
                # Functions stored as class descriptors bind; built-in functions
                # (eval, system, etc.) do not.
                return Bound(fn, obj) if isinstance(fn, Obj) and refname(fn.cls) == "types.FunctionType" else fn
        return None

    def attr(self, obj, name, default=UNKNOWN):
        if not isinstance(name, str):
            return Unknown("nonliteral attribute name")
        if name == "__dict__":
            if isinstance(obj, Obj):
                return obj.attrs
            return Namespace(obj)
        if isinstance(obj, Module):
            if name == "__builtins__": return Namespace(Module("builtins"))
            trusted = {"builtins": builtins, "os": os, "io": io, "functools": functools, "operator": operator, "types": types}
            if obj.name in trusted and name not in vars(trusted[obj.name]): return default
            if obj.name == "sys" and name == "modules":
                return Namespace(Module("sys.modules"))
            # Attribute paths can cross module boundaries; names stay symbolic.
            return Ref(canonical_name(canonical_module(obj.name) + "." + name))
        if isinstance(obj, Ref):
            if name == "__globals__":
                return Namespace(Module(obj.name.rsplit(".", 1)[0]), globals=True)
            if name == "__builtins__": return Namespace(Module("builtins"))
            if name == "__call__":
                return obj
            return Ref(canonical_name(obj.name + "." + name))
        if isinstance(obj, Obj):
            cn = refname(obj.cls)
            if cn == "functools.partial":
                if name == "func": return obj.value
                if name == "args": return obj.args
                if name == "keywords": return obj.state or {}
                if name == "__call__": return obj
            if name in obj.attrs:
                return obj.attrs[name]
            if cn == "types.FunctionType" and name == "__globals__":
                return obj.args[1] if len(obj.args) > 1 else {}
            if name == "__class__":
                return obj.cls
            hook = self.special(obj, name)
            if hook is not None:
                return hook
            fallback = self.special(obj, "__getattr__")
            if fallback is not None:
                return self.call(fallback, (name,))
            return Bound(Ref(cn + "." + name), obj)
        if isinstance(obj, (str, bytes, bytearray, tuple, list, dict, set, frozenset, int, float)):
            return Bound(Ref("builtins." + type(obj).__name__ + "." + name), obj)
        return default

    def new(self, cls, args=(), kwargs=None):
        # Per policy, NEWOBJ/NEWOBJ_EX and empty INST/OBJ are allocation only.
        obj = Obj(cls, seq(args))
        if refname(cls) == "functools.partial":
            obj.value = args[0] if args else UNKNOWN
            obj.args = tuple(args[1:])
            obj.state = dict(kwargs or {})
        return obj

    def build(self, obj, state):
        if not isinstance(obj, Obj):
            return
        name = refname(obj.cls)
        if name == "functools.partial":
            if isinstance(state, (tuple, list)) and len(state) == 4:
                obj.value, args, kw, attrs = state
                obj.args = seq(args)
                obj.state = dict(kw or {})
                if isinstance(attrs, dict): obj.attrs.update(attrs)
            else:
                raise AnalysisError("invalid partial state")
            return
        if name in ("io.BytesIO", "io.StringIO"):
            if isinstance(state, tuple) and state:
                obj.value = state[0]
                obj.position = state[1] if len(state) > 1 else 0
                if len(state) > 2 and isinstance(state[2], dict): obj.attrs.update(state[2])
            return
        hook = obj.attrs.get("__setstate__") or self.special(obj, "__setstate__")
        if hook is not None:
            self.call(hook, (state,))
            return
        obj.state = state
        if isinstance(state, dict):
            obj.attrs.update(state)
        elif isinstance(state, tuple) and len(state) == 2:
            for attrs in state:
                if isinstance(attrs, dict): obj.attrs.update(attrs)
        # Common data reducers' __setstate__ only restore data. Dangerous
        # custom state hooks with known semantics are handled explicitly.
        if name in ("numpy.core.multiarray._reconstruct", "numpy.ndarray"):
            if isinstance(state, tuple) and len(state) >= 5:
                obj.value = state[4]
        if name.startswith(("dill.", "cloudpickle.")) and "function" in name.lower():
            self.effect("code-execution", name + ".__setstate__")

    def call(self, fn, args=(), kwargs=None):
        self.context.tick()
        self.call_depth += 1
        try:
            if self.call_depth > 80:
                raise AnalysisError("cyclic or excessively deep symbolic call")
            return self._call(fn, seq(args), dict(kwargs or {}))
        finally:
            self.call_depth -= 1

    def _call(self, fn, args, kwargs):
        if isinstance(fn, Bound):
            return self.call(fn.function, (fn.receiver,) + args, kwargs)
        if isinstance(fn, Obj):
            name = refname(fn.cls)
            if name == "functools.partial":
                merged = dict(fn.state or {})
                merged.update(kwargs)
                return self.call(fn.value, fn.args + args, merged)
            if name in ("operator.attrgetter", "operator.itemgetter"):
                if not args: return UNKNOWN
                result = []
                for key in fn.args:
                    value = args[0]
                    if name == "operator.attrgetter" and isinstance(key, str):
                        for part in key.split("."): value = self.attr(value, part)
                    else:
                        value = self.getitem(value, key)
                    result.append(value)
                return result[0] if len(result) == 1 else tuple(result)
            if name == "operator.methodcaller":
                if not args or not fn.args: return UNKNOWN
                return self.call(self.attr(args[0], fn.args[0]), fn.args[1:], fn.state)
            if name == "types.FunctionType":
                self.effect("code-execution", "serialized Python function")
                return UNKNOWN
            if name == "types.MethodType" and len(fn.args) >= 2:
                return self.call(fn.args[0], (fn.args[1],) + args, kwargs)
            hook = self.special(fn, "__call__")
            if hook is not None: return self.call(hook, args, kwargs)
            if name == "builtins.type":
                obj = Obj(fn, args)
                init = self.special(obj, "__init__")
                if init is not None: self.call(init, args, kwargs)
                return obj
            raise AnalysisError("no callable summary for instance of " + name)
        if not isinstance(fn, Ref):
            raise AnalysisError("cannot resolve called value: " + (fn.reason if isinstance(fn, Unknown) else type(fn).__name__))
        name = canonical_name(fn.name)
        if name.endswith(".__call__"):
            base = name[:-len(".__call__")]
            if args and isinstance(args[0], (Ref, Obj, Bound)):
                return self.call(args[0], args[1:], kwargs)
            name = base
        if name.endswith(".__init__"):
            base = name[:-len(".__init__")]
            constructor = base.rsplit(".", 1)[-1]
            if constructor[:1].isupper() and (base in CALL_EFFECTS or base.startswith(("ftplib.", "smtplib.", "http.client.", "poplib.", "imaplib.", "telnetlib."))):
                return self.call(Ref(base), args[1:], kwargs)
        if name.endswith(".__new__"):
            cls = args[0] if args else Ref(name[:-8])
            return self.new(cls, args[1:], kwargs)
        if name == "builtins.getattr":
            return self.attr(args[0], args[1], args[2] if len(args) > 2 else UNKNOWN) if len(args) >= 2 else UNKNOWN
        if name in ("builtins.setattr", "builtins.delattr"):
            if len(args) >= 2 and isinstance(args[0], Obj) and isinstance(args[1], str):
                if len(args) > 2: args[0].attrs[args[1]] = args[2]
                else: args[0].attrs.pop(args[1], None)
            return None
        if name in ("builtins.__import__", "importlib.import_module", "importlib.__import__"):
            if args and isinstance(args[0], str):
                module = args[0]
                if name != "importlib.import_module" and not (args[3] if len(args) > 3 else kwargs.get("fromlist")):
                    module = module.split(".")[0]
                return Module(canonical_module(module))
            return Unknown("dynamic module import")
        if name in ("operator.attrgetter", "operator.itemgetter", "operator.methodcaller"):
            return Obj(Ref(name), args, state=kwargs)
        if name == "functools.partial":
            target = args[0] if args else UNKNOWN
            bound_args = args[1:]
            if isinstance(target, Obj) and refname(target.cls) == "functools.partial" and not target.attrs:
                bound_args = target.args + bound_args
                merged = dict(target.state or {})
                merged.update(kwargs)
                kwargs, target = merged, target.value
            return Obj(fn, bound_args, state=kwargs, value=target)
        if name == "functools.partial.__setstate__":
            if len(args) == 2: self.build(args[0], args[1])
            return None
        if name in ("operator.call", "builtins.apply"):
            if not args: return UNKNOWN
            return self.call(args[0], seq(args[1]) if name == "builtins.apply" and len(args) > 1 else args[1:], kwargs)
        if name == "numpy.lib.npyio.NpzFile.__getitem__":
            return self.getitem(args[0], args[1]) if len(args) > 1 else UNKNOWN
        if name in ("operator.getitem", "builtins.dict.__getitem__", "builtins.list.__getitem__", "builtins.tuple.__getitem__", "builtins.dict.get"):
            return self.getitem(args[0], args[1], args[2] if len(args) > 2 else UNKNOWN) if len(args) >= 2 else UNKNOWN
        if name in ("operator.setitem", "builtins.dict.__setitem__", "builtins.list.__setitem__"):
            if len(args) >= 3 and isinstance(args[0], (dict, list)): args[0][args[1]] = args[2]
            return None
        if name == "builtins.vars":
            return self.attr(args[0], "__dict__") if args else Namespace(Module("builtins"))
        if name in ("types.ModuleType",):
            return Module(args[0]) if args and isinstance(args[0], str) else UNKNOWN
        if name == "builtins.type":
            if len(args) == 3 and isinstance(args[2], dict):
                return Obj(fn, args, attrs=dict(args[2]))
            if len(args) == 1:
                return args[0].cls if isinstance(args[0], Obj) else Ref("builtins." + type(args[0]).__name__)
        if name in ("types.MethodType",):
            return Obj(fn, args)
        if name in ("types.FunctionType", "types.LambdaType"):
            self.effect("code-execution", name + " (materializes Python bytecode)")
            return Obj(Ref("types.FunctionType"), args)
        if name in ("marshal.loads", "marshal.load", "types.CodeType"):
            if name == "marshal.load" and args and not self.in_memory(args[0]):
                self.effect("file-access", name)
            return Obj(Ref("types.CodeType"), args)
        if name in ("torch._tensor._rebuild_from_type", "torch._tensor._rebuild_from_type_v2"):
            if len(args) < 3: return UNKNOWN
            value = self.call(args[0], args[2])
            if len(args) > 3:
                obj = Obj(args[1], value=value)
                self.build(obj, args[3])
                return obj
            return value
        if name in ("numpy.fromfunction", "numpy.apply_along_axis", "numpy.apply_over_axes"):
            return self.call(args[0], (UNKNOWN,)) if args else UNKNOWN
        if name == "copyreg._reconstructor":
            if len(args) < 3: return UNKNOWN
            obj = self.new(args[0])
            if isinstance(args[1], Ref) and args[1].name != "builtins.object":
                self.call(Ref(args[1].name + ".__init__"), (obj, args[2]))
            return obj
        if name in ("copyreg.__newobj__", "copyreg.__newobj_ex__"):
            return self.new(args[0], args[1:]) if args else UNKNOWN
        if name in NESTED_LOADERS:
            return self.nested(name, NESTED_LOADERS[name], args, kwargs)
        if name in ("pickle.Unpickler", "pickle._Unpickler"):
            return Obj(fn, args, value=args[0] if args else UNKNOWN)
        if name in ("io.BufferedReader", "io.BufferedRandom", "io.BufferedWriter", "io.TextIOWrapper"):
            if args and self.in_memory(args[0]):
                return Obj(Ref("io.BytesIO"), value=self.stream_bytes(args[0]))
            return Obj(fn, args)
        if name in ("io.BytesIO", "io.StringIO"):
            return Obj(fn, args, value=args[0] if args else (b"" if name == "io.BytesIO" else ""))
        if name.startswith(("io.BytesIO.", "io.StringIO.")):
            return self.stream_method(name, args)
        if name in ("pathlib.Path", "pathlib.PosixPath", "pathlib.WindowsPath", "pathlib.PurePath", "pathlib.PurePosixPath", "pathlib.PureWindowsPath"):
            parts = [self.path_value(a) for a in args]
            value = "/".join(parts) if all(isinstance(a, str) for a in parts) else UNKNOWN
            return Obj(fn, args, value=value)
        if name == "os.fspath": return self.path_value(args[0]) if args else UNKNOWN
        if name in ("ftplib.FTP", "ftplib.FTP_TLS", "smtplib.SMTP", "smtplib.SMTP_SSL", "smtplib.LMTP", "poplib.POP3", "poplib.POP3_SSL", "imaplib.IMAP4", "imaplib.IMAP4_SSL", "telnetlib.Telnet"):
            host = args[0] if args else kwargs.get("host", "")
            if host or name.startswith(("poplib.", "imaplib.")):
                self.effect("network", name + "(host)")
            return Obj(fn, args, attrs=dict(kwargs))
        if name in ("socket.socket", "http.client.HTTPConnection", "http.client.HTTPSConnection", "xmlrpc.client.ServerProxy", "xmlrpc.client.Server", "requests.Session", "requests.sessions.Session", "ftplib.FTP", "code.InteractiveInterpreter", "code.InteractiveConsole", "bdb.Bdb", "pdb.Pdb"):
            return Obj(fn, args, attrs=dict(kwargs))
        if name.startswith("xmlrpc.client.ServerProxy.") and not name.endswith((".__init__", ".__new__")):
            self.effect("network", name)
            return UNKNOWN
        if name.startswith("xmlrpc.client.Server."):
            self.effect("network", name)
            return UNKNOWN
        if name in ("logging.config.dictConfig", "logging.config.DictConfigurator.configure"):
            config = args[0] if args else UNKNOWN
            if isinstance(config, Obj): config = config.value
            return self.logging_config(config)
        if name == "logging.config.DictConfigurator":
            return Obj(fn, args, value=args[0] if args else UNKNOWN)
        if name == "logging.config.fileConfig":
            self.effect("file-access" if args and isinstance(args[0], (str, bytes)) else "code-execution", name)
            return None
        if name in ("gzip.GzipFile", "gzip.open", "bz2.BZ2File", "bz2.open", "lzma.LZMAFile", "lzma.open"):
            source = kwargs.get("fileobj", args[3] if name == "gzip.GzipFile" and len(args) > 3 else args[0] if args else UNKNOWN)
            if self.in_memory(source):
                mode = args[1] if len(args) > 1 else kwargs.get("mode", "rb")
                if isinstance(mode, str) and any(flag in mode for flag in "wax"):
                    return Obj(Ref("io.BytesIO"), value=b"")
                from modelgate.dispatch import decompress
                raw = self.stream_bytes(source)
                return Obj(Ref("io.BytesIO"), value=decompress(raw, self.context))
            self.file_effect(name, (source,))
            return Obj(fn, args)
        if name in ("gzip.decompress", "bz2.decompress", "lzma.decompress", "zlib.decompress"):
            if args and is_bytes(args[0]):
                from modelgate.dispatch import decompress
                return decompress(bytes(args[0]), self.context, codec=name.split(".")[0])
            return UNKNOWN
        if name.startswith(("cloudpickle.", "dill.")) and name.rsplit(".", 1)[-1] in (
            "_make_function", "_create_function", "_create_code", "_function_setstate", "_make_skeleton_class", "_create_type"):
            self.effect("code-execution", name)
            return Obj(Ref("types.FunctionType"), args)
        if name.startswith(("importlib.machinery.", "importlib._bootstrap_external.")) and name.endswith((".exec_module", ".load_module", ".get_code")):
            self.effect("code-execution", name)
            return UNKNOWN
        if name in ("sqlite3.connect", "sqlite3.dbapi2.connect"):
            path = args[0] if args else kwargs.get("database")
            if path == ":memory:" or (kwargs.get("uri") and isinstance(path, str) and (path.startswith("file::memory:") or "mode=memory" in path)):
                return Obj(Ref("sqlite3.Connection"), args)
        if name in ("zipfile.ZipFile", "zipfile.PyZipFile", "tarfile.open", "tarfile.TarFile.open", "h5py.File"):
            source = kwargs.get("fileobj", args[0] if args else kwargs.get("file", kwargs.get("name", UNKNOWN)))
            if self.in_memory(source):
                return Obj(Ref("zipfile.ZipFile" if name.startswith("zipfile.") else "tarfile.TarFile" if name.startswith("tarfile.") else name), value=self.stream_bytes(source))
            self.file_effect(name, (source,))
            return Obj(fn, args)
        if name == "zipfile.ZipFile.read" and len(args) > 1 and isinstance(args[0], Obj) and is_bytes(args[0].value) and isinstance(args[1], str):
            import zipfile
            with zipfile.ZipFile(io.BytesIO(args[0].value)) as archive:
                entry = archive.getinfo(args[1])
                self.context.account(entry.file_size)
                return archive.read(entry)
        if name == "tarfile.TarFile.extractfile" and len(args) > 1 and isinstance(args[0], Obj) and is_bytes(args[0].value) and isinstance(args[1], str):
            import tarfile
            with tarfile.open(fileobj=io.BytesIO(args[0].value), mode="r:*") as archive:
                entry = archive.getmember(args[1])
                self.context.account(entry.size)
                stream = archive.extractfile(entry)
                if stream is None: return None
                with stream: data = stream.read(self.context.max_bytes + 1)
                return Obj(Ref("io.BytesIO"), value=data)
        if name in ("ctypes.CDLL", "ctypes.PyDLL", "ctypes.WinDLL", "ctypes.OleDLL", "ctypes.cdll.LoadLibrary", "ctypes.pydll.LoadLibrary"):
            if args and args[0] is not None: self.effect("code-execution", name)
            return Obj(fn, args)
        if name == "threading.Thread":
            target = args[1] if len(args) > 1 else kwargs.get("target")
            parameters = args[3] if len(args) > 3 else kwargs.get("args", ())
            keywords = args[4] if len(args) > 4 else kwargs.get("kwargs", {})
            return Obj(fn, (target, parameters, keywords))
        if name == "threading.Timer":
            target = args[1] if len(args) > 1 else kwargs.get("function")
            parameters = args[2] if len(args) > 2 else kwargs.get("args", ())
            keywords = args[3] if len(args) > 3 else kwargs.get("kwargs", {})
            return Obj(fn, (target, parameters, keywords))
        if name in ("threading.Thread.start", "threading.Thread.run", "threading.Timer.start", "threading.Timer.run"):
            if args and isinstance(args[0], Obj) and len(args[0].args) == 3:
                target, params, kw = args[0].args
                if target is not None: self.call(target, params or (), kw or {})
            return None
        if name in ("concurrent.futures.ThreadPoolExecutor.submit", "concurrent.futures.thread.ThreadPoolExecutor.submit"):
            return self.call(args[1], args[2:], kwargs) if len(args) > 1 else UNKNOWN
        if name in ("timeit.Timer", "timeit.timeit", "timeit.repeat"):
            stmt = args[0] if args else kwargs.get("stmt")
            setup = args[1] if len(args) > 1 else kwargs.get("setup")
            if isinstance(stmt, str) or isinstance(setup, str):
                self.effect("code-execution", name + " (source compilation)")
            timer = Obj(Ref("timeit.Timer"), attrs={"stmt": stmt, "setup": setup})
            if name != "timeit.Timer": self.run_timer(timer)
            return timer if name == "timeit.Timer" else UNKNOWN
        if name in ("timeit.Timer.timeit", "timeit.Timer.repeat", "timeit.Timer.autorange"):
            if args and isinstance(args[0], Obj): self.run_timer(args[0])
            return UNKNOWN
        if name in ("bdb.Bdb.runcall", "pdb.Pdb.runcall", "pdb.runcall"):
            offset = 0 if name == "pdb.runcall" else 1
            return self.call(args[offset], args[offset + 1:], kwargs) if len(args) > offset else UNKNOWN
        effect = CALL_EFFECTS.get(name)
        if effect:
            if effect == "file-access": self.file_effect(name, args)
            else: self.effect(effect, name)
            return Obj(fn, args)
        result = self.pure_call(name, args, kwargs)
        if result is not NotImplemented:
            return result
        # Retain data constructors' types so later methods still undergo
        # capability analysis. A missing call summary is not evidence of safety.
        module, _, symbol = name.rpartition(".")
        scientific_class = (module.startswith(("sklearn.", "scipy.", "pandas.", "numpy.random."))
                            and symbol[:1].isupper() and not symbol.startswith("_"))
        exception_class = module == "builtins" and (symbol.endswith(("Error", "Warning", "Exception")) or symbol in ("Exception", "BaseException", "StopIteration"))
        data_method = (module in ("socket.socket", "http.client.HTTPConnection", "http.client.HTTPSConnection", "ftplib.FTP", "ftplib.FTP_TLS", "smtplib.SMTP", "smtplib.SMTP_SSL")
                       and symbol in ("close", "set_debuglevel", "set_pasv", "settimeout", "gettimeout", "setsockopt", "getsockopt"))
        if name in DATA_CALLS or module in PURE_MODULES or scientific_class or exception_class or data_method:
            return Obj(Ref(name), args, attrs=dict(kwargs))
        raise AnalysisError("no static call summary for " + name)

    def run_timer(self, timer):
        for value in (timer.attrs.get("setup"), timer.attrs.get("stmt")):
            if isinstance(value, (Ref, Obj, Bound)):
                self.call(value, ())

    def path_value(self, value):
        if isinstance(value, Obj) and refname(value.cls).startswith("pathlib."):
            return value.value
        return value

    def file_effect(self, name, args):
        paths = args[:2] if name.rsplit(".", 1)[-1] in ("copy", "copyfile", "copy2", "copytree", "move", "rename", "replace", "link", "symlink") else args[:1]
        if paths and all(self.in_memory(v) or self.context.internal(self.path_value(v)) for v in paths):
            return
        self.effect("file-access", name)

    def in_memory(self, value):
        return isinstance(value, Obj) and refname(value.cls) in ("io.BytesIO", "io.StringIO")

    def stream_bytes(self, value):
        if self.in_memory(value):
            raw = value.value
            pos = value.position
            if isinstance(raw, str): raw = raw.encode("latin1")
            if is_bytes(raw) and isinstance(pos, int): return bytes(raw[pos:])
        raise AnalysisError("nested loader source is not a known in-memory buffer")

    def stream_method(self, name, args):
        if not args or not self.in_memory(args[0]): return UNKNOWN
        obj = args[0]
        method = name.rsplit(".", 1)[-1]
        if method in ("getvalue", "getbuffer"): return obj.value
        if method == "tell": return obj.position
        if method == "seek":
            offset = args[1] if len(args) > 1 else 0
            whence = args[2] if len(args) > 2 else 0
            if not isinstance(offset, int): return UNKNOWN
            base = 0 if whence == 0 else obj.position if whence == 1 else len(obj.value)
            obj.position = base + offset
            return base + offset
        if method in ("read", "read1", "readall", "readline"):
            raw = obj.value
            pos = obj.position
            if not isinstance(raw, (str, bytes, bytearray)) or not isinstance(pos, int): return UNKNOWN
            size = args[1] if len(args) > 1 and isinstance(args[1], int) else -1
            end = len(raw) if size < 0 else pos + size
            if method == "readline":
                newline = raw.find(b"\n" if is_bytes(raw) else "\n", pos)
                if newline >= 0: end = min(end, newline + 1)
            result = raw[pos:end]
            obj.position = pos + len(result)
            return result
        if method == "write" and len(args) > 1:
            raw, value = obj.value, args[1]
            pos = obj.position
            if type(raw) is type(value) and isinstance(raw, (str, bytes)) and isinstance(pos, int):
                obj.value = raw[:pos] + value + raw[pos + len(value):]
                obj.position = pos + len(value)
                return len(value)
        if method == "__setstate__" and len(args) > 1: self.build(obj, args[1])
        return None

    def nested(self, name, mode, args, kwargs):
        source = args[0] if args else kwargs.get("file", kwargs.get("f", kwargs.get("file_obj", UNKNOWN)))
        if mode == "pickle-unpickler" and isinstance(source, Obj): source = source.value
        if mode in ("pickle-bytes", "torch-bytes"):
            if not is_bytes(source):
                raise AnalysisError(name + " needs known embedded bytes")
            raw = bytes(source)
        elif self.in_memory(source):
            raw = self.stream_bytes(source)
        else:
            # Bytes are filenames for these APIs, not serialized content.
            self.file_effect(name, (source,))
            return UNKNOWN
        location = self.where + " -> " + name
        if mode in ("pickle-bytes", "pickle-file", "pickle-unpickler"):
            child = Interpreter(location, self.context, self.depth + 1)
            reader = io.BytesIO(raw)
            value = child.read_one(reader, 0)
            self.findings.extend(child.findings)
            if self.in_memory(source): source.position += reader.tell()
            return value
        from modelgate.dispatch import inspect_loaded
        options = dict(kwargs)
        if mode == "numpy" and len(args) > 2: options["allow_pickle"] = args[2]
        if mode == "numpy-array" and len(args) > 1: options["allow_pickle"] = args[1]
        if name.startswith("pandas."): mode = "pandas"
        if name.startswith("joblib."): mode = "joblib"
        findings, value, consumed = inspect_loaded(raw, location, self.context, self.depth + 1, mode, options)
        self.findings.extend(findings)
        if self.in_memory(source): source.position += consumed
        return value

    def logging_config(self, config):
        if not isinstance(config, dict):
            raise AnalysisError("logging configuration is not a known mapping")
        if config.get("incremental"):
            return None
        for section in ("formatters", "filters", "handlers"):
            entries = config.get(section, {})
            if not isinstance(entries, dict): continue
            for entry in entries.values():
                if not isinstance(entry, dict): continue
                factory = entry.get("()")
                if factory is None and section == "handlers": factory = entry.get("class")
                if isinstance(factory, str):
                    factory = Ref(factory.removeprefix("ext://"))
                if factory is not None:
                    kw = {k: v for k, v in entry.items() if k not in ("()", "class", "level", "formatter", "filters", ".")}
                    self.call(factory, (), kw)
        return None

    def pure_call(self, name, args, kwargs):
        """Evaluate a small, explicit set of data transformations only."""
        if name in ("codecs.encode", "codecs.decode") and args:
            encoding = args[1] if len(args) > 1 else "utf-8"
            allowed = {"latin1", "latin-1", "iso8859-1", "utf8", "utf-8", "ascii", "unicode_escape", "raw_unicode_escape", "base64", "base64_codec", "hex", "hex_codec", "rot_13", "rot13", "bz2", "bz2_codec", "zlib", "zlib_codec"}
            if isinstance(encoding, str) and encoding.lower() in allowed and isinstance(args[0], (str, bytes)):
                if encoding in ("bz2", "bz2_codec", "zlib", "zlib_codec") and name.endswith("decode"):
                    from modelgate.dispatch import decompress
                    return decompress(args[0], self.context, codec=encoding.split("_")[0])
                try:
                    fn = codecs.encode if name.endswith("encode") else codecs.decode
                    return fn(args[0], encoding)
                except (ValueError, TypeError, UnicodeError): return UNKNOWN
            return UNKNOWN
        if name in ("base64.b64decode", "base64.standard_b64decode", "base64.urlsafe_b64decode", "base64.b32decode", "base64.b16decode", "base64.b85decode", "base64.a85decode", "binascii.a2b_base64", "binascii.unhexlify", "binascii.a2b_hex") and args:
            if isinstance(args[0], (str, bytes)):
                module, method = name.split(".")
                safe = {"base64": base64, "binascii": binascii}[module]
                try: return getattr(safe, method)(args[0])
                except (ValueError, TypeError): return UNKNOWN
            return UNKNOWN
        if name in ("builtins.bytes", "builtins.bytearray"):
            if not args: return b""
            value = args[0]
            if isinstance(value, str) and len(args) > 1 and isinstance(args[1], str):
                if args[1].lower() in ("latin1", "latin-1", "utf-8", "utf8", "ascii"): return value.encode(args[1])
            if is_bytes(value): return bytes(value)
            if isinstance(value, (tuple, list)) and all(isinstance(x, int) and 0 <= x < 256 for x in value): return bytes(value)
            if isinstance(value, int) and 0 <= value <= self.context.max_bytes: return bytes(value)
            return UNKNOWN
        if name == "builtins.str":
            if not args: return ""
            if isinstance(args[0], (str, int, float, bool, type(None))): return str(args[0])
            if is_bytes(args[0]) and len(args) > 1 and isinstance(args[1], str) and args[1].lower() in ("latin1", "latin-1", "utf8", "utf-8", "ascii"):
                return bytes(args[0]).decode(args[1])
            return UNKNOWN
        if name in ("builtins.str.encode", "builtins.bytes.decode", "builtins.bytearray.decode") and args:
            encoding = args[1] if len(args) > 1 else "utf-8"
            if encoding in ("latin1", "latin-1", "utf8", "utf-8", "ascii", "unicode_escape", "raw_unicode_escape") and isinstance(args[0], (str, bytes, bytearray)):
                try:
                    return args[0].encode(encoding) if name.endswith("encode") else args[0].decode(encoding)
                except (UnicodeError, AttributeError): return UNKNOWN
            return UNKNOWN
        if name in ("builtins.bytes.fromhex", "builtins.bytearray.fromhex") and args and isinstance(args[0], str):
            return bytes.fromhex(args[0])
        if name in ("builtins.str.join", "builtins.bytes.join") and len(args) > 1:
            if isinstance(args[0], (str, bytes)) and isinstance(args[1], (list, tuple)) and all(type(x) is type(args[0]) for x in args[1]):
                return args[0].join(args[1])
            return UNKNOWN
        if name in ("operator.add", "operator.concat") and len(args) == 2:
            if type(args[0]) is type(args[1]) and isinstance(args[0], (str, bytes, int, float, tuple, list)): return args[0] + args[1]
            return UNKNOWN
        if name == "operator.mul" and len(args) == 2:
            if isinstance(args[0], int) and isinstance(args[1], int): return args[0] * args[1]
            return UNKNOWN
        if name in ("builtins.tuple", "builtins.list", "builtins.set", "builtins.frozenset"):
            value = args[0] if args else ()
            if isinstance(value, Obj) and refname(value.cls) in ("builtins.map", "builtins.filter", "itertools.starmap", "builtins.iterator"):
                value = self.materialize(value)
            if isinstance(value, (tuple, list, set, frozenset, dict, str, bytes)):
                return {"builtins.tuple": tuple, "builtins.list": list, "builtins.set": set, "builtins.frozenset": frozenset}[name](value)
            return UNKNOWN
        if name in ("builtins.dict", "collections.OrderedDict", "collections.defaultdict", "collections.Counter"):
            values = args[1:] if name == "collections.defaultdict" else args
            result = {}
            if values and isinstance(values[0], (dict, list, tuple)):
                try: result.update(values[0])
                except (TypeError, ValueError): pass
            result.update(kwargs)
            return result
        if name in ("builtins.dict.update", "builtins.list.append", "builtins.list.extend", "builtins.set.add"):
            if len(args) > 1:
                obj = args[0]
                if name.endswith("update") and isinstance(obj, dict) and isinstance(args[1], dict): obj.update(args[1])
                elif name.endswith("append") and isinstance(obj, list): obj.append(args[1])
                elif name.endswith("extend") and isinstance(obj, list) and isinstance(args[1], (tuple, list)): obj.extend(args[1])
                elif name.endswith("add") and isinstance(obj, set): obj.add(args[1])
            return None
        if name.startswith(("builtins.str.", "builtins.bytes.", "builtins.bytearray.")) and args:
            method = name.rsplit(".", 1)[-1]
            safe_methods = {"lower", "upper", "casefold", "title", "capitalize", "swapcase", "strip", "lstrip", "rstrip", "removeprefix", "removesuffix", "replace", "split", "rsplit", "splitlines", "partition", "rpartition", "startswith", "endswith", "find", "rfind", "index", "rindex", "count", "ljust", "rjust", "center", "zfill", "expandtabs", "format", "format_map", "hex", "isascii", "isalpha", "isalnum", "isdecimal", "isdigit", "isnumeric", "isspace", "islower", "isupper", "istitle", "isidentifier", "isprintable", "__getitem__", "__add__", "__mul__", "__mod__"}
            if method in safe_methods and type(args[0]) in (str, bytes, bytearray):
                def literal(value, seen=None):
                    if type(value) in (str, bytes, int, float, bool, type(None), slice): return True
                    if type(value) not in (dict, tuple, list): return False
                    seen = set() if seen is None else seen
                    if id(value) in seen: return False
                    seen.add(id(value))
                    vals = list(value.items()) if type(value) is dict else value
                    return all(literal(v, seen) for v in vals)
                if all(literal(a) for a in args[1:]) and all(literal(v) for v in kwargs.values()):
                    try: return getattr(args[0], method)(*args[1:], **kwargs)
                    except (ValueError, TypeError, IndexError, KeyError, AttributeError): return UNKNOWN
            return UNKNOWN
        if name.startswith("os.path."):
            method = name.rsplit(".", 1)[-1]
            if method in ("join", "split", "splitext", "splitdrive", "basename", "dirname", "normcase", "normpath", "isabs", "commonprefix", "commonpath", "relpath", "abspath", "expandvars", "expanduser"):
                if all(isinstance(a, (str, bytes)) for a in args):
                    try: return getattr(os.path, method)(*args)
                    except (TypeError, ValueError): return UNKNOWN
                return UNKNOWN
        if name in ("builtins.dict.keys", "builtins.dict.values", "builtins.dict.items") and args and isinstance(args[0], dict):
            method = name.rsplit(".", 1)[-1]
            return list(getattr(args[0], method)())
        if name in ("builtins.list.__init__", "builtins.dict.__init__", "builtins.set.__init__", "builtins.object.__init__", "numpy.ndarray.__init__"):
            return None
        if name == "builtins.len" and args and isinstance(args[0], (str, bytes, tuple, list, dict, set)): return len(args[0])
        if name == "builtins.slice": return slice(*args)
        if name == "builtins.chr" and args and isinstance(args[0], int): return chr(args[0])
        if name == "builtins.ord" and args and isinstance(args[0], (str, bytes)): return ord(args[0])
        if name in ("builtins.int", "builtins.float", "builtins.bool") and args and isinstance(args[0], (str, bytes, int, float, bool)):
            try: return {"builtins.int": int, "builtins.float": float, "builtins.bool": bool}[name](args[0])
            except (ValueError, TypeError): return UNKNOWN
        if name in ("builtins.map", "builtins.filter", "itertools.starmap"):
            return Obj(Ref(name), args)
        if name == "functools.reduce" and len(args) >= 2 and isinstance(args[1], (list, tuple)):
            values = list(args[1])
            if len(args) > 2: result = args[2]
            elif values: result = values.pop(0)
            else: return UNKNOWN
            for value in values: result = self.call(args[0], (result, value))
            return result
        if name == "builtins.sorted" and args and isinstance(args[0], (tuple, list, dict, set)):
            if "key" in kwargs and kwargs["key"] is not None:
                for v in args[0]: self.call(kwargs["key"], (v,))
            return list(args[0])
        if name == "builtins.next" and args and isinstance(args[0], Obj):
            values = self.materialize(args[0], limit=1)
            return values[0] if values else UNKNOWN
        if name in ("builtins.min", "builtins.max"):
            values = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list, set, dict)) else args
            if "key" in kwargs and kwargs["key"] is not None:
                for value in values: self.call(kwargs["key"], (value,))
            return values[0] if isinstance(values, (tuple, list)) and values else UNKNOWN
        if name in ("heapq.nlargest", "heapq.nsmallest") and len(args) > 1:
            key = args[2] if len(args) > 2 else kwargs.get("key")
            if key is not None and isinstance(args[1], (tuple, list, set, dict)):
                for value in args[1]: self.call(key, (value,))
            return list(args[1]) if isinstance(args[1], (tuple, list)) else UNKNOWN
        if name in ("builtins.iter",):
            return Obj(Ref("builtins.iterator"), args)
        if name == "ast.literal_eval" and args and isinstance(args[0], (str, bytes)):
            try: return ast.literal_eval(args[0])
            except (ValueError, SyntaxError): return UNKNOWN
        return NotImplemented

    def materialize(self, obj, limit=100000):
        name = refname(obj.cls)
        if name == "builtins.iterator":
            if len(obj.args) == 2:
                return [self.call(obj.args[0], ())]  # iter(callable, sentinel)
            if obj.args and isinstance(obj.args[0], (tuple, list, str, bytes, dict)):
                return list(obj.args[0])[:limit]
            return []
        if len(obj.args) < 2: return []
        fn, *iterables = obj.args
        if not all(isinstance(v, (list, tuple, str, bytes, dict)) for v in iterables):
            raise AnalysisError("unknown lazy iterable")
        result = []
        for i, values in enumerate(zip(*iterables)):
            if i >= limit: break
            if name == "itertools.starmap": value = self.call(fn, seq(values[0]))
            elif name == "builtins.filter":
                self.call(fn, values) if fn is not None else None
                value = values[0]
            else: value = self.call(fn, values)
            result.append(value)
        return result

    def read_one(self, reader, stream_index=0):
        stack = []
        marks = []
        self.protocol = 0
        def pop_mark():
            if not marks: raise AnalysisError("missing pickle MARK")
            idx = marks.pop()
            result = stack[idx:]
            del stack[idx:]
            return result
        for op, arg, pos in pickletools.genops(reader):
            self.context.tick()
            name = op.name
            self.where = f"{self.location}:stream[{stream_index}]@{pos}"
            self.operation = name
            if name in ("PROTO", "FRAME"):
                if name == "PROTO":
                    if arg > 5: raise AnalysisError("unknown pickle protocol")
                    self.protocol = arg
            elif name == "MARK": marks.append(len(stack))
            elif name == "STOP":
                if not stack: raise AnalysisError("empty pickle result")
                return stack.pop()
            elif name == "POP":
                if marks and len(stack) == marks[-1]: marks.pop()
                else: stack.pop()
            elif name == "POP_MARK": pop_mark()
            elif name == "DUP": stack.append(stack[-1])
            elif name in ("INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT", "STRING", "BINSTRING", "SHORT_BINSTRING", "UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8"):
                stack.append(bytes(arg) if isinstance(arg, bytearray) else arg)
            elif name == "NONE": stack.append(None)
            elif name == "NEWTRUE": stack.append(True)
            elif name == "NEWFALSE": stack.append(False)
            elif name == "EMPTY_LIST": stack.append([])
            elif name == "EMPTY_TUPLE": stack.append(())
            elif name == "EMPTY_DICT": stack.append({})
            elif name == "EMPTY_SET": stack.append(set())
            elif name in ("TUPLE", "LIST", "DICT", "FROZENSET"):
                items = pop_mark()
                if name == "DICT":
                    if len(items) % 2: raise AnalysisError("odd DICT item count")
                    value = dict(zip(items[::2], items[1::2]))
                else: value = {"TUPLE": tuple, "LIST": list, "FROZENSET": frozenset}[name](items)
                stack.append(value)
            elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
                count = int(name[-1])
                items = stack[-count:]
                del stack[-count:]
                stack.append(tuple(items))
            elif name in ("APPEND", "APPENDS", "ADDITEMS"):
                values = [stack.pop()] if name == "APPEND" else pop_mark()
                target = stack[-1]
                if isinstance(target, list): target.extend(values)
                elif isinstance(target, set): target.update(values)
                else:
                    for value in values: self.call(self.attr(target, "add" if name == "ADDITEMS" else "append"), (value,))
            elif name in ("SETITEM", "SETITEMS"):
                if name == "SETITEM":
                    value, key = stack.pop(), stack.pop()
                    values = [key, value]
                else: values = pop_mark()
                target = stack[-1]
                if len(values) % 2: raise AnalysisError("odd SETITEMS item count")
                for key, value in zip(values[::2], values[1::2]):
                    if isinstance(target, dict): target[key] = value
                    else: self.call(self.attr(target, "__setitem__"), (key, value))
            elif name in ("PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE"):
                self.memo[len(self.memo) if name == "MEMOIZE" else int(arg)] = stack[-1]
            elif name in ("GET", "BINGET", "LONG_BINGET"):
                if int(arg) not in self.memo: raise AnalysisError("unbound pickle memo reference")
                stack.append(self.memo[int(arg)])
            elif name == "GLOBAL":
                module, symbol = arg.split(" ", 1)
                stack.append(global_ref(module, symbol, self.protocol))
            elif name == "STACK_GLOBAL":
                symbol, module = stack.pop(), stack.pop()
                stack.append(global_ref(module, symbol, self.protocol))
            elif name == "REDUCE":
                args, fn = stack.pop(), stack.pop()
                stack.append(self.call(fn, args))
            elif name in ("NEWOBJ", "NEWOBJ_EX"):
                kwargs = stack.pop() if name == "NEWOBJ_EX" else {}
                args, cls = stack.pop(), stack.pop()
                stack.append(self.new(cls, args, kwargs))
            elif name in ("INST", "OBJ"):
                items = pop_mark()
                if name == "INST":
                    module, symbol = arg.split(" ", 1)
                    cls, args = global_ref(module, symbol, self.protocol), tuple(items)
                else: cls, args = items[0], tuple(items[1:])
                stack.append(self.call(cls, args) if args else self.new(cls))
            elif name == "BUILD":
                state = stack.pop()
                self.build(stack[-1], state)
                self.joblib_array(stack, reader)
            elif name in ("PERSID", "BINPERSID"):
                pid = stack.pop() if name == "BINPERSID" else arg
                stack.append(Obj(Ref("persistent-storage"), value=pid))
            elif name in ("EXT1", "EXT2", "EXT4"):
                # Extension IDs depend on process-specific registration, not on
                # bytes in the artifact. Never consult or import that registry.
                stack.append(Unknown(f"pickle extension {arg}"))
            elif name == "NEXT_BUFFER": stack.append(Unknown("out-of-band buffer"))
            elif name == "READONLY_BUFFER": pass
            else: raise AnalysisError("unknown pickle opcode " + name)
        raise AnalysisError("pickle has no STOP")

    def joblib_array(self, stack, reader):
        obj = stack[-1]
        if not isinstance(obj, Obj) or refname(obj.cls) != "joblib.numpy_pickle.NumpyArrayWrapper": return
        dtype = obj.attrs.get("dtype")
        from modelgate.numpy_scan import dtype_info
        if isinstance(dtype, Obj) and refname(dtype.cls) == "numpy.dtype":
            desc = dtype.args[0] if dtype.args else None
            size, hasobject = dtype_info(desc)
            if isinstance(dtype.state, tuple) and len(dtype.state) > 5 and isinstance(dtype.state[5], int) and dtype.state[5] > 0:
                size = dtype.state[5]
            if isinstance(dtype.state, tuple) and len(dtype.state) > 4 and isinstance(dtype.state[4], dict):
                hasobject = hasobject or any(isinstance(v, tuple) and isinstance(v[0], Obj) and v[0].args and dtype_info(v[0].args[0])[1] for v in dtype.state[4].values())
        else: raise AnalysisError("unknown joblib array dtype")
        if hasobject:
            child = Interpreter(self.where + " -> joblib object array", self.context, self.depth + 1)
            stack[-1] = child.read_one(reader)
            self.findings.extend(child.findings)
        else:
            shape = obj.attrs.get("shape", ())
            if not isinstance(shape, tuple) or not all(isinstance(n, int) and n >= 0 for n in shape): raise AnalysisError("unknown joblib array shape")
            count = math.prod(shape)
            alignment = obj.attrs.get("numpy_array_alignment_bytes")
            if alignment is not None:
                pad = reader.read(1)
                if not pad: raise AnalysisError("truncated joblib alignment")
                reader.seek(pad[0], 1)
            end = reader.tell() + count * size
            if end > len(reader.getbuffer()): raise AnalysisError("truncated joblib array data")
            reader.seek(end)
            stack[-1] = Obj(Ref("numpy.ndarray"), value=UNKNOWN)


def scan_pickle(data: bytes, location: str, context=None, depth=0, sequence=True) -> list[Finding]:
    machine = Interpreter(location, context, depth)
    reader = io.BytesIO(data)
    count = 0
    legacy_torch = False
    try:
        while reader.tell() < len(data):
            machine.memo = {}
            value = machine.read_one(reader, count)
            if count == 0: legacy_torch = value == TORCH_MAGIC if isinstance(value, int) else False
            count += 1
            # Raw storages are bytes, not another pickle. Recognize the legacy
            # format by its magic value, never by the filename.
            if not sequence or (legacy_torch and count == 5): break
        if count == 0: raise AnalysisError("empty pickle")
    except Exception as exc:
        machine.findings.append(Finding("unsupported", machine.where, f"cannot analyse pickle: {type(exc).__name__}: {exc}"))
    return machine.findings
