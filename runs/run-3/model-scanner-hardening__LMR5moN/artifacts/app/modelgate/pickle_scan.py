"""A non-executing interpreter for pickle's stack machine.

Values from artifacts are literals or the symbolic objects below. GLOBAL does
not import anything, and REDUCE/BUILD never dispatch to a Python object from
an artifact. The only real operations performed are explicitly selected,
side-effect-free operations on literal data and container parsers.
"""
from __future__ import annotations

import _compat_pickle
import base64
import binascii
import io
import math
import operator
import pickletools
from dataclasses import dataclass, field

from modelgate.limits import Budget, ScanLimit
from modelgate.policy import CALL_CAPABILITIES, NESTED_LOADERS
from modelgate.report import Finding


@dataclass(eq=False)
class Unknown:
    description: str = "unknown value"


UNKNOWN = Unknown()
MARK = object()
TORCH_MAGIC = 0x1950A86A20F9469CFC6C


@dataclass(frozen=True)
class Ref:
    name: str


@dataclass(frozen=True)
class Module:
    name: str


@dataclass(eq=False)
class Instance:
    cls: object
    args: tuple = ()
    attrs: dict = field(default_factory=dict)
    state: object = None


@dataclass(eq=False)
class Partial:
    fn: object
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)
    has_dict: bool = False


@dataclass(eq=False)
class Bound:
    obj: object
    name: str


@dataclass(eq=False)
class Helper:
    kind: str
    args: tuple
    kwargs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Buffer:
    data: object = b""
    pos: int = 0


@dataclass(eq=False)
class Npz:
    data: bytes
    allow_pickle: bool


@dataclass(eq=False)
class Code:
    data: object


@dataclass(eq=False)
class Function:
    code: object
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class DynamicClass:
    name: str
    bases: tuple
    attrs: dict


@dataclass(eq=False)
class Lazy:
    kind: str
    fn: object
    sources: tuple
    pos: int = 0


ALIASES = {
    "posix": "os", "nt": "os", "_pickle": "pickle", "_io": "io",
    "_operator": "operator", "_functools": "functools", "_socket": "socket",
    "numpy.core": "numpy._core", "pathlib.PosixPath": "pathlib.Path",
    "pathlib.WindowsPath": "pathlib.Path", "socket.SocketType": "socket.socket",
    "socket._socketobject": "socket.socket",
}
REF_ALIASES = {
    "pickle._loads": "pickle.loads", "pickle._load": "pickle.load",
    "io._IOBase": "io.IOBase", "io._RawIOBase": "io.RawIOBase",
    "io._BufferedIOBase": "io.BufferedIOBase", "io._TextIOBase": "io.TextIOBase",
    "types.LambdaType": "types.FunctionType", "types.BuiltinFunctionType": "types.BuiltinFunctionType",
    "numpy._core.multiarray.scalar": "numpy._core.multiarray.scalar",
}


def canonical(name):
    for old, new in ALIASES.items():
        if name == old or name.startswith(old + "."):
            name = new + name[len(old):]
            break
    return REF_ALIASES.get(name, name)


def global_ref(module, name, protocol=4):
    if not isinstance(module, str) or not isinstance(name, str):
        return UNKNOWN
    if protocol < 3:
        if (module, name) in _compat_pickle.NAME_MAPPING:
            module, name = _compat_pickle.NAME_MAPPING[(module, name)]
        else:
            module = _compat_pickle.IMPORT_MAPPING.get(module, module)
    return Ref(canonical(module + "." + name))


def _class_name(value):
    if isinstance(value, Ref):
        return value.name
    if isinstance(value, Instance):
        return _class_name(value.cls)
    return ""


def _literal(value, depth=0):
    """Whether a value is exclusively inert builtin data (no symbolic object)."""
    if depth > 30:
        return False
    if type(value) in (str, bytes, bytearray, int, float, bool, complex, type(None)):
        return True
    if type(value) in (tuple, list, set, frozenset):
        return all(_literal(x, depth + 1) for x in value)
    if type(value) is dict:
        return all(_literal(k, depth + 1) and _literal(v, depth + 1) for k, v in value.items())
    return False


class Machine:
    def __init__(self, location, budget, depth=0):
        self.location = location
        self.budget = budget
        self.depth = depth
        self.findings = []
        self.memo = {}
        self.stack = []
        self.protocol = 0
        self.where = location
        self.operation = "call"
        self.call_depth = 0
        self.file = None

    def finding(self, kind, detail):
        if len(self.findings) < 200:
            self.findings.append(Finding(kind, self.where, detail))

    def unsupported(self, detail):
        self.finding("unsupported", detail)
        return UNKNOWN

    def get_attr(self, obj, name, default=UNKNOWN):
        if not isinstance(name, str):
            return UNKNOWN
        if isinstance(obj, Module):
            if name == "__dict__":
                return Helper("namespace", (obj,))
            return Ref(canonical(obj.name + "." + name))
        if isinstance(obj, Ref):
            if name == "__call__":
                return obj
            if name == "__module__":
                return obj.name.rsplit(".", 1)[0]
            if name in ("__name__", "__qualname__"):
                return obj.name.rsplit(".", 1)[-1]
            if name == "__dict__":
                return Helper("namespace", (obj,))
            return Ref(canonical(obj.name + "." + name))
        if isinstance(obj, Partial):
            if name == "__dict__":
                obj.has_dict = True
                return obj.attrs
            if name in ("func", "args", "keywords"):
                return {"func": obj.fn, "args": obj.args, "keywords": obj.kwargs}[name]
            if name in obj.attrs:
                return obj.attrs[name]
        if isinstance(obj, (Instance, Function, DynamicClass)):
            if name == "__dict__":
                return obj.attrs
            if name in obj.attrs:
                return obj.attrs[name]
            if isinstance(obj, Instance) and isinstance(obj.cls, DynamicClass) and name in obj.cls.attrs:
                value = obj.cls.attrs[name]
                return Helper("method", (value, obj)) if isinstance(value, Function) else value
        if name == "__class__":
            if isinstance(obj, Instance):
                return obj.cls
            if isinstance(obj, Partial):
                return Ref("functools.partial")
            if isinstance(obj, Buffer):
                return Ref("io.BytesIO" if isinstance(obj.data, bytes) else "io.StringIO")
            if type(obj) in (str, bytes, list, tuple, dict, set, frozenset, int):
                return Ref("builtins." + type(obj).__name__)
        if name == "__call__" and isinstance(obj, (Partial, Helper, Function)):
            return obj
        if isinstance(obj, Unknown):
            return default
        return Bound(obj, name)

    def get_item(self, obj, key):
        if isinstance(obj, Npz) and isinstance(key, str):
            import zipfile
            from modelgate.numpy_scan import scan_array
            with zipfile.ZipFile(io.BytesIO(obj.data)) as zf:
                entry = key if key in zf.namelist() else key + ".npy"
                if entry not in zf.namelist():
                    return UNKNOWN
                with zf.open(entry) as source:
                    data = self.budget.read(source)
            if data.startswith(b"\x93NUMPY"):
                findings, value = scan_array(data, self.where + "!" + entry, self.budget,
                                             self.depth + 1, obj.allow_pickle)
                self.findings.extend(findings)
                return value
            return data
        if isinstance(obj, Helper) and obj.kind == "namespace":
            return self.get_attr(obj.args[0], key)
        if isinstance(obj, Instance):
            items = obj.attrs.get("__items__", obj.attrs)
            if isinstance(items, dict):
                if key not in items and _class_name(obj) == "collections.defaultdict" and obj.args and obj.args[0] is not None:
                    items[key] = self.call(obj.args[0])
                return items.get(key, UNKNOWN)
            if isinstance(items, (list, tuple)) and isinstance(key, (int, slice)):
                return items[key]
        if type(obj) in (dict, list, tuple, str, bytes, bytearray) and type(key) in (str, int, bytes, tuple, slice):
            try:
                return obj[key]
            except (KeyError, IndexError, TypeError):
                return UNKNOWN
        return UNKNOWN

    def call(self, fn, args=(), kwargs=None):
        self.budget.check()
        self.call_depth += 1
        try:
            if self.call_depth > 80:
                raise ScanLimit("symbolic call recursion limit exceeded")
            if not isinstance(args, (tuple, list)):
                return self.unsupported("call arguments cannot be resolved statically")
            return self._call(fn, tuple(args), kwargs or {})
        finally:
            self.call_depth -= 1

    def _call(self, fn, args, kwargs):
        if isinstance(fn, Partial):
            return self.call(fn.fn, fn.args + args, {**fn.kwargs, **kwargs})
        if isinstance(fn, Function):
            self.finding("code-execution", f"{self.operation} serialized Python function")
            return UNKNOWN
        if isinstance(fn, Bound):
            return self.bound_call(fn.obj, fn.name, args, kwargs)
        if isinstance(fn, Helper):
            if fn.kind == "attrgetter" and args:
                values = []
                for path in fn.args:
                    obj = args[0]
                    if not isinstance(path, str):
                        return UNKNOWN
                    for name in path.split("."):
                        obj = self.get_attr(obj, name)
                    values.append(obj)
                return values[0] if len(values) == 1 else tuple(values)
            if fn.kind == "itemgetter" and args:
                values = [self.get_item(args[0], key) for key in fn.args]
                return values[0] if len(values) == 1 else tuple(values)
            if fn.kind == "methodcaller" and args:
                return self.call(self.get_attr(args[0], fn.args[0]), fn.args[1:], fn.kwargs)
            if fn.kind == "method":
                return self.call(fn.args[0], (fn.args[1],) + args, kwargs)
            if fn.kind == "remote-method":
                self.finding("network", f"{self.operation} XML-RPC method")
                return UNKNOWN
            return self.unsupported("unresolved symbolic helper call")
        if isinstance(fn, DynamicClass):
            obj = Instance(fn)
            for name in ("__new__", "__init__"):
                if name in fn.attrs:
                    self.call(fn.attrs[name], (obj,) + args, kwargs)
            return obj
        if isinstance(fn, Instance):
            return self.call(self.get_attr(fn, "__call__"), args, kwargs)
        if not isinstance(fn, Ref):
            return self.unsupported("executed callable cannot be resolved statically")
        name = canonical(fn.name)
        if name.endswith(".__init__") and args:
            return self.call(Ref(name[:-9]), args[1:], kwargs)
        if name.endswith(".__setstate__") and len(args) >= 2:
            self.build(args[0], args[1])
            return None
        # Calling a type's __new__ bypasses __init__, just like NEWOBJ.
        if name.endswith(".__new__"):
            cls = args[0] if args else Ref(name[:-8])
            return self.new_object(cls, args[1:], kwargs)
        if name in ("builtins.getattr", "builtins.object.__getattribute__", "builtins.type.__getattribute__", "operator.getattr"):
            return self.get_attr(args[0], args[1], args[2] if len(args) > 2 else UNKNOWN) if len(args) >= 2 else UNKNOWN
        if name in ("builtins.__import__", "importlib.import_module", "importlib.__import__"):
            if not args or not isinstance(args[0], str):
                return UNKNOWN
            module = args[0]
            if name != "importlib.import_module":
                fromlist = args[3] if len(args) > 3 else kwargs.get("fromlist", ())
                if not fromlist:
                    module = module.split(".")[0]
            return Module(canonical(module))
        if name in ("functools.partial", "functools.partialmethod"):
            return self.make_partial(args, kwargs)
        if name in ("operator.attrgetter", "operator.itemgetter", "operator.methodcaller"):
            return Helper(name.split(".")[-1], args, kwargs)
        if name in ("operator.call", "builtins.apply"):
            if not args:
                return UNKNOWN
            if name == "builtins.apply":
                return self.call(args[0], args[1] if len(args) > 1 else (), args[2] if len(args) > 2 else {})
            return self.call(args[0], args[1:], kwargs)
        if name in ("operator.getitem", "operator.__getitem__") and len(args) >= 2:
            return self.get_item(*args[:2])
        if name in ("builtins.setattr", "builtins.delattr") and len(args) >= 2:
            obj, key = args[:2]
            if isinstance(obj, (Instance, Partial, Function, DynamicClass)) and isinstance(key, str):
                if isinstance(obj, Partial):
                    obj.has_dict = True
                if len(args) == 3:
                    obj.attrs[key] = args[2]
                else:
                    obj.attrs.pop(key, None)
            return None
        if name == "builtins.vars" and args:
            return self.get_attr(args[0], "__dict__")
        if name == "builtins.type":
            if len(args) == 1:
                return self.get_attr(args[0], "__class__")
            if len(args) == 3 and isinstance(args[2], dict):
                return DynamicClass(str(args[0]), args[1], args[2])
        if name in ("types.FunctionType", "dill._dill._create_function", "dill.dill._create_function", "cloudpickle.cloudpickle._make_function", "cloudpickle.cloudpickle._make_skel_func"):
            self.finding("code-execution", f"{self.operation} {name}: materializes serialized Python bytecode")
            return Function(args[0] if args else UNKNOWN)
        if name in ("types.CodeType", "dill._dill._create_code"):
            return Code(args)
        if name in ("marshal.loads", "marshal.load"):
            if name.endswith(".load") and args and not isinstance(args[0], Buffer):
                self.finding("file-access", f"{self.operation} marshal.load reads an external stream")
            return Code(args[0] if args else UNKNOWN)
        if name == "types.MethodType" and len(args) >= 2:
            return Helper("method", args[:2])
        if name in ("io.BytesIO", "io.StringIO"):
            return Buffer(args[0] if args else (b"" if name.endswith("BytesIO") else ""))
        if name in ("gzip.GzipFile", "bz2.BZ2File", "lzma.LZMAFile", "gzip.open", "bz2.open", "lzma.open"):
            source = kwargs.get("fileobj", args[0] if args else kwargs.get("filename", UNKNOWN))
            if isinstance(source, Buffer) and isinstance(source.data, bytes):
                from modelgate.dispatch import decompress
                return Buffer(decompress(source.data[source.pos:], self.budget))
        if name in ("io.BufferedReader", "io.BufferedRandom", "io.BufferedWriter", "io.TextIOWrapper") and args and isinstance(args[0], Buffer):
            return args[0]
        if name in ("urllib.request.urlopen", "urllib.request.OpenerDirector.open", "urllib.request.URLopener.open", "urllib.request.FancyURLopener.open"):
            index = 1 if name != "urllib.request.urlopen" else 0
            url = args[index] if len(args) > index else kwargs.get("url", kwargs.get("fullurl", UNKNOWN))
            if isinstance(url, Instance) and _class_name(url) == "urllib.request.Request" and url.args:
                url = url.args[0]
            if isinstance(url, str) and url.startswith("data:"):
                from urllib.parse import unquote_to_bytes
                metadata, content = url[5:].split(",", 1)
                data = unquote_to_bytes(content)
                return Buffer(base64.b64decode(data) if metadata.endswith(";base64") else data)
            if isinstance(url, str) and url.startswith("file:"):
                self.finding("file-access", f"{self.operation} {name} reads a file URL")
                return UNKNOWN
        if name in NESTED_LOADERS:
            return self.nested_load(name, args, kwargs)
        if name in ("pickle.Unpickler", "pickle._Unpickler"):
            return Instance(Ref(name), args)
        if name in ("logging.config.dictConfig", "logging.config.DictConfigurator.configure"):
            config = args[0] if args else kwargs.get("config", {})
            if isinstance(config, Instance):
                config = config.args[0] if config.args else UNKNOWN
            self.logging_config(config)
            return None
        if name == "logging.config.fileConfig":
            self.finding("file-access", f"{self.operation} {name} reads configuration")
            return None
        if name in ("timeit.timeit", "timeit.repeat", "timeit.Timer", "timeit.Timer.timeit", "timeit.Timer.repeat", "timeit.Timer.autorange"):
            return self.timeit_call(name, args, kwargs)
        if name in ("ftplib.FTP", "ftplib.FTP_TLS", "smtplib.SMTP", "smtplib.SMTP_SSL", "telnetlib.Telnet"):
            host = args[0] if args else kwargs.get("host", "")
            if host:
                self.finding("network", f"{self.operation} {name} connects to host")
            return Instance(Ref(name), args)
        if name in ("poplib.POP3", "poplib.POP3_SSL", "imaplib.IMAP4", "imaplib.IMAP4_SSL", "imaplib.IMAP4_stream"):
            self.finding("process" if name.endswith("_stream") else "network", f"{self.operation} {name}")
            return Instance(Ref(name), args)
        if name in ("multiprocessing.Pool", "multiprocessing.pool.Pool", "multiprocessing.pool.ThreadPool"):
            if not name.endswith("ThreadPool"):
                self.finding("process", f"{self.operation} {name}")
            return Instance(Ref(name), args)
        if name in ("ctypes.CDLL", "ctypes.PyDLL", "ctypes.WinDLL", "ctypes.OleDLL", "ctypes.cdll.LoadLibrary", "ctypes.pydll.LoadLibrary"):
            if args and args[0] is not None:
                self.finding("code-execution", f"{self.operation} {name} loads native code")
            return Instance(Ref("ctypes.CDLL"), args)
        # Constructors for these clients allocate local state only. Their
        # connecting/starting methods are handled when actually invoked.
        if name in ("socket.socket", "http.client.HTTPConnection", "http.client.HTTPSConnection",
                    "xmlrpc.client.ServerProxy", "xmlrpc.client.Server", "threading.Thread",
                    "multiprocessing.Process", "multiprocessing.context.Process",
                    "concurrent.futures.ProcessPoolExecutor", "concurrent.futures.ThreadPoolExecutor"):
            return Instance(Ref(name), args, dict(kwargs))
        if name.startswith("pathlib.") and name.split(".")[-1] in ("Path", "PurePath", "PurePosixPath", "PureWindowsPath"):
            return Instance(Ref(name), args)
        if name in ("copyreg._reconstructor", "copyreg.__newobj__", "copyreg.__newobj_ex__"):
            if not args:
                return UNKNOWN
            ctorargs = args[1:] if name.endswith("__newobj__") else ()
            if name.endswith("__newobj_ex__") and len(args) >= 3:
                ctorargs, kwargs = args[1:3]
            return self.new_object(args[0], ctorargs, kwargs)
        # Data operations can return callable values, so propagate them instead
        # of treating all harmless calls as opaque.
        value, handled = self.data_call(name, args, kwargs)
        if handled:
            return value
        kind = CALL_CAPABILITIES.get(name)
        if kind:
            # SQLite's in-memory database does not access a file.
            if name == "sqlite3.connect" and args and args[0] == ":memory:":
                return Instance(Ref("sqlite3.Connection"), args)
            # Delayed FileHandlers do not open until the first emit.
            if name.startswith("logging.") and name.endswith("FileHandler") and kwargs.get("delay"):
                return Instance(Ref(name), args)
            self.finding(kind, f"{self.operation} {name}")
            return Instance(Ref(name), args)
        if self.extra_capability(name, args, kwargs):
            return Instance(Ref(name), args)
        # Ordinary model constructors/rebuilders are opaque data. Their state
        # and attributes remain available for subsequent symbolic operations.
        return Instance(Ref(name), args, dict(kwargs))

    def data_call(self, name, args, kwargs):
        if name in ("builtins.list", "builtins.tuple", "builtins.set", "builtins.frozenset"):
            seq = self.materialize(args[0]) if args else []
            if isinstance(seq, Unknown):
                return UNKNOWN, True
            return {"builtins.list": list, "builtins.tuple": tuple,
                    "builtins.set": set, "builtins.frozenset": frozenset}[name](seq), True
        if name in ("builtins.dict", "collections.OrderedDict"):
            if not args:
                return dict(kwargs), True
            if isinstance(args[0], dict):
                return {**args[0], **kwargs}, True
            values = self.materialize(args[0])
            if not isinstance(values, Unknown):
                try:
                    return {**dict(values), **kwargs}, True
                except (ValueError, TypeError):
                    pass
            return Instance(Ref(name), args), True
        if name == "collections.defaultdict":
            obj = Instance(Ref(name), args)
            obj.attrs["__items__"] = dict(args[1]) if len(args) > 1 and isinstance(args[1], dict) else {}
            return obj, True
        if name in ("builtins.map", "builtins.filter", "itertools.starmap", "itertools.filterfalse") and args:
            return Lazy(name.rsplit(".", 1)[-1], args[0], args[1:]), True
        if name == "builtins.iter" and args:
            if len(args) > 1:
                return Lazy("sentinel", args[0], (args[1],)), True
            return Lazy("iter", None, (args[0],)), True
        if name == "builtins.next" and args:
            values = self.materialize(args[0], count=1)
            return values[0] if isinstance(values, list) and values else args[1] if len(args) > 1 else UNKNOWN, True
        if name == "builtins.zip":
            return Lazy("zip", None, args), True
        if name == "builtins.enumerate" and args:
            return Lazy("enumerate", None, args), True
        if name in ("builtins.sorted", "builtins.min", "builtins.max") and args:
            values = self.materialize(args[0]) if len(args) == 1 or name.endswith("sorted") else list(args)
            if isinstance(values, Unknown):
                return UNKNOWN, True
            key = kwargs.get("key")
            if key is not None:
                for value in values:
                    self.call(key, (value,))
                return values if name.endswith("sorted") else UNKNOWN, True
            if _literal(values):
                try:
                    return {"builtins.sorted": sorted, "builtins.min": min, "builtins.max": max}[name](values), True
                except (TypeError, ValueError):
                    return UNKNOWN, True
            return values if name.endswith("sorted") else UNKNOWN, True
        if name in ("functools.reduce", "_functools.reduce") and len(args) >= 2:
            seq = self.materialize(args[1])
            if isinstance(seq, Unknown):
                return UNKNOWN, True
            if len(args) > 2:
                acc = args[2]
            elif seq:
                acc, seq = seq[0], seq[1:]
            else:
                return UNKNOWN, True
            for item in seq:
                acc = self.call(args[0], (acc, item))
            return acc, True
        if name in ("builtins.any", "builtins.all", "builtins.sum") and args:
            seq = self.materialize(args[0])
            if isinstance(seq, Unknown) or not _literal(seq):
                return UNKNOWN, True
            return {"builtins.any": any, "builtins.all": all, "builtins.sum": sum}[name](seq), True
        if name in ("builtins.bytes", "builtins.bytearray"):
            if args and isinstance(args[0], Buffer):
                return args[0].data, True
            if args and isinstance(args[0], int) and not 0 <= args[0] <= self.budget.max_blob:
                raise ScanLimit("oversized byte allocation")
            if _literal(args):
                return bytes(*args), True
            return UNKNOWN, True
        if name == "builtins.memoryview" and args:
            return args[0], True
        if name == "builtins.slice" and _literal(args):
            return slice(*args), True
        if name == "builtins.range" and _literal(args):
            value = range(*args)
            if len(value) > 100000:
                raise ScanLimit("oversized symbolic range")
            return list(value), True
        if name == "os.fspath" and args:
            value = args[0]
            if isinstance(value, Instance) and _class_name(value).startswith("pathlib."):
                return "/".join(str(x) for x in value.args), True
            return value, True
        if name == "builtins.len" and args and type(args[0]) in (str, bytes, list, tuple, dict, set, frozenset):
            return len(args[0]), True
        pure = {
            "builtins.str": str, "builtins.repr": repr, "builtins.int": int,
            "builtins.float": float, "builtins.bool": bool, "builtins.complex": complex,
            "builtins.chr": chr, "builtins.ord": ord, "builtins.abs": abs,
            "operator.add": operator.add, "operator.concat": operator.concat,
            "operator.sub": operator.sub, "operator.mod": operator.mod,
            "operator.truediv": operator.truediv, "operator.floordiv": operator.floordiv,
            "operator.neg": operator.neg, "operator.index": operator.index,
            "base64.b64decode": base64.b64decode, "base64.decodebytes": base64.decodebytes,
            "base64.standard_b64decode": base64.standard_b64decode,
            "base64.urlsafe_b64decode": base64.urlsafe_b64decode,
            "base64.b32decode": base64.b32decode, "base64.b16decode": base64.b16decode,
            "base64.b85decode": base64.b85decode, "base64.a85decode": base64.a85decode,
            "binascii.a2b_base64": binascii.a2b_base64, "binascii.unhexlify": binascii.unhexlify,
            "binascii.a2b_hex": binascii.unhexlify,
        }
        if name in pure and _literal(args) and _literal(kwargs):
            try:
                value = pure[name](*args, **kwargs)
                if isinstance(value, (bytes, str)) and len(value) > self.budget.max_blob:
                    raise ScanLimit("oversized decoded value")
                return value, True
            except (ValueError, TypeError, OverflowError, UnicodeError):
                return UNKNOWN, True
        if name in ("codecs.encode", "_codecs.encode", "codecs.decode", "_codecs.decode") and args:
            value = args[0]
            encoding = args[1] if len(args) > 1 else kwargs.get("encoding", "utf-8")
            if not isinstance(encoding, str):
                return UNKNOWN, True
            enc = encoding.lower().replace("_", "-")
            if enc in ("latin1", "latin-1", "iso-8859-1", "ascii", "utf8", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "raw-unicode-escape", "unicode-escape"):
                if name.endswith("encode") and isinstance(value, str):
                    return value.encode(encoding), True
                if name.endswith("decode") and isinstance(value, bytes):
                    return value.decode(encoding), True
            if enc in ("base64", "base64-codec", "hex", "hex-codec") and name.endswith("decode"):
                return (base64.b64decode(value) if enc.startswith("base64") else binascii.unhexlify(value)), True
            if enc in ("zlib", "zlib-codec", "bz2", "bz2-codec") and name.endswith("decode"):
                from modelgate.dispatch import decompress
                return decompress(value, self.budget), True
            return UNKNOWN, True
        if name in ("gzip.decompress", "bz2.decompress", "lzma.decompress", "zlib.decompress") and args and isinstance(args[0], bytes):
            from modelgate.dispatch import decompress
            return decompress(args[0], self.budget), True
        if name in ("json.loads", "ast.literal_eval") and args and isinstance(args[0], (str, bytes)):
            # No caller-supplied decoder hooks are executed by the scanner.
            if name == "json.loads":
                import json
                value = json.loads(args[0])
                for hook in ("object_hook", "object_pairs_hook", "parse_int", "parse_float", "parse_constant"):
                    if kwargs.get(hook) is not None:
                        self.call(kwargs[hook], (value,))
                return value, True
            import ast
            return ast.literal_eval(args[0]), True
        if name in ("builtins.str.encode", "builtins.bytes.decode", "builtins.bytes.fromhex", "builtins.bytearray.fromhex") and args:
            if name.endswith("fromhex") and isinstance(args[0], str):
                return bytes.fromhex(args[0]), True
            return self.bound_call(args[0], name.rsplit(".", 1)[-1], args[1:], kwargs), True
        if name.startswith(("builtins.dict.", "builtins.list.", "builtins.tuple.", "builtins.str.", "builtins.bytes.")) and args:
            return self.bound_call(args[0], name.rsplit(".", 1)[-1], args[1:], kwargs), True
        return None, False

    def materialize(self, obj, count=None):
        if type(obj) in (list, tuple, set, frozenset, dict, bytes, str):
            values = list(obj)
            return values if count is None else values[:count]
        if not isinstance(obj, Lazy):
            return UNKNOWN
        if obj.kind == "sentinel":
            # Reaching the iterator calls its callable even for an unknown
            # sentinel. Bound the loop; one call suffices to expose capability.
            return [self.call(obj.fn, ())]
        if obj.kind in ("enumerate", "iter", "filter", "filterfalse", "starmap"):
            sources = [self.materialize(obj.sources[0])]
        else:
            sources = [self.materialize(source) for source in obj.sources]
        if not sources or any(isinstance(s, Unknown) for s in sources):
            return UNKNOWN
        end = min(map(len, sources))
        if count is not None:
            end = min(end, obj.pos + count)
        values = []
        for i in range(obj.pos, end):
            self.budget.check()
            args = tuple(s[i] for s in sources)
            if obj.kind == "map":
                values.append(self.call(obj.fn, args))
            elif obj.kind == "starmap":
                values.append(self.call(obj.fn, args[0]))
            elif obj.kind in ("filter", "filterfalse"):
                keep = self.call(obj.fn, args) if obj.fn is not None else args[0]
                if bool(keep) == (obj.kind == "filter"):
                    values.append(args[0])
            elif obj.kind == "enumerate":
                values.append((i + (obj.sources[1] if len(obj.sources) > 1 else 0), args[0]))
            elif obj.kind == "zip":
                values.append(args)
            else:
                values.append(args[0])
        obj.pos = end
        return values

    def bound_call(self, obj, method, args, kwargs):
        if isinstance(obj, Npz):
            if method in ("get", "__getitem__") and args:
                return self.get_item(obj, args[0])
            return UNKNOWN
        if isinstance(obj, Buffer):
            if not isinstance(obj.data, (str, bytes)):
                return UNKNOWN
            if method in ("getvalue", "getbuffer"):
                return obj.data
            if method in ("read", "read1", "readline"):
                n = args[0] if args and isinstance(args[0], int) else -1
                end = len(obj.data) if n < 0 else obj.pos + n
                if method == "readline":
                    newline = b"\n" if isinstance(obj.data, bytes) else "\n"
                    found = obj.data.find(newline, obj.pos, end)
                    if found >= 0:
                        end = found + 1
                value = obj.data[obj.pos:end]
                obj.pos += len(value)
                return value
            if method == "seek" and args and isinstance(args[0], int):
                whence = args[1] if len(args) > 1 else 0
                obj.pos = args[0] + (obj.pos if whence == 1 else len(obj.data) if whence == 2 else 0)
                return obj.pos
            if method == "tell":
                return obj.pos
            if method == "write" and args and isinstance(args[0], type(obj.data)):
                value = args[0]
                obj.data = obj.data[:obj.pos] + value + obj.data[obj.pos + len(value):]
                obj.pos += len(value)
                return len(value)
            if method == "__setstate__" and args and isinstance(args[0], tuple):
                state = args[0]
                if len(state) >= 2:
                    obj.data, obj.pos = state[:2]
                return None
            return UNKNOWN
        if method == "__setstate__" and args:
            self.build(obj, args[0])
            return None
        if method == "__call__":
            if isinstance(obj, (Partial, Function, Helper)):
                return self.call(obj, args, kwargs)
            if isinstance(obj, Instance):
                name = _class_name(obj)
                if name:
                    return self.call(Ref(name + ".__call__"), (obj,) + args, kwargs)
                return self.unsupported("unknown callable instance")
        if method == "__getitem__" and args:
            return self.get_item(obj, args[0])
        if isinstance(obj, Helper) and obj.kind == "namespace":
            if method == "get" and args:
                return self.get_item(obj, args[0])
        if type(obj) is dict:
            if method in ("get", "pop") and args:
                if method == "pop":
                    return obj.pop(args[0], args[1] if len(args) > 1 else UNKNOWN)
                return obj.get(args[0], args[1] if len(args) > 1 else UNKNOWN)
            if method == "setdefault" and args:
                return obj.setdefault(args[0], args[1] if len(args) > 1 else None)
            if method == "__setitem__" and len(args) == 2:
                obj[args[0]] = args[1]
                return None
            if method in ("keys", "values", "items"):
                return list(getattr(obj, method)())
            if method in ("copy", "__copy__"):
                return dict(obj)
            if method == "update":
                if args and isinstance(args[0], dict):
                    obj.update(args[0])
                obj.update(kwargs)
                return None
        if type(obj) is list:
            if method == "append" and args:
                obj.append(args[0])
                return None
            if method == "extend" and args:
                values = self.materialize(args[0])
                if not isinstance(values, Unknown):
                    obj.extend(values)
                return None
            if method == "pop":
                return obj.pop(*args)
            if method == "__setitem__" and len(args) == 2:
                obj[args[0]] = args[1]
                return None
        if type(obj) in (str, bytes) and _literal(args) and _literal(kwargs):
            methods = {"encode", "decode", "join", "format", "replace", "strip", "lstrip", "rstrip", "split", "rsplit", "partition", "rpartition", "lower", "upper", "capitalize", "removeprefix", "removesuffix", "__add__", "__getitem__"}
            if method in methods and hasattr(type(obj), method):
                value = getattr(obj, method)(*args, **kwargs)
                if isinstance(value, (str, bytes)) and len(value) > self.budget.max_blob:
                    raise ScanLimit("oversized string operation")
                return value
        if isinstance(obj, Instance):
            name = _class_name(obj)
            if name in ("xmlrpc.client.ServerProxy", "xmlrpc.client.Server", "xmlrpc.client._Method"):
                if method not in ("__enter__", "__exit__"):
                    self.finding("network", f"{self.operation} {name}.{method}")
                return UNKNOWN
            if name.startswith("ctypes."):
                self.finding("code-execution", f"{self.operation} native function {method}")
                return UNKNOWN
            if name in ("pickle.Unpickler", "pickle._Unpickler") and method == "load":
                return self.nested_load(name + ".load", obj.args[:1], kwargs)
            if name == "timeit.Timer" and method in ("timeit", "repeat", "autorange"):
                return self.timeit_call("timeit.Timer." + method, (obj,) + args, kwargs)
            if name == "logging.config.DictConfigurator" and method == "configure":
                self.logging_config(obj.args[0] if obj.args else UNKNOWN)
                return None
            if method == "start" and name in ("multiprocessing.Process", "multiprocessing.context.Process", "multiprocessing.process.BaseProcess"):
                self.finding("process", f"{self.operation} {name}.start")
                return UNKNOWN
            if method == "start" and name == "threading.Thread":
                target = obj.attrs.get("target", obj.args[1] if len(obj.args) > 1 else None)
                if target is not None:
                    self.call(target, obj.attrs.get("args", ()), obj.attrs.get("kwargs", {}))
                return UNKNOWN
            if name.startswith("pathlib.") and method in ("__str__", "__fspath__"):
                return "/".join(str(x) for x in obj.args)
            if name and name not in ("builtins.str", "builtins.bytes"):
                # Prefix with the receiver to preserve class-specific method
                # semantics, including methods on objects allocated by NEWOBJ.
                return self.call(Ref(canonical(name + "." + method)), (obj,) + args, kwargs)
        return UNKNOWN

    def new_object(self, cls, args=(), kwargs=None):
        """NEWOBJ/NEWOBJ_EX and empty INST/OBJ never execute capability."""
        if _class_name(cls) == "functools.partial":
            return self.make_partial(args, kwargs or {})
        if _class_name(cls) in ("io.BytesIO", "io.StringIO"):
            return Buffer(b"" if _class_name(cls).endswith("BytesIO") else "")
        return Instance(cls, tuple(args), dict(kwargs or {}))

    def make_partial(self, args, kwargs):
        if not args:
            return Partial(UNKNOWN)
        fn, rest = args[0], tuple(args[1:])
        # CPython snapshots a plain partial when constructing another partial.
        # Partials with an instance dictionary retain the inner reference.
        if isinstance(fn, Partial) and not fn.has_dict:
            return Partial(fn.fn, fn.args + rest, {**fn.kwargs, **kwargs})
        return Partial(fn, rest, dict(kwargs))

    def build(self, obj, state):
        if isinstance(obj, Partial):
            if isinstance(state, (tuple, list)) and len(state) == 4:
                fn, args, kwargs, attrs = state
                # Mutate the existing object: all memo aliases must see the
                # new callable, arguments and keywords.
                obj.fn = fn
                obj.args = tuple(args) if isinstance(args, (list, tuple)) else ()
                obj.kwargs = kwargs if isinstance(kwargs, dict) else {}
                obj.attrs = attrs if isinstance(attrs, dict) else {}
                obj.has_dict = isinstance(attrs, dict)
            else:
                self.unsupported("invalid functools.partial state")
            return
        if isinstance(obj, Buffer):
            if isinstance(state, tuple) and len(state) >= 2:
                obj.data, obj.pos = state[:2]
            return
        if isinstance(obj, Instance):
            if _class_name(obj) in ("xmlrpc.client.ServerProxy", "xmlrpc.client.Server", "xmlrpc.client._Method"):
                self.finding("network", "BUILD invokes the XML-RPC proxy's remote __setstate__ method")
                return
            if isinstance(obj.cls, DynamicClass) and "__setstate__" in obj.cls.attrs:
                self.call(obj.cls.attrs["__setstate__"], (obj, state))
                return
            if "__setstate__" in obj.attrs:
                self.call(obj.attrs["__setstate__"], (state,))
                return
            obj.state = state
            slotstate = None
            if isinstance(state, tuple) and len(state) == 2:
                state, slotstate = state
            if isinstance(state, dict):
                obj.attrs.update(state)
            if isinstance(slotstate, dict):
                obj.attrs.update(slotstate)
            cls = _class_name(obj)
            if cls in ("joblib.numpy_pickle.NumpyArrayWrapper", "joblib.numpy_pickle.NDArrayWrapper",
                       "joblib.numpy_pickle_compat.NDArrayWrapper", "joblib.numpy_pickle_compat.ZNDArrayWrapper"):
                self.joblib_array(obj)
            # CPython encodes some objects (notably random.Random and numpy
            # dtypes/arrays) as tuple states. Retain, but never call setstate.
        elif isinstance(obj, Function) and isinstance(state, dict):
            obj.attrs.update(state)

    def joblib_array(self, obj):
        if _class_name(obj) != "joblib.numpy_pickle.NumpyArrayWrapper":
            self.finding("file-access", f"{self.operation} joblib array wrapper reads a sidecar file")
            return
        if self.file is None:
            return
        dtype = obj.attrs.get("dtype")
        shape = obj.attrs.get("shape", ())
        if not isinstance(dtype, Instance) or _class_name(dtype) != "numpy.dtype" or not dtype.args:
            raise ValueError("unresolved joblib array dtype")
        import numpy as np
        dt = np.dtype(dtype.args[0])
        # Structured dtype state: version, byteorder, subarray, names, fields,
        # elsize, alignment, flags. The raw byte count is supplied in elsize.
        state = dtype.state
        itemsize, hasobject = dt.itemsize, dt.hasobject
        if isinstance(state, tuple) and len(state) >= 8:
            if isinstance(state[5], int) and state[5] > 0:
                itemsize = state[5]
            if isinstance(state[7], int):
                hasobject = hasobject or bool(state[7] & 1)
        if hasobject:
            nested = Machine(self.where + ":joblib-object-array", self.budget, self.depth + 1)
            nested.run_stream(self.file)
            self.findings.extend(nested.findings)
            return
        if not isinstance(shape, (tuple, list)) or any(not isinstance(i, int) or i < 0 for i in shape):
            raise ValueError("unresolved joblib array shape")
        if obj.attrs.get("numpy_array_alignment_bytes") is not None:
            padding = self.file.read(1)
            if not padding:
                raise ValueError("missing joblib array padding")
            self.file.seek(padding[0], io.SEEK_CUR)
        size = math.prod(shape) * itemsize
        if size > self.budget.max_blob:
            raise ScanLimit("oversized joblib array")
        start = self.file.tell()
        self.file.seek(size, io.SEEK_CUR)
        if self.file.tell() > len(self.file.getbuffer()):
            raise ValueError(f"truncated joblib array at {start}")

    def nested_load(self, name, args, kwargs):
        from modelgate import dispatch
        from modelgate.numpy_scan import scan_array

        mode = NESTED_LOADERS[name]
        source = args[0] if args else kwargs.get("file", kwargs.get("f", kwargs.get("filename", kwargs.get("filepath_or_buffer", UNKNOWN))))
        self.budget.depth(self.depth + 1)
        if mode.endswith("bytes"):
            if isinstance(source, (bytes, bytearray)):
                data = bytes(source)
            else:
                return self.unsupported(f"{name} embedded bytes cannot be resolved")
        elif isinstance(source, Buffer):
            if not isinstance(source.data, (bytes, bytearray)):
                return self.unsupported(f"{name} embedded stream cannot be resolved")
            data = bytes(source.data[source.pos:])
        elif isinstance(source, (str, bytes)) or (isinstance(source, Instance) and _class_name(source).startswith("pathlib.")):
            self.finding("file-access", f"{self.operation} {name} reads an external path")
            return UNKNOWN
        else:
            # A file object acquired through open has already been reported.
            if isinstance(source, Instance) and _class_name(source) in ("builtins.open", "io.open", "io.FileIO"):
                self.finding("file-access", f"{self.operation} {name} reads an external file")
                return UNKNOWN
            return self.unsupported(f"{name} input cannot be resolved statically")
        loc = self.where + ":" + name
        try:
            if mode.startswith("pickle"):
                compression = kwargs.get("compression")
                if name.startswith("pandas.") and compression not in (None, "infer"):
                    expanded = dispatch.decompress(data, self.budget)
                    if expanded is not None:
                        data = expanded
                # loads/load consume exactly one pickle. Trailing bytes are
                # not executed here, unlike the registry's stream-sequence
                # loader used for a standalone raw pickle artifact.
                fs, value, end = analyze_pickle(data, loc, self.budget, self.depth + 1, multiple=False)
                if isinstance(source, Buffer):
                    source.pos += end
            elif mode in ("numpy", "npy"):
                import zipfile
                allow_index = 1 if mode == "npy" else 2
                allow = kwargs.get("allow_pickle", args[allow_index] if len(args) > allow_index else False)
                if data.startswith(b"\x93NUMPY"):
                    fs, value = scan_array(data, loc, self.budget, self.depth + 1, allow_pickle=bool(allow))
                elif mode == "numpy" and data.startswith((b"PK\x03\x04", b"PK\x05\x06")) and zipfile.is_zipfile(io.BytesIO(data)):
                    # numpy.load returns a lazy NpzFile. Each array is loaded
                    # only when the surrounding pickle actually indexes it.
                    fs, value = [], Npz(data, bool(allow))
                elif mode == "numpy" and not allow:
                    # Numeric arrays or an unopened lazy NPZ contain no code
                    # execution when allow_pickle=False.
                    fs, value = [], UNKNOWN
                else:
                    fs, value, _ = analyze_pickle(data, loc, self.budget, self.depth + 1, multiple=False)
            elif mode.startswith("torch"):
                # torch's zip check is deliberately stricter than zipfile's.
                if data.startswith(b"PK\x03\x04"):
                    fs, value = dispatch.scan_zip(data, loc, self.budget, self.depth + 1)
                else:
                    fs, value, _ = analyze_pickle(data, loc, self.budget, self.depth + 1)
            else:
                fs, value = dispatch.analyze_bytes(data, loc, self.budget, self.depth + 1)
            self.findings.extend(fs)
            return value
        except Exception as exc:
            return self.unsupported(f"{name} nested artifact cannot be analysed: {type(exc).__name__}: {exc}")

    def logging_config(self, config):
        if not isinstance(config, dict):
            self.unsupported("logging configuration cannot be resolved")
            return
        if config.get("incremental"):
            return
        for group in ("formatters", "filters", "handlers"):
            entries = config.get(group, {})
            if not isinstance(entries, dict):
                continue
            for entry in entries.values():
                self.budget.check()
                if not isinstance(entry, dict):
                    continue
                factory = entry.get("()")
                if factory is None and group in ("handlers", "formatters"):
                    factory = entry.get("class")
                if factory is None:
                    continue
                if isinstance(factory, str):
                    factory = Ref(canonical(factory.removeprefix("ext://")))
                parameters = {k: v for k, v in entry.items() if k not in ("()", ".", "class", "level", "formatter", "filters")}
                for key, value in parameters.items():
                    if isinstance(value, str) and value.startswith("ext://"):
                        parameters[key] = Ref(canonical(value[6:]))
                self.call(factory, (), parameters)

    def timeit_call(self, name, args, kwargs):
        if name.startswith("timeit.Timer.") and args and isinstance(args[0], Instance):
            obj = args[0]
            stmt, setup = obj.attrs.get("stmt"), obj.attrs.get("setup")
        else:
            stmt = args[0] if args else kwargs.get("stmt")
            setup = args[1] if len(args) > 1 else kwargs.get("setup")
        for value in (stmt, setup):
            if isinstance(value, (str, bytes, Code)):
                self.finding("code-execution", f"{self.operation} {name} compiles Python source")
            elif value is not None and name != "timeit.Timer":
                self.call(value, ())
        return Instance(Ref("timeit.Timer"), attrs={"stmt": stmt, "setup": setup}) if name == "timeit.Timer" else UNKNOWN

    def extra_capability(self, name, args, kwargs):
        kind = None
        # These families share an actual implementation/capability; matching is
        # restricted to their known module/class, not arbitrary method names.
        if name.startswith(("multiprocessing.popen_",)) and name.endswith(".Popen"):
            kind = "process"
        elif name in ("multiprocessing.process.BaseProcess.start", "multiprocessing.context.Process.start", "multiprocessing.Process.start"):
            kind = "process"
        elif name.startswith("concurrent.futures.ProcessPoolExecutor.") and name.rsplit(".", 1)[-1] in ("submit", "map"):
            kind = "process"
        elif name.startswith(("io.FileIO.", "io.TextIOWrapper.", "io.BufferedReader.", "io.BufferedWriter.", "io.BufferedRandom.")):
            if name.rsplit(".", 1)[-1] in ("read", "read1", "readall", "readline", "readlines", "write", "writelines", "truncate", "flush"):
                if args and isinstance(args[0], Instance) and args[0].args and isinstance(args[0].args[0], Buffer):
                    return False
                kind = "file-access"
        elif name in ("gzip.open", "gzip.GzipFile", "bz2.open", "bz2.BZ2File", "lzma.open", "lzma.LZMAFile", "tarfile.open", "tarfile.TarFile.open", "zipfile.ZipFile"):
            source = args[0] if args else kwargs.get("filename", kwargs.get("name", kwargs.get("fileobj", UNKNOWN)))
            if not isinstance(source, Buffer) and source is not None:
                kind = "file-access"
        elif name.startswith("numpy.") and name.rsplit(".", 1)[-1] in ("fromfile", "tofile", "loadtxt", "genfromtxt", "savetxt", "save", "savez", "savez_compressed", "memmap", "open_memmap"):
            source = args[0] if args else kwargs.get("file", kwargs.get("fname", UNKNOWN))
            if not isinstance(source, Buffer):
                kind = "file-access"
        elif name.startswith("pandas.") and name.rsplit(".", 1)[-1] in ("read_csv", "read_table", "read_json", "read_hdf", "read_excel", "read_parquet", "read_feather", "read_sql", "read_html"):
            if not args or not isinstance(args[0], Buffer):
                kind = "file-access"
        elif name in ("pickle.dump", "pickle.dumps"):
            if name == "pickle.dump" and len(args) > 1 and not isinstance(args[1], Buffer):
                kind = "file-access"
        elif name in ("torch.save", "joblib.dump"):
            if len(args) < 2 or not isinstance(args[1], Buffer):
                kind = "file-access"
        elif name in ("importlib.machinery.SourceFileLoader.load_module", "importlib.machinery.SourceFileLoader.exec_module",
                      "importlib.machinery.SourcelessFileLoader.load_module", "importlib.machinery.SourcelessFileLoader.exec_module",
                      "pkgutil.get_data", "pkg_resources.resource_string"):
            kind = "code-execution" if name.startswith("importlib.") else "file-access"
        if kind:
            self.finding(kind, f"{self.operation} {name}")
            return True
        return False

    def pop_mark(self):
        index = len(self.stack) - 1
        while index >= 0 and self.stack[index] is not MARK:
            index -= 1
        if index < 0:
            raise ValueError("pickle MARK missing")
        values = self.stack[index + 1:]
        del self.stack[index:]
        return values

    def set_items(self, obj, pairs):
        if len(pairs) % 2:
            raise ValueError("odd number of dictionary items")
        if isinstance(obj, dict):
            target = obj
        elif isinstance(obj, Instance):
            target = obj.attrs.setdefault("__items__", {})
        else:
            raise ValueError("SETITEM target is not a mapping")
        for i in range(0, len(pairs), 2):
            target[pairs[i]] = pairs[i + 1]

    def append_items(self, obj, items):
        if isinstance(obj, list):
            obj.extend(items)
        elif isinstance(obj, Instance):
            obj.attrs.setdefault("__items__", []).extend(items)
        else:
            raise ValueError("APPEND target is not a sequence")

    def run_stream(self, source, index=0):
        self.file = source
        self.stack = []
        self.memo = {}
        self.protocol = 0
        self.budget.depth(self.depth)
        stack = self.stack
        start = source.tell()
        for op, arg, pos in pickletools.genops(source):
            self.budget.check()
            self.where = f"{self.location}:stream[{index}]@{pos}"
            self.operation = op.name
            name = op.name
            if name == "PROTO":
                if arg > 5:
                    raise ValueError(f"unsupported pickle protocol {arg}")
                self.protocol = arg
            elif name == "FRAME":
                # FRAME specifies buffering, not control flow. Reading the
                # underlying bytes also supports joblib's inline array data.
                if source.tell() + arg > len(source.getbuffer()):
                    raise ValueError("truncated pickle frame")
            elif name in ("STRING", "BINSTRING", "SHORT_BINSTRING", "UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8",
                          "BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8",
                          "INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT"):
                stack.append(bytes(arg) if isinstance(arg, bytearray) else arg)
            elif name == "NONE":
                stack.append(None)
            elif name == "NEWTRUE":
                stack.append(True)
            elif name == "NEWFALSE":
                stack.append(False)
            elif name == "MARK":
                stack.append(MARK)
            elif name == "POP":
                stack.pop()
            elif name == "POP_MARK":
                self.pop_mark()
            elif name == "DUP":
                stack.append(stack[-1])
            elif name == "EMPTY_TUPLE":
                stack.append(())
            elif name == "EMPTY_LIST":
                stack.append([])
            elif name == "EMPTY_DICT":
                stack.append({})
            elif name == "EMPTY_SET":
                stack.append(set())
            elif name in ("TUPLE", "LIST", "DICT", "FROZENSET"):
                values = self.pop_mark()
                if name == "DICT":
                    value = {}
                    self.set_items(value, values)
                else:
                    value = {"TUPLE": tuple, "LIST": list, "FROZENSET": frozenset}[name](values)
                stack.append(value)
            elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
                n = int(name[-1])
                values = tuple(stack[-n:])
                del stack[-n:]
                stack.append(values)
            elif name == "APPEND":
                value = stack.pop()
                self.append_items(stack[-1], [value])
            elif name == "APPENDS":
                values = self.pop_mark()
                self.append_items(stack[-1], values)
            elif name == "SETITEM":
                value, key = stack.pop(), stack.pop()
                self.set_items(stack[-1], [key, value])
            elif name == "SETITEMS":
                values = self.pop_mark()
                self.set_items(stack[-1], values)
            elif name == "ADDITEMS":
                values = self.pop_mark()
                stack[-1].update(values)
            elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
                self.memo[int(arg)] = stack[-1]
            elif name == "MEMOIZE":
                self.memo[len(self.memo)] = stack[-1]
            elif name in ("GET", "BINGET", "LONG_BINGET"):
                stack.append(self.memo[int(arg)])
            elif name == "GLOBAL":
                module, attr = arg.split(" ", 1)
                stack.append(global_ref(module, attr, self.protocol))
            elif name == "STACK_GLOBAL":
                attr, module = stack.pop(), stack.pop()
                stack.append(global_ref(module, attr, self.protocol))
            elif name in ("EXT1", "EXT2", "EXT4"):
                stack.append(Unknown(f"extension code {arg}"))
            elif name == "REDUCE":
                args, fn = stack.pop(), stack.pop()
                stack.append(self.call(fn, args))
            elif name == "INST":
                args = self.pop_mark()
                module, attr = arg.split(" ", 1)
                cls = global_ref(module, attr, self.protocol)
                stack.append(self.call(cls, args) if args else self.new_object(cls))
            elif name == "OBJ":
                values = self.pop_mark()
                cls, args = values[0], values[1:]
                stack.append(self.call(cls, args) if args else self.new_object(cls))
            elif name == "NEWOBJ":
                args, cls = stack.pop(), stack.pop()
                stack.append(self.new_object(cls, args))
            elif name == "NEWOBJ_EX":
                kwargs, args, cls = stack.pop(), stack.pop(), stack.pop()
                stack.append(self.new_object(cls, args, kwargs))
            elif name == "BUILD":
                state = stack.pop()
                self.build(stack[-1], state)
            elif name == "PERSID":
                stack.append(Unknown("persistent storage"))
            elif name == "BINPERSID":
                stack.pop()
                stack.append(Unknown("persistent storage"))
            elif name == "NEXT_BUFFER":
                stack.append(Unknown("out-of-band buffer"))
            elif name == "READONLY_BUFFER":
                pass
            elif name == "STOP":
                if not stack:
                    raise ValueError("empty pickle stack at STOP")
                return stack.pop()
            else:
                raise ValueError(f"unhandled pickle opcode {name}")
        raise ValueError(f"pickle at offset {start} has no STOP")


def analyze_pickle(data: bytes, location: str, budget=None, depth=0, multiple=True):
    budget = budget or Budget()
    budget.depth(depth)
    machine = Machine(location, budget, depth)
    source = io.BytesIO(data)
    value, object_value = UNKNOWN, UNKNOWN
    index, legacy, legacy_candidate = 0, False, False
    try:
        while source.tell() < len(data):
            value = machine.run_stream(source, index)
            if index == 0 and isinstance(value, int) and value == TORCH_MAGIC and multiple:
                legacy_candidate = True
            elif index == 1:
                legacy_candidate = legacy_candidate and isinstance(value, int) and value == 1001
            elif index == 2:
                legacy = legacy_candidate and isinstance(value, dict) and "little_endian" in value
            if legacy and index == 3:
                object_value = value
            index += 1
            if not multiple:
                break
            # The fifth pickle ends legacy torch metadata. Everything after
            # it is raw storage, and must not be interpreted as opcodes.
            if legacy and index == 5:
                value = object_value
                break
        if not index:
            raise ValueError("empty pickle")
        if legacy and index < 5:
            raise ValueError("incomplete legacy torch checkpoint")
    except Exception as exc:
        machine.findings.append(Finding("unsupported", machine.where,
                                        f"could not interpret pickle: {type(exc).__name__}: {exc}"))
    return machine.findings, value, source.tell()


def scan_pickle(data: bytes, location: str, budget=None, depth=0, multiple=True) -> list[Finding]:
    return analyze_pickle(data, location, budget, depth, multiple)[0]
