"""What calls, attribute lookups and iteration do, symbolically.

The pickle simulator (unpickler.py) hands every operation that could run code
-- a call (REDUCE, INST, OBJ, BUILD's __setstate__, APPEND's extend ...), an
attribute lookup, an iteration -- to the ``Sem`` object below, which decides
from the stdlib index and the capability tables what the real object would do.
Capability-bearing calls are reported as findings; routes (getattr chains,
partials, map consumed by list, threads, ...) are followed; plain data
transformations (base64, zlib, string building) are computed so that nested
payloads can be judged after decoding.
"""
from __future__ import annotations

import time

from modelgate import pyindex
from modelgate.caps import CAPS, KIND
from modelgate.report import Finding
from modelgate.values import (
    AttrGetter, BoundMethod, ClassDict, ClassObj, CodeObj, Coroutine, FileObj, FunctionObj,
    Instance, ItemGetter, LazyIter, MethodCaller, Module, ModuleDict, Obj, Opaque, PByteArray,
    PDict, PList, PSet, Partial, Ref, SimError, SysModules, describe, is_hashable, is_native,
)

MAX_ITER = 20000          # elements produced when consuming one iterator
MAX_BYTES = 256 << 20     # largest value built by a data transformation
NOTFOUND = object()


class Budget(Exception):
    """The scan ran out of time or steps."""


class Context:
    """State shared by every load performed while scanning one artifact
    (several streams, nested loaders): module rebinding persists across them,
    as it would in the registry's process."""

    def __init__(self, deadline: float, max_steps: int = 30_000_000):
        self.findings: list[Finding] = []
        self._seen: set = set()
        self.modules: dict[str, Module] = {}
        self.sys_modules = SysModules()
        self.extensions: dict = {}
        self.class_overlay: dict = {}
        self.func_attrs: dict = {}
        self.deadline = deadline
        self.steps = 0
        self.max_steps = max_steps
        self.depth = 0
        self.loc = "?"
        self.nested = None  # callback installed by formats.py: (loader, payload, opts) -> value
        self.notes: list[str] = []

    def tick(self, n: int = 1):
        self.steps += n
        if self.steps > self.max_steps or (self.steps & 0x3FF) == 0 and time.monotonic() > self.deadline:
            raise Budget("analysis budget exhausted")

    def flag(self, letter: str, detail: str, location: str | None = None):
        kind = KIND.get(letter, letter)
        loc = location or self.loc
        key = (kind, loc, detail)
        if key in self._seen:
            return
        self._seen.add(key)
        self.findings.append(Finding(kind, loc, detail))

    def unsupported(self, detail: str, location: str | None = None):
        self.flag("unsupported", detail, location)


# --------------------------------------------------------------------------
# capability table, normalised through the index
_CAPS: dict | None = None
_HANDLERS: dict = {}
_LAZY_CAPS = {"glob.iglob", "fileinput.input", "fileinput.FileInput"}


def caps_table() -> dict:
    global _CAPS
    if _CAPS is None:
        out = {}
        for path, spec in CAPS.items():
            if spec is None:
                continue
            oid = pyindex.resolve_path(path)
            if oid is None or oid.startswith("m:"):
                oid = path
            out.setdefault(oid, spec)
            out.setdefault(path, spec)
        _CAPS = out
    return _CAPS


def handler(*paths):
    def deco(fn):
        for p in paths:
            _HANDLERS[p] = fn
        return fn
    return deco


_NORMALISED = False


def handlers() -> dict:
    global _NORMALISED
    if not _NORMALISED:
        import modelgate.handlers  # noqa: F401  (registers the modelled callables)
        extra = {}
        for path, fn in _HANDLERS.items():
            oid = pyindex.resolve_path(path)
            if oid and not oid.startswith("m:") and oid not in _HANDLERS:
                extra[oid] = fn
        _HANDLERS.update(extra)
        _NORMALISED = True
    return _HANDLERS


# --------------------------------------------------------------------------
# signatures
_SIG_CACHE: dict = {}


def parse_sig(sig: str):
    p = _SIG_CACHE.get(sig)
    if p is None:
        p = []
        if sig:
            for part in sig.split(","):
                name, rest = part.rsplit(":", 1)
                p.append((name, int(rest[0]), rest.endswith("=")))
        _SIG_CACHE[sig] = p
    return p


def bind(sig, args, kwargs):
    """Bind call arguments like CPython would; None when the call would
    raise TypeError before running."""
    if sig is None:
        return None
    params = parse_sig(sig)
    out = {}
    i = 0
    kw = dict(kwargs)
    for name, kind, has_def in params:
        if kind in (0, 1):
            if i < len(args):
                if kind == 1 and name in kw:
                    return None
                out[name] = args[i]
                i += 1
            elif kind == 1 and name in kw:
                out[name] = kw.pop(name)
            elif not has_def:
                return None
        elif kind == 2:
            out[name] = tuple(args[i:])
            i = len(args)
        elif kind == 3:
            if name in kw:
                out[name] = kw.pop(name)
            elif not has_def:
                return None
        elif kind == 4:
            out[name] = dict(kw)
            kw = {}
    if i < len(args) or kw:
        return None
    return out


def sig_of(ref: Ref):
    if ref.sig is None and ref.kind != "?":
        e = pyindex.object_entry(ref.oid)
        if e:
            if e[0] == "c":
                ref.sig = e[5] if len(e) > 5 else None
            elif len(e) > 2:
                ref.sig = e[2]
    return ref.sig


def short(v, n=60) -> str:
    s = describe(v)
    return s if len(s) <= n else s[: n - 3] + "..."


BUILTIN_TYPE_OIDS = {
    type(None): "types.NoneType", bool: "builtins.bool", int: "builtins.int",
    float: "builtins.float", complex: "builtins.complex", str: "builtins.str",
    bytes: "builtins.bytes", tuple: "builtins.tuple", frozenset: "builtins.frozenset",
    slice: "builtins.slice", range: "builtins.range",
}
CONTAINER_TYPE_OIDS = {
    PList: "builtins.list", PDict: "builtins.dict", PSet: "builtins.set",
    PByteArray: "builtins.bytearray", ModuleDict: "builtins.dict", SysModules: "builtins.dict",
    FileObj: "io.BytesIO", LazyIter: "builtins.map",
}

PY2_ONLY_MODULES = frozenset({
    "__builtin__", "copy_reg", "Queue", "SocketServer", "ConfigParser", "repr", "tkFileDialog",
    "tkSimpleDialog", "tkColorChooser", "tkCommonDialog", "Dialog", "Tkdnd", "tkFont",
    "tkMessageBox", "ScrolledText", "Tkconstants", "Tix", "ttk", "Tkinter", "markupbase",
    "_winreg", "thread", "dummy_thread", "dbhash", "dumbdbm", "gdbm", "xmlrpclib",
    "SimpleXMLRPCServer", "httplib", "htmlentitydefs", "HTMLParser", "Cookie", "cookielib",
    "BaseHTTPServer", "test.test_support", "commands", "urlparse", "robotparser", "urllib2",
    "anydbm", "_abcoll", "cPickle", "FileDialog", "SimpleDialog", "DocXMLRPCServer",
    "SimpleHTTPServer", "CGIHTTPServer", "UserDict", "UserList", "UserString", "whichdb",
    "StringIO", "cStringIO", "exceptions", "multiprocessing.forking",
})
MODULE_ALIASES = {"nt": "posix"}
# attribute names that, on a module outside the index (a third-party package),
# are almost always the stdlib module it imported (``torch.os`` is ``os``)
STDLIB_MODULE_ATTRS = frozenset({
    "os", "sys", "subprocess", "builtins", "shutil", "socket", "pickle", "marshal", "importlib", "io", "pty",
    "platform", "ctypes", "code", "codeop", "runpy", "pdb", "bdb", "timeit", "webbrowser", "urllib", "ftplib",
    "smtplib", "tempfile", "pathlib", "glob", "multiprocessing", "threading", "asyncio", "functools", "operator",
    "copyreg", "pydoc", "posix", "_io", "_pickle", "zipfile", "tarfile", "shelve", "sqlite3", "inspect",
    "linecache", "pkgutil", "_posixsubprocess", "posixpath",
})
UNINDEXED_SUBMODULES = frozenset({"encodings", "numpy", "test", "idlelib", "tkinter", "lib2to3",
                                  "turtledemo", "pydoc_data", "ensurepip", "venv", "distutils"})


_CANON: dict = {}


def _canonical(path):
    """Canonical object id for a dotted path (aliases resolved), or None."""
    if path in _CANON:
        return _CANON[path]
    r = pyindex.resolve_path(path)
    out = r if r and not r.startswith("m:") else None
    _CANON[path] = out
    return out


class Sem:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.caps = caps_table()
        self.handlers = handlers()

    # ------------------------------------------------------------ modules
    def module(self, name: str) -> Module:
        name = pyindex.module_canonical(name)
        m = self.ctx.modules.get(name)
        if m is None:
            m = Module(name, pyindex.module_table(name) is not None)
            self.ctx.modules[name] = m
        return m

    def import_module(self, name):
        """``__import__(name)`` followed by ``sys.modules[name]``."""
        if not isinstance(name, str):
            raise SimError("module name must be str")
        sm = self.ctx.sys_modules
        if name in sm.overlay:
            return sm.overlay[name]
        if not name or name.startswith(".") or name.endswith(".") or ".." in name:
            raise SimError(f"bad module name {name!r}")
        if name in PY2_ONLY_MODULES:
            raise SimError(f"No module named {name!r}")
        name = MODULE_ALIASES.get(name, name)
        if pyindex.module_table(name) is not None:
            return self.module(name)
        top = name.split(".")[0]
        if (top in pyindex.STDLIB_TOP and top not in UNINDEXED_SUBMODULES
                and pyindex.module_table(top) is not None):
            raise SimError(f"No module named {name!r}")
        return self.module(name)

    def ref(self, oid: str) -> Obj:
        if oid == "sys.modules":
            return self.ctx.sys_modules
        e = pyindex.object_entry(oid)
        if e is None:
            canon = _canonical(oid)
            if canon is not None and canon != oid:
                return self.ref(canon)
            return Ref(oid, "?")
        if e[0] == "i":
            return Opaque(oid)
        flags = e[1] if e[0] != "c" and len(e) > 1 else ""
        return Ref(oid, e[0], flags or "")

    def ref_value(self, r: str, path: str):
        if r.startswith("m:"):
            return self.module(r[2:])
        if r.startswith("o:"):
            return self.ref(r[2:])
        return Opaque(path)

    # ------------------------------------------------------------ getattr
    def getattr(self, obj, name):
        self.ctx.tick()
        if not isinstance(name, str):
            raise SimError("attribute name must be string")
        if isinstance(obj, Module):
            return self.module_getattr(obj, name)
        if isinstance(obj, Ref):
            return self.ref_getattr(obj, name)
        if isinstance(obj, Instance):
            return self.instance_getattr(obj, name)
        if isinstance(obj, ClassObj):
            v = self.classobj_lookup(obj, name)
            if v is NOTFOUND:
                if name == "__name__":
                    return obj.name
                if name == "__dict__":
                    return obj.ns
                if name == "__bases__":
                    return obj.bases
                raise SimError(f"type object has no attribute {name!r}")
            return v
        if isinstance(obj, BoundMethod):
            if name == "__self__":
                return obj.self_
            if name == "__func__":
                return obj.func
            if name == "__call__":
                return obj
            return Opaque(f"attribute {name} of a method")
        if isinstance(obj, Opaque):
            if obj.attrs and name in obj.attrs:
                return obj.attrs[name]
            return Opaque(f"{obj.desc}.{name}", obj.tainted)
        if isinstance(obj, (FunctionObj, CodeObj, Coroutine)):
            if name == "__call__":
                return obj
            return Opaque(f"{describe(obj)}.{name}")
        if isinstance(obj, SysModules) or isinstance(obj, ModuleDict) or isinstance(obj, ClassDict):
            try:
                return self.native_method(obj, "builtins.dict", name)
            except SimError:
                if isinstance(obj, ModuleDict) and obj.module.name == "builtins":
                    return self.module_getattr(obj.module, name)
                raise
        if isinstance(obj, LazyIter):
            if name in ("__next__", "__iter__", "send"):
                return BoundMethod(obj, Ref("builtins.iterator." + name, "b"))
            return Opaque(f"iterator.{name}")
        if isinstance(obj, FileObj):
            return BoundMethod(obj, Ref("io.BytesIO." + name, "md"))
        tid = CONTAINER_TYPE_OIDS.get(type(obj)) or BUILTIN_TYPE_OIDS.get(type(obj))
        if tid is None:
            raise SimError(f"no attribute {name!r}")
        return self.native_method(obj, tid, name)

    def native_method(self, obj, tid, name):
        if name == "__class__":
            return self.ref(tid)
        if tid in ("builtins.bytes", "builtins.bytearray") and name in ("tobytes", "toreadonly", "tolist", "release",
                                                                         "cast"):
            # memoryview(...) is modelled as the bytes it views
            return BoundMethod(obj, Ref("builtins.memoryview." + name, "md"))
        hit = pyindex.class_lookup(tid, name) or pyindex.class_lookup("builtins.object", name)
        if hit is None:
            raise SimError(f"{tid} object has no attribute {name!r}")
        r, owner = hit
        if r.startswith("cm:"):
            return BoundMethod(self.ref(tid), self.ref(r[3:]))
        if r.startswith("sm:"):
            return self.ref(r[3:])
        if r == "v":
            if is_native(obj):
                try:
                    v = getattr(obj, name)
                    if is_native(v):
                        return v
                except Exception:
                    pass
            return Opaque(f"{tid}.{name}")
        return BoundMethod(obj, self.ref(r[2:]))

    def module_getattr(self, m: Module, name: str):
        if name in m.overlay:
            return m.overlay[name]
        if name in m.deleted:
            raise SimError(f"module {m.name!r} has no attribute {name!r}")
        if name == "__dict__":
            return ModuleDict(m)
        if name == "__name__":
            return m.name
        if m.fresh:
            raise SimError(f"module {m.name!r} has no attribute {name!r}")
        if m.indexed:
            if name == "__builtins__":
                return ModuleDict(self.module("builtins"))
            r = pyindex.module_attr_ref(m.name, name)
            if r is not None:
                return self.ref_value(r, f"{m.name}.{name}")
            sub = f"{m.name}.{name}"
            if pyindex.module_table(sub) is not None:
                return self.module(sub)
            if pyindex.module_attr_ref(m.name, "__getattr__") is not None:
                # PEP 562 lazy attribute: look for it in the package's submodules
                prefix = m.name + "."
                for modname in pyindex.modules():
                    if modname.startswith(prefix):
                        r = pyindex.module_attr_ref(modname, name)
                        if r is not None and r.startswith("o:") and r[2:].startswith(prefix):
                            return self.ref_value(r, f"{m.name}.{name}")
                return Ref(f"{m.name}.{name}", "?")
            if name in ("__file__", "__cached__"):
                return f"/usr/local/lib/python3.12/{m.name.replace('.', '/')}.py"
            if name.startswith("__") and name.endswith("__"):
                return Opaque(f"{m.name}.{name}")
            raise SimError(f"module {m.name!r} has no attribute {name!r}")
        if name in STDLIB_MODULE_ATTRS:
            return self.module(name)
        if name == "__builtins__":
            return ModuleDict(self.module("builtins"))
        return Ref(f"{m.name}.{name}", "?")

    def class_lookup(self, cls, name):
        """Raw class attribute (unbound) as (tag, value) or NOTFOUND.
        tag: "func" (binds to instances), "cm", "sm", "plain", "data"."""
        if isinstance(cls, ClassObj):
            v = self.classobj_raw(cls, name)
            if v is NOTFOUND:
                return NOTFOUND
            return ("func" if self.is_python_function(v) else "plain", v)
        if not isinstance(cls, Ref):
            return NOTFOUND
        ov = self.ctx.class_overlay.get(cls.oid)
        if ov and name in ov:
            v = ov[name]
            return ("func" if self.is_python_function(v) else "plain", v)
        if cls.kind != "c":
            return NOTFOUND
        hit = pyindex.class_lookup(cls.oid, name)
        if hit is None:
            hit = pyindex.class_lookup("builtins.object", name)
            if hit is None:
                return NOTFOUND
        r, owner = hit
        if r.startswith("cm:"):
            return ("cm", self.ref(r[3:]))
        if r.startswith("sm:"):
            return ("sm", self.ref(r[3:]))
        if r == "v":
            return ("data", Opaque(f"{cls.oid}.{name}"))
        v = self.ref(r[2:])
        if isinstance(v, Ref) and v.kind == "p":
            return ("data", Opaque(f"{cls.oid}.{name}"))
        if isinstance(v, Ref) and v.kind in ("f", "md"):
            return ("func", v)
        return ("plain", v)

    @staticmethod
    def is_python_function(v):
        return isinstance(v, FunctionObj) or (isinstance(v, Ref) and v.kind in ("f", "md"))

    def classobj_raw(self, cls: ClassObj, name):
        if name in cls.ns.d:
            return cls.ns.d[name]
        for b in cls.bases:
            if isinstance(b, ClassObj):
                v = self.classobj_raw(b, name)
                if v is not NOTFOUND:
                    return v
            elif isinstance(b, Ref):
                t = self.class_lookup(b, name)
                if t is not NOTFOUND:
                    return t[1]
        return NOTFOUND

    def classobj_lookup(self, cls: ClassObj, name):
        return self.classobj_raw(cls, name)

    def ref_getattr(self, ref: Ref, name: str):
        k = ref.kind
        if k == "?":
            if name == "__name__":
                return ref.oid.rsplit(".", 1)[-1]
            if name == "__module__":
                return ref.oid.rsplit(".", 1)[0]
            if name == "__call__":
                return ref
            if name in STDLIB_MODULE_ATTRS:
                return self.module(name)
            if name == "__builtins__":
                return ModuleDict(self.module("builtins"))
            if name in ("__globals__", "__dict__"):
                owner = ref.oid if name == "__dict__" else ref.oid.rsplit(".", 1)[0]
                return ModuleDict(self.module(owner))
            return Ref(f"{ref.oid}.{name}", "?")
        if k == "c":
            t = self.class_lookup(ref, name)
            if t is not NOTFOUND:
                tag, v = t
                if tag == "cm":
                    return BoundMethod(ref, v)
                return v
            if name == "__name__":
                return ref.oid.rsplit(".", 1)[-1]
            if name in ("__qualname__",):
                return ref.oid.rsplit(".", 1)[-1]
            if name == "__module__":
                return pyindex.owner_module(ref.oid) or ref.oid.rsplit(".", 1)[0]
            if name == "__dict__":
                return ClassDict(ref)
            if name in ("__mro__", "__bases__"):
                info = pyindex.class_info(ref.oid)
                mro = tuple(self.ref(o) for o in (info[2] if info else []))
                return ((ref,) + mro) if name == "__mro__" else mro[:1]
            if name == "__base__":
                info = pyindex.class_info(ref.oid)
                return self.ref(info[2][0]) if info and info[2] else self.ref("builtins.object")
            t = pyindex.class_lookup("builtins.type", name)
            if t is not None:
                r = t[0]
                if r.startswith("o:"):
                    return BoundMethod(ref, self.ref(r[2:]))
                return Opaque(f"{ref.oid}.{name}")
            if name.startswith("__") and name.endswith("__"):
                return Opaque(f"{ref.oid}.{name}")
            raise SimError(f"type object {ref.oid!r} has no attribute {name!r}")
        # functions, builtins, descriptors, other callables
        if name == "__call__":
            return ref
        if name == "__name__" or name == "__qualname__":
            return ref.oid.rsplit(".", 1)[-1]
        if name == "__module__":
            return pyindex.owner_module(ref.oid)
        if k == "f" and name == "__globals__":
            mod = pyindex.owner_module(ref.oid)
            return ModuleDict(self.module(mod)) if mod else Opaque("globals")
        if k == "f" and name == "__builtins__":
            return ModuleDict(self.module("builtins"))
        if k == "f" and name == "__code__":
            return CodeObj("library function")
        if k == "b" and name == "__self__":
            mod = pyindex.owner_module(ref.oid)
            return self.module(mod) if mod else Opaque("self")
        if name == "__dict__":
            return self.ctx.func_attrs.setdefault(ref.oid, PDict())
        fa = self.ctx.func_attrs.get(ref.oid)
        if fa is not None and name in fa.d:
            return fa.d[name]
        if name in ("__wrapped__",):
            raise SimError("no __wrapped__")
        if name.startswith("__") and name.endswith("__"):
            return Opaque(f"{ref.oid}.{name}")
        if k in ("x", "bm", "i"):
            return Opaque(f"{ref.oid}.{name}")
        raise SimError(f"{ref.oid} has no attribute {name!r}")

    def instance_getattr(self, inst: Instance, name: str):
        if name == "__dict__":
            if inst.d is None:
                raise SimError("object has no __dict__")
            return inst.d
        if name == "__class__":
            return inst.cls
        if isinstance(inst, Partial):
            if name == "func":
                return inst.func
            if name == "args":
                return inst.args
            if name == "keywords":
                return PDict(inst.kw)
        cls = inst.cls
        t = self.class_lookup(cls, name) if isinstance(cls, (Ref, ClassObj)) else NOTFOUND
        if t is not NOTFOUND and t[0] == "data" and inst.d is not None and name in inst.d.d:
            return inst.d.d[name]
        if inst.d is not None and name in inst.d.d:
            return inst.d.d[name]
        if t is not NOTFOUND:
            tag, v = t
            if tag == "func":
                return BoundMethod(inst, v)
            if tag == "cm":
                return BoundMethod(cls, v)
            if tag == "data":
                ex = inst.extra.get("attr:" + name, NOTFOUND)
                return ex if ex is not NOTFOUND else v
            return v
        if isinstance(cls, Ref) and cls.oid == "xmlrpc.client.ServerProxy" and not name.startswith("__"):
            return BoundMethod(inst, Ref("xmlrpc.client._Method.__call__", "f"))
        if isinstance(cls, Ref) and cls.kind == "?":
            return BoundMethod(inst, Ref(f"{cls.oid}.{name}", "?"))
        if isinstance(cls, Ref) and cls.kind == "c":
            ga = pyindex.class_lookup(cls.oid, "__getattr__")
            if ga is not None:
                return Opaque(f"{cls.oid} dynamic attribute {name}")
            info = pyindex.class_info(cls.oid)
            if inst.extra.get("constructed") and (info is None or info[0]):
                # an attribute its __init__ would have set
                return Opaque(f"{cls.oid} instance attribute {name}")
        if isinstance(cls, ClassObj) and self.classobj_raw(cls, "__getattr__") is not NOTFOUND:
            return self.call(self.classobj_raw(cls, "__getattr__"), (inst, name), {}, "__getattr__")
        raise SimError(f"object has no attribute {name!r}")

    # ------------------------------------------------------------ setattr
    def setattr(self, obj, name, value):
        self.ctx.tick()
        if not isinstance(name, str):
            raise SimError("attribute name must be string")
        if isinstance(obj, Module):
            obj.overlay[name] = value
            obj.deleted.discard(name)
            return
        if isinstance(obj, Instance):
            if isinstance(obj, Partial) and name in ("func", "args", "keywords"):
                raise SimError("readonly attribute")
            if obj.d is None:
                cls = obj.cls
                if isinstance(cls, Ref) and cls.kind == "c":
                    t = self.class_lookup(cls, name)
                    if t is not NOTFOUND and t[0] == "data":
                        obj.extra["attr:" + name] = value
                        return
                raise SimError(f"cannot set attribute {name!r}")
            obj.d.d[name] = value
            return
        if isinstance(obj, ClassObj):
            obj.ns.d[name] = value
            return
        if isinstance(obj, Ref):
            if obj.kind == "c":
                e = pyindex.object_entry(obj.oid)
                if e and not e[1]:
                    raise SimError("cannot set attributes of a built-in type")
                self.ctx.class_overlay.setdefault(obj.oid, {})[name] = value
                return
            if obj.kind in ("f", "?"):
                self.ctx.func_attrs.setdefault(obj.oid, PDict()).d[name] = value
                if obj.kind == "?":
                    self.ctx.class_overlay.setdefault(obj.oid, {})[name] = value
                return
            raise SimError("cannot set attribute on builtin")
        if isinstance(obj, Opaque):
            if obj.attrs is None:
                obj.attrs = {}
            obj.attrs[name] = value
            return
        raise SimError(f"cannot set attribute {name!r} on {describe(obj)}")

    def delattr(self, obj, name):
        if isinstance(obj, Module):
            obj.overlay.pop(name, None)
            obj.deleted.add(name)
            return
        if isinstance(obj, Instance) and obj.d is not None:
            if name not in obj.d.d:
                raise SimError("no such attribute")
            del obj.d.d[name]
            return
        if isinstance(obj, Opaque):
            return
        raise SimError("cannot delete attribute")

    # ------------------------------------------------------------ items
    def getitem(self, obj, key):
        self.ctx.tick()
        if isinstance(obj, PDict):
            if not is_hashable(key):
                raise SimError("unhashable key")
            try:
                return obj.d[key]
            except KeyError:
                raise SimError(f"KeyError {short(key)}") from None
        if isinstance(obj, ModuleDict):
            try:
                return self.module_getattr(obj.module, key) if isinstance(key, str) else self._keyerr(key)
            except SimError:
                raise SimError(f"KeyError {short(key)}") from None
        if isinstance(obj, SysModules):
            if not isinstance(key, str):
                raise SimError("KeyError")
            if key in obj.overlay:
                return obj.overlay[key]
            if key in obj.deleted:
                raise SimError(f"KeyError {key!r}")
            return self.import_module(key)
        if isinstance(obj, ClassDict):
            t = self.class_lookup(obj.cls, key) if isinstance(key, str) else NOTFOUND
            if t is NOTFOUND:
                raise SimError("KeyError")
            return t[1]
        if isinstance(obj, (PList, tuple, str, bytes, PByteArray)):
            seq = obj.items if isinstance(obj, PList) else (obj.b if isinstance(obj, PByteArray) else obj)
            if not isinstance(key, (int, slice)):
                raise SimError("indices must be integers")
            try:
                v = seq[key]
            except (IndexError, ValueError, TypeError):
                raise SimError("IndexError") from None
            if isinstance(key, slice):
                if isinstance(obj, PList):
                    return PList(v)
                if isinstance(obj, PByteArray):
                    return PByteArray(v)
            return v
        if isinstance(obj, Instance):
            cls = obj.cls
            if isinstance(cls, Ref) and cls.oid == "collections.defaultdict":
                store = obj.extra.setdefault("items", {})
                if is_hashable(key) and key in store:
                    return store[key]
                fac = obj.extra.get("factory")
                if fac is None or fac is None.__class__:
                    raise SimError("KeyError")
                v = self.call(fac, (), {}, "defaultdict default_factory")
                if is_hashable(key):
                    store[key] = v
                return v
            if isinstance(cls, ClassObj):
                gi = self.classobj_raw(cls, "__getitem__")
                if gi is not NOTFOUND:
                    return self.call(gi, (obj, key), {}, "__getitem__")
            store = obj.extra.get("items")
            if store is not None and is_hashable(key) and key in store:
                return store[key]
            return Opaque(f"item of {describe(obj)}")
        if isinstance(obj, Ref) and obj.kind in ("c", "?", "x", "i", "bm"):
            # classes (generic aliases) and special objects such as typing.Optional
            return Opaque(f"{obj.oid}[...]")
        if isinstance(obj, Opaque):
            if obj.attrs and ("item", key) in obj.attrs if is_hashable(key) else False:
                return obj.attrs[("item", key)]
            return Opaque(f"item of {obj.desc}", obj.tainted)
        raise SimError(f"{describe(obj)} is not subscriptable")

    @staticmethod
    def _keyerr(key):
        raise SimError(f"KeyError {short(key)}")

    def setitem(self, obj, key, value):
        self.ctx.tick()
        if isinstance(obj, PDict):
            if not is_hashable(key):
                raise SimError("unhashable key")
            obj.d[key] = value
            return
        if isinstance(obj, PList):
            if not isinstance(key, int):
                raise SimError("list indices must be integers")
            try:
                obj.items[key] = value
            except IndexError:
                raise SimError("IndexError") from None
            return
        if isinstance(obj, ModuleDict):
            if isinstance(key, str):
                obj.module.overlay[key] = value
                obj.module.deleted.discard(key)
            return
        if isinstance(obj, SysModules):
            if isinstance(key, str):
                obj.overlay[key] = value
                obj.deleted.discard(key)
            return
        if isinstance(obj, PByteArray):
            return
        if isinstance(obj, Instance):
            cls = obj.cls
            if isinstance(cls, ClassObj):
                si = self.classobj_raw(cls, "__setitem__")
                if si is not NOTFOUND:
                    self.call(si, (obj, key, value), {}, "__setitem__")
                    return
            if isinstance(cls, Ref) and cls.kind == "c":
                if pyindex.class_lookup(cls.oid, "__setitem__") is None:
                    raise SimError("object does not support item assignment")
            if is_hashable(key):
                obj.extra.setdefault("items", {})[key] = value
            return
        if isinstance(obj, Opaque):
            if is_hashable(key):
                if obj.attrs is None:
                    obj.attrs = {}
                obj.attrs[("item", key)] = value
            return
        if isinstance(obj, Ref) and obj.kind == "?":
            return
        raise SimError(f"{describe(obj)} does not support item assignment")

    # ------------------------------------------------------------ iteration
    def iterate(self, obj, limit=MAX_ITER):
        """Consume an iterable the way list(obj) would; returns its elements.
        Consuming a lazy iterator runs the calls it was built from."""
        self.ctx.tick()
        if isinstance(obj, (tuple, frozenset)):
            return list(obj)[:limit]
        if isinstance(obj, PList):
            return list(obj.items[:limit])
        if isinstance(obj, PDict):
            return list(obj.d.keys())[:limit]
        if isinstance(obj, PSet):
            return list(obj.s)[:limit]
        if isinstance(obj, str):
            return list(obj[:limit])
        if isinstance(obj, bytes):
            return list(obj[:limit])
        if isinstance(obj, PByteArray):
            return list(obj.b[:limit])
        if isinstance(obj, range):
            return list(obj[:limit])
        if isinstance(obj, LazyIter):
            return self.run_lazy(obj, limit)
        if isinstance(obj, (ModuleDict, SysModules)):
            ov = obj.module.overlay if isinstance(obj, ModuleDict) else obj.overlay
            return list(ov.keys())[:limit] + [Opaque("key")]
        if isinstance(obj, FileObj):
            data = obj.data[obj.pos:]
            obj.pos = len(obj.data)
            return data.splitlines(True)[:limit]
        if isinstance(obj, Instance):
            cls = obj.cls
            if isinstance(cls, ClassObj):
                it = self.classobj_raw(cls, "__iter__")
                if it is not NOTFOUND:
                    return self.iterate(self.call(it, (obj,), {}, "__iter__"), limit)
            if isinstance(obj, Partial) or isinstance(obj, (AttrGetter, ItemGetter, MethodCaller)):
                raise SimError("object is not iterable")
            items = obj.extra.get("iterable")
            if items is not None:
                return list(items)[:limit]
            if isinstance(cls, Ref) and cls.kind == "c" and pyindex.class_lookup(cls.oid, "__iter__") is None \
                    and pyindex.class_lookup(cls.oid, "__getitem__") is None:
                raise SimError("object is not iterable")
            return [Opaque("element")]
        if isinstance(obj, Opaque):
            return [Opaque("element", obj.tainted)]
        if isinstance(obj, Coroutine):
            raise SimError("coroutine is not iterable")
        if isinstance(obj, Ref) and obj.kind == "?":
            return [Opaque("element")]
        raise SimError(f"{describe(obj)} is not iterable")

    def truthy(self, v):
        if is_native(v):
            try:
                return bool(v)
            except Exception:
                return True
        if isinstance(v, PList):
            return bool(v.items)
        if isinstance(v, PDict):
            return bool(v.d)
        if isinstance(v, PSet):
            return bool(v.s)
        if isinstance(v, PByteArray):
            return bool(v.b)
        if isinstance(v, Instance) and not isinstance(v, (Partial, AttrGetter, ItemGetter, MethodCaller)):
            if "items" in v.extra or "iterable" in v.extra:
                return bool(v.extra.get("items")) or bool(v.extra.get("iterable"))
        return True

    def run_lazy(self, it: LazyIter, limit=MAX_ITER):
        if it.consumed:
            return []
        it.consumed = True
        k, p = it.kind, it.params
        out = []
        if k == "list":
            return list(p["items"])[:limit]
        if k == "map":
            srcs = [self.iterate(s, limit) for s in p["iterables"]]
            n = min((len(s) for s in srcs), default=0)
            for i in range(min(n, limit)):
                out.append(self.call(p["func"], tuple(s[i] for s in srcs), {}, "map() consumed"))
            return out
        if k in ("filter", "filterfalse"):
            f = p["func"]
            for x in self.iterate(p["iterable"], limit):
                if f is None:
                    keep = self.truthy(x)
                else:
                    keep = self.truthy(self.call(f, (x,), {}, f"{k}() consumed"))
                if k == "filterfalse":
                    keep = not keep
                if keep:
                    out.append(x)
            return out
        if k == "starmap":
            for x in self.iterate(p["iterable"], limit):
                out.append(self.call(p["func"], tuple(self.iterate(x)), {}, "starmap() consumed"))
            return out
        if k in ("dropwhile", "takewhile"):
            f = p["func"]
            dropping = True
            for x in self.iterate(p["iterable"], limit):
                if k == "takewhile":
                    if not self.truthy(self.call(f, (x,), {}, "takewhile() consumed")):
                        break
                    out.append(x)
                else:
                    if dropping and self.truthy(self.call(f, (x,), {}, "dropwhile() consumed")):
                        continue
                    dropping = False
                    out.append(x)
            return out
        if k == "accumulate":
            f = p.get("func")
            acc = NOTFOUND
            if p.get("initial") is not None:
                acc = p["initial"]
                out.append(acc)
            for x in self.iterate(p["iterable"], limit):
                if acc is NOTFOUND:
                    acc = x
                elif f is None:
                    acc = self.pure_binop("add", acc, x)
                else:
                    acc = self.call(f, (acc, x), {}, "accumulate() consumed")
                out.append(acc)
            return out
        if k == "groupby":
            f = p.get("func")
            for x in self.iterate(p["iterable"], limit):
                key = self.call(f, (x,), {}, "groupby() consumed") if f is not None else x
                out.append((key, LazyIter("list", items=[x])))
            return out
        if k == "zip":
            srcs = [self.iterate(s, limit) for s in p["iterables"]]
            n = min((len(s) for s in srcs), default=0)
            return [tuple(s[i] for s in srcs) for i in range(n)]
        if k == "enumerate":
            start = p.get("start", 0)
            start = start if isinstance(start, int) else 0
            return [(start + i, x) for i, x in enumerate(self.iterate(p["iterable"], limit))]
        if k == "chain":
            for s in p["iterables"]:
                out.extend(self.iterate(s, limit))
            return out[:limit]
        if k == "chain_from":
            for s in self.iterate(p["iterable"], limit):
                out.extend(self.iterate(s, limit))
            return out[:limit]
        if k == "islice":
            src = self.iterate(p["iterable"], limit)
            try:
                return src[slice(*p["args"])]
            except Exception:
                return src
        if k == "reversed":
            return list(reversed(self.iterate(p["seq"], limit)))
        if k == "repeat":
            times = p.get("times")
            n = times if isinstance(times, int) and not isinstance(times, bool) else limit
            return [p["obj"]] * max(0, min(n, 64))
        if k in ("count",):
            return list(range(64))
        if k == "cycle":
            return self.iterate(p["iterable"], limit)
        if k == "callable_iter":
            f, sentinel = p["func"], p["sentinel"]
            for _ in range(min(limit, 64)):
                r = self.call(f, (), {}, "iter(callable, sentinel) consumed")
                if not is_native(r) or not is_native(sentinel):
                    break
                try:
                    if r == sentinel:
                        break
                except Exception:
                    break
                out.append(r)
            return out
        if k == "gen":
            self.call_ref(p["ref"], p["args"], p["kw"], p.get("how", "generator consumed"), lazy_ok=False)
            return [Opaque("element")]
        if k == "pairs":
            return list(p["items"])
        return [Opaque("element")]

    # ------------------------------------------------------------ calls
    def call(self, f, args, kwargs, how="call"):
        self.ctx.tick()
        if not isinstance(args, tuple):
            args = tuple(args)
        if isinstance(f, Ref):
            return self.call_ref(f, args, kwargs, how)
        if isinstance(f, BoundMethod):
            fn = f.func
            if isinstance(fn, Ref):
                return self.call_ref(fn, (f.self_,) + args, kwargs, how, bound=True)
            return self.call(fn, (f.self_,) + args, kwargs, how)
        if isinstance(f, Partial):
            kw = dict(f.kw)
            kw.update(kwargs)
            return self.call(f.func, f.args + args, kw, how + " via functools.partial")
        if isinstance(f, AttrGetter):
            if len(args) != 1 or kwargs:
                raise SimError("attrgetter expects one argument")
            res = []
            for dotted in f.names:
                v = args[0]
                for part in dotted.split("."):
                    v = self.getattr(v, part)
                res.append(v)
            return res[0] if len(res) == 1 else tuple(res)
        if isinstance(f, ItemGetter):
            if len(args) != 1 or kwargs:
                raise SimError("itemgetter expects one argument")
            res = [self.getitem(args[0], k) for k in f.keys]
            return res[0] if len(res) == 1 else tuple(res)
        if isinstance(f, MethodCaller):
            if len(args) != 1 or kwargs:
                raise SimError("methodcaller expects one argument")
            m = self.getattr(args[0], f.name)
            return self.call(m, f.args, f.kw, how + f" via operator.methodcaller({f.name!r})")
        if isinstance(f, ClassObj):
            return self.instantiate_classobj(f, args, kwargs, how)
        if isinstance(f, Instance):
            return self.call_instance(f, args, kwargs, how)
        if isinstance(f, FunctionObj):
            self.ctx.flag("A", f"{how}: calls a function materialised from serialized bytecode")
            return Opaque("result of materialised function")
        if isinstance(f, LazyIter):
            raise SimError("iterator is not callable")
        if isinstance(f, Opaque):
            if f.tainted:
                self.ctx.unsupported(f"{how}: calls an object obtained at run time ({f.desc}) that cannot be "
                                     f"resolved statically")
            return Opaque(f"result of calling {f.desc}", f.tainted)
        if isinstance(f, Coroutine):
            raise SimError("coroutine is not callable")
        if isinstance(f, (Module, ModuleDict, SysModules, ClassDict, CodeObj, FileObj)):
            raise SimError(f"{describe(f)} is not callable")
        raise SimError(f"{describe(f)} object is not callable")

    def call_instance(self, inst: Instance, args, kwargs, how):
        cls = inst.cls
        if isinstance(cls, ClassObj):
            c = self.classobj_raw(cls, "__call__")
            if c is NOTFOUND:
                raise SimError("instance is not callable")
            return self.call(c, (inst,) + tuple(args), kwargs, how + " via __call__")
        if isinstance(cls, Ref):
            fn = self.handlers.get(cls.oid + ".__call__")
            if fn is not None:
                return fn(self, (inst,) + tuple(args), kwargs, how)
            if cls.kind == "c":
                hit = pyindex.class_lookup(cls.oid, "__call__")
                if hit is None:
                    raise SimError(f"{cls.oid} object is not callable")
                r = hit[0]
                if r.startswith("o:"):
                    return self.call_ref(self.ref(r[2:]), (inst,) + tuple(args), kwargs, how, bound=True)
            return Opaque(f"result of calling {describe(inst)}")
        return Opaque("result")

    def instantiate_classobj(self, cls: ClassObj, args, kwargs, how):
        new = self.classobj_raw(cls, "__new__")
        inst = Instance(cls)
        if new is not NOTFOUND and not (isinstance(new, Ref) and new.oid == "builtins.object.__new__"):
            inst = self.call(new, (cls,) + tuple(args), kwargs, how + " via __new__")
            if not (isinstance(inst, Instance) and inst.cls is cls):
                return inst  # __new__ returned something else: __init__ is not called
        init = self.classobj_raw(cls, "__init__")
        if init is not NOTFOUND and not (isinstance(init, Ref) and init.oid == "builtins.object.__init__"):
            self.call(init, (inst,) + tuple(args), kwargs, how + " via __init__")
        return inst

    def check_cap(self, ref: Ref, args, kwargs, how, bound=False):
        """Report ``ref`` if calling it with these arguments carries a
        capability.  Returns True when a finding was recorded."""
        spec = self.caps.get(ref.oid)
        sig_ref = ref
        if spec is None and ref.oid.endswith(".__init__"):
            # Cls.__init__(obj, ...) of a capability-bearing class (a subclass
            # built by the artifact, or an explicit unbound call)
            cls_oid = ref.oid[: -len(".__init__")]
            spec = self.caps.get(cls_oid)
            if spec is not None:
                sig_ref = self.ref(cls_oid) if pyindex.object_entry(cls_oid) else ref
                args = tuple(args[1:])
                bound = False
        if spec is None:
            if ref.kind == "?" or ref.oid.startswith("webbrowser."):
                spec = self._third_party_cap(ref.oid)
            if spec is None:
                return False
        letter, cond = (spec, None) if isinstance(spec, str) else spec
        sig = sig_of(sig_ref) if isinstance(sig_ref, Ref) else None
        b = bind(sig, args, kwargs) if sig is not None else None
        if sig is not None and b is None:
            raise SimError(f"TypeError calling {ref.oid}")
        if cond is not None:
            ctype, pname = cond
            v = b.get(pname, NOTFOUND) if b is not None else (args[0] if args else kwargs.get(pname, NOTFOUND))
            if ctype == "truthy" and (v is NOTFOUND or not self.truthy(v)):
                return False
            if ctype == "not" and v is not NOTFOUND and self.truthy(v):
                return False
            if ctype == "path" and not self.is_pathlike(v):
                return False
            if ctype == "notmem" and isinstance(v, str) and v in (":memory:",):
                return False
            if ctype == "str" and not isinstance(v, (str, bytes)):
                return False
        arg_s = ", ".join(short(a, 40) for a in (args[1:] if bound else args))
        self.ctx.flag(letter, f"{how}: {ref.oid}({arg_s})")
        return True

    def _third_party_cap(self, oid):
        if oid.startswith("webbrowser.") and oid.rsplit(".", 1)[-1] in ("open", "open_new", "open_new_tab"):
            return "D"
        return None

    @staticmethod
    def is_pathlike(v):
        if isinstance(v, (str, bytes)):
            return True
        if isinstance(v, Instance) and isinstance(v.cls, Ref) and v.cls.oid.startswith("pathlib."):
            return True
        if isinstance(v, (FileObj, PByteArray)):
            return False
        if isinstance(v, Opaque):
            return True
        if isinstance(v, int) and not isinstance(v, bool):
            return True
        return False

    def call_ref(self, ref: Ref, args, kwargs, how="call", bound=False, lazy_ok=True):
        oid = ref.oid
        if lazy_ok and ("g" in ref.flags or oid in _LAZY_CAPS):
            return LazyIter("gen", ref=ref, args=tuple(args), kw=dict(kwargs), how=how + " (consumed)")
        if lazy_ok and "a" in ref.flags:
            return Coroutine(ref, tuple(args), dict(kwargs))
        h = self.handlers.get(oid)
        if h is not None:
            try:
                return h(self, tuple(args), dict(kwargs), how)
            except (SimError, Budget, RecursionError, MemoryError):
                raise
            except (IndexError, KeyError, TypeError, ValueError, AttributeError) as exc:
                # argument shapes the model does not expect: the real call raises too
                raise SimError(f"{oid} raised {type(exc).__name__}") from None
        if bound and oid.startswith("builtins.") and args and (is_native(args[0]) or isinstance(args[0], PByteArray)):
            from modelgate.handlers import native_method_call
            r = native_method_call(self, oid, tuple(args), dict(kwargs))
            if r is not NOTFOUND:
                return r
        if self.check_cap(ref, args, kwargs, how, bound):
            return Opaque(f"result of {oid}")
        if ref.kind in ("f", "b", "md"):
            sig = sig_of(ref)
            if sig is not None and bind(sig, args, kwargs) is None:
                raise SimError(f"TypeError: wrong arguments for {oid}")
        if ref.kind == "c":
            return self.instantiate(ref, args, kwargs, how)
        if oid.endswith(".__new__") and args and isinstance(args[0], (ClassObj, Ref)):
            # Base.__new__(cls, ...): a bare instance of cls
            target = args[0]
            if isinstance(target, Ref) and target.kind == "c":
                info = pyindex.class_info(target.oid)
                return Instance(target, d=True if (info is None or info[1]) else None)
            return Instance(target)
        if ref.kind == "?":
            ov = self.ctx.class_overlay.get(oid)
            if ov and "__init__" in ov:
                inst = Instance(ref)
                self.call(ov["__init__"], (inst,) + tuple(args), kwargs, how + " via __init__")
                return inst
            last = oid.rsplit(".", 1)[-1]
            if last[:1].isupper():
                return Instance(ref, extra={"args": tuple(args), "kw": dict(kwargs)})
        return Opaque(f"result of {oid}")

    def instantiate(self, cls: Ref, args, kwargs, how):
        """A full call of an indexed class: ``__new__`` then ``__init__``."""
        ov = self.ctx.class_overlay.get(cls.oid)
        if ov:
            for hook in ("__new__", "__init__"):
                if hook in ov:
                    target = Instance(cls)
                    self.call(ov[hook], ((cls,) if hook == "__new__" else (target,)) + tuple(args), kwargs,
                              how + f" via {hook}")
        info = pyindex.class_info(cls.oid)
        has_dict = info[1] if info else True
        return Instance(cls, d=True if has_dict else None,
                        extra={"args": tuple(args), "kw": dict(kwargs), "constructed": True})

    def new_object(self, cls, args, kwargs, how):
        """``cls.__new__(cls, *args, **kwargs)`` (NEWOBJ / NEWOBJ_EX,
        argument-less INST/OBJ, copyreg.__newobj__)."""
        if isinstance(cls, ClassObj):
            new = self.classobj_raw(cls, "__new__")
            if new is not NOTFOUND and not (isinstance(new, Ref) and new.oid == "builtins.object.__new__"):
                return self.call(new, (cls,) + tuple(args), kwargs, how + " via __new__")
            return Instance(cls)
        if not isinstance(cls, Ref):
            raise SimError("__new__ of a non-class")
        h = self.handlers.get(cls.oid + ".__new__")
        if h is not None:
            return h(self, (cls,) + tuple(args), dict(kwargs), how)
        ov = self.ctx.class_overlay.get(cls.oid)
        if ov and "__new__" in ov:
            return self.call(ov["__new__"], (cls,) + tuple(args), kwargs, how + " via __new__")
        if cls.kind == "c":
            info = pyindex.class_info(cls.oid)
            return Instance(cls, d=True if (info is None or info[1]) else None)
        return Instance(cls)

    def pure_binop(self, name, a, b):
        from modelgate.handlers import make_binop
        return make_binop(name)(self, (a, b), {}, name)
