"""The Python object model the pickle emulator runs against.

``Runtime`` answers the questions the unpickler asks of Python while it
loads: resolve a global, get or set an attribute, call a callable, build an
object, iterate an iterator.  It never imports what an artifact names and
never calls a real function an artifact names: capability-bearing callables
are recognised by identity through the knowledge base and reported when the
loader would call them; indirection gadgets (getattr, partial, map, ...) and
pure data transformations (base64, zlib, str.join, ...) are modelled so that
whatever the loader would end up calling is known.
"""
from __future__ import annotations

import base64
import binascii
import bz2
import codecs
import lzma
import re
import struct
import zlib

from modelgate import kb as kbmod
from modelgate.values import (
    AttrGetter, BoundMethod, BytesIOV, CmpKey, CodeObj, Coroutine, Decorator, Dtype,
    ExtRegistry, Glob, Inst, ItemGetter, Lazy, MethodCaller, ModDict, Mod, NpzObj, ODict,
    Opaque, Partial, Runner, Sym, SysModules, UnpicklerObj, Wrapper, XmlrpcMethod,
    XmlrpcProxy, describe,
)

CODE = "code-execution"
PROC = "process"
FILE = "file-access"
NET = "network"

MAX_CONCRETE = 64 * 1024 * 1024     # largest str/bytes the emulator materialises
MAX_ITER = 100_000                   # elements consumed from one iterator


class PickleStop(Exception):
    """The real loader raises here; nothing after this point runs."""


class ScanAbort(Exception):
    """Analysis budget exhausted; the artifact cannot be analysed."""


HANDLERS = {}


def handler(*paths):
    def deco(fn):
        for p in paths:
            HANDLERS[p] = fn
        return fn
    return deco


def is_concrete(v):
    return v is None or isinstance(v, (bool, int, float, complex, str, bytes, bytearray))


def is_text(v):
    return isinstance(v, (str, bytes, bytearray))


def deep_concrete(v, depth=0):
    if depth > 20:
        return False
    if is_concrete(v) or isinstance(v, (slice, range)):
        return True
    if isinstance(v, (tuple, list, frozenset, set)):
        return all(deep_concrete(x, depth + 1) for x in v)
    if isinstance(v, dict) and not isinstance(v, ODict):
        return all(deep_concrete(k, depth + 1) and deep_concrete(x, depth + 1) for k, x in v.items())
    return False


def class_like(name):
    last = name.rsplit(".", 1)[-1]
    return bool(last) and last[0].isupper()


def unorderable(v):
    """Values '<' and '+' certainly reject (functions, modules, classes, None)."""
    if isinstance(v, Glob):
        return v.kind in ("func", "class")  # an unknown name may be a plain constant
    return isinstance(v, (Mod, Partial, BoundMethod, Wrapper, AttrGetter, ItemGetter,
                          MethodCaller, Lazy, Decorator)) or v is None


def check_ordering(items):
    """sorted()/min()/max() compare elements: raise where CPython raises TypeError."""
    if len(items) < 2:
        return
    if any(unorderable(x) for x in items):
        raise PickleStop("'<' not supported between these instances")
    kinds = set()
    for x in items:
        if isinstance(x, bool) or isinstance(x, (int, float)):
            kinds.add("num")
        elif isinstance(x, (str, bytes, tuple, list)):
            kinds.add(type(x).__name__)
        elif is_concrete(x) or isinstance(x, (dict, set, frozenset)):
            kinds.add(type(x).__name__)
            if isinstance(x, (dict, complex)):
                raise PickleStop("'<' not supported for this type")
        else:
            return  # unknown objects may define ordering
    if len(kinds) > 1:
        raise PickleStop("'<' not supported between instances of different types")


def size_ok(v):
    return not isinstance(v, (str, bytes, bytearray)) or len(v) <= MAX_CONCRETE


_BUILTIN_TYPES = {
    "builtins.str": str, "builtins.bytes": bytes, "builtins.bytearray": bytearray,
    "builtins.int": int, "builtins.float": float, "builtins.complex": complex,
    "builtins.bool": bool, "builtins.list": list, "builtins.tuple": tuple,
    "builtins.dict": dict, "builtins.set": set, "builtins.frozenset": frozenset,
    "builtins.slice": slice, "builtins.range": range, "builtins.NoneType": type(None),
}
_TYPE_NAMES = {v: k for k, v in _BUILTIN_TYPES.items()}


def type_name(v):
    if isinstance(v, ODict):
        return "collections.OrderedDict"
    return _TYPE_NAMES.get(type(v), "builtins.object")


class Runtime:
    def __init__(self, ctx):
        self.ctx = ctx
        self.kb = ctx.kb
        canon = self.kb.canonical
        self.handlers = {}
        for path, fn in HANDLERS.items():
            self.handlers[canon(path)] = fn
            self.handlers.setdefault(path, fn)

    # ------------------------------------------------------------------ util
    def report(self, kind, site, detail):
        self.ctx.report(kind, site, detail)

    def canon(self, path):
        return self.kb.canonical(path)

    def glob(self, path, kind=None):
        name = self.kb.canonical(path)
        if kind is None:
            mod, _, attr = name.rpartition(".")
            desc = self.kb.module_attr(mod, attr) if mod else None
            known = self.kb.attr_kind(mod, attr) if mod else None
            if desc and desc[0] in "cf":
                kind = "class" if desc[0] == "c" else "func"
            elif name in _BUILTIN_TYPES or self.kb.class_info(name) is not None:
                kind = "class"
            elif known is not None:
                kind = "func" if known == "func" else "class"
            else:
                kind = "class" if class_like(name) else "unknown"
        return Glob(name, kind)

    def not_iterable(self, v):
        """True when iter(v) raises for certain (modules, functions, plain classes)."""
        if isinstance(v, Mod):
            return True
        if isinstance(v, Glob):
            if v.kind == "func":
                return True
            if v.kind == "class":
                mod, _, attr = v.name.rpartition(".")
                return self.kb.attr_kind(mod, attr) == "class" or v.name in _BUILTIN_TYPES
        if is_concrete(v) and not isinstance(v, (str, bytes, bytearray)):
            return True
        return False

    def from_desc(self, desc):
        tag, name = desc[0], desc[2:]
        if tag == "m":
            return Mod(name)
        if tag == "c":
            return Glob(name, "class")
        if tag in ("f", "s"):
            return Glob(name, "func")
        return self.special_object(name)

    def special_object(self, name):
        if name == self.canon("sys.modules"):
            return SysModules()
        if name == self.canon("copyreg._inverted_registry"):
            return ExtRegistry("inverted")
        if name == self.canon("copyreg._extension_cache"):
            return ExtRegistry("cache")
        if name == self.canon("copyreg._extension_registry"):
            return ExtRegistry("registry")
        return Glob(name, "obj")

    def module_of(self, name):
        parts = name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            cand = ".".join(parts[:i])
            if self.kb.is_module(cand):
                return self.kb.module_name(cand)
        return parts[0]

    def is_type(self, v):
        if isinstance(v, Glob):
            if v.kind == "class":
                return True
            if v.kind == "unknown":
                return class_like(v.name)
            return False
        return isinstance(v, Opaque)

    def could_be_type(self, v):
        """NEWOBJ's type check: unknown names (e.g. lowercase third-party classes) pass."""
        if isinstance(v, Glob):
            return v.kind in ("class", "unknown")
        return isinstance(v, Opaque)

    # instances of these classes are never callable
    PLAIN_INSTANCES = ("collections.deque", "types.SimpleNamespace", "argparse.Namespace",
                       "builtins.object", "types.ModuleType")

    def is_callable(self, v):
        if isinstance(v, Inst):
            return not (isinstance(v.cls, Glob) and v.cls.name in [self.canon(n) for n in self.PLAIN_INSTANCES])
        if isinstance(v, Opaque):
            return not v.nocall
        if isinstance(v, (Glob, Partial, BoundMethod, Wrapper, Decorator, AttrGetter, ItemGetter,
                          MethodCaller, CmpKey, XmlrpcMethod)):
            return not (isinstance(v, Glob) and v.kind == "obj" and v.name == self.canon("sys.modules"))
        return False

    def capability_of(self, v, depth=0):
        """The capability a value would exercise if called, or None."""
        if depth > 8:
            return None
        if isinstance(v, Glob):
            entry = self.kb.danger(v.name)
            if entry is not None:
                return entry[0]
            loader = self.kb.nested(v.name)
            if loader is not None:
                return CODE
            return None
        if isinstance(v, Partial):
            return self.capability_of(v.func, depth + 1)
        if isinstance(v, (Wrapper, BoundMethod)):
            return self.capability_of(v.func, depth + 1)
        if isinstance(v, Opaque):
            return v.taint
        if isinstance(v, MethodCaller):
            return None
        return None

    def taint_of_args(self, args, kwargs):
        for a in list(args) + list(kwargs.values()):
            k = self.capability_of(a)
            if k:
                return k
        return None

    def tick(self, n=1):
        self.ctx.tick(n)

    # --------------------------------------------------------------- imports
    def import_module(self, name, site):
        if not isinstance(name, str) or not name or name.startswith("."):
            raise PickleStop(f"cannot import module {name!r}")
        if name in self.ctx.sysmods_override:
            return self.ctx.sysmods_override[name]
        top = name.split(".")[0]
        if top in self.kb.py2_only and name not in kbmod.MODULE_ALIASES:
            raise PickleStop(f"no module named {name!r}")
        if name in kbmod.MODULE_ALIASES and kbmod.MODULE_ALIASES[name] in ("builtins",) and name != "builtins":
            # __builtin__/exceptions only exist through fix_imports
            raise PickleStop(f"no module named {name!r}")
        return Mod(self.kb.module_name(name))

    def resolve_dotted(self, name, site, colon_ok=True):
        """pkgutil.resolve_name / pydoc.locate / logging's resolve."""
        if not isinstance(name, str) or not name:
            raise PickleStop("bad name")
        if colon_ok and ":" in name:
            modname, _, rest = name.partition(":")
            obj = self.import_module(modname, site)
            for part in [p for p in rest.split(".") if p]:
                obj = self.getattr(obj, part, site)
            return obj
        parts = name.split(".")
        i = len(parts)
        while i > 1 and not self.kb.is_module(".".join(parts[:i])):
            i -= 1
        obj = self.import_module(".".join(parts[:i]), site)
        for part in parts[i:]:
            obj = self.getattr(obj, part, site)
        return obj

    # ------------------------------------------------------------ attributes
    def getattr(self, obj, name, site):
        self.tick()
        if not isinstance(name, str):
            raise PickleStop("attribute name must be a string")
        if isinstance(obj, Mod):
            return self.module_attr(obj.name, name)
        if isinstance(obj, Glob):
            return self.glob_attr(obj, name, site)
        if isinstance(obj, Inst):
            return self.inst_attr(obj, name)
        if isinstance(obj, Opaque):
            if obj.attrs and name in obj.attrs:
                return obj.attrs[name]
            return Opaque(f"attribute {name} of {obj.what}", obj.taint)
        if isinstance(obj, Partial):
            if name == "func":
                return obj.func
            if name == "args":
                return obj.args
            if name == "keywords":
                return obj.kwargs
            if name == "__call__":
                return obj
            if name == "__dict__":
                return obj.attrs
            if name in obj.attrs:
                return obj.attrs[name]
            if name in ("__setstate__", "__reduce__", "__class__"):
                if name == "__class__":
                    return self.glob("functools.partial", "class")
                return BoundMethod(obj, self.glob("functools.partial." + name, "func"))
            raise PickleStop(f"partial has no attribute {name}")
        if isinstance(obj, Wrapper):
            if name == "__wrapped__":
                return obj.func
            if name == "__call__":
                return obj
            if name == "__func__":
                return obj.func
            return Opaque(f"attribute {name} of {obj.describe()}")
        if isinstance(obj, BoundMethod):
            if name == "__self__":
                return obj.obj
            if name == "__func__":
                return obj.func
            if name == "__call__":
                return obj
            return Opaque(f"attribute {name} of a method")
        if isinstance(obj, (SysModules, ModDict, ExtRegistry)):
            if hasattr(dict, name) and not name.startswith("__") or name in ("__getitem__", "__setitem__", "__contains__"):
                return BoundMethod(obj, Glob("builtins.dict." + name, "func"))
            raise PickleStop(f"dict has no attribute {name}")
        if isinstance(obj, BytesIOV):
            if name in ("read", "getvalue", "seek", "tell", "readline", "getbuffer", "read1", "readinto", "peek"):
                return BoundMethod(obj, Glob("io.BytesIO." + name, "func"))
            return Opaque(f"attribute {name} of BytesIO")
        if isinstance(obj, UnpicklerObj):
            if name == "load":
                return BoundMethod(obj, Glob(self.canon("pickle.Unpickler.load"), "func"))
            return Opaque(f"attribute {name} of Unpickler")
        if isinstance(obj, Runner):
            return BoundMethod(obj, Glob("runner." + name, "func"))
        if isinstance(obj, XmlrpcProxy):
            return XmlrpcMethod(obj, name)
        if isinstance(obj, XmlrpcMethod):
            return XmlrpcMethod(obj.proxy, obj.name + "." + name)
        if isinstance(obj, Lazy):
            if name in ("__next__", "send"):
                return BoundMethod(obj, Glob(self.canon("builtins.next"), "func"))
            if name == "__iter__":
                return BoundMethod(obj, Glob(self.canon("builtins.iter"), "func"))
            raise PickleStop(f"iterator has no attribute {name}")
        if isinstance(obj, NpzObj):
            if name == "files":
                return list(obj.members)
            if name in ("__getitem__", "get"):
                return BoundMethod(obj, Glob("npz.__getitem__", "func"))
            return Opaque("NpzFile attribute")
        if isinstance(obj, (CodeObj, Coroutine, Dtype, CmpKey, Decorator, AttrGetter, ItemGetter, MethodCaller)):
            if isinstance(obj, Dtype) and name in obj.attrs:
                return obj.attrs[name]
            return Opaque(f"attribute {name} of {describe(obj)}")
        # concrete Python data
        return self.concrete_attr(obj, name)

    def concrete_attr(self, obj, name):
        if isinstance(obj, ODict):
            if name in obj.attrs:
                return obj.attrs[name]
            if name == "__dict__":
                return obj.attrs
            if name == "default_factory":
                return obj.factory
        t = dict if isinstance(obj, ODict) else type(obj)
        tname = _TYPE_NAMES.get(t)
        if tname is None:
            raise PickleStop(f"unsupported attribute access on {type(obj).__name__}")
        if name == "__class__":
            return Glob(tname, "class")
        try:
            raw = getattr(t, name)
        except AttributeError:
            raise PickleStop(f"{t.__name__!r} object has no attribute {name!r}") from None
        if callable(raw) and not isinstance(raw, type):
            static = None
            for klass in t.__mro__:
                if name in klass.__dict__:
                    static = klass.__dict__[name]
                    break
            if isinstance(static, (classmethod, staticmethod)) or \
                    type(static).__name__ in ("classmethod_descriptor", "builtin_function_or_method"):
                return Glob(f"{tname}.{name}", "func")  # class/static methods do not bind the instance
            return BoundMethod(obj, Glob(f"{tname}.{name}", "func"))
        try:
            val = getattr(obj, name)
        except Exception as exc:  # noqa: BLE001
            raise PickleStop(str(exc)) from None
        return val if is_concrete(val) else Opaque(f"{tname}.{name}")

    def module_attr(self, module, name):
        ov = self.ctx.mod_overrides.get(module)
        if ov is not None and name in ov:
            return ov[name]
        if name == "__dict__":
            return ModDict(module)
        if name == "__builtins__":
            return ModDict("builtins")
        if name == "__name__":
            return module
        if name in ("__spec__", "__loader__", "__path__", "__file__", "__package__", "__doc__", "__cached__"):
            return Opaque(f"{module}.{name}")
        desc = self.kb.module_attr(module, name)
        if desc is not None:
            return self.from_desc(desc)
        sub = f"{module}.{name}"
        if self.kb.is_module(sub):
            return Mod(self.kb.module_name(sub))
        return self.glob(sub)

    def glob_attr(self, g, name, site):
        cov = self.ctx.cls_overrides.get(g.name)
        if cov is not None and name in cov:
            return cov[name]
        if g.name == self.canon("ctypes.pythonapi"):
            return Opaque(f"Python C API function {name}", CODE)
        if g.name == self.canon("ctypes.cdll") or g.name == self.canon("ctypes.pydll"):
            if name in ("LoadLibrary",):
                return Glob(self.canon("ctypes.LibraryLoader.LoadLibrary"), "func")
            if not name.startswith("_"):
                self.report(CODE, site, f"loads native library {name!r} through {g.name}")
                return Opaque(f"native library {name}", CODE)
        info = self.kb.class_info(g.name)
        if info is not None:
            desc = info["attrs"].get(name)
            if desc is not None:
                return self.from_desc(desc)
        if name == "__call__" and g.kind != "class":
            return g
        if name in ("__name__", "__qualname__"):
            return g.name.rsplit(".", 1)[-1]
        if name == "__module__":
            return self.module_of(g.name)
        if self.is_type(g) or g.kind == "class":
            if name == "__new__":
                return Glob(self.canon("builtins.object.__new__"), "func")
            if name == "__init__":
                return Glob(f"{g.name}.__init__", "func")
            if name == "__subclasses__":
                return BoundMethod(g, Glob("builtins.type.__subclasses__", "func"))
            if name == "__class__":
                return Glob("builtins.type", "class")
            if name == "__dict__":
                return Opaque(f"{g.name}.__dict__")
            if name == "__call__":
                return BoundMethod(g, Glob(self.canon("builtins.type.__call__"), "func"))
            if name in ("__getattribute__", "__setattr__", "__reduce_ex__"):
                return Glob(self.canon("builtins.object." + name), "func")
            if g.name in _BUILTIN_TYPES:
                t = _BUILTIN_TYPES[g.name]
                if not hasattr(t, name):
                    raise PickleStop(f"type object {t.__name__!r} has no attribute {name!r}")
            return self.glob(f"{g.name}.{name}")
        if name == "__self__":
            return Mod(self.module_of(g.name))
        if name == "__globals__":
            return ModDict(self.module_of(g.name))
        if name == "__builtins__":
            return ModDict("builtins")
        if name == "__wrapped__":
            raise PickleStop(f"{g.name} has no attribute __wrapped__")
        if name == "__code__":
            return CodeObj(None)
        if name == "__class__":
            return Glob("builtins.function", "class")
        if name == "__dict__":
            return {}
        return self.glob(f"{g.name}.{name}")

    def inst_attr(self, inst, name):
        if name in inst.attrs:
            return inst.attrs[name]
        if name == "__dict__":
            return inst.attrs
        if name == "__class__":
            return inst.cls
        cls = inst.cls
        if isinstance(cls, Glob):
            cov = self.ctx.cls_overrides.get(cls.name)
            if cov is not None and name in cov:
                return cov[name]
            info = self.kb.class_info(cls.name)
            if info is not None:
                desc = info["attrs"].get(name)
                if desc is not None:
                    val = self.from_desc(desc)
                    if isinstance(val, Glob) and val.kind == "func" and desc[0] != "s":
                        return BoundMethod(inst, val)
                    return val
            if cls.name in (self.canon("types.SimpleNamespace"), self.canon("types.ModuleType")):
                raise PickleStop(f"object has no attribute {name!r}")
            return BoundMethod(inst, Glob(f"{cls.name}.{name}", "func"))
        return Opaque(f"attribute {name} of an instance")

    def setattr(self, obj, name, value, site):
        if not isinstance(name, str):
            raise PickleStop("attribute name must be a string")
        if isinstance(obj, Mod):
            self.ctx.mod_overrides.setdefault(obj.name, {})[name] = value
            return
        if isinstance(obj, Inst):
            inst_dict = obj.attrs
            inst_dict[name] = value
            return
        if isinstance(obj, Glob):
            if self.is_type(obj):
                info = self.kb.class_info(obj.name)
                if obj.name in _BUILTIN_TYPES or (info is not None and not info.get("heap", True)):
                    raise PickleStop(f"cannot set attributes of built-in type {obj.name}")
                self.ctx.cls_overrides.setdefault(obj.name, {})[name] = value
            return
        if isinstance(obj, (Partial, ODict)):
            obj.attrs[name] = value
            return
        if isinstance(obj, (Opaque, Dtype)):
            if isinstance(obj, Dtype):
                obj.attrs[name] = value
            return
        if isinstance(obj, (Wrapper, Runner)):
            return
        raise PickleStop(f"cannot set attribute {name!r} on {describe(obj)}")

    # ------------------------------------------------------------ containers
    def getitem(self, obj, key, site):
        self.tick()
        if isinstance(obj, (dict, ODict)) or (isinstance(obj, Inst) and isinstance(obj.items, dict)):
            self.hashable(key)
        if isinstance(obj, ODict):
            try:
                if key in obj:
                    return obj[key]
            except TypeError as exc:
                raise PickleStop(str(exc)) from None
            if obj.factory is not None:
                val = self.call(obj.factory, (), {}, site, "defaultdict default_factory")
                obj[key] = val
                return val
            raise PickleStop(f"KeyError {describe(key)}")
        if isinstance(obj, (dict, list, tuple, str, bytes, bytearray, range)):
            try:
                val = obj[key]
            except Exception as exc:  # noqa: BLE001
                raise PickleStop(f"{type(exc).__name__}: {exc}") from None
            return val
        if isinstance(obj, SysModules):
            if not isinstance(key, str):
                raise PickleStop("KeyError")
            if key in self.ctx.sysmods_override:
                return self.ctx.sysmods_override[key]
            return self.import_module(key, site)
        if isinstance(obj, ModDict):
            if not isinstance(key, str):
                raise PickleStop("KeyError")
            return self.module_attr(obj.module, key)
        if isinstance(obj, Opaque):
            return Opaque(f"item of {obj.what}", obj.taint)
        if isinstance(obj, Inst):
            if obj.items is not None and isinstance(obj.items, dict):
                try:
                    if key in obj.items:
                        return obj.items[key]
                except TypeError:
                    pass
            return Opaque(f"item of {obj.describe()}")
        if isinstance(obj, NpzObj):
            return self.ctx.npz_member(obj, key, site)
        if isinstance(obj, ExtRegistry):
            return Opaque("registry entry")
        if isinstance(obj, Glob) and self.is_type(obj):
            return Opaque(f"{obj.name}[...]")
        if isinstance(obj, Dtype):
            return Opaque("dtype field")
        raise PickleStop(f"{describe(obj)} is not subscriptable")

    def setitem(self, obj, key, value, site):
        self.tick()
        if isinstance(obj, (dict, ExtRegistry, Inst)):
            self.hashable(key)
        if isinstance(obj, (dict, list, bytearray)):
            try:
                obj[key] = value
            except Exception as exc:  # noqa: BLE001
                raise PickleStop(f"{type(exc).__name__}: {exc}") from None
            return
        if isinstance(obj, ModDict):
            if isinstance(key, str):
                self.ctx.mod_overrides.setdefault(obj.module, {})[key] = value
            return
        if isinstance(obj, SysModules):
            if isinstance(key, str):
                self.ctx.sysmods_override[key] = value
            return
        if isinstance(obj, ExtRegistry):
            if obj.which == "inverted":
                self.ctx.ext_registry[key] = value
            elif obj.which == "cache":
                self.ctx.ext_cache[key] = value
            return
        if isinstance(obj, Inst):
            if obj.items is None:
                obj.items = {}
            try:
                obj.items[key] = value
            except TypeError:
                pass
            return
        if isinstance(obj, Opaque):
            return
        raise PickleStop(f"{describe(obj)} does not support item assignment")

    def hashable(self, key):
        """Raise PickleStop where hashing ``key`` would fail.

        CPython hashes nested tuples recursively in C with no depth guard, so a
        very deep key would crash the interpreter (the real loader dies there);
        measure the depth iteratively before ever calling hash().
        """
        if isinstance(key, (tuple, frozenset)):
            stack = [(key, 1)]
            seen = 0
            while stack:
                node, depth = stack.pop()
                if depth > 2000:
                    raise PickleStop("key nested too deeply to hash (the interpreter would crash)")
                seen += 1
                if seen > 2_000_000:
                    break
                for child in node:
                    if isinstance(child, (tuple, frozenset)):
                        stack.append((child, depth + 1))
        try:
            hash(key)
        except TypeError as exc:
            raise PickleStop(f"unhashable key: {exc}") from None
        except RecursionError:
            raise PickleStop("key too deeply nested") from None

    # ------------------------------------------------------------- iteration
    def iterate(self, value, site, limit=MAX_ITER):
        """Consume an iterable as a builtin consumer would; returns its elements."""
        self.tick()
        if isinstance(value, (list, tuple, str, bytes, bytearray, range, set, frozenset, dict)):
            if isinstance(value, range) and len(value) > limit:
                return list(value[:limit])
            if isinstance(value, (bytes, bytearray)):
                return list(value[:limit])
            if isinstance(value, str):
                return list(value[:limit])
            return list(value)[:limit]
        if isinstance(value, Lazy):
            return self.consume_lazy(value, site, limit)
        if isinstance(value, Inst) and isinstance(value.items, list):
            return list(value.items)[:limit]
        if isinstance(value, (Opaque, Inst)):
            return [Opaque("element of an unknown iterable", getattr(value, "taint", None))]
        if isinstance(value, (SysModules, ModDict, ExtRegistry)):
            return [Opaque("key")]
        if isinstance(value, NpzObj):
            return list(value.members)
        if isinstance(value, BytesIOV):
            return [Opaque("line")]
        if isinstance(value, Glob) and not self.not_iterable(value):
            return [Opaque("element of an unknown iterable")]  # e.g. an Enum class
        raise PickleStop(f"{describe(value)} is not iterable")

    def consume_lazy(self, lz, site, limit):
        if lz.consumed:
            return []  # an iterator is consumed once
        lz.consumed = 1
        kind = lz.kind
        how = f"{kind}() consumed"
        out = []
        if kind in ("map", "starmap", "filter", "filterfalse", "takewhile", "dropwhile"):
            if kind == "map":
                cols = [self.iterate(s, site, limit) for s in lz.sources]
                n = min((len(c) for c in cols), default=0)
                for i in range(n):
                    out.append(self.call(lz.func, tuple(c[i] for c in cols), {}, site, f"map() consumed: calls"))
                return out
            src = self.iterate(lz.sources[0], site, limit) if lz.sources else []
            for x in src:
                if kind == "starmap":
                    out.append(self.call(lz.func, tuple(self.iterate(x, site, 64)), {}, site, "itertools.starmap() consumed: calls"))
                    continue
                if lz.func is None:
                    keep = x
                else:
                    keep = self.call(lz.func, (x,), {}, site, f"{kind}() consumed: calls")
                if kind == "filter":
                    if not (is_concrete(keep) and not keep):
                        out.append(x)
                elif kind == "filterfalse":
                    if not (is_concrete(keep) and keep):
                        out.append(x)
                elif kind == "takewhile":
                    if is_concrete(keep) and not keep:
                        break
                    out.append(x)
                else:
                    out.append(x)
            return out
        if kind == "chain":
            for s in lz.sources:
                out.extend(self.iterate(s, site, limit))
            return out[:limit]
        if kind in ("zip", "zip_longest"):
            cols = [self.iterate(s, site, limit) for s in lz.sources]
            n = min((len(c) for c in cols), default=0) if kind == "zip" else max((len(c) for c in cols), default=0)
            return [tuple(c[i] if i < len(c) else None for c in cols) for i in range(n)]
        if kind == "enumerate":
            start = lz.extra or 0
            return [(start + i, x) for i, x in enumerate(self.iterate(lz.sources[0], site, limit))]
        if kind == "islice":
            n = lz.extra if isinstance(lz.extra, int) else limit
            src = lz.sources[0]
            if isinstance(src, Lazy):
                return self.consume_partial(src, site, n)
            return self.iterate(src, site, limit)[:n]
        if kind == "accumulate":
            items = self.iterate(lz.sources[0], site, limit)
            if not items:
                return []
            acc = items[0]
            out = [acc]
            for x in items[1:]:
                acc = self.call(lz.func, (acc, x), {}, site, "itertools.accumulate() consumed: calls") if lz.func is not None else Opaque("sum")
                out.append(acc)
            return out
        if kind == "groupby":
            items = self.iterate(lz.sources[0], site, limit)
            if lz.func is not None:
                for x in items:
                    self.call(lz.func, (x,), {}, site, "itertools.groupby() consumed: calls key")
            return [(Opaque("key"), Opaque("group")) for _ in items[:1]]
        if kind == "callable_iter":
            for _ in range(3):
                val = self.call(lz.func, (), {}, site, "iter(callable, sentinel) consumed: calls")
                if not is_concrete(val) or val == lz.extra:
                    break
                out.append(val)
            return out
        if kind == "reversed":
            return list(reversed(self.iterate(lz.sources[0], site, limit)))
        if kind in ("cycle", "repeat", "count"):
            src = self.iterate(lz.sources[0], site, limit) if lz.sources else []
            return src[:limit]
        if kind == "iter":
            return self.iterate(lz.sources[0], site, limit)
        if kind == "merge":
            for s in lz.sources:
                out.extend(self.iterate(s, site, limit))
            return out
        # pairwise, batched, compress, tee and other wrappers
        for s in lz.sources:
            out.extend(self.iterate(s, site, limit))
        return out[:limit]

    def consume_partial(self, lz, site, n):
        """Take only the first n elements of a lazy iterator (next(), islice)."""
        if n <= 0:
            return []
        if lz.kind == "map":
            cols = [self.iterate(s, site, n) if not isinstance(s, Lazy) else self.consume_partial(s, site, n)
                    for s in lz.sources]
            m = min((len(c) for c in cols), default=0)
            return [self.call(lz.func, tuple(c[i] for c in cols), {}, site, "map() consumed: calls") for i in range(min(m, n))]
        if lz.kind in ("starmap",):
            src = self.iterate(lz.sources[0], site, n)[:n]
            return [self.call(lz.func, tuple(self.iterate(x, site, 64)), {}, site, "itertools.starmap() consumed: calls") for x in src]
        return self.consume_lazy(lz, site, MAX_ITER)[:n]

    # ----------------------------------------------------------------- calls
    def call(self, f, args, kwargs, site, how="calls"):
        self.tick()
        ctx = self.ctx
        ctx.call_depth += 1
        try:
            if ctx.call_depth > 200:
                raise PickleStop("recursion too deep")
            return self._call(f, tuple(args), dict(kwargs), site, how)
        finally:
            ctx.call_depth -= 1

    def _call(self, f, args, kwargs, site, how):
        if isinstance(f, Glob):
            return self.call_glob(f, args, kwargs, site, how)
        if isinstance(f, Partial):
            return self.call(f.func, f.args + args, {**f.kwargs, **kwargs}, site, how + " functools.partial of")
        if isinstance(f, BoundMethod):
            return self.call_bound(f, args, kwargs, site, how)
        if isinstance(f, Wrapper):
            if f.how == "singledispatch" and not args:
                raise PickleStop("singledispatch function requires at least 1 positional argument")
            return self.call(f.func, args, kwargs, site, how + f" {f.how} wrapper of")
        if isinstance(f, Decorator):
            if len(args) != 1:
                raise PickleStop("decorator expects one argument")
            if f.how in ("lru_cache", "cache"):
                return Wrapper(args[0], "functools.lru_cache")
            return args[0]
        if isinstance(f, AttrGetter):
            if len(args) != 1 or kwargs:
                raise PickleStop("attrgetter expects one argument")
            vals = []
            for nm in f.names:
                obj = args[0]
                for part in nm.split("."):
                    obj = self.getattr(obj, part, site)
                vals.append(obj)
            return vals[0] if len(vals) == 1 else tuple(vals)
        if isinstance(f, ItemGetter):
            if len(args) != 1 or kwargs:
                raise PickleStop("itemgetter expects one argument")
            vals = [self.getitem(args[0], it, site) for it in f.items]
            return vals[0] if len(vals) == 1 else tuple(vals)
        if isinstance(f, MethodCaller):
            if len(args) != 1 or kwargs:
                raise PickleStop("methodcaller expects one argument")
            meth = self.getattr(args[0], f.name, site)
            return self.call(meth, f.args, f.kwargs, site, how + f" methodcaller({f.name!r}) ->")
        if isinstance(f, XmlrpcMethod):
            self.report(NET, site, f"{how} XML-RPC method {f.name!r} on {describe(f.proxy.uri)}")
            return Opaque("XML-RPC result")
        if isinstance(f, CmpKey):
            return Opaque("cmp_to_key object")
        if isinstance(f, Opaque):
            if f.nocall:
                raise PickleStop(f"{f.what} is not callable")
            if f.taint:
                self.report(f.taint, site, f"{how} {f.what}, which may be a capability-bearing callable")
            return Opaque(f"result of calling {f.what}", f.taint or self.taint_of_args(args, kwargs))
        if isinstance(f, Inst):
            return self.call_instance(f, args, kwargs, site, how)
        raise PickleStop(f"{describe(f)} is not callable")

    def call_instance(self, inst, args, kwargs, site, how):
        if not self.is_callable(inst):
            raise PickleStop(f"{inst.describe()} is not callable")
        cls = inst.cls
        if isinstance(cls, Glob):
            cov = self.ctx.cls_overrides.get(cls.name, {})
            target = cov.get("__call__")
            if target is not None:
                return self.call(target, (inst,) + args, kwargs, site, how + " __call__ of")
            info = self.kb.class_info(cls.name)
            if info is not None and "__call__" in info["attrs"]:
                return self.call(self.from_desc(info["attrs"]["__call__"]), (inst,) + args, kwargs, site, how)
        taint = self.instance_taint(inst)
        if taint:
            self.report(taint, site, f"{how} an instance of {describe(cls)} holding a capability-bearing callable")
        return Opaque(f"result of calling {inst.describe()}")

    def instance_taint(self, inst):
        for v in list(inst.attrs.values()) + list((inst.kwargs or {}).values()) + list(inst.args or ()):
            k = self.capability_of(v)
            if k:
                return k
        return None

    def call_bound(self, bm, args, kwargs, site, how):
        obj, func = bm.obj, bm.func
        if isinstance(obj, Runner):
            return self.runner_method(obj, func.name.split(".", 1)[1] if isinstance(func, Glob) else "", args, kwargs, site, how)
        if isinstance(func, Glob):
            name = func.name
            if isinstance(obj, (SysModules, ModDict, ExtRegistry)) and name.startswith("builtins.dict."):
                return self.mapping_method(obj, name[len("builtins.dict."):], args, kwargs, site)
            if isinstance(obj, BytesIOV) and name.startswith("io.BytesIO."):
                return self.bytesio_method(obj, name[len("io.BytesIO."):], args, kwargs, site)
            if isinstance(obj, NpzObj) and name == "npz.__getitem__":
                if not args:
                    raise PickleStop("missing key")
                return self.ctx.npz_member(obj, args[0], site)
            if isinstance(obj, Inst) and name not in self.handlers and self.kb.danger(name) is None \
                    and self.kb.nested(name) is None:
                taint = self.instance_taint(obj)
                if taint:
                    self.report(taint, site, f"{how} method {name} of an instance holding a capability-bearing callable")
                    return Opaque(f"result of {name}")
            return self.call_glob(func, (obj,) + args, kwargs, site, how)
        return self.call(func, (obj,) + args, kwargs, site, how)

    def call_glob(self, g, args, kwargs, site, how):
        name = g.name
        loader = self.kb.nested(name)
        entry = self.kb.danger(name) if loader is None else None
        if (loader is not None or entry is not None) and not self.kb.accepts(name, len(args), kwargs):
            raise PickleStop(f"{name}() rejects these arguments (TypeError before it runs)")
        if loader is not None:
            return self.nested_load(loader, name, args, kwargs, site, how)
        if entry is not None and self.condition_holds(entry[1], args, kwargs):
            self.report(entry[0], site, f"{how} {name}")
            h = self.handlers.get(name)
            if h is not None:
                try:
                    return self.run_handler(h, args, kwargs, site, how, name)
                except PickleStop:
                    return Opaque(f"result of {name}")
            return Inst(g, args, kwargs) if self.is_type(g) else Opaque(f"result of {name}", entry[0])
        if not self.kb.accepts(name, len(args), kwargs):
            raise PickleStop(f"{name}() rejects these arguments")
        h = self.handlers.get(name)
        if h is not None:
            return self.run_handler(h, args, kwargs, site, how, name)
        if self.is_type(g):
            return self.construct(g, args, kwargs, site, how)
        return Opaque(f"result of {name}", self.taint_of_args(args, kwargs))

    def run_handler(self, h, args, kwargs, site, how, name):
        try:
            return h(self, args, kwargs, site, how, name)
        except (PickleStop, ScanAbort):
            raise
        except (TypeError, ValueError, IndexError, KeyError, AttributeError, OverflowError,
                UnicodeError, LookupError, ZeroDivisionError) as exc:
            # the real callable rejects these arguments the same way
            raise PickleStop(f"{name}: {type(exc).__name__}: {exc}") from None

    def construct(self, g, args, kwargs, site, how):
        info = self.kb.class_info(g.name)
        if info is not None:
            init = info["attrs"].get("__init__")
            if init is not None and init.startswith("f:"):
                entry = self.kb.danger(init[2:])
                if entry is not None:
                    cond = entry[1]
                    if isinstance(cond, list) and cond and cond[0] == "shift":
                        cond = cond[1]
                    if self.condition_holds(cond, args, kwargs):
                        self.report(entry[0], site, f"{how} {g.name} (constructor {init[2:]})")
        cov = self.ctx.cls_overrides.get(g.name, {})
        inst = Inst(g, args, kwargs)
        for hook in ("__new__", "__init__"):
            target = cov.get(hook)
            if target is not None:
                self.call(target, ((g,) if hook == "__new__" else (inst,)) + tuple(args), kwargs, site, how + f" {hook} of")
        return inst

    def condition_holds(self, cond, args, kwargs):
        if cond is None:
            return True
        if isinstance(cond, list) and cond and cond[0] == "shift":
            cond = cond[1]
            args = args[1:]
        kind = cond[0]
        if kind == "kw":
            v = kwargs.get(cond[1])
            return v is not None and not (is_concrete(v) and not v)
        idx, kw = cond[1], cond[2]
        present = len(args) > idx or kw in kwargs
        v = args[idx] if len(args) > idx else kwargs.get(kw)
        if kind == "truthy":
            return present and not (is_concrete(v) and not v)
        if kind == "notnone":
            return present and v is not None
        if kind == "path":
            if not present:
                return False
            return not isinstance(v, (BytesIOV, Inst)) or (isinstance(v, Inst) and "path" in describe(v.cls).lower())
        if kind == "str":
            return present and (isinstance(v, (str, Opaque)))
        if kind == "notmemory":
            if not present:
                return False
            if isinstance(v, (str, bytes)):
                s = v.decode("latin-1") if isinstance(v, bytes) else v
                return s not in ("", ":memory:") and not s.startswith("file::memory:")
            return True
        return True

    # ---------------------------------------------------------------- objects
    def new_instance(self, cls, args, kwargs, site, how="__new__"):
        """``cls.__new__(cls, *args, **kwargs)`` (NEWOBJ, NEWOBJ_EX, copyreg.__newobj__)."""
        if not self.could_be_type(cls):
            raise PickleStop(f"__new__ target {describe(cls)} is not a class")
        if isinstance(cls, Opaque):
            return Opaque(f"instance of {cls.what}", cls.taint)
        name = cls.name
        if name == self.canon("functools.partial"):
            if not args or not self.is_callable(args[0]):
                raise PickleStop("the first argument must be callable")
            return Partial(args[0], args[1:], kwargs)
        if name == self.canon("types.MethodType"):
            return h_methodtype(self, args, kwargs, site, how, name)
        if name in ("builtins.tuple", "builtins.frozenset", "builtins.bytes", "builtins.str",
                    "builtins.int", "builtins.float", "builtins.complex", "builtins.bool",
                    "builtins.slice", "builtins.range"):
            return self.call_glob(cls, args, kwargs, site, how)
        if name in ("builtins.list", "builtins.dict", "builtins.set", "builtins.bytearray"):
            return _BUILTIN_TYPES[name]()
        if name == "builtins.type":
            if len(args) != 3:
                raise PickleStop("type.__new__() takes exactly 3 arguments")
            return h_type(self, args, kwargs, site, how, name)
        if name in (self.canon("collections.OrderedDict"), self.canon("collections.Counter"),
                    self.canon("collections.defaultdict")):
            return ODict()
        lazy = self.handlers.get(name)
        if lazy is not None and getattr(lazy, "lazy_ctor", False):
            return self.run_handler(lazy, args, kwargs, site, how, name)
        if name == self.canon("builtins.object") and args:
            raise PickleStop("object.__new__() takes exactly one argument")
        return Inst(cls)

    def instantiate(self, cls, args, site, how):
        """INST / OBJ: argument-less on a class runs __new__ only."""
        if not args and self.is_type(cls):
            return self.new_instance(cls, (), {}, site, how)
        return self.call(cls, args, {}, site, how)

    def lookup_setstate(self, inst):
        if isinstance(inst, Inst):
            if "__setstate__" in inst.attrs:
                return inst.attrs["__setstate__"]
            if isinstance(inst.cls, Glob):
                cov = self.ctx.cls_overrides.get(inst.cls.name, {})
                if "__setstate__" in cov:
                    return cov["__setstate__"]
                info = self.kb.class_info(inst.cls.name)
                if info is not None and "__setstate__" in info["attrs"]:
                    return BoundMethod(inst, self.from_desc(info["attrs"]["__setstate__"]))
            return None
        if isinstance(inst, Mod):
            ov = self.ctx.mod_overrides.get(inst.name, {})
            return ov.get("__setstate__")
        if isinstance(inst, Partial):
            if "__setstate__" in inst.attrs:
                return inst.attrs["__setstate__"]
            return BoundMethod(inst, self.glob("functools.partial.__setstate__", "func"))
        if isinstance(inst, Glob) and self.is_type(inst):
            cov = self.ctx.cls_overrides.get(inst.name, {})
            if "__setstate__" in cov:
                return cov["__setstate__"]
        return None

    def build(self, inst, state, site, python=False):
        """BUILD: ``inst.__setstate__(state)`` or update ``inst.__dict__``."""
        self.tick()
        setstate = self.lookup_setstate(inst)
        if setstate is not None:
            self.call(setstate, (state,), {}, site, "BUILD calls __setstate__")
            return
        if isinstance(inst, (Opaque, Dtype, Wrapper, Runner, CodeObj, Lazy, BoundMethod, Decorator,
                             AttrGetter, ItemGetter, MethodCaller, XmlrpcProxy, UnpicklerObj, BytesIOV)):
            if isinstance(inst, Dtype) and isinstance(state, dict):
                inst.attrs.update({k: v for k, v in state.items() if isinstance(k, str)})
            return
        if isinstance(inst, Inst) and not self._default_setstate_only(inst):
            # the class may define its own __setstate__ taking any state; keep what
            # a dict-updating one would record so later lookups still see it
            if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
                for k, v in state[1].items():
                    if isinstance(k, str):
                        inst.attrs[k] = v
                state = state[0]
            if isinstance(state, dict):
                for k, v in state.items():
                    try:
                        inst.attrs[k] = v
                    except TypeError:
                        pass
            elif state is not None:
                inst.attrs.setdefault("__state__", state)
            return
        slotstate = None
        if isinstance(state, tuple) and len(state) == 2:
            state, slotstate = state
        if (state if python else state is not None):
            if not isinstance(state, dict):
                raise PickleStop("state is not a dictionary")
            target = self.instance_dict(inst)
            for k, v in state.items():
                target[k] = v
        if (slotstate if python else slotstate is not None):
            if not isinstance(slotstate, dict):
                raise PickleStop("slot state is not a dictionary")
            for k, v in slotstate.items():
                self.setattr(inst, k, v, site)

    def _default_setstate_only(self, inst):
        """True when the instance's class is known to have no custom __setstate__."""
        cls = inst.cls
        if not isinstance(cls, Glob):
            return False
        return cls.name in [self.canon(n) for n in self.PLAIN_INSTANCES]

    def instance_dict(self, inst):
        if isinstance(inst, Inst):
            return inst.attrs
        if isinstance(inst, Mod):
            return self.ctx.mod_overrides.setdefault(inst.name, {})
        if isinstance(inst, (Partial, ODict)):
            return inst.attrs
        if isinstance(inst, Glob):
            if self.is_type(inst):
                raise PickleStop("'mappingproxy' object does not support item assignment")
            return {}
        raise PickleStop(f"{describe(inst)} has no __dict__")

    def append(self, target, items, site):
        """APPEND/APPENDS on a non-list object: extend(), else append() per item."""
        if isinstance(target, list):
            target.extend(items)
            return
        if isinstance(target, (Opaque,)):
            return
        if isinstance(target, Inst):
            for name in ("extend", "append"):
                meth = target.attrs.get(name)
                if meth is None and isinstance(target.cls, Glob):
                    meth = self.ctx.cls_overrides.get(target.cls.name, {}).get(name)
                if meth is not None:
                    if name == "extend":
                        self.call(meth, (list(items),), {}, site, "APPENDS calls extend")
                    else:
                        for it in items:
                            self.call(meth, (it,), {}, site, "APPENDS calls append")
                    return
            if target.items is None:
                target.items = []
            if isinstance(target.items, list):
                target.items.extend(items)
            return
        if isinstance(target, bytearray):
            for it in items:
                if not isinstance(it, int) or not 0 <= it < 256:
                    raise PickleStop("bytearray append needs ints")
                target.append(it)
            return
        if isinstance(target, Mod):
            ov = self.ctx.mod_overrides.get(target.name, {})
            for name in ("extend", "append"):
                if name in ov:
                    if name == "extend":
                        self.call(ov[name], (list(items),), {}, site, "APPENDS calls extend")
                    else:
                        for it in items:
                            self.call(ov[name], (it,), {}, site, "APPENDS calls append")
                    return
        raise PickleStop(f"{describe(target)} has no append/extend")

    def additems(self, target, items, site):
        if isinstance(target, set):
            for it in items:
                self.hashable(it)
                target.add(it)
            return
        if isinstance(target, Opaque):
            return
        if isinstance(target, Inst):
            meth = target.attrs.get("add")
            if meth is not None:
                for it in items:
                    self.call(meth, (it,), {}, site, "ADDITEMS calls add")
            return
        raise PickleStop(f"{describe(target)} has no add")

    # --------------------------------------------------------- nested loads
    def nested_load(self, loader, name, args, kwargs, site, how):
        ctx = self.ctx

        def arg(i, kw, default=None):
            if len(args) > i:
                return args[i]
            return kwargs.get(kw, default)

        def undetermined(what):
            self.report(CODE, site, f"{how} {name} on {what} the scanner cannot determine")
            return Opaque(f"result of {name}", CODE)

        def pathlike(v):
            return isinstance(v, (str, bytes)) or (isinstance(v, Inst) and "path" in describe(v.cls).lower())

        if loader in ("pickle-bytes", "pickle-bytes-py", "grammar-loads", "pickle-bytes-ext"):
            if loader == "grammar-loads":
                data = args[-1] if args else kwargs.get("pkl")
            elif loader == "pickle-bytes-ext":
                ext = arg(0, "extension")
                if not (isinstance(ext, str) and ext in ("pickle", "pkl")):
                    return Opaque(f"result of {name}")
                data = arg(1, "data")
            else:
                data = arg(0, "data")
            if isinstance(data, (bytes, bytearray)):
                enc = kwargs.get("encoding", "ASCII")
                return ctx.load_pickle_bytes(bytes(data), python=loader.endswith("-py"),
                                             encoding=enc if isinstance(enc, str) else "ASCII",
                                             site=f"{site} > {name}")
            if isinstance(data, (str, int, float, type(None), list, tuple, dict, BytesIOV, Mod, Glob)):
                raise PickleStop(f"{name}: a bytes-like object is required")
            return undetermined("data")
        if loader in ("pickle-file", "pickle-file-py", "unpickler-load"):
            f = args[0].file if loader == "unpickler-load" and args and isinstance(args[0], UnpicklerObj) else arg(0, "file")
            python = loader.endswith("-py") or (loader == "unpickler-load" and args and isinstance(args[0], UnpicklerObj) and args[0].python)
            if isinstance(f, BytesIOV):
                if f.data is None:
                    return undetermined("an in-memory file")
                enc = kwargs.get("encoding", "ASCII")
                val, end = ctx.load_pickle_stream(f.data, f.pos, python=bool(python),
                                                  encoding=enc if isinstance(enc, str) else "ASCII",
                                                  site=f"{site} > {name}")
                f.pos = end
                return val
            if isinstance(f, (str, bytes, int, float, type(None), list, tuple, dict, Mod)):
                raise PickleStop(f"{name}: file must have read and readline attributes")
            return undetermined("a file object")
        if loader in ("unpickler", "unpickler-py"):
            f = arg(0, "file")
            return UnpicklerObj(f, loader == "unpickler-py")
        if loader in ("torch", "torch-bytes"):
            f = arg(0, "f") if loader == "torch" else arg(0, "b")
            if loader == "torch":
                # torch >= 2.6: weights_only defaults to True unless a pickle_module
                # is given; the weights-only unpickler refuses arbitrary globals.
                # (torch.storage._load_from_bytes always passes weights_only=False.)
                wo = kwargs.get("weights_only")
                if wo is None:
                    wo = arg(2, "pickle_module") is None
                if is_concrete(wo) and wo and not pathlike(f):
                    return Opaque("weights-only torch.load result")
            if loader == "torch" and pathlike(f):
                self.report(FILE, site, f"{how} {name} on a file path {describe(f)}")
                return Opaque(f"result of {name}", CODE)
            data = f.data if isinstance(f, BytesIOV) else f if isinstance(f, (bytes, bytearray)) else None
            if data is not None:
                if isinstance(f, BytesIOV):
                    data = data[f.pos:]
                return ctx.load_torch_bytes(bytes(data), site=f"{site} > {name}")
            if isinstance(f, (int, float, type(None), list, tuple, dict, Mod)):
                raise PickleStop(f"{name}: invalid file argument")
            return undetermined("data")
        if loader in ("numpy", "npy-read-array"):
            f = arg(0, "file" if loader == "numpy" else "fp")
            allow = arg(2, "allow_pickle", False) if loader == "numpy" else arg(1, "allow_pickle", False)
            allow = not (is_concrete(allow) and not allow)
            if pathlike(f):
                self.report(FILE, site, f"{how} {name} on a file path {describe(f)}")
                return Opaque(f"result of {name}", CODE if allow else None)
            data = f.data[f.pos:] if isinstance(f, BytesIOV) and f.data is not None else None
            if data is not None:
                return ctx.load_numpy_bytes(bytes(data), allow, npz_ok=(loader == "numpy"),
                                            site=f"{site} > {name}")
            if allow and not isinstance(f, (int, float, type(None), list, tuple, dict, Mod)):
                return undetermined("a file object")
            return Opaque(f"result of {name}")
        if loader == "joblib":
            f = arg(0, "filename")
            if pathlike(f):
                self.report(FILE, site, f"{how} {name} on a file path {describe(f)}")
                return Opaque(f"result of {name}", CODE)
            if isinstance(f, BytesIOV) and f.data is not None:
                return ctx.load_joblib_bytes(bytes(f.data[f.pos:]), site=f"{site} > {name}")
            return undetermined("a file object")
        if loader == "pandas":
            f = arg(0, "filepath_or_buffer")
            if pathlike(f):
                self.report(FILE, site, f"{how} {name} on a file path {describe(f)}")
                return Opaque(f"result of {name}", CODE)
            if isinstance(f, BytesIOV) and f.data is not None:
                val, end = ctx.load_pickle_stream(f.data, f.pos, python=False, encoding="ASCII",
                                                  site=f"{site} > {name}")
                f.pos = end
                return val
            return undetermined("a file object")
        if loader == "keras":
            f = arg(0, "filepath")
            if pathlike(f):
                self.report(FILE, site, f"{how} {name} on a file path {describe(f)}")
            return Opaque(f"result of {name}")
        if loader == "shelf":
            self.report(FILE, site, f"{how} {name}")
            return Opaque(f"result of {name}")
        return undetermined("input")

    # ----------------------------------------------------- special receivers
    def mapping_method(self, obj, meth, args, kwargs, site):
        if meth in ("get", "__getitem__", "pop", "setdefault"):
            if not args:
                raise PickleStop("missing key")
            try:
                return self.getitem(obj, args[0], site)
            except PickleStop:
                if meth in ("get", "pop", "setdefault") and len(args) > 1:
                    if meth == "setdefault":
                        self.setitem(obj, args[0], args[1], site)
                    return args[1]
                if meth == "get":
                    return None
                raise
        if meth in ("__setitem__",):
            if len(args) != 2:
                raise PickleStop("__setitem__ expects 2 arguments")
            self.setitem(obj, args[0], args[1], site)
            return None
        if meth == "update":
            src = args[0] if args else {}
            pairs = list(src.items()) if isinstance(src, dict) else []
            pairs += list(kwargs.items())
            for k, v in pairs:
                self.setitem(obj, k, v, site)
            return None
        return Opaque(f"dict.{meth} result")

    def bytesio_method(self, bio, meth, args, kwargs, site):
        if bio.data is None:
            return Opaque("BytesIO data")
        if meth in ("getvalue", "getbuffer"):
            return bio.data
        if meth in ("read", "read1"):
            n = args[0] if args and isinstance(args[0], int) and args[0] >= 0 else None
            out = bio.data[bio.pos:] if n is None else bio.data[bio.pos:bio.pos + n]
            bio.pos += len(out)
            return out
        if meth == "readline":
            i = bio.data.find(b"\n", bio.pos)
            end = len(bio.data) if i < 0 else i + 1
            out = bio.data[bio.pos:end]
            bio.pos = end
            return out
        if meth == "seek":
            if args and isinstance(args[0], int):
                whence = args[1] if len(args) > 1 and isinstance(args[1], int) else 0
                base = 0 if whence == 0 else bio.pos if whence == 1 else len(bio.data)
                bio.pos = max(0, base + args[0])
            return bio.pos
        if meth == "tell":
            return bio.pos
        return Opaque("BytesIO result")

    def runner_method(self, runner, meth, args, kwargs, site, how):
        what = runner.what
        if meth in ("start", "run", "timeit", "repeat", "autorange") and what in ("thread", "timer", "process"):
            if what == "process" and meth == "start":
                self.report(PROC, site, f"{how} multiprocessing.Process.start")
            if runner.target is not None:
                self.call(runner.target, runner.args, runner.kwargs, site, f"{how} {what}.{meth} -> target")
            return None
        if what == "executor" and meth in ("submit", "apply", "apply_async", "map", "starmap", "map_async",
                                             "imap", "imap_unordered", "starmap_async"):
            if not args:
                raise PickleStop("missing callable")
            fn, rest = args[0], args[1:]
            if meth in ("submit", "apply", "apply_async"):
                if meth in ("apply", "apply_async") and rest:
                    rest = tuple(self.iterate(rest[0], site, 64))
                self.call(fn, rest, kwargs if meth == "submit" else {}, site, f"{how} executor.{meth} ->")
            elif meth in ("starmap", "starmap_async"):
                for item in (self.iterate(rest[0], site) if rest else []):
                    self.call(fn, tuple(self.iterate(item, site, 64)), {}, site, f"{how} executor.{meth} ->")
            else:
                cols = [self.iterate(s, site) for s in rest]
                n = min((len(c) for c in cols), default=0)
                for i in range(n):
                    self.call(fn, tuple(c[i] for c in cols), {}, site, f"{how} executor.{meth} ->")
            return Opaque("future")
        if what == "scheduler":
            if meth in ("enter", "enterabs"):
                action = args[2] if len(args) > 2 else kwargs.get("action")
                argument = args[3] if len(args) > 3 else kwargs.get("argument", ())
                kw = args[4] if len(args) > 4 else kwargs.get("kwargs", {})
                runner.queue.append((action, argument, kw))
                return Opaque("event")
            if meth == "run":
                for action, argument, kw in runner.queue:
                    self.call(action, tuple(self.iterate(argument, site, 64)), kw if isinstance(kw, dict) else {}, site,
                              f"{how} sched.scheduler.run ->")
                runner.queue = []
            return None
        if what == "loop" and meth in ("run_until_complete",):
            if args and isinstance(args[0], Coroutine):
                self.report(args[0].kind, site, f"{how} event loop runs {args[0].detail}")
            return Opaque("loop result")
        return Opaque(f"{what}.{meth} result")


# =====================================================================
# Handlers.  Signature: (rt, args, kwargs, site, how) -> value
# =====================================================================

def _arg(args, kwargs, i, kw, default=None):
    if len(args) > i:
        return args[i]
    return kwargs.get(kw, default)


def _need_str(v, what):
    if not isinstance(v, str):
        raise PickleStop(f"{what} must be a string, not {describe(v)}")
    return v


@handler("builtins.getattr", "builtins.object.__getattribute__", "builtins.type.__getattribute__",
         "inspect.getattr_static")
def h_getattr(rt, args, kwargs, site, how, name=""):
    if len(args) < 2:
        raise PickleStop("getattr expects at least 2 arguments")
    name = _need_str(args[1], "attribute name")
    try:
        return rt.getattr(args[0], name, site)
    except PickleStop:
        if len(args) > 2:
            return args[2]
        raise


@handler("builtins.setattr", "builtins.object.__setattr__")
def h_setattr(rt, args, kwargs, site, how, name=""):
    if len(args) != 3:
        raise PickleStop("setattr expects 3 arguments")
    rt.setattr(args[0], _need_str(args[1], "attribute name"), args[2], site)
    return None


@handler("builtins.delattr")
def h_delattr(rt, args, kwargs, site, how, name=""):
    if len(args) == 2 and isinstance(args[0], Mod) and isinstance(args[1], str):
        rt.ctx.mod_overrides.get(args[0].name, {}).pop(args[1], None)
    return None


@handler("builtins.hasattr", "builtins.callable", "builtins.isinstance", "builtins.issubclass")
def h_bool(rt, args, kwargs, site, how, name=""):
    return Opaque("bool")


@handler("builtins.vars")
def h_vars(rt, args, kwargs, site, how, name=""):
    if not args:
        return Opaque("locals()", CODE, nocall=True)
    obj = args[0]
    if isinstance(obj, Mod):
        return ModDict(obj.name)
    if isinstance(obj, (Inst, Partial, ODict)):
        return obj.attrs
    if isinstance(obj, Glob):
        return Opaque(f"vars({obj.name})")
    if isinstance(obj, Opaque):
        return Opaque(f"vars of {obj.what}", obj.taint)
    raise PickleStop("vars() argument must have __dict__ attribute")


@handler("builtins.globals", "builtins.locals", "sys._getframe", "inspect.currentframe",
         "gc.get_objects", "gc.get_referrers", "gc.get_referents", "builtins.type.__subclasses__",
         "builtins.object.__subclasses__")
def h_reflective(rt, args, kwargs, site, how, name=""):
    # a dict, list or frame: not callable itself, but what it leads to may be
    return Opaque("reflective access to interpreter objects", CODE, nocall=True)


@handler("builtins.__import__", "importlib.__import__")
def h_import(rt, args, kwargs, site, how, name=""):
    name = _need_str(_arg(args, kwargs, 0, "name"), "module name")
    fromlist = _arg(args, kwargs, 3, "fromlist", ())
    level = _arg(args, kwargs, 4, "level", 0)
    if isinstance(level, int) and level != 0:
        raise PickleStop("relative import outside a package")
    mod = rt.import_module(name, site)
    if fromlist and not (is_concrete(fromlist) and not fromlist):
        return mod
    return rt.import_module(name.split(".")[0], site)


@handler("importlib.import_module", "importlib._bootstrap._gcd_import", "pydoc.safeimport")
def h_import_module(rt, args, kwargs, site, how, name=""):
    name = _need_str(_arg(args, kwargs, 0, "name"), "module name")
    if name.startswith("."):
        pkg = _arg(args, kwargs, 1, "package")
        if not isinstance(pkg, str):
            raise PickleStop("relative import requires a package")
        name = pkg + name
        name = name.replace("..", ".")
    return rt.import_module(name, site)


@handler("importlib.reload")
def h_reload(rt, args, kwargs, site, how, name=""):
    return args[0] if args else None


@handler("pkgutil.resolve_name", "logging.config._resolve", "unittest.mock._importer")
def h_resolve_name(rt, args, kwargs, site, how, name=""):
    name = _need_str(_arg(args, kwargs, 0, "name"), "name")
    return rt.resolve_dotted(name, site, colon_ok=True)


@handler("pydoc.locate")
def h_locate(rt, args, kwargs, site, how, name=""):
    name = _need_str(_arg(args, kwargs, 0, "path"), "path")
    try:
        return rt.resolve_dotted(name, site, colon_ok=False)
    except PickleStop:
        return None


@handler("logging.config.BaseConfigurator.resolve")
def h_cfg_resolve(rt, args, kwargs, site, how, name=""):
    name = _need_str(args[-1] if args else kwargs.get("s"), "name")
    return rt.resolve_dotted(name, site, colon_ok=False)


@handler("unittest.mock._dot_lookup")
def h_dot_lookup(rt, args, kwargs, site, how, name=""):
    if len(args) < 3:
        raise PickleStop("bad args")
    return rt.getattr(args[0], _need_str(args[1], "name"), site)


@handler("operator.attrgetter")
def h_attrgetter(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("attrgetter expected 1 argument")
    for a in args:
        _need_str(a, "attribute name")
    return AttrGetter(args)


h_attrgetter.lazy_ctor = True


@handler("operator.itemgetter")
def h_itemgetter(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("itemgetter expected 1 argument")
    return ItemGetter(args)


h_itemgetter.lazy_ctor = True


@handler("operator.methodcaller")
def h_methodcaller(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("methodcaller needs at least one argument")
    return MethodCaller(_need_str(args[0], "method name"), args[1:], kwargs)


h_methodcaller.lazy_ctor = True


@handler("operator.getitem", "builtins.dict.__getitem__", "builtins.list.__getitem__",
         "builtins.tuple.__getitem__", "builtins.str.__getitem__", "builtins.bytes.__getitem__")
def h_getitem(rt, args, kwargs, site, how, name=""):
    if len(args) != 2:
        raise PickleStop("getitem expects 2 arguments")
    return rt.getitem(args[0], args[1], site)


@handler("operator.setitem", "builtins.dict.__setitem__")
def h_setitem(rt, args, kwargs, site, how, name=""):
    if len(args) != 3:
        raise PickleStop("setitem expects 3 arguments")
    if isinstance(args[0], dict):
        rt.hashable(args[1])
    rt.setitem(args[0], args[1], args[2], site)
    return None


@handler("operator.delitem")
def h_delitem(rt, args, kwargs, site, how, name=""):
    return None


@handler("operator.call")
def h_opcall(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("call expects a callable")
    return rt.call(args[0], args[1:], kwargs, site, how + " operator.call ->")


@handler("builtins.type.__call__")
def h_type_call(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("type.__call__ needs a class")
    return rt.call(args[0], args[1:], kwargs, site, how)


@handler("builtins.object.__new__", "copyreg.__newobj__")
def h_object_new(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("__new__ needs a class")
    return rt.new_instance(args[0], args[1:], kwargs, site, how)


@handler("builtins.type.__new__")
def h_type_new(rt, args, kwargs, site, how, name=""):
    return h_type(rt, args[1:], kwargs, site, how, name)


@handler("copyreg.__newobj_ex__")
def h_newobj_ex(rt, args, kwargs, site, how, name=""):
    if len(args) != 3:
        raise PickleStop("__newobj_ex__ expects 3 arguments")
    cls, a, kw = args
    if not isinstance(a, tuple) or not isinstance(kw, dict):
        raise PickleStop("bad __newobj_ex__ arguments")
    return rt.new_instance(cls, a, kw, site, how)


@handler("copyreg._reconstructor")
def h_reconstructor(rt, args, kwargs, site, how, name=""):
    if len(args) != 3:
        raise PickleStop("_reconstructor expects 3 arguments")
    cls, base, state = args
    if isinstance(base, Glob) and base.name == rt.canon("builtins.object"):
        return rt.new_instance(cls, (), {}, site, how)
    # base.__new__(cls, state); base.__init__(obj, state)
    if isinstance(base, Glob):
        if base.name in ("builtins.list", "builtins.set", "builtins.dict", "builtins.bytearray",
                         rt.canon("collections.deque"), rt.canon("collections.OrderedDict"),
                         rt.canon("collections.Counter")):
            rt.iterate(state, site)
        elif base.name in ("builtins.tuple", "builtins.frozenset", "builtins.str", "builtins.bytes",
                           "builtins.int", "builtins.float", "builtins.complex"):
            rt.call(base, (state,), {}, site, how + " copyreg._reconstructor ->")
        else:
            rt.call(base, (state,), {}, site, how + " copyreg._reconstructor ->")
    elif isinstance(base, (Partial, Opaque)):
        rt.call(base, (state,), {}, site, how + " copyreg._reconstructor ->")
    else:
        raise PickleStop("bad _reconstructor base")
    return Inst(cls) if isinstance(cls, Glob) else Opaque("reconstructed object")


@handler("copyreg.add_extension")
def h_add_extension(rt, args, kwargs, site, how, name=""):
    if len(args) != 3:
        raise PickleStop("add_extension expects 3 arguments")
    module, name, code = args
    if not isinstance(code, int) or not 1 <= code <= 0x7FFFFFFF:
        raise PickleStop("code out of range")
    rt.ctx.ext_registry[code] = (module, name)
    return None


@handler("copyreg.constructor", "copyreg.pickle")
def h_copyreg_misc(rt, args, kwargs, site, how, name=""):
    return None


@handler("functools.partial")
def h_partial(rt, args, kwargs, site, how, name=""):
    if not args or not rt.is_callable(args[0]):
        raise PickleStop("the first argument must be callable")
    func, rest = args[0], args[1:]
    if isinstance(func, Partial) and not func.attrs:
        return Partial(func.func, func.args + tuple(rest), {**func.kwargs, **kwargs})
    return Partial(func, rest, kwargs)


h_partial.lazy_ctor = True


@handler("functools.partial.__setstate__")
def h_partial_setstate(rt, args, kwargs, site, how, name=""):
    if len(args) != 2 or not isinstance(args[0], Partial):
        raise PickleStop("partial.__setstate__ needs a partial")
    p, state = args
    if not isinstance(state, tuple) or len(state) != 4:
        raise PickleStop("argument to __setstate__ must be a tuple")
    func, fargs, kwds, ns = state
    if not rt.is_callable(func) or not isinstance(fargs, tuple) or \
            (kwds is not None and not isinstance(kwds, dict)) or (ns is not None and not isinstance(ns, dict)):
        raise PickleStop("invalid partial state")
    p.func, p.args, p.kwargs = func, fargs, dict(kwds or {})
    if ns:
        p.attrs = dict(ns)
    return None


@handler("functools.partial.__call__")
def h_partial_call(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("partial.__call__ needs self")
    return rt.call(args[0], args[1:], kwargs, site, how)


@handler("functools.reduce")
def h_reduce(rt, args, kwargs, site, how, name=""):
    if len(args) < 2:
        raise PickleStop("reduce expects at least 2 arguments")
    f = args[0]
    items = rt.iterate(args[1], site)
    if len(args) > 2:
        acc = args[2]
    else:
        if not items:
            raise PickleStop("reduce() of empty iterable with no initial value")
        acc, items = items[0], items[1:]
    for x in items:
        acc = rt.call(f, (acc, x), {}, site, how + " functools.reduce ->")
    return acc


@handler("functools.lru_cache")
def h_lru_cache(rt, args, kwargs, site, how, name=""):
    maxsize = _arg(args, kwargs, 0, "maxsize", 128)
    if rt.is_callable(maxsize) and not is_concrete(maxsize):
        return Wrapper(maxsize, "functools.lru_cache")
    if maxsize is None or isinstance(maxsize, int):
        return Decorator("lru_cache")
    raise PickleStop("Expected first argument to be an integer, a callable, or None")


@handler("functools.cache")
def h_cache(rt, args, kwargs, site, how, name=""):
    if len(args) != 1:
        raise PickleStop("cache expects one argument")
    return Wrapper(args[0], "functools.cache")


@handler("functools._lru_cache_wrapper")
def h_lru_wrapper(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("missing function")
    return Wrapper(args[0], "functools._lru_cache_wrapper")


@handler("functools.singledispatch")
def h_singledispatch(rt, args, kwargs, site, how, name=""):
    if len(args) != 1:
        raise PickleStop("singledispatch expects one argument")
    return Wrapper(args[0], "singledispatch")


@handler("functools.wraps")
def h_wraps(rt, args, kwargs, site, how, name=""):
    return Decorator("wraps")


@handler("functools.update_wrapper")
def h_update_wrapper(rt, args, kwargs, site, how, name=""):
    if len(args) < 2:
        raise PickleStop("update_wrapper expects 2 arguments")
    return args[0]


@handler("functools.cmp_to_key")
def h_cmp_to_key(rt, args, kwargs, site, how, name=""):
    if len(args) != 1:
        raise PickleStop("cmp_to_key expects one argument")
    return CmpKey(args[0])


@handler("functools.partialmethod", "functools.cached_property", "builtins.property", "builtins.classmethod",
         "builtins.super")
def h_descriptor(rt, args, kwargs, site, how, name=""):
    # the wrapped function stays reachable (fget, __func__, func)
    return Opaque("descriptor object", rt.taint_of_args(args, kwargs))


@handler("builtins.staticmethod")
def h_staticmethod(rt, args, kwargs, site, how, name=""):
    if len(args) != 1:
        raise PickleStop("staticmethod expects one argument")
    return Wrapper(args[0], "staticmethod")


@handler("contextlib.contextmanager")
def h_contextmanager(rt, args, kwargs, site, how, name=""):
    if len(args) != 1:
        raise PickleStop("contextmanager expects one argument")
    return Wrapper(args[0], "contextlib.contextmanager")


@handler("numpy.vectorize", "numpy.frompyfunc")
def h_vectorize(rt, args, kwargs, site, how, name=""):
    f = _arg(args, kwargs, 0, "pyfunc")
    if f is None:
        return Decorator("identity")
    return Wrapper(f, "numpy.vectorize")


@handler("numpy.apply_along_axis", "numpy.apply_over_axes")
def h_apply_axis(rt, args, kwargs, site, how, name=""):
    f = _arg(args, kwargs, 0, "func1d")
    arr = args[2] if len(args) > 2 else kwargs.get("arr", Opaque("array"))
    rt.call(f, (arr,) + tuple(args[3:]), {}, site, how + " numpy apply ->")
    return Opaque("array")


@handler("numpy.fromfunction")
def h_fromfunction(rt, args, kwargs, site, how, name=""):
    f = _arg(args, kwargs, 0, "function")
    rt.call(f, (Opaque("index array"),), {}, site, how + " numpy.fromfunction ->")
    return Opaque("array")


@handler("numpy.piecewise")
def h_piecewise(rt, args, kwargs, site, how, name=""):
    funcs = _arg(args, kwargs, 2, "funclist")
    for f in (funcs if isinstance(funcs, (list, tuple)) else []):
        if rt.is_callable(f):
            rt.call(f, (Opaque("array"),), {}, site, how + " numpy.piecewise ->")
    return Opaque("array")


@handler("numpy.fromiter")
def h_fromiter(rt, args, kwargs, site, how, name=""):
    if args:
        rt.iterate(args[0], site)
    return Opaque("array")


@handler("types.MethodType")
def h_methodtype(rt, args, kwargs, site, how, name=""):
    if len(args) != 2 or not rt.is_callable(args[0]):
        raise PickleStop("MethodType expects a callable and an object")
    return BoundMethod(args[1], args[0])


@handler("typing.cast")
def h_cast(rt, args, kwargs, site, how, name=""):
    if len(args) != 2:
        raise PickleStop("cast expects 2 arguments")
    return args[1]


@handler("copy.copy", "copy.deepcopy")
def h_copy(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("copy expects an argument")
    v = args[0]
    if isinstance(v, (list, dict, set, bytearray)) and not isinstance(v, ODict):
        return type(v)(v)
    return v


@handler("sys.intern", "os.fspath")
def h_identity(rt, args, kwargs, site, how, name=""):
    if len(args) != 1:
        raise PickleStop("expects one argument")
    v = args[0]
    if isinstance(v, Inst):
        return Opaque("path string")
    return v


@handler("sys.exit")
def h_exit(rt, args, kwargs, site, how, name=""):
    raise PickleStop("SystemExit")


@handler("sys.getrefcount", "builtins.id", "builtins.hash", "os.getenv")
def h_opaque_value(rt, args, kwargs, site, how, name=""):
    return Opaque("value")


@handler("builtins.print")
def h_print(rt, args, kwargs, site, how, name=""):
    return None


@handler("builtins.dir")
def h_dir(rt, args, kwargs, site, how, name=""):
    return [Opaque("name")]


@handler("builtins.type")
def h_type(rt, args, kwargs, site, how, name=""):
    if len(args) == 1:
        v = args[0]
        if isinstance(v, Inst):
            return v.cls
        if isinstance(v, Mod):
            return Glob("builtins.module", "class")
        if isinstance(v, Glob):
            return Glob("builtins.type" if rt.is_type(v) else "builtins.builtin_function_or_method", "class")
        if isinstance(v, Sym):
            return Opaque(f"type of {describe(v)}")
        return Glob(type_name(v), "class")
    if len(args) == 3:
        ns = args[2]
        taint = None
        if isinstance(ns, dict):
            for v in ns.values():
                taint = taint or rt.capability_of(v)
        return Opaque(f"class {args[0]!r} created with type()", taint)
    raise PickleStop("type() takes 1 or 3 arguments")


@handler("types.new_class", "collections.namedtuple")
def h_new_class(rt, args, kwargs, site, how, name=""):
    return Opaque("dynamically created class")


@handler("types.SimpleNamespace")
def h_namespace(rt, args, kwargs, site, how, name=""):
    if args:
        raise PickleStop("SimpleNamespace: no positional arguments expected")
    inst = Inst(rt.glob("types.SimpleNamespace", "class"))
    inst.attrs.update(kwargs)
    return inst


@handler("types.ModuleType")
def h_moduletype(rt, args, kwargs, site, how, name=""):
    return Inst(rt.glob("types.ModuleType", "class"))


@handler("builtins.object")
def h_object(rt, args, kwargs, site, how, name=""):
    if args or kwargs:
        raise PickleStop("object() takes no arguments")
    return Inst(rt.glob("builtins.object", "class"))


@handler("builtins.object.__reduce_ex__")
def h_reduce_ex(rt, args, kwargs, site, how, name=""):
    return Opaque("reduce tuple")


@handler("marshal.loads", "marshal.load")
def h_marshal(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("marshal.loads expects data")
    return CodeObj(args[0])


def _plain_dumps(obj, protocol=None, *a, **k):
    import pickle
    return pickle.dumps(obj, protocol if isinstance(protocol, int) else None)


def _gzip_compress(data, compresslevel=9, *, mtime=None):
    import gzip
    return gzip.compress(data, compresslevel, mtime=0 if mtime is None else mtime)




@handler("marshal.dumps", "pickle.dumps", "_pickle.dumps")
def h_dumps(rt, args, kwargs, site, how, name=""):
    """Serialising plain data runs no code; anything else stays unknown."""
    if not args or not all(deep_concrete(a) for a in args):
        return Opaque("serialized data")
    try:
        if name.startswith("marshal"):
            import marshal
            return marshal.dumps(args[0])
        return _plain_dumps(*args)
    except Exception as exc:  # noqa: BLE001
        raise PickleStop(f"{name}: {exc}") from None


@handler("types.FunctionType", "types.CodeType", "builtins.compile")
def h_code_object(rt, args, kwargs, site, how, name=""):
    # reported as code execution by the danger table before reaching here
    return CodeObj("materialised") if name.endswith(("CodeType", "compile")) else Wrapper(Glob(rt.canon("builtins.exec"), "func"), "materialised bytecode")


@handler("io.BytesIO")
def h_bytesio(rt, args, kwargs, site, how, name=""):
    init = _arg(args, kwargs, 0, "initial_bytes", b"")
    if init is None:
        init = b""
    if isinstance(init, (bytes, bytearray)):
        return BytesIOV(bytes(init))
    if isinstance(init, (str, int, float, list, tuple, dict)):
        raise PickleStop("a bytes-like object is required")
    return BytesIOV(None)


h_bytesio.lazy_ctor = False


@handler("io.StringIO", "io.TextIOWrapper", "io.BufferedReader")
def h_textio(rt, args, kwargs, site, how, name=""):
    if args and isinstance(args[0], BytesIOV):
        return args[0]
    return Opaque("text stream")


# ----------------------------------------------------------- builtin types
def _consume_arg(rt, args, site):
    return rt.iterate(args[0], site) if args else []


@handler("builtins.list")
def h_list(rt, args, kwargs, site, how, name=""):
    if len(args) > 1 or kwargs:
        raise PickleStop("list expected at most 1 argument")
    return list(_consume_arg(rt, args, site))


@handler("builtins.tuple")
def h_tuple(rt, args, kwargs, site, how, name=""):
    if len(args) > 1 or kwargs:
        raise PickleStop("tuple expected at most 1 argument")
    return tuple(_consume_arg(rt, args, site))


@handler("builtins.set", "builtins.frozenset")
def h_set(rt, args, kwargs, site, how, name=""):
    if len(args) > 1 or kwargs:
        raise PickleStop("set expected at most 1 argument")
    items = _consume_arg(rt, args, site)
    for it in items:
        rt.hashable(it)
    return set(items)


@handler("builtins.frozenset")
def h_frozenset(rt, args, kwargs, site, how, name=""):
    if len(args) > 1 or kwargs:
        raise PickleStop("frozenset expected at most 1 argument")
    items = _consume_arg(rt, args, site)
    for it in items:
        rt.hashable(it)
    return frozenset(items)


def _pairs(rt, value, site):
    if isinstance(value, dict):
        return list(value.items())
    out = []
    for item in rt.iterate(value, site):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            out.append((item[0], item[1]))
        elif isinstance(item, (Opaque, Inst)):
            out.append((Opaque("key"), Opaque("value")))
        else:
            raise PickleStop("dictionary update sequence element has wrong length")
    return out


@handler("builtins.dict")
def h_dict(rt, args, kwargs, site, how, name=""):
    d = {}
    if args:
        for k, v in _pairs(rt, args[0], site):
            rt.hashable(k)
            d[k] = v
    d.update(kwargs)
    return d


@handler("collections.OrderedDict", "collections.Counter", "collections.UserDict", "collections.ChainMap")
def h_odict(rt, args, kwargs, site, how, name=""):
    d = ODict()
    if args:
        if "Counter" in name:
            for k in rt.iterate(args[0], site):
                rt.hashable(k)
                d[k] = d.get(k, 0) + 1 if not isinstance(d.get(k, 0), Sym) else d.get(k)
        else:
            for k, v in _pairs(rt, args[0], site):
                rt.hashable(k)
                d[k] = v
    d.update(kwargs)
    return d


@handler("collections.defaultdict")
def h_defaultdict(rt, args, kwargs, site, how, name=""):
    d = ODict()
    if args:
        if args[0] is not None and not rt.is_callable(args[0]):
            raise PickleStop("first argument must be callable or None")
        d.factory = args[0]
        if len(args) > 1:
            for k, v in _pairs(rt, args[1], site):
                rt.hashable(k)
                d[k] = v
    return d


@handler("collections.deque", "collections.UserList")
def h_deque(rt, args, kwargs, site, how, name=""):
    inst = Inst(rt.glob("collections.deque", "class"))
    inst.items = list(_consume_arg(rt, args, site))
    return inst


@handler("array.array")
def h_array(rt, args, kwargs, site, how, name=""):
    if len(args) > 1:
        rt.iterate(args[1], site)
    return Opaque("array")


@handler("builtins.sorted", "heapq.nsmallest", "heapq.nlargest", "builtins.list.sort")
def h_sorted(rt, args, kwargs, site, how, name=""):
    if "heapq" in name:
        items = rt.iterate(args[1], site) if len(args) > 1 else []
    else:
        if not args:
            raise PickleStop("sorted expected 1 argument")
        items = rt.iterate(args[0], site)
    key = kwargs.get("key")
    if key is not None:
        if isinstance(key, CmpKey):
            if len(items) >= 2:
                rt.call(key.func, (items[0], items[1]), {}, site, how + " sort comparison ->")
        else:
            keys = [rt.call(key, (x,), {}, site, how + " sort key ->") for x in items]
            check_ordering(keys)
    else:
        check_ordering(items)
    try:
        if all(isinstance(x, (int, float, str)) for x in items) and len({type(x) for x in items}) <= 1:
            return sorted(items) if key is None else list(items)
    except Exception:  # noqa: BLE001
        pass
    return list(items)


@handler("builtins.min", "builtins.max")
def h_minmax(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("min/max expected at least 1 argument")
    items = list(args) if len(args) > 1 else rt.iterate(args[0], site)
    key = kwargs.get("key")
    if key is not None:
        check_ordering([rt.call(key, (x,), {}, site, how + " key ->") for x in items])
    else:
        check_ordering(items)
    if not items:
        if "default" in kwargs:
            return kwargs["default"]
        raise PickleStop("min/max arg is an empty sequence")
    if len(items) == 1:
        return items[0]
    try:
        if all(isinstance(x, (int, float)) for x in items) or all(isinstance(x, str) for x in items):
            return (min if name.endswith("min") else max)(items) if key is None else items[0]
    except Exception:  # noqa: BLE001
        pass
    return items[0]


@handler("builtins.sum", "builtins.any", "builtins.all", "statistics.mean", "statistics.median",
         "statistics.fmean", "math.fsum", "math.prod", "operator.countOf", "operator.indexOf",
         "operator.contains", "builtins.dict.fromkeys", "builtins.frozenset.union", "builtins.set.union")
def h_consumer(rt, args, kwargs, site, how, name=""):
    if name == "builtins.sum" and args:
        items = rt.iterate(args[0], site)
        start = args[1] if len(args) > 1 else kwargs.get("start", 0)
        for x in [start] + items:
            if isinstance(x, (str, bytes, bytearray)) or unorderable(x) or isinstance(x, (dict, set)):
                raise PickleStop("unsupported operand type(s) for +")
        return Opaque("sum")
    for a in args:
        if isinstance(a, Lazy):
            rt.iterate(a, site)
    if args and not isinstance(args[0], Lazy) and not name.endswith(("contains", "countOf", "indexOf", "fromkeys", "union")):
        rt.iterate(args[0], site)
    return Opaque("aggregate")


@handler("builtins.next")
def h_next(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("next expected at least 1 argument")
    it = args[0]
    if isinstance(it, Lazy):
        got = rt.consume_partial(it, site, 1)
        if got:
            return got[0]
        if len(args) > 1:
            return args[1]
        raise PickleStop("StopIteration")
    if isinstance(it, (Opaque, Inst)):
        return Opaque("next element", getattr(it, "taint", None))
    raise PickleStop(f"{describe(it)} is not an iterator")


@handler("builtins.iter")
def h_iter(rt, args, kwargs, site, how, name=""):
    if len(args) == 2:
        if not rt.is_callable(args[0]):
            raise PickleStop("iter(v, w): v must be callable")
        return Lazy("callable_iter", args[0], (), args[1])
    if len(args) != 1:
        raise PickleStop("iter expected 1 or 2 arguments")
    if isinstance(args[0], Lazy):
        return args[0]
    if is_concrete(args[0]) and not isinstance(args[0], (str, bytes, bytearray)):
        raise PickleStop(f"{describe(args[0])} is not iterable")
    return Lazy("iter", None, (args[0],))


def _lazy(kind, needs_func):
    def make(rt, args, kwargs, site, how, name=""):
        if needs_func:
            if not args:
                raise PickleStop(f"{kind} needs arguments")
            func, sources = args[0], args[1:]
            # the function is only called (and checked) when the iterator is consumed
            if kind in ("map", "starmap", "filter", "filterfalse", "takewhile", "dropwhile") and not sources:
                raise PickleStop(f"{kind} needs an iterable")
            for s in sources:
                if rt.not_iterable(s):
                    raise PickleStop(f"{describe(s)} is not iterable")
            return Lazy(kind, func, sources)
        arity = {"accumulate": (1, 2), "groupby": (1, 2), "islice": (2, 4), "enumerate": (1, 2),
                 "reversed": (1, 1), "compress": (2, 2), "cycle": (1, 1), "tee": (1, 2),
                 "pairwise": (1, 1), "batched": (2, 2), "zip_longest": (0, 99), "zip": (0, 99),
                 "chain": (0, 99), "merge": (0, 99), "product": (0, 99), "permutations": (1, 2),
                 "combinations": (2, 2), "combinations_with_replacement": (2, 2)}.get(kind, (0, 99))
        if not arity[0] <= len(args) <= arity[1]:
            raise PickleStop(f"{kind} expected {arity[0]} to {arity[1]} arguments, got {len(args)}")
        # chain() and heapq.merge() keep their iterables untouched until consumed
        iter_args = {"chain": (), "merge": (), "zip": args, "zip_longest": args, "product": args}.get(kind, args[:1])
        for s in iter_args:
            if rt.not_iterable(s):
                raise PickleStop(f"{describe(s)} is not iterable")
        if kind == "enumerate" and not isinstance(_arg(args, kwargs, 1, "start", 0), int):
            raise PickleStop("enumerate start must be an integer")
        if kind == "accumulate":
            return Lazy(kind, _arg(args, kwargs, 1, "func"), args[:1])
        if kind == "groupby":
            return Lazy(kind, _arg(args, kwargs, 1, "key"), args[:1])
        if kind == "islice":
            n = args[1] if len(args) == 2 else (args[2] if len(args) > 2 else None)
            return Lazy(kind, None, args[:1], n if isinstance(n, int) else None)
        if kind == "enumerate":
            return Lazy(kind, None, args[:1], _arg(args, kwargs, 1, "start", 0))
        if kind == "product":
            for s in args:
                rt.iterate(s, site)
            return Lazy("iter", None, ([],))
        if kind in ("permutations", "combinations", "combinations_with_replacement"):
            if args:
                rt.iterate(args[0], site)
            return Lazy("iter", None, ([],))
        if kind == "reversed":
            if args and isinstance(args[0], (list, tuple, str, bytes, range)):
                return Lazy("iter", None, (list(reversed(args[0])),))
            if args and isinstance(args[0], Lazy):
                raise PickleStop("object is not reversible")
            return Lazy(kind, None, args[:1])
        return Lazy(kind, None, args)
    make.lazy_ctor = True
    return make


for _name, _kind, _fn in (
    ("builtins.map", "map", True), ("builtins.filter", "filter", True),
    ("itertools.starmap", "starmap", True), ("itertools.filterfalse", "filterfalse", True),
    ("itertools.takewhile", "takewhile", True), ("itertools.dropwhile", "dropwhile", True),
    ("builtins.zip", "zip", False), ("itertools.zip_longest", "zip_longest", False),
    ("builtins.enumerate", "enumerate", False), ("builtins.reversed", "reversed", False),
    ("itertools.chain", "chain", False), ("itertools.islice", "islice", False),
    ("itertools.accumulate", "accumulate", False), ("itertools.groupby", "groupby", False),
    ("itertools.compress", "compress", False), ("itertools.cycle", "cycle", False),
    ("itertools.tee", "tee", False), ("itertools.pairwise", "pairwise", False),
    ("itertools.batched", "batched", False), ("itertools.product", "product", False),
    ("itertools.permutations", "permutations", False), ("itertools.combinations", "combinations", False),
    ("itertools.combinations_with_replacement", "combinations_with_replacement", False),
    ("heapq.merge", "merge", False),
):
    HANDLERS[_name] = _lazy(_kind, _fn)


@handler("itertools.chain.from_iterable")
def h_chain_from(rt, args, kwargs, site, how, name=""):
    if len(args) != 1:
        raise PickleStop("from_iterable expects one argument")
    return Lazy("chain", None, rt.iterate(args[0], site))


@handler("itertools.repeat")
def h_repeat(rt, args, kwargs, site, how, name=""):
    n = args[1] if len(args) > 1 and isinstance(args[1], int) else 3
    return Lazy("iter", None, ([args[0]] * max(0, min(n, 1000)),)) if args else Lazy("iter", None, ([],))


@handler("itertools.count")
def h_count(rt, args, kwargs, site, how, name=""):
    return Lazy("iter", None, (list(range(3)),))


# ------------------------------------------------------- threads & co
@handler("threading.Thread")
def h_thread(rt, args, kwargs, site, how, name=""):
    target = _arg(args, kwargs, 1, "target")
    targs = _arg(args, kwargs, 3, "args", ())
    tkw = _arg(args, kwargs, 4, "kwargs", None)
    return Runner("thread", target, rt.iterate(targs, site, 64) if targs is not None else (), tkw)


@handler("threading.Timer")
def h_timer(rt, args, kwargs, site, how, name=""):
    target = _arg(args, kwargs, 1, "function")
    targs = _arg(args, kwargs, 2, "args", None) or ()
    tkw = _arg(args, kwargs, 3, "kwargs", None)
    return Runner("timer", target, rt.iterate(targs, site, 64), tkw)


@handler("multiprocessing.Process")
def h_process(rt, args, kwargs, site, how, name=""):
    target = _arg(args, kwargs, 1, "target")
    targs = _arg(args, kwargs, 3, "args", ())
    tkw = _arg(args, kwargs, 4, "kwargs", None)
    return Runner("process", target, rt.iterate(targs, site, 64) if targs is not None else (), tkw)


@handler("threading.Thread.start", "threading.Thread.run")
def h_thread_start(rt, args, kwargs, site, how, name=""):
    if args and isinstance(args[0], Runner):
        return rt.runner_method(args[0], "start", (), {}, site, how)
    if args and isinstance(args[0], Inst):
        target = args[0].attrs.get("_target")
        if target is not None:
            targs = args[0].attrs.get("_args", ())
            tkw = args[0].attrs.get("_kwargs", {})
            rt.call(target, tuple(rt.iterate(targs, site, 64)) if targs else (), tkw if isinstance(tkw, dict) else {},
                    site, how + " Thread.run -> target")
    return None


@handler("concurrent.futures.ThreadPoolExecutor", "concurrent.futures.thread.ThreadPoolExecutor",
         "multiprocessing.pool.ThreadPool", "multiprocessing.dummy.Pool")
def h_executor(rt, args, kwargs, site, how, name=""):
    return Runner("executor")


@handler("concurrent.futures.ThreadPoolExecutor.submit", "concurrent.futures.Executor.map",
         "multiprocessing.pool.Pool.apply", "multiprocessing.pool.Pool.map",
         "multiprocessing.pool.Pool.apply_async", "multiprocessing.pool.Pool.starmap")
def h_executor_method(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("missing self")
    meth = name.rsplit(".", 1)[-1]
    runner = args[0] if isinstance(args[0], Runner) else Runner("executor")
    return rt.runner_method(runner, meth, args[1:], kwargs, site, how)


@handler("_thread.start_new_thread", "_thread.start_new", "threading._start_new_thread")
def h_start_new_thread(rt, args, kwargs, site, how, name=""):
    if len(args) < 2:
        raise PickleStop("start_new_thread expects 2 arguments")
    if not isinstance(args[1], tuple):
        raise PickleStop("2nd arg must be a tuple")
    kw = args[2] if len(args) > 2 and isinstance(args[2], dict) else {}
    rt.call(args[0], args[1], kw, site, how + " new thread ->")
    return Opaque("thread id")


@handler("atexit.register")
def h_atexit(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("register expects a callable")
    rt.call(args[0], args[1:], kwargs, site, how + " atexit callback ->")
    return args[0]


@handler("weakref.finalize")
def h_finalize(rt, args, kwargs, site, how, name=""):
    if len(args) < 2:
        raise PickleStop("finalize expects an object and a callable")
    rt.call(args[1], args[2:], kwargs, site, how + " weakref.finalize callback ->")
    return Opaque("finalizer")


@handler("sched.scheduler")
def h_sched(rt, args, kwargs, site, how, name=""):
    return Runner("scheduler")


@handler("sched.scheduler.enter", "sched.scheduler.enterabs", "sched.scheduler.run")
def h_sched_method(rt, args, kwargs, site, how, name=""):
    if not args or not isinstance(args[0], Runner):
        return Opaque("sched result")
    meth = name.rsplit(".", 1)[-1]
    return rt.runner_method(args[0], meth, args[1:], kwargs, site, how)


@handler("bdb.Bdb.runcall", "profile.Profile.runcall", "cProfile.Profile.runcall", "trace.Trace.runfunc")
def h_runcall_method(rt, args, kwargs, site, how, name=""):
    if len(args) < 2:
        raise PickleStop("runcall expects self and a callable")
    return rt.call(args[1], args[2:], kwargs, site, how + " runcall ->")


@handler("pdb.runcall")
def h_pdb_runcall(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("runcall expects a callable")
    return rt.call(args[0], args[1:], kwargs, site, how + " pdb.runcall ->")


@handler("contextlib.ExitStack.callback")
def h_exitstack_cb(rt, args, kwargs, site, how, name=""):
    return args[1] if len(args) > 1 else None


@handler("unittest.FunctionTestCase")
def h_functestcase(rt, args, kwargs, site, how, name=""):
    return Inst(rt.glob("unittest.FunctionTestCase", "class"), args, kwargs)


@handler("asyncio.create_subprocess_shell", "asyncio.create_subprocess_exec",
         "asyncio.subprocess.create_subprocess_shell", "asyncio.subprocess.create_subprocess_exec")
def h_async_subprocess(rt, args, kwargs, site, how, name=""):
    return Coroutine(PROC, "asyncio subprocess")


@handler("asyncio.open_connection", "asyncio.start_server")
def h_async_net(rt, args, kwargs, site, how, name=""):
    return Coroutine(NET, "asyncio connection")


@handler("asyncio.run")
def h_async_run(rt, args, kwargs, site, how, name=""):
    if args and isinstance(args[0], Coroutine):
        rt.report(args[0].kind, site, f"{how} asyncio.run of {args[0].detail}")
    elif args and isinstance(args[0], Opaque) and args[0].taint:
        rt.report(args[0].taint, site, f"{how} asyncio.run of {args[0].what}")
    return Opaque("coroutine result")


@handler("asyncio.get_event_loop", "asyncio.new_event_loop")
def h_loop(rt, args, kwargs, site, how, name=""):
    return Runner("loop")


@handler("asyncio.BaseEventLoop.run_until_complete", "asyncio.events.AbstractEventLoop.run_until_complete")
def h_run_until_complete(rt, args, kwargs, site, how, name=""):
    if len(args) > 1 and isinstance(args[1], Coroutine):
        rt.report(args[1].kind, site, f"{how} event loop runs {args[1].detail}")
    return Opaque("loop result")


@handler("xmlrpc.client.ServerProxy")
def h_serverproxy(rt, args, kwargs, site, how, name=""):
    return XmlrpcProxy(_arg(args, kwargs, 0, "uri"))


@handler("timeit.Timer")
def h_timeit_timer(rt, args, kwargs, site, how, name=""):
    stmt = _arg(args, kwargs, 0, "stmt", "pass")
    return Runner("timer", stmt if rt.is_callable(stmt) else None)


@handler("socket.socket")
def h_socket(rt, args, kwargs, site, how, name=""):
    return Inst(rt.glob("socket.socket", "class"), args, kwargs)


@handler("torch._tensor._rebuild_from_type_v2", "torch._utils._rebuild_from_type_v2")
def h_rebuild_from_type(rt, args, kwargs, site, how, name=""):
    if len(args) < 3:
        raise PickleStop("_rebuild_from_type_v2 expects (func, type, args, state)")
    fargs = args[2]
    if not isinstance(fargs, tuple):
        fargs = tuple(rt.iterate(fargs, site, 64))
    return rt.call(args[0], fargs, {}, site, how + " torch _rebuild_from_type_v2 ->")


@handler("cloudpickle.cloudpickle._builtin_type", "cloudpickle.cloudpickle_fast._builtin_type",
         "dill._dill._load_type")
def h_builtin_type(rt, args, kwargs, site, how, name=""):
    tname = _need_str(_arg(args, kwargs, 0, "name"), "type name")
    if tname == "ClassType":
        return Glob("builtins.type", "class")
    return rt.getattr(Mod("types"), tname, site)


@handler("dill._dill._get_attr")
def h_dill_get_attr(rt, args, kwargs, site, how, name=""):
    if len(args) != 2:
        raise PickleStop("_get_attr expects 2 arguments")
    return rt.getattr(args[0], _need_str(args[1], "attribute name"), site)


@handler("dill._dill._import_module")
def h_dill_import(rt, args, kwargs, site, how, name=""):
    return rt.import_module(_need_str(_arg(args, kwargs, 0, "import_name"), "module name"), site)


@handler("numpy.dtype")
def h_np_dtype(rt, args, kwargs, site, how, name=""):
    return Dtype(args[0] if args else None)


# ------------------------------------------------------- pure functions
_CODEC_ALIASES = {
    "utf8": "utf-8", "utf_8": "utf-8", "u8": "utf-8", "utf": "utf-8", "utf-8": "utf-8",
    "ascii": "ascii", "us-ascii": "ascii", "646": "ascii", "us_ascii": "ascii",
    "latin1": "latin-1", "latin-1": "latin-1", "latin_1": "latin-1", "iso-8859-1": "latin-1",
    "iso8859-1": "latin-1", "iso_8859_1": "latin-1", "8859": "latin-1", "cp819": "latin-1", "l1": "latin-1",
    "utf-16": "utf-16", "utf16": "utf-16", "utf_16": "utf-16", "utf-16-le": "utf-16-le", "utf_16_le": "utf-16-le",
    "utf-16-be": "utf-16-be", "utf_16_be": "utf-16-be", "utf-32": "utf-32", "utf_32": "utf-32",
    "utf-32-le": "utf-32-le", "utf-32-be": "utf-32-be",
    "raw_unicode_escape": "raw_unicode_escape", "raw-unicode-escape": "raw_unicode_escape",
    "unicode_escape": "unicode_escape", "unicode-escape": "unicode_escape",
    "base64": "base64", "base_64": "base64", "base64_codec": "base64", "base64-codec": "base64",
    "hex": "hex", "hex_codec": "hex", "hex-codec": "hex",
    "zlib": "zlib", "zip": "zlib", "zlib_codec": "zlib", "zlib-codec": "zlib",
    "bz2": "bz2", "bz2_codec": "bz2", "bz2-codec": "bz2",
    "rot13": "rot13", "rot_13": "rot13", "rot-13": "rot13",
    "uu": "uu", "uu_codec": "uu", "quopri": "quopri", "quotedprintable": "quopri",
    "quoted_printable": "quopri", "quopri_codec": "quopri", "cp1252": "cp1252", "windows-1252": "cp1252",
    "cp437": "cp437", "unicode_internal": None, "escape": "escape",
}

_TEXT_CODECS = {"utf-8", "ascii", "latin-1", "utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le",
                "utf-32-be", "raw_unicode_escape", "unicode_escape", "cp1252", "cp437"}


def bounded_decompress(kind, data, limit=MAX_CONCRETE, wbits=None):
    if kind == "zlib":
        d = zlib.decompressobj(wbits if wbits is not None else zlib.MAX_WBITS)
        out = d.decompress(data, limit + 1)
        if len(out) > limit:
            raise PickleStop("decompressed data too large")
        if not d.eof:
            raise PickleStop("incomplete or truncated zlib stream")
        return out
    if kind == "bz2":
        out = bytearray()
        rest = data
        while rest:
            d = bz2.BZ2Decompressor()
            out += d.decompress(rest, limit + 1 - len(out))
            if len(out) > limit:
                raise PickleStop("decompressed data too large")
            if not d.eof:
                raise PickleStop("truncated bz2 stream")
            rest = d.unused_data
        return bytes(out)
    if kind in ("lzma", "xz"):
        out = bytearray()
        rest = data
        while rest:
            d = lzma.LZMADecompressor()
            out += d.decompress(rest, limit + 1 - len(out))
            if len(out) > limit:
                raise PickleStop("decompressed data too large")
            if not d.eof:
                raise PickleStop("truncated lzma stream")
            rest = d.unused_data
            if rest and not rest.startswith((b"\xfd7zXZ", b"]\x00\x00")):
                break
        return bytes(out)
    if kind == "gzip":
        out = bytearray()
        rest = data
        while rest:
            if rest[:2] != b"\x1f\x8b":
                break
            d = zlib.decompressobj(16 + zlib.MAX_WBITS)
            out += d.decompress(rest, limit + 1 - len(out))
            if len(out) > limit:
                raise PickleStop("decompressed data too large")
            if not d.eof:
                raise PickleStop("truncated gzip stream")
            rest = d.unused_data.lstrip(b"\x00")
        return bytes(out)
    raise PickleStop("unknown compression")


def codec_apply(direction, obj, encoding, errors="strict"):
    if not isinstance(encoding, str):
        raise PickleStop("encoding must be a string")
    enc = _CODEC_ALIASES.get(encoding.strip().lower().replace(" ", "_"))
    if enc is None:
        raise PickleStop(f"unsupported codec {encoding!r}")
    if not isinstance(errors, str):
        errors = "strict"
    try:
        if enc in _TEXT_CODECS:
            if direction == "decode":
                if not isinstance(obj, (bytes, bytearray)):
                    raise PickleStop("decoding needs bytes")
                return bytes(obj).decode(enc, errors)
            if not isinstance(obj, str):
                raise PickleStop("encoding needs str")
            return obj.encode(enc, errors)
        if enc == "rot13":
            if not isinstance(obj, str):
                raise PickleStop("rot13 needs str")
            return codecs.encode(obj, "rot13")
        if enc == "escape":
            if direction == "decode":
                return codecs.escape_decode(bytes(obj) if isinstance(obj, (bytes, bytearray)) else obj.encode("latin-1"))[0]
            return codecs.escape_encode(bytes(obj))[0]
        if isinstance(obj, str):
            obj = obj.encode("ascii") if direction == "decode" else obj.encode("utf-8")
        data = bytes(obj)
        if enc == "base64":
            return base64.decodebytes(data) if direction == "decode" else base64.encodebytes(data)
        if enc == "hex":
            return binascii.unhexlify(data) if direction == "decode" else binascii.hexlify(data)
        if enc == "zlib":
            return bounded_decompress("zlib", data) if direction == "decode" else zlib.compress(data)
        if enc == "bz2":
            return bounded_decompress("bz2", data) if direction == "decode" else bz2.compress(data)
        if enc == "uu":
            return codecs.decode(data, "uu") if direction == "decode" else codecs.encode(data, "uu")
        if enc == "quopri":
            return codecs.decode(data, "quopri") if direction == "decode" else codecs.encode(data, "quopri")
    except PickleStop:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PickleStop(f"codec error: {exc}") from None
    raise PickleStop("unsupported codec")


@handler("codecs.decode", "_codecs.decode")
def h_codecs_decode(rt, args, kwargs, site, how, name=""):
    obj = _arg(args, kwargs, 0, "obj")
    enc = _arg(args, kwargs, 1, "encoding", "utf-8")
    errors = _arg(args, kwargs, 2, "errors", "strict")
    if not is_concrete(obj) or not isinstance(enc, str):
        return Opaque("decoded data")
    return codec_apply("decode", obj, enc, errors)


@handler("codecs.encode", "_codecs.encode")
def h_codecs_encode(rt, args, kwargs, site, how, name=""):
    obj = _arg(args, kwargs, 0, "obj")
    enc = _arg(args, kwargs, 1, "encoding", "utf-8")
    errors = _arg(args, kwargs, 2, "errors", "strict")
    if not is_concrete(obj) or not isinstance(enc, str):
        return Opaque("encoded data")
    return codec_apply("encode", obj, enc, errors)


@handler("codecs.escape_decode", "_codecs.escape_decode")
def h_escape_decode(rt, args, kwargs, site, how, name=""):
    obj = _arg(args, kwargs, 0, "data")
    if isinstance(obj, (bytes, bytearray, str)):
        try:
            return codecs.escape_decode(obj)
        except Exception as exc:  # noqa: BLE001
            raise PickleStop(str(exc)) from None
    return Opaque("decoded data")


def _pure(fn, convert=None):
    def run(rt, args, kwargs, site, how, name=""):
        if not all(deep_concrete(a) for a in args) or not all(deep_concrete(v) for v in kwargs.values()):
            return Opaque("derived data")
        try:
            out = fn(*args, **kwargs)
        except PickleStop:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PickleStop(f"{type(exc).__name__}: {exc}") from None
        if not size_ok(out):
            return Opaque("large data")
        return out
    return run


for _name, _fn in (
    ("base64.b64decode", base64.b64decode), ("base64.b64encode", base64.b64encode),
    ("base64.standard_b64decode", base64.standard_b64decode),
    ("base64.standard_b64encode", base64.standard_b64encode),
    ("base64.urlsafe_b64decode", base64.urlsafe_b64decode),
    ("base64.urlsafe_b64encode", base64.urlsafe_b64encode),
    ("base64.b32decode", base64.b32decode), ("base64.b32encode", base64.b32encode),
    ("base64.b32hexdecode", base64.b32hexdecode), ("base64.b16decode", base64.b16decode),
    ("base64.b16encode", base64.b16encode), ("base64.a85decode", base64.a85decode),
    ("base64.a85encode", base64.a85encode), ("base64.b85decode", base64.b85decode),
    ("base64.b85encode", base64.b85encode), ("base64.decodebytes", base64.decodebytes),
    ("base64.encodebytes", base64.encodebytes), ("base64.decodestring", base64.decodebytes),
    ("base64.encodestring", base64.encodebytes),
    ("binascii.a2b_base64", binascii.a2b_base64), ("binascii.b2a_base64", binascii.b2a_base64),
    ("binascii.unhexlify", binascii.unhexlify), ("binascii.hexlify", binascii.hexlify),
    ("binascii.a2b_hex", binascii.a2b_hex), ("binascii.b2a_hex", binascii.b2a_hex),
    ("binascii.a2b_uu", binascii.a2b_uu), ("binascii.a2b_qp", binascii.a2b_qp),
    ("quopri.decodestring", None),
    ("builtins.bytes.fromhex", bytes.fromhex), ("builtins.bytearray.fromhex", bytearray.fromhex),
    ("builtins.chr", chr), ("builtins.ord", ord), ("builtins.hex", hex), ("builtins.oct", oct),
    ("builtins.bin", bin), ("builtins.abs", abs), ("builtins.round", round), ("builtins.divmod", divmod),
    ("builtins.ascii", ascii),
    ("builtins.int.from_bytes", int.from_bytes),
    ("json.loads", None), ("ast.literal_eval", None),
    ("operator.sub", lambda a, b: a - b), ("operator.neg", lambda a: -a),
    ("operator.xor", lambda a, b: a ^ b), ("operator.and_", lambda a, b: a & b),
    ("operator.or_", lambda a, b: a | b), ("operator.invert", lambda a: ~a),
    ("operator.inv", lambda a: ~a), ("operator.floordiv", lambda a, b: a // b),
    ("operator.truediv", lambda a, b: a / b), ("operator.index", lambda a: a.__index__()),
    ("operator.not_", lambda a: not a), ("operator.truth", lambda a: bool(a)),
    ("operator.eq", lambda a, b: a == b), ("operator.ne", lambda a, b: a != b),
    ("operator.lt", lambda a, b: a < b), ("operator.le", lambda a, b: a <= b),
    ("operator.gt", lambda a, b: a > b), ("operator.ge", lambda a, b: a >= b),
    ("operator.is_", lambda a, b: a is b), ("operator.is_not", lambda a, b: a is not b),
    ("operator.length_hint", lambda a, d=0: len(a)),
    ("builtins.str.upper", str.upper), ("builtins.str.lower", str.lower),
    ("builtins.str.strip", str.strip), ("builtins.str.lstrip", str.lstrip), ("builtins.str.rstrip", str.rstrip),
    ("builtins.str.split", str.split), ("builtins.str.rsplit", str.rsplit), ("builtins.str.swapcase", str.swapcase),
    ("builtins.str.title", str.title), ("builtins.str.capitalize", str.capitalize),
    ("builtins.str.casefold", str.casefold), ("builtins.str.translate", str.translate),
    ("builtins.str.maketrans", str.maketrans), ("builtins.str.removeprefix", str.removeprefix),
    ("builtins.str.removesuffix", str.removesuffix), ("builtins.str.partition", str.partition),
    ("builtins.str.rpartition", str.rpartition), ("builtins.str.splitlines", str.splitlines),
    ("builtins.bytes.decode", None), ("builtins.bytearray.decode", None), ("builtins.str.encode", None),
    ("builtins.bytes.hex", bytes.hex), ("builtins.bytes.strip", bytes.strip),
    ("builtins.bytes.lstrip", bytes.lstrip), ("builtins.bytes.rstrip", bytes.rstrip),
    ("builtins.bytes.translate", bytes.translate), ("builtins.bytes.maketrans", bytes.maketrans),
    ("builtins.bytes.upper", bytes.upper), ("builtins.bytes.lower", bytes.lower),
    ("builtins.bytes.swapcase", bytes.swapcase), ("builtins.bytes.split", bytes.split),
    ("builtins.bytes.removeprefix", bytes.removeprefix), ("builtins.bytes.removesuffix", bytes.removesuffix),
    ("builtins.list.index", list.index), ("builtins.tuple.index", tuple.index),
):
    if _fn is not None:
        HANDLERS[_name] = _pure(_fn)


def _json_loads(s, *a, **k):
    import json
    return json.loads(s)


def _literal_eval(s):
    import ast
    return ast.literal_eval(s)


@handler("builtins.len")
def h_len(rt, args, kwargs, site, how, name=""):
    if len(args) != 1:
        raise PickleStop("len() takes exactly one argument")
    v = args[0]
    if isinstance(v, (str, bytes, bytearray, list, tuple, dict, set, frozenset, range)):
        return len(v)
    if is_concrete(v) or unorderable(v) or isinstance(v, Lazy):
        raise PickleStop(f"object of type {describe(v)} has no len()")
    return Opaque("length")


for _name, _fn in (("zlib.compress", zlib.compress), ("bz2.compress", bz2.compress),
                   ("lzma.compress", lzma.compress), ("gzip.compress", _gzip_compress),
                   ("struct.unpack", struct.unpack), ("binascii.crc32", binascii.crc32)):
    HANDLERS[_name] = _pure(_fn)

HANDLERS["json.loads"] = _pure(_json_loads)
HANDLERS["ast.literal_eval"] = _pure(_literal_eval)
HANDLERS["quopri.decodestring"] = _pure(lambda s, header=False: codecs.decode(s, "quopri"))


@handler("builtins.bytes.decode", "builtins.bytearray.decode")
def h_bytes_decode(rt, args, kwargs, site, how, name=""):
    if not args or not isinstance(args[0], (bytes, bytearray)):
        return Opaque("text")
    enc = _arg(args, kwargs, 1, "encoding", "utf-8")
    errors = _arg(args, kwargs, 2, "errors", "strict")
    if not isinstance(enc, str):
        return Opaque("text")
    out = codec_apply("decode", args[0], enc, errors)
    if not isinstance(out, str):
        raise PickleStop("bytes.decode needs a text codec")
    return out


@handler("builtins.str.encode")
def h_str_encode(rt, args, kwargs, site, how, name=""):
    if not args or not isinstance(args[0], str):
        return Opaque("bytes")
    enc = _arg(args, kwargs, 1, "encoding", "utf-8")
    errors = _arg(args, kwargs, 2, "errors", "strict")
    if not isinstance(enc, str):
        return Opaque("bytes")
    out = codec_apply("encode", args[0], enc, errors)
    if not isinstance(out, bytes):
        raise PickleStop("str.encode needs a text codec")
    return out


def _check_size(n):
    if n > MAX_CONCRETE:
        raise PickleStop("result too large")


@handler("builtins.str.join", "builtins.bytes.join", "builtins.bytearray.join")
def h_join(rt, args, kwargs, site, how, name=""):
    if len(args) != 2:
        raise PickleStop("join expects one argument")
    sep = args[0]
    items = rt.iterate(args[1], site)
    if isinstance(sep, str) and all(isinstance(x, str) for x in items):
        _check_size(sum(len(x) for x in items) + len(sep) * len(items))
        return sep.join(items)
    if isinstance(sep, (bytes, bytearray)) and all(isinstance(x, (bytes, bytearray)) for x in items):
        _check_size(sum(len(x) for x in items) + len(sep) * len(items))
        out = bytes(sep).join(bytes(x) for x in items)
        return bytearray(out) if isinstance(sep, bytearray) else out
    if any(is_concrete(x) and not isinstance(x, (str, bytes, bytearray)) for x in items):
        raise PickleStop("join: sequence item is not a string")
    return Opaque("joined text")


def _binop(op):
    def run(rt, args, kwargs, site, how, name=""):
        if len(args) != 2:
            raise PickleStop("binary operator expects 2 arguments")
        a, b = args
        if unorderable(a) or unorderable(b):
            raise PickleStop("unsupported operand type(s)")  # functions, modules, classes, None
        if not (deep_concrete(a) and deep_concrete(b)):
            if isinstance(a, list) and isinstance(b, list) and op in ("add", "iadd", "concat", "iconcat"):
                if op.startswith("i"):
                    a.extend(b)
                    return a
                return a + b
            if isinstance(a, tuple) and isinstance(b, tuple) and op in ("add", "concat"):
                return a + b
            return Opaque("derived value")
        try:
            if op in ("add", "iadd", "concat", "iconcat"):
                if op in ("concat", "iconcat") and not hasattr(a, "__getitem__"):
                    raise PickleStop("object can't be concatenated")
                if isinstance(a, (str, bytes, bytearray, list, tuple)):
                    _check_size(len(a) + (len(b) if hasattr(b, "__len__") else 0))
                if op == "iadd" and isinstance(a, (list, bytearray)):
                    a += b
                    return a
                return a + b
            if op in ("mul", "imul"):
                seq, n = (a, b) if isinstance(b, int) else (b, a)
                if isinstance(seq, (str, bytes, bytearray, list, tuple)) and isinstance(n, int):
                    _check_size(len(seq) * max(n, 0))
                elif isinstance(a, int) and isinstance(b, int) and (a.bit_length() + b.bit_length()) > 100000:
                    raise PickleStop("number too large")
                return a * b
            if op in ("mod", "imod"):
                if isinstance(a, (str, bytes)):
                    if re.search(r"%[-+ #0]*(\d{6,}|\*)", a if isinstance(a, str) else a.decode("latin-1")):
                        raise PickleStop("format width too large")
                    out = a % b
                    _check_size(len(out))
                    return out
                return a % b
            if op == "pow":
                if isinstance(b, int) and isinstance(a, int) and abs(b) > 10000:
                    raise PickleStop("number too large")
                return a ** b
            if op == "lshift":
                if isinstance(b, int) and b > 100000:
                    raise PickleStop("number too large")
                return a << b
            if op == "rshift":
                return a >> b
        except PickleStop:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PickleStop(f"{type(exc).__name__}: {exc}") from None
        return Opaque("derived value")
    return run


for _name, _op in (("operator.add", "add"), ("operator.iadd", "iadd"), ("operator.concat", "concat"),
                   ("operator.iconcat", "iconcat"), ("operator.mul", "mul"), ("operator.imul", "imul"),
                   ("operator.mod", "mod"), ("operator.imod", "imod"), ("operator.pow", "pow"),
                   ("builtins.pow", "pow"), ("operator.lshift", "lshift"), ("operator.rshift", "rshift"),
                   ("builtins.str.__add__", "add"), ("builtins.bytes.__add__", "add"),
                   ("builtins.bytearray.__add__", "add"), ("builtins.str.__mul__", "mul"),
                   ("builtins.bytes.__mul__", "mul"), ("builtins.str.__mod__", "mod"),
                   ("builtins.bytes.__mod__", "mod")):
    HANDLERS[_name] = _binop(_op)


@handler("builtins.str.replace", "builtins.bytes.replace")
def h_replace(rt, args, kwargs, site, how, name=""):
    if len(args) < 3 or not all(deep_concrete(a) for a in args):
        return Opaque("text")
    s, old, new = args[0], args[1], args[2]
    try:
        count = args[3] if len(args) > 3 else -1
        n = s.count(old) if old else len(s) + 1
        if count >= 0:
            n = min(n, count)
        _check_size(len(s) + n * max(0, len(new) - len(old)))
        return s.replace(old, new, count)
    except PickleStop:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PickleStop(str(exc)) from None


@handler("builtins.str.zfill", "builtins.str.center", "builtins.str.ljust", "builtins.str.rjust",
         "builtins.str.expandtabs")
def h_pad(rt, args, kwargs, site, how, name=""):
    if len(args) >= 2 and isinstance(args[1], int) and args[1] > 1_000_000:
        raise PickleStop("width too large")
    if not all(deep_concrete(a) for a in args):
        return Opaque("text")
    attr = name.rsplit(".", 1)[-1]
    try:
        return getattr(str, attr)(*args)
    except Exception as exc:  # noqa: BLE001
        raise PickleStop(str(exc)) from None


@handler("builtins.str.format", "builtins.str.format_map")
def h_format(rt, args, kwargs, site, how, name=""):
    if not args or not isinstance(args[0], str):
        return Opaque("text")
    fmt = args[0]
    if not all(deep_concrete(a) for a in args[1:]) or not all(deep_concrete(v) for v in kwargs.values()):
        return Opaque("formatted text")
    if re.search(r"\{[^}]*[:!][^}]*\}", fmt) or len(fmt) > 100000:
        return Opaque("formatted text")
    try:
        if name.endswith("format_map"):
            out = fmt.format_map(args[1]) if len(args) > 1 and isinstance(args[1], dict) else Opaque("text")
        else:
            out = fmt.format(*args[1:], **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise PickleStop(str(exc)) from None
    return out


@handler("builtins.format")
def h_builtin_format(rt, args, kwargs, site, how, name=""):
    if not 1 <= len(args) <= 2:
        raise PickleStop("format expected 1 or 2 arguments")
    if isinstance(args[0], (str, int, float)) and (len(args) < 2 or args[1] in ("", "s", "d")):
        return format(*args)
    return Opaque("formatted text")


@handler("builtins.str", "builtins.repr")
def h_str(rt, args, kwargs, site, how, name=""):
    if not args and not kwargs:
        return ""
    obj = _arg(args, kwargs, 0, "object")
    if len(args) > 1 or "encoding" in kwargs:
        enc = _arg(args, kwargs, 1, "encoding", "utf-8")
        errors = _arg(args, kwargs, 2, "errors", "strict")
        if isinstance(obj, (bytes, bytearray)) and isinstance(enc, str):
            out = codec_apply("decode", obj, enc, errors)
            if isinstance(out, str):
                return out
        if isinstance(obj, (str, int, float, type(None), list, tuple, dict, Glob, Mod, Partial)):
            raise PickleStop("decoding to str: need a bytes-like object")
        return Opaque("text")
    if is_concrete(obj):
        if name.endswith("repr"):
            return repr(obj)
        return str(obj)
    if isinstance(obj, Mod):
        return f"<module {obj.name!r}>"
    return Opaque("text")


@handler("builtins.bytes", "builtins.bytearray", "builtins.memoryview")
def h_bytes(rt, args, kwargs, site, how, name=""):
    want = bytearray if name.endswith("bytearray") else bytes
    src = _arg(args, kwargs, 0, "source", b"")
    if not args and not kwargs:
        return want()
    if isinstance(src, str):
        enc = _arg(args, kwargs, 1, "encoding")
        if not isinstance(enc, str):
            raise PickleStop("string argument without an encoding")
        out = codec_apply("encode", src, enc, _arg(args, kwargs, 2, "errors", "strict"))
        return want(out)
    if isinstance(src, (bytes, bytearray)):
        return want(src)
    if isinstance(src, int) and not isinstance(src, bool):
        if src < 0:
            raise PickleStop("negative count")
        _check_size(src)
        return want(src)
    if isinstance(src, (list, tuple, Lazy, range, set, frozenset, dict)):
        items = rt.iterate(src, site)
        if all(isinstance(x, int) and 0 <= x < 256 for x in items):
            return want(items)
        if any(is_concrete(x) and not isinstance(x, int) for x in items):
            raise PickleStop("bytes must be in range(0, 256)")
        return Opaque("bytes")
    if isinstance(src, BytesIOV) and src.data is not None:
        raise PickleStop("cannot convert BytesIO to bytes")
    return Opaque("bytes")


@handler("builtins.int", "builtins.float", "builtins.complex", "builtins.bool")
def h_number(rt, args, kwargs, site, how, name=""):
    t = {"builtins.int": int, "builtins.float": float, "builtins.complex": complex}.get(name, bool)
    if not all(deep_concrete(a) for a in args) or kwargs and not all(deep_concrete(v) for v in kwargs.values()):
        return Opaque("number")
    try:
        if t is int and args and isinstance(args[0], str) and len(args[0]) > 5000:
            raise PickleStop("int string too long")
        return t(*args, **kwargs)
    except PickleStop:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PickleStop(f"{type(exc).__name__}: {exc}") from None


@handler("builtins.slice")
def h_slice(rt, args, kwargs, site, how, name=""):
    if not 1 <= len(args) <= 3:
        raise PickleStop("slice expected 1 to 3 arguments")
    return slice(*args)


@handler("builtins.range")
def h_range(rt, args, kwargs, site, how, name=""):
    if not 1 <= len(args) <= 3 or not all(isinstance(a, int) for a in args):
        if all(deep_concrete(a) for a in args):
            raise PickleStop("range arguments must be integers")
        return Opaque("range")
    try:
        return range(*args)
    except Exception as exc:  # noqa: BLE001
        raise PickleStop(str(exc)) from None


@handler("builtins.int.to_bytes")
def h_to_bytes(rt, args, kwargs, site, how, name=""):
    if not args or not isinstance(args[0], int):
        return Opaque("bytes")
    length = _arg(args, kwargs, 1, "length", 1)
    if isinstance(length, int) and length > MAX_CONCRETE:
        raise PickleStop("too large")
    return _pure(int.to_bytes)(rt, args, kwargs, site, how, name)


@handler("struct.pack")
def h_struct_pack(rt, args, kwargs, site, how, name=""):
    if args and isinstance(args[0], (str, bytes)):
        fmt = args[0] if isinstance(args[0], str) else args[0].decode("latin-1")
        if any(int(n) > 1_000_000 for n in re.findall(r"\d+", fmt)):
            raise PickleStop("struct format too large")
    return _pure(struct.pack)(rt, args, kwargs, site, how, name)


def _decompressor(kind):
    def run(rt, args, kwargs, site, how, name=""):
        data = _arg(args, kwargs, 0, "data")
        if not isinstance(data, (bytes, bytearray)):
            if deep_concrete(data):
                raise PickleStop("a bytes-like object is required")
            return Opaque("decompressed data")
        wbits = _arg(args, kwargs, 1, "wbits", zlib.MAX_WBITS) if kind == "zlib" else None
        if kind == "zlib" and not isinstance(wbits, int):
            raise PickleStop("bad wbits")
        try:
            return bounded_decompress(kind, bytes(data), wbits=wbits)
        except PickleStop:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PickleStop(f"{kind} error: {exc}") from None
    return run


HANDLERS["zlib.decompress"] = _decompressor("zlib")
HANDLERS["gzip.decompress"] = _decompressor("gzip")
HANDLERS["bz2.decompress"] = _decompressor("bz2")
HANDLERS["lzma.decompress"] = _decompressor("lzma")


# --------------------------------------------------------- mutating methods
@handler("builtins.list.append", "builtins.list.extend", "builtins.list.insert", "builtins.list.pop",
         "builtins.list.reverse", "builtins.list.copy", "builtins.bytearray.extend",
         "builtins.bytearray.append", "builtins.set.add", "builtins.set.update", "builtins.set.pop",
         "builtins.dict.get", "builtins.dict.setdefault", "builtins.dict.update", "builtins.dict.pop",
         "builtins.dict.items", "builtins.dict.keys", "builtins.dict.values", "builtins.dict.copy")
def h_container_method(rt, args, kwargs, site, how, name=""):
    if not args:
        raise PickleStop("method needs self")
    obj, rest = args[0], args[1:]
    meth = name.rsplit(".", 1)[-1]
    if isinstance(obj, (SysModules, ModDict, ExtRegistry)):
        return rt.mapping_method(obj, meth, rest, kwargs, site)
    if isinstance(obj, Sym) and not isinstance(obj, ODict):
        if meth in ("extend", "update"):
            for r in rest:
                if isinstance(r, Lazy):
                    rt.iterate(r, site)
        return Opaque(f"{meth} result", getattr(obj, "taint", None))
    try:
        if meth == "extend":
            items = rt.iterate(rest[0], site) if rest else []
            if isinstance(obj, bytearray):
                if not all(isinstance(x, int) and 0 <= x < 256 for x in items):
                    return None
            obj.extend(items)
            return None
        if meth == "update" and isinstance(obj, (set,)):
            for r in rest:
                for it in rt.iterate(r, site):
                    rt.hashable(it)
                    obj.add(it)
            return None
        if meth == "update" and isinstance(obj, dict):
            if rest:
                for k, v in _pairs(rt, rest[0], site):
                    rt.hashable(k)
                    obj[k] = v
            obj.update(kwargs)
            return None
        if meth == "add":
            rt.hashable(rest[0])
        if meth in ("get", "setdefault", "pop") and isinstance(obj, dict) and rest:
            rt.hashable(rest[0])
        if meth in ("items", "keys", "values"):
            return list(getattr(obj, meth)())
        return getattr(obj, meth)(*rest, **kwargs)
    except PickleStop:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PickleStop(f"{type(exc).__name__}: {exc}") from None


@handler("os.path.join")
def h_path_join(rt, args, kwargs, site, how, name=""):
    if args and all(isinstance(a, str) for a in args):
        import posixpath
        return posixpath.join(*args)
    return Opaque("path")


@handler("re.sub", "re.subn")
def h_re_sub(rt, args, kwargs, site, how, name=""):
    return Opaque("text")


@handler("string.Template")
def h_template(rt, args, kwargs, site, how, name=""):
    return Inst(rt.glob("string.Template", "class"), args, kwargs)


@handler("importlib.util.find_spec", "importlib.util.resolve_name")
def h_spec(rt, args, kwargs, site, how, name=""):
    return Opaque("module spec")
