"""Symbolic model of the Python operations an unpickler can trigger.

``Interp`` holds the state shared by every pickle stream of one artifact
(module attributes rebound by BUILD, ``sys.modules`` replacements, the copyreg
extension registry) and implements what the C unpickler does with the
objects it builds: resolve globals (``find_class``), call callables, run
``__new__``, apply BUILD state, append and set items.  Whenever the modelled
program would call a callable that carries a capability of the policy, a
finding is recorded.

Real objects are only ever obtained from standard-library modules that are
safe to import; nothing from the artifact is executed.
"""
from __future__ import annotations

import ast
import base64
import binascii
import codecs
import functools
import importlib
import inspect
import itertools
import operator
import re
import sys
import time
import types
import zlib

from modelgate import kb
from modelgate.symbolic import (
    AttrGetter, Bound, CodeObj, Coro, Exhausted, FileObj, Func, Inst, ItemGetter, Iter,
    LoadError, MethodCaller, Namespace, Opaque, Partial, Real, Sym, SymClass, SysModules,
    Unres, Wrapper, describe, is_plain,
)

STDLIB = set(sys.stdlib_module_names)
# stdlib modules never imported by the scanner (side effects on import, GUI,
# test suites, or not available on this platform)
NO_IMPORT = {
    "antigravity", "this", "idlelib", "tkinter", "turtle", "turtledemo", "test",
    "lib2to3", "__hello__", "__phello__", "readline", "rlcompleter", "pydoc_data",
    "msilib", "winreg", "winsound", "_msi", "_winapi", "msvcrt", "nt", "_overlapped",
    "curses", "_curses", "_curses_panel", "xxsubtype", "xxlimited", "xxlimited_35",
    "_tkinter", "_testcapi", "_testinternalcapi", "_testbuffer", "_testimportmultiple",
    "_testmultiphase", "_xxtestfuzz", "_ctypes_test", "_testclinic", "_testsinglephase",
    "__main__", "_xxsubinterpreters", "_xxinterpchannels", "_wmi", "_scproxy", "_osx_support",
    "_aix_support", "_android_support", "_ios_support", "_pyrepl", "_bootsubprocess",
}
# names of Python 2 modules (and modules removed from Python 3.12) that the
# loader cannot import
try:
    import _compat_pickle

    PY2_ONLY = (set(_compat_pickle.IMPORT_MAPPING) | {k[0] for k in _compat_pickle.NAME_MAPPING}
                | {"exceptions", "sets", "md5", "sha", "new", "imp", "asynchat", "asyncore",
                   "smtpd", "distutils", "commands", "popen2", "cPickle", "cStringIO",
                   "UserDict", "UserList", "UserString", "__builtin__", "copy_reg", "urllib2",
                   "httplib", "thread", "dummy_thread", "dummy_threading", "string_old",
                   "SocketServer", "Queue", "StringIO"})
    PY2_ONLY -= STDLIB - {"dbm", "urllib", "socket", "itertools", "multiprocessing",
                          "multiprocessing.forking", "multiprocessing.process", "_socket",
                          "_multiprocessing"}
    PY2_ONLY = {m for m in PY2_ONLY if m.split(".")[0] not in STDLIB}
except Exception:  # noqa: BLE001
    _compat_pickle = None
    PY2_ONLY = set()

MAX_BYTES = 64 * 1024 * 1024  # largest value the scanner materialises
MAX_ITEMS = 20000  # items consumed from one iterator
MAX_NEST = 40  # nested loader depth

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def norm_module(name):
    if not isinstance(name, str):
        return str(name)
    if name in kb.MODULE_ALIASES:
        return kb.MODULE_ALIASES[name]
    top, dot, rest = name.partition(".")
    if dot and top in kb.MODULE_ALIASES:
        return kb.MODULE_ALIASES[top] + "." + rest
    return name


def canonical(obj, _depth=0):
    """``module.qualname`` of a real object, module aliases normalised."""
    if _depth > 5:
        return None
    try:
        if isinstance(obj, types.ModuleType):
            return norm_module(obj.__name__)
        if isinstance(obj, type):
            return f"{norm_module(obj.__module__)}.{obj.__qualname__}"
        if isinstance(obj, types.FunctionType):
            return f"{norm_module(obj.__module__)}.{obj.__qualname__}"
        if isinstance(obj, types.BuiltinFunctionType):
            owner = obj.__self__
            if owner is None or isinstance(owner, types.ModuleType):
                mod = obj.__module__ or (owner.__name__ if owner is not None else "builtins")
                return f"{norm_module(mod)}.{obj.__qualname__}"
            if isinstance(owner, type):
                return f"{canonical(owner, _depth + 1)}.{obj.__name__}"
            return f"{canonical(type(owner), _depth + 1)}.{obj.__name__}"
        if isinstance(obj, (types.MethodDescriptorType, types.WrapperDescriptorType,
                            types.ClassMethodDescriptorType, types.GetSetDescriptorType,
                            types.MemberDescriptorType)):
            return f"{canonical(obj.__objclass__, _depth + 1)}.{obj.__name__}"
        if isinstance(obj, types.MethodWrapperType):
            return f"{canonical(type(obj.__self__), _depth + 1)}.{obj.__name__}"
        if isinstance(obj, types.MethodType):
            return canonical(obj.__func__, _depth + 1)
        if isinstance(obj, (staticmethod, classmethod)):
            return canonical(obj.__func__, _depth + 1)
    except Exception:  # noqa: BLE001
        return None
    return None


def unres_norm(name):
    for old, new in kb.UNRES_PREFIX_ALIASES:
        if name == old or (old.endswith(".") and name.startswith(old)):
            return new + name[len(old):] if old.endswith(".") else new
    top, dot, rest = name.partition(".")
    if top in kb.MODULE_ALIASES:
        return kb.MODULE_ALIASES[top] + dot + rest
    return name


_UNKNOWN_SETSTATE = object()
_MISSING = object()


class AttrDict(dict):
    """An instance of a dict subclass (OrderedDict, defaultdict, Counter):
    behaves as a dict and also has an instance ``__dict__``."""

    def __init__(self, *a, clsname="collections.OrderedDict", factory=None):
        super().__init__(*a)
        self.attrs = {}
        self.clsname = clsname
        self.factory = factory

    __hash__ = None


class Interp:
    """State and semantics shared by all pickle streams of one artifact."""

    def __init__(self, deadline=None, max_steps=400_000_000):
        self.findings = []  # (kind, location, detail)
        self.seen = set()
        self.mod_over = {}  # id(real object) -> {attr: value}  (modules, classes, functions)
        self._keep = []  # keep overridden real objects alive (ids stay valid)
        self.sysmod_over = {}  # module name -> value placed in sys.modules
        self.ext_registry = {}  # copyreg extension code -> (module, name)
        self.unres = {}  # interned Unres by name
        self._loc = "?"  # location of the opcode being executed: stream ...
        self._pos = None  # ... and opcode offset
        self.steps = 0
        self.max_steps = max_steps
        self.deadline = deadline
        self.depth = 0
        self._ns_owner = {}
        self._sig_cache = {}
        self.sys_modules = SysModules()
        self.internal_errors = []

    # ------------------------------------------------------------------ util
    @property
    def site(self):
        return self._loc if self._pos is None else f"{self._loc}@{self._pos}"

    @site.setter
    def site(self, value):
        self._loc = value
        self._pos = None

    def tick(self, n=1):
        self.steps += n
        if (self.steps & 1023) == 0 or n > 1:
            self.check_budget()

    def check_budget(self):
        if self.steps > self.max_steps:
            raise Exhausted("step budget exhausted")
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise Exhausted("time budget exhausted")

    def finding(self, kind, detail, location=None):
        loc = location or self.site
        key = (kind, loc, detail)
        if key in self.seen:
            return
        self.seen.add(key)
        if len(self.findings) < 200:
            self.findings.append(key)

    def unres_obj(self, name, is_module=False):
        u = self.unres.get(name)
        if u is None:
            u = Unres(name, is_module)
            self.unres[name] = u
        elif is_module:
            u.is_module = True
        return u

    # ------------------------------------------------------------ importing
    def import_module(self, name):
        """``PyImport_Import(name)``: the object ``sys.modules[name]`` holds
        after importing it; raises LoadError when the import fails."""
        if not isinstance(name, str):
            raise LoadError("module name must be str")
        if name in self.sysmod_over:
            return self.sysmod_over[name]
        if not name or not all(_IDENT.match(p) for p in name.split(".")):
            raise LoadError(f"no module named {name!r}")
        top = name.split(".")[0]
        if name in kb.MODULE_ALIASES and kb.MODULE_ALIASES[name] in ("os",) and top not in ("posix",):
            name = kb.MODULE_ALIASES[name]  # nt -> os
            top = name
        if top in PY2_ONLY or name in PY2_ONLY:
            raise LoadError(f"no module named {name!r}")
        if top in STDLIB:
            if top in NO_IMPORT or "__main__" in name.split("."):
                return self.unres_obj(name, True)
            return self._real_import(name)
        if top not in kb.KNOWN_THIRD_PARTY:
            raise LoadError(f"no module named {name!r}")
        return self.unres_obj(name, True)

    def _real_import(self, name):
        mod = sys.modules.get(name)
        if mod is None:
            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod = importlib.import_module(name)
            except BaseException as exc:  # noqa: BLE001 - ImportError, SystemExit...
                raise LoadError(f"cannot import {name}: {type(exc).__name__}") from None
            mod = sys.modules.get(name, mod)
        return Real(mod, name)

    def find_class(self, module, name, proto, fix_imports=True):
        """The C unpickler's ``find_class``."""
        if not isinstance(module, str) or not isinstance(name, str):
            raise LoadError("find_class arguments must be str")
        if proto < 3 and fix_imports and _compat_pickle is not None:
            if (module, name) in _compat_pickle.NAME_MAPPING:
                module, name = _compat_pickle.NAME_MAPPING[(module, name)]
            elif module in _compat_pickle.IMPORT_MAPPING:
                module = _compat_pickle.IMPORT_MAPPING[module]
        mod = self.import_module(module)
        if proto >= 4:
            obj = mod
            for part in name.split("."):
                if part == "<locals>":
                    raise LoadError("Can't get local attribute")
                obj = self.getattr(obj, part)
            return obj
        return self.getattr(mod, name)

    # -------------------------------------------------------------- wrapping
    def ns_owner(self, d):
        m = self._ns_owner.get(id(d))
        if m is not None and m.__dict__ is d:
            return m
        for mod in list(sys.modules.values()):
            md = getattr(mod, "__dict__", None)
            if md is d:
                self._ns_owner[id(d)] = mod
                return mod
        return None

    def wrap(self, v, name):
        """Model value for a real object reached by attribute access."""
        if v is None or isinstance(v, (bool, int, float, complex, str, bytes, range)):
            return v
        if isinstance(v, tuple):
            if len(v) > 10000:
                return Opaque(f"tuple {name}")
            return tuple(self.wrap(x, f"{name}[{i}]") for i, x in enumerate(v))
        if isinstance(v, frozenset):
            if all(isinstance(x, (str, int, bytes, float)) for x in v):
                return v
            return Opaque(f"frozenset {name}")
        if v is sys.modules:
            return self.sys_modules
        if isinstance(v, dict):
            mod = self.ns_owner(v)
            if mod is not None:
                return Namespace(Real(mod, mod.__name__))
            if v is builtins_dict():
                return Namespace(Real(sys.modules["builtins"], "builtins"))
            return Opaque(f"dict {name}")
        if isinstance(v, types.ModuleType):
            return Real(v, name)
        if isinstance(v, types.BuiltinFunctionType):
            owner = v.__self__
            if owner is not None and not isinstance(owner, (types.ModuleType, type)):
                fn = getattr(type(owner), v.__name__, None)
                if fn is not None:
                    return Bound(self.wrap(owner, name.rsplit(".", 1)[0]), Real(fn, canonical(fn) or name), v.__name__)
            return Real(v, name)
        if isinstance(v, types.MethodType):
            owner = v.__self__
            if isinstance(owner, type):
                return Bound(Real(owner, canonical(owner) or name), Real(v.__func__, canonical(v.__func__) or name), v.__name__)
            return Bound(self.wrap(owner, name), Real(v.__func__, canonical(v.__func__) or name), v.__name__)
        if isinstance(v, (type, types.FunctionType, types.MethodDescriptorType, types.WrapperDescriptorType,
                          types.ClassMethodDescriptorType, types.MethodWrapperType, staticmethod, classmethod,
                          functools.partial, types.CodeType)):
            return Real(v, name)
        if isinstance(v, (list, set, bytearray)):
            return Opaque(f"{type(v).__name__} {name}")
        return Real(v, name)

    # ----------------------------------------------------------- attributes
    def _over(self, obj):
        return self.mod_over.get(id(obj))

    def set_over(self, obj, attr, value):
        d = self.mod_over.get(id(obj))
        if d is None:
            d = self.mod_over[id(obj)] = {}
            self._keep.append(obj)
        d[attr] = value

    def getattr(self, obj, name):
        """``getattr(obj, name)``; LoadError when it raises AttributeError."""
        self.tick()
        if not isinstance(name, str):
            if isinstance(name, Sym):
                return self.unknown_attr(obj)
            raise LoadError("attribute name must be string")
        if isinstance(obj, Real):
            return self._getattr_real(obj, name)
        if isinstance(obj, Unres):
            if name in obj.attrs:
                return obj.attrs[name]
            last = obj.name.rsplit(".", 1)[-1]
            modlike = obj.is_module or last[:1].islower() or last[:1] == "_"
            if modlike and name in kb.COMMON_MODULE_ATTRS and obj.name not in ("numpy", "keras", "tensorflow"):
                try:
                    return self.import_module(name)
                except LoadError:
                    pass
            if name == "__dict__" and obj.is_module:
                return Namespace(obj)
            return self.unres_obj(f"{obj.name}.{name}", False)
        if isinstance(obj, Inst):
            return self._getattr_inst(obj, name)
        if isinstance(obj, SymClass):
            if name in obj.ns:
                return obj.ns[name]
            for b in obj.bases:
                try:
                    return self.getattr(b, name)
                except LoadError:
                    continue
            if name == "__dict__":
                return obj.ns
            raise LoadError(f"type object has no attribute {name!r}")
        if isinstance(obj, Partial):
            if name == "func":
                return obj.func
            if name == "args":
                return obj.args
            if name == "keywords":
                return obj.kwargs
            if name == "__dict__":
                return obj.dict
            if name in obj.dict:
                return obj.dict[name]
            if name == "__setstate__":
                return Bound(obj, Real(functools.partial.__setstate__, "functools.partial.__setstate__"), name)
            if name == "__call__":
                return obj
            return self._method_of(obj, functools.partial, name)
        if isinstance(obj, Namespace):
            if name in ("get", "__getitem__", "setdefault", "update", "__setitem__", "pop", "keys",
                        "values", "items", "copy", "__contains__"):
                return Bound(obj, Real(getattr(dict, name), f"builtins.dict.{name}"), name)
            raise LoadError(f"'dict' object has no attribute {name!r}")
        if isinstance(obj, SysModules):
            if hasattr(dict, name):
                return Bound(obj, Real(getattr(dict, name), f"builtins.dict.{name}"), name)
            raise LoadError(f"'dict' object has no attribute {name!r}")
        if isinstance(obj, AttrDict):
            if name in obj.attrs:
                return obj.attrs[name]
            if name == "__dict__":
                return obj.attrs
            if name == "default_factory":
                return obj.factory
            return self._method_of(obj, dict, name)
        if isinstance(obj, (AttrGetter, ItemGetter, MethodCaller, Iter, Wrapper, Func, CodeObj, Coro)):
            if name == "__call__" and isinstance(obj, (AttrGetter, ItemGetter, MethodCaller, Wrapper, Func)):
                return obj
            if isinstance(obj, Iter) and name == "__next__":
                return Bound(obj, Real(next, "builtins.next"), "__next__")
            if isinstance(obj, Wrapper) and name in ("__wrapped__", "__func__"):
                return obj.target
            return Opaque(f"attribute {name} of {obj.describe()}")
        if isinstance(obj, FileObj):
            return Bound(obj, Opaque(f"file method {name}"), name) if name not in FILE_METHODS else \
                Bound(obj, Real(FILE_METHODS[name], f"io.BytesIO.{name}"), name)
        if isinstance(obj, Bound):
            if name == "__self__":
                return obj.obj
            if name in ("__func__",):
                return obj.func
            return Opaque(f"attribute {name} of bound method")
        if isinstance(obj, Opaque):
            return Opaque(f"{obj.desc}.{name}")
        # plain data
        return self._method_of(obj, type(obj), name)

    def unknown_attr(self, obj):
        return Opaque(f"unknown attribute of {describe(obj)}")

    def _method_of(self, obj, cls, name):
        try:
            attr = inspect.getattr_static(cls, name)
        except AttributeError:
            raise LoadError(f"{cls.__name__!r} object has no attribute {name!r}") from None
        if name == "__class__":
            return Real(type(obj), canonical(type(obj)) or "type") if not isinstance(obj, Sym) else Opaque("class")
        real = getattr(cls, name, None)
        if isinstance(attr, (staticmethod,)):
            return Real(real, canonical(real) or name)
        if isinstance(attr, classmethod) or isinstance(real, types.BuiltinMethodType) and getattr(real, "__self__", None) is cls:
            return Real(real, canonical(real) or name)
        if callable(attr) or isinstance(attr, (types.MethodDescriptorType, types.WrapperDescriptorType)):
            return Bound(obj, Real(real if real is not None else attr, canonical(real if real is not None else attr) or name), name)
        if isinstance(obj, (str, bytes, int, float, complex, tuple, frozenset, range)) and not isinstance(obj, Sym):
            try:
                v = getattr(obj, name)
            except Exception:  # noqa: BLE001
                raise LoadError(f"no attribute {name!r}") from None
            if is_plain(v):
                return v
        return Opaque(f"attribute {name}")

    def _getattr_real(self, obj, name):
        o = obj.obj
        ov = self._over(o)
        if ov is not None and name in ov:
            return ov[name]
        if isinstance(o, types.ModuleType):
            if name == "__dict__":
                return Namespace(obj)
            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    v = getattr(o, name)
            except AttributeError:
                # a submodule that is not imported yet is not an attribute
                raise LoadError(f"module {o.__name__!r} has no attribute {name!r}") from None
            except Exception:  # noqa: BLE001
                raise LoadError(f"module {o.__name__!r} attribute {name!r} failed") from None
            return self.wrap(v, f"{obj.name}.{name}")
        if isinstance(o, type):
            # class-level overrides of base classes
            for base in o.__mro__[1:]:
                bov = self._over(base)
                if bov is not None and name in bov:
                    return bov[name]
            if name == "__dict__":
                return Opaque(f"mappingproxy of {obj.name}")
            try:
                v = getattr(o, name)
            except AttributeError:
                raise LoadError(f"type object {o.__name__!r} has no attribute {name!r}") from None
            except Exception:  # noqa: BLE001
                return Opaque(f"{obj.name}.{name}")
            if name == "__subclasses__":
                return Opaque(f"{obj.name}.__subclasses__")
            return self.wrap(v, f"{obj.name}.{name}")
        if isinstance(o, types.FunctionType):
            if name == "__globals__":
                mod = sys.modules.get(o.__module__)
                if mod is not None and mod.__dict__ is o.__globals__:
                    return Namespace(Real(mod, mod.__name__))
                return Opaque("function globals")
            if name == "__builtins__":
                return Namespace(Real(sys.modules["builtins"], "builtins"))
            if name in ("__code__", "__closure__", "__defaults__", "__kwdefaults__"):
                return Opaque(f"{obj.name}.{name}")
            try:
                v = getattr(o, name)
            except AttributeError:
                raise LoadError(f"function has no attribute {name!r}") from None
            return self.wrap(v, f"{obj.name}.{name}")
        if isinstance(o, (types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType,
                          types.ClassMethodDescriptorType, types.MethodWrapperType, staticmethod, classmethod,
                          functools.partial)):
            if name == "__self__" and isinstance(o, types.BuiltinFunctionType):
                return self.wrap(o.__self__, f"{obj.name}.__self__")
            if name == "__call__":
                return obj
            try:
                v = getattr(o, name)
            except AttributeError:
                raise LoadError(f"no attribute {name!r}") from None
            except Exception:  # noqa: BLE001
                return Opaque(f"{obj.name}.{name}")
            if isinstance(v, (str, int, type(None), bool, tuple)):
                return self.wrap(v, name)
            return self.wrap(v, f"{obj.name}.{name}")
        # some other real object (an instance living in a module): model its
        # methods by its type
        cls = type(o)
        try:
            attr = inspect.getattr_static(o, name)
        except AttributeError:
            raise LoadError(f"object has no attribute {name!r}") from None
        if isinstance(attr, (types.FunctionType, types.MethodDescriptorType, types.WrapperDescriptorType)):
            fn = getattr(cls, name, attr)
            return Bound(obj, Real(fn, canonical(fn) or name), name)
        if isinstance(attr, (property, types.GetSetDescriptorType, types.MemberDescriptorType,
                             functools.cached_property)):
            return Opaque(f"{obj.name}.{name}")  # never run property getters
        try:
            v = getattr(o, name)
        except Exception:  # noqa: BLE001
            return Opaque(f"{obj.name}.{name}")
        return self.wrap(v, f"{obj.name}.{name}")

    def _getattr_inst(self, inst, name):
        if name in inst.attrs:
            return inst.attrs[name]
        if name == "__dict__":
            return inst.attrs
        if name == "__class__":
            return inst.cls
        cls = inst.cls
        if isinstance(cls, Real) and isinstance(cls.obj, type):
            o = cls.obj
            for base in o.__mro__:
                bov = self._over(base)
                if bov is not None and name in bov:
                    v = bov[name]
                    if isinstance(v, Real) and isinstance(v.obj, types.FunctionType):
                        return Bound(inst, v, name)
                    return v
            if o is type and name in ("__subclasses__", "mro"):
                return Opaque("type method")
            try:
                attr = inspect.getattr_static(o, name)
            except AttributeError:
                # a class-level __getattr__ answers attribute misses
                ga = None
                if not (name.startswith("__") and name.endswith("__")):
                    try:
                        ga = inspect.getattr_static(o, "__getattr__")
                    except AttributeError:
                        ga = None
                if isinstance(ga, types.FunctionType):
                    gname = canonical(ga)
                    if gname in HANDLERS:
                        return HANDLERS[gname](self, Real(ga, gname), (inst, name), {})
                    return Opaque(f"{cls.name}.{name} (via __getattr__)")
                # set in __init__ (which the scanner does not run) or missing
                return self.inst_dynamic_attr(inst, name)
            if isinstance(attr, staticmethod):
                return self.wrap(attr.__func__, f"{cls.name}.{name}")
            if isinstance(attr, classmethod):
                return Bound(cls, Real(attr.__func__, canonical(attr.__func__) or name), name)
            if isinstance(attr, types.BuiltinFunctionType):
                # builtin functions are not descriptors: no binding
                return self.wrap(attr, f"{cls.name}.{name}")
            if isinstance(attr, (types.FunctionType, types.MethodDescriptorType, types.WrapperDescriptorType)):
                fn = getattr(o, name, attr)
                if isinstance(fn, types.MethodType):
                    fn = fn.__func__
                return Bound(inst, Real(fn, canonical(fn) or f"{cls.name}.{name}"), name)
            if isinstance(attr, (property, types.GetSetDescriptorType, types.MemberDescriptorType)):
                return self.inst_dynamic_attr(inst, name)
            if isinstance(attr, functools.cached_property):
                return Opaque(f"{cls.name}.{name}")
            try:
                v = getattr(o, name)
            except Exception:  # noqa: BLE001
                return Opaque(f"{cls.name}.{name}")
            return self.wrap(v, f"{cls.name}.{name}")
        if isinstance(cls, SymClass):
            try:
                v = self.getattr(cls, name)
            except LoadError:
                return Opaque(f"attribute {name}")
            if isinstance(v, (Real, Func, Partial)) and not isinstance(v, Wrapper):
                if isinstance(v, Real) and isinstance(v.obj, types.FunctionType) or isinstance(v, Func):
                    return Bound(inst, v, name)
            if isinstance(v, Wrapper) and v.desc == "staticmethod":
                return v.target
            return v
        if isinstance(cls, Unres):
            if name.startswith("__") and name.endswith("__") and name not in ("__call__", "__getitem__"):
                return Opaque(f"{cls.name}.{name}")
            return Bound(inst, self.unres_obj(f"{cls.name}.{name}"), name)
        return Opaque(f"attribute {name}")

    def inst_dynamic_attr(self, inst, name):
        """An attribute that ``__init__`` would set: derive the known ones
        from the constructor arguments."""
        cls = inst.cls
        cname = canonical(cls.obj) if isinstance(cls, Real) else None
        if cname and inst.inited:
            bound = self.bind_ctor(cls.obj, inst)
            if bound is not None:
                mapping = INIT_ATTRS.get(cname)
                if mapping and name in mapping:
                    return bound.get(mapping[name])
                for base in getattr(cls.obj, "__mro__", ())[1:]:
                    m2 = INIT_ATTRS.get(canonical(base))
                    if m2 and name in m2:
                        return bound.get(m2[name])
        return Opaque(f"attribute {name} of {inst.describe()}")

    def bind_ctor(self, cls, inst):
        try:
            sig = self.signature(cls)
            if sig is None:
                return None
            ba = sig.bind(*inst.args, **inst.kwargs)
            ba.apply_defaults()
            return dict(ba.arguments)
        except Exception:  # noqa: BLE001
            return None

    def setattr(self, obj, name, value):
        self.tick()
        if not isinstance(name, str):
            if isinstance(name, Sym):
                return
            raise LoadError("attribute name must be string")
        if isinstance(obj, Real):
            o = obj.obj
            if isinstance(o, types.ModuleType):
                self.set_over(o, name, value)
                return
            if isinstance(o, type):
                if not (o.__flags__ & (1 << 9)):  # Py_TPFLAGS_HEAPTYPE
                    raise LoadError(f"cannot set {name!r} attribute of immutable type")
                self.set_over(o, name, value)
                return
            if isinstance(o, types.FunctionType):
                self.set_over(o, name, value)
                return
            raise LoadError(f"cannot set attribute {name!r}")
        if isinstance(obj, Unres):
            obj.attrs[name] = value
            return
        if isinstance(obj, Inst):
            inst_cls = obj.cls
            if isinstance(inst_cls, Real) and inst_cls.obj in (object,):
                raise LoadError("'object' object has no attribute")
            obj.attrs[name] = value
            return
        if isinstance(obj, SymClass):
            obj.ns[name] = value
            return
        if isinstance(obj, Partial):
            obj.dict[name] = value
            return
        if isinstance(obj, AttrDict):
            obj.attrs[name] = value
            return
        if isinstance(obj, (Opaque, FileObj, Iter, Wrapper)):
            return
        raise LoadError(f"'{type(obj).__name__}' object attribute {name!r} is read-only")

    # ------------------------------------------------------------- items
    def getitem(self, obj, key):
        self.tick()
        if isinstance(obj, Namespace):
            if not isinstance(key, str):
                if isinstance(key, Sym):
                    return self.unknown_attr(obj.module)
                raise LoadError("KeyError")
            if obj.module is not None:
                try:
                    return self.getattr(obj.module, key)
                except LoadError:
                    raise LoadError(f"KeyError: {key!r}") from None
        if isinstance(obj, SysModules):
            if isinstance(key, Sym):
                return Opaque("sys.modules entry")
            if not isinstance(key, str):
                raise LoadError("KeyError")
            try:
                return self.import_module(key)
            except LoadError:
                raise LoadError(f"KeyError: {key!r}") from None
        if isinstance(obj, AttrDict):
            try:
                return obj[key]
            except KeyError:
                if obj.factory is not None:
                    v = self.call(obj.factory, ())
                    obj[key] = v
                    return v
                raise LoadError("KeyError") from None
            except TypeError:
                raise LoadError("unhashable key") from None
        if isinstance(obj, dict):
            try:
                return obj[key]
            except (KeyError, TypeError):
                raise LoadError("KeyError") from None
        if isinstance(obj, (list, tuple, str, bytes, bytearray, range)):
            if isinstance(key, Sym):
                if isinstance(obj, (list, tuple)) and obj:
                    return obj[0] if len(obj) == 1 else Opaque("element")
                return Opaque("element")
            try:
                v = obj[key]
            except Exception:  # noqa: BLE001
                raise LoadError("IndexError/TypeError in subscript") from None
            return v
        if isinstance(obj, Inst):
            m = self.getattr(obj, "__getitem__") if not isinstance(obj.cls, Unres) else None
            if isinstance(m, (Bound, Real)):
                return self.call(m, (key,))
            return Opaque(f"item of {obj.describe()}")
        if isinstance(obj, Real) and isinstance(obj.obj, type):
            # Cls[...] -> __class_getitem__: typing alias, harmless
            return Opaque(f"{obj.name}[...]")
        if isinstance(obj, (int, float, type(None), bool, complex)):
            raise LoadError("object is not subscriptable")
        return Opaque("item")

    def setitem(self, obj, key, value):
        self.tick()
        if isinstance(obj, Namespace):
            if isinstance(key, str):
                if isinstance(obj.module, Real):
                    self.set_over(obj.module.obj, key, value)
                elif isinstance(obj.module, Unres):
                    obj.module.attrs[key] = value
            return
        if isinstance(obj, SysModules):
            if isinstance(key, str):
                self.sysmod_over[key] = value
            return
        if isinstance(obj, dict):
            try:
                obj[key] = value
            except TypeError:
                raise LoadError("unhashable type") from None
            return
        if isinstance(obj, list):
            if isinstance(key, int) and not isinstance(key, bool):
                try:
                    obj[key] = value
                except IndexError:
                    raise LoadError("list assignment index out of range") from None
                return
            if isinstance(key, Sym):
                return
            raise LoadError("list indices must be integers")
        if isinstance(obj, Inst):
            if isinstance(obj.cls, Real):
                try:
                    m = self.getattr(obj, "__setitem__")
                except LoadError:
                    raise LoadError("object does not support item assignment") from None
                if isinstance(m, (Bound, Real, Partial)):
                    self.call(m, (key, value))
            return
        if isinstance(obj, (Opaque, Unres)):
            return
        raise LoadError(f"'{type(obj).__name__}' object does not support item assignment")

    # --------------------------------------------------------------- calls
    def signature(self, obj):
        key = id(obj)
        if key in self._sig_cache:
            return self._sig_cache[key][1]
        try:
            sig = inspect.signature(obj)
        except Exception:  # noqa: BLE001
            sig = None
        self._sig_cache[key] = (obj, sig)
        return sig

    def check_call(self, obj, args, kwargs):
        """LoadError if calling the real ``obj`` with these arguments raises
        TypeError before doing anything (wrong number of arguments)."""
        sig = self.signature(obj)
        if sig is None:
            return
        try:
            sig.bind(*args, **kwargs)
        except TypeError:
            raise LoadError(f"{canonical(obj)}: arguments do not match the signature") from None

    def call(self, f, args, kwargs=None):
        """Call ``f(*args, **kwargs)`` as the loader would."""
        self.tick()
        kwargs = kwargs or {}
        if not isinstance(args, tuple):
            args = tuple(args)
        if isinstance(f, Real):
            return self.call_real(f, args, kwargs)
        if isinstance(f, Unres):
            return self.call_unres(f, args, kwargs)
        if isinstance(f, Partial):
            return self.call(f.func, f.args + args, {**f.kwargs, **kwargs})
        if isinstance(f, Bound):
            return self.call(f.func, (f.obj,) + args, kwargs) if not isinstance(f.func, Opaque) else \
                Opaque(f"result of {f.name}")
        if isinstance(f, AttrGetter):
            if len(args) != 1 or kwargs:
                raise LoadError("attrgetter expected 1 argument")
            vals = []
            for a in f.attrs:
                obj = args[0]
                for part in a.split("."):
                    obj = self.getattr(obj, part)
                vals.append(obj)
            return vals[0] if len(vals) == 1 else tuple(vals)
        if isinstance(f, ItemGetter):
            if len(args) != 1 or kwargs:
                raise LoadError("itemgetter expected 1 argument")
            vals = [self.getitem(args[0], k) for k in f.items]
            return vals[0] if len(vals) == 1 else tuple(vals)
        if isinstance(f, MethodCaller):
            if len(args) != 1 or kwargs:
                raise LoadError("methodcaller expected 1 argument")
            m = self.getattr(args[0], f.name)
            return self.call(m, f.args, f.kwargs)
        if isinstance(f, Wrapper):
            if f.target is None:
                # a decorator: lru_cache(maxsize)(fn) wraps fn, wraps(x)(fn) is fn
                if not args:
                    raise LoadError("decorator expects a function")
                if f.desc == "decorator:identity":
                    return args[0]
                return Wrapper(args[0], "functools.lru_cache wrapper")
            return self.call(f.target, args, kwargs)
        if isinstance(f, Func):
            self.finding(kb.CODE, f"call of a function built from serialized code ({f.desc})")
            return Opaque("result of materialised function")
        if isinstance(f, SymClass):
            return self.instantiate_symclass(f, args, kwargs)
        if isinstance(f, Inst):
            return self.call_instance(f, args, kwargs)
        if isinstance(f, Iter):
            raise LoadError("iterator is not callable")
        if isinstance(f, Opaque):
            return Opaque(f"result of calling {f.desc}")
        if isinstance(f, (Namespace, SysModules, FileObj, CodeObj, Coro)):
            raise LoadError("object is not callable")
        raise LoadError(f"'{type(f).__name__}' object is not callable")

    def call_real(self, f, args, kwargs):
        o = f.obj
        if isinstance(o, types.ModuleType):
            raise LoadError("'module' object is not callable")
        name = canonical(o) or f.name
        # wrong arguments raise TypeError before the callable does anything
        self.check_call(o, args, kwargs)
        self.validate(name, args, kwargs)
        h = HANDLERS.get(name)
        if h is not None:
            if isinstance(o, types.FunctionType) and "." in o.__qualname__ and "<" not in o.__qualname__ \
                    and args and not isinstance(args[0], Sym) and not name.startswith("modelgate."):
                raise LoadError(f"{name}: invalid self argument")
            return h(self, f, args, kwargs)
        cap = kb.CAPS.get(name)
        if cap is not None:
            return self.capability(o, name, cap, args, kwargs)
        if isinstance(o, type):
            return self.instantiate(f, args, kwargs)
        if isinstance(o, (types.WrapperDescriptorType, types.MethodDescriptorType)) and o.__name__ == "__call__":
            if not args:
                raise LoadError("__call__ needs an object")
            return self.call(args[0], args[1:], kwargs)
        if name.endswith(".__init__"):
            # Cls.__init__(obj, *args): the constructor's capability
            owner = name[: -len(".__init__")]
            cap = kb.CAPS.get(owner)
            if cap is not None:
                if not args:
                    raise LoadError("__init__ needs an object")
                return self.capability(None, owner, cap, args[1:], kwargs)
            return Opaque(f"result of {name}")
        mod, _, attr = name.rpartition(".")
        if mod in ("os",) and attr not in kb.OS_HARMLESS and not attr.startswith("_"):
            # an os function the tables do not classify: file or process
            self.check_call(o, args, kwargs)
            self.finding(kb.FILE, f"call of {name}{self.fmt_args(args)}")
            return Opaque(f"result of {name}")
        if kb.MODULE_DEFAULTS.get(mod) and not attr.startswith("_"):
            self.check_call(o, args, kwargs)
            self.finding(kb.MODULE_DEFAULTS[mod], f"call of {name}{self.fmt_args(args)}")
            return Opaque(f"result of {name}")
        if name in PURE or (mod in PURE_MODULES and not attr.startswith("_") and attr not in ("main", "test")):
            return self.eval_pure(o, name, args, kwargs)
        if isinstance(o, types.BuiltinFunctionType) and isinstance(o.__self__, type) and o.__self__ in (str, bytes, bytearray, int, float, dict):
            return self.eval_pure(o, name, args, kwargs)
        if isinstance(o, (types.MethodDescriptorType, types.WrapperDescriptorType)) and \
                o.__objclass__ in (str, bytes, bytearray, int, float, tuple, complex, frozenset):
            return self.eval_pure(o, name, args, kwargs)
        if isinstance(o, (types.MethodDescriptorType, types.WrapperDescriptorType)) and \
                o.__objclass__ in (list, dict, set):
            return self.container_method(o, name, args, kwargs)
        ret = kb.RETURNS.get(name)
        if ret is not None:
            return self.instance_of(ret)
        # any other function: no capability of its own; callables it receives
        # are not called by it as far as the scanner knows
        return Opaque(f"result of {name}")

    def instance_of(self, clsname):
        """A model instance of the standard-library class ``clsname``."""
        mod, _, qual = clsname.rpartition(".")
        try:
            cls = self.find_class(mod, qual, 4, False)
        except LoadError:
            parts = clsname.split(".")
            cls = None
            for k in range(len(parts) - 1, 0, -1):
                try:
                    cls = self.find_class(".".join(parts[:k]), ".".join(parts[k:]), 4, False)
                    break
                except LoadError:
                    continue
            if cls is None:
                return Opaque(f"{clsname} instance")
        return Inst(cls, (), {}, True)

    def fmt_args(self, args):
        return "(" + ", ".join(describe(a, 60) for a in args[:3]) + (", ..." if len(args) > 3 else "") + ")"

    def validate(self, name, args, kwargs):
        """LoadError for argument errors the callable raises before acting."""
        validator = kb.ARG_VALIDATORS.get(name)
        if validator is not None:
            try:
                valid = validator(args, kwargs)
            except Exception:  # noqa: BLE001
                valid = True
            if not valid:
                raise LoadError(f"{name}: invalid arguments")
        check = kb.ARG_CHECKS.get(name)
        if check is not None:
            idx, pname, types_ = check
            v = args[idx] if len(args) > idx else kwargs.get(pname, _MISSING)
            if v is not _MISSING and not isinstance(v, Sym) and not isinstance(v, types_):
                raise LoadError(f"{name}: argument {pname!r} has the wrong type")
            if v is not _MISSING and isinstance(v, Sym) and (
                    not_value_like(v) or isinstance(v, Iter) and not (set(types_) & {list, tuple})):
                raise LoadError(f"{name}: argument {pname!r} has the wrong type")

    def capability(self, o, name, cap, args, kwargs):
        kind, cond = cap
        if o is not None:
            self.check_call(o, args, kwargs)
        self.validate(name, args, kwargs)
        if o is not None and not isinstance(o, type) and "." in getattr(o, "__qualname__", "") \
                and not isinstance(o, types.BuiltinFunctionType):
            # a method of a class called with plain data as ``self`` fails
            # (AttributeError / descriptor TypeError) before acting
            if args and not isinstance(args[0], Sym):
                raise LoadError(f"{name}: invalid self argument")
        if kind == kb.FILE and name.split(".")[0] in ("os", "shutil", "io", "builtins", "codecs", "pathlib",
                                                      "tempfile", "linecache", "shelve", "dbm") \
                and not name.startswith(("pathlib.Path.", "tempfile.")):
            first = args[0] if args else None
            is_open = name in ("io.open", "builtins.open", "io.FileIO", "codecs.open", "os.fdopen")
            if isinstance(first, int) and is_open:
                return Opaque(f"file object for descriptor {first}")  # an fd, not a path
            if first is not None and (not_value_like(first) or isinstance(first, (Iter, list, tuple, dict, set,
                                                                                    frozenset, float, complex,
                                                                                    bool))):
                raise LoadError(f"{name}: path should be string, bytes or os.PathLike")
        if cond is not None:
            try:
                ok = cond(args, kwargs)
            except Exception:  # noqa: BLE001
                ok = True
            if not ok:
                if isinstance(o, type):
                    return Inst(Real(o, name), args, kwargs, True)
                return Opaque(f"result of {name}")
        self.finding(kind, f"call of {name}{self.fmt_args(args)}")
        if isinstance(o, type):
            inst = Inst(Real(o, name), args, kwargs, True)
            inst.extra["capability"] = kind
            return inst
        return Opaque(f"result of {name}")

    def call_unres(self, f, args, kwargs):
        name = unres_norm(f.name)
        nested = kb.NESTED_LOADERS.get(name)
        if nested is not None:
            return self.nested_loader(nested, name, args, kwargs)
        h = HANDLERS.get(name)
        if h is not None and name.split(".")[0] not in STDLIB:
            return h(self, f, args, kwargs)
        cap = kb.CAPS.get(name)
        if cap is None:
            cap = kb.NEW_CAPS.get(name)
        if cap is not None:
            return self.capability(None, name, cap, args, kwargs)
        last = name.rsplit(".", 1)[-1]
        if last in kb.REEXPORTED and "." in name:
            target = kb.REEXPORTED[last]
            mod, _, qual = target.rpartition(".")
            try:
                real = self.find_class(mod, qual, 4, False)
            except LoadError:
                real = None
            if real is not None:
                return self.call(real, args, kwargs)
        # a method of a third-party object bound to an instance
        return Inst(f, args, kwargs, True)

    def call_instance(self, inst, args, kwargs):
        cls = inst.cls
        if isinstance(cls, Real) and isinstance(cls.obj, type):
            try:
                m = self.getattr(inst, "__call__")
            except LoadError:
                raise LoadError("object is not callable") from None
            if isinstance(m, Opaque):
                return Opaque("result of calling an object")
            return self.call(m, args, kwargs)
        if isinstance(cls, SymClass):
            m = self.getattr(inst, "__call__")
            return self.call(m, args, kwargs)
        return Opaque(f"result of calling {inst.describe()}")

    # ------------------------------------------------------------- classes
    def instantiate(self, f, args, kwargs):
        """``cls(*args, **kwargs)`` for a real class without a handler."""
        o = f.obj
        # capability of a base class (a subclass of Popen, of TCPServer ...)
        for base in o.__mro__[1:]:
            bname = canonical(base)
            if bname in HANDLERS and bname not in ("builtins.object",):
                h = HANDLERS[bname]
                if getattr(h, "inheritable", False):
                    return h(self, f, args, kwargs)
            cap = kb.CAPS.get(bname)
            if cap is not None:
                return self.capability(o, bname, cap, args, kwargs)
        for base in o.__mro__:
            bov = self._over(base)
            if bov is not None and ("__init__" in bov or "__new__" in bov):
                inst = Inst(f, args, kwargs, True)
                if "__new__" in bov:
                    self.call(bov["__new__"], (f,) + args, kwargs)
                if "__init__" in bov:
                    self.call(bov["__init__"], (inst,) + args, kwargs)
                return inst
        if issubclass(o, dict):
            return AttrDict(clsname=canonical(o) or f.name)
        if issubclass(o, BaseException):
            return Inst(f, args, kwargs, True)
        return Inst(f, args, kwargs, True)

    def instantiate_symclass(self, c, args, kwargs):
        inst = Inst(c, args, kwargs, True)
        new = c.ns.get("__new__")
        if new is not None:
            target = new.target if isinstance(new, Wrapper) else new
            self.call(target, (c,) + args, kwargs)
        init = c.ns.get("__init__")
        if init is not None:
            self.call(init, (inst,) + args, kwargs)
        else:
            for b in c.bases:
                if isinstance(b, Real) and isinstance(b.obj, type):
                    for base in b.obj.__mro__:
                        cap = kb.CAPS.get(canonical(base))
                        if cap is not None:
                            self.capability(base, canonical(base), cap, args, kwargs)
                            return inst
        return inst

    def new_object(self, cls, args, kwargs, opname="NEWOBJ"):
        """``cls.__new__(cls, *args, **kwargs)`` (NEWOBJ, NEWOBJ_EX, and
        argument-less INST/OBJ on a class)."""
        self.tick()
        if isinstance(cls, Real):
            o = cls.obj
            if not isinstance(o, type):
                raise LoadError(f"{opname} class argument must be a type")
            name = canonical(o)
            h = NEW_HANDLERS.get(name)
            if h is not None:
                return h(self, cls, args, kwargs)
            for base in o.__mro__:
                bov = self._over(base)
                if bov is not None and "__new__" in bov:
                    self.call(bov["__new__"], (cls,) + tuple(args), kwargs)
                    break
            if issubclass(o, dict):
                return AttrDict(clsname=name or cls.name)
            if issubclass(o, list) and o is list:
                return []
            if o is set:
                return set()
            if o is object:
                return Inst(cls, (), {}, False)
            return Inst(cls, args, kwargs, False)
        if isinstance(cls, Unres):
            cap = kb.NEW_CAPS.get(unres_norm(cls.name))
            if cap is not None:
                self.capability(None, cls.name, cap, args, kwargs)
            return Inst(cls, args, kwargs, False)
        if isinstance(cls, SymClass):
            new = cls.ns.get("__new__")
            if new is not None:
                target = new.target if isinstance(new, Wrapper) else new
                self.call(target, (cls,) + tuple(args), kwargs)
            return Inst(cls, args, kwargs, False)
        if isinstance(cls, Opaque):
            return Opaque("object built by __new__ of an unknown class")
        raise LoadError(f"{opname} class argument must be a type")

    def instantiate_noargs(self, cls):
        """INST/OBJ with no arguments: ``cls.__new__(cls)`` for a class,
        a plain call for any other callable."""
        if isinstance(cls, Real):
            if isinstance(cls.obj, type):
                if hasattr(cls.obj, "__getinitargs__"):
                    return self.call(cls, ())
                return self.new_object(cls, (), {}, "INST")
            return self.call(cls, ())
        if isinstance(cls, SymClass):
            return self.new_object(cls, (), {}, "INST")
        if isinstance(cls, Unres):
            last = cls.name.rsplit(".", 1)[-1]
            if last[:1].isupper():
                return Inst(cls, (), {}, False)
            return self.call(cls, ())
        return self.call(cls, ())

    # --------------------------------------------------------------- BUILD
    def build(self, inst, state):
        self.tick()
        setstate = self.lookup_setstate(inst)
        if setstate is _UNKNOWN_SETSTATE:
            parts = state if isinstance(state, tuple) and len(state) == 2 else (state,)
            for part in parts:
                if isinstance(part, dict):
                    for k, v in list(part.items()):
                        if isinstance(k, str):
                            inst.attrs[k] = v
            inst.state = state
            return
        if setstate is not None:
            self.call(setstate, (state,))
            return
        slotstate = None
        if isinstance(state, tuple) and len(state) == 2:
            state, slotstate = state
        if state is not None:
            if not isinstance(state, dict):
                raise LoadError("state is not a dictionary")
            target = self.instance_dict(inst)
            for k, v in list(state.items()):
                if isinstance(target, Namespace):
                    self.setitem(target, k, v)
                else:
                    try:
                        target[k] = v
                    except TypeError:
                        raise LoadError("unhashable key in state") from None
        if slotstate is not None:
            if not isinstance(slotstate, dict):
                raise LoadError("slot state is not a dictionary")
            for k, v in list(slotstate.items()):
                self.setattr(inst, k, v)

    def lookup_setstate(self, inst):
        """``getattr(inst, '__setstate__', None)`` (Python >= 3.11 has no
        default ``object.__setstate__``)."""
        if isinstance(inst, Inst):
            if "__setstate__" in inst.attrs:
                return inst.attrs["__setstate__"]
            cls = inst.cls
            if isinstance(cls, Real) and isinstance(cls.obj, type):
                for base in cls.obj.__mro__:
                    bov = self._over(base)
                    if bov is not None and "__setstate__" in bov:
                        v = bov["__setstate__"]
                        if isinstance(v, Real) and isinstance(v.obj, types.FunctionType):
                            return Bound(inst, v, "__setstate__")
                        return v
                fn = getattr(cls.obj, "__setstate__", None)
                if fn is None:
                    return None
                if isinstance(fn, types.MethodType):
                    fn = fn.__func__
                return Bound(inst, Real(fn, canonical(fn) or "__setstate__"), "__setstate__")
            if isinstance(cls, SymClass):
                try:
                    v = self.getattr(cls, "__setstate__")
                except LoadError:
                    return None
                if isinstance(v, Real) and isinstance(v.obj, types.FunctionType) or isinstance(v, Func):
                    return Bound(inst, v, "__setstate__")
                return v
            # a third-party class: its own __setstate__ (if any) is taken to be
            # benign; the state is kept as the instance's attributes
            return _UNKNOWN_SETSTATE
        if isinstance(inst, Real):
            o = inst.obj
            ov = self._over(o)
            if ov is not None and "__setstate__" in ov:
                return ov["__setstate__"]
            if isinstance(o, type):
                fn = getattr(o, "__setstate__", None)
                if fn is not None:
                    return Real(fn, canonical(fn) or "__setstate__")
            return None
        if isinstance(inst, Unres):
            return inst.attrs.get("__setstate__")
        if isinstance(inst, Partial):
            if "__setstate__" in inst.dict:
                return inst.dict["__setstate__"]
            return Bound(inst, Real(functools.partial.__setstate__, "functools.partial.__setstate__"), "__setstate__")
        if isinstance(inst, AttrDict):
            if "__setstate__" in inst.attrs:
                return inst.attrs["__setstate__"]
            return None
        return None

    def instance_dict(self, inst):
        if isinstance(inst, Inst):
            if isinstance(inst.cls, Real) and inst.cls.obj is object:
                raise LoadError("'object' object has no attribute '__dict__'")
            return inst.attrs
        if isinstance(inst, Real):
            o = inst.obj
            if isinstance(o, types.ModuleType):
                return Namespace(inst)
            if isinstance(o, types.FunctionType):
                d = self.mod_over.get(id(o))
                if d is None:
                    d = self.mod_over[id(o)] = {}
                    self._keep.append(o)
                return d
            raise LoadError("object has no writable __dict__")
        if isinstance(inst, Unres):
            return inst.attrs
        if isinstance(inst, Partial):
            return inst.dict
        if isinstance(inst, AttrDict):
            return inst.attrs
        if isinstance(inst, SymClass):
            raise LoadError("mappingproxy does not support item assignment")
        if isinstance(inst, Opaque):
            return {}
        raise LoadError(f"'{type(inst).__name__}' object has no attribute '__dict__'")

    # ----------------------------------------------- APPEND/SETITEMS targets
    def extend(self, target, items):
        if isinstance(target, list):
            target.extend(items)
            return
        try:
            ext = self.getattr(target, "extend")
        except LoadError:
            ext = None
        if ext is not None and not isinstance(ext, Opaque):
            self.call(ext, (list(items),))
            return
        app = self.getattr(target, "append")
        if isinstance(app, Opaque):
            return
        for it in items:
            self.call(app, (it,))

    def add_items(self, target, items):
        if isinstance(target, set):
            try:
                target.update(items)
            except TypeError:
                raise LoadError("unhashable type") from None
            return
        add = self.getattr(target, "add")
        if isinstance(add, Opaque):
            return
        for it in items:
            self.call(add, (it,))

    # ------------------------------------------------------------ iteration
    def iterate(self, x):
        """Generator over the items ``iter(x)`` yields (lazily)."""
        if isinstance(x, (list, tuple, str, bytes, bytearray, range)):
            yield from x
            return
        if isinstance(x, (set, frozenset)):
            yield from list(x)
            return
        if isinstance(x, dict):
            yield from list(x.keys())
            return
        if isinstance(x, Iter):
            yield from x.gen
            return
        if isinstance(x, FileObj):
            if x.data is not None:
                for line in x.data[x.pos:].splitlines(True)[:MAX_ITEMS]:
                    yield line
            else:
                yield Opaque("line of a file")
            return
        if isinstance(x, Inst) and isinstance(x.cls, Real):
            cname = canonical(x.cls.obj)
            if cname == "fileinput.FileInput":
                self.finding(kb.FILE, "iteration of fileinput.FileInput opens files")
            m = None
            try:
                m = self.getattr(x, "__iter__")
            except LoadError:
                m = None
            if m is None:
                raise LoadError("object is not iterable")
            yield Opaque(f"element of {x.describe()}")
            return
        if isinstance(x, (int, float, complex, bool, type(None))):
            raise LoadError(f"'{type(x).__name__}' object is not iterable")
        if isinstance(x, Real) and isinstance(x.obj, (types.FunctionType, types.BuiltinFunctionType, types.ModuleType)):
            raise LoadError("object is not iterable")
        yield Opaque(f"element of {describe(x)}")

    def consume(self, x, limit=MAX_ITEMS):
        out = []
        for item in self.iterate(x):
            self.tick()
            out.append(item)
            if len(out) >= limit:
                break
        return out

    # ------------------------------------------------------------- values
    def to_bytes(self, v):
        """Concrete bytes for a bytes-like model value, or None."""
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
        if isinstance(v, memoryview):
            return v.tobytes()
        return None

    def eval_pure(self, o, name, args, kwargs):
        """Evaluate a side-effect-free function on plain arguments."""
        if any(isinstance(a, Iter) for a in args) and name in CONSUMING:
            # these builtins iterate what they are given
            args = tuple(self.consume(a) if isinstance(a, Iter) else a for a in args)
        if not (all(is_plain(a) for a in args) and all(is_plain(v) for v in kwargs.values())):
            return Opaque(f"result of {name}")
        if not size_ok(name, args, kwargs):
            return Opaque(f"result of {name} (not evaluated)")
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = o(*args, **kwargs)
        except (MemoryError, RecursionError, OverflowError):
            return Opaque(f"result of {name}")
        except Exception as exc:  # noqa: BLE001 - the same call raises in the loader
            raise LoadError(f"{name} raised {type(exc).__name__}") from None
        if isinstance(r, (zip, map, filter, enumerate, reversed)) or type(r).__name__.endswith("iterator"):
            return Iter(iter(list(itertools.islice(r, MAX_ITEMS))), f"iterator from {name}")
        if isinstance(r, memoryview):
            return r.tobytes()
        if is_plain(r) or isinstance(r, range):
            if isinstance(r, (bytes, str, bytearray)) and len(r) > MAX_BYTES:
                return Opaque(f"result of {name}")
            return r
        return Opaque(f"result of {name}")

    def container_method(self, o, name, args, kwargs):
        """Methods of list/dict/set called on model values."""
        if not args:
            raise LoadError(f"{name} needs self")
        self_ = args[0]
        rest = args[1:]
        meth = o.__name__
        if isinstance(self_, (Namespace, SysModules)):
            return self.mapping_method(self_, meth, rest, kwargs)
        objclass = o.__objclass__
        if not isinstance(self_, objclass):
            raise LoadError(f"descriptor {meth!r} requires a {objclass.__name__!r} object")
        if meth == "sort" and isinstance(self_, list):
            key = kwargs.get("key")
            if key is not None:
                for item in list(self_):
                    self.call(key, (item,))
            return None
        if meth in ("extend", "update") and rest and isinstance(rest[0], (Iter, Inst, Opaque)):
            items = self.consume(rest[0])
            if meth == "extend":
                self_.extend(items)
                return None
            for kv in items:
                if isinstance(kv, (tuple, list)) and len(kv) == 2:
                    self.setitem(self_, kv[0], kv[1])
            return None
        if meth in ("__getitem__", "get") and isinstance(self_, dict):
            key = rest[0] if rest else None
            try:
                return self_[key]
            except (KeyError, TypeError):
                if meth == "get":
                    return rest[1] if len(rest) > 1 else None
                if isinstance(self_, AttrDict) and self_.factory is not None:
                    return self.getitem(self_, key)
                raise LoadError("KeyError") from None
        try:
            r = o(self_, *rest, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LoadError(f"{name} raised {type(exc).__name__}") from None
        return r if (is_plain(r) or isinstance(r, Sym)) else Opaque(f"result of {name}")

    def mapping_method(self, m, meth, rest, kwargs):
        if meth in ("get", "__getitem__"):
            if not rest:
                raise LoadError("missing key")
            try:
                return self.getitem(m, rest[0])
            except LoadError:
                if meth == "get":
                    return rest[1] if len(rest) > 1 else None
                raise
        if meth in ("__setitem__",):
            self.setitem(m, rest[0], rest[1])
            return None
        if meth == "setdefault":
            try:
                return self.getitem(m, rest[0])
            except LoadError:
                self.setitem(m, rest[0], rest[1] if len(rest) > 1 else None)
                return rest[1] if len(rest) > 1 else None
        if meth == "update":
            src = rest[0] if rest else {}
            if isinstance(src, dict):
                for k, v in list(src.items()):
                    self.setitem(m, k, v)
            for k, v in kwargs.items():
                self.setitem(m, k, v)
            return None
        if meth == "pop":
            try:
                return self.getitem(m, rest[0])
            except LoadError:
                if len(rest) > 1:
                    return rest[1]
                raise
        if meth == "__contains__":
            return Opaque("bool")
        return Opaque(f"result of dict.{meth}")

    # -------------------------------------------------------- nested loaders
    def nested_loader(self, kind, name, args, kwargs):
        from modelgate import nested

        return nested.run(self, kind, name, args, kwargs)


def not_value_like(v):
    """True for model values that are certainly not str/bytes/path/sequence
    data (functions, getters, iterators, modules ...)."""
    if isinstance(v, (AttrGetter, ItemGetter, MethodCaller, Partial, Wrapper, Func, CodeObj,
                      Namespace, SysModules, Coro, Bound)):
        return True
    if isinstance(v, Real) and isinstance(v.obj, (types.FunctionType, types.BuiltinFunctionType, type,
                                                   types.ModuleType, types.MethodDescriptorType,
                                                   types.WrapperDescriptorType)):
        return True
    return False


def builtins_dict():
    import builtins

    return builtins.__dict__


# ----------------------------------------------------------------------
# plain-function evaluation

PURE_MODULES = {
    # modules whose public functions are side-effect free on plain data (they
    # are evaluated for real, on concrete arguments, under size guards)
    "base64", "binascii", "operator", "string", "json", "shlex", "textwrap", "html",
    "urllib.parse", "os.path", "math", "struct", "fnmatch", "keyword", "colorsys",
    "unicodedata", "heapq", "bisect", "stringprep", "zlib", "ipaddress", "posixpath",
    "reprlib", "numbers", "statistics",
}
PURE = set("""
    builtins.str builtins.bytes builtins.bytearray builtins.int builtins.float builtins.complex
    builtins.bool builtins.len builtins.abs builtins.chr builtins.ord builtins.hex builtins.oct
    builtins.bin builtins.repr builtins.ascii builtins.round builtins.divmod builtins.hash
    builtins.format builtins.slice builtins.range builtins.pow builtins.memoryview
    builtins.isinstance builtins.issubclass builtins.callable builtins.id
    codecs.decode codecs.encode codecs.escape_decode codecs.escape_encode codecs.lookup
    codecs.latin_1_decode codecs.latin_1_encode codecs.utf_8_decode codecs.utf_8_encode
    codecs.raw_unicode_escape_decode codecs.unicode_escape_decode codecs.ascii_decode
    codecs.charmap_decode codecs.readbuffer_encode
    ast.literal_eval os.fspath os.fsencode os.fsdecode gzip.decompress gzip.compress
    bz2.decompress bz2.compress lzma.decompress lzma.compress
    builtins.bytes.fromhex builtins.bytearray.fromhex builtins.str.maketrans
    builtins.bytes.maketrans builtins.int.from_bytes builtins.dict.fromkeys
""".split())
CONSUMING = {"builtins.str.join", "builtins.bytes.join", "builtins.bytearray.join", "builtins.bytes",
             "builtins.bytearray", "builtins.sum", "builtins.dict.fromkeys", "builtins.int.from_bytes",
             "math.fsum", "math.prod", "builtins.bytearray.extend", "heapq.heapify", "builtins.str.maketrans",
             "statistics.mean", "statistics.fmean", "statistics.median", "statistics.mode",
             "statistics.stdev", "statistics.variance", "statistics.pstdev", "statistics.pvariance",
             "statistics.geometric_mean", "statistics.harmonic_mean", "statistics.multimode",
             "statistics.median_low", "statistics.median_high", "statistics.quantiles"}
_UNSAFE_PURE = {"random.seed", "time.sleep", "uuid.getnode", "uuid.uuid1", "uuid._ip_getnode",
                "uuid._ifconfig_getnode", "uuid._arp_getnode", "uuid._lanscan_getnode",
                "uuid._netstat_getnode", "uuid._unix_getnode", "uuid._popen", "time.tzset",
                "datetime.datetime.now", "random.SystemRandom", "calendar.prmonth", "calendar.prcal",
                "calendar.main", "copy.copy", "copy.deepcopy", "statistics.mean"}
_FMT_WIDTH = re.compile(r"(?:%|\{[^}]*:)[^a-zA-Z%}]*?\d{5,}|%[^a-zA-Z%]*\*|\{[^}]*:[^}]*\{")


def _plain_size(v, depth=0):
    if isinstance(v, (str, bytes, bytearray)):
        return len(v)
    if isinstance(v, (list, tuple, set, frozenset)) and depth < 3:
        return sum(_plain_size(x, depth + 1) for x in v) + len(v)
    if isinstance(v, dict) and depth < 3:
        return sum(_plain_size(k, depth + 1) + _plain_size(x, depth + 1) for k, x in v.items())
    if isinstance(v, int) and not isinstance(v, bool):
        return v.bit_length() // 8
    return 8


def size_ok(name, args, kwargs):
    """Refuse evaluations whose result could be huge or slow."""
    if name in _UNSAFE_PURE:
        return False
    total = sum(_plain_size(a) for a in args) + sum(_plain_size(v) for v in kwargs.values())
    if total > MAX_BYTES:
        return False
    short = name.rsplit(".", 1)[-1]
    nums = [a for a in list(args) + list(kwargs.values()) if isinstance(a, int) and not isinstance(a, bool)]
    if short in ("mul", "__mul__", "__rmul__", "imul", "repeat"):
        seq = [a for a in args if isinstance(a, (str, bytes, bytearray, list, tuple))]
        if seq and nums:
            return max(1, len(seq[0])) * max(0, nums[0]) <= MAX_BYTES
        if len(nums) >= 2:
            return sum(n.bit_length() for n in nums) < 1_000_000
        return True
    if short in ("pow", "__pow__", "ipow", "lshift", "__lshift__", "ilshift", "factorial", "comb",
                 "perm", "ldexp"):
        return all(abs(n) < 100_000 for n in nums[1:]) and (not nums or abs(nums[0]).bit_length() < 100_000)
    if short in ("center", "ljust", "rjust", "zfill", "expandtabs", "__format__", "format"):
        if any(abs(n) > MAX_BYTES for n in nums):
            return False
        if short in ("format", "__format__") and any(isinstance(a, str) and (len(a) > 4096 or _FMT_WIDTH.search(a))
                                                     for a in args):
            return False
    if short in ("mod", "__mod__", "imod", "format", "format_map", "str", "__format__"):
        for a in args:
            if isinstance(a, (str, bytes, bytearray)):
                if len(a) > 4096:
                    return False
                if _FMT_WIDTH.search(a if isinstance(a, str) else bytes(a).decode("latin-1")):
                    return False
    if short in ("replace",) and len(args) >= 3:
        s, old, new = args[0], args[1], args[2]
        if isinstance(s, (str, bytes, bytearray)) and isinstance(new, (str, bytes, bytearray)):
            if (len(s) + 1) * (len(new) + 1) > MAX_BYTES * 4 and len(old) < 2:
                return False
    if name in ("builtins.bytes", "builtins.bytearray", "builtins.range") and nums:
        return all(n <= MAX_BYTES for n in nums)
    if short in ("decompress",):
        return True  # handled by the guarded handlers below
    if name.startswith("struct."):
        fmt = args[0] if args else ""
        try:
            import struct

            return struct.calcsize(fmt) <= MAX_BYTES
        except Exception:  # noqa: BLE001
            return True
    if name.startswith("math.") and nums and any(abs(n) > 10**6 for n in nums):
        return False
    if short in ("factorial", "comb", "perm", "isqrt", "lcm", "gcd") and any(abs(n) > 10**4 for n in nums):
        return False
    if short == "to_bytes" and any(abs(n) > MAX_BYTES for n in nums):
        return False
    return True


FILE_METHODS = {}


def _register_handlers():
    from modelgate import handlers

    return handlers.HANDLERS, handlers.NEW_HANDLERS, handlers.INIT_ATTRS


HANDLERS, NEW_HANDLERS, INIT_ATTRS = {}, {}, {}


def _late_init():
    h, n, i = _register_handlers()
    HANDLERS.update(h)
    NEW_HANDLERS.update(n)
    INIT_ATTRS.update(i)
    from modelgate import handlers

    FILE_METHODS.update(handlers.FILE_METHODS)


_late_init()
