"""Callables whose behaviour the simulator models explicitly.

Each handler receives (sem, args, kwargs, how) and returns the value the real
call would return, reporting findings for capabilities it reaches and raising
SimError where the real call would raise.  Pure data transformations run the
real stdlib function on plain values only (never on symbolic objects), with
size limits.
"""
from __future__ import annotations

import base64
import binascii
import bz2
import codecs
import lzma
import operator
import struct
import zlib

from modelgate.semantics import MAX_BYTES, NOTFOUND, handler, short
from modelgate.values import (
    AttrGetter, BoundMethod, ClassDict, ClassObj, CodeObj, Coroutine, FileObj, FunctionObj,
    Instance, ItemGetter, LazyIter, MethodCaller, Module, ModuleDict, Opaque, PByteArray,
    PDict, PList, PSet, Partial, Ref, SimError, SysModules, describe, is_hashable, is_native,
)

H = handler


# ---------------------------------------------------------------- helpers
def arg(args, kwargs, i, name, default=NOTFOUND):
    if i is not None and i < len(args):
        return args[i]
    if name is not None and name in kwargs:
        return kwargs[name]
    if default is NOTFOUND:
        raise SimError(f"missing argument {name}")
    return default


def callable_value(v):
    if isinstance(v, (Ref, BoundMethod, Partial, AttrGetter, ItemGetter, MethodCaller, ClassObj,
                      FunctionObj, Opaque)):
        return True
    if isinstance(v, Instance):
        return True
    return False


def as_bytes(v):
    """Concrete bytes for a bytes-like value, else None."""
    if isinstance(v, bytes):
        return v
    if isinstance(v, PByteArray):
        return bytes(v.b)
    return None


def guard_size(n):
    if n > MAX_BYTES:
        raise SimError("value too large")


def plain(v):
    """Real Python value for plain data (lists/dicts converted), else NOTFOUND."""
    if is_native(v):
        return v
    if isinstance(v, PByteArray):
        return bytearray(v.b)
    if isinstance(v, PList):
        out = [plain(x) for x in v.items]
        return NOTFOUND if any(x is NOTFOUND for x in out) else out
    if isinstance(v, PDict):
        out = {}
        for k, x in v.d.items():
            px = plain(x)
            if px is NOTFOUND or not is_native(k):
                return NOTFOUND
            out[k] = px
        return out
    return NOTFOUND


def wrap(v):
    """Symbolic value for a real Python value produced by a pure function."""
    if is_native(v) or isinstance(v, (slice, range)):
        return v
    if isinstance(v, bytearray):
        return PByteArray(v)
    if isinstance(v, memoryview):
        return bytes(v)
    if isinstance(v, list):
        return PList([wrap(x) for x in v])
    if isinstance(v, dict):
        return PDict({k: wrap(x) for k, x in v.items()})
    if isinstance(v, set):
        return PSet(v)
    if isinstance(v, tuple):
        return tuple(wrap(x) for x in v)
    return Opaque(type(v).__name__)


def pure(fn, args, kwargs, desc):
    pa = [plain(a) for a in args]
    pk = {k: plain(v) for k, v in kwargs.items()}
    if any(a is NOTFOUND for a in pa) or any(v is NOTFOUND for v in pk.values()):
        return Opaque(f"result of {desc}")
    try:
        r = fn(*pa, **pk)
    except (MemoryError, OverflowError, RecursionError):
        raise SimError(f"{desc} failed") from None
    except Exception as exc:
        raise SimError(f"{desc} raised {type(exc).__name__}") from None
    if isinstance(r, (bytes, bytearray, str)):
        guard_size(len(r))
    return wrap(r)


def _limited_numbers(args, kwargs, limit=1 << 22):
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, int) and not isinstance(a, bool) and abs(a) > limit:
            raise SimError("argument too large to model")


def to_str(sem, v):
    if isinstance(v, str):
        return v
    if is_native(v):
        return str(v)
    return Opaque("str")


# ---------------------------------------------------------------- getattr family
@H("builtins.getattr")
def _getattr(sem, args, kwargs, how):
    if kwargs or not 2 <= len(args) <= 3:
        raise SimError("getattr expects 2 or 3 arguments")
    try:
        return sem.getattr(args[0], args[1])
    except SimError:
        if len(args) == 3:
            return args[2]
        raise


@H("builtins.setattr")
def _setattr(sem, args, kwargs, how):
    if kwargs or len(args) != 3:
        raise SimError("setattr expects 3 arguments")
    sem.setattr(args[0], args[1], args[2])
    return None


@H("builtins.delattr")
def _delattr(sem, args, kwargs, how):
    if len(args) != 2:
        raise SimError("delattr expects 2 arguments")
    sem.delattr(args[0], args[1])
    return None


@H("builtins.hasattr")
def _hasattr(sem, args, kwargs, how):
    if len(args) != 2:
        raise SimError("hasattr expects 2 arguments")
    try:
        sem.getattr(args[0], args[1])
        return True
    except SimError:
        return False


@H("builtins.vars")
def _vars(sem, args, kwargs, how):
    if not args:
        return Opaque("locals")
    o = args[0]
    if isinstance(o, Module):
        return ModuleDict(o)
    if isinstance(o, Instance):
        if o.d is None:
            raise SimError("vars() argument must have __dict__")
        return o.d
    if isinstance(o, Ref) and o.kind == "c":
        return ClassDict(o)
    if isinstance(o, ClassObj):
        return o.ns
    if isinstance(o, Ref) and o.kind == "?":
        return ModuleDict(sem.module(o.oid))
    if isinstance(o, Opaque):
        return Opaque("vars")
    raise SimError("vars() argument must have __dict__")


@H("builtins.dir")
def _dir(sem, args, kwargs, how):
    return PList([])


@H("builtins.__import__", "importlib.__import__", "importlib._bootstrap.__import__")
def _dunder_import(sem, args, kwargs, how):
    name = arg(args, kwargs, 0, "name")
    fromlist = arg(args, kwargs, 3, "fromlist", ())
    level = arg(args, kwargs, 4, "level", 0)
    if not isinstance(name, str):
        raise SimError("__import__ name must be str")
    if level not in (0, None):
        raise SimError("relative import")
    mod = sem.import_module(name)
    if sem.truthy(fromlist) and fromlist is not None:
        return mod
    top = name.split(".")[0]
    return sem.import_module(top) if top != name else mod


@H("importlib.import_module", "importlib._bootstrap._gcd_import")
def _import_module(sem, args, kwargs, how):
    name = arg(args, kwargs, 0, "name")
    package = arg(args, kwargs, 1, "package", None)
    if not isinstance(name, str):
        raise SimError("module name must be str")
    if name.startswith("."):
        if not isinstance(package, str):
            raise SimError("relative import requires package")
        level = len(name) - len(name.lstrip("."))
        base = package.rsplit(".", level - 1)[0] if level > 1 else package
        name = f"{base}.{name.lstrip('.')}" if name.lstrip(".") else base
    return sem.import_module(name)


@H("importlib.reload")
def _reload(sem, args, kwargs, how):
    m = arg(args, kwargs, 0, "module")
    if not isinstance(m, Module):
        raise SimError("reload() argument must be a module")
    return m


def resolve_dotted(sem, name, colon_ok=True):
    if not isinstance(name, str):
        raise SimError("name must be str")
    if ":" in name and colon_ok:
        pkg, _, obj = name.partition(":")
        cur = sem.import_module(pkg)
        parts = obj.split(".") if obj else []
    else:
        parts = name.split(".")
        modname = parts.pop(0)
        cur = sem.import_module(modname)
        if isinstance(cur, Module) and not cur.indexed:
            v = Ref(name, "?")
            return v
        while parts:
            s = f"{modname}.{parts[0]}"
            try:
                m = sem.import_module(s)
            except SimError:
                break
            if isinstance(m, Module) and not m.indexed and isinstance(cur, Module) and cur.indexed:
                break
            cur = m
            modname = s
            parts.pop(0)
    for p in parts:
        cur = sem.getattr(cur, p)
    return cur


@H("pkgutil.resolve_name")
def _resolve_name(sem, args, kwargs, how):
    return resolve_dotted(sem, arg(args, kwargs, 0, "name"))


@H("pydoc.locate", "pydoc.safeimport")
def _locate(sem, args, kwargs, how):
    try:
        return resolve_dotted(sem, arg(args, kwargs, 0, "path"), colon_ok=False)
    except SimError:
        return None


@H("logging.config._resolve", "logging.config.BaseConfigurator.resolve", "unittest.mock._importer",
   "unittest.mock._dot_lookup")
def _cfg_resolve(sem, args, kwargs, how):
    name = args[-1] if args else kwargs.get("s", kwargs.get("target"))
    return resolve_dotted(sem, name, colon_ok=False)


# ---------------------------------------------------------------- operator
@H("operator.attrgetter")
def _attrgetter(sem, args, kwargs, how):
    if not args or kwargs or not all(isinstance(a, str) for a in args):
        raise SimError("attrgetter expects attribute names")
    return AttrGetter(sem.ref("operator.attrgetter"), args)


@H("operator.attrgetter.__new__")
def _attrgetter_new(sem, args, kwargs, how):
    return _attrgetter(sem, args[1:], kwargs, how)


@H("operator.itemgetter")
def _itemgetter(sem, args, kwargs, how):
    if not args or kwargs:
        raise SimError("itemgetter expects items")
    return ItemGetter(sem.ref("operator.itemgetter"), args)


@H("operator.itemgetter.__new__")
def _itemgetter_new(sem, args, kwargs, how):
    return _itemgetter(sem, args[1:], kwargs, how)


@H("operator.methodcaller")
def _methodcaller(sem, args, kwargs, how):
    if not args or not isinstance(args[0], str):
        raise SimError("methodcaller needs a method name")
    return MethodCaller(sem.ref("operator.methodcaller"), args[0], args[1:], kwargs)


@H("operator.methodcaller.__new__")
def _methodcaller_new(sem, args, kwargs, how):
    return _methodcaller(sem, args[1:], kwargs, how)


@H("operator.call")
def _op_call(sem, args, kwargs, how):
    if not args:
        raise SimError("call() needs a callable")
    return sem.call(args[0], args[1:], kwargs, how + " via operator.call")


@H("operator.getitem", "operator.__getitem__")
def _op_getitem(sem, args, kwargs, how):
    if len(args) != 2:
        raise SimError("getitem expects 2 arguments")
    return sem.getitem(args[0], args[1])


@H("operator.setitem", "operator.__setitem__")
def _op_setitem(sem, args, kwargs, how):
    if len(args) != 3:
        raise SimError("setitem expects 3 arguments")
    sem.setitem(args[0], args[1], args[2])
    return None


@H("operator.delitem", "operator.__delitem__")
def _op_delitem(sem, args, kwargs, how):
    if len(args) != 2:
        raise SimError("delitem expects 2 arguments")
    o, k = args
    if isinstance(o, SysModules) and isinstance(k, str):
        o.overlay.pop(k, None)
        o.deleted.add(k)
    elif isinstance(o, PDict) and is_hashable(k):
        o.d.pop(k, None)
    elif isinstance(o, ModuleDict) and isinstance(k, str):
        sem.delattr(o.module, k)
    return None


@H("operator.contains", "operator.countOf", "operator.indexOf", "operator.__contains__")
def _op_contains(sem, args, kwargs, how):
    if len(args) != 2:
        raise SimError("expects 2 arguments")
    items = sem.iterate(args[0])
    if args[1] in [x for x in items if is_native(x)] if is_native(args[1]) else False:
        return True
    return Opaque("comparison")


_BINOPS = {
    "add": operator.add, "concat": operator.add, "iadd": operator.add, "iconcat": operator.add,
    "sub": operator.sub, "isub": operator.sub, "mul": operator.mul, "imul": operator.mul,
    "mod": operator.mod, "imod": operator.mod, "truediv": operator.truediv, "floordiv": operator.floordiv,
    "and_": operator.and_, "or_": operator.or_, "xor": operator.xor, "iand": operator.and_,
    "ior": operator.or_, "ixor": operator.xor, "lshift": operator.lshift, "rshift": operator.rshift,
    "pow": operator.pow, "eq": operator.eq, "ne": operator.ne, "lt": operator.lt, "le": operator.le,
    "gt": operator.gt, "ge": operator.ge, "is_": operator.is_, "is_not": operator.is_not,
}


def safe_binop(name, a, b):
    fn = _BINOPS[name]
    if name in ("mul", "imul"):
        for x, y in ((a, b), (b, a)):
            if isinstance(x, (str, bytes, tuple, list, bytearray)) and isinstance(y, int):
                guard_size(len(x) * max(y, 0))
    if name in ("pow", "lshift"):
        if isinstance(b, int) and abs(b) > 4096:
            raise SimError("exponent too large to model")
    return fn(a, b)


def make_binop(name):
    def h(sem, args, kwargs, how):
        if len(args) != 2:
            raise SimError(f"{name} expects 2 arguments")
        a, b = args
        if isinstance(a, PList) and isinstance(b, PList) and name in ("add", "concat", "iadd", "iconcat"):
            return PList(a.items + b.items)
        return pure(lambda x, y: safe_binop(name, x, y), (a, b), {}, f"operator.{name}")
    return h


for _n in _BINOPS:
    H(f"operator.{_n}", f"operator.__{_n.rstrip('_')}__")(make_binop(_n))


def _unop(fn, name):
    def h(sem, args, kwargs, how):
        if len(args) != 1:
            raise SimError(f"{name} expects 1 argument")
        return pure(fn, args, {}, name)
    return h


for _n, _f in {"neg": operator.neg, "pos": operator.pos, "invert": operator.invert, "inv": operator.inv,
               "not_": operator.not_, "truth": operator.truth, "abs": operator.abs, "index": operator.index}.items():
    H(f"operator.{_n}")(_unop(_f, _n))


# ---------------------------------------------------------------- builtins
@H("builtins.iter")
def _iter(sem, args, kwargs, how):
    if len(args) == 2:
        if not callable_value(args[0]):
            raise SimError("iter(v, w): v must be callable")
        return LazyIter("callable_iter", func=args[0], sentinel=args[1])
    if len(args) != 1:
        raise SimError("iter expects 1 or 2 arguments")
    o = args[0]
    if isinstance(o, LazyIter):
        return o
    if isinstance(o, (Opaque, Instance)):
        return LazyIter("pairs", items=[Opaque("element")]) if isinstance(o, Opaque) else LazyIter(
            "list", items=sem.iterate(o))
    return LazyIter("list", items=sem.iterate(o))


@H("builtins.iterator.__next__", "builtins.next")
def _next(sem, args, kwargs, how):
    if not args:
        raise SimError("next expects an iterator")
    it = args[0]
    if not isinstance(it, LazyIter):
        if isinstance(it, Opaque):
            return Opaque("element")
        raise SimError("object is not an iterator")
    items = sem.run_lazy(it, 1)
    if items:
        return items[0]
    if len(args) > 1:
        return args[1]
    raise SimError("StopIteration")


@H("builtins.map")
def _map(sem, args, kwargs, how):
    if len(args) < 2 or kwargs:
        raise SimError("map() must have at least two arguments")
    for s in args[1:]:
        _check_iterable(sem, s)
    return LazyIter("map", func=args[0], iterables=list(args[1:]))


@H("builtins.map.__new__")
def _map_new(sem, args, kwargs, how):
    return _map(sem, args[1:], kwargs, how)


def _check_iterable(sem, s):
    if isinstance(s, (int, float, complex, bool, type(None), Module, Ref)) and not (
            isinstance(s, Ref) and s.kind == "?"):
        raise SimError(f"{describe(s)} is not iterable")


@H("builtins.filter")
def _filter(sem, args, kwargs, how):
    if len(args) != 2 or kwargs:
        raise SimError("filter expected 2 arguments")
    _check_iterable(sem, args[1])
    return LazyIter("filter", func=args[0], iterable=args[1])


@H("builtins.filter.__new__")
def _filter_new(sem, args, kwargs, how):
    return _filter(sem, args[1:], kwargs, how)


@H("builtins.zip")
def _zip(sem, args, kwargs, how):
    for s in args:
        _check_iterable(sem, s)
    return LazyIter("zip", iterables=list(args))


@H("builtins.zip.__new__")
def _zip_new(sem, args, kwargs, how):
    return _zip(sem, args[1:], kwargs, how)


@H("builtins.enumerate", "builtins.enumerate.__new__")
def _enumerate(sem, args, kwargs, how):
    if args and isinstance(args[0], Ref) and args[0].oid == "builtins.enumerate":
        args = args[1:]
    it = arg(args, kwargs, 0, "iterable")
    _check_iterable(sem, it)
    return LazyIter("enumerate", iterable=it, start=arg(args, kwargs, 1, "start", 0))


@H("builtins.reversed", "builtins.reversed.__new__")
def _reversed(sem, args, kwargs, how):
    if args and isinstance(args[0], Ref) and args[0].oid == "builtins.reversed":
        args = args[1:]
    if len(args) != 1:
        raise SimError("reversed expected 1 argument")
    if isinstance(args[0], LazyIter):
        raise SimError("argument to reversed() must be a sequence")
    return LazyIter("reversed", seq=args[0])


def _consume_to_list(sem, args, kwargs, how, name):
    if not args:
        return []
    if len(args) > 1:
        raise SimError(f"{name} expected at most 1 argument")
    return sem.iterate(args[0])


@H("builtins.list")
def _list(sem, args, kwargs, how):
    return PList(_consume_to_list(sem, args, kwargs, how, "list"))


@H("builtins.tuple", "builtins.tuple.__new__")
def _tuple(sem, args, kwargs, how):
    if args and isinstance(args[0], Ref) and args[0].oid == "builtins.tuple":
        args = args[1:]
    return tuple(_consume_to_list(sem, args, kwargs, how, "tuple"))


@H("builtins.set")
def _set(sem, args, kwargs, how):
    items = _consume_to_list(sem, args, kwargs, how, "set")
    if not all(is_hashable(x) for x in items):
        raise SimError("unhashable type")
    return PSet(items)


@H("builtins.frozenset", "builtins.frozenset.__new__")
def _frozenset(sem, args, kwargs, how):
    if args and isinstance(args[0], Ref) and args[0].oid == "builtins.frozenset":
        args = args[1:]
    items = _consume_to_list(sem, args, kwargs, how, "frozenset")
    if not all(is_hashable(x) for x in items):
        raise SimError("unhashable type")
    return frozenset(items)


@H("builtins.dict")
def _dict(sem, args, kwargs, how):
    d = PDict()
    if args:
        src = args[0]
        if isinstance(src, PDict):
            d.d.update(src.d)
        elif isinstance(src, (ModuleDict, SysModules, Opaque, Instance)):
            pass
        else:
            for item in sem.iterate(src):
                pair = sem.iterate(item) if not isinstance(item, tuple) else list(item)
                if len(pair) != 2:
                    raise SimError("dictionary update sequence element has wrong length")
                if not is_hashable(pair[0]):
                    raise SimError("unhashable type")
                d.d[pair[0]] = pair[1]
    d.d.update(kwargs)
    return d


@H("builtins.list.__new__", "builtins.dict.__new__", "builtins.set.__new__",
   "builtins.bytearray.__new__", "collections.deque.__new__", "collections.OrderedDict.__new__")
def _empty_new(sem, args, kwargs, how):
    cls = args[0] if args else None
    oid = cls.oid if isinstance(cls, Ref) else ""
    if oid == "builtins.list":
        return PList()
    if oid == "builtins.dict":
        return PDict()
    if oid == "builtins.set":
        return PSet()
    if oid == "builtins.bytearray":
        return PByteArray()
    return Instance(cls) if cls is not None else Opaque("object")


def _collection_ctor(path):
    def h(sem, args, kwargs, how):
        items = []
        if args and not isinstance(args[0], (PDict, Opaque, Instance)):
            items = sem.iterate(args[0])
        if path == "heapq.heapify":
            return None
        inst = Instance(sem.ref(path), extra={"constructed": True})
        inst.extra["iterable"] = items
        if args and isinstance(args[0], PDict):
            inst.extra["items"] = dict(args[0].d)
        return inst
    return h


for _p in ("collections.OrderedDict", "collections.Counter", "collections.deque", "collections.ChainMap",
           "collections.UserDict", "collections.UserList", "heapq.heapify"):
    H(_p)(_collection_ctor(_p))


@H("collections.defaultdict")
def _defaultdict(sem, args, kwargs, how):
    fac = args[0] if args else None
    if fac is not None and not callable_value(fac):
        raise SimError("first argument must be callable or None")
    inst = Instance(sem.ref("collections.defaultdict"))
    inst.extra["factory"] = fac
    inst.extra["items"] = {}
    if len(args) > 1 and isinstance(args[1], PDict):
        inst.extra["items"].update(args[1].d)
    return inst


@H("collections.defaultdict.__missing__")
def _dd_missing(sem, args, kwargs, how):
    if len(args) != 2:
        raise SimError("__missing__ expects 1 argument")
    inst = args[0]
    fac = inst.extra.get("factory") if isinstance(inst, Instance) else None
    if fac is None:
        raise SimError("KeyError")
    return sem.call(fac, (), {}, how + " via defaultdict.__missing__")


@H("builtins.sorted")
def _sorted(sem, args, kwargs, how):
    if len(args) != 1:
        raise SimError("sorted expected 1 argument")
    items = sem.iterate(args[0])
    key = kwargs.get("key")
    if key is not None:
        for x in items:
            sem.call(key, (x,), {}, how + " via sorted(key=)")
    return PList(items)


def _minmax(name):
    def h(sem, args, kwargs, how):
        if not args:
            raise SimError(f"{name} expected at least 1 argument")
        items = sem.iterate(args[0]) if len(args) == 1 else list(args)
        key = kwargs.get("key")
        if not items:
            if "default" in kwargs:
                return kwargs["default"]
            raise SimError(f"{name}() arg is an empty sequence")
        if key is not None:
            for x in items:
                sem.call(key, (x,), {}, how + f" via {name}(key=)")
        nat = [x for x in items if is_native(x)]
        if len(nat) == len(items):
            try:
                return (min if name == "min" else max)(nat)
            except Exception:
                pass
        return items[0]
    return h


H("builtins.min")(_minmax("min"))
H("builtins.max")(_minmax("max"))


@H("builtins.sum", "math.fsum", "math.prod", "statistics.mean", "statistics.median", "statistics.fmean")
def _sum(sem, args, kwargs, how):
    if not args:
        raise SimError("sum expected an iterable")
    items = sem.iterate(args[0])
    if all(isinstance(x, (int, float)) for x in items):
        try:
            return sum(items) + (args[1] if len(args) > 1 and isinstance(args[1], (int, float)) else 0)
        except Exception:
            pass
    return Opaque("sum")


@H("builtins.any", "builtins.all")
def _anyall(sem, args, kwargs, how):
    if len(args) != 1:
        raise SimError("expected 1 argument")
    items = sem.iterate(args[0])
    vals = [sem.truthy(x) for x in items]
    return any(vals) if "any" in how else all(vals)


@H("builtins.len")
def _len(sem, args, kwargs, how):
    if len(args) != 1:
        raise SimError("len expects 1 argument")
    o = args[0]
    if isinstance(o, (str, bytes, tuple, frozenset, range)):
        return len(o)
    if isinstance(o, PList):
        return len(o.items)
    if isinstance(o, PDict):
        return len(o.d)
    if isinstance(o, PSet):
        return len(o.s)
    if isinstance(o, PByteArray):
        return len(o.b)
    if isinstance(o, Instance) and isinstance(o.cls, ClassObj):
        f = sem.classobj_raw(o.cls, "__len__")
        if f is not NOTFOUND:
            return sem.call(f, (o,), {}, how + " via __len__")
    if isinstance(o, (int, float, Module, Ref, LazyIter)):
        raise SimError("object has no len()")
    return Opaque("len")


def _dunder_route(sem, o, dunder, how):
    if isinstance(o, Instance) and isinstance(o.cls, ClassObj):
        f = sem.classobj_raw(o.cls, dunder)
        if f is not NOTFOUND:
            return sem.call(f, (o,), {}, how + f" via {dunder}")
    return NOTFOUND


@H("builtins.repr", "builtins.ascii")
def _repr(sem, args, kwargs, how):
    if len(args) != 1:
        raise SimError("repr expects 1 argument")
    r = _dunder_route(sem, args[0], "__repr__", how)
    if r is not NOTFOUND:
        return r
    if is_native(args[0]):
        return pure(repr, args, {}, "repr")
    return Opaque("repr string")


@H("builtins.str")
def _str(sem, args, kwargs, how):
    if not args and not kwargs:
        return ""
    o = arg(args, kwargs, 0, "object", "")
    if len(args) > 1 or "encoding" in kwargs or "errors" in kwargs:
        b = as_bytes(o)
        if b is None:
            return Opaque("str")
        return pure(lambda *a, **k: str(*a, **k), (b,) + tuple(args[1:]),
                    {k: v for k, v in kwargs.items() if k != "object"}, "str")
    r = _dunder_route(sem, o, "__str__", how)
    if r is not NOTFOUND:
        return r
    if is_native(o):
        return pure(str, (o,), {}, "str")
    if isinstance(o, PByteArray):
        return str(bytearray(o.b))
    return Opaque("str")


@H("builtins.str.__new__")
def _str_new(sem, args, kwargs, how):
    return _str(sem, args[1:], kwargs, how)


@H("builtins.format")
def _format(sem, args, kwargs, how):
    if not args:
        raise SimError("format expects arguments")
    r = _dunder_route(sem, args[0], "__format__", how)
    if r is not NOTFOUND:
        return r
    return pure(format, args, kwargs, "format")


@H("builtins.hash")
def _hash(sem, args, kwargs, how):
    if len(args) != 1:
        raise SimError("hash expects 1 argument")
    r = _dunder_route(sem, args[0], "__hash__", how)
    if r is not NOTFOUND:
        return r
    if not is_hashable(args[0]):
        raise SimError("unhashable type")
    return Opaque("hash")


@H("builtins.print")
def _print(sem, args, kwargs, how):
    for a in args:
        _dunder_route(sem, a, "__str__", how)
    f = kwargs.get("file")
    if isinstance(f, FileObj):
        pass
    return None


@H("builtins.input")
def _input(sem, args, kwargs, how):
    return Opaque("input line")


@H("builtins.breakpoint", "builtins.help", "builtins.copyright", "builtins.credits")
def _harmless(sem, args, kwargs, how):
    return None


@H("builtins.exit", "builtins.quit", "sys.exit", "os._exit", "os.abort", "os.kill", "os.killpg",
   "signal.raise_signal", "_thread.interrupt_main", "threading._thread.interrupt_main")
def _exit(sem, args, kwargs, how):
    raise SimError("the load exits the interpreter")


@H("builtins.callable")
def _callable(sem, args, kwargs, how):
    return callable_value(args[0]) if args else False


@H("builtins.isinstance", "builtins.issubclass")
def _isinstance(sem, args, kwargs, how):
    return Opaque("bool")


@H("builtins.id")
def _id(sem, args, kwargs, how):
    return 140000000000000


def _make_pure_builtin(fn, name):
    def h(sem, args, kwargs, how):
        _limited_numbers(args, kwargs, 1 << 31)
        return pure(fn, args, kwargs, name)
    return h


for _n, _f in {"chr": chr, "ord": ord, "hex": hex, "oct": oct, "bin": bin, "abs": abs, "round": round,
               "divmod": divmod, "float": float, "complex": complex, "bool": bool}.items():
    H(f"builtins.{_n}")(_make_pure_builtin(_f, _n))


@H("builtins.pow")
def _pow(sem, args, kwargs, how):
    if len(args) >= 2 and isinstance(args[1], int) and abs(args[1]) > 4096 and len(args) < 3:
        raise SimError("exponent too large to model")
    return pure(pow, args, kwargs, "pow")


@H("builtins.int", "builtins.int.__new__")
def _int(sem, args, kwargs, how):
    if args and isinstance(args[0], Ref) and args[0].oid == "builtins.int":
        args = args[1:]
    if args and isinstance(args[0], (str, bytes)) and len(args[0]) > 5000:
        raise SimError("int string too long")
    return pure(int, args, kwargs, "int")


@H("builtins.bytes", "builtins.bytes.__new__")
def _bytes(sem, args, kwargs, how):
    if args and isinstance(args[0], Ref) and args[0].oid == "builtins.bytes":
        args = args[1:]
    if not args and not kwargs:
        return b""
    src = arg(args, kwargs, 0, "source", b"")
    if isinstance(src, int) and not isinstance(src, bool):
        guard_size(src)
        return bytes(src)
    if isinstance(src, (LazyIter, PList, tuple)) and len(args) == 1:
        items = sem.iterate(src)
        if all(isinstance(x, int) and 0 <= x < 256 for x in items):
            return bytes(items)
        return Opaque("bytes")
    if isinstance(src, PByteArray):
        return bytes(src.b)
    return pure(bytes, args, kwargs, "bytes")


@H("builtins.bytearray")
def _bytearray(sem, args, kwargs, how):
    r = _bytes(sem, args, kwargs, how)
    if isinstance(r, bytes):
        return PByteArray(r)
    return Opaque("bytearray")


@H("builtins.memoryview")
def _memoryview(sem, args, kwargs, how):
    if len(args) != 1:
        raise SimError("memoryview expects 1 argument")
    b = as_bytes(args[0])
    if b is None:
        if isinstance(args[0], Opaque):
            return Opaque("memoryview")
        raise SimError("memoryview: a bytes-like object is required")
    return b


@H("builtins.range")
def _range(sem, args, kwargs, how):
    return pure(range, args, kwargs, "range")


@H("builtins.slice")
def _slice(sem, args, kwargs, how):
    if not 1 <= len(args) <= 3:
        raise SimError("slice expected 1 to 3 arguments")
    if all(is_native(a) for a in args):
        return slice(*args)
    return Opaque("slice")


@H("builtins.object")
def _object(sem, args, kwargs, how):
    if args or kwargs:
        raise SimError("object() takes no arguments")
    return Instance(sem.ref("builtins.object"), d=None)


@H("builtins.object.__new__", "copyreg.__newobj__")
def _object_new(sem, args, kwargs, how):
    if not args:
        raise SimError("object.__new__(): not enough arguments")
    return sem.new_object(args[0], args[1:], kwargs, how) if not (
        isinstance(args[0], Ref) and args[0].oid == "builtins.object") else Instance(args[0], d=None)


@H("copyreg.__newobj_ex__")
def _newobj_ex(sem, args, kwargs, how):
    if len(args) != 3:
        raise SimError("__newobj_ex__ expects 3 arguments")
    cls, a, k = args
    return sem.new_object(cls, tuple(sem.iterate(a)), dict(k.d) if isinstance(k, PDict) else {}, how)


@H("builtins.type")
def _type(sem, args, kwargs, how):
    if len(args) == 1:
        o = args[0]
        if isinstance(o, Instance):
            return o.cls
        for t, oid in ((PList, "builtins.list"), (PDict, "builtins.dict"), (PSet, "builtins.set"),
                       (PByteArray, "builtins.bytearray")):
            if isinstance(o, t):
                return sem.ref(oid)
        if is_native(o):
            from modelgate.semantics import BUILTIN_TYPE_OIDS
            return sem.ref(BUILTIN_TYPE_OIDS.get(type(o), "builtins.object"))
        if isinstance(o, Module):
            return sem.ref("types.ModuleType")
        if isinstance(o, (Ref, ClassObj)):
            return sem.ref("builtins.type")
        return Opaque("type")
    if len(args) != 3:
        raise SimError("type() takes 1 or 3 arguments")
    name, bases, ns = args
    if not isinstance(name, str) or not isinstance(bases, tuple) or not isinstance(ns, PDict):
        raise SimError("type() argument types")
    cls = ClassObj(name, bases, PDict(ns.d))
    for b in bases:
        if isinstance(b, Ref) and b.oid in sem.ctx.class_overlay:
            pass
    init_sub = None
    for b in bases:
        if isinstance(b, ClassObj):
            v = sem.classobj_raw(b, "__init_subclass__")
            if v is not NOTFOUND:
                init_sub = v
    if init_sub is not None:
        sem.call(init_sub, (cls,), {}, how + " via __init_subclass__")
    for v in list(ns.d.values()):
        if isinstance(v, Instance) and isinstance(v.cls, ClassObj):
            sn = sem.classobj_raw(v.cls, "__set_name__")
            if sn is not NOTFOUND:
                sem.call(sn, (v, cls, "attr"), {}, how + " via __set_name__")
    return cls


@H("builtins.type.__new__")
def _type_new(sem, args, kwargs, how):
    return _type(sem, args[1:], kwargs, how)


@H("builtins.type.__call__")
def _type_call(sem, args, kwargs, how):
    if not args:
        raise SimError("type.__call__ needs a class")
    return sem.call(args[0], args[1:], kwargs, how)


@H("types.new_class")
def _new_class(sem, args, kwargs, how):
    name = arg(args, kwargs, 0, "name")
    bases = arg(args, kwargs, 1, "bases", ())
    body = arg(args, kwargs, 3, "exec_body", None)
    ns = PDict()
    if body is not None:
        sem.call(body, (ns,), {}, how + " via types.new_class(exec_body=)")
    return ClassObj(name if isinstance(name, str) else "C", bases if isinstance(bases, tuple) else (), ns)


@H("builtins.__build_class__")
def _build_class(sem, args, kwargs, how):
    if len(args) < 2:
        raise SimError("__build_class__: not enough arguments")
    ns = PDict()
    sem.call(args[0], (), {}, how + " via __build_class__ body")
    return ClassObj(args[1] if isinstance(args[1], str) else "C", tuple(args[2:]), ns)


@H("builtins.super", "builtins.property", "builtins.classmethod")
def _misc_wrappers(sem, args, kwargs, how):
    return Opaque("descriptor")


@H("builtins.staticmethod")
def _staticmethod(sem, args, kwargs, how):
    if len(args) != 1:
        raise SimError("staticmethod expects 1 argument")
    return Partial(sem.ref("builtins.staticmethod"), args[0])


@H("builtins.compile")
def _compile(sem, args, kwargs, how):
    sem.check_cap(sem.ref("builtins.compile"), args, kwargs, how)
    return CodeObj("compile()")


@H("builtins.eval", "builtins.exec")
def _eval(sem, args, kwargs, how):
    if not args:
        raise SimError("eval expected at least 1 argument")
    src = args[0]
    if not isinstance(src, (str, bytes, CodeObj, Opaque, PByteArray)):
        raise SimError("eval() arg 1 must be a string, bytes or code object")
    sem.ctx.flag("A", f"{how}: evaluates {short(src, 70)}")
    return Opaque("result of eval")


@H("types.FunctionType", "types.FunctionType.__new__")
def _function_type(sem, args, kwargs, how):
    if args and isinstance(args[0], Ref) and args[0].oid == "types.FunctionType":
        args = args[1:]
    code = arg(args, kwargs, 0, "code")
    if not isinstance(code, (CodeObj, Opaque)):
        raise SimError("function() argument 'code' must be code")
    sem.ctx.flag("A", f"{how}: types.FunctionType materialises {describe(code)} into a function")
    return FunctionObj(code)


@H("types.CodeType", "types.CodeType.__new__", "types.CodeType.replace", "dill._dill._create_code")
def _code_type(sem, args, kwargs, how):
    return CodeObj("CodeType")


@H("types.MethodType", "types.MethodType.__new__")
def _method_type(sem, args, kwargs, how):
    if args and isinstance(args[0], Ref) and args[0].oid == "types.MethodType":
        args = args[1:]
    if len(args) != 2:
        raise SimError("method expected 2 arguments")
    if not callable_value(args[0]):
        raise SimError("first argument must be callable")
    return BoundMethod(args[1], args[0])


@H("types.SimpleNamespace")
def _simplens(sem, args, kwargs, how):
    inst = Instance(sem.ref("types.SimpleNamespace"))
    if args and isinstance(args[0], PDict):
        inst.d.d.update(args[0].d)
    inst.d.d.update(kwargs)
    return inst


@H("types.ModuleType", "types.ModuleType.__new__")
def _module_type(sem, args, kwargs, how):
    if args and isinstance(args[0], Ref) and args[0].oid == "types.ModuleType":
        args = args[1:]
    name = arg(args, kwargs, 0, "name", "module")
    if not isinstance(name, str):
        raise SimError("module name must be str")
    return Module(name, indexed=False, fresh=True)


@H("types.MappingProxyType")
def _mappingproxy(sem, args, kwargs, how):
    return args[0] if args else Opaque("mappingproxy")


# ---------------------------------------------------------------- functools
@H("functools.partial")
def _partial(sem, args, kwargs, how):
    if not args:
        raise SimError("partial() needs a callable")
    f = args[0]
    if not callable_value(f):
        raise SimError("the first argument must be callable")
    if isinstance(f, Partial) and isinstance(f.cls, Ref) and f.cls.oid == "functools.partial" and f.d is not None \
            and not f.d.d:
        kw = dict(f.kw)
        kw.update(kwargs)
        return Partial(sem.ref("functools.partial"), f.func, f.args + tuple(args[1:]), kw)
    return Partial(sem.ref("functools.partial"), f, args[1:], kwargs)


@H("functools.partial.__new__")
def _partial_new(sem, args, kwargs, how):
    return _partial(sem, args[1:], kwargs, how)


@H("functools.partial.__setstate__")
def _partial_setstate(sem, args, kwargs, how):
    if len(args) != 2:
        raise SimError("__setstate__ expects 1 argument")
    p, state = args
    if not isinstance(p, Partial):
        raise SimError("descriptor requires a partial")
    if not isinstance(state, tuple) or len(state) != 4:
        raise SimError("invalid partial state")
    fn, fnargs, kw, d = state
    if not callable_value(fn) or not isinstance(fnargs, tuple) or not (kw is None or isinstance(kw, PDict)):
        raise SimError("invalid partial state")
    if not (d is None or isinstance(d, PDict)):
        raise SimError("invalid partial state")
    p.func = fn
    p.args = fnargs
    p.kw = dict(kw.d) if isinstance(kw, PDict) else {}
    p.d = d if isinstance(d, PDict) else PDict()
    return None


@H("functools.partial.__call__")
def _partial_call(sem, args, kwargs, how):
    if not args:
        raise SimError("descriptor needs an argument")
    return sem.call(args[0], args[1:], kwargs, how)


@H("functools.reduce")
def _reduce(sem, args, kwargs, how):
    if not 2 <= len(args) <= 3:
        raise SimError("reduce expected 2 or 3 arguments")
    f, it = args[0], args[1]
    items = sem.iterate(it)
    if len(args) == 3:
        acc, rest = args[2], items
    else:
        if not items:
            raise SimError("reduce() of empty iterable with no initial value")
        acc, rest = items[0], items[1:]
    for x in rest:
        acc = sem.call(f, (acc, x), {}, how + " via functools.reduce")
    return acc


@H("functools.lru_cache")
def _lru_cache(sem, args, kwargs, how):
    a = arg(args, kwargs, 0, "maxsize", 128)
    if callable_value(a) and not isinstance(a, (int, type(None))):
        return Partial(sem.ref("functools._lru_cache_wrapper"), a)
    return Partial(sem.ref("functools.partial"), sem.ref("functools.cache"))


@H("functools.cache", "functools._lru_cache_wrapper", "functools.singledispatch", "contextlib.contextmanager",
   "functools.total_ordering", "functools.cached_property", "atexit.unregister")
def _wrap_func(sem, args, kwargs, how):
    if not args:
        raise SimError("missing function")
    f = args[0]
    if not callable_value(f):
        raise SimError("argument must be callable")
    if "contextmanager" in how:
        return Opaque("context manager factory")
    return Partial(sem.ref("functools._lru_cache_wrapper"), f)


@H("functools.update_wrapper")
def _update_wrapper(sem, args, kwargs, how):
    return arg(args, kwargs, 0, "wrapper")


@H("functools.wraps")
def _wraps(sem, args, kwargs, how):
    return Partial(sem.ref("functools.partial"), sem.ref("functools.update_wrapper"), (),
                   {"wrapped": arg(args, kwargs, 0, "wrapped")})


@H("functools.cmp_to_key")
def _cmp_to_key(sem, args, kwargs, how):
    return Opaque("key wrapper class")


# ---------------------------------------------------------------- itertools
def _lazy(kind, **params):
    return LazyIter(kind, **params)


@H("itertools.starmap", "itertools.starmap.__new__")
def _starmap(sem, args, kwargs, how):
    if args and isinstance(args[0], Ref) and args[0].oid == "itertools.starmap":
        args = args[1:]
    if len(args) != 2:
        raise SimError("starmap expected 2 arguments")
    return _lazy("starmap", func=args[0], iterable=args[1])


@H("itertools.filterfalse", "itertools.dropwhile", "itertools.takewhile")
def _pred_iter(sem, args, kwargs, how):
    if len(args) != 2:
        raise SimError("expected 2 arguments")
    kind = "filterfalse" if "filterfalse" in how else ("dropwhile" if "dropwhile" in how else "takewhile")
    return _lazy(kind, func=args[0], iterable=args[1])


@H("itertools.accumulate")
def _accumulate(sem, args, kwargs, how):
    it = arg(args, kwargs, 0, "iterable")
    return _lazy("accumulate", iterable=it, func=arg(args, kwargs, 1, "func", None),
                 initial=kwargs.get("initial"))


@H("itertools.groupby")
def _groupby(sem, args, kwargs, how):
    return _lazy("groupby", iterable=arg(args, kwargs, 0, "iterable"), func=arg(args, kwargs, 1, "key", None))


@H("itertools.chain")
def _chain(sem, args, kwargs, how):
    return _lazy("chain", iterables=list(args))


@H("itertools.chain.from_iterable")
def _chain_from(sem, args, kwargs, how):
    if args and isinstance(args[0], Ref) and args[0].oid == "itertools.chain":
        args = args[1:]
    return _lazy("chain_from", iterable=args[0] if args else ())


@H("itertools.islice")
def _islice(sem, args, kwargs, how):
    if not args:
        raise SimError("islice expected arguments")
    return _lazy("islice", iterable=args[0], args=tuple(a if isinstance(a, int) or a is None else None
                                                         for a in args[1:]))


@H("itertools.repeat")
def _repeat(sem, args, kwargs, how):
    return _lazy("repeat", obj=arg(args, kwargs, 0, "object"), times=arg(args, kwargs, 1, "times", None))


@H("itertools.count")
def _count(sem, args, kwargs, how):
    return _lazy("count")


@H("itertools.cycle")
def _cycle(sem, args, kwargs, how):
    return _lazy("cycle", iterable=args[0] if args else ())


@H("itertools.zip_longest", "itertools.product", "itertools.compress", "itertools.pairwise",
   "itertools.batched", "itertools.permutations", "itertools.combinations",
   "itertools.combinations_with_replacement")
def _zip_like(sem, args, kwargs, how):
    return _lazy("zip", iterables=[a for a in args if not isinstance(a, int)])


@H("itertools.tee")
def _tee(sem, args, kwargs, how):
    src = args[0] if args else ()
    items = sem.iterate(src) if isinstance(src, LazyIter) else None
    n = args[1] if len(args) > 1 and isinstance(args[1], int) else 2
    if items is None:
        return tuple(_lazy("list", items=sem.iterate(src)) for _ in range(min(n, 8)))
    return tuple(_lazy("list", items=list(items)) for _ in range(min(n, 8)))


@H("heapq.nsmallest", "heapq.nlargest")
def _nsmallest(sem, args, kwargs, how):
    if len(args) < 2:
        raise SimError("expected 2 arguments")
    items = sem.iterate(args[1])
    key = kwargs.get("key", args[2] if len(args) > 2 else None)
    if key is not None:
        for x in items:
            sem.call(key, (x,), {}, how + " via key=")
    return PList(items)


@H("heapq.merge")
def _hmerge(sem, args, kwargs, how):
    key = kwargs.get("key")
    if key is None:
        return _lazy("chain", iterables=list(args))
    return _lazy("map", func=key, iterables=[_lazy("chain", iterables=list(args))])


# ---------------------------------------------------------------- copy / copyreg
@H("copy.copy", "copy.deepcopy")
def _copy(sem, args, kwargs, how):
    if not args:
        raise SimError("copy expects an argument")
    x = args[0]
    if isinstance(x, PList):
        return PList(x.items)
    if isinstance(x, PDict):
        return PDict(x.d)
    if isinstance(x, PSet):
        return PSet(x.s)
    if isinstance(x, PByteArray):
        return PByteArray(x.b)
    if isinstance(x, Partial):
        p = Partial(x.cls, x.func, x.args, x.kw)
        p.d = PDict(x.d.d) if x.d is not None else None
        return p
    if isinstance(x, Instance) and not isinstance(x, (AttrGetter, ItemGetter, MethodCaller)):
        cls = x.cls
        if isinstance(cls, ClassObj):
            for hook in ("__copy__", "__deepcopy__", "__reduce_ex__", "__reduce__"):
                f = sem.classobj_raw(cls, hook)
                if f is not NOTFOUND:
                    return sem.call(f, (x,) if hook in ("__copy__", "__reduce__") else (x, PDict() if
                                    hook == "__deepcopy__" else 4), {}, how + f" via {hook}")
        y = Instance(cls, d=PDict(x.d.d) if x.d is not None else None, extra=dict(x.extra))
        return y
    return x


@H("copyreg._reconstructor")
def _reconstructor(sem, args, kwargs, how):
    if len(args) != 3:
        raise SimError("_reconstructor expects 3 arguments")
    cls, base, state = args
    if isinstance(base, Ref) and base.oid == "builtins.object":
        return sem.new_object(cls, (), {}, how)
    sem.new_object(base, (state,), {}, how + " via copyreg._reconstructor")
    if isinstance(base, (Ref, ClassObj)):
        init_hit = sem.class_lookup(base, "__init__")
        if init_hit is not NOTFOUND:
            tag, init = init_hit
            if not (isinstance(init, Ref) and init.oid in ("builtins.object.__init__",)):
                sem.call(base, (state,), {}, how + " via copyreg._reconstructor (base.__init__)")
    return sem.new_object(cls, (), {}, how) if isinstance(cls, (Ref, ClassObj)) else Opaque("object")


@H("copyreg.add_extension")
def _add_extension(sem, args, kwargs, how):
    if len(args) != 3:
        raise SimError("add_extension expects 3 arguments")
    module, name, code = args
    if not isinstance(code, int) or not 1 <= code <= 0x7FFFFFFF:
        raise SimError("code out of range")
    sem.ctx.extensions[code] = (module, name)
    return None


@H("copyreg.remove_extension", "copyreg.clear_extension_cache")
def _remove_extension(sem, args, kwargs, how):
    if len(args) == 3 and isinstance(args[2], int):
        sem.ctx.extensions.pop(args[2], None)
    return None


@H("copyreg.pickle", "copyreg.constructor")
def _copyreg_pickle(sem, args, kwargs, how):
    return None


# ---------------------------------------------------------------- container methods
def _self(args):
    if not args:
        raise SimError("descriptor needs an argument")
    return args[0], args[1:]


def _mapping_get(sem, o, key):
    if isinstance(o, PDict):
        if not is_hashable(key):
            raise SimError("unhashable key")
        return o.d.get(key, NOTFOUND)
    try:
        return sem.getitem(o, key)
    except SimError:
        return NOTFOUND


@H("builtins.dict.get")
def _dict_get(sem, args, kwargs, how):
    o, rest = _self(args)
    if not rest:
        raise SimError("get expected at least 1 argument")
    v = _mapping_get(sem, o, rest[0])
    if v is NOTFOUND:
        return rest[1] if len(rest) > 1 else None
    return v


@H("builtins.dict.__getitem__")
def _dict_getitem(sem, args, kwargs, how):
    o, rest = _self(args)
    if len(rest) != 1:
        raise SimError("__getitem__ expects 1 argument")
    return sem.getitem(o, rest[0])


@H("builtins.dict.__setitem__", "builtins.list.__setitem__")
def _dict_setitem(sem, args, kwargs, how):
    o, rest = _self(args)
    if len(rest) != 2:
        raise SimError("__setitem__ expects 2 arguments")
    sem.setitem(o, rest[0], rest[1])
    return None


@H("builtins.dict.setdefault")
def _dict_setdefault(sem, args, kwargs, how):
    o, rest = _self(args)
    if not rest:
        raise SimError("setdefault expects arguments")
    v = _mapping_get(sem, o, rest[0])
    if v is NOTFOUND:
        v = rest[1] if len(rest) > 1 else None
        sem.setitem(o, rest[0], v)
    return v


@H("builtins.dict.pop")
def _dict_pop(sem, args, kwargs, how):
    o, rest = _self(args)
    if not rest:
        raise SimError("pop expects arguments")
    v = _mapping_get(sem, o, rest[0])
    if v is NOTFOUND:
        if len(rest) > 1:
            return rest[1]
        raise SimError("KeyError")
    if isinstance(o, PDict):
        o.d.pop(rest[0], None)
    return v


@H("builtins.dict.update")
def _dict_update(sem, args, kwargs, how):
    o, rest = _self(args)
    if rest:
        src = rest[0]
        if isinstance(src, PDict):
            for k, v in src.d.items():
                sem.setitem(o, k, v)
        elif not isinstance(src, (Opaque, ModuleDict, SysModules)):
            for item in sem.iterate(src):
                pair = list(item) if isinstance(item, tuple) else sem.iterate(item)
                if len(pair) == 2:
                    sem.setitem(o, pair[0], pair[1])
    for k, v in kwargs.items():
        sem.setitem(o, k, v)
    return None


@H("builtins.dict.keys", "builtins.dict.values", "builtins.dict.items", "builtins.dict.copy")
def _dict_views(sem, args, kwargs, how):
    o, rest = _self(args)
    if isinstance(o, PDict):
        if how.endswith("copy") or "copy" in how:
            return PDict(o.d)
        return LazyIter("list", items=list(o.d.items()) if "items" in how else (
            list(o.d.values()) if "values" in how else list(o.d.keys())))
    return Opaque("view")


@H("builtins.dict.fromkeys")
def _dict_fromkeys(sem, args, kwargs, how):
    items = sem.iterate(args[1]) if len(args) > 1 else []
    v = args[2] if len(args) > 2 else None
    d = PDict()
    for k in items:
        if is_hashable(k):
            d.d[k] = v
    return d


@H("builtins.list.append")
def _list_append(sem, args, kwargs, how):
    o, rest = _self(args)
    if len(rest) != 1:
        raise SimError("append expects 1 argument")
    if isinstance(o, PList):
        o.items.append(rest[0])
    return None


@H("builtins.list.extend", "builtins.list.__iadd__")
def _list_extend(sem, args, kwargs, how):
    o, rest = _self(args)
    if len(rest) != 1:
        raise SimError("extend expects 1 argument")
    items = sem.iterate(rest[0])
    if isinstance(o, PList):
        o.items.extend(items)
    return o if "iadd" in how else None


@H("builtins.list.insert")
def _list_insert(sem, args, kwargs, how):
    o, rest = _self(args)
    if len(rest) == 2 and isinstance(o, PList) and isinstance(rest[0], int):
        o.items.insert(rest[0], rest[1])
    return None


@H("builtins.list.sort")
def _list_sort(sem, args, kwargs, how):
    o, rest = _self(args)
    key = kwargs.get("key")
    if key is not None and isinstance(o, PList):
        for x in list(o.items):
            sem.call(key, (x,), {}, how + " via list.sort(key=)")
    return None


@H("builtins.list.pop")
def _list_pop(sem, args, kwargs, how):
    o, rest = _self(args)
    if isinstance(o, PList):
        if not o.items:
            raise SimError("pop from empty list")
        i = rest[0] if rest and isinstance(rest[0], int) else -1
        try:
            return o.items.pop(i)
        except IndexError:
            raise SimError("pop index out of range") from None
    return Opaque("element")


@H("builtins.list.__getitem__", "builtins.tuple.__getitem__", "builtins.str.__getitem__",
   "builtins.bytes.__getitem__")
def _seq_getitem(sem, args, kwargs, how):
    o, rest = _self(args)
    if len(rest) != 1:
        raise SimError("__getitem__ expects 1 argument")
    return sem.getitem(o, rest[0])


@H("builtins.set.add")
def _set_add(sem, args, kwargs, how):
    o, rest = _self(args)
    if len(rest) != 1:
        raise SimError("add expects 1 argument")
    if isinstance(o, PSet):
        if not is_hashable(rest[0]):
            raise SimError("unhashable type")
        o.s.add(rest[0])
    return None


@H("builtins.set.update")
def _set_update(sem, args, kwargs, how):
    o, rest = _self(args)
    for r in rest:
        items = sem.iterate(r)
        if isinstance(o, PSet):
            for x in items:
                if not is_hashable(x):
                    raise SimError("unhashable type")
                o.s.add(x)
    return None


@H("builtins.str.join", "builtins.bytes.join", "builtins.bytearray.join")
def _join(sem, args, kwargs, how):
    o, rest = _self(args)
    if len(rest) != 1:
        raise SimError("join expects 1 argument")
    items = sem.iterate(rest[0])
    if isinstance(o, str) and all(isinstance(x, str) for x in items):
        guard_size(sum(len(x) for x in items) + len(o) * len(items))
        return o.join(items)
    sep = as_bytes(o)
    if sep is not None:
        bs = [as_bytes(x) for x in items]
        if all(b is not None for b in bs):
            guard_size(sum(len(b) for b in bs) + len(sep) * len(bs))
            r = sep.join(bs)
            return PByteArray(r) if isinstance(o, PByteArray) else r
    if isinstance(o, (str, bytes)) and any(is_native(x) and not isinstance(x, type(o)) for x in items):
        raise SimError("sequence item: expected str instance")
    return Opaque("joined string")


_SIZE_METHODS = {"center", "ljust", "rjust", "zfill", "expandtabs", "__mul__", "__rmul__", "to_bytes"}


def native_method_call(sem, fn_oid, args, kwargs):
    """Call a method of str/bytes/int/float/tuple/bytearray on plain values."""
    tname, _, mname = fn_oid.rpartition(".")
    tmap = {"builtins.str": str, "builtins.bytes": bytes, "builtins.int": int, "builtins.float": float,
            "builtins.tuple": tuple, "builtins.frozenset": frozenset, "builtins.bytearray": bytearray,
            "builtins.complex": complex, "builtins.bool": bool}
    t = tmap.get(tname)
    if t is None:
        return NOTFOUND
    if mname in _SIZE_METHODS:
        _limited_numbers(args[1:], kwargs, 1 << 22)
    if mname == "replace" and args and isinstance(args[0], (str, bytes)) and len(args) > 2 \
            and isinstance(args[2], (str, bytes)):
        guard_size(len(args[0]) * (len(args[2]) + 1))
    try:
        meth = getattr(t, mname)
    except AttributeError:
        return NOTFOUND
    if mname in ("format", "format_map"):
        if not all(is_native(a) for a in args) or not all(is_native(v) for v in kwargs.values()):
            return Opaque("formatted string")
    return pure(meth, args, kwargs, fn_oid)


# ---------------------------------------------------------------- in-memory files
@H("io.BytesIO", "_io.BytesIO", "io.StringIO", "_io.StringIO")
def _bytesio(sem, args, kwargs, how):
    init = arg(args, kwargs, 0, "initial_bytes", None)
    if init is None:
        init = kwargs.get("initial_value", b"")
    text = "StringIO" in how
    if init is None:
        init = "" if text else b""
    b = as_bytes(init)
    if text and isinstance(init, str):
        return FileObj(init, text=True)
    if b is None:
        if isinstance(init, str):
            raise SimError("a bytes-like object is required")
        if isinstance(init, Opaque):
            return Opaque("in-memory file with unknown content")
        raise SimError("a bytes-like object is required")
    return FileObj(b)


@H("io.BufferedReader", "io.BufferedRandom", "io.TextIOWrapper", "io.BufferedRWPair", "codecs.EncodedFile",
   "codecs.getreader")
def _wrap_file(sem, args, kwargs, how):
    return args[0] if args and isinstance(args[0], FileObj) else Opaque("file wrapper")


def _file_method(name):
    def h(sem, args, kwargs, how):
        f, rest = _self(args)
        if not isinstance(f, FileObj):
            return Opaque(f"file.{name}")
        data = f.data
        if name in ("read", "read1"):
            n = rest[0] if rest and isinstance(rest[0], int) and rest[0] >= 0 else None
            out = data[f.pos:] if n is None else data[f.pos:f.pos + n]
            f.pos += len(out)
            return out
        if name == "readline":
            nl = "\n" if f.text else b"\n"
            i = data.find(nl, f.pos)
            end = len(data) if i < 0 else i + 1
            out = data[f.pos:end]
            f.pos = end
            return out
        if name == "readlines":
            out = data[f.pos:].splitlines(True)
            f.pos = len(data)
            return PList(out)
        if name == "getvalue" or name == "getbuffer":
            return data
        if name == "seek":
            off = rest[0] if rest and isinstance(rest[0], int) else 0
            whence = rest[1] if len(rest) > 1 and isinstance(rest[1], int) else 0
            base = 0 if whence == 0 else (f.pos if whence == 1 else len(data))
            f.pos = max(0, base + off)
            return f.pos
        if name == "tell":
            return f.pos
        if name == "write":
            v = rest[0] if rest else b""
            if f.text and isinstance(v, str):
                f.data = data[:f.pos] + v + data[f.pos + len(v):]
                f.pos += len(v)
                return len(v)
            b = as_bytes(v)
            if b is not None and not f.text:
                f.data = data[:f.pos] + b + data[f.pos + len(b):]
                f.pos += len(b)
                return len(b)
            return Opaque("count")
        if name in ("__enter__",):
            return f
        return None
    return h


for _n in ("read", "read1", "readline", "readlines", "getvalue", "getbuffer", "seek", "tell", "write", "close",
           "flush", "__enter__", "__exit__"):
    H(f"io.BytesIO.{_n}", f"io.StringIO.{_n}")(_file_method(_n))


# ---------------------------------------------------------------- data transforms
def _cap_decompress(kind, data, wbits=None):
    if kind == "zlib":
        d = zlib.decompressobj(wbits if wbits is not None else zlib.MAX_WBITS)
        out = d.decompress(data, MAX_BYTES)
        if d.unconsumed_tail:
            raise SimError("decompressed value too large")
        if not d.eof:
            raise SimError("incomplete compressed data")
        return out
    if kind == "gzip":
        from modelgate import containers
        out = containers.gunzip(data, MAX_BYTES)
        if out is None:
            raise SimError("bad gzip data")
        return out
    if kind == "bz2":
        out = []
        total = 0
        while data:
            d = bz2.BZ2Decompressor()
            chunk = d.decompress(data, MAX_BYTES - total)
            total += len(chunk)
            out.append(chunk)
            if not d.eof:
                raise SimError("bz2 data incomplete or too large")
            data = d.unused_data
        return b"".join(out)
    if kind == "lzma":
        out = []
        total = 0
        while data:
            d = lzma.LZMADecompressor()
            chunk = d.decompress(data, MAX_BYTES - total)
            total += len(chunk)
            out.append(chunk)
            if not d.eof:
                raise SimError("lzma data incomplete or too large")
            data = d.unused_data
        return b"".join(out)
    raise SimError("unknown compression")


def _decompress_handler(kind):
    def h(sem, args, kwargs, how):
        data = as_bytes(arg(args, kwargs, 0, "data"))
        if data is None:
            return Opaque(f"{kind}-decompressed data")
        wbits = arg(args, kwargs, 1, "wbits", None) if kind == "zlib" else None
        try:
            return _cap_decompress(kind, data, wbits if isinstance(wbits, int) else None)
        except SimError:
            raise
        except Exception as exc:
            raise SimError(f"{kind} decompress failed: {exc}") from None
    return h


H("zlib.decompress")(_decompress_handler("zlib"))
H("gzip.decompress")(_decompress_handler("gzip"))
H("bz2.decompress")(_decompress_handler("bz2"))
H("lzma.decompress")(_decompress_handler("lzma"))

def _codec(sem, args, kwargs, how, encode):
    obj = arg(args, kwargs, 0, "obj")
    enc = arg(args, kwargs, 1, "encoding", "utf-8")
    errors = arg(args, kwargs, 2, "errors", "strict")
    if not isinstance(enc, str) or not isinstance(errors, str):
        return Opaque("codec result")
    norm = enc.lower().replace("-", "_").replace(" ", "_")
    if isinstance(obj, PByteArray):
        obj = bytes(obj.b)
    if not is_native(obj):
        return Opaque("codec result")
    if norm in ("zlib", "zip", "zlib_codec", "bz2", "bz2_codec"):
        if encode:
            return pure(codecs.encode, (obj, enc, errors), {}, "codecs.encode")
        b = as_bytes(obj)
        if b is None:
            raise SimError("bytes required")
        return _cap_decompress("zlib" if norm.startswith("z") else "bz2", b)
    try:
        codecs.lookup(enc)
    except LookupError:
        raise SimError(f"unknown encoding {enc!r}") from None
    return pure(codecs.encode if encode else codecs.decode, (obj, enc, errors), {},
                "codecs.encode" if encode else "codecs.decode")


@H("codecs.encode", "_codecs.encode")
def _codecs_encode(sem, args, kwargs, how):
    return _codec(sem, args, kwargs, how, True)


@H("codecs.decode", "_codecs.decode")
def _codecs_decode(sem, args, kwargs, how):
    return _codec(sem, args, kwargs, how, False)


def _make_pure(fn, name):
    def h(sem, args, kwargs, how):
        return pure(fn, args, kwargs, name)
    return h


for _name in ("b64decode", "b64encode", "standard_b64decode", "standard_b64encode", "urlsafe_b64decode",
              "urlsafe_b64encode", "b32decode", "b32encode", "b32hexdecode", "b32hexencode", "b16decode",
              "b16encode", "a85decode", "a85encode", "b85decode", "b85encode", "decodebytes", "encodebytes",
              "z85decode", "z85encode"):
    if hasattr(base64, _name):
        H(f"base64.{_name}")(_make_pure(getattr(base64, _name), f"base64.{_name}"))
for _name in ("a2b_base64", "b2a_base64", "a2b_hex", "b2a_hex", "hexlify", "unhexlify", "a2b_uu", "b2a_uu",
              "a2b_qp", "b2a_qp", "crc32", "crc_hqx"):
    if hasattr(binascii, _name):
        H(f"binascii.{_name}")(_make_pure(getattr(binascii, _name), f"binascii.{_name}"))
for _name, _fn in (("zlib.compress", zlib.compress), ("zlib.crc32", zlib.crc32), ("zlib.adler32", zlib.adler32),
                   ("struct.pack", struct.pack), ("struct.unpack", struct.unpack),
                   ("struct.calcsize", struct.calcsize), ("bz2.compress", bz2.compress),
                   ("lzma.compress", lzma.compress)):
    H(_name)(_make_pure(_fn, _name))


@H("gzip.compress")
def _gzip_compress(sem, args, kwargs, how):
    import gzip
    return pure(gzip.compress, args, {}, "gzip.compress")


@H("json.loads")
def _json_loads(sem, args, kwargs, how):
    import json
    if kwargs:
        return Opaque("json value")
    return pure(json.loads, args, {}, "json.loads")


@H("ast.literal_eval")
def _literal_eval(sem, args, kwargs, how):
    import ast
    s = args[0] if args else None
    if isinstance(s, (str, bytes)) and len(s) > 100000:
        return Opaque("literal")
    return pure(ast.literal_eval, args, {}, "ast.literal_eval")


@H("ast.parse")
def _ast_parse(sem, args, kwargs, how):
    return Opaque("AST")


@H("builtins.bytes.fromhex", "builtins.bytearray.fromhex")
def _fromhex(sem, args, kwargs, how):
    s = args[-1] if args else None
    if not isinstance(s, str):
        raise SimError("fromhex() argument must be str")
    try:
        b = bytes.fromhex(s)
    except ValueError:
        raise SimError("non-hexadecimal number found in fromhex() arg") from None
    return PByteArray(b) if "bytearray" in how or (args and isinstance(args[0], Ref)
                                                    and args[0].oid == "builtins.bytearray") else b


@H("builtins.int.from_bytes")
def _from_bytes(sem, args, kwargs, how):
    rest = args[1:] if args and isinstance(args[0], Ref) else args
    return pure(int.from_bytes, rest, kwargs, "int.from_bytes")


@H("marshal.loads", "marshal.load")
def _marshal_loads(sem, args, kwargs, how):
    src = arg(args, kwargs, 0, "bytes" if "loads" in how else "file")
    if isinstance(src, FileObj):
        data = src.data[src.pos:]
        src.pos = len(src.data)
    else:
        data = as_bytes(src)
    if data is None:
        if isinstance(src, (Opaque,)):
            return Opaque("unmarshalled value")
        raise SimError("marshal.loads() argument must be bytes-like")
    if not data:
        raise SimError("EOF read where object expected")
    t = data[0] & 0x7F
    if t == ord("c"):
        return CodeObj("marshal.loads")
    try:
        import marshal
        if t in b"sutaAzZiNTFgy":
            return wrap(marshal.loads(data))
    except Exception:
        raise SimError("bad marshal data") from None
    return Opaque("unmarshalled value")


# ---------------------------------------------------------------- nested loaders
def _payload(v):
    """(bytes, FileObj or None) for data handed to a nested loader."""
    if isinstance(v, FileObj):
        d = v.data
        if isinstance(d, str):
            d = d.encode("latin-1", "replace")
        return d[v.pos:], v
    b = as_bytes(v)
    if b is not None:
        return b, None
    return None, None


def nested_load(sem, loader, src, how, opts=None, path_param=True):
    """Run a nested loader on embedded data and return what it loads."""
    opts = opts or {}
    data, fobj = _payload(src)
    if data is None:
        if path_param and Sem_is_path(sem, src):
            sem.ctx.flag("C", f"{how}: nested loader {loader} reads the file {short(src, 60)}")
            return Opaque(f"object loaded from file by {loader}")
        if isinstance(src, (Opaque, Instance)):
            sem.ctx.unsupported(f"{how}: nested loader {loader} fed data the scanner could not reconstruct")
            return Opaque("nested load result")
        raise SimError(f"{loader}: argument is not bytes or a file")
    if sem.ctx.nested is None:
        sem.ctx.unsupported(f"{how}: nested loader {loader} not analysable here")
        return Opaque("nested load result")
    value, consumed = sem.ctx.nested(loader, data, dict(opts, how=how))
    if fobj is not None and isinstance(consumed, int):
        fobj.pos += consumed
    return value


def Sem_is_path(sem, v):
    if isinstance(v, (str, bytes)):
        return True
    if isinstance(v, Instance) and isinstance(v.cls, Ref) and v.cls.oid.startswith("pathlib."):
        return True
    if isinstance(v, int) and not isinstance(v, bool):
        return True
    return False


def _pickle_opts(kwargs, start=1, args=()):
    enc = kwargs.get("encoding", "ASCII")
    fix = kwargs.get("fix_imports", True)
    return {"encoding": enc if isinstance(enc, str) else "ASCII", "fix_imports": bool(fix) if is_native(fix) else True}


@H("pickle.loads")
def _pickle_loads(sem, args, kwargs, how):
    if len(args) != 1:
        raise SimError("loads() takes exactly one positional argument")
    src = args[0]
    if isinstance(src, str):
        raise SimError("a bytes-like object is required, not 'str'")
    return nested_load(sem, "pickle", src, how + " -> pickle.loads", _pickle_opts(kwargs), path_param=False)


@H("pickle._loads")
def _pickle_py_loads(sem, args, kwargs, how):
    if len(args) != 1:
        raise SimError("loads() takes exactly one positional argument")
    return nested_load(sem, "pickle-py", args[0], how + " -> pickle._loads", _pickle_opts(kwargs), path_param=False)


@H("pickle.load", "pickle._load")
def _pickle_load(sem, args, kwargs, how):
    f = arg(args, kwargs, 0, "file")
    if not isinstance(f, (FileObj, Opaque, Instance)):
        raise SimError("file must have read and readline attributes")
    flavor = "pickle-py" if "_load" in how.split(".")[-1] and "pickle._load" in how else "pickle"
    return nested_load(sem, flavor, f, how + " -> pickle.load", _pickle_opts(kwargs), path_param=False)


@H("pickle.Unpickler", "pickle._Unpickler")
def _unpickler(sem, args, kwargs, how):
    f = arg(args, kwargs, 0, "file")
    inst = Instance(sem.ref("pickle.Unpickler"))
    inst.extra["file"] = f
    inst.extra["opts"] = _pickle_opts(kwargs)
    inst.extra["flavor"] = "pickle-py" if "_Unpickler" in how else "pickle"
    return inst


@H("pickle.Unpickler.load", "pickle._Unpickler.load")
def _unpickler_load(sem, args, kwargs, how):
    inst = args[0] if args else None
    if not isinstance(inst, Instance) or "file" not in inst.extra:
        return Opaque("unpickled object")
    return nested_load(sem, inst.extra.get("flavor", "pickle"), inst.extra["file"],
                       how + " -> Unpickler.load", inst.extra.get("opts"), path_param=False)


@H("dill.loads", "dill._dill.loads", "dill.load", "dill._dill.load", "cloudpickle.loads", "cloudpickle.load",
   "srsly.pickle_loads", "joblib.externals.cloudpickle.loads")
def _dill_loads(sem, args, kwargs, how):
    src = arg(args, kwargs, 0, "str")
    return nested_load(sem, "pickle-py" if "dill" in how else "pickle", src, how + " -> nested pickle", {},
                       path_param=False)


@H("numpy.load", "numpy.lib.npyio.load", "numpy.lib._npyio_impl.load")
def _numpy_load(sem, args, kwargs, how):
    from modelgate.semantics import bind
    b = bind("file:1,mmap_mode:1=,allow_pickle:1=,fix_imports:1=,encoding:1=,max_header_size:3=", args, kwargs)
    if b is None:
        raise SimError("numpy.load: bad arguments")
    allow = b.get("allow_pickle", False)
    return nested_load(sem, "numpy", b["file"], how + " -> numpy.load",
                       {"allow_pickle": sem.truthy(allow), "encoding": b.get("encoding", "ASCII")})


@H("numpy.lib.format.read_array", "numpy.lib._format_impl.read_array")
def _read_array(sem, args, kwargs, how):
    from modelgate.semantics import bind
    b = bind("fp:1,allow_pickle:1=,pickle_kwargs:1=,max_header_size:3=", args, kwargs)
    if b is None:
        raise SimError("read_array: bad arguments")
    return nested_load(sem, "npy", b["fp"], how + " -> numpy.lib.format.read_array",
                       {"allow_pickle": sem.truthy(b.get("allow_pickle", False))})


@H("torch.load", "torch.serialization.load")
def _torch_load(sem, args, kwargs, how):
    f = arg(args, kwargs, 0, "f")
    pickle_module = arg(args, kwargs, 2, "pickle_module", None)
    wo = kwargs.get("weights_only", None)
    if wo is None:
        # torch >= 2.6 (the registry's standard loader passes weights_only=False explicitly):
        # the default is a weights-only load unless a pickle_module is given
        wo = pickle_module is None
    if Sem_is_path(sem, f):
        sem.ctx.flag("C", f"{how}: nested loader torch.load reads the file {short(f, 60)}")
        return Opaque("object loaded from file by torch.load")
    if sem.truthy(wo):
        return Opaque("weights-only load (restricted unpickler)")
    return nested_load(sem, "torch", f, how + " -> torch.load", {})


@H("torch.storage._load_from_bytes", "torch.serialization._load_from_bytes")
def _torch_from_bytes(sem, args, kwargs, how):
    b = arg(args, kwargs, 0, "b")
    return nested_load(sem, "torch", b, how + " -> torch.storage._load_from_bytes", {}, path_param=False)


@H("joblib.load", "joblib.numpy_pickle.load", "sklearn.externals.joblib.load")
def _joblib_load(sem, args, kwargs, how):
    return nested_load(sem, "joblib", arg(args, kwargs, 0, "filename"), how + " -> joblib.load", {})


@H("pandas.read_pickle", "pandas.io.pickle.read_pickle")
def _pandas_read_pickle(sem, args, kwargs, how):
    src = arg(args, kwargs, 0, "filepath_or_buffer")
    if isinstance(src, str) and src.split(":", 1)[0].lower() in ("http", "https", "ftp", "s3", "gs", "gcs"):
        sem.ctx.flag("D", f"{how}: pandas.read_pickle fetches {short(src)}")
        return Opaque("remote object")
    return nested_load(sem, "pandas", src, how + " -> pandas.read_pickle", {})


@H("numpy.frombuffer", "numpy.core.multiarray.frombuffer")
def _frombuffer(sem, args, kwargs, how):
    return Instance(sem.ref("numpy.ndarray"))


# ---------------------------------------------------------------- legit model helpers
@H("numpy.dtype")
def _np_dtype(sem, args, kwargs, how):
    inst = Instance(sem.ref("numpy.dtype"), d=None)
    inst.extra["typestr"] = args[0] if args else None
    return inst


@H("numpy.dtype.__setstate__")
def _np_dtype_setstate(sem, args, kwargs, how):
    if args and isinstance(args[0], Instance) and len(args) > 1:
        args[0].extra["state"] = args[1]
    return None


@H("numpy.core.multiarray._reconstruct", "numpy._core.multiarray._reconstruct")
def _np_reconstruct(sem, args, kwargs, how):
    return Instance(sem.ref("numpy.ndarray"), d=None)


@H("numpy.ndarray.__setstate__", "numpy.ma.core.MaskedArray.__setstate__", "numpy.generic.__setstate__")
def _np_setstate(sem, args, kwargs, how):
    return None


@H("torch._tensor._rebuild_from_type_v2", "torch._tensor._rebuild_from_type")
def _rebuild_from_type(sem, args, kwargs, how):
    if len(args) < 3:
        raise SimError("_rebuild_from_type_v2: not enough arguments")
    func, new_type, fargs = args[0], args[1], args[2]
    r = sem.call(func, tuple(sem.iterate(fargs)), {}, how + " via torch._tensor._rebuild_from_type_v2")
    return r


@H("torch._utils._rebuild_parameter_with_state", "torch._utils._rebuild_parameter")
def _rebuild_param(sem, args, kwargs, how):
    return Opaque("tensor")


# ---------------------------------------------------------------- execution routes
def _route(sem, f, fargs, fkw, how):
    if f is None:
        return None
    return sem.call(f, tuple(fargs), dict(fkw), how)


def _as_args(sem, v):
    if v is None or v is NOTFOUND:
        return ()
    if isinstance(v, tuple):
        return v
    return tuple(sem.iterate(v))


def _as_kw(v):
    if isinstance(v, PDict):
        return {k: x for k, x in v.d.items() if isinstance(k, str)}
    return {}


@H("threading.Thread")
def _thread(sem, args, kwargs, how):
    from modelgate.semantics import bind
    b = bind("group:1=,target:1=,name:1=,args:1=,kwargs:1=,daemon:3=", args, kwargs)
    if b is None:
        raise SimError("Thread: bad arguments")
    if b.get("group") is not None:
        raise SimError("group argument must be None for now")
    inst = Instance(sem.ref("threading.Thread"))
    inst.d.d.update({"_target": b.get("target"), "_args": b.get("args", ()), "_kwargs": b.get("kwargs")})
    return inst


@H("threading.Timer")
def _timer(sem, args, kwargs, how):
    from modelgate.semantics import bind
    b = bind("interval:1,function:1,args:1=,kwargs:1=", args, kwargs)
    if b is None:
        raise SimError("Timer: bad arguments")
    inst = Instance(sem.ref("threading.Timer"))
    inst.d.d.update({"function": b["function"], "args": b.get("args") or (), "kwargs": b.get("kwargs") or PDict()})
    return inst


def _thread_like_run(sem, inst, how, target_keys):
    if not isinstance(inst, Instance) or inst.d is None:
        return None
    t, a, k = target_keys
    f = inst.d.d.get(t)
    if f is None:
        return None
    return _route(sem, f, _as_args(sem, inst.d.d.get(a)), _as_kw(inst.d.d.get(k)), how)


@H("threading.Thread.start", "threading.Thread.run", "threading.Thread._bootstrap",
   "threading.Thread._bootstrap_inner")
def _thread_start(sem, args, kwargs, how):
    inst = args[0] if args else None
    if isinstance(inst, Instance) and inst.d is not None and "function" in inst.d.d and "_target" not in inst.d.d:
        _thread_like_run(sem, inst, how + " via threading.Timer", ("function", "args", "kwargs"))
        return None
    _thread_like_run(sem, inst, how + " via threading.Thread", ("_target", "_args", "_kwargs"))
    return None


@H("threading.Timer.run")
def _timer_run(sem, args, kwargs, how):
    _thread_like_run(sem, args[0] if args else None, how + " via threading.Timer", ("function", "args", "kwargs"))
    return None


@H("threading._start_new_thread", "_thread.start_new_thread", "_thread.start_new", "threading._start_joinable_thread")
def _start_new_thread(sem, args, kwargs, how):
    if len(args) < 2:
        raise SimError("start_new_thread expected at least 2 arguments")
    f, a = args[0], args[1]
    if not isinstance(a, tuple):
        raise SimError("2nd arg must be a tuple")
    k = args[2] if len(args) > 2 else None
    _route(sem, f, a, _as_kw(k), how + " via _thread.start_new_thread")
    return 1


@H("multiprocessing.context.Process", "multiprocessing.Process", "multiprocessing.process.BaseProcess")
def _mp_process(sem, args, kwargs, how):
    from modelgate.semantics import bind
    b = bind("group:1=,target:1=,name:1=,args:1=,kwargs:1=,daemon:3=", args, kwargs)
    if b is None:
        raise SimError("Process: bad arguments")
    inst = Instance(sem.ref("multiprocessing.context.Process"))
    inst.d.d.update({"_target": b.get("target"), "_args": b.get("args", ()), "_kwargs": b.get("kwargs")})
    return inst


@H("multiprocessing.process.BaseProcess.start")
def _mp_start(sem, args, kwargs, how):
    sem.ctx.flag("B", f"{how}: multiprocessing Process.start() starts a process")
    _thread_like_run(sem, args[0] if args else None, how + " in a child process", ("_target", "_args", "_kwargs"))
    return None


@H("multiprocessing.process.BaseProcess.run")
def _mp_run(sem, args, kwargs, how):
    _thread_like_run(sem, args[0] if args else None, how + " via Process.run", ("_target", "_args", "_kwargs"))
    return None


@H("concurrent.futures.ThreadPoolExecutor.submit", "concurrent.futures.thread.ThreadPoolExecutor.submit",
   "concurrent.futures.ProcessPoolExecutor.submit", "concurrent.futures.process.ProcessPoolExecutor.submit",
   "multiprocessing.pool.Pool.apply", "multiprocessing.pool.Pool.apply_async")
def _submit(sem, args, kwargs, how):
    if len(args) < 2:
        raise SimError("submit expected a callable")
    if "Process" in how:
        sem.ctx.flag("B", f"{how}: ProcessPoolExecutor.submit starts worker processes")
    if ".apply" in how:
        fargs = _as_args(sem, args[2]) if len(args) > 2 else ()
        fkw = _as_kw(args[3]) if len(args) > 3 else {}
        _route(sem, args[1], fargs, fkw, how + " via Pool.apply")
        return Opaque("result")
    _route(sem, args[1], args[2:], kwargs, how + " via Executor.submit")
    return Opaque("future")


@H("concurrent.futures.Executor.map", "concurrent.futures._base.Executor.map",
   "concurrent.futures.ProcessPoolExecutor.map", "concurrent.futures.process.ProcessPoolExecutor.map",
   "multiprocessing.pool.Pool.map",
   "multiprocessing.pool.Pool.starmap", "multiprocessing.pool.Pool.imap", "multiprocessing.pool.Pool.map_async",
   "multiprocessing.pool.Pool.imap_unordered", "multiprocessing.pool.Pool.starmap_async")
def _executor_map(sem, args, kwargs, how):
    if len(args) < 3:
        raise SimError("map expected a callable and iterables")
    if "Process" in how:
        sem.ctx.flag("B", f"{how}: ProcessPoolExecutor.map starts worker processes")
    star = "starmap" in how
    srcs = [sem.iterate(s) for s in args[2:3 if "Pool." in how else None]]
    n = min((len(s) for s in srcs), default=0)
    for i in range(n):
        fargs = tuple(sem.iterate(srcs[0][i])) if star else tuple(s[i] for s in srcs)
        _route(sem, args[1], fargs, {}, how + " via Executor.map")
    return Opaque("results")


@H("multiprocessing.pool.ThreadPool", "multiprocessing.dummy.Pool", "concurrent.futures.ThreadPoolExecutor",
   "concurrent.futures.ProcessPoolExecutor", "concurrent.futures.thread.ThreadPoolExecutor",
   "concurrent.futures.process.ProcessPoolExecutor")
def _pool_ctor(sem, args, kwargs, how):
    if "ThreadPoolExecutor" in how:
        oid = "concurrent.futures.thread.ThreadPoolExecutor"
    elif "ProcessPoolExecutor" in how:
        oid = "concurrent.futures.process.ProcessPoolExecutor"
    else:
        oid = "multiprocessing.pool.ThreadPool"
    init = kwargs.get("initializer", args[1] if "Pool" in how and len(args) > 1 else None)
    if init is not None and "Process" not in how:
        _route(sem, init, _as_args(sem, kwargs.get("initargs", ())), {}, how + " via pool initializer")
    return Instance(sem.ref(oid))


@H("weakref.finalize")
def _finalize(sem, args, kwargs, how):
    if len(args) < 2:
        raise SimError("finalize expected obj and func")
    target = args[0]
    if isinstance(target, (int, float, complex, str, bytes, tuple, type(None), PList, PDict, PByteArray)):
        raise SimError("cannot create weak reference to this object")
    _route(sem, args[1], args[2:], kwargs, how + " via weakref.finalize (runs at collection or exit)")
    return Instance(sem.ref("weakref.finalize"))


@H("atexit.register")
def _atexit(sem, args, kwargs, how):
    if not args:
        raise SimError("register() takes at least 1 argument")
    _route(sem, args[0], args[1:], kwargs, how + " via atexit.register (runs at exit)")
    return args[0]


@H("sched.scheduler.enter", "sched.scheduler.enterabs")
def _sched_enter(sem, args, kwargs, how):
    from modelgate.semantics import bind
    b = bind("self:1,time:1,priority:1,action:1,argument:1=,kwargs:1=", args, kwargs) or \
        bind("self:1,delay:1,priority:1,action:1,argument:1=,kwargs:1=", args, kwargs)
    if b is None:
        raise SimError("enter: bad arguments")
    inst = b["self"]
    if isinstance(inst, Instance):
        inst.extra.setdefault("events", []).append((b["action"], b.get("argument", ()), b.get("kwargs")))
    return Opaque("event")


@H("sched.scheduler.run")
def _sched_run(sem, args, kwargs, how):
    inst = args[0] if args else None
    if isinstance(inst, Instance):
        for action, a, k in inst.extra.pop("events", []):
            _route(sem, action, _as_args(sem, a), _as_kw(k), how + " via sched.scheduler.run")
    return None


@H("contextlib.ExitStack.callback", "contextlib._BaseExitStack.callback")
def _exitstack_cb(sem, args, kwargs, how):
    if len(args) < 2:
        raise SimError("callback expected a callable")
    inst = args[0]
    if isinstance(inst, Instance):
        inst.extra.setdefault("callbacks", []).append((args[1], args[2:], kwargs))
    return args[1]


@H("contextlib.ExitStack.close", "contextlib.ExitStack.__exit__", "contextlib._BaseExitStack.close")
def _exitstack_close(sem, args, kwargs, how):
    inst = args[0] if args else None
    if isinstance(inst, Instance):
        for f, a, k in reversed(inst.extra.pop("callbacks", [])):
            _route(sem, f, a, k, how + " via ExitStack callback")
    return None


@H("bdb.Bdb.runcall", "profile.Profile.runcall", "cProfile.Profile.runcall", "trace.Trace.runfunc")
def _runcall(sem, args, kwargs, how):
    if len(args) < 2:
        raise SimError("runcall expected a function")
    return _route(sem, args[1], args[2:], kwargs, how + " via runcall")


@H("pdb.runcall")
def _pdb_runcall(sem, args, kwargs, how):
    if not args:
        raise SimError("runcall expected a function")
    return _route(sem, args[0], args[1:], kwargs, how + " via pdb.runcall")


def _timeit_common(sem, b, how, ctor=False):
    for label in ("stmt", "setup"):
        if label not in b:
            continue  # the default "pass" is the library's own source
        v = b[label]
        if isinstance(v, (str, bytes)):
            if (v.decode("latin-1") if isinstance(v, bytes) else v).strip() in ("", "pass"):
                continue  # the documented no-op statement
            sem.ctx.flag("A", f"{how}: timeit compiles and runs {label} {short(v, 60)}")
        elif not ctor:
            _route(sem, v, (), {}, how + f" via timeit ({label} callable)")


@H("timeit.timeit", "timeit.repeat")
def _timeit(sem, args, kwargs, how):
    from modelgate.semantics import bind
    b = bind("stmt:1=,setup:1=,timer:1=,number:1=,globals:1=", args, kwargs) or \
        bind("stmt:1=,setup:1=,timer:1=,repeat:1=,number:1=,globals:1=", args, kwargs)
    if b is None:
        raise SimError("timeit: bad arguments")
    _timeit_common(sem, b, how)
    return 0.0


@H("timeit.Timer")
def _timer_cls(sem, args, kwargs, how):
    from modelgate.semantics import bind
    b = bind("stmt:1=,setup:1=,timer:1=,globals:1=", args, kwargs)
    if b is None:
        raise SimError("Timer: bad arguments")
    _timeit_common(sem, b, how, ctor=True)
    inst = Instance(sem.ref("timeit.Timer"))
    inst.extra["b"] = b
    return inst


@H("timeit.Timer.timeit", "timeit.Timer.repeat", "timeit.Timer.autorange")
def _timer_timeit(sem, args, kwargs, how):
    inst = args[0] if args else None
    if isinstance(inst, Instance) and "b" in inst.extra:
        b = inst.extra["b"]
        for label in ("setup", "stmt"):
            v = b.get(label)
            if label in b and not isinstance(v, (str, bytes)):
                _route(sem, v, (), {}, how + " via Timer.timeit")
    return 0.0


@H("asyncio.run", "asyncio.runners.run", "asyncio.base_events.BaseEventLoop.run_until_complete",
   "asyncio.runners.Runner.run", "asyncio.ensure_future", "asyncio.create_task", "asyncio.tasks.ensure_future",
   "asyncio.wait_for", "asyncio.gather")
def _asyncio_run(sem, args, kwargs, how):
    for c in args:
        if isinstance(c, Instance) and len(args) == 1:
            continue
        run_coroutine(sem, c, how)
    return Opaque("coroutine result")


def run_coroutine(sem, c, how):
    if not isinstance(c, Coroutine):
        return
    ref = c.func
    if isinstance(ref, Ref) and ref.oid in ("asyncio.to_thread", "asyncio.threads.to_thread"):
        if c.args:
            _route(sem, c.args[0], c.args[1:], c.kw, how + " via asyncio.to_thread")
        return
    if isinstance(ref, Ref):
        sem.call_ref(ref, c.args, c.kw, how + " (awaited)", lazy_ok=False)


@H("asyncio.base_events.BaseEventLoop.run_in_executor")
def _run_in_executor(sem, args, kwargs, how):
    if len(args) < 3:
        raise SimError("run_in_executor expects a callable")
    _route(sem, args[2], args[3:], {}, how + " via loop.run_in_executor")
    return Opaque("future")


@H("unittest.mock.Mock", "unittest.mock.MagicMock", "unittest.mock.NonCallableMock",
   "unittest.mock.NonCallableMagicMock", "unittest.mock.AsyncMock")
def _mock(sem, args, kwargs, how):
    inst = Instance(sem.ref("unittest.mock.Mock"))
    se = kwargs.get("side_effect", args[1] if len(args) > 1 else None)
    inst.d.d["side_effect"] = se
    inst.d.d["return_value"] = kwargs.get("return_value", Opaque("mock result"))
    return inst


@H("unittest.mock.Mock.__call__", "unittest.mock.CallableMixin.__call__", "unittest.mock.CallableMixin._mock_call",
   "unittest.mock.CallableMixin._execute_mock_call")
def _mock_call(sem, args, kwargs, how):
    inst = args[0] if args else None
    if isinstance(inst, Instance) and inst.d is not None:
        se = inst.d.d.get("side_effect")
        if se is not None and callable_value(se):
            return sem.call(se, args[1:], kwargs, how + " via Mock(side_effect=)")
        if se is not None and not callable_value(se):
            items = sem.iterate(se) if isinstance(se, (PList, tuple, LazyIter)) else []
            return items[0] if items else Opaque("mock result")
        return inst.d.d.get("return_value", Opaque("mock result"))
    return Opaque("mock result")


@H("logging.config.dictConfig", "logging.config.DictConfigurator.configure")
def _dictconfig(sem, args, kwargs, how):
    cfg = args[-1] if args else None
    if isinstance(cfg, Instance) and "args" in cfg.extra:
        cfg = cfg.extra["args"][0] if cfg.extra["args"] else None
    if not isinstance(cfg, PDict):
        sem.ctx.flag("A", f"{how}: logging.config.dictConfig with a configuration the scanner could not "
                          f"reconstruct (it can instantiate arbitrary callables)")
        return None
    _walk_logging_config(sem, cfg, how)
    return None


def _cfg_value(sem, v, how):
    if isinstance(v, str) and v.startswith("ext://"):
        try:
            return resolve_dotted(sem, v[6:], colon_ok=False)
        except SimError:
            return Opaque("ext value")
    return v


def _cfg_factory(sem, spec: PDict, how, default_cls=None, positional=None):
    d = spec.d
    target = d.get("()", NOTFOUND)
    if target is NOTFOUND:
        target = d.get("class", NOTFOUND)
        skip = {"class", "level", "formatter", "filters", "."}
        if target is NOTFOUND:
            return
    else:
        skip = {"()", "."}
    if isinstance(target, str):
        name = target[6:] if target.startswith("ext://") else target
        try:
            target = resolve_dotted(sem, name, colon_ok=False)
        except SimError:
            return
    kw = {k: _cfg_value(sem, v, how) for k, v in d.items() if isinstance(k, str) and k not in skip}
    if positional is not None and "()" not in d:
        args = tuple(kw.get(p) for p in positional if p in kw)
        sem.call(target, args, {}, how + " via logging.config.dictConfig")
        return
    sem.call(target, (), kw, how + " via logging.config.dictConfig")


def _walk_logging_config(sem, cfg: PDict, how):
    d = cfg.d
    if d.get("version") != 1:
        raise SimError("Unsupported version in logging configuration")
    if sem.truthy(d.get("incremental", False)):
        return  # incremental configuration only adjusts levels
    for section in ("formatters", "filters", "handlers"):
        sec = d.get(section)
        if not isinstance(sec, PDict):
            continue
        for name, spec in sec.d.items():
            if not isinstance(spec, PDict):
                continue
            if section == "formatters" and "()" not in spec.d:
                _cfg_factory(sem, spec, how, positional=("format", "datefmt", "style"))
            elif section == "filters" and "()" not in spec.d:
                continue
            else:
                _cfg_factory(sem, spec, how)


# ---------------------------------------------------------------- factories
# functions whose result is an object with capability-bearing methods
FACTORY_RESULT = {
    "urllib.request.build_opener": "urllib.request.OpenerDirector",
    "webbrowser.get": "webbrowser.GenericBrowser",
    "multiprocessing.get_context": "multiprocessing.context.DefaultContext",
    "multiprocessing.context.BaseContext.get_context": "multiprocessing.context.DefaultContext",
    "asyncio.new_event_loop": "asyncio.unix_events._UnixSelectorEventLoop",
    "asyncio.get_event_loop": "asyncio.unix_events._UnixSelectorEventLoop",
    "asyncio.get_running_loop": "asyncio.unix_events._UnixSelectorEventLoop",
    "asyncio.events.new_event_loop": "asyncio.unix_events._UnixSelectorEventLoop",
    "tarfile.open": "tarfile.TarFile",
    "gzip.open": "gzip.GzipFile",
    "bz2.open": "bz2.BZ2File",
    "lzma.open": "lzma.LZMAFile",
    "sqlite3.connect": "sqlite3.Connection",
    "shelve.open": "shelve.DbfilenameShelf",
    "ssl.create_default_context": "ssl.SSLContext",
    "ssl.SSLContext.wrap_socket": "ssl.SSLSocket",
    "socket.fromfd": "socket.socket",
    "socket.socket.dup": "socket.socket",
    "logging.getLogger": "logging.Logger",
    "zipfile.ZipFile.open": "zipfile.ZipExtFile",
}


for _f, _r in FACTORY_RESULT.items():
    def _make(fpath, rpath):
        def h(sem, args, kwargs, how):
            ref = sem.ref(fpath) if not isinstance(sem.ref(fpath), Opaque) else None
            if isinstance(ref, Ref) and ref.oid in sem.caps:
                sem.check_cap(ref, args, kwargs, how)
            return Instance(sem.ref(rpath), extra={"constructed": True, "args": tuple(args)})
        return h
    H(_f)(_make(_f, _r))


@H("importlib.util.spec_from_file_location", "importlib.util.spec_from_loader")
def _spec_from_file(sem, args, kwargs, how):
    loader = Instance(sem.ref("importlib.machinery.SourceFileLoader"), extra={"constructed": True})
    spec = Instance(sem.ref("importlib.machinery.ModuleSpec"), extra={"constructed": True})
    spec.d.d["loader"] = loader
    return spec


@H("importlib.util.module_from_spec")
def _module_from_spec(sem, args, kwargs, how):
    spec = args[0] if args else None
    if isinstance(spec, Instance) and spec.d is not None and "loader" in spec.d.d:
        m = Module("loaded module", indexed=False, fresh=True)
        m.overlay["__loader__"] = spec.d.d["loader"]
        return m
    return Opaque("module")


@H("builtins.globals", "builtins.locals")
def _globals2(sem, args, kwargs, how):
    m = Module("__main__", indexed=False, fresh=True)
    m.overlay["__builtins__"] = sem.module("builtins")
    return ModuleDict(m)


@H("builtins.memoryview.tobytes", "builtins.memoryview.toreadonly", "builtins.memoryview.cast",
   "builtins.memoryview.release", "builtins.memoryview.tolist")
def _memoryview_methods(sem, args, kwargs, how):
    o = args[0] if args else None
    if "tolist" in how:
        return PList(list(o)) if isinstance(o, bytes) else Opaque("list")
    if "release" in how:
        return None
    if isinstance(o, PByteArray):
        return bytes(o.b)
    return o


# ---------------------------------------------------------------- cloudpickle / dill helpers
_DILL_TYPES = {"FunctionType": "types.FunctionType", "CodeType": "types.CodeType", "LambdaType": "types.FunctionType",
               "MethodType": "types.MethodType", "ModuleType": "types.ModuleType", "CellType": "types.CellType",
               "BuiltinFunctionType": "types.BuiltinFunctionType", "PartialType": "functools.partial"}


@H("cloudpickle.cloudpickle._builtin_type", "cloudpickle.cloudpickle_fast._builtin_type",
   "srsly.cloudpickle.cloudpickle._builtin_type", "joblib.externals.cloudpickle.cloudpickle._builtin_type")
def _cp_builtin_type(sem, args, kwargs, how):
    name = arg(args, kwargs, 0, "name")
    if name == "ClassType":
        return sem.ref("builtins.type")
    if not isinstance(name, str):
        raise SimError("type name must be str")
    return sem.getattr(sem.module("types"), name)


@H("dill._dill._load_type", "dill._dill._get_type")
def _dill_load_type(sem, args, kwargs, how):
    name = arg(args, kwargs, 0, "name")
    if not isinstance(name, str):
        raise SimError("type name must be str")
    oid = _DILL_TYPES.get(name)
    if oid:
        return sem.ref(oid)
    try:
        return sem.getattr(sem.module("types"), name)
    except SimError:
        return sem.getattr(sem.module("builtins"), name)


@H("dill._dill._get_attr", "dill._dill._getattr")
def _dill_getattr(sem, args, kwargs, how):
    if len(args) < 2:
        raise SimError("_get_attr expects 2 arguments")
    return sem.getattr(args[0], args[1])


@H("dill._dill._import_module", "cloudpickle.cloudpickle.subimport", "cloudpickle.cloudpickle_fast.subimport",
   "joblib.externals.cloudpickle.cloudpickle.subimport")
def _dill_import(sem, args, kwargs, how):
    name = arg(args, kwargs, 0, "import_name")
    if not isinstance(name, str):
        raise SimError("module name must be str")
    return sem.import_module(name)


# ---------------------------------------------------------------- __new__ of C types whose tp_new does the work
_NEW_LIKE_CALL = ("itertools.chain", "itertools.islice", "itertools.accumulate", "itertools.filterfalse",
                  "itertools.dropwhile", "itertools.takewhile", "itertools.groupby", "itertools.zip_longest",
                  "itertools.product", "itertools.permutations", "itertools.combinations",
                  "itertools.combinations_with_replacement", "itertools.compress", "itertools.cycle",
                  "itertools.repeat", "itertools.count", "itertools.pairwise", "itertools.batched",
                  "builtins.staticmethod")


def _new_like_call(path):
    def h(sem, args, kwargs, how):
        if not args:
            raise SimError("__new__ needs a type")
        return sem.call(sem.ref(path), tuple(args[1:]), kwargs, how)
    return h


for _p in _NEW_LIKE_CALL:
    H(_p + ".__new__")(_new_like_call(_p))


# ---------------------------------------------------------------- tuple subclasses usable as REDUCE arguments
_STRUCTSEQ_FIELDS = {"os.terminal_size": 2, "os.stat_result": 10, "os.statvfs_result": 10, "time.struct_time": 9,
                     "os.times_result": 5, "os.uname_result": 5, "os.waitid_result": 5, "pwd.struct_passwd": 7,
                     "grp.struct_group": 4, "resource.struct_rusage": 16, "os.sched_param": 1}


def _structseq(path, n):
    def h(sem, args, kwargs, how):
        seq = arg(args, kwargs, 0, "sequence")
        items = tuple(sem.iterate(seq))
        if len(items) < n:
            raise SimError(f"{path}() takes an at least {n}-sequence")
        return items[:n]
    return h


for _p, _n in _STRUCTSEQ_FIELDS.items():
    H(_p)(_structseq(_p, _n))


@H("collections.namedtuple", "typing.NamedTuple")
def _namedtuple(sem, args, kwargs, how):
    name = arg(args, kwargs, 0, "typename", "T")
    fields = arg(args, kwargs, 1, "field_names", ())
    if isinstance(fields, str):
        names = fields.replace(",", " ").split()
    else:
        names = []
        for f in sem.iterate(fields):
            if isinstance(f, tuple) and f:
                f = f[0]
            names.append(f if isinstance(f, str) else "f")
    ns = PDict({"__new__": Ref("collections._namedtuple_new", "b"), "_fields": tuple(names)})
    return ClassObj(name if isinstance(name, str) else "T", (sem.ref("builtins.tuple"),), ns)


@H("collections._namedtuple_new")
def _namedtuple_new(sem, args, kwargs, how):
    if not args or not isinstance(args[0], ClassObj):
        raise SimError("__new__ needs the namedtuple class")
    fields = args[0].ns.d.get("_fields", ())
    vals = list(args[1:])
    for f in fields[len(vals):]:
        if f not in kwargs:
            raise SimError("missing namedtuple field")
        vals.append(kwargs[f])
    if len(vals) != len(fields):
        raise SimError("namedtuple arity mismatch")
    return tuple(vals)


# ---------------------------------------------------------------- objects found dynamically at run time
@H("builtins.type.__subclasses__", "builtins.object.__subclasses__", "sys._getframe", "inspect.currentframe",
   "inspect.stack", "inspect.trace", "gc.get_objects", "gc.get_referrers", "gc.get_referents", "sys._current_frames",
   "traceback.walk_stack", "inspect.getouterframes", "sys.exc_info", "builtins.object.__reduce_ex__",
   "builtins.object.__reduce__")
def _dynamic(sem, args, kwargs, how):
    return Opaque(f"run-time object from {how.split()[-1]}", tainted=True)


# ---------------------------------------------------------------- more call routes
@H("contextvars.Context.run")
def _context_run(sem, args, kwargs, how):
    if len(args) < 2:
        raise SimError("run() missing callable")
    return sem.call(args[1], args[2:], kwargs, how + " via contextvars.Context.run")


@H("contextvars.copy_context")
def _copy_context(sem, args, kwargs, how):
    return Instance(sem.ref("contextvars.Context"), d=None, extra={"constructed": True})


@H("importlib._bootstrap._call_with_frames_removed", "_frozen_importlib._call_with_frames_removed")
def _call_frames_removed(sem, args, kwargs, how):
    if not args:
        raise SimError("missing callable")
    return sem.call(args[0], args[1:], kwargs, how + " via _call_with_frames_removed")


@H("sys.call_tracing")
def _call_tracing(sem, args, kwargs, how):
    if len(args) != 2 or not isinstance(args[1], tuple):
        raise SimError("call_tracing(func, args) expects a tuple")
    return sem.call(args[0], args[1], {}, how + " via sys.call_tracing")
