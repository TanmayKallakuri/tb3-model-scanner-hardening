"""Semantics of the callables through which an unpickler reaches a
capability: indirection (getattr, attrgetter, methodcaller, partial,
__import__ ...), callers (map, filter, sorted, reduce, threads, executors ...)
and constructors whose ``__new__`` does the work.

Each handler is ``fn(interp, f, args, kwargs) -> value`` and is keyed by the
canonical name of the callable (see ``interp.canonical``).
"""
from __future__ import annotations

import codecs
import functools
import io
import itertools
import operator
import types

from modelgate import kb
from modelgate.interp import AttrDict, MAX_ITEMS, canonical
from modelgate.symbolic import (
    AttrGetter, Bound, CodeObj, Coro, FileObj, Func, Inst, ItemGetter, Iter, LoadError,
    MethodCaller, Namespace, Opaque, Partial, Real, Sym, SymClass, SysModules, Unres,
    Wrapper, describe, is_plain,
)

HANDLERS: dict = {}
NEW_HANDLERS: dict = {}
FILE_METHODS: dict = {}

# attributes that __init__ sets from constructor parameters, for objects the
# scanner builds without running __init__: class -> {attribute: parameter}
INIT_ATTRS = {
    "threading.Thread": {"_target": "target", "_args": "args", "_kwargs": "kwargs"},
    "threading.Timer": {"function": "function", "args": "args", "kwargs": "kwargs",
                        "interval": "interval"},
    "multiprocessing.process.BaseProcess": {"_target": "target", "_args": "args", "_kwargs": "kwargs"},
    "unittest.mock.NonCallableMock": {"side_effect": "side_effect", "_mock_wraps": "wraps",
                                      "return_value": "return_value"},
    "timeit.Timer": {"stmt": "stmt", "setup": "setup"},
}


def h(*names, inheritable=False, new=False):
    def deco(fn):
        fn.inheritable = inheritable
        for n in names:
            HANDLERS[n] = fn
            if new:
                NEW_HANDLERS[n] = fn
        return fn
    return deco


def arg(args, kwargs, idx, name, default=None):
    if len(args) > idx:
        return args[idx]
    return kwargs.get(name, default)


def need(args, lo, hi=None, kwargs=None, name="function"):
    hi = lo if hi is None else hi
    if not (lo <= len(args) <= hi):
        raise LoadError(f"{name} takes {lo}..{hi} positional arguments")


def is_callable_value(v):
    """False only for values that certainly are not callable."""
    if isinstance(v, (int, float, complex, str, bytes, bytearray, list, tuple, dict, set,
                      frozenset, type(None), range)):
        return isinstance(v, type)
    if isinstance(v, (Namespace, SysModules, FileObj, CodeObj, Iter, Coro)):
        return False
    return True


def check_iterable(v):
    if isinstance(v, (int, float, complex, bool, type(None))):
        raise LoadError(f"'{type(v).__name__}' object is not iterable")
    if isinstance(v, Real) and isinstance(v.obj, (types.FunctionType, types.BuiltinFunctionType,
                                                  types.ModuleType)):
        raise LoadError("object is not iterable")
    if isinstance(v, Real) and isinstance(v.obj, type):
        import enum

        if not issubclass(v.obj, enum.Enum):
            raise LoadError("'type' object is not iterable")
    if isinstance(v, (AttrGetter, ItemGetter, MethodCaller, Partial, Wrapper, Func, CodeObj, Coro)):
        raise LoadError("object is not iterable")


# ----------------------------------------------------------------- attributes
@h("builtins.getattr")
def _getattr(I, f, args, kwargs):
    if kwargs:
        raise LoadError("getattr() takes no keyword arguments")
    need(args, 2, 3, name="getattr")
    obj, name = args[0], args[1]
    if not isinstance(name, str):
        if isinstance(name, Sym):
            return I.unknown_attr(obj)
        raise LoadError("attribute name must be string")
    try:
        return I.getattr(obj, name)
    except LoadError:
        if len(args) == 3:
            return args[2]
        raise


@h("builtins.object.__getattribute__", "builtins.type.__getattribute__",
   "builtins.module.__getattribute__", "inspect.getattr_static")
def _getattribute(I, f, args, kwargs):
    if len(args) < 2:
        raise LoadError("__getattribute__ needs 2 arguments")
    try:
        return I.getattr(args[0], args[1])
    except LoadError:
        if len(args) > 2:
            return args[2]
        raise


@h("builtins.setattr", "builtins.object.__setattr__", "builtins.type.__setattr__",
   "builtins.module.__setattr__")
def _setattr(I, f, args, kwargs):
    need(args, 3, name="setattr")
    I.setattr(args[0], args[1], args[2])
    return None


@h("builtins.delattr", "builtins.object.__delattr__")
def _delattr(I, f, args, kwargs):
    need(args, 2, name="delattr")
    return None


@h("builtins.hasattr")
def _hasattr(I, f, args, kwargs):
    need(args, 2, name="hasattr")
    try:
        I.getattr(args[0], args[1])
        return True
    except LoadError:
        return False


@h("builtins.vars")
def _vars(I, f, args, kwargs):
    if not args:
        return Opaque("vars() of the calling frame")
    obj = args[0]
    if isinstance(obj, Real) and isinstance(obj.obj, types.ModuleType):
        return Namespace(obj)
    if isinstance(obj, Unres) and obj.is_module:
        return Namespace(obj)
    if isinstance(obj, Inst):
        return obj.attrs
    if isinstance(obj, AttrDict):
        return obj.attrs
    if isinstance(obj, Partial):
        return obj.dict
    if isinstance(obj, Real) and isinstance(obj.obj, type):
        return Opaque(f"mappingproxy of {obj.name}")
    if isinstance(obj, (int, str, float, bytes, list, tuple, dict, type(None))):
        raise LoadError("vars() argument must have __dict__ attribute")
    return Opaque("vars()")


@h("builtins.globals", "builtins.locals")
def _globals(I, f, args, kwargs):
    return Opaque("namespace of the calling frame")


@h("builtins.dir")
def _dir(I, f, args, kwargs):
    return Opaque("dir()")


# ------------------------------------------------------------------- imports
@h("builtins.__import__", "importlib.__import__", "importlib._bootstrap.__import__")
def _import(I, f, args, kwargs):
    name = arg(args, kwargs, 0, "name")
    fromlist = arg(args, kwargs, 3, "fromlist", ())
    level = arg(args, kwargs, 4, "level", 0)
    if not isinstance(name, str):
        if isinstance(name, Opaque):
            return Opaque("module imported by an unknown name")
        raise LoadError("__import__() argument 1 must be str")
    if level not in (0, None) and level != 0:
        raise LoadError("relative import without a package")
    mod = I.import_module(name)
    if not fromlist:
        top = name.split(".")[0]
        return I.import_module(top)
    return mod


@h("importlib.import_module", "importlib._bootstrap._gcd_import")
def _import_module(I, f, args, kwargs):
    name = arg(args, kwargs, 0, "name")
    package = arg(args, kwargs, 1, "package")
    if not isinstance(name, str):
        if isinstance(name, Opaque):
            return Opaque("module imported by an unknown name")
        raise LoadError("module name must be a string")
    if name.startswith("."):
        if not isinstance(package, str):
            raise LoadError("relative import requires package")
        level = len(name) - len(name.lstrip("."))
        base = package.rsplit(".", level - 1)[0] if level > 1 else package
        name = base + "." + name.lstrip(".") if name.lstrip(".") else base
    return I.import_module(name)


def _resolve_dotted(I, name):
    """Import the longest importable prefix of ``name`` and getattr the rest
    (pkgutil.resolve_name / pydoc.locate / logging.config resolution)."""
    if ":" in name:
        modname, _, attrs = name.partition(":")
        obj = I.import_module(modname)
        if attrs:
            for part in attrs.split("."):
                obj = I.getattr(obj, part)
        return obj
    parts = name.split(".")
    obj = None
    used = 0
    for i in range(len(parts), 0, -1):
        modname = ".".join(parts[:i])
        try:
            m = I.import_module(modname)
        except LoadError:
            continue
        if isinstance(m, Unres) and i > 1:
            # third-party: assume the first component is the package
            continue
        obj, used = m, i
        break
    if obj is None:
        raise LoadError(f"cannot resolve {name!r}")
    for part in parts[used:]:
        obj = I.getattr(obj, part)
    return obj


@h("pkgutil.resolve_name", "pydoc.locate", "logging.config._resolve", "unittest.mock._importer",
   "unittest.mock._dot_lookup")
def _resolve_name(I, f, args, kwargs):
    name = args[0] if args else kwargs.get("name", kwargs.get("path", kwargs.get("target")))
    if canonical(f.obj) == "unittest.mock._dot_lookup" and len(args) >= 3:
        try:
            return I.getattr(args[0], args[1])
        except LoadError:
            return I.import_module(args[2])
    if not isinstance(name, str):
        if isinstance(name, Sym):
            return Opaque("object resolved from an unknown name")
        raise LoadError("name must be str")
    try:
        return _resolve_dotted(I, name)
    except LoadError:
        if canonical(f.obj) == "pydoc.locate":
            return None
        raise


@h("logging.config.BaseConfigurator.resolve")
def _cfg_resolve(I, f, args, kwargs):
    if len(args) < 2 or not isinstance(args[1], str):
        return Opaque("resolved object")
    return _resolve_dotted(I, args[1])


# ------------------------------------------------------ operator / functools
@h("operator.attrgetter", new=True)
def _attrgetter(I, f, args, kwargs):
    if not args:
        raise LoadError("attrgetter expected 1 argument")
    for a in args:
        if not isinstance(a, str):
            raise LoadError("attribute name must be a string")
    return AttrGetter(args)


@h("operator.itemgetter", new=True)
def _itemgetter(I, f, args, kwargs):
    if not args:
        raise LoadError("itemgetter expected 1 argument")
    return ItemGetter(args)


@h("operator.methodcaller", new=True)
def _methodcaller(I, f, args, kwargs):
    if not args or not isinstance(args[0], str):
        raise LoadError("method name must be a string")
    return MethodCaller(args[0], args[1:], kwargs)


@h("operator.getitem", "operator.__getitem__", "builtins.object.__getitem__")
def _getitem(I, f, args, kwargs):
    need(args, 2, name="getitem")
    return I.getitem(args[0], args[1])


@h("operator.setitem", "operator.__setitem__")
def _setitem(I, f, args, kwargs):
    need(args, 3, name="setitem")
    I.setitem(args[0], args[1], args[2])
    return None


@h("operator.delitem", "operator.__delitem__")
def _delitem(I, f, args, kwargs):
    return None


@h("operator.call", "operator.__call__")
def _opcall(I, f, args, kwargs):
    if not args:
        raise LoadError("call() missing argument")
    return I.call(args[0], args[1:], kwargs)


@h("functools.partial", new=True)
def _partial(I, f, args, kwargs):
    if not args:
        raise LoadError("type 'partial' takes at least one argument")
    func = args[0]
    if not is_callable_value(func):
        raise LoadError("the first argument must be callable")
    if isinstance(func, Partial):
        return Partial(func.func, func.args + tuple(args[1:]), {**func.kwargs, **kwargs})
    return Partial(func, args[1:], kwargs)


@h("functools.partial.__setstate__")
def _partial_setstate(I, f, args, kwargs):
    need(args, 2, name="partial.__setstate__")
    p, state = args
    if not isinstance(p, Partial):
        raise LoadError("descriptor '__setstate__' requires a 'functools.partial' object")
    if not isinstance(state, tuple) or len(state) != 4:
        raise LoadError("invalid partial state")
    func, fargs, fkw, fdict = state
    if not is_callable_value(func) or not isinstance(fargs, tuple) or \
            not (fkw is None or isinstance(fkw, dict)) or not (fdict is None or isinstance(fdict, dict)):
        raise LoadError("invalid partial state")
    p.func = func
    p.args = fargs
    p.kwargs = dict(fkw or {})
    p.dict = fdict if fdict is not None else {}
    return None


@h("functools.reduce", "_functools.reduce")
def _reduce(I, f, args, kwargs):
    need(args, 2, 3, name="reduce")
    func, it = args[0], args[1]
    items = I.consume(it)
    if len(args) == 3:
        acc = args[2]
    else:
        if not items:
            raise LoadError("reduce() of empty iterable with no initial value")
        acc, items = items[0], items[1:]
    for item in items:
        acc = I.call(func, (acc, item))
    return acc


@h("functools.lru_cache")
def _lru_cache(I, f, args, kwargs):
    maxsize = arg(args, kwargs, 0, "maxsize", 128)
    if isinstance(maxsize, Sym) and is_callable_value(maxsize):
        return Wrapper(maxsize, "functools.lru_cache wrapper")
    if maxsize is not None and not isinstance(maxsize, int):
        raise LoadError("Expected first argument to be an integer, a callable, or None")
    return Wrapper(None, "decorator:wrap")


@h("functools.cache", "functools.singledispatch", "builtins.staticmethod",
   "contextlib.contextmanager", "contextlib.asynccontextmanager", "numpy.vectorize",
   "atexit.register")
def _wrap(I, f, args, kwargs):
    name = canonical(f.obj) if isinstance(f, Real) else getattr(f, "name", "")
    if not args:
        raise LoadError("missing function")
    target = args[0]
    if name == "atexit.register":
        # runs at interpreter exit, still a consequence of loading
        I.call(target, args[1:], kwargs)
        return target
    if name == "builtins.staticmethod":
        return Wrapper(target, "staticmethod")
    return Wrapper(target, f"{name} wrapper")


@h("numpy.frompyfunc")
def _frompyfunc(I, f, args, kwargs):
    if not args:
        raise LoadError("missing function")
    return Wrapper(args[0], "numpy ufunc")


@h("functools.wraps")
def _wraps(I, f, args, kwargs):
    return Wrapper(None, "decorator:identity")


@h("functools.update_wrapper")
def _update_wrapper(I, f, args, kwargs):
    if not args:
        raise LoadError("missing wrapper")
    return args[0]


@h("copy.copy", "copy.deepcopy")
def _copy(I, f, args, kwargs):
    if not args:
        raise LoadError("copy() missing argument")
    x = args[0]
    if isinstance(x, Partial):
        return Partial(x.func, x.args, x.kwargs, dict(x.dict))
    if isinstance(x, list):
        return list(x)
    if isinstance(x, AttrDict):
        d = AttrDict(x, clsname=x.clsname, factory=x.factory)
        d.attrs = dict(x.attrs)
        return d
    if isinstance(x, dict):
        return dict(x)
    if isinstance(x, set):
        return set(x)
    return x


@h("types.MethodType", "builtins.method")
def _methodtype(I, f, args, kwargs):
    need(args, 2, name="method")
    if not is_callable_value(args[0]):
        raise LoadError("first argument must be callable")
    return Bound(args[1], args[0], "method")


@h("builtins.callable")
def _callable(I, f, args, kwargs):
    need(args, 1, name="callable")
    return is_callable_value(args[0])


@h("builtins.type")
def _type(I, f, args, kwargs):
    if len(args) == 1:
        x = args[0]
        if isinstance(x, Inst):
            return x.cls
        if isinstance(x, Real):
            t = type(x.obj)
            return Real(t, canonical(t) or "type")
        if isinstance(x, Partial):
            return Real(functools.partial, "functools.partial")
        if isinstance(x, AttrDict):
            return Opaque(x.clsname)
        if not isinstance(x, Sym):
            return Real(type(x), canonical(type(x)) or "type")
        return Opaque("type of an unknown object")
    if len(args) == 3:
        name, bases, ns = args
        if not isinstance(name, str) or not isinstance(bases, tuple) or not isinstance(ns, dict):
            raise LoadError("type() arguments")
        return SymClass(name, bases, ns)
    raise LoadError("type() takes 1 or 3 arguments")


# --------------------------------------------------------------- copyreg
@h("copyreg._reconstructor")
def _reconstructor(I, f, args, kwargs):
    need(args, 3, name="_reconstructor")
    cls, base, state = args
    if isinstance(base, Real) and base.obj is object:
        return I.new_object(cls, (), {})
    obj = I.new_object(cls, (state,), {}) if isinstance(cls, (Real, Unres, SymClass)) else Opaque("object")
    init_call(I, base, obj, (state,))
    return obj


def init_call(I, base, obj, args):
    """``base.__init__(obj, *args)`` where base is a class."""
    if isinstance(base, Real) and isinstance(base.obj, type):
        for b in base.obj.__mro__:
            bname = canonical(b)
            cap = kb.CAPS.get(bname)
            if cap is not None:
                I.capability(None, bname, cap, args, {})
                return
            hh = HANDLERS.get(bname)
            if hh is not None and getattr(hh, "inheritable", False):
                hh(I, base, args, {})
                return
        bov = I._over(base.obj)
        if bov and "__init__" in bov:
            I.call(bov["__init__"], (obj,) + tuple(args))
    elif isinstance(base, SymClass) and "__init__" in base.ns:
        I.call(base.ns["__init__"], (obj,) + tuple(args))


@h("copyreg.__newobj__")
def _newobj(I, f, args, kwargs):
    if not args:
        raise LoadError("__newobj__ missing class")
    return I.new_object(args[0], args[1:], {})


@h("copyreg.__newobj_ex__")
def _newobj_ex(I, f, args, kwargs):
    need(args, 3, name="__newobj_ex__")
    cls, a, k = args
    if not isinstance(a, tuple) or not isinstance(k, dict):
        raise LoadError("__newobj_ex__ arguments")
    return I.new_object(cls, a, k)


@h("copyreg.add_extension")
def _add_extension(I, f, args, kwargs):
    need(args, 3, name="add_extension")
    module, name, code = args
    if not isinstance(code, int) or not (1 <= code <= 0x7FFFFFFF):
        raise LoadError("code out of range")
    I.ext_registry[code] = (module, name)
    return None


@h("builtins.object.__new__")
def _object_new(I, f, args, kwargs):
    if not args:
        raise LoadError("object.__new__(): not enough arguments")
    return I.new_object(args[0], args[1:], kwargs)


@h("builtins.object.__reduce_ex__", "builtins.object.__reduce__", "copyreg._reduce_ex")
def _reduce_ex(I, f, args, kwargs):
    return Opaque("reduce value")


# ------------------------------------------------- code objects and sources
def _source_ok(v):
    if isinstance(v, Real):
        return isinstance(v.obj, types.CodeType)
    if isinstance(v, (Partial, AttrGetter, ItemGetter, MethodCaller, Iter, Wrapper, Func, Namespace,
                      SysModules, FileObj, Bound)):
        return False
    if isinstance(v, (str, bytes, bytearray, CodeObj, Sym)):
        return True
    return False


def _mapping_ok(v):
    return v is None or isinstance(v, (dict, Namespace, Opaque, Inst, Unres))


@h("builtins.eval", "builtins.exec")
def _eval(I, f, args, kwargs):
    src = arg(args, kwargs, 0, "source")
    if src is None and not args and "source" not in kwargs:
        raise LoadError("eval expected at least 1 argument")
    if len(args) > 3:
        raise LoadError("too many arguments")
    if len(args) > 1 and not _mapping_ok(args[1]):
        raise LoadError("globals must be a dict")
    if not _source_ok(src):
        raise LoadError("eval() arg 1 must be a string, bytes or code object")
    I.finding(kb.CODE, f"call of {canonical(f.obj)}({describe(src, 60)})")
    return Opaque("result of eval")


@h("builtins.compile")
def _compile(I, f, args, kwargs):
    if len(args) + len(kwargs) < 3:
        raise LoadError("compile() missing required argument")
    src = arg(args, kwargs, 0, "source")
    if not _source_ok(src) and not isinstance(src, (Inst, Opaque)):
        raise LoadError("compile() arg 1 must be a string, bytes or AST object")
    I.finding(kb.CODE, f"call of builtins.compile({describe(src, 60)})")
    return CodeObj("code compiled from source")


@h("builtins.function", "types.FunctionType", "types.LambdaType")
def _function(I, f, args, kwargs):
    code = arg(args, kwargs, 0, "code")
    globs = arg(args, kwargs, 1, "globals")
    if code is None or (not isinstance(code, Sym)):
        raise LoadError("function() argument 'code' must be code")
    if globs is not None and not isinstance(globs, (dict, Namespace, Sym)):
        raise LoadError("function() argument 'globals' must be dict")
    I.finding(kb.CODE, f"materialises serialized bytecode into a function ({describe(code, 60)})")
    return Func("function built from serialized bytecode")


@h("builtins.code", "types.CodeType")
def _codetype(I, f, args, kwargs):
    return CodeObj("code object built from raw fields")


@h("marshal.loads")
def _marshal_loads(I, f, args, kwargs):
    data = arg(args, kwargs, 0, "bytes")
    b = I.to_bytes(data)
    if b is None:
        if isinstance(data, Sym):
            return CodeObj("unmarshalled object (unknown data)")
        raise LoadError("marshal.loads() argument must be bytes-like")
    if b and (b[0] & 0x7F) == ord("c"):
        return CodeObj("code object unmarshalled from embedded bytes")
    return Opaque("unmarshalled data")


@h("marshal.load")
def _marshal_load(I, f, args, kwargs):
    return CodeObj("unmarshalled object from a file")


# ------------------------------------------------------------- iterators
def _zip_gen(I, iters):
    its = [I.iterate(x) for x in iters]
    if not its:
        return
    while True:
        items = []
        for it in its:
            try:
                items.append(next(it))
            except StopIteration:
                return
        yield tuple(items)


@h("builtins.map", new=True)
def _map(I, f, args, kwargs):
    if kwargs:
        raise LoadError("map() takes no keyword arguments")
    if len(args) < 2:
        raise LoadError("map() must have at least two arguments")
    func, its = args[0], args[1:]
    for x in its:
        check_iterable(x)

    def gen():
        for items in _zip_gen(I, its):
            yield I.call(func, items)
    return Iter(gen(), f"map({describe(func, 40)})")


@h("builtins.filter", new=True)
def _filter(I, f, args, kwargs):
    need(args, 2, name="filter")
    func, it = args
    check_iterable(it)

    def gen():
        for item in I.iterate(it):
            if func is None:
                yield item
            else:
                r = I.call(func, (item,))
                if not (r is False or r == 0 or r is None) or isinstance(r, Sym):
                    yield item
    return Iter(gen(), f"filter({describe(func, 40)})")


@h("itertools.filterfalse", new=True)
def _filterfalse(I, f, args, kwargs):
    need(args, 2, name="filterfalse")
    func, it = args
    check_iterable(it)

    def gen():
        for item in I.iterate(it):
            if func is not None:
                I.call(func, (item,))
            yield item
    return Iter(gen(), "filterfalse")


@h("itertools.starmap", new=True)
def _starmap(I, f, args, kwargs):
    need(args, 2, name="starmap")
    func, it = args
    check_iterable(it)

    def gen():
        for item in I.iterate(it):
            yield I.call(func, tuple(I.consume(item)))
    return Iter(gen(), f"starmap({describe(func, 40)})")


@h("itertools.takewhile", "itertools.dropwhile", new=True)
def _takewhile(I, f, args, kwargs):
    need(args, 2, name="takewhile")
    func, it = args
    check_iterable(it)

    def gen():
        for item in I.iterate(it):
            I.call(func, (item,))
            yield item
    return Iter(gen(), "takewhile")


@h("itertools.accumulate", new=True)
def _accumulate(I, f, args, kwargs):
    it = arg(args, kwargs, 0, "iterable")
    func = arg(args, kwargs, 1, "func")
    check_iterable(it)

    def gen():
        acc = None
        first = True
        for item in I.iterate(it):
            if first:
                acc, first = item, False
            elif func is not None:
                acc = I.call(func, (acc, item))
            else:
                acc = Opaque("sum")
            yield acc
    return Iter(gen(), "accumulate")


@h("itertools.groupby", new=True)
def _groupby(I, f, args, kwargs):
    it = arg(args, kwargs, 0, "iterable")
    key = arg(args, kwargs, 1, "key")
    check_iterable(it)

    def gen():
        for item in I.iterate(it):
            k = I.call(key, (item,)) if key is not None else item
            yield (k, Iter(iter([item]), "group"))
    return Iter(gen(), "groupby")


@h("builtins.zip", new=True)
def _zip(I, f, args, kwargs):
    for x in args:
        check_iterable(x)
    return Iter(_zip_gen(I, args), "zip")


@h("itertools.zip_longest", new=True)
def _zip_longest(I, f, args, kwargs):
    for x in args:
        check_iterable(x)

    def gen():
        lists = [I.consume(x) for x in args]
        n = max((len(x) for x in lists), default=0)
        for i in range(n):
            yield tuple(x[i] if i < len(x) else kwargs.get("fillvalue") for x in lists)
    return Iter(gen(), "zip_longest")


@h("builtins.enumerate", new=True)
def _enumerate(I, f, args, kwargs):
    it = arg(args, kwargs, 0, "iterable")
    start = arg(args, kwargs, 1, "start", 0)
    check_iterable(it)

    def gen():
        i = start if isinstance(start, int) else 0
        for item in I.iterate(it):
            yield (i, item)
            i += 1
    return Iter(gen(), "enumerate")


@h("builtins.reversed", new=True)
def _reversed(I, f, args, kwargs):
    need(args, 1, name="reversed")
    x = args[0]
    if isinstance(x, (list, tuple, str, bytes, bytearray, range)):
        return Iter(iter(list(reversed(x))), "reversed")
    if isinstance(x, (Iter, int, float, type(None), set, frozenset)):
        raise LoadError("argument to reversed() must be a sequence")
    if isinstance(x, dict):
        return Iter(iter(list(reversed(list(x.keys())))), "reversed")
    return Iter(I.iterate(x), "reversed")


@h("builtins.iter")
def _iter(I, f, args, kwargs):
    need(args, 1, 2, name="iter")
    if len(args) == 2:
        func, sentinel = args

        def gen():
            for _ in range(MAX_ITEMS):
                r = I.call(func, ())
                if not isinstance(r, Sym) and r == sentinel:
                    return
                yield r
        return Iter(gen(), "iter(callable, sentinel)")
    x = args[0]
    if isinstance(x, Iter):
        return x
    check_iterable(x)
    return Iter(I.iterate(x), "iterator")


@h("builtins.next")
def _next(I, f, args, kwargs):
    need(args, 1, 2, name="next")
    it = args[0]
    if not isinstance(it, Iter):
        if isinstance(it, (Opaque, Inst, Unres)):
            return Opaque("next item")
        raise LoadError("object is not an iterator")
    try:
        return next(it.gen)
    except StopIteration:
        if len(args) == 2:
            return args[1]
        raise LoadError("StopIteration") from None


@h("itertools.chain", new=True)
def _chain(I, f, args, kwargs):
    for x in args:
        check_iterable(x)

    def gen():
        for x in args:
            yield from I.iterate(x)
    return Iter(gen(), "chain")


@h("itertools.chain.from_iterable")
def _chain_from(I, f, args, kwargs):
    need(args, 1, name="from_iterable")
    outer = args[0]
    check_iterable(outer)

    def gen():
        for x in I.iterate(outer):
            yield from I.iterate(x)
    return Iter(gen(), "chain")


@h("itertools.islice", new=True)
def _islice(I, f, args, kwargs):
    need(args, 2, 4, name="islice")
    it = args[0]
    check_iterable(it)
    try:
        sl = slice(*args[1:])
        start, stop, step = sl.start or 0, sl.stop, sl.step or 1
    except Exception:  # noqa: BLE001
        raise LoadError("islice arguments") from None

    def gen():
        for i, item in enumerate(I.iterate(it)):
            if stop is not None and isinstance(stop, int) and i >= stop:
                return
            if isinstance(start, int) and i >= start and (i - start) % max(1, step if isinstance(step, int) else 1) == 0:
                yield item
    return Iter(gen(), "islice")


@h("itertools.cycle", new=True)
def _cycle(I, f, args, kwargs):
    need(args, 1, name="cycle")
    check_iterable(args[0])

    def gen():
        saved = []
        for item in I.iterate(args[0]):
            saved.append(item)
            yield item
        if not saved:
            return
        for _ in range(MAX_ITEMS):
            yield from saved
    return Iter(gen(), "cycle")


@h("itertools.repeat", new=True)
def _repeat(I, f, args, kwargs):
    obj = arg(args, kwargs, 0, "object")
    times = arg(args, kwargs, 1, "times")
    n = times if isinstance(times, int) else MAX_ITEMS
    return Iter(iter([obj] * max(0, min(n, MAX_ITEMS))), "repeat")


@h("itertools.count", new=True)
def _count(I, f, args, kwargs):
    start = arg(args, kwargs, 0, "start", 0)
    return Iter((start + i if isinstance(start, int) else Opaque("count") for i in range(MAX_ITEMS)), "count")


@h("itertools.compress", "itertools.pairwise", "itertools.product", "itertools.permutations",
   "itertools.combinations", "itertools.combinations_with_replacement", "itertools.batched",
   "itertools.tee", new=True)
def _itertools_other(I, f, args, kwargs):
    for x in args:
        if not isinstance(x, int):
            check_iterable(x)

    def gen():
        for x in args:
            if not isinstance(x, int):
                for item in I.iterate(x):
                    yield Opaque("combination")
    return Iter(gen(), "itertools")


# ------------------------------------------------------------- consumers
def _items_of(I, x):
    check_iterable(x)
    return I.consume(x)


@h("builtins.list")
def _list(I, f, args, kwargs):
    if not args:
        return []
    need(args, 1, name="list")
    return list(_items_of(I, args[0]))


NEW_HANDLERS["builtins.list"] = lambda I, f, args, kwargs: []


@h("builtins.tuple", new=True)
def _tuple(I, f, args, kwargs):
    if not args:
        return ()
    need(args, 1, name="tuple")
    return tuple(_items_of(I, args[0]))


@h("builtins.set")
def _set(I, f, args, kwargs):
    if kwargs or len(args) > 1:
        raise LoadError("set expected at most 1 argument")
    if not args:
        return set()
    items = _items_of(I, args[0])
    try:
        return set(items)
    except TypeError:
        raise LoadError("unhashable type") from None


NEW_HANDLERS["builtins.set"] = lambda I, f, args, kwargs: set()


@h("builtins.frozenset", new=True)
def _frozenset(I, f, args, kwargs):
    if not args:
        return frozenset()
    items = _items_of(I, args[0])
    try:
        return frozenset(items)
    except TypeError:
        raise LoadError("unhashable type") from None


def _dict_from(I, src, target):
    if isinstance(src, dict):
        for k, v in list(src.items()):
            I.setitem(target, k, v)
        return
    if isinstance(src, (Namespace, SysModules, Opaque, Inst, Unres)):
        return
    for kv in _items_of(I, src):
        if isinstance(kv, (tuple, list)):
            if len(kv) != 2:
                raise LoadError("dictionary update sequence element has wrong length")
            I.setitem(target, kv[0], kv[1])
        elif isinstance(kv, (str, bytes)) and len(kv) == 2:
            I.setitem(target, kv[0:1], kv[1:2])
        elif isinstance(kv, Sym):
            continue
        else:
            raise LoadError("cannot convert dictionary update sequence element")


@h("builtins.dict")
def _dict(I, f, args, kwargs):
    d = {}
    if args:
        need(args, 1, name="dict")
        _dict_from(I, args[0], d)
    d.update(kwargs)
    return d


NEW_HANDLERS["builtins.dict"] = lambda I, f, args, kwargs: {}


@h("collections.OrderedDict", "collections.Counter", "collections.UserDict")
def _odict(I, f, args, kwargs):
    d = AttrDict(clsname=canonical(f.obj) if isinstance(f, Real) else "collections.OrderedDict")
    if args:
        _dict_from(I, args[0], d)
    for k, v in kwargs.items():
        d[k] = v
    return d


NEW_HANDLERS["collections.OrderedDict"] = lambda I, f, args, kwargs: AttrDict(clsname="collections.OrderedDict")
NEW_HANDLERS["collections.Counter"] = lambda I, f, args, kwargs: AttrDict(clsname="collections.Counter")


@h("collections.defaultdict")
def _defaultdict(I, f, args, kwargs):
    factory = args[0] if args else None
    if factory is not None and not is_callable_value(factory):
        raise LoadError("first argument must be callable or None")
    d = AttrDict(clsname="collections.defaultdict", factory=factory)
    if len(args) > 1:
        _dict_from(I, args[1], d)
    return d


NEW_HANDLERS["collections.defaultdict"] = lambda I, f, args, kwargs: AttrDict(clsname="collections.defaultdict")


@h("collections.deque")
def _deque(I, f, args, kwargs):
    it = arg(args, kwargs, 0, "iterable", ())
    return list(_items_of(I, it)) if it != () else []


def _key_calls(I, items, key):
    if key is not None:
        for item in items:
            I.call(key, (item,))


@h("builtins.sorted")
def _sorted(I, f, args, kwargs):
    need(args, 1, name="sorted")
    items = list(_items_of(I, args[0]))
    _key_calls(I, items, kwargs.get("key"))
    if all(is_plain(x) for x in items):
        try:
            return sorted(items, reverse=bool(kwargs.get("reverse"))) if kwargs.get("key") is None else items
        except TypeError:
            raise LoadError("'<' not supported between instances") from None
    return items


@h("builtins.min", "builtins.max")
def _minmax(I, f, args, kwargs):
    if not args:
        raise LoadError("min/max expected at least 1 argument")
    items = list(_items_of(I, args[0])) if len(args) == 1 else list(args)
    _key_calls(I, items, kwargs.get("key"))
    if not items:
        if "default" in kwargs:
            return kwargs["default"]
        raise LoadError("min/max() arg is an empty sequence")
    if all(is_plain(x) for x in items) and kwargs.get("key") is None:
        try:
            return (min if canonical(f.obj) == "builtins.min" else max)(items)
        except TypeError:
            raise LoadError("not comparable") from None
    return items[0]


@h("builtins.sum")
def _sum(I, f, args, kwargs):
    need(args, 1, 2, name="sum")
    items = _items_of(I, args[0])
    start = arg(args, kwargs, 1, "start", 0)
    if all(is_plain(x) for x in items) and is_plain(start):
        if sum(len(x) for x in items if isinstance(x, (list, tuple))) > MAX_ITEMS * 10:
            return Opaque("sum")
        try:
            return sum(items, start)
        except Exception:  # noqa: BLE001 - e.g. sum() of bytes raises TypeError
            raise LoadError("sum() failed") from None
    return Opaque("sum")


@h("builtins.any", "builtins.all")
def _anyall(I, f, args, kwargs):
    need(args, 1, name="any")
    items = _items_of(I, args[0])
    if all(is_plain(x) for x in items):
        return (any if canonical(f.obj) == "builtins.any" else all)(items)
    return Opaque("bool")


@h("builtins.str.join", "builtins.bytes.join", "builtins.bytearray.join")
def _join(I, f, args, kwargs):
    need(args, 2, name="join")
    sep, it = args
    items = _items_of(I, it)
    if all(is_plain(x) for x in items) and is_plain(sep):
        try:
            return f.obj(sep, items)
        except Exception:  # noqa: BLE001
            raise LoadError("join failed") from None
    return Opaque("joined string")


@h("heapq.nsmallest", "heapq.nlargest")
def _nsmallest(I, f, args, kwargs):
    need(args, 2, 3, name="nsmallest")
    items = _items_of(I, args[1])
    _key_calls(I, items, arg(args, kwargs, 2, "key"))
    return items


@h("heapq.merge")
def _merge(I, f, args, kwargs):
    key = kwargs.get("key")

    def gen():
        for x in args:
            for item in I.iterate(x):
                if key is not None:
                    I.call(key, (item,))
                yield item
    return Iter(gen(), "heapq.merge")


@h("operator.countOf", "operator.indexOf", "operator.contains", "builtins.len")
def _count_of(I, f, args, kwargs):
    if not args:
        raise LoadError("missing argument")
    x = args[0]
    if isinstance(x, Iter):
        if canonical(f.obj) == "builtins.len":
            raise LoadError("object of type 'map' has no len()")
        I.consume(x)
        return Opaque("count")
    if is_plain(x) and all(is_plain(a) for a in args):
        try:
            return f.obj(*args)
        except Exception:  # noqa: BLE001
            raise LoadError("call failed") from None
    return Opaque("count")


def _consuming_any(I, f, args, kwargs):
    for a in args:
        if isinstance(a, Iter):
            I.consume(a)
    return Opaque(f"result of {canonical(f.obj) if isinstance(f, Real) else f.name}")


for _n in ("statistics.mean", "statistics.median", "statistics.mode", "statistics.fmean",
           "statistics.stdev", "statistics.variance", "math.fsum", "math.prod", "numpy.fromiter",
           "array.array", "statistics.multimode", "statistics.geometric_mean",
           "statistics.harmonic_mean", "statistics.median_low", "statistics.median_high",
           "statistics.pstdev", "statistics.pvariance"):
    HANDLERS[_n] = _consuming_any


@h("builtins.bytes", "builtins.bytearray", "builtins.str", new=True)
def _bytes(I, f, args, kwargs):
    name = canonical(f.obj)
    if args and isinstance(args[0], Iter):
        if name == "builtins.str":
            return Opaque("str of an iterator")  # str() does not iterate
        items = I.consume(args[0])
        if all(isinstance(x, int) for x in items):
            try:
                return f.obj(items)
            except Exception:  # noqa: BLE001
                raise LoadError("bytes() failed") from None
        if any(is_plain(x) and not isinstance(x, int) for x in items):
            raise LoadError("'str' object cannot be interpreted as an integer")
        return Opaque("bytes")
    return I.eval_pure(f.obj, name, args, kwargs)


# ----------------------------------------------------------- invokers
def _thread_target(I, inst):
    if not isinstance(inst, Inst):
        return None, (), {}
    target = inst.attrs.get("_target")
    targs = inst.attrs.get("_args")
    tkw = inst.attrs.get("_kwargs")
    bound = None
    if isinstance(inst.cls, Real) and inst.inited:
        import threading

        try:
            bound = I.signature(threading.Thread).bind(*inst.args, **inst.kwargs).arguments
        except Exception:  # noqa: BLE001
            bound = None
    if bound is not None:
        target = bound.get("target", target) if target is None else target
        targs = bound.get("args", targs) if targs is None else targs
        tkw = bound.get("kwargs", tkw) if tkw is None else tkw
    return target, targs or (), tkw or {}


@h("threading.Thread.start", "multiprocessing.process.BaseProcess.run")
def _thread_start(I, f, args, kwargs):
    need(args, 1, name="start")
    inst = args[0]
    run = I.getattr(inst, "run") if isinstance(inst, (Inst, SymClass)) else None
    if run is None or isinstance(run, Opaque):
        return None
    I.call(run, ())
    return None


@h("threading.Thread.run")
def _thread_run(I, f, args, kwargs):
    need(args, 1, name="run")
    target, targs, tkw = _thread_target(I, args[0])
    if target is not None:
        if not isinstance(targs, tuple):
            targs = tuple(I.consume(targs))
        I.call(target, targs, tkw if isinstance(tkw, dict) else {})
    return None


@h("threading.Timer.run")
def _timer_run(I, f, args, kwargs):
    need(args, 1, name="run")
    inst = args[0]
    if not isinstance(inst, Inst):
        return None
    func = inst.attrs.get("function")
    fargs = inst.attrs.get("args")
    fkw = inst.attrs.get("kwargs")
    if inst.inited:
        func = func if func is not None else arg(inst.args, inst.kwargs, 1, "function")
        fargs = fargs if fargs is not None else arg(inst.args, inst.kwargs, 2, "args")
        fkw = fkw if fkw is not None else arg(inst.args, inst.kwargs, 3, "kwargs")
    if func is not None:
        I.call(func, tuple(I.consume(fargs)) if fargs is not None else (), fkw if isinstance(fkw, dict) else {})
    return None


@h("_thread.start_new_thread", "_thread.start_new", "_thread.start_joinable_thread")
def _start_new_thread(I, f, args, kwargs):
    need(args, 1, 3, name="start_new_thread")
    func = args[0]
    fargs = args[1] if len(args) > 1 else ()
    fkw = args[2] if len(args) > 2 else {}
    if canonical(f.obj) != "_thread.start_joinable_thread" and not isinstance(fargs, tuple) and not isinstance(fargs, Sym):
        raise LoadError("2nd arg must be a tuple")
    I.call(func, tuple(I.consume(fargs)) if not isinstance(fargs, tuple) else fargs,
           fkw if isinstance(fkw, dict) else {})
    return Opaque("thread id")


@h("concurrent.futures.thread.ThreadPoolExecutor.submit", "concurrent.futures._base.Executor.submit",
   "multiprocessing.pool.Pool.apply", "multiprocessing.pool.Pool.apply_async",
   "asyncio.events.AbstractEventLoop.run_in_executor", "asyncio.base_events.BaseEventLoop.run_in_executor",
   "sched.scheduler.enter", "sched.scheduler.enterabs")
def _submit(I, f, args, kwargs):
    name = canonical(f.obj)
    if name.startswith("multiprocessing.pool.Pool.apply"):
        need(args, 2, 4, name=name)
        fn = args[1]
        fargs = arg(args, kwargs, 2, "args", ())
        fkw = arg(args, kwargs, 3, "kwds", {})
        I.call(fn, tuple(I.consume(fargs)), fkw if isinstance(fkw, dict) else {})
        return Opaque("result")
    if name.endswith("run_in_executor"):
        need(args, 3, 99, name=name)
        I.call(args[2], args[3:])
        return Opaque("future")
    if name.startswith("sched."):
        # enter(delay, priority, action, argument=()): runs on scheduler.run()
        action = arg(args, kwargs, 3, "action")
        aargs = arg(args, kwargs, 4, "argument", ())
        akw = arg(args, kwargs, 5, "kwargs", {})
        st = args[0] if args else None
        if isinstance(st, Inst) and action is not None:
            st.extra.setdefault("events", []).append((action, aargs, akw))
        return Opaque("event")
    need(args, 2, 99, name="submit")
    I.call(args[1], args[2:], kwargs)
    return Opaque("future")


@h("concurrent.futures._base.Executor.map", "multiprocessing.pool.Pool.map",
   "multiprocessing.pool.Pool.map_async", "multiprocessing.pool.Pool.imap",
   "multiprocessing.pool.Pool.imap_unordered", "multiprocessing.pool.Pool.starmap",
   "multiprocessing.pool.Pool.starmap_async")
def _executor_map(I, f, args, kwargs):
    need(args, 3, 99, name="map")
    fn = args[1]
    star = canonical(f.obj).endswith(("starmap", "starmap_async"))
    pool_map = canonical(f.obj).startswith("multiprocessing.pool.Pool")
    its = args[2:3] if pool_map else args[2:]
    results = []
    for items in _zip_gen(I, its):
        if star:
            items = tuple(I.consume(items[0]))
        results.append(I.call(fn, items))
    return Iter(iter(results), "executor map")


@h("weakref.finalize")
def _finalize(I, f, args, kwargs):
    need(args, 2, 99, name="finalize")
    I.call(args[1], args[2:], kwargs)
    return Opaque("finalizer")


@h("bdb.Bdb.runcall", "cProfile.Profile.runcall", "profile.Profile.runcall", "trace.Trace.runfunc")
def _runcall(I, f, args, kwargs):
    name = canonical(f.obj)
    need(args, 2, 99, name=name)
    I.call(args[1], args[2:], kwargs)
    return Opaque("result")


@h("contextlib.ExitStack.callback", "contextlib._BaseExitStack.callback")
def _exitstack_callback(I, f, args, kwargs):
    # registered, runs only when the stack is closed
    need(args, 2, 99, name="callback")
    st = args[0]
    if isinstance(st, Inst):
        st.extra.setdefault("callbacks", []).append((args[1], args[2:], kwargs))
    return args[1]


@h("contextlib.ExitStack.close", "contextlib._BaseExitStack.close", "contextlib.ExitStack.__exit__",
   "contextlib._BaseExitStack.__exit__")
def _exitstack_close(I, f, args, kwargs):
    st = args[0] if args else None
    if isinstance(st, Inst):
        for fn, a, k in reversed(st.extra.pop("callbacks", [])):
            I.call(fn, a, k)
    return None


@h("pdb.runcall")
def _pdb_runcall(I, f, args, kwargs):
    need(args, 1, 99, name="runcall")
    I.call(args[0], args[1:], kwargs)
    return Opaque("result")


@h("unittest.mock.CallableMixin.__call__", "unittest.mock.CallableMixin._mock_call",
   "unittest.mock.CallableMixin._execute_mock_call")
def _mock_call(I, f, args, kwargs):
    need(args, 1, 99, name="mock call")
    inst = args[0]
    if isinstance(inst, Inst):
        se = inst.attrs.get("side_effect")
        wraps = inst.attrs.get("_mock_wraps")
        if inst.inited:
            try:
                import unittest.mock as um

                bound = I.signature(um.NonCallableMock.__init__).bind(inst, *inst.args, **inst.kwargs).arguments
            except Exception:  # noqa: BLE001
                bound = {}
            if se is None:
                se = bound.get("side_effect")
            if wraps is None:
                wraps = bound.get("wraps")
            if se is None and "kwargs" in bound and isinstance(bound["kwargs"], dict):
                se = bound["kwargs"].get("side_effect")
        for target in (se, wraps):
            if target is not None and is_callable_value(target) and not isinstance(target, (Iter,)):
                if isinstance(target, Real) and isinstance(target.obj, type) and issubclass(target.obj, BaseException):
                    continue
                I.call(target, args[1:], kwargs)
                break
    return Opaque("mock result")


@h("asyncio.runners.run", "asyncio.base_events.BaseEventLoop.run_until_complete",
   "asyncio.runners.Runner.run")
def _asyncio_run(I, f, args, kwargs):
    for a in args:
        if isinstance(a, Coro):
            I.finding(a.kind, a.detail)
    return Opaque("coroutine result")


def _coro(kind):
    def fn(I, f, args, kwargs):
        return Coro(kind, f"awaited {canonical(f.obj) if isinstance(f, Real) else f.name}{I.fmt_args(args)}")
    return fn


for _n in ("asyncio.streams.open_connection", "asyncio.streams.start_server",
           "asyncio.streams.open_unix_connection", "asyncio.base_events.BaseEventLoop.create_connection",
           "asyncio.base_events.BaseEventLoop.create_server"):
    HANDLERS[_n] = _coro(kb.NET)
for _n in ("asyncio.subprocess.create_subprocess_shell", "asyncio.subprocess.create_subprocess_exec",
           "asyncio.base_events.BaseEventLoop.subprocess_shell",
           "asyncio.base_events.BaseEventLoop.subprocess_exec"):
    HANDLERS[_n] = _coro(kb.PROC)
for _n in ("asyncio.streams.open_connection", "asyncio.streams.start_server"):
    kb.CAPS.pop(_n, None)
kb.CAPS.pop("asyncio.subprocess.create_subprocess_shell", None)
kb.CAPS.pop("asyncio.subprocess.create_subprocess_exec", None)
kb.CAPS.pop("asyncio.base_events.BaseEventLoop.create_connection", None)


@h("timeit.Timer.timeit", "timeit.Timer.repeat", "timeit.Timer.autorange")
def _timer_timeit(I, f, args, kwargs):
    # Timer() already compiled its template source; calling timeit runs the
    # statement and, when it is a callable, calls it
    inst = args[0] if args else None
    I.finding(kb.CODE, f"call of {canonical(f.obj)} (executes compiled source)")
    if isinstance(inst, Inst):
        stmt = arg(inst.args, inst.kwargs, 0, "stmt", "pass")
        if isinstance(stmt, Sym) and is_callable_value(stmt):
            I.call(stmt, ())
    return Opaque("timing")


@h("timeit.timeit", "timeit.repeat")
def _timeit(I, f, args, kwargs):
    stmt = arg(args, kwargs, 0, "stmt", "pass")
    setup = arg(args, kwargs, 1, "setup", "pass")
    I.finding(kb.CODE, f"call of {canonical(f.obj)}({describe(stmt, 60)}) compiles and runs source")
    for s in (setup, stmt):
        if isinstance(s, Sym) and is_callable_value(s):
            I.call(s, ())
    return Opaque("timing")


@h("timeit.Timer")
def _timer(I, f, args, kwargs):
    I.finding(kb.CODE, f"timeit.Timer({describe(arg(args, kwargs, 0, 'stmt', 'pass'), 60)}) compiles source")
    return Inst(f, args, kwargs, True)


# numpy helpers that call a function
@h("numpy.apply_along_axis", "numpy.apply_over_axes", "numpy.lib._shape_base_impl.apply_along_axis")
def _np_apply(I, f, args, kwargs):
    if args:
        I.call(args[0], (Opaque("array"),) + tuple(args[3:]))
    return Opaque("array")


@h("numpy.piecewise")
def _np_piecewise(I, f, args, kwargs):
    funcs = arg(args, kwargs, 2, "funclist", [])
    if isinstance(funcs, (list, tuple)):
        for fn in funcs:
            if isinstance(fn, Sym) and is_callable_value(fn):
                I.call(fn, (Opaque("array"),))
    return Opaque("array")


# ------------------------------------------------------------- file objects
def _bytesio(I, f, args, kwargs):
    data = arg(args, kwargs, 0, "initial_bytes", b"")
    if isinstance(data, (bytes, bytearray)):
        return FileObj(bytes(data))
    if isinstance(data, str) and canonical(f.obj) in ("io.StringIO",):
        return FileObj(data)
    if isinstance(data, Sym):
        return FileObj(None)
    if data is None:
        return FileObj(b"")
    raise LoadError("a bytes-like object is required")


for _n in ("io.BytesIO", "io.StringIO"):
    HANDLERS[_n] = _bytesio
    NEW_HANDLERS[_n] = lambda I, f, args, kwargs: FileObj(b"")


@h("io.BufferedReader", "io.BufferedRandom", "io.TextIOWrapper", "io.BufferedRWPair")
def _buffered(I, f, args, kwargs):
    raw = arg(args, kwargs, 0, "raw")
    if isinstance(raw, FileObj):
        return raw
    return FileObj(None)


def _fm_read(I, fo, args, kwargs):
    if fo.data is None:
        return Opaque("data read from a file")
    n = args[0] if args else -1
    if not isinstance(n, int) or n is None or n < 0:
        out = fo.data[fo.pos:]
    else:
        out = fo.data[fo.pos:fo.pos + n]
    fo.pos += len(out)
    return out


def _fm_getvalue(I, fo, args, kwargs):
    return fo.data if fo.data is not None else Opaque("buffer content")


def _fm_seek(I, fo, args, kwargs):
    if args and isinstance(args[0], int) and fo.data is not None:
        whence = args[1] if len(args) > 1 else 0
        base = {0: 0, 1: fo.pos, 2: len(fo.data)}.get(whence, 0)
        fo.pos = max(0, base + args[0])
    return fo.pos


def _fm_write(I, fo, args, kwargs):
    if fo.path is not None:
        I.finding(kb.FILE, f"write to file {describe(fo.path)}")
        return Opaque("written")
    if args and isinstance(args[0], (bytes, bytearray, str)) and fo.data is not None and \
            type(fo.data) is type(bytes(args[0]) if isinstance(args[0], bytearray) else args[0]):
        b = bytes(args[0]) if isinstance(args[0], bytearray) else args[0]
        fo.data = fo.data[:fo.pos] + b + fo.data[fo.pos + len(b):]
        fo.pos += len(b)
        return len(b)
    fo.data = None
    return Opaque("written")


def _fm_readline(I, fo, args, kwargs):
    if fo.data is None:
        return Opaque("line")
    nl = b"\n" if isinstance(fo.data, bytes) else "\n"
    i = fo.data.find(nl, fo.pos)
    end = len(fo.data) if i < 0 else i + 1
    out = fo.data[fo.pos:end]
    fo.pos = end
    return out


def _fm_self(I, fo, args, kwargs):
    return fo


def _fm_none(I, fo, args, kwargs):
    return None


class _FileMethod:
    def __init__(self, fn, name):
        self.fn = fn
        self.__name__ = name


FILE_METHODS.update({
    "read": _fm_read, "read1": _fm_read, "getvalue": _fm_getvalue, "getbuffer": _fm_getvalue,
    "seek": _fm_seek, "write": _fm_write, "readline": _fm_readline, "__enter__": _fm_self,
    "close": _fm_none, "flush": _fm_none, "__exit__": _fm_none, "tell": lambda I, fo, a, k: fo.pos,
    "readable": lambda I, fo, a, k: True, "seekable": lambda I, fo, a, k: True,
})


def _file_method_handler(I, f, args, kwargs):
    # f is Real(<python function from FILE_METHODS>)
    if not args or not isinstance(args[0], FileObj):
        return Opaque("file method result")
    return f.obj(I, args[0], args[1:], kwargs)


for _fn in set(FILE_METHODS.values()):
    _key = f"modelgate.handlers.{getattr(_fn, '__qualname__', '')}"
    HANDLERS[_key] = _file_method_handler


# --------------------------------------------------------- nested loaders
def _nested(kind):
    def fn(I, f, args, kwargs):
        name = canonical(f.obj) if isinstance(f, Real) else f.name
        return I.nested_loader(kind, name, args, kwargs)
    return fn


for _n, _kind in kb.NESTED_LOADERS.items():
    if _kind != "none":
        HANDLERS[_n] = _nested(_kind)
HANDLERS["pickle.Unpickler"] = lambda I, f, args, kwargs: Inst(f, args, kwargs, True)
HANDLERS["pickle._Unpickler"] = HANDLERS["pickle.Unpickler"]


@h("pickle.Unpickler.load", "pickle._Unpickler.load")
def _unpickler_load(I, f, args, kwargs):
    need(args, 1, name="load")
    u = args[0]
    fileobj = None
    if isinstance(u, Inst):
        fileobj = arg(u.args, u.kwargs, 0, "file")
    return I.nested_loader("pickle-file", "pickle.Unpickler.load", (fileobj,), {})


@h("numpy.lib.npyio.NpzFile.__getitem__", "numpy.lib._npyio_impl.NpzFile.__getitem__")
def _npz_getitem(I, f, args, kwargs):
    return Opaque("array")


# ------------------------------------------------------------- decoding
def _limited_decompress(data, wbits=15, limit=64 * 1024 * 1024):
    import zlib

    return zlib.decompressobj(wbits).decompress(data, limit)


@h("zlib.decompress")
def _zlib_decompress(I, f, args, kwargs):
    data = arg(args, kwargs, 0, "data")
    wbits = arg(args, kwargs, 1, "wbits", 15)
    b = I.to_bytes(data)
    if b is None or not isinstance(wbits, int):
        if isinstance(data, Sym) and _maybe_bytes(data) and (isinstance(wbits, (int, Opaque))):
            return Opaque("decompressed data")
        raise LoadError("a bytes-like object is required")
    try:
        import zlib

        d = zlib.decompressobj(wbits)
        out = d.decompress(b, 64 * 1024 * 1024)
        if not d.eof and not d.unconsumed_tail:
            raise LoadError("incomplete or truncated stream")
        return out
    except LoadError:
        raise
    except Exception:  # noqa: BLE001
        raise LoadError("zlib.error") from None


@h("gzip.decompress")
def _gzip_decompress(I, f, args, kwargs):
    from modelgate import containers

    raw = arg(args, kwargs, 0, "data")
    data = I.to_bytes(raw)
    if data is None:
        if isinstance(raw, Sym) and _maybe_bytes(raw):
            return Opaque("decompressed data")
        raise LoadError("a bytes-like object is required")
    out, ok = containers.gunzip(data)
    if not ok:
        raise LoadError("gzip error")
    return out


@h("bz2.decompress")
def _bz2_decompress(I, f, args, kwargs):
    from modelgate import containers

    raw = arg(args, kwargs, 0, "data")
    data = I.to_bytes(raw)
    if data is None:
        if isinstance(raw, Sym) and _maybe_bytes(raw):
            return Opaque("decompressed data")
        raise LoadError("a bytes-like object is required")
    out, ok = containers.bunzip(data)
    if not ok:
        raise LoadError("bz2 error")
    return out


@h("lzma.decompress")
def _lzma_decompress(I, f, args, kwargs):
    from modelgate import containers

    raw = arg(args, kwargs, 0, "data")
    data = I.to_bytes(raw)
    if data is None:
        if isinstance(raw, Sym) and _maybe_bytes(raw):
            return Opaque("decompressed data")
        raise LoadError("a bytes-like object is required")
    out, ok = containers.unxz(data, auto=True)
    if not ok:
        raise LoadError("lzma error")
    return out


_SAFE_CODECS = {
    "base64", "base_64", "base64_codec", "hex", "hex_codec", "zlib", "zip", "zlib_codec",
    "bz2", "bz2_codec", "rot13", "rot_13", "latin1", "latin-1", "latin_1", "iso-8859-1",
    "iso8859-1", "l1", "utf-8", "utf8", "utf_8", "u8", "ascii", "us-ascii", "unicode_escape",
    "raw_unicode_escape", "utf-16", "utf-32", "utf_16", "utf_32", "cp1252", "uu", "uu_codec",
    "quopri", "quopri_codec", "quotedprintable", "utf-16-le", "utf-16-be", "punycode", "idna",
    "unicode-escape", "raw-unicode-escape", "string_escape", "escape", "charmap",
}


@h("codecs.decode", "codecs.encode")
def _codecs(I, f, args, kwargs):
    obj = arg(args, kwargs, 0, "obj")
    enc = arg(args, kwargs, 1, "encoding", "utf-8")
    errors = arg(args, kwargs, 2, "errors", "strict")
    if isinstance(enc, str):
        try:
            codecs.lookup(enc)
        except LookupError:
            raise LoadError(f"unknown encoding: {enc}") from None
    if not is_plain(obj) or not isinstance(enc, str) or not isinstance(errors, str):
        return Opaque("encoded/decoded data")
    norm = enc.lower().replace(" ", "")
    if norm not in _SAFE_CODECS and norm.replace("-", "_") not in _SAFE_CODECS:
        return Opaque("encoded/decoded data")
    if norm in ("zlib", "zip", "zlib_codec", "bz2", "bz2_codec") and canonical(f.obj) == "codecs.decode":
        b = I.to_bytes(obj)
        if b is None:
            raise LoadError("bad input")
        from modelgate import containers

        if norm.startswith("bz2"):
            out, ok = containers.bunzip(b)
        else:
            try:
                out = _limited_decompress(b)
                ok = True
            except Exception:  # noqa: BLE001
                ok = False
        if not ok:
            raise LoadError("decompression failed")
        return out
    return I.eval_pure(f.obj, canonical(f.obj), args, kwargs)


@h("_codecs.encode")
def _codecs_encode(I, f, args, kwargs):
    return _codecs(I, f, args, kwargs)


# --------------------------------------------------------------- logging
@h("logging.config.dictConfig")
def _dictconfig(I, f, args, kwargs):
    I.finding(kb.CODE, f"call of logging.config.dictConfig({describe(arg(args, kwargs, 0, 'config'), 60)})")
    return None


@h("xmlrpc.client.ServerProxy.__getattr__")
def _serverproxy_getattr(I, f, args, kwargs):
    # proxy.<name> is a remote method; calling it opens a connection
    return Wrapper(I.unres_obj("xmlrpc.client._Method.__call__"), "xmlrpc remote method")


@h("sched.scheduler.run")
def _sched_run(I, f, args, kwargs):
    st = args[0] if args else None
    if isinstance(st, Inst):
        for action, aargs, akw in st.extra.pop("events", []):
            I.call(action, tuple(I.consume(aargs)), akw if isinstance(akw, dict) else {})
    return None


# bytearray.__new__ makes an empty array (the content is set by __init__)
NEW_HANDLERS["builtins.bytearray"] = lambda I, f, args, kwargs: bytearray()
# type.__new__(type, name, bases, ns) builds a class
def _type_new(I, f, args, kwargs):
    if len(args) in (1, 3):
        return _type(I, f, args, kwargs)
    raise LoadError("type.__new__() takes exactly 3 arguments")


NEW_HANDLERS["builtins.type"] = _type_new


def _maybe_bytes(v):
    from modelgate.nested import _maybe_bytes as mb
    return mb(v)



# --------------------------------------------------- third-party callers
@h("torch._tensor._rebuild_from_type_v2", "torch._tensor._rebuild_from_type",
   "torch.tensor._rebuild_from_type_v2", "torch.tensor._rebuild_from_type")
def _torch_rebuild_from_type(I, f, args, kwargs):
    """``ret = func(*args)`` then a subclass cast: func is called."""
    func = arg(args, kwargs, 0, "func")
    fargs = arg(args, kwargs, 2, "args", ())
    if func is not None:
        I.call(func, tuple(I.consume(fargs)) if not isinstance(fargs, tuple) else fargs)
    return Opaque("tensor")


_CLOUDPICKLE_TYPES = {"CodeType": "builtins.code", "FunctionType": "builtins.function",
                      "LambdaType": "builtins.function", "MethodType": "builtins.method"}


@h("cloudpickle.cloudpickle._builtin_type", "cloudpickle.cloudpickle_fast._builtin_type")
def _cloudpickle_builtin_type(I, f, args, kwargs):
    name = arg(args, kwargs, 0, "name")
    if name == "CodeType":
        return Real(types.CodeType, "builtins.code")
    if name in ("FunctionType", "LambdaType"):
        return Real(types.FunctionType, "builtins.function")
    if name == "MethodType":
        return Real(types.MethodType, "builtins.method")
    return Opaque(f"types.{name}")
