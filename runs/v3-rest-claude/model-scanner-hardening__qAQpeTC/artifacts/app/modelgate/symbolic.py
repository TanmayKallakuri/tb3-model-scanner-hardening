"""Values manipulated by the pickle emulator.

Plain data (None, bool, int, float, complex, str, bytes, bytearray, tuple,
list, dict, set, frozenset) is represented by the real Python value; the
containers may hold symbolic values.  Everything else is a ``Sym``: a
reference to a global, an object built by a call, a lazy iterator and so on.
Symbols hash and compare by identity, like the objects they stand for.
"""
from __future__ import annotations

PLAIN_SCALARS = (type(None), bool, int, float, complex, str, bytes)
PLAIN_CONTAINERS = (tuple, list, dict, set, frozenset, bytearray)


class LoadError(Exception):
    """The real loader raises here: nothing after this point executes."""


class Exhausted(BaseException):
    """The analysis budget for one artifact is spent (a BaseException so that
    no ``except Exception`` on the way swallows it)."""


class Sym:
    """Base of all symbolic values."""

    def describe(self) -> str:  # pragma: no cover - overridden
        return type(self).__name__


class Real(Sym):
    """A real object (module, class, function, descriptor) of a module the
    scanner can import without side effects."""

    def __init__(self, obj, name: str):
        self.obj = obj
        self.name = name

    def describe(self):
        return self.name


class Unres(Sym):
    """A global of a module the scanner does not import (third-party)."""

    def __init__(self, name: str, is_module: bool = False):
        self.name = name
        self.is_module = is_module
        self.attrs: dict = {}  # attributes rebound by the pickle

    def describe(self):
        return self.name


class Inst(Sym):
    """An object created by calling a class or by ``cls.__new__``."""

    def __init__(self, cls, args=(), kwargs=None, inited=True):
        self.cls = cls
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.inited = inited
        self.attrs: dict = {}  # instance __dict__ written by BUILD / setattr
        self.state = None
        self.extra: dict = {}

    def describe(self):
        cls = self.cls.describe() if isinstance(self.cls, Sym) else repr(self.cls)
        return f"<{cls} instance>"


class SymClass(Sym):
    """A class created by the pickle itself (``type(name, bases, ns)``)."""

    def __init__(self, name, bases, ns):
        self.name = name
        self.bases = tuple(bases)
        self.ns = ns  # a dict (the class namespace; may be mutated)

    def describe(self):
        return f"<class {self.name}>"


class Partial(Sym):
    def __init__(self, func, args=(), kwargs=None, dct=None):
        self.func = func
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.dict = dct if dct is not None else {}

    def describe(self):
        f = self.func.describe() if isinstance(self.func, Sym) else repr(self.func)
        return f"functools.partial({f})"


class AttrGetter(Sym):
    def __init__(self, attrs):
        self.attrs = tuple(attrs)

    def describe(self):
        return f"operator.attrgetter{self.attrs!r}"


class ItemGetter(Sym):
    def __init__(self, items):
        self.items = tuple(items)

    def describe(self):
        return "operator.itemgetter"


class MethodCaller(Sym):
    def __init__(self, name, args=(), kwargs=None):
        self.name = name
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})

    def describe(self):
        return f"operator.methodcaller({self.name!r})"


class Bound(Sym):
    """``obj.method``: calling it calls ``func(obj, *args)``."""

    def __init__(self, obj, func, name: str):
        self.obj = obj
        self.func = func
        self.name = name

    def describe(self):
        return f"bound method {self.name}"


class Iter(Sym):
    """A lazy iterator; ``gen`` is a Python generator that performs the
    symbolic calls (map, filter, ...) as it is advanced."""

    def __init__(self, gen, desc: str):
        self.gen = gen
        self.desc = desc

    def describe(self):
        return self.desc


class Opaque(Sym):
    """A value the scanner does not model (result of an unknown call)."""

    def __init__(self, desc: str = "unknown value"):
        self.desc = desc

    def describe(self):
        return self.desc


class Namespace(Sym):
    """``vars(module)`` / ``module.__dict__`` / ``function.__globals__``."""

    def __init__(self, module):
        self.module = module  # a Real (module) or Unres module

    def describe(self):
        return f"vars({self.module.describe()})"


class SysModules(Sym):
    def describe(self):
        return "sys.modules"


class FileObj(Sym):
    """A file object: in-memory (``io.BytesIO(data)``) or a real file."""

    def __init__(self, data=None, path=None):
        self.data = data  # bytes, or None when the content is unknown
        self.path = path
        self.pos = 0

    def describe(self):
        return "io.BytesIO" if self.path is None else f"file {self.path!r}"


class CodeObj(Sym):
    """A code object (from marshal.loads, compile, types.CodeType)."""

    def __init__(self, desc: str):
        self.desc = desc

    def describe(self):
        return self.desc


class Func(Sym):
    """A function materialised from bytecode or source by the artifact."""

    def __init__(self, desc: str):
        self.desc = desc

    def describe(self):
        return self.desc


class Wrapper(Sym):
    """A callable that forwards to ``target`` (lru_cache, staticmethod...)."""

    def __init__(self, target, desc: str):
        self.target = target
        self.desc = desc

    def describe(self):
        return self.desc


class Coro(Sym):
    """A coroutine object; running it performs ``kind``."""

    def __init__(self, kind, detail):
        self.kind = kind
        self.detail = detail

    def describe(self):
        return "coroutine"


def is_plain(v, depth: int = 0) -> bool:
    """True if ``v`` is plain data all the way down (no symbols)."""
    if isinstance(v, PLAIN_SCALARS):
        return True
    if depth > 50:
        return False
    if isinstance(v, (tuple, list, set, frozenset)):
        return all(is_plain(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return all(is_plain(k, depth + 1) and is_plain(x, depth + 1) for k, x in v.items())
    if isinstance(v, bytearray):
        return True
    return False


_REPR = None


def describe(v, limit: int = 80) -> str:
    """Short human-readable rendering (bounded work for huge values)."""
    global _REPR
    if isinstance(v, Sym):
        s = v.describe()
    else:
        try:
            if isinstance(v, (str, bytes, bytearray)) and len(v) > limit:
                s = repr(v[:limit])
            else:
                if _REPR is None:
                    import reprlib

                    _REPR = reprlib.Repr()
                    _REPR.maxstring = limit
                    _REPR.maxother = limit
                    _REPR.maxlevel = 3
                s = _REPR.repr(v)
        except Exception:  # noqa: BLE001
            s = f"<{type(v).__name__}>"
    if len(s) > limit:
        s = s[: limit - 3] + "..."
    return s
