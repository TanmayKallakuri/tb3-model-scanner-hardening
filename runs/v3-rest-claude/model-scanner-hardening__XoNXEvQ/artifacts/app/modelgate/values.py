"""Symbolic Python values used by the pickle simulator.

Nothing here ever imports or calls an object named by an artifact.  Plain
data (None, bool, int, float, complex, str, bytes, and tuples / frozensets of
plain data) is represented by the real Python value; everything else by one of
the classes below, compared by identity like the real objects would be.
"""
from __future__ import annotations


class SimError(Exception):
    """The real loader would raise at this point: the load stops here."""


class Obj:
    """Base of all symbolic objects (identity semantics, hashable)."""

    __slots__ = ("__weakref__",)
    unhashable = False

    def describe(self) -> str:
        return type(self).__name__


class Ref(Obj):
    """A named object that lives in a module: function, class, builtin...

    ``kind`` is the kind recorded in the stdlib index ("f" python function,
    "b" builtin, "c" class, "md" method descriptor, "x" other callable,
    "i" special instance) or "?" for objects outside the index (third-party
    packages), which may be modules, classes or functions.
    """

    __slots__ = ("oid", "kind", "flags", "sig")

    def __init__(self, oid: str, kind: str = "?", flags: str = "", sig=None):
        self.oid = oid
        self.kind = kind
        self.flags = flags
        self.sig = sig

    def describe(self) -> str:
        return self.oid

    def __repr__(self):
        return f"<Ref {self.oid}>"


class Module(Obj):
    """A module object; ``overlay`` holds attributes rebound by the artifact."""

    __slots__ = ("name", "overlay", "deleted", "indexed", "fresh")

    def __init__(self, name: str, indexed: bool, fresh: bool = False):
        self.name = name
        self.overlay: dict = {}
        self.deleted: set = set()
        self.indexed = indexed
        self.fresh = fresh  # created with types.ModuleType(): no indexed attributes

    def describe(self) -> str:
        return f"module {self.name}"

    def __repr__(self):
        return f"<Module {self.name}>"


class PList(Obj):
    __slots__ = ("items",)
    unhashable = True

    def __init__(self, items=None):
        self.items = list(items) if items is not None else []

    def describe(self):
        return "list"


class PDict(Obj):
    __slots__ = ("d",)
    unhashable = True

    def __init__(self, d=None):
        self.d = dict(d) if d is not None else {}

    def describe(self):
        return "dict"


class PSet(Obj):
    __slots__ = ("s",)
    unhashable = True

    def __init__(self, s=None):
        self.s = set(s) if s is not None else set()

    def describe(self):
        return "set"


class PByteArray(Obj):
    __slots__ = ("b",)
    unhashable = True

    def __init__(self, b=b""):
        self.b = bytearray(b)

    def describe(self):
        return "bytearray"


class Instance(Obj):
    """An instance of a class (a Ref class or a ClassObj).

    ``d`` is the instance ``__dict__`` (a PDict) or None when the class has
    no instance dictionary; ``extra`` keeps constructor arguments for the few
    classes whose behaviour the simulator models.
    """

    __slots__ = ("cls", "d", "extra")

    def __init__(self, cls, d=True, extra=None):
        self.cls = cls
        self.d = PDict() if d is True else d
        self.extra = extra if extra is not None else {}

    def describe(self):
        c = self.cls
        return f"instance of {c.describe() if isinstance(c, Obj) else c}"


class Partial(Instance):
    """functools.partial (also used for other forwarding wrappers)."""

    __slots__ = ("func", "args", "kw")

    def __init__(self, cls, func, args=(), kw=None):
        super().__init__(cls)
        self.func = func
        self.args = tuple(args)
        self.kw = dict(kw or {})

    def describe(self):
        f = self.func
        return f"partial({f.describe() if isinstance(f, Obj) else repr(f)[:40]})"


class AttrGetter(Instance):
    __slots__ = ("names",)

    def __init__(self, cls, names):
        super().__init__(cls, d=None)
        self.names = list(names)


class ItemGetter(Instance):
    __slots__ = ("keys",)

    def __init__(self, cls, keys):
        super().__init__(cls, d=None)
        self.keys = list(keys)


class MethodCaller(Instance):
    __slots__ = ("name", "args", "kw")

    def __init__(self, cls, name, args=(), kw=None):
        super().__init__(cls, d=None)
        self.name = name
        self.args = tuple(args)
        self.kw = dict(kw or {})


class BoundMethod(Obj):
    """``obj.method``: calling it calls ``func`` with ``obj`` prepended."""

    __slots__ = ("self_", "func")

    def __init__(self, self_, func):
        self.self_ = self_
        self.func = func

    def describe(self):
        f = self.func
        return f"bound {f.describe() if isinstance(f, Obj) else f}"


class LazyIter(Obj):
    """A lazy iterator (map, filter, starmap, generator...): nothing runs
    until something consumes it."""

    __slots__ = ("kind", "params", "consumed")

    def __init__(self, kind, **params):
        self.kind = kind
        self.params = params
        self.consumed = False

    def describe(self):
        return f"{self.kind} iterator"


class ClassObj(Obj):
    """A class created by the artifact (type(name, bases, ns))."""

    __slots__ = ("name", "bases", "ns")

    def __init__(self, name, bases, ns):
        self.name = name
        self.bases = tuple(bases)
        self.ns = ns  # PDict

    def describe(self):
        return f"class {self.name!r} built by the artifact"


class FunctionObj(Obj):
    """A function materialised from bytecode by the artifact."""

    __slots__ = ("code", "name")

    def __init__(self, code, name="<function>"):
        self.code = code
        self.name = name

    def describe(self):
        return "function materialised from bytecode"


class CodeObj(Obj):
    __slots__ = ("origin",)

    def __init__(self, origin):
        self.origin = origin

    def describe(self):
        return f"code object ({self.origin})"


class Coroutine(Obj):
    __slots__ = ("func", "args", "kw")

    def __init__(self, func, args, kw):
        self.func = func
        self.args = args
        self.kw = kw


class FileObj(Obj):
    """An in-memory file (io.BytesIO / io.StringIO) built from artifact data."""

    __slots__ = ("data", "pos", "text")

    def __init__(self, data, text=False):
        self.data = data
        self.pos = 0
        self.text = text


class ModuleDict(Obj):
    """``vars(module)`` / ``module.__dict__``."""

    __slots__ = ("module",)
    unhashable = True

    def __init__(self, module):
        self.module = module


class ClassDict(Obj):
    """``vars(cls)`` / ``cls.__dict__`` (a read-only mappingproxy)."""

    __slots__ = ("cls",)
    unhashable = True

    def __init__(self, cls):
        self.cls = cls


class SysModules(Obj):
    """``sys.modules``: maps module names to module objects."""

    __slots__ = ("overlay", "deleted")
    unhashable = True

    def __init__(self):
        self.overlay: dict = {}
        self.deleted: set = set()


class Opaque(Obj):
    """A value the simulator does not model in detail (the result of a call
    with no capability, a data attribute, ...).  ``tainted`` marks values found
    dynamically at run time (``object.__subclasses__()``, frames, the garbage
    collector): they may be anything, so calling one cannot be judged."""

    __slots__ = ("desc", "attrs", "tainted")

    def __init__(self, desc="object", tainted=False):
        self.desc = desc
        self.attrs = None
        self.tainted = tainted

    def describe(self):
        return self.desc

    def __repr__(self):
        return f"<Opaque {self.desc}>"


NATIVE_SCALARS = (type(None), bool, int, float, complex, str, bytes)


def is_native(v) -> bool:
    """Plain data whose real Python value is used directly."""
    if isinstance(v, NATIVE_SCALARS):
        return True
    if isinstance(v, (tuple, frozenset)):
        return all(is_native(x) for x in v)
    return False


def is_hashable(v) -> bool:
    if isinstance(v, Obj):
        return not v.unhashable
    if isinstance(v, tuple):
        return all(is_hashable(x) for x in v)
    return True


def describe(v) -> str:
    if isinstance(v, Obj):
        return v.describe()
    r = repr(v)
    return r if len(r) <= 60 else r[:57] + "..."
