"""Values of the pickle emulator.

Concrete Python data (None, bool, int, float, complex, str, bytes, bytearray,
tuple, list, dict, set, frozenset, slice, range) is represented by itself.
Everything the unpickler would obtain from code (modules, globals, objects
built by calls) is represented by the symbolic classes below.  None of them
ever wraps a real object from an artifact.
"""
from __future__ import annotations


class Sym:
    """Base class of symbolic values (hashable by identity)."""

    __slots__ = ("__weakref__",)

    def describe(self) -> str:
        return type(self).__name__


class Mod(Sym):
    """A module object."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def describe(self):
        return f"module {self.name}"


class Glob(Sym):
    """A global: function, class or other module-level object.

    ``name`` is the canonical dotted name when the knowledge base knows the
    object, otherwise the dotted path it was reached by.  ``kind`` is
    ``func``, ``class``, ``obj`` or ``unknown``.
    """

    __slots__ = ("name", "kind")

    def __init__(self, name, kind="unknown"):
        self.name = name
        self.kind = kind

    def describe(self):
        return self.name


class Inst(Sym):
    """An instance of a class, with its instance ``__dict__``."""

    __slots__ = ("cls", "attrs", "args", "kwargs", "items")

    def __init__(self, cls, args=None, kwargs=None):
        self.cls = cls
        self.attrs = {}
        self.args = args
        self.kwargs = kwargs
        self.items = None

    def describe(self):
        return f"instance of {describe(self.cls)}"


class Opaque(Sym):
    """A value the emulator cannot know (result of an unmodeled call).

    ``taint`` holds the capability kind when the value may be, or give
    access to, a capability-bearing callable.
    """

    __slots__ = ("what", "taint", "attrs", "nocall")

    def __init__(self, what="value", taint=None, nocall=False):
        self.what = what
        self.taint = taint
        self.attrs = None
        self.nocall = nocall  # known not to be callable (a dict, a list, a frame)

    def describe(self):
        return self.what


class Partial(Sym):
    __slots__ = ("func", "args", "kwargs", "attrs")

    def __init__(self, func, args=(), kwargs=None):
        self.func = func
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.attrs = {}

    def describe(self):
        return f"functools.partial({describe(self.func)})"


class BoundMethod(Sym):
    """``obj.method``: ``func`` is called with ``obj`` prepended."""

    __slots__ = ("obj", "func")

    def __init__(self, obj, func):
        self.obj = obj
        self.func = func

    def describe(self):
        return f"{describe(self.func)} bound to {describe(self.obj)}"


class Wrapper(Sym):
    """A callable that calls ``func`` with its own arguments when called."""

    __slots__ = ("func", "how")

    def __init__(self, func, how):
        self.func = func
        self.how = how

    def describe(self):
        return f"{self.how}({describe(self.func)})"


class Decorator(Sym):
    """A decorator factory result: calling it with ``f`` gives ``Wrapper(f)`` or ``f``."""

    __slots__ = ("how",)

    def __init__(self, how):
        self.how = how


class AttrGetter(Sym):
    __slots__ = ("names",)

    def __init__(self, names):
        self.names = tuple(names)

    def describe(self):
        return f"operator.attrgetter{self.names!r}"


class ItemGetter(Sym):
    __slots__ = ("items",)

    def __init__(self, items):
        self.items = tuple(items)


class MethodCaller(Sym):
    __slots__ = ("name", "args", "kwargs")

    def __init__(self, name, args, kwargs):
        self.name = name
        self.args = tuple(args)
        self.kwargs = dict(kwargs)

    def describe(self):
        return f"operator.methodcaller({self.name!r})"


class Lazy(Sym):
    """A lazy iterator; consuming it runs the calls it would make."""

    __slots__ = ("kind", "func", "sources", "extra", "consumed")

    def __init__(self, kind, func=None, sources=(), extra=None):
        self.kind = kind
        self.func = func
        self.sources = list(sources)
        self.extra = extra
        self.consumed = 0

    def describe(self):
        return f"lazy {self.kind} iterator"


class BytesIOV(Sym):
    """An in-memory binary file; ``data`` is None when unknown."""

    __slots__ = ("data", "pos")

    def __init__(self, data, pos=0):
        self.data = data
        self.pos = pos


class ModDict(Sym):
    """``vars(module)`` / ``module.__dict__``."""

    __slots__ = ("module",)

    def __init__(self, module):
        self.module = module


class SysModules(Sym):
    __slots__ = ()


class ExtRegistry(Sym):
    """``copyreg._inverted_registry`` / ``_extension_cache`` / ``_extension_registry``."""

    __slots__ = ("which",)

    def __init__(self, which):
        self.which = which


class CodeObj(Sym):
    __slots__ = ("origin",)

    def __init__(self, origin):
        self.origin = origin


class Coroutine(Sym):
    __slots__ = ("kind", "detail")

    def __init__(self, kind, detail):
        self.kind = kind
        self.detail = detail


class Runner(Sym):
    """A thread, timer, process, executor, scheduler or event loop."""

    __slots__ = ("what", "target", "args", "kwargs", "queue")

    def __init__(self, what, target=None, args=(), kwargs=None):
        self.what = what
        self.target = target
        self.args = tuple(args) if isinstance(args, (tuple, list)) else ()
        self.kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
        self.queue = []


class UnpicklerObj(Sym):
    __slots__ = ("file", "python")

    def __init__(self, file, python):
        self.file = file
        self.python = python


class XmlrpcProxy(Sym):
    __slots__ = ("uri",)

    def __init__(self, uri):
        self.uri = uri


class XmlrpcMethod(Sym):
    __slots__ = ("proxy", "name")

    def __init__(self, proxy, name):
        self.proxy = proxy
        self.name = name


class NpzObj(Sym):
    __slots__ = ("members", "allow_pickle")

    def __init__(self, members, allow_pickle):
        self.members = members
        self.allow_pickle = allow_pickle


class Dtype(Sym):
    """A numpy dtype built in the pickle (joblib needs its item size)."""

    __slots__ = ("spec", "attrs")

    def __init__(self, spec):
        self.spec = spec
        self.attrs = {}


class CmpKey(Sym):
    __slots__ = ("func",)

    def __init__(self, func):
        self.func = func


class ODict(dict):
    """An OrderedDict/defaultdict/Counter: a dict that also has a __dict__."""

    __slots__ = ("attrs", "factory")

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.attrs = {}
        self.factory = None

    __hash__ = None


def describe(value) -> str:
    if isinstance(value, Sym):
        return value.describe()
    text = repr(value) if not isinstance(value, (list, dict, set, tuple)) else type(value).__name__
    return text if len(text) <= 80 else text[:77] + "..."
