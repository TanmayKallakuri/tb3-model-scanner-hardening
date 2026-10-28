"""Inert abstract values. None of these objects can run artifact code."""
from dataclasses import dataclass, field


@dataclass(eq=False)
class Ref:
    name: str


@dataclass(eq=False)
class Module:
    name: str
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Obj:
    cls: str
    args: tuple = ()
    attrs: dict = field(default_factory=dict)
    state: object = None


@dataclass(eq=False)
class Partial:
    fn: object
    args: tuple = ()
    kw: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Bound:
    owner: object
    name: str


@dataclass(eq=False)
class Adapter:
    kind: str
    args: tuple = ()
    kw: dict = field(default_factory=dict)


@dataclass(eq=False)
class Lazy:
    kind: str
    args: tuple
    index: int = 0
    done: bool = False
    state: object = None


@dataclass(eq=False)
class Buffer:
    data: object
    pos: int = 0


@dataclass(eq=False)
class Namespace:
    owner: object


class Unknown:
    pass


UNKNOWN = Unknown()
MARK = object()


class LoadError(Exception):
    """The standard loader necessarily stops before another capability call."""


class Unresolved(Exception):
    """A static operation cannot be analyzed; the report must fail closed."""


class EndIteration(Exception):
    pass


class FindingRaised(Exception):
    """The verdict is established; stop before doing unnecessary work."""
