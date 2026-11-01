"""Values used by the pickle interpreter. None are artifact Python objects."""
from dataclasses import dataclass, field


@dataclass(eq=False)
class Global:
    name: str
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Module:
    name: str
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Namespace:
    owner: object


@dataclass(eq=False)
class Instance:
    cls: str
    args: tuple = ()
    attrs: dict = field(default_factory=dict)
    value: object = None


@dataclass(eq=False)
class Bound:
    obj: object
    name: str


@dataclass(eq=False)
class Partial:
    func: object
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Getter:
    kind: str
    args: tuple
    kwargs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Wrapper:
    func: object
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Lazy:
    kind: str
    args: tuple
    index: int = 0
    state: object = None


@dataclass(eq=False)
class Buffer:
    data: object = b""
    pos: int = 0
    text: bool = False


@dataclass(eq=False)
class Unknown:
    reason: str = "unknown value"


@dataclass(eq=False)
class Code:
    data: object = None


MARK = object()
MISSING = object()
