"""Inert values used by the pickle interpreter. None are artifact objects."""
from __future__ import annotations

from dataclasses import dataclass, field
import time


class LoadError(Exception):
    """The real loader would stop here, before executing subsequent opcodes."""


class LimitError(Exception):
    pass


class Detected(Exception):
    def __init__(self, finding):
        self.finding = finding


@dataclass(eq=False)
class Value:
    kind: str
    name: str = ''
    args: tuple = ()
    attrs: dict = field(default_factory=dict)


class Mapping(dict):
    """A data-only mapping with the namespace of an OrderedDict/UserDict."""
    def __init__(self, name, data=()):
        super().__init__(data)
        self.name = name
        self.attrs = {}


UNKNOWN = Value('unknown')
MARK = Value('mark')
MISSING = Value('missing')


class Context:
    def __init__(self):
        self.modules = {}
        self.globals = {}
        self.steps = 0
        self.expanded = 0
        self.deadline = time.monotonic() + 40
        self.nesting = 0

    def tick(self, count=1):
        self.steps += count
        if self.steps > 2_000_000 or time.monotonic() > self.deadline:
            raise LimitError('static analysis work limit exceeded')

    def allocate(self, size):
        self.expanded += size
        if size > 256 * 1024 * 1024 or self.expanded > 512 * 1024 * 1024:
            raise LimitError('expanded data limit exceeded')


def literal(value, depth=0):
    """Bounded exact-builtin trees, with cycles and expansion bombs rejected."""
    remaining = 100000
    active = set()
    def visit(obj, level):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or level > 20:
            return False
        if type(obj) in (str, bytes, bytearray, int, float, bool, complex, type(None), slice, range):
            return True
        if type(obj) not in (list, tuple, set, frozenset, dict) or id(obj) in active:
            return False
        if len(obj) > 100000:
            return False
        active.add(id(obj))
        try:
            if type(obj) is dict:
                return all(visit(k, level + 1) and visit(v, level + 1) for k, v in obj.items())
            return all(visit(v, level + 1) for v in obj)
        finally:
            active.remove(id(obj))
    return visit(value, depth)
