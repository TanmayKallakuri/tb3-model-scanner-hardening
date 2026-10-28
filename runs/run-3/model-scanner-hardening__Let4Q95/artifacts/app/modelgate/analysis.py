"""Shared bounds for static parsing, including recursive deserialization."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


class Unsupported(ValueError):
    """The input cannot be interpreted safely and completely."""


@dataclass
class Budget:
    deadline: float = field(default_factory=lambda: time.monotonic() + 45)
    steps: int = 0
    expanded: int = 0

    def tick(self, count=1):
        self.steps += count
        if self.steps > 4_000_000 or time.monotonic() > self.deadline:
            raise Unsupported('static analysis resource limit exceeded')

    def expand(self, size):
        self.expanded += size
        if self.expanded > 512 * 1024 * 1024:
            raise Unsupported('expanded artifact exceeds analysis limit')
        self.tick()

    def nest(self, depth):
        if depth > 32:
            raise Unsupported('nested serialization exceeds analysis limit')
        self.tick()


MAX_BYTES = 256 * 1024 * 1024
