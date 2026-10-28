"""Shared, deterministic work limits for recursive static parsing."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


class ScanLimit(ValueError):
    pass


@dataclass
class Budget:
    deadline: float = field(default_factory=lambda: time.monotonic() + 45.0)
    remaining_bytes: int = 512 * 1024 * 1024
    remaining_steps: int = 2_000_000
    max_blob: int = 256 * 1024 * 1024

    def check(self, steps=1):
        self.remaining_steps -= steps
        if self.remaining_steps < 0 or time.monotonic() > self.deadline:
            raise ScanLimit("static analysis work limit exceeded")

    def bytes(self, size):
        self.check()
        self.remaining_bytes -= size
        if size > self.max_blob or self.remaining_bytes < 0:
            raise ScanLimit("expanded artifact exceeds analysis size limit")

    def depth(self, depth):
        self.check()
        if depth > 40:
            raise ScanLimit("recursive artifact exceeds analysis depth limit")

    def read(self, stream):
        data = stream.read(min(self.max_blob, self.remaining_bytes) + 1)
        self.bytes(len(data))
        return data
