"""Shared, deterministic work limits for recursive static parsing."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


class AnalysisError(Exception):
    """The input cannot be safely or completely analysed."""


@dataclass
class Context:
    artifact: str = ""
    deadline: float = field(default_factory=lambda: time.monotonic() + 50)
    steps: int = 0
    expanded: int = 0
    max_bytes: int = 512 * 1024 * 1024

    def tick(self, count=1):
        self.steps += count
        if self.steps > 4_000_000 or time.monotonic() > self.deadline:
            raise AnalysisError("static analysis work limit exceeded")

    def account(self, size):
        self.tick()
        self.expanded += size
        if size > self.max_bytes or self.expanded > 2 * self.max_bytes:
            raise AnalysisError("expanded artifact exceeds analysis limit")

    def internal(self, path):
        if not self.artifact or not isinstance(path, (str, bytes)):
            return False
        try:
            candidate = os.path.abspath(os.fsdecode(path))
            root = os.path.abspath(self.artifact)
            return candidate == root or (os.path.isdir(root) and os.path.commonpath([candidate, root]) == root)
        except (ValueError, TypeError):
            return False
