"""Shared resource limits and findings for a static analysis."""
from __future__ import annotations

import os
import time

from modelgate.report import Finding


class LimitError(Exception):
    pass


class Context:
    def __init__(self, artifact=''):
        self.findings = []
        self.artifact = os.path.abspath(artifact) if artifact else ''
        self.is_directory = bool(artifact and os.path.isdir(artifact))
        self.deadline = time.monotonic() + 45
        self.steps = 0
        self.expanded = 0
        self.max_bytes = 512 * 1024 * 1024
        # These are symbolic module namespaces, never actual imported modules.
        self.modules = {}
        self.module_dicts = {}

    def tick(self, amount=1):
        self.steps += amount
        if self.steps > 2000000 or time.monotonic() > self.deadline:
            raise LimitError('static analysis resource limit')

    def account(self, size):
        self.expanded += size
        self.tick()
        if self.expanded > self.max_bytes:
            raise LimitError('expanded artifact exceeds analysis byte limit')

    def add(self, kind, location, detail):
        if len(self.findings) < 100:
            self.findings.append(Finding(kind, str(location), str(detail)))

    def outside(self, path):
        if not isinstance(path, (str, bytes)):
            return True
        try:
            target = os.path.abspath(os.fsdecode(path))
            if self.is_directory:
                return os.path.commonpath((target, self.artifact)) != self.artifact
            return target != self.artifact
        except (ValueError, TypeError):
            return True
