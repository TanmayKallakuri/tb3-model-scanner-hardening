"""Shared work limits. No artifact content is ever imported or executed."""
import time

MAX_BYTES = 256 * 1024 * 1024
MAX_ITEMS = 100_000


class AnalysisLimit(Exception):
    pass


class Budget:
    def __init__(self):
        self.deadline = time.monotonic() + 45
        self.work = 0
        self.expanded = 0

    def tick(self, n=1):
        self.work += n
        if self.work > 2_000_000 or time.monotonic() > self.deadline:
            raise AnalysisLimit('static analysis work limit exceeded')

    def add_bytes(self, n):
        self.expanded += n
        self.tick()
        if n > MAX_BYTES or self.expanded > 2 * MAX_BYTES:
            raise AnalysisLimit('expanded artifact size limit exceeded')
