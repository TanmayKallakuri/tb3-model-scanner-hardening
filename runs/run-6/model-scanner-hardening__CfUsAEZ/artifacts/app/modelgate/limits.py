"""Shared work limits for parsers and the abstract interpreter."""
import time


class AnalysisLimit(Exception):
    pass


class Budget:
    def __init__(self):
        self.deadline = time.monotonic() + 45
        self.steps = 0
        self.expanded = 0

    def tick(self, count=1):
        self.steps += count
        if self.steps > 2000000 or time.monotonic() > self.deadline:
            raise AnalysisLimit("static analysis work limit exceeded")

    def expand(self, size):
        self.expanded += size
        if self.expanded > 512 * 1024 * 1024:
            raise AnalysisLimit("expanded artifact exceeds 512 MiB")
        self.tick()
