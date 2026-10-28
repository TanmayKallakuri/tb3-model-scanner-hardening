"""Shared work budget for recursive, entirely in-memory inspection."""
import time

MAX_BYTES = 256 * 1024 * 1024
MAX_ITEMS = 250_000


class LimitExceeded(Exception):
    pass


class Budget:
    def __init__(self):
        self.deadline = time.monotonic() + 45
        self.steps = 0
        self.bytes = 0

    def tick(self, n=1):
        self.steps += n
        if self.steps > 2_000_000 or time.monotonic() > self.deadline:
            raise LimitExceeded("static analysis work limit exceeded")

    def add_bytes(self, n):
        self.bytes += n
        if self.bytes > MAX_BYTES:
            raise LimitExceeded("expanded artifact exceeds analysis byte limit")
        self.tick()
