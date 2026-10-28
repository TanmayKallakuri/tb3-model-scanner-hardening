"""Pickle stream scanning.

Walks the opcodes of a pickle with ``pickletools.genops`` and reports any
global reference on the blocklist.  ``GLOBAL`` carries the module and name in
its argument; ``STACK_GLOBAL`` (protocol 4 and later) takes them from the two
strings pushed just before it.
"""
from __future__ import annotations

import pickletools

from modelgate.policy import PICKLE_BLOCKLIST, SUSPICIOUS_STRINGS
from modelgate.report import Finding

_STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "STRING", "SHORT_BINSTRING", "BINSTRING"}
_MEMO_PUT_OPS = {"MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"}  # the pickler memoizes every string it writes


def scan_pickle(data: bytes, location: str) -> list[Finding]:
    findings: list[Finding] = []
    recent: list[str] = []  # the last two string pushes, for STACK_GLOBAL
    try:
        for op, arg, pos in pickletools.genops(data):
            if op.name in _STRING_OPS:
                recent = (recent + [str(arg)])[-2:]
                for needle in SUSPICIOUS_STRINGS:
                    if needle in str(arg):
                        findings.append(Finding("code-execution", f"{location}@{pos}", f"suspicious string {needle!r} in data"))
                        break
                continue
            if op.name == "GLOBAL":
                module, name = str(arg).split(" ", 1)
                _check_global(module, name, f"{location}@{pos}", findings)
            elif op.name == "STACK_GLOBAL":
                if len(recent) == 2:
                    _check_global(recent[0], recent[1], f"{location}@{pos}", findings)
            elif op.name == "STOP":
                break
            if op.name not in _STRING_OPS and op.name not in _MEMO_PUT_OPS:
                recent = []
    except Exception as exc:  # noqa: BLE001 - truncated or odd pickles are not our problem
        findings.append(Finding("unsupported", location, f"could not walk pickle opcodes: {exc}"))
    return findings


def _check_global(module: str, name: str, location: str, findings: list[Finding]) -> None:
    key = f"{module}.{name}"
    kind = PICKLE_BLOCKLIST.get(key)
    if kind:
        findings.append(Finding(kind, location, f"dangerous global {key}"))
