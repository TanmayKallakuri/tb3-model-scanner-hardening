"""Shared fixtures and raw pickle opcode helpers for the oracle scanner tests.

Malicious inputs are built only as opcode bytes; nothing dangerous is ever
pickled, unpickled or called.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SOLUTION = os.path.join(ROOT, "tasks", "model-scanner-hardening", "solution")
REFS = os.path.join(ROOT, "tmp", "refs")
if SOLUTION not in sys.path:
    sys.path.insert(0, SOLUTION)

import pytest  # noqa: E402

# --- protocol 4 opcode helpers -----------------------------------------------
PROTO2 = b"\x80\x02"
PROTO3 = b"\x80\x03"
PROTO4 = b"\x80\x04"
PROTO5 = b"\x80\x05"
STOP = b"."
MARK = b"("
TUPLE = b"t"
TUPLE1 = b"\x85"
TUPLE2 = b"\x86"
TUPLE3 = b"\x87"
EMPTY_TUPLE = b")"
EMPTY_DICT = b"}"
EMPTY_LIST = b"]"
APPENDS = b"e"
SETITEM = b"s"
REDUCE = b"R"
NEWOBJ = b"\x81"
NEWOBJ_EX = b"\x92"
BUILD = b"b"
MEMOIZE = b"\x94"
POP = b"0"
NONE = b"N"
BINPERSID = b"Q"


def su(s):
    b = s.encode("utf-8")
    assert len(b) < 256
    return b"\x8c" + bytes([len(b)]) + b


def sb(b):
    assert len(b) < 256
    return b"C" + bytes([len(b)]) + b


def bb(b):
    return b"B" + len(b).to_bytes(4, "little") + b


def sg(mod, name):
    """STACK_GLOBAL mod.name"""
    return su(mod) + su(name) + b"\x93"


def g(mod, name):
    """text GLOBAL mod.name"""
    return b"c" + mod.encode() + b"\n" + name.encode() + b"\n"


def i1(n):
    return b"K" + bytes([n])


def binget(i):
    return b"h" + bytes([i])


def long_binget(i):
    return b"j" + i.to_bytes(4, "little")


def binput(i):
    return b"q" + bytes([i])


def args(*items):
    if not items:
        return EMPTY_TUPLE
    if len(items) == 1:
        return items[0] + TUPLE1
    if len(items) == 2:
        return b"".join(items) + TUPLE2
    if len(items) == 3:
        return b"".join(items) + TUPLE3
    return MARK + b"".join(items) + TUPLE


def call(mod, name, *items):
    """REDUCE of mod.name(*items), items already encoded."""
    return sg(mod, name) + args(*items) + REDUCE


def p4(body):
    return PROTO4 + body + STOP


OS_SYSTEM_P4 = p4(call("os", "system", su("id")))
BENIGN_P4 = p4(EMPTY_DICT + su("k") + i1(1) + SETITEM)


def kinds(findings):
    return sorted({f.kind for f in findings})


@pytest.fixture(scope="session")
def refs():
    if not os.path.isdir(REFS):
        pytest.skip("reference files not present")
    return REFS
