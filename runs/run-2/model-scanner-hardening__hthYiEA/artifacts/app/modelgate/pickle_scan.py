"""Non-executing pickle analysis.

The scanner must never run the artifact, so it cannot call ``pickle.load``.
Instead it walks the opcode stream with ``pickletools.genops`` and runs a
symbolic machine that mirrors CPython's unpickler stack: it tracks what each
stack entry *is* (a global reference, a module, a ``functools.partial``, a
constant, ...) without importing or calling anything.  A finding is raised at
the opcode that would actually invoke a capability-bearing callable -- REDUCE,
an argument-bearing INST/OBJ, or a retargeted partial reached through the memo,
``getattr``, ``attrgetter``, ``methodcaller`` or a nested stream -- which is
exactly the policy's test (see ``/app/docs/threat-model.md`` section 3).
"""
from __future__ import annotations

import pickletools

from modelgate import policy
from modelgate.report import Finding

# Tunable guards so a hostile or pathological artifact cannot exhaust time or
# stack.  Pickles in real models are tiny; these ceilings are far above them.
_MAX_OPS = 2_000_000
_MAX_DEPTH = 6

_MARK = object()


# Symbolic values are small sequences tagged by their first element:
#   ("global", module, name)      a GLOBAL / STACK_GLOBAL / INST class
#   ("module", name)              a module object (from __import__/import_module)
#   ("instance", module, name)    an object constructed from that class, so a
#                                 later method call can be resolved on it
#   ["partial", func, [args]]     a functools.partial -- a *list*, so a BUILD
#                                 retarget is seen through the memo and stack
#   ("attrgetter", [names])       operator.attrgetter(...)
#   ("methodcaller", name, [args])operator.methodcaller(...)
#   ("const", value)              a str/bytes/int/float/bool/None literal
#   ("tuple", [items])            a tuple
#   ("list", [items])             a list (contents tracked best-effort)
#   ("opaque",)                   anything whose identity does not matter
_OPAQUE = ("opaque",)


def _const_str(sym):
    if sym and sym[0] == "const" and isinstance(sym[1], str):
        return sym[1]
    return None


def _const_bytes(sym):
    if sym and sym[0] == "const" and isinstance(sym[1], (bytes, bytearray)):
        return bytes(sym[1])
    return None


# Pure data transforms an attacker may wrap embedded loader input in.  Each
# maps a constant to a constant; applying them unpacks bytes (like the archive
# layer) and never runs artifact code.
_TRANSFORMS = {
    "zlib.decompress", "gzip.decompress", "bz2.decompress", "lzma.decompress",
    "base64.b64decode", "base64.standard_b64decode", "base64.urlsafe_b64decode",
    "base64.b85decode", "base64.b32decode", "base64.b16decode", "base64.a85decode",
    "binascii.a2b_base64", "binascii.unhexlify", "binascii.a2b_hex",
    "codecs.decode",
}
_TRANSFORM_CAP = 256 * 1024 * 1024


def _apply_transform(full, args):
    raw = None
    if args:
        raw = _const_bytes(args[0])
        if raw is None:
            s = _const_str(args[0])
            if s is not None:
                raw = s.encode("latin1", "ignore")
    if raw is None or len(raw) > _TRANSFORM_CAP:
        return None
    try:
        mod, fn = full.split(".", 1)
        if full == "zlib.decompress":
            import zlib
            out = zlib.decompress(raw)
        elif full == "gzip.decompress":
            import gzip
            out = gzip.decompress(raw)
        elif full == "bz2.decompress":
            import bz2
            out = bz2.decompress(raw)
        elif full == "lzma.decompress":
            import lzma
            out = lzma.decompress(raw)
        elif mod == "base64":
            import base64
            out = getattr(base64, fn)(raw)
        elif mod == "binascii":
            import binascii
            out = getattr(binascii, fn)(raw)
        elif full == "codecs.decode":
            enc = _const_str(args[1]) if len(args) > 1 else "utf-8"
            import codecs
            out = codecs.decode(raw, enc or "utf-8")
            if not isinstance(out, (bytes, bytearray)):
                return None
        else:
            return None
    except Exception:
        return None
    if len(out) > _TRANSFORM_CAP:
        return None
    return bytes(out)


class _Machine:
    def __init__(self, data, location, stream, sink, depth, budget):
        self.data = data
        self.location = location
        self.stream = stream
        self.sink = sink            # list to append Findings to
        self.depth = depth
        self.budget = budget        # dict with an "ops" counter
        self.stack = []
        self.memo = {}

    # -- stack helpers -----------------------------------------------------
    def _pop(self):
        return self.stack.pop() if self.stack else _OPAQUE

    def _pop_mark(self):
        items = []
        while self.stack:
            top = self.stack.pop()
            if top is _MARK:
                break
            items.append(top)
        items.reverse()
        return items

    def _finding(self, kind, pos, detail):
        where = f"{self.location}#s{self.stream}@{pos}"
        self.sink.append(Finding(kind, where, detail))

    # -- run ---------------------------------------------------------------
    def run(self, gen, base):
        """Drive the opcode generator, rebasing offsets onto ``base``.  Returns
        the STOP offset (relative to base) or None.  Consumes lazily and stops
        at the op budget so a huge pickle is never fully materialised."""
        for op, arg, off in gen:
            self.budget["ops"] += 1
            if self.budget["ops"] > _MAX_OPS:
                return None
            name = op.name
            if name == "STOP":
                return off
            handler = _HANDLERS.get(name)
            if handler is not None:
                handler(self, arg, base + off)
            # opcodes without a handler have no effect we care about
        return None

    # -- the one interesting operation: a call --------------------------
    def invoke(self, func, args, pos):
        """Model calling ``func(*args)`` and return the symbolic result.

        Raises findings into the sink as a side effect.  ``args`` is a list of
        symbolic values.
        """
        tag = func[0] if func else "opaque"

        if tag == "partial":
            return self.invoke(func[1], list(func[2]) + list(args), pos)

        if tag == "attrgetter":
            # Retrieving an attribute is not itself a call; it yields the
            # attribute so a later call can reach it.
            if func[1] and args:
                return self._attr_of(args[0], func[1][0])
            return _OPAQUE

        if tag == "methodcaller":
            if args:
                target = self._attr_of(args[0], func[1])
                if target[0] == "global":
                    return self.invoke(target, func[2], pos)
            return _OPAQUE

        if tag != "global":
            return _OPAQUE

        module, name = policy.resolve(func[1], func[2])
        full = f"{module}.{name}"

        if full in _TRANSFORMS:
            # A pure data transform (decompress/decode): compute it on the
            # constant so a nested loader fed the *result* is seen through.
            # This runs no artifact code -- it only unpacks bytes, as the
            # container layer already does -- and is capped against bombs.
            out = _apply_transform(full, args)
            return ("const", out) if out is not None else _OPAQUE

        kind = policy.INDIRECTION.get(full)
        if kind is not None:
            return self._indirect(kind, args, pos)

        fmt = policy.NESTED_LOADERS.get(full)
        if fmt is not None:
            self._nested(fmt, full, args, pos)
            return _OPAQUE

        cap = policy.CAPABILITIES.get(full)
        if cap is not None:
            kind, trigger = cap
            if trigger == "call" or (trigger == "call_args" and len(args) >= 1):
                self._finding(kind, pos, f"call {full}")
            return _OPAQUE

        # Not a capability: calling a class yields an instance of it, so a
        # later method call (via getattr/methodcaller) can be resolved.
        return ("instance", module, name)

    def _attr_of(self, obj, attr):
        if not isinstance(attr, str) or not obj:
            return _OPAQUE
        if obj[0] == "module":
            return ("global", obj[1], attr)
        if obj[0] in ("global", "instance"):
            return ("global", obj[1], f"{obj[2]}.{attr}")
        return _OPAQUE

    def _indirect(self, kind, args, pos):
        if kind == "getattr":
            if len(args) >= 2:
                attr = _const_str(args[1])
                if attr is not None:
                    return self._attr_of(args[0], attr)
            return _OPAQUE
        if kind == "import":
            if args:
                mod = _const_str(args[0])
                if mod is not None:
                    return ("module", mod)
            return _OPAQUE
        if kind == "attrgetter":
            names = [s for s in (_const_str(a) for a in args) if s is not None]
            return ("attrgetter", names)
        if kind == "methodcaller":
            if args:
                mname = _const_str(args[0])
                if mname is not None:
                    return ("methodcaller", mname, list(args[1:]))
            return _OPAQUE
        if kind == "partial":
            if args:
                return ["partial", args[0], list(args[1:])]
            return _OPAQUE
        return _OPAQUE

    def _nested(self, fmt, full, args, pos):
        if not args:
            return
        data = _const_bytes(args[0])
        if data is not None:
            if self.depth >= _MAX_DEPTH:
                return
            where = f"{self.location}#s{self.stream}@{pos}->{full}"
            if fmt == "pickle":
                self.sink.extend(scan_pickle(data, where, self.depth + 1, self.budget))
            else:
                from modelgate import containers
                if fmt == "numpy":
                    self.sink.extend(containers.scan_npy_bytes(data, where, self.depth + 1, self.budget))
                elif fmt == "torch":
                    self.sink.extend(containers.scan_bytes(data, where, self.depth + 1, self.budget))
            return
        path = _const_str(args[0])
        if path is not None:
            # A loader handed a path reads that path: file access (C).
            self._finding(policy.FILE, pos, f"{full} reads path {path!r}")


# --------------------------------------------------------------------------
# Opcode handlers.  (STOP is handled inline by the run loop.)


def _push(value):
    def handler(m, arg, pos):
        m.stack.append(value() if callable(value) else value)
    return handler


def _push_const(m, arg, pos):
    m.stack.append(("const", arg))


def _op_mark(m, arg, pos):
    m.stack.append(_MARK)


def _op_pop(m, arg, pos):
    m._pop()


def _op_pop_mark(m, arg, pos):
    m._pop_mark()


def _op_dup(m, arg, pos):
    if m.stack:
        m.stack.append(m.stack[-1])


def _op_tuple(m, arg, pos):
    m.stack.append(("tuple", m._pop_mark()))


def _op_tuple1(m, arg, pos):
    a = m._pop()
    m.stack.append(("tuple", [a]))


def _op_tuple2(m, arg, pos):
    b = m._pop(); a = m._pop()
    m.stack.append(("tuple", [a, b]))


def _op_tuple3(m, arg, pos):
    c = m._pop(); b = m._pop(); a = m._pop()
    m.stack.append(("tuple", [a, b, c]))


def _op_list(m, arg, pos):
    m.stack.append(("list", m._pop_mark()))


def _op_dict(m, arg, pos):
    m._pop_mark()
    m.stack.append(("dict",))


def _op_set(m, arg, pos):
    m._pop_mark()
    m.stack.append(_OPAQUE)


def _op_append(m, arg, pos):
    v = m._pop()
    if m.stack and m.stack[-1] and m.stack[-1][0] == "list":
        m.stack[-1][1].append(v)


def _op_appends(m, arg, pos):
    vs = m._pop_mark()
    if m.stack and m.stack[-1] and m.stack[-1][0] == "list":
        m.stack[-1][1].extend(vs)


def _op_setitem(m, arg, pos):
    m._pop(); m._pop()


def _op_setitems(m, arg, pos):
    m._pop_mark()


def _op_additems(m, arg, pos):
    m._pop_mark()


def _op_build(m, arg, pos):
    state = m._pop()
    if not m.stack:
        return
    obj = m.stack[-1]
    # BUILD calls __setstate__; for a functools.partial that retargets it.
    # The partial is a mutable list, so the memo and stack (which alias the
    # same object, as in CPython) both see the new target.
    if isinstance(obj, list) and obj and obj[0] == "partial" and state and state[0] == "tuple" and state[1]:
        items = state[1]
        obj[1] = items[0]
        obj[2] = list(items[1][1]) if len(items) > 1 and items[1] and items[1][0] == "tuple" else []


def _op_reduce(m, arg, pos):
    args = m._pop()
    func = m._pop()
    arglist = args[1] if args and args[0] == "tuple" else []
    m.stack.append(m.invoke(func, arglist, pos))


def _instance_of(cls):
    if cls and cls[0] == "global":
        return ("instance", cls[1], cls[2])
    return _OPAQUE


def _op_inst(m, arg, pos):
    args = m._pop_mark()
    module, _, name = str(arg).partition(" ")
    cls = ("global", module, name)
    if args:  # argument-bearing INST constructs fully (runs __init__)
        m.invoke(cls, args, pos)
    m.stack.append(_instance_of(cls))


def _op_obj(m, arg, pos):
    items = m._pop_mark()
    cls = _OPAQUE
    if items:
        cls = items[0]
        rest = items[1:]
        if rest:
            m.invoke(cls, rest, pos)
    m.stack.append(_instance_of(cls))


def _op_newobj(m, arg, pos):
    m._pop()                    # args
    cls = m._pop()              # cls -> __new__ only, no capability executed
    m.stack.append(_instance_of(cls))


def _op_newobj_ex(m, arg, pos):
    m._pop(); m._pop()          # kwargs, args
    cls = m._pop()
    m.stack.append(_instance_of(cls))


def _op_global(m, arg, pos):
    module, _, name = str(arg).partition(" ")
    m.stack.append(("global", module, name))


def _op_stack_global(m, arg, pos):
    name = m._pop()
    module = m._pop()
    ms = _const_str(module)
    ns = _const_str(name)
    if ms is not None and ns is not None:
        m.stack.append(("global", ms, ns))
    else:
        m.stack.append(_OPAQUE)


def _op_get(m, arg, pos):
    m.stack.append(m.memo.get(arg, _OPAQUE))


def _op_put(m, arg, pos):
    if m.stack:
        m.memo[arg] = m.stack[-1]


def _op_memoize(m, arg, pos):
    if m.stack:
        m.memo[len(m.memo)] = m.stack[-1]


def _op_persid(m, arg, pos):
    m.stack.append(_OPAQUE)     # resolved by the loader's storage, executes nothing


def _op_binpersid(m, arg, pos):
    m._pop()
    m.stack.append(_OPAQUE)


def _op_ext(m, arg, pos):
    m.stack.append(_OPAQUE)     # copyreg extension registry; not resolvable statically


def _op_next_buffer(m, arg, pos):
    m.stack.append(_OPAQUE)


def _op_readonly_buffer(m, arg, pos):
    pass


_HANDLERS = {
    "MARK": _op_mark,
    "POP": _op_pop,
    "POP_MARK": _op_pop_mark,
    "DUP": _op_dup,
    "NONE": _push(lambda: ("const", None)),
    "NEWTRUE": _push(lambda: ("const", True)),
    "NEWFALSE": _push(lambda: ("const", False)),
    "INT": _push_const, "LONG": _push_const,
    "BININT": _push_const, "BININT1": _push_const, "BININT2": _push_const,
    "LONG1": _push_const, "LONG4": _push_const,
    "FLOAT": _push_const, "BINFLOAT": _push_const,
    "STRING": _push_const, "BINSTRING": _push_const, "SHORT_BINSTRING": _push_const,
    "BINBYTES": _push_const, "SHORT_BINBYTES": _push_const, "BINBYTES8": _push_const,
    "BYTEARRAY8": _push_const,
    "UNICODE": _push_const, "SHORT_BINUNICODE": _push_const,
    "BINUNICODE": _push_const, "BINUNICODE8": _push_const,
    "EMPTY_LIST": _push(lambda: ("list", [])),
    "EMPTY_TUPLE": _push(lambda: ("tuple", [])),
    "EMPTY_DICT": _push(lambda: ("dict",)),
    "EMPTY_SET": _push(lambda: _OPAQUE),
    "TUPLE": _op_tuple, "TUPLE1": _op_tuple1, "TUPLE2": _op_tuple2, "TUPLE3": _op_tuple3,
    "LIST": _op_list,
    "DICT": _op_dict,
    "FROZENSET": _op_set,
    "APPEND": _op_append, "APPENDS": _op_appends,
    "SETITEM": _op_setitem, "SETITEMS": _op_setitems,
    "ADDITEMS": _op_additems,
    "BUILD": _op_build,
    "REDUCE": _op_reduce,
    "INST": _op_inst, "OBJ": _op_obj,
    "NEWOBJ": _op_newobj, "NEWOBJ_EX": _op_newobj_ex,
    "GLOBAL": _op_global, "STACK_GLOBAL": _op_stack_global,
    "GET": _op_get, "BINGET": _op_get, "LONG_BINGET": _op_get,
    "PUT": _op_put, "BINPUT": _op_put, "LONG_BINPUT": _op_put,
    "MEMOIZE": _op_memoize,
    "PERSID": _op_persid, "BINPERSID": _op_binpersid,
    "EXT1": _op_ext, "EXT2": _op_ext, "EXT4": _op_ext,
    "NEXT_BUFFER": _op_next_buffer, "READONLY_BUFFER": _op_readonly_buffer,
}


# --------------------------------------------------------------------------
def scan_pickle(data, location, depth=0, budget=None):
    """Scan one file that is a sequence of one or more concatenated pickles.

    Every stream is analysed (a payload may hide in the second stream, as the
    policy notes); trailing non-pickle bytes -- the raw storages of a legacy
    torch checkpoint -- end the walk once at least one stream has parsed.
    """
    if budget is None:
        budget = {"ops": 0}
    findings: list[Finding] = []
    pos = 0
    stream = 0
    n = len(data)
    parsed_any = False
    while pos < n and budget["ops"] <= _MAX_OPS:
        stream += 1
        machine = _Machine(data, location, stream, findings, depth, budget)
        try:
            stop_off = machine.run(pickletools.genops(data[pos:]), pos)
        except Exception:
            # A stream that will not parse: trailing raw bytes after an earlier
            # pickle (a legacy torch checkpoint's storages) end the walk; a
            # first stream that will not parse at all is fail-closed.
            if parsed_any or findings:
                break
            return [Finding("unsupported", location, "could not parse pickle opcodes")]
        parsed_any = True
        if stop_off is None:
            break                   # no STOP: truncated tail or op budget hit
        pos += stop_off + 1
    if not parsed_any:
        return [Finding("unsupported", location, "empty pickle")]
    return findings


def looks_like_pickle(data):
    """Best-effort: does a complete pickle stream parse at offset 0?"""
    if not data:
        return False
    try:
        for op, arg, off in pickletools.genops(data):
            if op.name == "STOP":
                return True
            if off > 4_000_000:
                return False
    except Exception:
        return False
    return False
