"""Pickle scanning by abstract interpretation of the opcode stream.

The scanner walks every pickle in the file (``pickle.load`` called repeatedly
until end of file), maintaining a symbolic stack and memo.  A finding is
emitted only where a capability-bearing callable is actually *executed*:

* ``REDUCE``/``INST``/``OBJ`` (with arguments) call the callable;
* ``BUILD`` runs ``__setstate__`` (used to retarget a ``functools.partial``);
* a ``functools.partial`` fired by a later call runs its target;
* a capability reached through the memo, a ``getattr`` chain,
  ``operator.attrgetter``/``methodcaller``, ``functools.partial``,
  ``__import__``/``import_module`` or an alias is the call that reaches it.

A global merely stored as data (in a container, as an attribute, memoised and
never called) is not a finding, whatever its name.
"""
from __future__ import annotations

import io
import pickletools

from modelgate import policy
from modelgate.report import Finding

# ----- symbolic values ------------------------------------------------------

class Global:
    __slots__ = ("module", "name")
    def __init__(self, module, name):
        self.module, self.name = policy.resolve_alias(str(module), str(name))
    @property
    def full(self):
        return f"{self.module}.{self.name}"

class ModuleRef:
    __slots__ = ("name",)
    def __init__(self, name): self.name = str(name)

class Partial:
    __slots__ = ("func", "args")
    def __init__(self, func, args): self.func, self.args = func, args

class Attrgetter:
    __slots__ = ("attr",)
    def __init__(self, attr): self.attr = attr

class Methodcaller:
    __slots__ = ("method", "args")
    def __init__(self, method, args): self.method, self.args = method, args

class Instance:
    """Result of constructing a tracked class (e.g. a socket)."""
    __slots__ = ("tag",)
    def __init__(self, tag): self.tag = tag

class Bytes:
    __slots__ = ("data",)
    def __init__(self, data): self.data = data

class Opaque:
    __slots__ = ()
OPAQUE = Opaque()

# opcode groups
_STR_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
            "STRING", "SHORT_BINSTRING", "BINSTRING"}
_BYTES_OPS = {"SHORT_BINBYTES", "BINBYTES", "BINBYTES8", "BYTEARRAY8"}
_INT_OPS = {"BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "INT"}

MAX_DEPTH = 6
MAX_OPCODES = 20_000_000  # per-artifact ceiling, guards against opcode bombs


class _Machine:
    def __init__(self, location, depth):
        self.location = location
        self.depth = depth
        self.stack = []
        self.marks = []
        self.memo = {}
        self.findings: list[Finding] = []
        self._seen = set()

    # stack helpers (guarded) -------------------------------------------------
    def push(self, v): self.stack.append(v)

    def pop(self):
        return self.stack.pop() if self.stack else OPAQUE

    def pop_to_mark(self):
        if not self.marks:
            items = self.stack[:]; self.stack = []; return items
        i = self.marks.pop()
        items = self.stack[i:]; del self.stack[i:]; return items

    def report(self, kind, pos, detail):
        key = (kind, detail)
        if key in self._seen:
            return
        self._seen.add(key)
        self.findings.append(Finding(kind, f"{self.location}@{pos}", detail))

    # ------------------------------------------------------------------------
    def step(self, name, arg, pos):
        st = self.stack
        if name in _STR_OPS:
            self.push(str(arg))
        elif name in _BYTES_OPS:
            self.push(Bytes(bytes(arg) if not isinstance(arg, (bytes, bytearray)) else bytes(arg)))
        elif name in _INT_OPS:
            self.push(arg)
        elif name in ("BINFLOAT", "FLOAT"):
            self.push(arg)
        elif name == "NONE":
            self.push(None)
        elif name == "NEWTRUE":
            self.push(True)
        elif name == "NEWFALSE":
            self.push(False)
        elif name in ("EMPTY_DICT", "DICT"):
            if name == "DICT": self.pop_to_mark()
            self.push({})
        elif name in ("EMPTY_LIST", "LIST"):
            if name == "LIST": self.pop_to_mark()
            self.push([])
        elif name in ("EMPTY_SET", "FROZENSET"):
            if name == "FROZENSET": self.pop_to_mark()
            self.push(OPAQUE)
        elif name == "EMPTY_TUPLE":
            self.push(())
        elif name == "MARK":
            self.marks.append(len(st))
        elif name == "TUPLE":
            self.push(tuple(self.pop_to_mark()))
        elif name == "TUPLE1":
            a = self.pop(); self.push((a,))
        elif name == "TUPLE2":
            b = self.pop(); a = self.pop(); self.push((a, b))
        elif name == "TUPLE3":
            c = self.pop(); b = self.pop(); a = self.pop(); self.push((a, b, c))
        elif name == "GLOBAL":
            m, n = str(arg).split(" ", 1)
            self.push(Global(m, n))
        elif name == "STACK_GLOBAL":
            n = self.pop(); m = self.pop()
            self.push(Global(_s(m), _s(n)))
        elif name in ("GET", "BINGET", "LONG_BINGET"):
            self.push(self.memo.get(_key(arg), OPAQUE))
        elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
            if st: self.memo[_key(arg)] = st[-1]
        elif name == "MEMOIZE":
            if st: self.memo[len(self.memo)] = st[-1]
        elif name == "POP":
            self.pop()
        elif name == "POP_MARK":
            self.pop_to_mark()
        elif name == "DUP":
            if st: self.push(st[-1])
        elif name in ("APPEND",):
            self.pop()
        elif name in ("APPENDS", "ADDITEMS"):
            self.pop_to_mark()
        elif name == "SETITEM":
            self.pop(); self.pop()
        elif name == "SETITEMS":
            self.pop_to_mark()
        elif name in ("PERSID",):
            self.push(OPAQUE)
        elif name == "BINPERSID":
            self.pop(); self.push(OPAQUE)
        elif name in ("EXT1", "EXT2", "EXT4"):
            self.push(OPAQUE)
        elif name == "NEXT_BUFFER":
            self.push(OPAQUE)
        elif name == "READONLY_BUFFER":
            pass  # wraps the top-of-stack buffer in place
        elif name == "REDUCE":
            a = self.pop(); f = self.pop()
            self.push(self.call(f, _astuple(a), pos))
        elif name in ("INST", "OBJ"):
            items = self.pop_to_mark()
            if name == "INST":
                m, n = str(arg).split(" ", 1)
                func = Global(m, n); args = tuple(items)
            else:
                func = items[0] if items else OPAQUE
                args = tuple(items[1:])
            if args:  # argless INST/OBJ call __new__ only -> no capability
                self.push(self.call(func, args, pos))
            else:
                self.push(OPAQUE)
        elif name == "NEWOBJ":
            self.pop(); cls = self.pop()
            self.push(self._instance_of(cls))
        elif name == "NEWOBJ_EX":
            self.pop(); self.pop(); cls = self.pop()
            self.push(self._instance_of(cls))
        elif name == "BUILD":
            state = self.pop()
            obj = st[-1] if st else OPAQUE
            self.build(obj, state, pos)
        elif name == "STOP":
            self.pop()
            raise _Stop
        elif name in ("PROTO", "FRAME"):
            pass
        else:
            # unknown opcode: leave the stack as-is
            pass

    # ------------------------------------------------------------------------
    def _instance_of(self, cls):
        if isinstance(cls, Global):
            tag = policy.TYPED_INSTANCES.get(cls.full)
            if tag:
                return Instance(tag)
        return OPAQUE

    def build(self, obj, state, pos):
        # functools.partial retargets itself through __setstate__
        if isinstance(obj, Partial):
            s = _astuple(state)
            if s:
                obj.func = s[0]
                if len(s) > 1:
                    obj.args = _astuple(s[1])

    # the heart: a call reaches and executes something ------------------------
    def call(self, func, args, pos):
        if isinstance(func, _BoundDanger):
            self.report(func.kind, pos, f"call {func.label}")
            return OPAQUE
        if isinstance(func, Partial):
            return self.call(func.func, tuple(_astuple(func.args)) + tuple(args), pos)
        if isinstance(func, Attrgetter):
            base = args[0] if args else OPAQUE
            return self.resolve_attr(base, func.attr)
        if isinstance(func, Methodcaller):
            base = args[0] if args else OPAQUE
            return self.invoke_method(base, func.method, func.args, pos)
        if not isinstance(func, Global):
            return OPAQUE
        full = func.full

        # buffer wrappers pass embedded bytes through to a nested loader
        if full in ("io.BytesIO", "_io.BytesIO", "builtins.bytearray", "builtins.bytes"):
            if args and isinstance(args[0], Bytes):
                return args[0]
            return OPAQUE

        # reach mechanisms -----------------------------------------------------
        tag = policy.REACH.get(full)
        if tag == "getattr":
            base = args[0] if args else OPAQUE
            attr = args[1] if len(args) > 1 else ""
            return self.resolve_attr(base, _s(attr))
        if tag == "import":
            mod = _s(args[0]) if args else ""
            return ModuleRef(mod)
        if tag == "partial":
            return Partial(args[0] if args else OPAQUE, tuple(args[1:]))
        if tag == "attrgetter":
            return Attrgetter(_s(args[0]) if args else "")
        if tag == "methodcaller":
            return Methodcaller(_s(args[0]) if args else "", tuple(args[1:]))
        if tag in ("reduce", "apply"):
            # both ultimately call args[0]
            if args:
                self.call(args[0], (OPAQUE, OPAQUE), pos)
            return OPAQUE

        # string building used to compute a global/attribute name
        if full in ("operator.add", "operator.concat", "operator.iadd",
                    "builtins.str.__add__"):
            if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str):
                return args[0] + args[1]
            return OPAQUE
        if full in ("str.join", "builtins.str.join"):
            return OPAQUE

        # nested loaders -------------------------------------------------------
        if full in policy.NESTED_LOADERS:
            self.nested(full, args, pos)
            return OPAQUE

        # host-sensitive client constructors ----------------------------------
        if full in policy.HOST_CONSTRUCTORS and args and _is_host(args[0]):
            self.report(policy.HOST_CONSTRUCTORS[full], pos,
                        f"{full}(host) opens a connection")
            return self._instance_of(func)

        # direct capability ----------------------------------------------------
        kind = policy.CAPABILITIES.get(full)
        if kind:
            self.report(kind, pos, f"call {full}")
            return OPAQUE

        return self._instance_of(func)

    def resolve_attr(self, base, attr):
        attr = _s(attr)
        if isinstance(base, ModuleRef):
            return Global(base.name, attr)
        if isinstance(base, Global):
            # getattr on a module/class global extends the dotted path
            return Global(base.module, f"{base.name}.{attr}")
        if isinstance(base, Instance):
            methods = policy.INSTANCE_METHODS.get(base.tag, {})
            if attr in methods:
                return _BoundDanger(methods[attr], f"{base.tag}.{attr}")
        return OPAQUE

    def invoke_method(self, base, method, margs, pos):
        method = _s(method)
        if isinstance(base, (ModuleRef, Global)):
            target = self.resolve_attr(base, method)
            return self.call(target, tuple(margs), pos)
        if isinstance(base, Instance):
            methods = policy.INSTANCE_METHODS.get(base.tag, {})
            if method in methods:
                self.report(methods[method], pos, f"{base.tag}.{method}() connects")
        return OPAQUE

    def nested(self, full, args, pos):
        data = args[0] if args else OPAQUE
        if isinstance(data, Bytes):
            if self.depth < MAX_DEPTH:
                from modelgate import dispatch
                loc = f"{self.location}:{full}()"
                self.findings.extend(
                    dispatch.scan_embedded(data.data, loc, self.depth + 1))
            return
        if isinstance(data, str):
            # a loader handed a path reads that path
            self.report(policy.FILE, pos, f"{full}() reads path {data!r:.60}")
            return
        # non-literal argument to a re-entrant loader: treat as code execution
        self.report(policy.CODE, pos, f"call {full} (re-entrant deserialization)")


class _BoundDanger:
    """A resolved bound method that is itself a capability when called."""
    __slots__ = ("kind", "label")
    def __init__(self, kind, label): self.kind, self.label = kind, label


class _Stop(Exception):
    pass


class _OpcodeBudget(BaseException):
    pass


def _s(v):
    return v if isinstance(v, str) else ""

def _key(arg):
    try:
        return int(arg)
    except (TypeError, ValueError):
        return arg

def _astuple(v):
    if isinstance(v, tuple):
        return v
    if isinstance(v, list):
        return tuple(v)
    return (v,)

def _is_host(v):
    return isinstance(v, str) and len(v) > 0


def scan_pickle(data: bytes, location: str, depth: int = 0) -> list[Finding]:
    """Scan every pickle stream in *data* until end of file."""
    findings: list[Finding] = []
    pos = 0
    n = len(data)
    streams = 0
    budget = [MAX_OPCODES]
    while pos < n:
        sub = data[pos:]
        m = _Machine(location if streams == 0 else f"{location}#{streams}", depth)
        try:
            advanced = _run_stream(m, sub, budget)
        except _OpcodeBudget:
            findings.extend(m.findings)
            raise ValueError("opcode budget exceeded")
        findings.extend(m.findings)
        if advanced is None:
            # not a pickle at this offset
            if streams == 0:
                raise ValueError("not a pickle stream")
            break  # trailing non-pickle bytes (e.g. legacy torch storages)
        if advanced <= 0:
            break
        pos += advanced
        streams += 1
        if streams > 64:
            break
    if streams == 0:
        raise ValueError("not a pickle stream")
    return findings


def _run_stream(machine, data, budget):
    """Run one pickle; return bytes consumed, or None if not a pickle."""
    consumed = None
    try:
        for op, arg, pos in pickletools.genops(io.BytesIO(data)):
            budget[0] -= 1
            if budget[0] <= 0:
                raise _OpcodeBudget
            try:
                machine.step(op.name, arg, pos)
            except _Stop:
                return pos + 1
            except Exception:
                pass
        # genops ended without STOP
        return None if consumed is None else consumed
    except Exception:
        return None
