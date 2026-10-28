"""Pickle scanning by abstract interpretation of the pickle virtual machine.

Rather than matching opcode names or data strings, this walks the opcodes with
``pickletools.genops`` (which never executes anything) while maintaining an
abstract stack and memo that mirror what CPython's unpickler would build.  A
finding is raised only when a capability-bearing callable would actually be
*called* -- through REDUCE, INST/OBJ, a retargeted partial, getattr/attrgetter
chains, operator.call, functools.reduce, a consumed map/filter, or a nested
loader handed embedded data -- so a global that is merely referenced, stored or
imported is never a finding, whatever its name (threat model section 3).
"""
from __future__ import annotations

import base64
import binascii
import bz2
import codecs
import gzip
import lzma
import operator as _operator
import pickletools
import zlib

from modelgate import policy
from modelgate.report import Finding

# Pure data transforms the policy names as ways embedded payloads are assembled
# before a nested loader sees them (base64, hex, zlib, concatenation, bytearray
# literals).  Computing them on concrete constants is side-effect free, so the
# scanner may evaluate them to recover the payload it must then scan.
_PURE = {
    "base64.b64decode": base64.b64decode,
    "base64.standard_b64decode": base64.standard_b64decode,
    "base64.urlsafe_b64decode": base64.urlsafe_b64decode,
    "base64.b16decode": base64.b16decode,
    "base64.b32decode": base64.b32decode,
    "base64.a85decode": base64.a85decode,
    "base64.b85decode": base64.b85decode,
    "binascii.a2b_base64": binascii.a2b_base64,
    "binascii.unhexlify": binascii.unhexlify,
    "binascii.a2b_hex": binascii.a2b_hex,
    "codecs.decode": codecs.decode,
    "zlib.decompress": lambda d, *a: zlib.decompressobj().decompress(
        bytes(d), 64 * 1024 * 1024),
    "bz2.decompress": lambda d: bz2.BZ2Decompressor().decompress(
        bytes(d), 64 * 1024 * 1024),
    "lzma.decompress": lambda d, *a: lzma.LZMADecompressor().decompress(
        bytes(d), 64 * 1024 * 1024),
    "gzip.decompress": gzip.decompress,
    "builtins.bytes.fromhex": bytes.fromhex,
    "builtins.bytes": bytes,
    "builtins.bytearray": bytearray,
    "operator.add": _operator.add,
    "operator.concat": _operator.concat,
    "builtins.bytes.join": lambda sep, it: sep.join(it),
    "builtins.str.join": lambda sep, it: sep.join(it),
    "builtins.bytearray.fromhex": bytearray.fromhex,
}
_PURE_MAX_IN = 8 * 1024 * 1024

# Top-level modules the resolver may import to canonicalise an attribute chain.
# All are standard library (plus numpy, which is installed) and safe to import:
# importing them runs no artefact code.  A module named by the artefact that is
# not on this list is never imported; it falls back to static name tables.
SAFE_IMPORT = {
    "os", "posix", "nt", "sys", "io", "_io", "builtins", "subprocess", "shutil",
    "socket", "operator", "functools", "_functools", "itertools", "collections",
    "importlib", "pkgutil", "pickle", "_pickle", "marshal", "types", "code",
    "codeop", "pty", "pydoc", "runpy", "timeit", "bdb", "pdb", "logging",
    "urllib", "http", "ftplib", "smtplib", "poplib", "imaplib", "nntplib",
    "telnetlib", "ssl", "webbrowser", "pathlib", "copy", "copyreg", "base64",
    "zlib", "bz2", "lzma", "gzip", "codecs", "_codecs", "numpy", "fileinput",
    "dbm", "shelve", "tarfile", "zipfile", "cProfile", "profile",
}

_MAX_OPS = 2_000_000       # opcode budget per stream (bomb guard)
_MAX_NEST = 6              # nested-loader recursion guard
_MAX_FINDINGS = 64

_introspect_cache: dict = {}


# --------------------------------------------------------------------------
# Abstract values
# --------------------------------------------------------------------------
class AV:
    pass


class Const(AV):
    __slots__ = ("val",)

    def __init__(self, val):
        self.val = val


class TupleV(AV):
    __slots__ = ("items",)

    def __init__(self, items):
        self.items = list(items)


class ListV(AV):
    __slots__ = ("items",)

    def __init__(self, items=None):
        self.items = list(items or [])


class DictV(AV):
    __slots__ = ("pairs",)

    def __init__(self):
        self.pairs = []          # list of (key_av, val_av)


class G(AV):
    __slots__ = ("canonical", "is_module")

    def __init__(self, canonical, is_module=False):
        self.canonical = canonical
        self.is_module = is_module


class ModuleV(AV):
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name


class Partial(AV):
    __slots__ = ("func", "args")

    def __init__(self, func, args):
        self.func = func
        self.args = list(args)


class AttrGetter(AV):
    __slots__ = ("names",)

    def __init__(self, names):
        self.names = names


class MethodCaller(AV):
    __slots__ = ("name", "args")

    def __init__(self, name, args):
        self.name = name
        self.args = list(args)


class Lazy(AV):
    __slots__ = ("func",)

    def __init__(self, func):
        self.func = func


class Obj(AV):
    __slots__ = ("cls",)

    def __init__(self, cls=None):
        self.cls = cls


class Unknown(AV):
    pass


_UNKNOWN = Unknown()
_MARK = object()


# --------------------------------------------------------------------------
# Global resolution
# --------------------------------------------------------------------------
def _introspect(module: str, name: str):
    """Resolve ``module`` + (possibly dotted) ``name`` via a real import of an
    allow-listed safe module.  Returns a canonical string, ``("module", name)``
    for a module object, or ``None`` if resolution is unavailable."""
    top = (module or "").split(".", 1)[0]
    if top not in SAFE_IMPORT:
        return None
    key = (module, name)
    if key in _introspect_cache:
        return _introspect_cache[key]
    result = None
    try:
        import importlib
        obj = importlib.import_module(module)
        for part in name.split(".") if name else []:
            obj = getattr(obj, part)
        import types as _types
        if isinstance(obj, _types.ModuleType):
            result = ("module", obj.__name__)
        else:
            mod = getattr(obj, "__module__", None)
            qual = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
            if mod and qual:
                result = policy.normalize(mod, qual)
            elif qual:
                result = policy.normalize(module, name)
    except Exception:
        result = None
    _introspect_cache[key] = result
    return result


def resolve_global(module: str, name: str) -> AV:
    res = _introspect(module, name)
    if isinstance(res, tuple) and res[0] == "module":
        return ModuleV(res[1])
    if isinstance(res, str):
        return G(res, is_module=False)
    canonical = policy.normalize(module, name)
    # a bare module reference (no name) becomes a module value
    if "." not in canonical and not name:
        return ModuleV(canonical)
    return G(canonical, is_module=False)


def g_from_canonical(canonical: str) -> AV:
    return G(canonical, is_module=False)


# --------------------------------------------------------------------------
# Concrete-value extraction
# --------------------------------------------------------------------------
_NOPE = object()


def to_py(av, depth=0):
    if depth > 8:
        return _NOPE
    if isinstance(av, Const):
        return av.val
    if isinstance(av, TupleV):
        if len(av.items) > 4096:
            return _NOPE
        out = []
        for it in av.items:
            v = to_py(it, depth + 1)
            if v is _NOPE:
                return _NOPE
            out.append(v)
        return tuple(out)
    if isinstance(av, ListV):
        if len(av.items) > 4096:
            return _NOPE
        out = []
        for it in av.items:
            v = to_py(it, depth + 1)
            if v is _NOPE:
                return _NOPE
            out.append(v)
        return out
    return _NOPE


def _as_args(argtuple):
    """Return a python list of argument AVs, or None if unknown."""
    if isinstance(argtuple, TupleV):
        return list(argtuple.items)
    if isinstance(argtuple, Const) and isinstance(argtuple.val, tuple):
        return [Const(x) for x in argtuple.val]
    if isinstance(argtuple, ListV):
        return list(argtuple.items)
    return None


def _truthy_host(av) -> bool:
    v = to_py(av)
    if v is _NOPE:
        return True            # unknown host -> assume a connection (fail closed)
    if isinstance(v, (tuple, list)):
        return len(v) > 0 and bool(v[0])
    return bool(v)


# --------------------------------------------------------------------------
# Interpreter
# --------------------------------------------------------------------------
class _Scan:
    def __init__(self, location, findings, depth):
        self.loc = location
        self.findings = findings
        self.depth = depth
        self.seen = set()
        self.module_overrides = {}     # modname -> {attr: AV}

    def resolve_op(self, module, name):
        """Resolve a GLOBAL/STACK_GLOBAL, honouring any attribute a prior BUILD
        rebound on that module object (threat model: BUILD on a module rebinds
        its attributes for every later GLOBAL)."""
        if self.module_overrides:
            for mod in (module, policy.MODULE_ALIASES.get(module, module)):
                ov = self.module_overrides.get(mod)
                if not ov:
                    continue
                top = name.split(".", 1)[0]
                if top in ov:
                    base = ov[top]
                    rest = name[len(top):].lstrip(".")
                    if not rest:
                        return base
                    v = base
                    for part in rest.split("."):
                        v = self.getattr_av(v, Const(part))
                    return v
        return resolve_global(module, name)

    def emit(self, kind, where, detail):
        key = (kind, detail)
        if key in self.seen or len(self.findings) >= _MAX_FINDINGS:
            return
        self.seen.add(key)
        self.findings.append(Finding(kind, where, detail))

    # -- calling ----------------------------------------------------------
    def call(self, func, argtuple, where, op="REDUCE"):
        args = _as_args(argtuple)
        return self._call(func, args, where, op)

    def _call(self, func, args, where, op):
        if isinstance(func, G):
            return self._call_global(func.canonical, args, where, op)
        if isinstance(func, Partial):
            merged = list(func.args) + (args or [])
            return self._call(func.func, merged if args is not None else None, where, op)
        if isinstance(func, AttrGetter):
            if args:
                v = args[0]
                for nm in func.names:
                    v = self.getattr_av(v, Const(nm))
                return v
            return _UNKNOWN
        if isinstance(func, MethodCaller):
            if args:
                target = self.getattr_av(args[0], Const(func.name))
                return self._call(target, list(func.args), where, op)
            return _UNKNOWN
        if isinstance(func, Lazy):
            return func
        return _UNKNOWN

    def _call_global(self, c, args, where, op):
        kind = policy.KIND_OF.get(c)
        if c in policy.NESTED_LOADERS:
            self._nested(c, args, where)
            return _UNKNOWN
        if kind is not None:
            self.emit(kind, where, f"{op} {c}")
            return _UNKNOWN
        if c in policy.HOST_SENSITIVE:
            if args is None or (args and _truthy_host(args[0])):
                self.emit("network", where, f"{op} {c}(host)")
            return Obj(cls=g_from_canonical(c))
        if c in policy.GETATTR:
            if args and len(args) >= 2:
                return self.getattr_av(args[0], args[1])
            return _UNKNOWN
        if c in policy.PARTIAL:
            if args:
                return Partial(args[0], args[1:])
            return _UNKNOWN
        if c in policy.IMPORTERS:
            if args:
                nm = to_py(args[0])
                if isinstance(nm, str):
                    return ModuleV(nm)
            return _UNKNOWN
        if c in policy.RESOLVE_NAME:
            if args:
                nm = to_py(args[0])
                if isinstance(nm, str):
                    return self._resolve_dotted(nm)
            return _UNKNOWN
        if c in policy.ATTRGETTER:
            names = []
            for a in (args or []):
                v = to_py(a)
                if isinstance(v, str):
                    names.append(v)
            return AttrGetter(names)
        if c in policy.METHODCALLER:
            if args:
                nm = to_py(args[0])
                if isinstance(nm, str):
                    return MethodCaller(nm, args[1:])
            return _UNKNOWN
        if c in policy.CALLERS:
            if args:
                return self._call(args[0], args[1:], where, "operator.call")
            return _UNKNOWN
        if c in ("operator.getitem", "operator.__getitem__"):
            # sys.modules['os'] / a modules mapping lookup -> the module
            if args and isinstance(args[0], G) and args[0].canonical == "sys.modules":
                k = to_py(args[1]) if len(args) >= 2 else None
                if isinstance(k, str):
                    return ModuleV(k)
            return _UNKNOWN
        if c in policy.REDUCER:
            if args:
                self._invoke(args[0], where, "functools.reduce")
            return _UNKNOWN
        if c in policy.MAPPERS:
            if args:
                return Lazy(args[0])
            return _UNKNOWN
        if c in policy.CONSUMERS or c in policy.KEYED:
            for a in (args or []):
                if isinstance(a, Lazy):
                    self._invoke(a.func, where, "consumed-iterator")
            return _UNKNOWN
        if c in policy.THREAD_STARTERS:
            # positional target is the 2nd argument of Thread/Process
            if args and len(args) >= 2:
                self._invoke(args[1], where, "thread-target")
            return _UNKNOWN
        if c in policy.WRAPPERS:
            # copy.copy / lru_cache / ... hand the callable back; keep the chain
            return args[0] if args else _UNKNOWN
        if c in ("io.BytesIO", "io.StringIO", "_io.BytesIO", "_io.StringIO"):
            # a byte/str buffer wrapping embedded data; carry the content so a
            # nested loader reading from it can be judged by what it would load
            if args:
                v = to_py(args[0])
                if isinstance(v, (bytes, bytearray, str)):
                    return Const(v)
            return Obj()
        if c in _PURE and args is not None:
            for a in args:
                if isinstance(a, Lazy):
                    self._invoke(a.func, where, "consumed-iterator")
            res = self._pure(c, args)
            if res is not None:
                return res
        # a benign callable: constructing it yields an opaque object
        return Obj(cls=g_from_canonical(c))

    def _pure(self, c, args):
        """Evaluate a pure data transform on concrete constants, so an embedded
        payload assembled by base64/hex/zlib/concatenation can be recovered."""
        concrete = []
        for a in args:
            v = to_py(a)
            if v is _NOPE:
                return None
            if isinstance(v, (bytes, bytearray, str)) and len(v) > _PURE_MAX_IN:
                return None
            concrete.append(v)
        try:
            out = _PURE[c](*concrete)
        except Exception:
            return None
        if isinstance(out, (bytes, bytearray, str, int, float, bool, type(None), tuple)):
            return Const(out)
        return None

    def _invoke(self, func, where, via):
        """A callable is going to be called indirectly with arguments we do not
        model; treat it as called with a non-empty argument."""
        self._call(func, [Const("<indirect>")], where, via)

    def _nested(self, c, args, where):
        payload = None
        path = None
        if args:
            for a in args:
                v = to_py(a)
                if isinstance(v, (bytes, bytearray)):
                    payload = bytes(v)
                    break
                if isinstance(v, str) and path is None:
                    path = v
        if payload is not None:
            if self.depth < _MAX_NEST:
                sub = _scan_bytes_any(payload, where + "»nested", self.depth + 1)
                for f in sub:
                    self.emit(f.kind, f.location, f.detail)
            else:
                self.emit("code-execution", where, f"{c} nested too deep")
            return
        if path is not None:
            self.emit("file-access", where, f"{c} reads path {path!r}")
            return
        # unknown payload handed to a loader -> fail closed
        self.emit("code-execution", where, f"{c} on unresolved data")

    # -- attribute access -------------------------------------------------
    def getattr_av(self, base, name_av) -> AV:
        name = to_py(name_av)
        if not isinstance(name, str):
            return _UNKNOWN
        if isinstance(base, ModuleV):
            ov = self.module_overrides.get(base.name, {})
            if name in ov:
                return ov[name]
            return resolve_global(base.name, name)
        if isinstance(base, G):
            if base.is_module:
                return resolve_global(base.canonical, name)
            return g_from_canonical(f"{base.canonical}.{name}")
        if isinstance(base, Obj) and isinstance(base.cls, G):
            return g_from_canonical(f"{base.cls.canonical}.{name}")
        return _UNKNOWN

    def _resolve_dotted(self, dotted: str) -> AV:
        if ":" in dotted:
            mod, _, attr = dotted.partition(":")
        elif "." in dotted:
            mod, _, attr = dotted.rpartition(".")
        else:
            return ModuleV(dotted)
        return resolve_global(mod, attr)

    # -- build (setstate) -------------------------------------------------
    def build(self, obj, state):
        if isinstance(obj, Partial):
            items = _as_args(state)
            if items:
                obj.func = items[0]
                if len(items) >= 2:
                    ia = _as_args(items[1])
                    obj.args = ia if ia is not None else []
            return obj
        if isinstance(obj, ModuleV) and isinstance(state, DictV):
            ov = self.module_overrides.setdefault(obj.name, {})
            for k, v in state.pairs:
                kk = to_py(k)
                if isinstance(kk, str):
                    ov[kk] = v
            return obj
        return obj


# --------------------------------------------------------------------------
# Single-stream opcode walk
# --------------------------------------------------------------------------
_STRING_OPS = {
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
}
_PUSH_STR = {"STRING", "BINSTRING", "SHORT_BINSTRING"}
_PUSH_BYTES = {"BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8"}
_PUSH_INT = {"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"}
_PUSH_FLOAT = {"FLOAT", "BINFLOAT"}


def _walk_stream(data, start, scan, base_loc):
    """Interpret one pickle stream beginning at ``data[start:]``.
    Returns the offset just past STOP, or raises if the stream is malformed."""
    stack = []
    memo = {}
    ops = 0
    for op, arg, pos in pickletools.genops(data[start:] if start else data):
        ops += 1
        if ops > _MAX_OPS:
            raise ValueError("opcode budget exceeded")
        nm = op.name
        where = f"{base_loc}@{start + pos}"

        if nm in _STRING_OPS or nm in _PUSH_STR:
            stack.append(Const(arg if isinstance(arg, str) else str(arg)))
        elif nm in _PUSH_BYTES:
            stack.append(Const(bytes(arg) if isinstance(arg, (bytes, bytearray)) else arg))
        elif nm in _PUSH_INT:
            stack.append(Const(arg))
        elif nm in _PUSH_FLOAT:
            stack.append(Const(arg))
        elif nm == "NONE":
            stack.append(Const(None))
        elif nm == "NEWTRUE":
            stack.append(Const(True))
        elif nm == "NEWFALSE":
            stack.append(Const(False))
        elif nm == "EMPTY_LIST":
            stack.append(ListV())
        elif nm == "EMPTY_DICT":
            stack.append(DictV())
        elif nm == "EMPTY_SET":
            stack.append(ListV())
        elif nm == "EMPTY_TUPLE":
            stack.append(TupleV([]))
        elif nm == "MARK":
            stack.append(_MARK)
        elif nm == "DUP":
            if stack:
                stack.append(stack[-1])
        elif nm == "POP":
            if stack:
                stack.pop()
        elif nm == "POP_MARK":
            _pop_to_mark(stack)
        elif nm == "GLOBAL":
            module, _, name = str(arg).partition(" ")
            stack.append(scan.resolve_op(module, name))
        elif nm == "STACK_GLOBAL":
            name_av = stack.pop() if stack else _UNKNOWN
            mod_av = stack.pop() if stack else _UNKNOWN
            module = to_py(mod_av)
            name = to_py(name_av)
            if isinstance(module, str) and isinstance(name, str):
                stack.append(scan.resolve_op(module, name))
            else:
                stack.append(_UNKNOWN)
        elif nm in ("EXT1", "EXT2", "EXT4"):
            stack.append(_UNKNOWN)
        elif nm in ("PERSID", "BINPERSID"):
            if nm == "BINPERSID" and stack:
                stack.pop()
            stack.append(_UNKNOWN)       # loader resolves; executes nothing
        elif nm == "REDUCE":
            argtuple = stack.pop() if stack else _UNKNOWN
            func = stack.pop() if stack else _UNKNOWN
            stack.append(scan.call(func, argtuple, where, "REDUCE"))
        elif nm == "INST":
            args = _pop_to_mark(stack)
            module, _, name = str(arg).partition(" ")
            func = resolve_global(module, name)
            stack.append(scan._call(func, args, where, "INST"))
        elif nm == "OBJ":
            items = _pop_to_mark(stack)
            func = items[0] if items else _UNKNOWN
            stack.append(scan._call(func, items[1:], where, "OBJ"))
        elif nm in ("NEWOBJ", "NEWOBJ_EX"):
            if nm == "NEWOBJ_EX":
                kwargs = stack.pop() if stack else _UNKNOWN  # noqa: F841
            argtuple = stack.pop() if stack else _UNKNOWN
            cls = stack.pop() if stack else _UNKNOWN
            stack.append(_newobj(cls, argtuple))
        elif nm == "BUILD":
            state = stack.pop() if stack else _UNKNOWN
            obj = stack[-1] if stack else _UNKNOWN
            if stack:
                stack[-1] = scan.build(obj, state)
        elif nm == "APPEND":
            item = stack.pop() if stack else _UNKNOWN
            if stack and isinstance(stack[-1], ListV):
                stack[-1].items.append(item)
        elif nm == "APPENDS":
            items = _pop_to_mark(stack)
            if stack and isinstance(stack[-1], ListV):
                stack[-1].items.extend(items)
        elif nm == "SETITEM":
            val = stack.pop() if stack else _UNKNOWN
            key = stack.pop() if stack else _UNKNOWN
            if stack and isinstance(stack[-1], DictV):
                stack[-1].pairs.append((key, val))
        elif nm == "SETITEMS":
            items = _pop_to_mark(stack)
            if stack and isinstance(stack[-1], DictV):
                for i in range(0, len(items) - 1, 2):
                    stack[-1].pairs.append((items[i], items[i + 1]))
        elif nm == "ADDITEMS":
            items = _pop_to_mark(stack)
            if stack and isinstance(stack[-1], ListV):
                stack[-1].items.extend(items)
        elif nm == "FROZENSET":
            items = _pop_to_mark(stack)
            stack.append(TupleV(items))
        elif nm == "DICT":
            items = _pop_to_mark(stack)
            d = DictV()
            for i in range(0, len(items) - 1, 2):
                d.pairs.append((items[i], items[i + 1]))
            stack.append(d)
        elif nm == "LIST":
            items = _pop_to_mark(stack)
            stack.append(ListV(items))
        elif nm == "TUPLE":
            items = _pop_to_mark(stack)
            stack.append(TupleV(items))
        elif nm == "TUPLE1":
            a = stack.pop() if stack else _UNKNOWN
            stack.append(TupleV([a]))
        elif nm == "TUPLE2":
            b = stack.pop() if stack else _UNKNOWN
            a = stack.pop() if stack else _UNKNOWN
            stack.append(TupleV([a, b]))
        elif nm == "TUPLE3":
            c = stack.pop() if stack else _UNKNOWN
            b = stack.pop() if stack else _UNKNOWN
            a = stack.pop() if stack else _UNKNOWN
            stack.append(TupleV([a, b, c]))
        elif nm == "MEMOIZE":
            if stack:
                memo[len(memo)] = stack[-1]
        elif nm in ("PUT", "BINPUT", "LONG_BINPUT"):
            if stack:
                memo[arg] = stack[-1]
        elif nm in ("GET", "BINGET", "LONG_BINGET"):
            stack.append(memo.get(arg, _UNKNOWN))
        elif nm in ("PROTO", "FRAME"):
            pass
        elif nm in ("NEXT_BUFFER", "READONLY_BUFFER"):
            if nm == "READONLY_BUFFER" and stack:
                pass
            else:
                stack.append(_UNKNOWN)
        elif nm == "STOP":
            if stack:
                stack.pop()
            return start + pos + 1
        else:
            # unhandled opcode: keep the stack roughly balanced
            delta = len(op.stack_after) - len(op.stack_before)
            if delta < 0:
                for _ in range(-delta):
                    if stack:
                        stack.pop()
            else:
                for _ in range(delta):
                    stack.append(_UNKNOWN)
    raise ValueError("pickle stream ended without STOP")


def _pop_to_mark(stack):
    items = []
    while stack:
        top = stack.pop()
        if top is _MARK:
            items.reverse()
            return items
        items.append(top)
    items.reverse()
    return items


def _newobj(cls, argtuple):
    if isinstance(cls, G) and cls.canonical in policy.PARTIAL:
        args = _as_args(argtuple) or []
        if args:
            return Partial(args[0], args[1:])
    return Obj(cls=cls if isinstance(cls, G) else None)


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def scan_pickle(data: bytes, location: str, depth: int = 0) -> list:
    """Scan one or more concatenated pickle streams.  Returns findings.

    Streams are read until end of file (threat model: a pickle file may be a
    sequence of pickles).  Trailing bytes that do not begin a valid pickle --
    e.g. the raw storage of a legacy torch checkpoint -- are ignored.  If the
    very first stream is unparsable the artefact is reported unsupported."""
    findings: list = []
    scan = _Scan(location, findings, depth)
    off = 0
    n = len(data)
    parsed = 0
    while off < n:
        try:
            nxt = _walk_stream(data, off, scan, location)
        except Exception as exc:
            if parsed == 0:
                return [Finding("unsupported", f"{location}@{off}",
                                f"not a valid pickle: {type(exc).__name__}: {exc}")]
            break
        if nxt <= off:
            break
        off = nxt
        parsed += 1
        if parsed > 64:
            break
    return findings


def _scan_bytes_any(data: bytes, location: str, depth: int) -> list:
    """Scan embedded bytes whose format is decided by content (for nested
    loaders): a zip, a numpy .npy, or one/more pickle streams."""
    from modelgate import formats
    return formats.scan_blob(data, location, depth)
