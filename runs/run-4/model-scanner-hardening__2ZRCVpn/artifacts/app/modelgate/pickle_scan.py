"""Static abstract interpretation of pickle streams.

We never execute a pickle.  Instead we walk its opcodes with
``pickletools.genops`` and run an abstract machine that mirrors what CPython's
unpickler would do: it builds a shadow stack and memo of *abstract values* and
reports a finding exactly when a capability-bearing callable would be invoked
(REDUCE / INST / OBJ / BUILD-retargeting / a consumed map / a nested loader
fed dangerous data).  A capability that is only stored as data is never a
finding, and names reached indirectly (getattr, partial, attrgetter,
methodcaller, import, a later stream, ...) are followed to the call that
reaches them.
"""
from __future__ import annotations

import base64
import binascii
import bz2
import codecs as _codecs_mod
import io
import lzma
import pickletools
import zlib

from modelgate import policy
from modelgate.report import Finding

MAX_OPCODES = 1_500_000
MAX_STREAMS = 4096
MAX_NEST = 6
MAX_COMPUTE_BYTES = 64 * 1024 * 1024

_STRING_OPS = {
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
    "STRING", "SHORT_BINSTRING", "BINSTRING",
}
_BYTES_OPS = {
    "SHORT_BINBYTES", "BINBYTES", "BINBYTES8", "BYTEARRAY8",
}


class V:
    """An abstract value on the pickle machine's shadow stack."""

    __slots__ = ("kind", "data", "path", "cls", "func", "args", "kwargs",
                 "names", "name", "items", "pairs")

    def __init__(self, kind, *, data=None, path=None, cls=None, func=None,
                 args=None, kwargs=None, names=None, name=None, items=None,
                 pairs=None):
        self.kind = kind
        self.data = data
        self.path = path
        self.cls = cls
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.names = names
        self.name = name
        self.items = items
        self.pairs = pairs


OPAQUE = V("opaque")
_MARK = V("mark")


def _const(x):
    return V("const", data=x)


def _concrete(v):
    """Return the concrete Python value of ``v`` or ``_NOPE`` if not known."""
    if v is None:
        return _NOPE
    if v.kind == "const":
        return v.data
    if v.kind == "tuple":
        out = []
        for it in v.items:
            c = _concrete(it)
            if c is _NOPE:
                return _NOPE
            out.append(c)
        return tuple(out)
    if v.kind == "list":
        out = []
        for it in v.items:
            c = _concrete(it)
            if c is _NOPE:
                return _NOPE
            out.append(c)
        return out
    return _NOPE


class _Nope:
    pass


_NOPE = _Nope()


class Ctx:
    def __init__(self, location, torch_context=False, nest=0):
        self.findings: list[Finding] = []
        self.location = location
        self.torch_context = torch_context
        self.nest = nest
        self.overrides: dict[str, dict] = {}
        self.ops = 0
        self.proto = 0
        self.seen_finding_keys = set()

    def emit(self, kind, where, detail):
        key = (kind, detail)
        if key in self.seen_finding_keys:
            return
        self.seen_finding_keys.add(key)
        self.findings.append(Finding(kind, where, detail))


def scan_pickle(data, location, torch_context=False, nest=0):
    ctx = Ctx(location, torch_context=torch_context, nest=nest)
    _run_all_streams(data, ctx)
    return ctx.findings


def _run_all_streams(data, ctx):
    if not data:
        ctx.emit("unsupported", ctx.location, "empty pickle stream")
        return
    bio = io.BytesIO(data)
    n = len(data)
    streams = 0
    while bio.tell() < n:
        start = bio.tell()
        try:
            completed = _run_one_stream(bio, ctx)
        except _StopStream:
            # find_class raised (e.g. a dotted name below protocol 4) or a
            # plain pickle hit a persistent id: pickle.load aborts here, so no
            # further stream in the file is loaded either.
            return
        except _Unsupported as exc:
            if ctx.ops == 0:
                ctx.emit("unsupported", ctx.location,
                         f"could not walk pickle opcodes: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            if ctx.ops == 0:
                ctx.emit("unsupported", ctx.location,
                         f"could not walk pickle opcodes: "
                         f"{type(exc).__name__}: {exc}")
            return
        streams += 1
        if not completed or bio.tell() <= start:
            return
        if streams >= MAX_STREAMS or ctx.ops >= MAX_OPCODES:
            return


class _Unsupported(Exception):
    pass


class _StopStream(Exception):
    pass


def _run_one_stream(bio, ctx):
    """Interpret opcodes from ``bio`` up to and including the next STOP.

    Returns True if a STOP terminated the stream (more streams may follow),
    False if the stream ran out without a STOP (stop scanning).
    """
    stack: list[V] = []
    memo: dict = {}
    ctx.proto = 0
    try:
        gen = pickletools.genops(bio)
        for opcode, arg, _pos in gen:
            ctx.ops += 1
            if ctx.ops > MAX_OPCODES:
                raise _Unsupported("opcode budget exceeded")
            name = opcode.name
            stop = _step(name, arg, stack, memo, ctx)
            if stop:
                return True
    except (_Unsupported, _StopStream):
        raise
    except ValueError as exc:
        # genops failed to decode an opcode: either a truncated/odd pickle or
        # trailing non-pickle bytes (torch legacy storage after the streams).
        raise _Unsupported(str(exc))
    # Ran out of data with no STOP.
    return False


def _pop_to_mark(stack):
    items = []
    while stack:
        v = stack.pop()
        if v is _MARK:
            items.reverse()
            return items
        items.append(v)
    # No mark found: treat the whole stack as the items (defensive).
    items.reverse()
    return items


def _step(name, arg, stack, memo, ctx):
    # --- constants -------------------------------------------------------
    if name in ("INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1",
                "LONG4"):
        stack.append(_const(arg))
    elif name in ("FLOAT", "BINFLOAT"):
        stack.append(_const(arg))
    elif name == "NONE":
        stack.append(_const(None))
    elif name == "NEWTRUE":
        stack.append(_const(True))
    elif name == "NEWFALSE":
        stack.append(_const(False))
    elif name in _STRING_OPS:
        stack.append(_const(arg if isinstance(arg, str) else str(arg)))
    elif name in _BYTES_OPS:
        # genops already returns a bytes object here; avoid copying large
        # tensor buffers.
        stack.append(_const(arg if arg is not None else b""))
    elif name in ("NEXT_BUFFER", "READONLY_BUFFER"):
        if name == "NEXT_BUFFER":
            stack.append(OPAQUE)
    # --- marks / stack ops ----------------------------------------------
    elif name == "MARK":
        stack.append(_MARK)
    elif name == "POP":
        if stack:
            stack.pop()
    elif name == "POP_MARK":
        _pop_to_mark(stack)
    elif name == "DUP":
        if stack:
            stack.append(stack[-1])
    # --- memo ------------------------------------------------------------
    elif name == "MEMOIZE":
        if stack:
            memo[len(memo)] = stack[-1]
    elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
        if stack:
            memo[arg] = stack[-1]
    elif name in ("GET", "BINGET", "LONG_BINGET"):
        stack.append(memo.get(arg, OPAQUE))
    # --- containers ------------------------------------------------------
    elif name == "EMPTY_LIST":
        stack.append(V("list", items=[]))
    elif name == "EMPTY_DICT":
        stack.append(V("dict", pairs=[]))
    elif name == "EMPTY_TUPLE":
        stack.append(V("tuple", items=[]))
    elif name == "EMPTY_SET":
        stack.append(V("set", items=[]))
    elif name == "LIST":
        stack.append(V("list", items=_pop_to_mark(stack)))
    elif name == "TUPLE":
        stack.append(V("tuple", items=_pop_to_mark(stack)))
    elif name == "SET":
        stack.append(V("set", items=_pop_to_mark(stack)))
    elif name == "FROZENSET":
        stack.append(V("set", items=_pop_to_mark(stack)))
    elif name == "DICT":
        flat = _pop_to_mark(stack)
        pairs = [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
        stack.append(V("dict", pairs=pairs))
    elif name == "TUPLE1":
        a = _safe_pop(stack)
        stack.append(V("tuple", items=[a]))
    elif name == "TUPLE2":
        b = _safe_pop(stack)
        a = _safe_pop(stack)
        stack.append(V("tuple", items=[a, b]))
    elif name == "TUPLE3":
        c = _safe_pop(stack)
        b = _safe_pop(stack)
        a = _safe_pop(stack)
        stack.append(V("tuple", items=[a, b, c]))
    elif name == "APPEND":
        v = _safe_pop(stack)
        if stack and stack[-1].kind == "list":
            stack[-1].items.append(v)
    elif name == "APPENDS":
        vs = _pop_to_mark(stack)
        if stack and stack[-1].kind == "list":
            stack[-1].items.extend(vs)
    elif name == "ADDITEMS":
        vs = _pop_to_mark(stack)
        if stack and stack[-1].kind == "set":
            stack[-1].items.extend(vs)
    elif name == "SETITEM":
        val = _safe_pop(stack)
        key = _safe_pop(stack)
        if stack and stack[-1].kind == "dict":
            stack[-1].pairs.append((key, val))
    elif name == "SETITEMS":
        flat = _pop_to_mark(stack)
        if stack and stack[-1].kind == "dict":
            for i in range(0, len(flat) - 1, 2):
                stack[-1].pairs.append((flat[i], flat[i + 1]))
    # --- globals ---------------------------------------------------------
    elif name == "GLOBAL":
        s = str(arg)
        module, _, nm = s.partition(" ")
        stack.append(_resolve_global(module, nm, ctx))
    elif name == "STACK_GLOBAL":
        nm = _concrete(_safe_pop(stack))
        module = _concrete(_safe_pop(stack))
        if isinstance(module, str) and isinstance(nm, str):
            stack.append(_resolve_global(module, nm, ctx))
        else:
            stack.append(OPAQUE)
    elif name in ("EXT1", "EXT2", "EXT4"):
        stack.append(OPAQUE)  # copyreg extension registry: not attacker code
    # --- construction / calls -------------------------------------------
    elif name == "REDUCE":
        argtup = _safe_pop(stack)
        callable_v = _safe_pop(stack)
        args = argtup.items if argtup.kind == "tuple" else [argtup]
        stack.append(_invoke(callable_v, args, ctx))
    elif name == "INST":
        s = str(arg)
        module, _, nm = s.partition(" ")
        args = _pop_to_mark(stack)
        ref = _resolve_global(module, nm, ctx)
        stack.append(_invoke(ref, args, ctx))
    elif name == "OBJ":
        items = _pop_to_mark(stack)
        if items:
            stack.append(_invoke(items[0], items[1:], ctx))
        else:
            stack.append(OPAQUE)
    elif name == "NEWOBJ":
        argtup = _safe_pop(stack)
        cls = _safe_pop(stack)
        stack.append(_construct_new(cls))
    elif name == "NEWOBJ_EX":
        _kwargs = _safe_pop(stack)
        _argtup = _safe_pop(stack)
        cls = _safe_pop(stack)
        stack.append(_construct_new(cls))
    elif name == "BUILD":
        state = _safe_pop(stack)
        if stack:
            _build(stack[-1], state, ctx)
    # --- persistent ids --------------------------------------------------
    elif name in ("PERSID", "BINPERSID"):
        if name == "PERSID":
            pass  # arg carried the id inline
        else:
            _safe_pop(stack)  # the id object
        if ctx.torch_context:
            stack.append(OPAQUE)  # resolved by torch's storage mechanism
        else:
            # Plain pickle.load has no persistent_load: it raises here and
            # nothing further in the stream executes.
            raise _StopStream()
    # --- end -------------------------------------------------------------
    elif name == "STOP":
        return True
    elif name == "PROTO":
        ctx.proto = arg if isinstance(arg, int) else ctx.proto
    elif name == "FRAME":
        pass
    else:
        # Unknown/odd opcode: leave stack as-is (defensive, no crash).
        pass
    return False


def _safe_pop(stack):
    while stack:
        v = stack.pop()
        if v is _MARK:
            continue
        return v
    return OPAQUE


def _resolve_global(module, name, ctx):
    # The unpickler imports ``module`` then resolves ``name`` attribute by
    # attribute, but only at protocol 4+.  Below protocol 4 a dotted ``name``
    # is one getattr on the literal "a.b" string, which raises and aborts the
    # whole load, executing nothing further.
    module, name = policy.apply_compat(module, name)
    if "." in name and ctx.proto < 4:
        raise _StopStream()
    cur = V("ref", path=policy.normalize_path(module))
    for seg in name.split("."):
        if seg:
            cur = _resolve_attr(cur, seg, ctx)
    return cur


def _construct_new(cls):
    # NEWOBJ / NEWOBJ_EX call __new__ only: never executes __init__, so even a
    # process/network class carries no capability here.
    if cls is not None and cls.kind == "ref":
        return V("object", cls=cls.path)
    return V("object", cls=None)


def _build(obj, state, ctx):
    if obj.kind == "partial":
        # __setstate__ on functools.partial retargets it with
        # (func, args, keywords, dict).
        if state.kind == "tuple" and state.items:
            obj.func = state.items[0]
            if len(state.items) > 1 and state.items[1].kind == "tuple":
                obj.args = list(state.items[1].items)
            if len(state.items) > 2:
                obj.kwargs = _dict_mapping(state.items[2]) or {}
        return
    if obj.kind == "ref":
        # Writing state into a module's __dict__ rebinds its attributes.
        mapping = _dict_mapping(state)
        if mapping is not None:
            ctx.overrides.setdefault(obj.path, {}).update(mapping)
        return
    if obj.kind == "object":
        obj.data = state


def _dict_mapping(state):
    if state is None or state.kind != "dict":
        return None
    out = {}
    for k, v in state.pairs:
        kc = _concrete(k)
        if isinstance(kc, str):
            out[kc] = v
    return out or None


def _invoke(func, args, ctx):
    """Abstractly invoke ``func`` with ``args`` (full call: __new__+__init__)."""
    k = func.kind
    if k == "ref":
        return _invoke_ref(func.path, args, ctx)
    if k == "partial":
        if func.kwargs:
            _maybe_key_call(func.func, func.kwargs, ctx)
        effargs = (func.args or []) + list(args)
        if func.kwargs and not effargs:
            # Keyword arguments count as arguments for if_args capabilities
            # (e.g. Popen(args=...), FTP(host=...)) bound through the partial.
            effargs = [OPAQUE]
        return _invoke(func.func, effargs, ctx)
    if k == "attrgetter":
        if args and func.names:
            return _resolve_attr(args[0], func.names[0], ctx)
        return OPAQUE
    if k == "methodcaller":
        if args:
            method = _resolve_attr(args[0], func.name, ctx)
            return _invoke(method, func.args or [], ctx)
        return OPAQUE
    if k == "wrapper":
        return _invoke(func.func, args, ctx)
    if k == "boundmethod":
        return _invoke_boundmethod(func, args)
    if k == "threadstart":
        if func.func is not None:
            _invoke(func.func, [OPAQUE], ctx)
        return OPAQUE
    # const string/bytes receiver being "called": not meaningful; opaque.
    return OPAQUE


def _invoke_boundmethod(func, args):
    """A method on a concrete str/bytes receiver, used to assemble data."""
    recv = func.func.data
    name = func.name
    try:
        vals = [_concrete(a) for a in args]
        if any(v is _NOPE for v in vals):
            return OPAQUE
        if name in ("join", "format", "replace", "encode", "decode",
                    "format_map", "removeprefix", "removesuffix", "strip",
                    "lstrip", "rstrip", "upper", "lower", "title"):
            result = getattr(recv, name)(*vals)
        elif name == "fromhex":
            result = bytes.fromhex(vals[0]) if vals else None
        else:
            return OPAQUE
    except Exception:  # noqa: BLE001
        return OPAQUE
    if isinstance(result, (str, bytes, bytearray)):
        if len(result) > MAX_COMPUTE_BYTES:
            return OPAQUE
        return _const(result)
    return OPAQUE


_CALLABLE_KINDS = ("ref", "partial", "attrgetter", "methodcaller", "wrapper",
                   "lazymap", "threadstart")


def _invoke_ref(path, args, ctx):
    path = policy.normalize_path(path)
    if path in policy.SINKS:
        for a in args:
            if a.kind in _CALLABLE_KINDS:
                _invoke(a, [OPAQUE], ctx)
        return OPAQUE
    if path in policy.THREAD_CTORS:
        # Thread(group, target, ...): the target runs only on .start(); store it.
        target = args[1] if len(args) > 1 else None
        return V("object", cls=path, func=target)
    if path == "operator.getitem" and len(args) == 2:
        container = args[0]
        key = _concrete(args[1])
        if container.kind == "ref" and container.path.endswith("sys.modules") \
                and isinstance(key, str):
            return V("ref", path=policy.normalize_path(key))
    cat, info = policy.classify(path)
    if cat == "cap":
        kind, argpolicy = info
        if argpolicy == "always" or (argpolicy == "if_args" and len(args) >= 1):
            ctx.emit(kind, ctx.location, f"call {path}")
        return V("object", cls=path)
    if cat == "loader":
        _nested_loader(path, info, args, ctx)
        return OPAQUE
    if cat == "transform":
        return _apply_transform(info, args, ctx)
    if cat == "consumer":
        _consume(args, ctx)
        return OPAQUE
    if cat == "bytesio":
        if args and args[0].kind == "const" and \
                isinstance(args[0].data, (bytes, bytearray, str)):
            d = args[0].data
            if isinstance(d, str):
                d = d.encode("latin-1", "ignore")
            return V("filelike", data=bytes(d))
        return V("filelike", data=None)
    if cat == "puredata":
        return _compute(path, args)
    return V("object", cls=path)  # plain class/function construction


def _apply_transform(tkind, args, ctx):
    if tkind == "getattr":
        if len(args) >= 2:
            nm = _concrete(args[1])
            if isinstance(nm, str):
                return _resolve_attr(args[0], nm, ctx)
        return OPAQUE
    if tkind == "import":
        nm = _concrete(args[0]) if args else _NOPE
        if isinstance(nm, str):
            return V("ref", path=policy.normalize_path(nm))
        return OPAQUE
    if tkind == "attrgetter":
        names = [c for c in (_concrete(a) for a in args) if isinstance(c, str)]
        return V("attrgetter", names=names) if names else OPAQUE
    if tkind == "methodcaller":
        if args:
            nm = _concrete(args[0])
            if isinstance(nm, str):
                return V("methodcaller", name=nm, args=list(args[1:]))
        return OPAQUE
    if tkind == "partial":
        if args:
            return V("partial", func=args[0], args=list(args[1:]))
        return OPAQUE
    if tkind == "resolve_name":
        nm = _concrete(args[0]) if args else _NOPE
        if isinstance(nm, str):
            return _resolve_dotted(nm, ctx)
        return OPAQUE
    if tkind == "identity":
        return args[0] if args else OPAQUE
    if tkind == "wrap":
        return V("wrapper", func=args[0]) if args else OPAQUE
    if tkind == "map":
        return V("lazymap", func=args[0]) if args else OPAQUE
    if tkind == "reduce_fn":
        if args:
            _invoke(args[0], [OPAQUE, OPAQUE], ctx)
        return OPAQUE
    if tkind == "call":
        if args:
            return _invoke(args[0], list(args[1:]), ctx)
        return OPAQUE
    return OPAQUE


def _consume(args, ctx):
    for a in args:
        if a.kind == "lazymap":
            _invoke(a.func, [OPAQUE], ctx)


_KEY_CALLERS = {
    "builtins.sorted", "builtins.min", "builtins.max",
    "heapq.nlargest", "heapq.nsmallest", "itertools.groupby",
}


def _maybe_key_call(func, kwargs, ctx):
    """sorted/min/max(..., key=f) invokes f on elements."""
    if func is None or func.kind != "ref":
        return
    if policy.normalize_path(func.path) in _KEY_CALLERS:
        key = kwargs.get("key")
        if key is not None:
            _invoke(key, [OPAQUE], ctx)


def _resolve_dotted(s, ctx):
    s = s.replace(":", ".")
    parts = s.split(".")
    if not parts:
        return OPAQUE
    cur = V("ref", path=policy.normalize_path(parts[0]))
    for p in parts[1:]:
        cur = _resolve_attr(cur, p, ctx)
    return cur


def _resolve_attr(obj, name, ctx):
    if "." in name:
        cur = obj
        for p in name.split("."):
            cur = _resolve_attr(cur, p, ctx)
        return cur
    base = None
    if obj.kind == "ref":
        base = obj.path
    elif obj.kind == "object" and obj.cls:
        # Starting/running a stored thread/process target executes it.
        if obj.cls in policy.THREAD_CTORS and name in ("start", "run"):
            return V("threadstart", func=obj.func)
        base = obj.cls
    elif obj.kind == "module":
        base = obj.path
    if base is None:
        # Attribute off a concrete string/bytes: used for join/format/fromhex.
        if obj.kind == "const" and isinstance(obj.data, (str, bytes, bytearray)):
            return V("boundmethod", func=obj, name=name)
        return OPAQUE
    # Module override from a prior BUILD.
    ov = ctx.overrides.get(base)
    if ov is not None and name in ov:
        return ov[name]
    # Reaching a module through another module's attribute (shutil.os is os).
    if policy.is_submodule_attr(name):
        return V("ref", path=policy.normalize_path(name))
    return V("ref", path=f"{base}.{name}")


def _nested_loader(path, kind, args, ctx):
    if ctx.nest >= MAX_NEST:
        ctx.emit("unsupported", ctx.location, f"nested loader {path} too deep")
        return
    data_arg = args[0] if args else None
    where = f"{ctx.location}->{path}"
    if data_arg is None:
        ctx.emit("code-execution", where, f"{path} on non-constant data")
        return
    # Concrete embedded bytes: recurse into the appropriate loader.
    raw = None
    if data_arg.kind == "const" and isinstance(data_arg.data, (bytes, bytearray)):
        raw = bytes(data_arg.data)
    elif data_arg.kind == "filelike" and data_arg.data is not None:
        raw = data_arg.data
    if raw is not None:
        _recurse_nested(kind, raw, where, ctx)
        return
    # A path string: the loader reads that path (file access).
    if data_arg.kind == "const" and isinstance(data_arg.data, str):
        ctx.emit("file-access", where, f"{path} reads path {data_arg.data!r}"[:200])
        return
    if data_arg.kind == "filelike" and data_arg.data is None:
        # Came from open(path): the open() already produced a file-access
        # finding; the load adds nothing provable.
        return
    # Opaque data we cannot prove benign: fail closed.
    ctx.emit("code-execution", where, f"{path} on non-constant data")


def _recurse_nested(kind, raw, where, ctx):
    if kind in ("pickle-bytes", "pickle-file", "joblib", "pandas"):
        sub = scan_pickle(raw, where, torch_context=False, nest=ctx.nest + 1)
        _merge(ctx, sub)
    elif kind == "torch-bytes":
        sub = scan_pickle(raw, where, torch_context=True, nest=ctx.nest + 1)
        _merge(ctx, sub)
    elif kind in ("numpy", "numpy-array"):
        from modelgate import container
        sub = container.analyze_npy_bytes(raw, where, nest=ctx.nest + 1)
        _merge(ctx, sub)
    elif kind == "torch":
        from modelgate import container
        sub = container.analyze_artifact(raw, where, nest=ctx.nest + 1)
        _merge(ctx, sub)


def _merge(ctx, findings):
    for f in findings:
        ctx.emit(f.kind, f.location, f.detail)


# ---------------------------------------------------------------------------
# Side-effect-free computation of embedded data.
# ---------------------------------------------------------------------------
def _compute(path, args):
    try:
        vals = [_concrete(a) for a in args]
        if any(v is _NOPE for v in vals):
            return OPAQUE
        result = _compute_impl(path, vals)
    except Exception:  # noqa: BLE001 - computation is best-effort only
        return OPAQUE
    if result is None:
        return OPAQUE
    if isinstance(result, (bytes, bytearray, str)) and len(result) > MAX_COMPUTE_BYTES:
        return OPAQUE
    return _const(result)


_TEXT_CODECS = {
    "latin_1", "latin1", "latin", "l1", "iso_8859_1", "8859", "cp819",
    "utf_8", "utf8", "u8", "utf", "ascii", "us_ascii", "646",
    "raw_unicode_escape", "unicode_escape", "utf_16", "utf_16_le",
    "utf_16_be", "utf_32", "utf_32_le", "utf_32_be", "hex", "hex_codec",
    "base64", "base64_codec",
}


def _bounded(decompressor, data):
    """Decompress with an output cap so a bomb cannot exhaust memory."""
    if not isinstance(data, (bytes, bytearray)):
        return None
    out = decompressor.decompress(bytes(data), MAX_COMPUTE_BYTES + 1)
    if len(out) > MAX_COMPUTE_BYTES:
        return None
    return out


def _compute_impl(path, vals):
    if path == "base64.b64decode" or path == "base64.standard_b64decode":
        return base64.b64decode(vals[0])
    if path == "base64.urlsafe_b64decode":
        return base64.urlsafe_b64decode(vals[0])
    if path == "base64.b16decode":
        return base64.b16decode(vals[0])
    if path == "base64.b32decode":
        return base64.b32decode(vals[0])
    if path == "base64.b85decode":
        return base64.b85decode(vals[0])
    if path == "base64.a85decode":
        return base64.a85decode(vals[0])
    if path == "base64.decodebytes":
        return base64.decodebytes(vals[0])
    if path == "binascii.a2b_base64":
        return binascii.a2b_base64(vals[0])
    if path in ("binascii.unhexlify", "binascii.a2b_hex"):
        return binascii.unhexlify(vals[0])
    if path == "zlib.decompress":
        return _bounded(zlib.decompressobj(), vals[0])
    if path == "gzip.decompress":
        return _bounded(zlib.decompressobj(31), vals[0])
    if path == "bz2.decompress":
        return _bounded(bz2.BZ2Decompressor(), vals[0])
    if path == "lzma.decompress":
        return _bounded(lzma.LZMADecompressor(), vals[0])
    if path in ("codecs.encode", "codecs.decode"):
        enc = (vals[1] if len(vals) > 1 else "utf-8")
        if not isinstance(enc, str) or enc.lower().replace("-", "_") not in \
                _TEXT_CODECS:
            return None  # skip compression codecs (bomb-safe); result unused
        fn = _codecs_mod.encode if path == "codecs.encode" else _codecs_mod.decode
        return fn(vals[0], enc)
    if path in ("operator.add", "operator.concat"):
        if len(vals) == 2 and type(vals[0]) is type(vals[1]) and \
                isinstance(vals[0], (str, bytes)):
            return vals[0] + vals[1]
        return None
    if path == "operator.mod":
        if len(vals) == 2 and isinstance(vals[0], (str, bytes)):
            return vals[0] % vals[1]
        return None
    if path == "operator.getitem":
        if len(vals) == 2:
            try:
                return vals[0][vals[1]]
            except Exception:  # noqa: BLE001
                return None
        return None
    if path == "builtins.bytes":
        if len(vals) == 1 and isinstance(vals[0], (list, tuple)):
            return bytes(vals[0])
        if len(vals) >= 2 and isinstance(vals[0], str):
            return bytes(vals[0], vals[1])
        return None
    if path == "builtins.bytearray":
        if len(vals) == 1 and isinstance(vals[0], (list, tuple)):
            return bytes(bytearray(vals[0]))
        return None
    if path == "builtins.str":
        if vals and isinstance(vals[0], str):
            return vals[0]
        return None
    if path in ("builtins.bytes.fromhex", "builtins.bytearray.fromhex"):
        if vals and isinstance(vals[0], str):
            return bytes.fromhex(vals[0])
        return None
    return None
