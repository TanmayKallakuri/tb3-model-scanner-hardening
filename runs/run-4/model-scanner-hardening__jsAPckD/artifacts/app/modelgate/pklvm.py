"""Abstract interpreter for pickle streams.

The scanner never unpickles.  It walks opcodes with ``pickletools.genops`` and
simulates the unpickler's stack and memo with *abstract* values, so it can
tell a capability that is merely stored as data (placed in a container, bound
as an attribute, memoised and never called) from one that is actually
executed.  A finding is emitted only when a capability-bearing callable is
invoked -- by REDUCE, INST, OBJ, an argument-less INST/OBJ, a retargeted
partial, a consumed lazy map, or a nested loader fed a payload that itself
does something.

See /app/docs/threat-model.md section 3 for the pickle semantics this follows.
"""
from __future__ import annotations

import io
import pickletools

from modelgate import policy
from modelgate.report import Finding

MAX_DEPTH = 8
_MAX_OPS = 2_000_000  # guard against pathological streams


# --------------------------------------------------------------------------
# Abstract values
# --------------------------------------------------------------------------
class Val:
    __slots__ = ("tag", "path", "is_module", "func", "args", "names", "value")

    def __init__(self, tag, *, path=None, is_module=False, func=None,
                 args=None, names=None, value=None):
        self.tag = tag
        self.path = path            # 'ref': dotted path
        self.is_module = is_module  # 'ref': path names a module object
        self.func = func            # 'partial'/'lazy': wrapped Val
        self.args = args            # 'partial'/'methodcaller'/container items
        self.names = names          # 'attrgetter': list of attribute names
        self.value = value          # 'bytes'/'str'/'int'/'const': concrete value


UNKNOWN = Val("unknown")
MARK = Val("mark")


def _ref(path, is_module=False):
    return Val("ref", path=path, is_module=is_module)


def _const_bytes(b):
    return Val("bytes", value=b)


def _const_str(s):
    return Val("str", value=s)


# --------------------------------------------------------------------------
# Name resolution
# --------------------------------------------------------------------------
def resolve_path(module: str, name: str) -> str:
    """Canonical dotted path for a GLOBAL(module, name), applying aliases."""
    # Python-2 name renames (e.g. __builtin__.unicode -> builtins.str).
    if (module, name) in policy.NAME_ALIASES:
        module, name = policy.NAME_ALIASES[(module, name)]
    module = policy.alias_module(module)
    parts = name.split(".") if name else []
    cur = module
    out = []
    i = 0
    # Walk leading components that are themselves modules (shutil.os -> os).
    while i < len(parts):
        key = (cur, parts[i])
        if key in policy.SUBMODULE_ALIASES:
            cur = policy.SUBMODULE_ALIASES[key]
            i += 1
        else:
            break
    out = [cur] + parts[i:]
    return ".".join(p for p in out if p)


def classify(path: str):
    """Return (role, kind) for a dotted path.

    role: 'cap', 'netctor', 'loader', 'enabler', 'consumer', 'transform',
          'module', 'benign'.
    """
    if path in policy.CAP:
        return "cap", policy.CAP[path]
    if path in policy.NETCTOR:
        return "netctor", policy.NET
    if path in policy.ALL_LOADERS:
        return "loader", policy.CODE
    if path in policy.ENABLERS:
        return "enabler", policy.ENABLERS[path]
    if path in policy.CONSUMERS:
        return "consumer", None
    if path in policy.TRANSFORMS:
        return "transform", None
    return "benign", None


# --------------------------------------------------------------------------
# Interpreter
# --------------------------------------------------------------------------
class _Stop(Exception):
    """The modelled load would raise here (e.g. persistent id in plain pickle)."""


class Interp:
    def __init__(self, location, persid_safe, module_overrides, depth):
        self.loc = location
        self.persid_safe = persid_safe
        self.overrides = module_overrides  # path -> Val, from BUILD/setattr on a module
        self.depth = depth
        self.findings: list[Finding] = []
        self.stack: list[Val] = []
        self.memo: dict = {}
        self._memo_auto = 0
        self.proto = 0  # set by the PROTO opcode; governs dotted-name resolution

    def _resolve_ref(self, module, name):
        """Build a ref Val for GLOBAL/STACK_GLOBAL/INST, honouring the fact that
        a dotted name only resolves attribute-by-attribute at protocol 4+, and
        below that the load raises (so nothing further executes)."""
        if "." in (name or "") and self.proto < 4:
            raise _Stop()
        return _ref(resolve_path(module, name))

    # -- helpers -----------------------------------------------------------
    def emit(self, kind, pos, detail):
        self.findings.append(Finding(kind, f"{self.loc}@{pos}", detail))

    def push(self, v):
        self.stack.append(v)

    def pop(self):
        return self.stack.pop() if self.stack else UNKNOWN

    def pop_to_mark(self):
        items = []
        while self.stack:
            v = self.stack.pop()
            if v is MARK:
                break
            items.append(v)
        items.reverse()
        return items

    # -- call modelling ----------------------------------------------------
    def container_items(self, v):
        if v is None:
            return []
        if v.tag in ("tuple", "list", "set"):
            return v.args or []
        return None

    def do_call(self, func: Val, args: list, pos: int, _depth: int = 0) -> Val:
        """Model calling ``func`` with positional ``args``; may emit; returns result."""
        if func is None or _depth > 50:
            return UNKNOWN
        tag = func.tag
        if tag == "ref":
            return self._call_ref(func, args, pos, _depth)
        if tag == "partial":
            return self.do_call(func.func, (func.args or []) + list(args), pos, _depth + 1)
        if tag == "attrgetter":
            if args and func.names:
                base = args[0]
                r = base
                for nm in func.names:
                    r = self.getattr_val(r, nm)
                return r
            return UNKNOWN
        if tag == "methodcaller":
            if args:
                m = self.getattr_val(args[0], func.names[0]) if func.names else UNKNOWN
                return self.do_call(m, func.args or [], pos, _depth + 1)
            return UNKNOWN
        if tag == "lazy":
            # Calling the result of map()/filter() isn't how it runs; ignore.
            return UNKNOWN
        return UNKNOWN

    def _call_ref(self, func: Val, args: list, pos: int, _depth: int = 0) -> Val:
        path = func.path
        # a module override may have retargeted this name
        if path in self.overrides:
            return self.do_call(self.overrides[path], args, pos, _depth + 1)
        role, kind = classify(path)
        if role == "cap":
            self.emit(kind, pos, f"call {path}")
            return UNKNOWN
        if role == "netctor":
            if args:  # a host argument means it connects
                self.emit(policy.NET, pos, f"call {path}(host)")
            return UNKNOWN
        if role == "loader":
            return self._call_loader(path, args, pos)
        if role == "enabler":
            return self._call_enabler(kind, args, pos, _depth)
        if role == "consumer":
            self._consume(args, pos)
            return UNKNOWN
        if role == "transform":
            return self._transform(path, args)
        # benign callable or module: a module isn't callable, a benign global
        # constructs/returns something opaque.
        return UNKNOWN

    def _consume(self, args, pos):
        for a in args:
            if a is not None and a.tag == "lazy":
                self._run_lazy(a, pos)
            items = self.container_items(a)
            if items:
                for it in items:
                    if it is not None and it.tag == "lazy":
                        self._run_lazy(it, pos)

    def _run_lazy(self, lazy, pos):
        f = lazy.func
        if f is None:
            return
        if f.tag == "ref":
            role, kind = classify(f.path)
            if role == "cap":
                self.emit(kind, pos, f"map/consume {f.path}")
            elif role == "netctor":
                self.emit(policy.NET, pos, f"map/consume {f.path}")

    def _call_enabler(self, which, args, pos, _depth: int = 0) -> Val:
        if which == "getattr":
            if len(args) >= 2:
                return self.getattr_val(args[0], self._as_str(args[1]))
            return UNKNOWN
        if which == "setattr":
            # setattr(module, name, value): retarget a later GLOBAL lookup.
            if len(args) >= 3 and args[0] is not None and args[0].tag == "ref" and args[0].is_module:
                nm = self._as_str(args[2 - 1])
                if nm:
                    self.overrides[f"{args[0].path}.{nm}"] = args[2]
            return UNKNOWN
        if which == "import":
            nm = self._as_str(args[0]) if args else None
            if nm:
                return _ref(policy.alias_module(nm), is_module=True)
            return UNKNOWN
        if which == "resolve_name":
            nm = self._as_str(args[0]) if args else None
            if nm:
                return _ref(resolve_path(*_split_resolve(nm)))
            return UNKNOWN
        if which == "partial":
            if args:
                return Val("partial", func=args[0], args=list(args[1:]))
            return UNKNOWN
        if which == "attrgetter":
            names = []
            for a in args:
                s = self._as_str(a)
                if s is None:
                    return UNKNOWN
                names.extend(s.split("."))
            return Val("attrgetter", names=names)
        if which == "methodcaller":
            if args:
                nm = self._as_str(args[0])
                if nm:
                    return Val("methodcaller", names=[nm], args=list(args[1:]))
            return UNKNOWN
        if which == "call":
            if args:
                return self.do_call(args[0], list(args[1:]), pos, _depth + 1)
            return UNKNOWN
        if which == "reduce":
            # functools.reduce(func, seq[, init]) applies func immediately.
            if args:
                f = args[0]
                if f is not None and f.tag == "ref":
                    role, kind = classify(f.path)
                    if role in ("cap", "netctor"):
                        self.emit(kind or policy.NET, pos, f"reduce {f.path}")
            return UNKNOWN
        if which in ("map", "filter"):
            if args:
                return Val("lazy", func=args[0])
            return UNKNOWN
        if which == "identity":
            return args[0] if args else UNKNOWN
        if which == "vars":
            return args[0] if args else UNKNOWN
        if which == "getitem":
            if len(args) >= 2:
                base, key = args[0], args[1]
                ks = self._as_str(key)
                if base is not None and base.tag == "ref" and ks:
                    if base.path in ("sys.modules", "os.sys.modules"):
                        return _ref(policy.alias_module(ks), is_module=True)
                    if base.is_module:
                        return self.getattr_val(base, ks)
            return UNKNOWN
        return UNKNOWN

    def getattr_val(self, base: Val, name) -> Val:
        if base is None or name is None:
            return UNKNOWN
        if base.tag == "ref":
            newpath = f"{base.path}.{name}"
            if newpath in self.overrides:
                return self.overrides[newpath]
            return _ref(newpath, is_module=base.is_module)
        return UNKNOWN

    def _call_loader(self, path, args, pos) -> Val:
        arg0 = args[0] if args else None
        if path in policy.LOADER_MARSHAL:
            self.emit(policy.CODE, pos, f"call {path} (bytecode)")
            return UNKNOWN
        data = arg0.value if (arg0 is not None and arg0.tag == "bytes") else None
        if data is not None:
            nested = self._analyse_nested(path, data, pos)
            self.findings.extend(nested)
            return UNKNOWN
        if arg0 is not None and arg0.tag == "str":
            # a path argument: the loader reads that file
            self.emit(policy.FILE, pos, f"call {path}('{_trim(arg0.value)}')")
            return UNKNOWN
        # unknown payload handed to a deserializer: fail closed
        self.emit(policy.CODE, pos, f"call {path}(<unresolved payload>)")
        return UNKNOWN

    def _analyse_nested(self, path, data, pos):
        from modelgate import container  # late import to avoid a cycle
        loc = f"{self.loc}@{pos}->{path}"
        try:
            if path in policy.LOADER_PICKLE:
                return analyse_streams(data, loc, persid_safe=False, depth=self.depth + 1)
            if path in policy.LOADER_TORCH:
                return container.analyse_blob(data, loc, self.depth + 1, torch=True)
            return container.analyse_blob(data, loc, self.depth + 1)
        except Exception:  # noqa: BLE001 - a nested-analysis bug must not crash the scan
            return []

    # -- constant folding --------------------------------------------------
    def _as_str(self, v):
        if v is not None and v.tag == "str":
            return v.value
        return None

    def _as_bytes(self, v):
        if v is None:
            return None
        if v.tag == "bytes":
            return v.value
        return None

    def _transform(self, path, args) -> Val:
        try:
            return self._transform_inner(path, args)
        except Exception:
            return UNKNOWN

    def _transform_inner(self, path, args) -> Val:
        import base64 as _b64
        import binascii as _ba

        def arg(i):
            return args[i] if i < len(args) else None

        if path in ("operator.add", "operator.concat", "operator.iadd"):
            a, b = self._as_bytes(arg(0)), self._as_bytes(arg(1))
            if a is not None and b is not None:
                return _const_bytes(a + b)
            sa, sb = self._as_str(arg(0)), self._as_str(arg(1))
            if sa is not None and sb is not None:
                return _const_str(sa + sb)
            return UNKNOWN
        if path in ("builtins.bytes", "builtins.bytearray"):
            b = self._as_bytes(arg(0))
            if b is not None:
                return _const_bytes(bytes(b))
            return UNKNOWN
        if path == "builtins.bytes.fromhex":
            s = self._as_str(arg(0))
            if s is not None:
                return _const_bytes(bytes.fromhex(s))
            return UNKNOWN
        if path in ("builtins.str.encode", "builtins.bytes.decode", "builtins.str"):
            s = self._as_str(arg(0))
            if s is not None:
                return _const_bytes(s.encode()) if "encode" in path else _const_str(s)
            return UNKNOWN

        payload = self._as_bytes(arg(0))
        if payload is None:
            s = self._as_str(arg(0))
            payload = s.encode("latin-1", "ignore") if s is not None else None
        if payload is None:
            return UNKNOWN
        if len(payload) > 64 * 1024 * 1024:
            return UNKNOWN

        if path in ("base64.b64decode", "base64.standard_b64decode", "base64.decodebytes"):
            return _const_bytes(_b64.b64decode(payload))
        if path == "base64.urlsafe_b64decode":
            return _const_bytes(_b64.urlsafe_b64decode(payload))
        if path == "base64.b16decode":
            return _const_bytes(_b64.b16decode(payload))
        if path == "base64.b32decode":
            return _const_bytes(_b64.b32decode(payload))
        if path == "base64.b85decode":
            return _const_bytes(_b64.b85decode(payload))
        if path == "base64.a85decode":
            return _const_bytes(_b64.a85decode(payload))
        if path in ("binascii.unhexlify", "binascii.a2b_hex"):
            return _const_bytes(_ba.unhexlify(payload))
        if path == "binascii.a2b_base64":
            return _const_bytes(_ba.a2b_base64(payload))
        if path == "zlib.decompress":
            import zlib
            return _const_bytes(zlib.decompress(payload))
        if path == "gzip.decompress":
            import gzip
            return _const_bytes(gzip.decompress(payload))
        if path == "bz2.decompress":
            import bz2
            return _const_bytes(bz2.decompress(payload))
        if path == "lzma.decompress":
            import lzma
            return _const_bytes(lzma.decompress(payload))
        if path in ("codecs.decode", "_codecs.decode", "codecs.encode", "_codecs.encode"):
            enc = (self._as_str(arg(1)) or "").lower()
            if enc in ("base64", "base64_codec"):
                return _const_bytes(_b64.b64decode(payload))
            if enc in ("hex", "hex_codec"):
                return _const_bytes(_ba.unhexlify(payload))
            if enc in ("zlib", "zlib_codec"):
                import zlib
                return _const_bytes(zlib.decompress(payload))
            if enc in ("bz2", "bz2_codec"):
                import bz2
                return _const_bytes(bz2.decompress(payload))
            if enc in ("", "latin-1", "latin1", "utf-8", "utf8", "ascii"):
                return _const_bytes(payload)
            return UNKNOWN
        return UNKNOWN

    # -- opcode loop -------------------------------------------------------
    def run(self, bio: io.BufferedIOBase):
        count = 0
        for op, arg, pos in pickletools.genops(bio):
            count += 1
            if count > _MAX_OPS:
                break
            nm = op.name
            h = _HANDLERS.get(nm)
            if h is not None:
                h(self, arg, pos)
            else:
                self._generic(op, pos)
            if nm == "STOP":
                break

    # -- generic fallback for opcodes without a dedicated handler ----------
    def _generic(self, op, pos):
        before = op.stack_before
        n = len(before)
        if n:
            for _ in range(min(n, len(self.stack))):
                self.stack.pop()
        for _ in op.stack_after:
            self.stack.append(UNKNOWN)


# --------------------------------------------------------------------------
# Opcode handlers
# --------------------------------------------------------------------------
def _h_push_unknown(self, arg, pos):
    self.push(UNKNOWN)


def _h_int(self, arg, pos):
    self.push(Val("int", value=arg))


def _h_str(self, arg, pos):
    self.push(_const_str(arg if isinstance(arg, str) else str(arg)))


def _h_bytes(self, arg, pos):
    if isinstance(arg, (bytes, bytearray)):
        self.push(_const_bytes(bytes(arg)))
    else:
        self.push(Val("bytes", value=None))


def _h_none(self, arg, pos):
    self.push(Val("const", value=None))


def _h_bool(self, arg, pos):
    self.push(Val("const", value=arg))


def _h_mark(self, arg, pos):
    self.push(MARK)


def _h_pop(self, arg, pos):
    self.pop()


def _h_pop_mark(self, arg, pos):
    self.pop_to_mark()


def _h_dup(self, arg, pos):
    if self.stack:
        self.push(self.stack[-1])
    else:
        self.push(UNKNOWN)


def _h_global(self, arg, pos):
    s = arg if isinstance(arg, str) else str(arg)
    if " " in s:
        module, name = s.split(" ", 1)
    else:
        module, name = s, ""
    self.push(self._resolve_ref(module, name))


def _h_stack_global(self, arg, pos):
    name_v = self.pop()
    module_v = self.pop()
    module = self._as_str(module_v)
    name = self._as_str(name_v)
    if module is None:
        module = ""
    if name is None:
        name = ""
    self.push(self._resolve_ref(module, name))


_NONTUPLE_TAGS = {"str", "int", "bytes", "const", "none", "bool",
                  "list", "set", "dict", "ref", "partial", "attrgetter",
                  "methodcaller", "lazy"}


def _h_reduce(self, arg, pos):
    argtuple = self.pop()
    func = self.pop()
    if argtuple is not None and argtuple.tag in _NONTUPLE_TAGS:
        # REDUCE requires a tuple; the C unpickler raises before calling, so
        # nothing here executes and the load stops.
        raise _Stop()
    items = self.container_items(argtuple)
    if items is None:
        items = []
    self.push(self.do_call(func, items, pos))


def _h_inst(self, arg, pos):
    # INST module name : args are everything back to the mark
    args = self.pop_to_mark()
    s = arg if isinstance(arg, str) else str(arg)
    if " " in s:
        module, name = s.split(" ", 1)
    else:
        module, name = s, ""
    func = self._resolve_ref(module, name)
    self._construct(func, args, pos)


def _h_obj(self, arg, pos):
    items = self.pop_to_mark()
    if items:
        func = items[0]
        args = items[1:]
    else:
        func, args = UNKNOWN, []
    self._construct(func, args, pos)


def _construct(self, func, args, pos):
    # INST/OBJ: with arguments -> full construction (calls the callable);
    # argument-less on a class -> __new__ only (no finding), on a function
    # -> the function is called.
    if not args and func is not None and func.tag == "ref":
        role, kind = classify(func.path)
        if role in ("cap",) and _is_constructor(func.path):
            self.push(UNKNOWN)
            return
    self.push(self.do_call(func, args, pos))


Interp._construct = _construct


def _h_newobj(self, arg, pos):
    self.pop()  # args tuple
    self.pop()  # cls
    self.push(UNKNOWN)  # __new__ only: no user code for our capabilities


def _h_newobj_ex(self, arg, pos):
    self.pop()  # kwargs
    self.pop()  # args
    self.pop()  # cls
    self.push(UNKNOWN)


def _h_build(self, arg, pos):
    state = self.pop()
    obj = self.stack[-1] if self.stack else UNKNOWN
    if obj is not None and obj.tag == "partial":
        # __setstate__ on a partial retargets func/args from a tuple state.
        items = self.container_items(state)
        if items:
            obj.func = items[0]
            if len(items) > 1:
                a2 = self.container_items(items[1])
                obj.args = a2 if a2 is not None else (obj.args or [])
    elif obj is not None and obj.tag == "ref" and obj.is_module:
        # writing into a module __dict__ rebinds its attributes.
        if state is not None and state.tag == "dict":
            kv = state.args or []
            for i in range(0, len(kv) - 1, 2):
                k = self._as_str(kv[i])
                if k:
                    self.overrides[f"{obj.path}.{k}"] = kv[i + 1]
    # BUILD leaves the object on the stack.


def _h_memoize(self, arg, pos):
    if self.stack:
        self.memo[self._memo_auto] = self.stack[-1]
    self._memo_auto += 1


def _h_put(self, arg, pos):
    if self.stack:
        self.memo[arg] = self.stack[-1]
    try:
        self._memo_auto = max(self._memo_auto, int(arg) + 1)
    except (TypeError, ValueError):
        pass


def _h_get(self, arg, pos):
    self.push(self.memo.get(arg, UNKNOWN))


def _tuple_n(n):
    def h(self, arg, pos):
        items = [self.pop() for _ in range(n)][::-1] if n else []
        self.push(Val("tuple", args=items))
    return h


def _h_tuple(self, arg, pos):
    items = self.pop_to_mark()
    self.push(Val("tuple", args=items))


def _h_empty_tuple(self, arg, pos):
    self.push(Val("tuple", args=[]))


def _h_empty_list(self, arg, pos):
    self.push(Val("list", args=[]))


def _h_empty_dict(self, arg, pos):
    self.push(Val("dict", args=[]))


def _h_empty_set(self, arg, pos):
    self.push(Val("set", args=[]))


def _h_list(self, arg, pos):
    items = self.pop_to_mark()
    self.push(Val("list", args=items))


def _h_dict(self, arg, pos):
    items = self.pop_to_mark()
    self.push(Val("dict", args=items))


def _h_frozenset(self, arg, pos):
    items = self.pop_to_mark()
    self.push(Val("set", args=items))


def _h_append(self, arg, pos):
    v = self.pop()
    if self.stack and self.stack[-1] is not None and self.stack[-1].tag == "list":
        lst = self.stack[-1]
        (lst.args if lst.args is not None else []).append(v)
        if lst.args is None:
            lst.args = [v]


def _h_appends(self, arg, pos):
    items = self.pop_to_mark()
    if self.stack and self.stack[-1] is not None and self.stack[-1].tag in ("list", "set"):
        lst = self.stack[-1]
        if lst.args is None:
            lst.args = []
        lst.args.extend(items)


def _h_setitem(self, arg, pos):
    v = self.pop()
    k = self.pop()
    if self.stack and self.stack[-1] is not None and self.stack[-1].tag == "dict":
        d = self.stack[-1]
        if d.args is None:
            d.args = []
        d.args.extend([k, v])


def _h_setitems(self, arg, pos):
    items = self.pop_to_mark()
    if self.stack and self.stack[-1] is not None and self.stack[-1].tag == "dict":
        d = self.stack[-1]
        if d.args is None:
            d.args = []
        d.args.extend(items)


def _h_additems(self, arg, pos):
    items = self.pop_to_mark()
    if self.stack and self.stack[-1] is not None and self.stack[-1].tag == "set":
        s = self.stack[-1]
        if s.args is None:
            s.args = []
        s.args.extend(items)


def _h_persid(self, arg, pos):
    if self.persid_safe:
        self.push(Val("persid"))
    else:
        raise _Stop()


def _h_binpersid(self, arg, pos):
    self.pop()  # the pid object
    if self.persid_safe:
        self.push(Val("persid"))
    else:
        raise _Stop()


def _h_proto(self, arg, pos):
    try:
        self.proto = int(arg)
    except (TypeError, ValueError):
        pass


def _h_noop(self, arg, pos):
    pass


def _h_stop(self, arg, pos):
    pass


def _is_constructor(path):
    # capability callables that are classes whose __new__ alone is harmless
    return path in ("subprocess.Popen",) or path in policy.NETCTOR


_HANDLERS = {
    "PROTO": _h_proto, "FRAME": _h_noop, "STOP": _h_stop,
    "INT": _h_int, "BININT": _h_int, "BININT1": _h_int, "BININT2": _h_int,
    "LONG": _h_int, "LONG1": _h_int, "LONG4": _h_int,
    "FLOAT": _h_bool, "BINFLOAT": _h_bool,
    "NONE": _h_none, "NEWTRUE": _h_bool, "NEWFALSE": _h_bool,
    "STRING": _h_str, "BINSTRING": _h_str, "SHORT_BINSTRING": _h_str,
    "UNICODE": _h_str, "BINUNICODE": _h_str, "SHORT_BINUNICODE": _h_str,
    "BINUNICODE8": _h_str,
    "BINBYTES": _h_bytes, "SHORT_BINBYTES": _h_bytes, "BINBYTES8": _h_bytes,
    "BYTEARRAY8": _h_bytes,
    "NEXT_BUFFER": _h_push_unknown, "READONLY_BUFFER": _h_noop,
    "GLOBAL": _h_global, "STACK_GLOBAL": _h_stack_global,
    "REDUCE": _h_reduce, "INST": _h_inst, "OBJ": _h_obj,
    "NEWOBJ": _h_newobj, "NEWOBJ_EX": _h_newobj_ex, "BUILD": _h_build,
    "MARK": _h_mark, "POP": _h_pop, "POP_MARK": _h_pop_mark, "DUP": _h_dup,
    "MEMOIZE": _h_memoize, "PUT": _h_put, "BINPUT": _h_put, "LONG_BINPUT": _h_put,
    "GET": _h_get, "BINGET": _h_get, "LONG_BINGET": _h_get,
    "EMPTY_TUPLE": _h_empty_tuple, "TUPLE": _h_tuple,
    "TUPLE1": _tuple_n(1), "TUPLE2": _tuple_n(2), "TUPLE3": _tuple_n(3),
    "EMPTY_LIST": _h_empty_list, "LIST": _h_list, "APPEND": _h_append, "APPENDS": _h_appends,
    "EMPTY_DICT": _h_empty_dict, "DICT": _h_dict, "SETITEM": _h_setitem, "SETITEMS": _h_setitems,
    "EMPTY_SET": _h_empty_set, "FROZENSET": _h_frozenset, "ADDITEMS": _h_additems,
    "PERSID": _h_persid, "BINPERSID": _h_binpersid,
    "EXT1": _h_push_unknown, "EXT2": _h_push_unknown, "EXT4": _h_push_unknown,
}


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def analyse_streams(data, location, persid_safe=False, depth=0, torch_legacy=None,
                    fail_closed=False):
    """Walk every pickle stream in ``data`` until end of file.

    Returns a list of Findings.  Streams are read one after another (the raw
    pickle format and torch's legacy checkpoint are sequences of pickles);
    trailing bytes that are not a valid pickle are ignored, because the loader
    stops there.  ``fail_closed`` makes an unparseable *top-level* artifact
    report ``unsupported``; a nested member that will not parse is simply not
    a code-bearing member and stays clean.
    """
    if depth > MAX_DEPTH:
        return []
    if not data:
        return []
    bio = io.BytesIO(data if isinstance(data, (bytes, bytearray)) else bytes(data))
    size = len(data)
    # Detect torch legacy checkpoint (first stream is the magic number), which
    # makes persistent ids safe for the whole file.
    if torch_legacy is None:
        torch_legacy = _looks_torch_legacy(bio)
    persid_safe = persid_safe or torch_legacy

    findings: list[Finding] = []
    overrides: dict = {}
    stream_idx = 0
    while True:
        start = bio.tell()
        if start >= size:
            break
        interp = Interp(f"{location}#{stream_idx}", persid_safe, overrides, depth)
        try:
            interp.run(bio)
        except _Stop:
            findings.extend(interp.findings)
            break
        except Exception as exc:  # noqa: BLE001
            findings.extend(interp.findings)
            if fail_closed and stream_idx == 0 and not findings:
                # could not parse the submitted artifact at all: fail closed
                return [Finding("unsupported", location,
                                f"unparseable pickle stream: {type(exc).__name__}: {_trim(str(exc))}")]
            break
        findings.extend(interp.findings)
        stream_idx += 1
        if bio.tell() <= start:
            break
    return findings


def _looks_torch_legacy(bio):
    MAGIC = 0x1950A86A20F9469CFC6C
    pos = bio.tell()
    try:
        ops = []
        for op, arg, _p in pickletools.genops(bio):
            ops.append((op.name, arg))
            if op.name == "STOP":
                break
        if len(ops) >= 2 and ops[0][0] == "PROTO":
            for name, arg in ops:
                if name in ("LONG1", "LONG", "LONG4", "BININT", "INT") and arg == MAGIC:
                    return True
    except Exception:  # noqa: BLE001
        pass
    finally:
        bio.seek(pos)
    return False


def _split_resolve(name):
    # pkgutil.resolve_name accepts "pkg.mod:attr" or "pkg.mod.attr"
    if ":" in name:
        mod, _, attr = name.partition(":")
        return mod, attr
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return name, ""


def _trim(s, n=160):
    s = str(s)
    return s if len(s) <= n else s[:n] + "..."
