"""Pickle stream scanning by abstract interpretation.

Rather than match names textually, we walk the opcodes keeping a shadow of the
unpickler's stack and memo, tracking only the things that matter for the
policy: references to globals, imported modules, the results of
getattr/attrgetter/methodcaller/partial, bytes/str literals, and tuples (so a
REDUCE's argument list can be read).  A finding is raised only when a
capability-bearing callable is actually *called* -- through REDUCE, an
argument-bearing INST/OBJ, a methodcaller/partial, or a retargeted partial --
never when a dangerous name is merely stored as data.

The interpreter also resolves the indirections the policy calls out (module
aliases, the memo, getattr chains, operator.attrgetter / methodcaller,
functools.partial, __import__/import_module) and follows nested loaders into
embedded data, seeing through pure byte transforms (base64 / compression).
"""
from __future__ import annotations

import io
import pickletools

from modelgate import policy
from modelgate.report import Finding

# Abstract values are small tagged tuples (fast, hashable-ish, no execution):
#   ("g", module, name)  a global reference  module.name
#   ("mod", module)      an imported module object
#   ("ag", (attr, ...))  operator.attrgetter(attr, ...)
#   ("mc", name)         operator.methodcaller(name, ...)
#   ("pa", target)       functools.partial(target, ...)  (target is a value)
#   ("by", bytes)        a bytes literal (or BytesIO/str wrapping known bytes)
#   ("st", str)          a str literal
#   ("tup", [values])    a tuple (so REDUCE args can be inspected)
#   ("op",)              anything else (opaque data)
OPAQUE = ("op",)

_STRING_OPS = {
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
    "STRING", "SHORT_BINSTRING", "BINSTRING",
}
_BYTES_OPS = {"SHORT_BINBYTES", "BINBYTES", "BINBYTES8", "BYTEARRAY8"}

_MAX_DEPTH = 6
_MAX_OPS = 2_000_000


class _Mark:
    __slots__ = ()


MARK = _Mark()


def scan_pickle(data: bytes, location: str, depth: int = 0) -> list[Finding]:
    """Scan a buffer that may hold one or several concatenated pickle streams."""
    findings: list[Finding] = []
    if depth > _MAX_DEPTH:
        return [Finding("unsupported", location, "pickle nesting too deep")]
    pos = 0
    n = len(data)
    any_stopped = False
    while pos < n:
        consumed, stopped = _scan_stream(data, pos, location, depth, findings)
        any_stopped = any_stopped or stopped
        if consumed <= 0:
            break
        pos += consumed
        if not stopped:
            # a stream that did not reach STOP ends the sequence: the real
            # loader (repeated pickle.load) would stop here too
            break
    if depth == 0 and not findings and not any_stopped:
        # Nothing matched any format and no complete pickle stream was found:
        # this is not an artifact we can analyse.  The policy is fail-closed.
        return [Finding("unsupported", location,
                        "not a recognised artifact format")]
    return findings


def _scan_stream(data, start, location, depth, findings):
    """Interpret one pickle stream starting at ``start``.

    Returns (bytes_consumed, reached_stop).  Appends to ``findings``.
    """
    bio = io.BytesIO(data)
    bio.seek(start)
    stack: list = []
    memo: dict = {}
    ops = 0
    stopped = False
    try:
        for opcode, arg, pos in pickletools.genops(bio):
            ops += 1
            if ops > _MAX_OPS:
                break
            name = opcode.name
            loc = f"{location}@{pos}"
            if name in _STRING_OPS:
                stack.append(("st", arg if isinstance(arg, str) else str(arg)))
            elif name in _BYTES_OPS:
                stack.append(("by", arg if isinstance(arg, (bytes, bytearray)) else b""))
            elif name == "MARK":
                stack.append(MARK)
            elif name == "STOP":
                stopped = True
                break
            elif name == "GLOBAL":
                mod, _, nm = str(arg).partition(" ")
                stack.append(("g", mod, nm))
            elif name == "STACK_GLOBAL":
                nm = stack.pop() if stack else OPAQUE
                mod = stack.pop() if stack else OPAQUE
                if mod[0] == "st" and nm[0] == "st":
                    stack.append(("g", mod[1], nm[1]))
                else:
                    stack.append(OPAQUE)
            elif name == "REDUCE":
                args = stack.pop() if stack else OPAQUE
                func = stack.pop() if stack else OPAQUE
                stack.append(_call(func, args, loc, depth, findings))
            elif name in ("INST", "OBJ"):
                items = _pop_to_mark(stack)
                if name == "INST":
                    mod, _, nm = str(arg).partition(" ")
                    func = ("g", mod, nm)
                    call_args = items
                else:  # OBJ: class is first item on the mark slice
                    func = items[0] if items else OPAQUE
                    call_args = items[1:]
                # INST/OBJ with arguments construct fully (run __init__);
                # argument-less ones call __new__ only and execute nothing.
                if call_args:
                    stack.append(_call(func, ("tup", call_args), loc, depth, findings))
                else:
                    stack.append(OPAQUE)
            elif name == "NEWOBJ":
                if stack:
                    stack.pop()  # args
                if stack:
                    stack.pop()  # cls
                stack.append(OPAQUE)
            elif name == "NEWOBJ_EX":
                for _ in range(3):
                    if stack:
                        stack.pop()
                stack.append(OPAQUE)
            elif name == "BUILD":
                state = stack.pop() if stack else OPAQUE
                target = stack[-1] if stack else OPAQUE
                _build(target, state, loc, depth, findings)
            elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
                k = int(name[-1])
                vals = []
                for _ in range(k):
                    vals.append(stack.pop() if stack else OPAQUE)
                vals.reverse()
                stack.append(("tup", vals))
            elif name == "TUPLE":
                vals = _pop_to_mark(stack)
                stack.append(("tup", vals))
            elif name == "EMPTY_TUPLE":
                stack.append(("tup", []))
            elif name in ("EMPTY_LIST", "EMPTY_DICT", "EMPTY_SET"):
                stack.append(OPAQUE)
            elif name in ("LIST", "DICT", "FROZENSET"):
                _pop_to_mark(stack)
                stack.append(OPAQUE)
            elif name == "APPEND":
                if stack:
                    stack.pop()
            elif name == "APPENDS":
                _pop_to_mark(stack)
            elif name == "SETITEM":
                for _ in range(2):
                    if stack:
                        stack.pop()
            elif name in ("SETITEMS", "ADDITEMS"):
                _pop_to_mark(stack)
            elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
                if stack:
                    memo[int(arg)] = stack[-1]
            elif name == "MEMOIZE":
                if stack:
                    memo[len(memo)] = stack[-1]
            elif name in ("GET", "BINGET", "LONG_BINGET"):
                stack.append(memo.get(int(arg), OPAQUE))
            elif name == "POP":
                if stack:
                    stack.pop()
            elif name == "POP_MARK":
                _pop_to_mark(stack)
            elif name == "DUP":
                if stack:
                    stack.append(stack[-1])
            elif name in ("PERSID", "BINPERSID"):
                if name == "BINPERSID" and stack:
                    stack.pop()
                stack.append(OPAQUE)  # resolved by the loader's storage; executes nothing
            elif name in ("PROTO", "FRAME"):
                pass
            elif name in ("NONE", "NEWTRUE", "NEWFALSE", "INT", "BININT",
                          "BININT1", "BININT2", "LONG", "LONG1", "LONG4",
                          "FLOAT", "BINFLOAT"):
                stack.append(OPAQUE)
            elif name in ("EXT1", "EXT2", "EXT4"):
                stack.append(OPAQUE)  # copyreg extension registry; not in scope
            elif name == "NEXT_BUFFER":
                stack.append(OPAQUE)  # protocol-5 out-of-band buffer (data)
            elif name == "READONLY_BUFFER":
                if stack:
                    stack.pop()
                stack.append(OPAQUE)
            else:
                # Unknown opcode: keep the stack roughly balanced.
                _generic(opcode, stack)
    except Exception:  # noqa: BLE001
        # Truncated or malformed: keep whatever findings we reached (the loader
        # would reach the same opcodes before raising).  Whether this counts as
        # an unanalysable artifact is decided by scan_pickle (fail-closed only
        # when no complete stream was found at the top level).
        return bio.tell() - start, stopped
    return bio.tell() - start, stopped


def _pop_to_mark(stack: list) -> list:
    """Pop values back to (and including) the last MARK; return them in order."""
    items: list = []
    while stack:
        v = stack.pop()
        if v is MARK:
            items.reverse()
            return items
        items.append(v)
    items.reverse()
    return items


def _generic(opcode, stack):
    before = getattr(opcode, "stack_before", []) or []
    after = getattr(opcode, "stack_after", []) or []
    has_mark = any(getattr(d, "name", "") == "mark" for d in before)
    if has_mark:
        _pop_to_mark(stack)
    else:
        for _ in range(len(before)):
            if stack:
                stack.pop()
    for _ in range(max(1, len(after)) if after else 1):
        stack.append(OPAQUE)


def _args_list(args):
    if isinstance(args, tuple) and args and args[0] == "tup":
        return args[1]
    return None


def _call(func, args, loc, depth, findings):
    """Model calling ``func`` with ``args`` (a tuple value). Return result value."""
    arglist = _args_list(args)
    if func[0] == "g":
        return _call_global(func[1], func[2], arglist, loc, depth, findings)
    if func[0] == "ag":
        # attrgetter(attrs)(obj): retrieves attributes, does not call them.
        if arglist:
            return _apply_attr(arglist[0], func[1])
        return OPAQUE
    if func[0] == "mc":
        # methodcaller(name, ...)(obj): CALLS obj.name(...) -> finding if capability
        if arglist:
            target = _apply_attr(arglist[0], (func[1],))
            if target[0] == "g":
                _flag_global(target[1], target[2], True, loc, findings, via="methodcaller")
        return OPAQUE
    if func[0] == "pa":
        # calling a partial calls its (possibly retargeted) target
        _invoke_value(func[1][0], True, loc, depth, findings, via="partial")
        return OPAQUE
    return OPAQUE


def _invoke_value(val, has_args, loc, depth, findings, via=""):
    if val[0] == "g":
        _flag_global(val[1], val[2], has_args, loc, findings, via=via)
    elif val[0] == "pa":
        _invoke_value(val[1][0], has_args, loc, depth, findings, via=via)
    elif val[0] == "mc":
        pass


def _apply_attr(obj, attrs):
    """getattr(obj, a) for each a in attrs (dotted names allowed)."""
    cur = obj
    for a in attrs:
        parts = a.split(".") if isinstance(a, str) else [a]
        for p in parts:
            if cur[0] == "mod":
                cur = ("g", cur[1], p)
            elif cur[0] == "g":
                cur = ("g", cur[1], cur[2] + "." + p)
            else:
                return OPAQUE
    return cur


def _call_global(module, name, arglist, loc, depth, findings):
    mod, nm = policy.resolve(module, name)
    full = f"{mod}.{nm}"

    indirection = policy.INDIRECTION.get(full)
    if indirection:
        return _indirect(indirection, arglist, loc)

    if full in policy.BYTESIO:
        if arglist and arglist[0][0] == "by":
            return ("by", arglist[0][1])
        return OPAQUE

    if full in policy.BYTE_TRANSFORMS:
        return _transform(full, arglist)

    nested = policy.NESTED_LOADERS.get(full)
    if nested:
        _nested_loader(full, nested, arglist, loc, depth, findings)
        return OPAQUE

    _flag_global(mod, nm, True if arglist is None else bool(arglist), loc,
                 findings, args_seen=arglist is not None)
    return OPAQUE


def _flag_global(module, name, has_positional, loc, findings, via="", args_seen=True):
    mod, nm = policy.resolve(module, name)
    full = f"{mod}.{nm}"
    cap = policy.CAPABILITIES.get(full)
    if not cap:
        return
    kind, mode = cap
    if mode == "needarg":
        # finding only when a positional argument is supplied (host given).
        if args_seen and not has_positional:
            return
    detail = f"call {full}"
    if via:
        detail += f" via {via}"
    findings.append(Finding(kind, loc, detail))


def _indirect(kind, arglist, loc):
    if kind == "getattr":
        if arglist and len(arglist) >= 2 and arglist[1][0] == "st":
            return _apply_attr(arglist[0], (arglist[1][1],))
        return OPAQUE
    if kind == "import":
        if arglist and arglist[0][0] == "st":
            return ("mod", arglist[0][1])
        return OPAQUE
    if kind == "attrgetter":
        attrs = tuple(a[1] for a in (arglist or []) if a[0] == "st")
        return ("ag", attrs)
    if kind == "methodcaller":
        if arglist and arglist[0][0] == "st":
            return ("mc", arglist[0][1])
        return OPAQUE
    if kind == "partial":
        if arglist:
            # mutable 1-cell so a later BUILD can retarget the partial in place
            return ("pa", [arglist[0]])
        return ("pa", [OPAQUE])
    return OPAQUE


def _transform(full, arglist):
    if not arglist or arglist[0][0] not in ("by", "st"):
        return OPAQUE
    raw = arglist[0][1]
    try:
        import base64 as _b64
        import binascii as _ba
        if full in ("zlib.decompress",):
            import zlib
            out = zlib.decompress(raw if isinstance(raw, bytes) else raw.encode(), 47)
        elif full == "gzip.decompress":
            import gzip
            out = gzip.decompress(raw if isinstance(raw, bytes) else raw.encode())
        elif full == "bz2.decompress":
            import bz2
            out = bz2.decompress(raw if isinstance(raw, bytes) else raw.encode())
        elif full == "lzma.decompress":
            import lzma
            out = lzma.decompress(raw if isinstance(raw, bytes) else raw.encode())
        elif full in ("base64.b64decode", "base64.standard_b64decode", "binascii.a2b_base64"):
            out = _b64.b64decode(raw)
        elif full == "base64.urlsafe_b64decode":
            out = _b64.urlsafe_b64decode(raw)
        elif full == "base64.b32decode":
            out = _b64.b32decode(raw)
        elif full == "base64.b16decode":
            out = _b64.b16decode(raw)
        elif full == "base64.a85decode":
            out = _b64.a85decode(raw)
        elif full == "base64.b85decode":
            out = _b64.b85decode(raw)
        elif full == "base64.decodebytes":
            out = _b64.decodebytes(raw if isinstance(raw, bytes) else raw.encode())
        elif full in ("binascii.unhexlify", "binascii.a2b_hex", "bytes.fromhex"):
            out = _ba.unhexlify(raw)
        elif full == "codecs.decode":
            enc = (arglist[1][1] if len(arglist) > 1 and arglist[1][0] == "st" else "").lower()
            # Only evaluate known byte-transform codecs, so attacker-controlled
            # data can never make the scanner import an arbitrary codec module.
            if enc.replace("-", "_") not in {
                    "hex", "hex_codec", "base64", "base64_codec", "zlib", "zlib_codec",
                    "bz2", "bz2_codec", "uu", "uu_codec", "quopri", "quopri_codec"}:
                return OPAQUE
            import codecs
            out = codecs.decode(raw if isinstance(raw, (bytes, bytearray)) else raw, enc)
            if isinstance(out, str):
                out = out.encode("latin1", "ignore")
        else:
            return OPAQUE
        if isinstance(out, (bytes, bytearray)) and len(out) <= 64 * 1024 * 1024:
            return ("by", bytes(out))
    except Exception:  # noqa: BLE001
        return OPAQUE
    return OPAQUE


def _nested_loader(full, mode, arglist, loc, depth, findings):
    arg0 = arglist[0] if arglist else None
    if arg0 is not None and arg0[0] == "by":
        # embedded data: judged by what it would load
        from modelgate import dispatch
        sub = dispatch.scan_bytes(arg0[1], f"{loc}->{full}", depth + 1)
        for f in sub:
            findings.append(Finding(
                "code-execution" if f.kind == "unsupported" else f.kind,
                f.location, f"via {full}: {f.detail}"))
        return
    if mode == "pathorbytes" and arg0 is not None and arg0[0] == "st":
        # a loader handed a filesystem path reads that path
        findings.append(Finding("file-access", loc, f"{full} reads path {arg0[1]!r}"))
        return
    # argument not resolvable: a re-entrant deserializer on data we cannot see.
    findings.append(Finding("code-execution", loc,
                            f"{full} on unresolved data (re-entrant deserialization)"))


def _build(target, state, loc, depth, findings):
    """BUILD calls __setstate__; model functools.partial being retargeted.

    partial.__setstate__ receives (func, args, kwds, namespace); a later call to
    the partial then runs ``func``.  The partial is held in a mutable cell so the
    retarget is visible when the value is called downstream.
    """
    if target[0] != "pa":
        return
    sl = _args_list(state)
    newfunc = sl[0] if sl else None
    if isinstance(newfunc, tuple) and newfunc and newfunc[0] in ("g", "pa", "mc"):
        target[1][0] = newfunc
