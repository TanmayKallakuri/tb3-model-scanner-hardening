"""Abstract interpretation of pickle protocols 0–5. No unpickler is used.

The stack and memo contain only primitives and symbolic objects. Executing a
pickle opcode means applying its *abstract* semantics, never its Python target.
"""
from __future__ import annotations

import io
import pickletools
import math
import re

from modelgate.symbolic import Context, Symbol, UNKNOWN, LoadError, AnalysisError
from modelgate.limits import LimitError, MAX_BYTES
from modelgate.report import Finding

MARK = object()
TORCH_MAGIC = 0x1950a86a20f9469cfc6c
STRING_OPS = {"STRING", "BINSTRING", "SHORT_BINSTRING", "UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"}
NUMBER_OPS = {"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT"}


def dtype_info(dtype):
    """Object flag and byte width, sufficient to skip joblib inline arrays."""
    if isinstance(dtype, Symbol):
        desc = dtype.args[0] if dtype.args else None
        state = dtype.attrs.get("state")
        if isinstance(state, tuple) and len(state) >= 6:
            fields = state[4]
            if isinstance(fields, dict):
                infos = [dtype_info(v[0]) for v in fields.values() if isinstance(v, tuple) and v]
                return any(v[0] for v in infos), max(0, state[5])
        return dtype_info(desc)
    if isinstance(dtype, str):
        s = dtype.lstrip("<>=|")
        if s in ("object", "object_", "O"): return True, 8
        widths = {"bool": 1, "bool_": 1, "?": 1, "b": 1, "B": 1, "h": 2, "H": 2, "i": 4, "I": 4, "l": 8, "L": 8, "q": 8, "Q": 8, "f": 4, "d": 8, "g": 16, "F": 8, "D": 16}
        if s in widths: return False, widths[s]
        match = re.fullmatch(r"([A-Za-z?]+)(\d+)(?:\[.*\])?", s)
        if match:
            kind, n = match[1], int(match[2])
            if kind in ("int", "uint", "float", "complex"): n //= 8
            if kind == "U": n *= 4
            return kind == "O", n
    if isinstance(dtype, list):
        infos = [dtype_info(v[1]) for v in dtype if isinstance(v, tuple) and len(v) >= 2]
        return any(v[0] for v in infos), sum(v[1] for v in infos)
    if isinstance(dtype, tuple) and len(dtype) == 2:
        hasobject, n = dtype_info(dtype[0])
        return hasobject, n * (math.prod(dtype[1]) if isinstance(dtype[1], tuple) else dtype[1])
    raise AnalysisError("cannot determine array dtype")


def _joblib_array(obj, fh, ctx, location):
    hasobject, width = dtype_info(obj.attrs.get("dtype"))
    if hasobject:
        pos = fh.tell()
        value, end, ok = analyze_pickle(fh.getvalue()[pos:], location + "/object-array", ctx, streams=1)
        fh.seek(pos + end)
        if not ok: raise LoadError("invalid joblib object array")
        obj.kind, obj.name = "instance", "numpy.ndarray"
        obj.attrs["data"] = value
    else:
        count = math.prod(obj.attrs.get("shape", ()))
        size = count * width
        if size < 0 or size > MAX_BYTES: raise LimitError("joblib array size limit")
        if obj.attrs.get("numpy_array_alignment_bytes") is not None:
            b = fh.read(1)
            if not b: raise AnalysisError("truncated joblib alignment")
            fh.seek(b[0], 1)
        end = fh.tell() + size
        if end > len(fh.getbuffer()): raise AnalysisError("truncated joblib array")
        fh.seek(end)
        obj.kind, obj.name = "instance", "numpy.ndarray"


def analyze_pickle(data: bytes, location: str, ctx: Context, *, streams=None,
                   persistent=False, joblib=True, torch=False):
    """Return (abstract result, consumed byte count, successful load).

    streams=None follows the registry's sequence-of-pickles convention. Nested
    pickle.load(s) instead passes streams=1. Legacy torch storage bytes are
    never parsed as another pickle.
    """
    fh = io.BytesIO(data)
    result = UNKNOWN
    stream = 0
    legacy = torch
    while fh.tell() < len(data) and (streams is None or stream < streams):
        stack, memo = [], {}
        protocol = 0
        start = fh.tell()
        stopped = False
        frame_end = None

        def pop():
            if not stack or stack[-1] is MARK: raise LoadError("pickle stack underflow")
            return stack.pop()

        def markitems():
            i = len(stack) - 1
            while i >= 0 and stack[i] is not MARK: i -= 1
            if i < 0: raise LoadError("missing pickle MARK")
            values = stack[i + 1:]
            del stack[i:]
            return values

        try:
            for opcode, arg, pos in pickletools.genops(fh):
                ctx.budget.tick()
                op = opcode.name
                ctx.loc = f"{location}:stream[{stream}]@{pos}"
                ctx.op = op
                if frame_end is not None and pos >= frame_end: frame_end = None
                if op == "PROTO":
                    if arg > 5: raise AnalysisError("unsupported pickle protocol")
                    protocol = arg
                elif op == "FRAME":
                    if frame_end is not None and pos < frame_end: raise LoadError("nested pickle frame")
                    frame_end = fh.tell() + arg
                    if frame_end > len(data): raise LoadError("truncated pickle frame")
                elif op in STRING_OPS or op in NUMBER_OPS or op in ("BINBYTES", "SHORT_BINBYTES", "BINBYTES8"):
                    stack.append(arg)
                elif op == "BYTEARRAY8": stack.append(bytearray(arg))
                elif op == "NONE": stack.append(None)
                elif op == "NEWTRUE": stack.append(True)
                elif op == "NEWFALSE": stack.append(False)
                elif op == "MARK": stack.append(MARK)
                elif op == "EMPTY_TUPLE": stack.append(())
                elif op == "EMPTY_LIST": stack.append([])
                elif op == "EMPTY_DICT": stack.append({})
                elif op == "EMPTY_SET": stack.append(set())
                elif op in ("TUPLE", "LIST", "DICT", "FROZENSET"):
                    values = markitems()
                    if op == "TUPLE": stack.append(tuple(values))
                    elif op == "LIST": stack.append(values)
                    elif op == "FROZENSET": stack.append(frozenset(values))
                    else:
                        if len(values) % 2: raise LoadError("odd DICT items")
                        stack.append(dict(zip(values[::2], values[1::2])))
                elif op in ("TUPLE1", "TUPLE2", "TUPLE3"):
                    n = int(op[-1])
                    stack.append(tuple(reversed([pop() for _ in range(n)])))
                elif op in ("APPEND", "APPENDS", "ADDITEMS"):
                    values = [pop()] if op == "APPEND" else markitems()
                    if not stack: raise LoadError("missing container")
                    target = stack[-1]
                    if type(target) is list: target.extend(values)
                    elif type(target) is set: target.update(values)
                    elif isinstance(target, Symbol):
                        for v in values: ctx.call(ctx.attr(target, "add" if op == "ADDITEMS" else "append"), (v,))
                    else: raise LoadError("invalid container")
                elif op in ("SETITEM", "SETITEMS"):
                    if op == "SETITEM":
                        value, key = pop(), pop()
                        values = [key, value]
                    else: values = markitems()
                    if len(values) % 2 or not stack: raise LoadError("invalid SETITEMS")
                    for k, v in zip(values[::2], values[1::2]): ctx.setitem(stack[-1], k, v)
                elif op == "POP":
                    if not stack: raise LoadError("empty POP")
                    stack.pop()
                elif op == "POP_MARK": markitems()
                elif op == "DUP":
                    if not stack: raise LoadError("empty DUP")
                    stack.append(stack[-1])
                elif op in ("PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE"):
                    if not stack: raise LoadError("empty memo assignment")
                    memo[len(memo) if op == "MEMOIZE" else int(arg)] = stack[-1]
                elif op in ("GET", "BINGET", "LONG_BINGET"):
                    if int(arg) not in memo: raise LoadError("unknown memo index")
                    stack.append(memo[int(arg)])
                elif op in ("GLOBAL", "STACK_GLOBAL"):
                    if op == "GLOBAL": module, name = arg.split(" ", 1)
                    else: name, module = pop(), pop()
                    stack.append(ctx.global_ref(module, name, protocol))
                elif op == "REDUCE":
                    args, target = pop(), pop()
                    stack.append(ctx.call(target, args))
                elif op in ("NEWOBJ", "NEWOBJ_EX"):
                    kwargs = pop() if op == "NEWOBJ_EX" else None
                    args, target = pop(), pop()
                    stack.append(ctx.newobj(target, args, kwargs))
                elif op in ("INST", "OBJ"):
                    values = markitems()
                    if op == "INST":
                        module, name = arg.split(" ", 1)
                        target, args = ctx.global_ref(module, name, protocol), tuple(values)
                    else:
                        if not values: raise LoadError("empty OBJ")
                        target, args = values[0], tuple(values[1:])
                    stack.append(ctx.newobj(target, ()) if not args and ctx.is_class(target) else ctx.call(target, args))
                elif op == "BUILD":
                    state = pop()
                    if not stack: raise LoadError("empty BUILD")
                    target = stack[-1]
                    ctx.build(target, state)
                    if joblib and isinstance(target, Symbol) and target.name in ("joblib.numpy_pickle.NumpyArrayWrapper", "joblib.numpy_pickle_compat.NDArrayWrapper"):
                        if target.name.endswith("NDArrayWrapper"):
                            ctx.finding("file-access", "joblib array stored in an external file")
                        else:
                            # Joblib commits its pickle frame before writing
                            # array bytes directly to the underlying stream.
                            if frame_end == fh.tell(): frame_end = None
                            _joblib_array(target, fh, ctx, ctx.loc)
                elif op in ("PERSID", "BINPERSID"):
                    if op == "BINPERSID": pop()
                    if not (persistent or legacy): raise LoadError("no persistent-id handler")
                    stack.append(Symbol("instance", "torch.Storage"))
                elif op in ("EXT1", "EXT2", "EXT4"):
                    raise AnalysisError("pickle extension registry is not supplied by the artifact")
                elif op == "NEXT_BUFFER":
                    raise LoadError("out-of-band buffers were not supplied")
                elif op == "READONLY_BUFFER":
                    if not stack: raise LoadError("missing buffer")
                elif op == "STOP":
                    result = pop()
                    stopped = True
                    break
                else:
                    raise AnalysisError("unmodelled pickle opcode " + op)
                if frame_end is not None and fh.tell() > frame_end:
                    raise LoadError("opcode crosses a pickle frame")
            if not stopped: raise AnalysisError("pickle has no STOP")
        except LoadError:
            return result, fh.tell(), False
        except (AnalysisError, LimitError) as exc:
            ctx.finding("unsupported", str(exc))
            return result, fh.tell(), False
        except Exception as exc:
            ctx.finding("unsupported", f"cannot parse pickle: {type(exc).__name__}: {exc}")
            return result, fh.tell(), False
        stream += 1
        if streams is None and stream == 1 and isinstance(result, int) and not isinstance(result, bool) and result == TORCH_MAGIC:
            legacy = True
            streams = 5
        if stream == 4 and legacy: legacy_object = result
        if stream >= 5 and legacy:
            return locals().get("legacy_object", result), fh.tell(), True
        if fh.tell() <= start: raise AnalysisError("pickle parser made no progress")
    return result, fh.tell(), True


def scan_pickle(data: bytes, location: str, **kwargs) -> list[Finding]:
    ctx = Context()
    analyze_pickle(data, location, ctx, **kwargs)
    return ctx.findings
