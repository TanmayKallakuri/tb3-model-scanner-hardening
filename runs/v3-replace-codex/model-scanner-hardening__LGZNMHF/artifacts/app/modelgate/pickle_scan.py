"""A non-executing abstract interpreter for Python pickle opcodes.

Only the pickle syntax parser is used. Globals, instances, calls, iterators,
modules and buffers are scanner-owned symbolic values; no unpickler, import
from an artifact, or callable named by an artifact is ever run.
"""
from __future__ import annotations

import io
import os
import pickletools
import time

from modelgate.capabilities import CLASS_NAMES, canonical
from modelgate.pickle_vm import Calls
from modelgate.report import Finding
from modelgate.symbols import (Bound, Buffer, Code, Global, Instance, MARK, MISSING,
                               Module, Namespace, Partial, Unknown, Wrapper)

START_BYTES = {op.code.encode("latin1") for op in pickletools.opcodes} - {b"\x00"}
TORCH_MAGIC = 119547037146038801333356


class LoadError(Exception):
    """A known loader error: execution stops, so subsequent bytes are inert."""


class AnalysisError(Exception):
    """A construct whose behaviour cannot be determined statically."""


class Interpreter(Calls):
    def __init__(self, artifact_path=None):
        self.findings = []
        self.modules = {}
        self.globals = {}
        self.location = "pickle"
        self.protocol = 0
        self.artifact_path = os.path.abspath(artifact_path) if artifact_path else None
        self.artifact_dir = bool(artifact_path and os.path.isdir(artifact_path))
        self.deadline = time.monotonic() + 35
        self.steps = 0
        self.depth = 0
        self.call_depth = 0
        self.LoadError = LoadError
        self.AnalysisError = AnalysisError

    def check_budget(self):
        self.steps += 1
        if self.steps > 2000000 or time.monotonic() > self.deadline:
            raise AnalysisError("static analysis resource limit")

    def finding(self, kind, detail):
        if not self.findings:
            self.findings.append(Finding(kind, self.location, detail))

    def module(self, name):
        name = canonical(name)
        if name not in self.modules:
            self.modules[name] = Module(name)
        return self.modules[name]

    def ref(self, name):
        if name not in self.globals:
            self.globals[name] = Global(name)
        return self.globals[name]

    def global_ref(self, module, name):
        module, name = canonical(module, name, self.protocol)
        if "." in name and self.protocol < 4:
            raise LoadError("dotted globals require protocol 4")
        obj = self.module(module)
        for part in name.split("."):
            obj = self.attr(obj, part)
        return obj

    def is_class(self, value):
        if not isinstance(value, Global):
            return False
        path = value.name
        return path in CLASS_NAMES or path.rsplit(".", 1)[-1][:1].isupper() or path in {
            "builtins.object", "builtins.type", "builtins.memoryview", "builtins.property",
        }

    def construct(self, cls, args, kwargs=None, new_only=False):
        kwargs = kwargs or {}
        if not isinstance(args, tuple) or not isinstance(kwargs, dict):
            raise LoadError("invalid construction arguments")
        if not new_only:
            return self.call(cls, args, kwargs)
        if not self.is_class(cls):
            raise LoadError("NEWOBJ requires a type")
        name = cls.name
        if name == "functools.partial":
            return self.make_partial(args, kwargs)
        if name in ("types.FunctionType", "builtins.function"):
            return self.call(cls, args, kwargs)
        if name == "builtins.object":
            return Instance(name)
        if name in ("builtins.tuple", "builtins.list", "builtins.dict", "builtins.set", "builtins.frozenset", "builtins.str", "builtins.bytes", "builtins.bytearray", "builtins.int", "builtins.float", "builtins.complex"):
            return self.call(cls, args, kwargs)
        return Instance(name, args)

    def build(self, obj, state):
        if isinstance(obj, Partial):
            if not isinstance(state, tuple) or len(state) != 4:
                raise LoadError("invalid partial state")
            func, args, kwargs, attrs = state
            if not isinstance(args, tuple) or not isinstance(kwargs, dict):
                raise LoadError("invalid partial state")
            obj.func, obj.args, obj.kwargs = func, args, dict(kwargs)
            obj.attrs = attrs if isinstance(attrs, dict) else {}
            return
        if isinstance(obj, Instance) and obj.cls == "numpy.dtype":
            obj.attrs["state"] = state
            return
        setter = self.attr(obj, "__setstate__", default=MISSING)
        if setter is not MISSING and not (isinstance(setter, Bound) and isinstance(obj, Instance)):
            self.call(setter, (state,))
            return
        if isinstance(obj, (Instance, Module, Global, Wrapper, Buffer)):
            slots = None
            if isinstance(state, tuple) and len(state) == 2:
                state, slots = state
            if isinstance(state, dict):
                if isinstance(obj, Buffer):
                    # BytesIO's native state: (contents, position, dict).
                    return
                obj.attrs.update(state)
            if isinstance(slots, dict) and hasattr(obj, "attrs"):
                obj.attrs.update(slots)
            if isinstance(obj, Instance):
                obj.attrs.setdefault("_pickle_state", state)
                if obj.cls.startswith("joblib.") and obj.cls.endswith("ArrayWrapper"):
                    obj.attrs["_joblib_array"] = True
            return
        if isinstance(obj, Buffer) and isinstance(state, tuple):
            obj.data, obj.pos = state[:2]
            return
        # Built-in immutable values have no __dict__ and reject BUILD.
        raise LoadError("object cannot accept pickle state")

    def load(self, data: bytes, location: str, *, all_streams=True, persistent=False, abort_on_error=False):
        self.depth += 1
        if self.depth > 24:
            self.depth -= 1
            raise AnalysisError("nested deserialization limit")
        previous = self.location, self.protocol
        stream = io.BytesIO(data)
        result = None
        count = 0
        legacy = False
        saved_object = None
        try:
            while stream.tell() < len(data):
                self.check_budget()
                start = stream.tell()
                self.location = f"{location}:stream[{count}]"
                result = self.run_stream(stream, persistent=persistent or (legacy and count == 3))
                count += 1
                if count == 1 and result == TORCH_MAGIC:
                    legacy = True
                if legacy and count == 4:
                    saved_object = result
                if self.findings or not all_streams or (legacy and count == 5):
                    break
                if stream.tell() <= start:
                    raise AnalysisError("pickle parser made no progress")
            return saved_object if legacy and count == 5 else result
        except LoadError:
            # Unlike an unknown operation, an operation known to raise cannot
            # execute a capability in later opcodes or later streams.
            if abort_on_error:
                raise
            return Unknown("load stopped with a known error")
        except Exception as exc:
            self.finding("unsupported", f"cannot analyse pickle: {type(exc).__name__}: {exc}")
            return Unknown(str(exc))
        finally:
            self.location, self.protocol = previous
            self.depth -= 1

    def run_stream(self, stream, *, persistent=False):
        stack = []
        memo = {}
        self.protocol = 0
        base_location = self.location
        frame_end = None

        def marked():
            for index in range(len(stack) - 1, -1, -1):
                if stack[index] is MARK:
                    values = stack[index + 1:]
                    del stack[index:]
                    return values
            raise LoadError("missing MARK")

        def pairs(values):
            if len(values) % 2:
                raise LoadError("odd dictionary items")
            return zip(values[::2], values[1::2])

        for op, arg, pos in pickletools.genops(stream):
            self.check_budget()
            name = op.name
            self.location = f"{base_location}@{pos} ({name})"
            end = stream.tell()
            if frame_end is not None:
                if pos < frame_end and end > frame_end:
                    raise LoadError("opcode crosses pickle frame")
                if pos >= frame_end:
                    frame_end = None
            if name == "PROTO":
                if arg > 5:
                    raise LoadError("unsupported pickle protocol")
                self.protocol = arg
            elif name == "FRAME":
                if frame_end is not None:
                    raise LoadError("nested pickle frame")
                frame_end = end + arg
                if frame_end > len(stream.getbuffer()):
                    raise LoadError("truncated pickle frame")
            elif name == "MARK":
                stack.append(MARK)
            elif name == "POP":
                stack.pop()
            elif name == "POP_MARK":
                marked()
            elif name == "DUP":
                stack.append(stack[-1])
            elif name in {"NONE", "NEWTRUE", "NEWFALSE"}:
                stack.append({"NONE": None, "NEWTRUE": True, "NEWFALSE": False}[name])
            elif name in {"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT", "UNICODE", "BINUNICODE", "BINUNICODE8", "SHORT_BINUNICODE", "STRING", "BINSTRING", "SHORT_BINSTRING", "BINBYTES", "BINBYTES8", "SHORT_BINBYTES"}:
                stack.append(arg)
            elif name == "BYTEARRAY8":
                stack.append(bytearray(arg))
            elif name in {"EMPTY_TUPLE", "EMPTY_LIST", "EMPTY_DICT", "EMPTY_SET"}:
                stack.append({"EMPTY_TUPLE": tuple, "EMPTY_LIST": list, "EMPTY_DICT": dict, "EMPTY_SET": set}[name]())
            elif name in {"TUPLE", "LIST", "DICT", "FROZENSET"}:
                values = marked()
                if name == "TUPLE":
                    stack.append(tuple(values))
                elif name == "LIST":
                    stack.append(values)
                elif name == "DICT":
                    target = {}
                    for key, value in pairs(values):
                        self.setitem(target, key, value)
                    stack.append(target)
                else:
                    stack.append(frozenset(values))
            elif name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
                size = int(name[-1])
                values = tuple(stack[-size:])
                del stack[-size:]
                stack.append(values)
            elif name == "APPEND":
                value = stack.pop()
                self.append(stack[-1], [value])
            elif name in {"APPENDS", "ADDITEMS"}:
                values = marked()
                target = stack[-1]
                if name == "ADDITEMS" and isinstance(target, set):
                    target.update(values)
                else:
                    self.append(target, values)
            elif name == "SETITEM":
                value, key = stack.pop(), stack.pop()
                self.setitem(stack[-1], key, value)
            elif name == "SETITEMS":
                values = marked()
                for key, value in pairs(values):
                    self.setitem(stack[-1], key, value)
            elif name in {"PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE"}:
                memo[len(memo) if name == "MEMOIZE" else arg] = stack[-1]
            elif name in {"GET", "BINGET", "LONG_BINGET"}:
                if arg not in memo:
                    raise LoadError("missing memo key")
                stack.append(memo[arg])
            elif name == "GLOBAL":
                module, attr = arg.split(" ", 1)
                stack.append(self.global_ref(module, attr))
            elif name == "STACK_GLOBAL":
                attr, module = stack.pop(), stack.pop()
                if not isinstance(attr, str) or not isinstance(module, str):
                    raise LoadError("STACK_GLOBAL requires strings")
                stack.append(self.global_ref(module, attr))
            elif name == "REDUCE":
                args, func = stack.pop(), stack.pop()
                if not isinstance(args, tuple):
                    raise LoadError("REDUCE requires tuple arguments")
                stack.append(self.call(func, args))
            elif name in {"NEWOBJ", "NEWOBJ_EX"}:
                kwargs = stack.pop() if name == "NEWOBJ_EX" else {}
                args, cls = stack.pop(), stack.pop()
                stack.append(self.construct(cls, args, kwargs, new_only=True))
            elif name in {"INST", "OBJ"}:
                args = marked()
                if name == "INST":
                    module, attr = arg.split(" ", 1)
                    cls = self.global_ref(module, attr)
                else:
                    cls, *args = args
                stack.append(self.construct(cls, tuple(args), new_only=not args and self.is_class(cls)))
            elif name == "BUILD":
                state = stack.pop()
                obj = stack[-1]
                self.build(obj, state)
                if isinstance(obj, Instance) and obj.attrs.get("_joblib_array"):
                    self.read_joblib_array(obj, stream)
            elif name in {"PERSID", "BINPERSID"}:
                if name == "BINPERSID":
                    arg = stack.pop()
                if not persistent:
                    raise LoadError("no persistent-id handler")
                stack.append(Instance("torch.Storage", (arg,)))
            elif name in {"EXT1", "EXT2", "EXT4"}:
                raise LoadError("unregistered extension code")
            elif name == "NEXT_BUFFER":
                raise LoadError("no out-of-band buffers supplied")
            elif name == "READONLY_BUFFER":
                if not stack:
                    raise LoadError("missing buffer")
            elif name == "STOP":
                if not stack or stack[-1] is MARK:
                    raise LoadError("empty pickle result")
                if frame_end is not None:
                    stream.seek(frame_end)
                return stack[-1]
            else:
                raise AnalysisError(f"unhandled opcode {name}")
            if self.findings:
                return Unknown("capability found")
        raise AnalysisError("pickle has no STOP")

    def read_joblib_array(self, obj, stream):
        dtype = obj.attrs.get("dtype")
        descr = dtype.args[0] if isinstance(dtype, Instance) and dtype.args else dtype
        if isinstance(descr, str) and descr.lstrip("<>=|").startswith("O"):
            # joblib delegates object arrays to a second unpickler.
            self.run_stream(stream)
            return
        shape = obj.attrs.get("shape", ())
        count = 1
        for size in shape:
            count *= size
        import re
        match = re.search(r"(\d+)$", str(descr))
        if not match:
            raise AnalysisError("unknown joblib array dtype")
        size = int(match[1]) * count
        if str(descr).lstrip("<>=|").startswith("U"):
            size *= 4
        if obj.attrs.get("numpy_array_alignment_bytes") is not None:
            padding = stream.read(1)
            if not padding:
                raise LoadError("truncated joblib array")
            stream.seek(padding[0], 1)
        stream.seek(size, 1)


def scan_pickle(data: bytes, location: str) -> list[Finding]:
    vm = Interpreter()
    vm.load(data, location)
    return vm.findings
