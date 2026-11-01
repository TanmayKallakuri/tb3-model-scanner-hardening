"""A symbolic re-implementation of CPython's unpickler.

It follows the opcode semantics of ``_pickle.c`` (``pickle.load``: flavor
"c") and of the pure-Python ``pickle._Unpickler`` that ``joblib.load`` uses
(flavor "py"), but every object is symbolic (see values.py): GLOBAL resolves
names through the offline stdlib index, and every call is handed to
semantics.Sem, which reports capabilities instead of running anything.

Errors the real unpickler would raise end the load at the same opcode
(SimError); whatever ran before the error has already been reported.
"""
from __future__ import annotations

import _compat_pickle
import codecs
import struct

from modelgate import pyindex
from modelgate.semantics import NOTFOUND, Sem, short
from modelgate.values import (
    ClassDict, ClassObj, Instance, LazyIter, Module, ModuleDict, Opaque, PByteArray, PDict, PList, PSet, Ref,
    SimError, describe, is_hashable,
)

HIGHEST_PROTOCOL = 5
NAME_MAPPING = _compat_pickle.NAME_MAPPING
IMPORT_MAPPING = _compat_pickle.IMPORT_MAPPING

# names torch.load's find_class maps to a storage type without importing anything
TORCH_STORAGE_TYPES = frozenset({
    "DoubleStorage", "FloatStorage", "HalfStorage", "LongStorage", "IntStorage", "ShortStorage", "CharStorage",
    "ByteStorage", "BoolStorage", "BFloat16Storage", "ComplexDoubleStorage", "ComplexFloatStorage", "QInt8Storage",
    "QUInt8Storage", "QInt32Storage", "QUInt4x2Storage", "QUInt2x4Storage", "Float8_e5m2Storage",
    "Float8_e4m3fnStorage", "Float8_e5m2fnuzStorage", "Float8_e4m3fnuzStorage", "UInt16Storage", "UInt32Storage",
    "UInt64Storage",
})
LAZY_TYPES = {"map": "builtins.map", "filter": "builtins.filter", "zip": "builtins.zip",
              "enumerate": "builtins.enumerate", "reversed": "builtins.reversed", "starmap": "itertools.starmap",
              "chain": "itertools.chain", "chain_from": "itertools.chain", "islice": "itertools.islice",
              "accumulate": "itertools.accumulate", "filterfalse": "itertools.filterfalse",
              "dropwhile": "itertools.dropwhile", "takewhile": "itertools.takewhile", "groupby": "itertools.groupby",
              "cycle": "itertools.cycle", "repeat": "itertools.repeat", "count": "itertools.count",
              "callable_iter": "builtins.iter"}
JOBLIB_WRAPPERS = ("joblib.numpy_pickle.NumpyArrayWrapper", "sklearn.externals.joblib.numpy_pickle.NumpyArrayWrapper")


class Stop(Exception):
    def __init__(self, value):
        self.value = value


class Unpickler:
    """One ``load()`` of one pickle stream starting at ``start``."""

    def __init__(self, sem: Sem, data: bytes, start: int = 0, *, flavor: str = "c", encoding: str = "ASCII",
                 errors: str = "strict", fix_imports: bool = True, persistent=None, joblib: bool = False,
                 label: str = "pickle", torch: bool = False):
        self.sem = sem
        self.ctx = sem.ctx
        self.data = data
        self.pos = start
        self.start = start
        self.py = flavor == "py"
        self.encoding = encoding
        self.errors = errors
        self.fix_imports = fix_imports
        self.persistent = persistent
        self.joblib = joblib
        self.label = label
        self.torch = torch
        self.proto = 0
        self.stack: list = []
        self.marks: list = []
        self.memo: dict = {}
        self.frame_end = 0
        self.op_pos = start

    # ------------------------------------------------------------ input
    def read(self, n: int) -> bytes:
        if n < 0:
            raise SimError("negative byte count")
        p = self.pos
        if p + n > len(self.data):
            raise SimError("pickle data was truncated")
        self.pos = p + n
        return self.data[p:p + n]

    def readline(self) -> bytes:
        i = self.data.find(b"\n", self.pos)
        if i < 0:
            if self.py:
                line = self.data[self.pos:]
                self.pos = len(self.data)
                return line
            raise SimError("pickle data was truncated")
        line = self.data[self.pos:i + 1]
        self.pos = i + 1
        return line

    def readline_strip(self) -> bytes:
        line = self.readline()
        if not self.py and len(line) < 2:
            raise SimError("pickle data was truncated")
        return line[:-1] if line.endswith(b"\n") else line

    # ------------------------------------------------------------ stack
    @property
    def fence(self):
        return self.marks[-1] if self.marks else 0

    def push(self, v):
        self.stack.append(v)

    def pop(self):
        if len(self.stack) <= self.fence:
            raise SimError("unpickling stack underflow")
        return self.stack.pop()

    def top(self):
        if len(self.stack) <= self.fence:
            raise SimError("unpickling stack underflow")
        return self.stack[-1]

    def pop_mark(self):
        if not self.marks:
            raise SimError("could not find MARK")
        m = self.marks.pop()
        items = self.stack[m:]
        del self.stack[m:]
        return items

    # ------------------------------------------------------------ main loop
    def load(self):
        """Run until STOP; returns (value, end_position)."""
        dispatch = DISPATCH
        data = self.data
        n = len(data)
        ctx = self.ctx
        while True:
            if self.pos >= n:
                if self.pos == self.start:
                    raise SimError("EOF: ran out of input")
                raise SimError("pickle data was truncated")
            self.op_pos = self.pos
            op = data[self.pos]
            self.pos += 1
            fn = dispatch[op]
            if fn is None:
                raise SimError(f"invalid load key, {bytes([op])!r}")
            ctx.tick()
            try:
                fn(self)
            except Stop as s:
                return s.value, self.pos

    def where(self):
        return f"{self.label}@{self.op_pos}"

    def set_loc(self):
        self.ctx.loc = self.where()

    # ------------------------------------------------------------ opcodes
    def op_mark(self):
        self.marks.append(len(self.stack))

    def op_stop(self):
        if self.py:
            if not self.stack:
                raise SimError("pop from empty list")
            raise Stop(self.stack.pop())
        raise Stop(self.pop())

    def op_pop(self):
        if self.marks and self.marks[-1] == len(self.stack):
            self.marks.pop()
            return
        self.pop()

    def op_pop_mark(self):
        self.pop_mark()

    def op_dup(self):
        self.push(self.top())

    def op_float(self):
        s = self.readline_strip()
        try:
            self.push(float(s))
        except ValueError:
            raise SimError("could not convert string to float") from None

    def op_int(self):
        s = self.readline_strip()
        if s == b"00":
            self.push(False)
        elif s == b"01":
            self.push(True)
        else:
            self.push(_parse_int(s, 0))

    def op_binint(self):
        self.push(struct.unpack("<i", self.read(4))[0])

    def op_binint1(self):
        self.push(self.read(1)[0])

    def op_binint2(self):
        self.push(struct.unpack("<H", self.read(2))[0])

    def op_long(self):
        s = self.readline_strip()
        if s.endswith(b"L"):
            s = s[:-1]
        self.push(_parse_int(s, 0))

    def op_long1(self):
        n = self.read(1)[0]
        self.push(int.from_bytes(self.read(n), "little", signed=True))

    def op_long4(self):
        n = struct.unpack("<i", self.read(4))[0]
        if n < 0:
            raise SimError("LONG pickle has negative byte count")
        if n > 1 << 20:
            raise SimError("LONG too large to model")
        self.push(int.from_bytes(self.read(n), "little", signed=True))

    def op_none(self):
        self.push(None)

    def op_newtrue(self):
        self.push(True)

    def op_newfalse(self):
        self.push(False)

    def _decode_str(self, b: bytes):
        if self.encoding == "bytes":
            return b
        try:
            return b.decode(self.encoding, self.errors)
        except (UnicodeDecodeError, LookupError):
            raise SimError("could not decode STRING") from None

    def op_string(self):
        s = self.readline_strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0:1] in (b"'", b'"'):
            s = s[1:-1]
        else:
            raise SimError("the STRING opcode argument must be quoted")
        try:
            b = codecs.escape_decode(s)[0]
        except Exception:
            raise SimError("bad STRING escape") from None
        self.push(self._decode_str(b))

    def op_binstring(self):
        n = struct.unpack("<i", self.read(4))[0]
        if n < 0:
            raise SimError("BINSTRING pickle has negative byte count")
        self.push(self._decode_str(self.read(n)))

    def op_short_binstring(self):
        n = self.read(1)[0]
        self.push(self._decode_str(self.read(n)))

    def op_unicode(self):
        s = self.readline()
        if s.endswith(b"\n"):
            s = s[:-1]
        elif not self.py:
            raise SimError("pickle data was truncated")
        try:
            self.push(s.decode("raw-unicode-escape"))
        except UnicodeDecodeError:
            raise SimError("bad UNICODE escape") from None

    def _utf8(self, b):
        try:
            return b.decode("utf-8", "surrogatepass")
        except UnicodeDecodeError:
            raise SimError("invalid utf-8 in BINUNICODE") from None

    def op_binunicode(self):
        n = struct.unpack("<I", self.read(4))[0]
        self.push(self._utf8(self.read(n)))

    def op_short_binunicode(self):
        n = self.read(1)[0]
        self.push(self._utf8(self.read(n)))

    def op_binunicode8(self):
        n = struct.unpack("<Q", self.read(8))[0]
        self.push(self._utf8(self.read(n)))

    def op_binbytes(self):
        n = struct.unpack("<I", self.read(4))[0]
        self.push(self.read(n))

    def op_short_binbytes(self):
        n = self.read(1)[0]
        self.push(self.read(n))

    def op_binbytes8(self):
        n = struct.unpack("<Q", self.read(8))[0]
        self.push(self.read(n))

    def op_bytearray8(self):
        n = struct.unpack("<Q", self.read(8))[0]
        self.push(PByteArray(self.read(n)))

    def op_next_buffer(self):
        raise SimError("pickle stream refers to out-of-band data but no buffers were given")

    def op_readonly_buffer(self):
        v = self.top()
        if isinstance(v, PByteArray):
            self.stack[-1] = bytes(v.b)
        elif not isinstance(v, bytes):
            raise SimError("a bytes-like object is required")

    def op_binfloat(self):
        self.push(struct.unpack(">d", self.read(8))[0])

    def op_empty_tuple(self):
        self.push(())

    def op_tuple(self):
        self.push(tuple(self.pop_mark()))

    def op_tuple1(self):
        self.push((self.pop(),))

    def op_tuple2(self):
        b = self.pop()
        a = self.pop()
        self.push((a, b))

    def op_tuple3(self):
        c = self.pop()
        b = self.pop()
        a = self.pop()
        self.push((a, b, c))

    def op_empty_list(self):
        self.push(PList())

    def op_list(self):
        self.push(PList(self.pop_mark()))

    def op_empty_dict(self):
        self.push(PDict())

    def op_dict(self):
        items = self.pop_mark()
        if len(items) % 2:
            raise SimError("odd number of items for DICT")
        d = PDict()
        for i in range(0, len(items), 2):
            if not is_hashable(items[i]):
                raise SimError("unhashable type in DICT")
            d.d[items[i]] = items[i + 1]
        self.push(d)

    def op_empty_set(self):
        self.push(PSet())

    def op_frozenset(self):
        items = self.pop_mark()
        if not all(is_hashable(x) for x in items):
            raise SimError("unhashable type in FROZENSET")
        self.push(frozenset(items))

    def op_additems(self):
        items = self.pop_mark()
        if not self.stack or len(self.stack) <= self.fence:
            raise SimError("unpickling stack underflow")
        target = self.stack[-1]
        self.set_loc()
        if not items and not self.py:
            return
        if isinstance(target, PSet):
            for x in items:
                if not is_hashable(x):
                    raise SimError("unhashable type")
                target.s.add(x)
            return
        add = self.sem.getattr(target, "add")
        for x in items:
            self.sem.call(add, (x,), {}, "ADDITEMS calls .add")

    def op_append(self):
        value = self.pop()
        target = self.top()
        self.set_loc()
        if isinstance(target, PList):
            target.items.append(value)
            return
        if self.py:
            self.sem.call(self.sem.getattr(target, "append"), (value,), {}, "APPEND calls .append")
            return
        self._extend_or_append(target, [value], "APPEND")

    def op_appends(self):
        items = self.pop_mark()
        target = self.top()
        self.set_loc()
        if isinstance(target, PList):
            target.items.extend(items)
            return
        if not items and not self.py:
            return
        if self.py:
            try:
                ext = self.sem.getattr(target, "extend")
            except SimError:
                ext = None
            if ext is not None:
                self.sem.call(ext, (PList(items),), {}, "APPENDS calls .extend")
                return
            app = self.sem.getattr(target, "append")
            for x in items:
                self.sem.call(app, (x,), {}, "APPENDS calls .append")
            return
        self._extend_or_append(target, items, "APPENDS")

    def _extend_or_append(self, target, items, opname):
        try:
            ext = self.sem.getattr(target, "extend")
        except SimError:
            ext = None
        if ext is not None:
            self.sem.call(ext, (PList(items),), {}, f"{opname} calls .extend")
            return
        app = self.sem.getattr(target, "append")
        for x in items:
            self.sem.call(app, (x,), {}, f"{opname} calls .append")

    def op_setitem(self):
        if len(self.stack) - 2 <= self.fence:
            raise SimError("unpickling stack underflow")
        value = self.stack.pop()
        key = self.stack.pop()
        target = self.stack[-1]
        self.set_loc()
        self.sem.setitem(target, key, value)

    def op_setitems(self):
        items = self.pop_mark()
        if len(items) % 2:
            raise SimError("odd number of items for SETITEMS")
        target = self.top()
        self.set_loc()
        if not items:
            return
        for i in range(0, len(items), 2):
            self.sem.setitem(target, items[i], items[i + 1])

    # ---- memo
    def op_get(self):
        s = self.readline_strip()
        self._get(_parse_int(s, 10))

    def op_binget(self):
        self._get(self.read(1)[0])

    def op_long_binget(self):
        self._get(struct.unpack("<I", self.read(4))[0])

    def _get(self, i):
        try:
            self.push(self.memo[i])
        except KeyError:
            raise SimError(f"Memo value not found at index {i}") from None

    def op_put(self):
        s = self.readline_strip()
        i = _parse_int(s, 10)
        if i < 0:
            raise SimError("negative PUT argument")
        self.memo[i] = self.top()

    def op_binput(self):
        i = self.read(1)[0]
        self.memo[i] = self.top()

    def op_long_binput(self):
        i = struct.unpack("<I", self.read(4))[0]
        self.memo[i] = self.top()

    def op_memoize(self):
        self.memo[len(self.memo)] = self.top()

    # ---- framing / protocol
    def op_proto(self):
        p = self.read(1)[0]
        if p > HIGHEST_PROTOCOL:
            raise SimError(f"unsupported pickle protocol: {p}")
        self.proto = p

    def op_frame(self):
        n = struct.unpack("<Q", self.read(8))[0]
        if self.pos + n > len(self.data):
            raise SimError("pickle data was truncated")
        self.frame_end = self.pos + n

    # ---- globals
    def find_class(self, module, name):
        self.set_loc()
        sem = self.sem
        if self.proto < 3 and self.fix_imports:
            key = (module, name)
            if key in NAME_MAPPING:
                module, name = NAME_MAPPING[key]
            elif module in IMPORT_MAPPING:
                module = IMPORT_MAPPING[module]
        if self.torch:
            if isinstance(name, str) and name in TORCH_STORAGE_TYPES:
                return Opaque(f"torch storage type {name}")
            if module == "torch.tensor":
                module = "torch._tensor"
        bmod = self.ctx.modules.get("builtins")
        if bmod is not None and "__import__" in bmod.overlay:
            # PyImport_Import calls builtins.__import__(name, globals, globals, ['__doc__'], 0)
            g = Opaque("unpickler globals")
            sem.call(bmod.overlay["__import__"], (module, g, g, PList(["__doc__"]), 0), {},
                     "GLOBAL calls the rebound builtins.__import__")
        mod = sem.import_module(module)
        if self.proto >= 4:
            parts = name.split(".")
            if "<locals>" in parts:
                raise SimError(f"Can't get local attribute {name!r}")
            obj = mod
            for p in parts:
                try:
                    obj = sem.getattr(obj, p)
                except SimError:
                    raise SimError(f"Can't get attribute {name!r} on {describe(mod)}") from None
            return obj
        try:
            return sem.getattr(mod, name)
        except SimError:
            raise SimError(f"Can't get attribute {name!r} on {describe(mod)}") from None

    def _decode_global(self, b, ascii_only):
        try:
            return b.decode("ascii" if ascii_only else "utf-8")
        except UnicodeDecodeError:
            raise SimError("bad GLOBAL name") from None

    def op_global(self):
        module = self._decode_global(self.readline_strip(), False)
        name = self._decode_global(self.readline_strip(), False)
        self.push(self.find_class(module, name))

    def op_stack_global(self):
        name = self.pop()
        module = self.pop()
        if type(name) is not str or type(module) is not str:
            raise SimError("STACK_GLOBAL requires str")
        self.push(self.find_class(module, name))

    def op_ext(self, nbytes):
        code = int.from_bytes(self.read(nbytes), "little")
        if code <= 0 and nbytes == 4:
            raise SimError("EXT specifies code <= 0")
        key = self.ctx.extensions.get(code)
        if key is None:
            raise SimError(f"unregistered extension code {code}")
        module, name = key
        if not isinstance(module, str) or not isinstance(name, str):
            raise SimError("bad extension registry entry")
        self.push(self.find_class(module, name))

    def op_ext1(self):
        self.op_ext(1)

    def op_ext2(self):
        self.op_ext(2)

    def op_ext4(self):
        self.op_ext(4)

    # ---- object construction
    def _is_type(self, cls):
        if isinstance(cls, ClassObj):
            return True
        if isinstance(cls, Ref):
            if cls.kind == "c":
                return True
            if cls.kind == "?":
                last = cls.oid.rsplit(".", 1)[-1]
                return last[:1].isupper()
        return False

    def _instantiate(self, cls, args, opname):
        self.set_loc()
        if not args and self._is_type(cls):
            has_initargs = False
            if isinstance(cls, ClassObj):
                has_initargs = self.sem.classobj_raw(cls, "__getinitargs__") is not NOTFOUND
            elif isinstance(cls, Ref) and cls.kind == "c":
                has_initargs = self.sem.class_lookup(cls, "__getinitargs__") is not NOTFOUND
            if not has_initargs:
                return self.sem.new_object(cls, (), {}, f"{opname} (no arguments: __new__ only)")
        return self.sem.call(cls, tuple(args), {}, f"{opname} {short(cls)}")

    def op_inst(self):
        if self.py:
            module = self._decode_global(self.readline_strip(), True)
            name = self._decode_global(self.readline_strip(), True)
            cls = self.find_class(module, name)
            args = self.pop_mark()
        else:
            if not self.marks:
                raise SimError("could not find MARK")
            module = self._decode_global(self.readline_strip(), True)
            name = self._decode_global(self.readline_strip(), True)
            cls = self.find_class(module, name)
            args = self.pop_mark()
        self.push(self._instantiate(cls, args, "INST"))

    def op_obj(self):
        items = self.pop_mark()
        if not items:
            raise SimError("unpickling stack underflow")
        cls, args = items[0], items[1:]
        self.push(self._instantiate(cls, args, "OBJ"))

    def op_newobj(self):
        args = self.pop()
        cls = self.pop()
        self.set_loc()
        if self.py:
            if not (isinstance(cls, ClassObj) or (isinstance(cls, Ref) and cls.kind in ("c", "?"))):
                raise SimError("__new__(X): X is not a type object")
            args = tuple(self.sem.iterate(args))
        else:
            if not self._is_type(cls) and not (isinstance(cls, Ref) and cls.kind == "?"):
                raise SimError("NEWOBJ class argument must be a type")
            if not isinstance(args, tuple):
                raise SimError("NEWOBJ args argument must be a tuple")
        self.push(self.sem.new_object(cls, args, {}, f"NEWOBJ {short(cls)}"))

    def op_newobj_ex(self):
        kwargs = self.pop()
        args = self.pop()
        cls = self.pop()
        self.set_loc()
        if self.py:
            if not (isinstance(cls, ClassObj) or (isinstance(cls, Ref) and cls.kind in ("c", "?"))):
                raise SimError("__new__(X): X is not a type object")
            args = tuple(self.sem.iterate(args))
            if not isinstance(kwargs, PDict):
                raise SimError("argument after ** must be a mapping")
        else:
            if not self._is_type(cls) and not (isinstance(cls, Ref) and cls.kind == "?"):
                raise SimError("NEWOBJ_EX class argument must be a type")
            if not isinstance(args, tuple):
                raise SimError("NEWOBJ_EX args argument must be a tuple")
            if not isinstance(kwargs, PDict):
                raise SimError("NEWOBJ_EX kwargs argument must be a dict")
        kw = {}
        for k, v in kwargs.d.items():
            if not isinstance(k, str):
                raise SimError("keywords must be strings")
            kw[k] = v
        self.push(self.sem.new_object(cls, args, kw, f"NEWOBJ_EX {short(cls)}"))

    def op_reduce(self):
        args = self.pop()
        if self.py:
            func = self.top()
            self.stack.pop()
        else:
            func = self.pop()
        self.set_loc()
        if self.py:
            if isinstance(args, PDict):
                args = tuple(args.d.keys())
            elif not isinstance(args, tuple):
                args = tuple(self.sem.iterate(args))
        elif not isinstance(args, tuple):
            raise SimError("argument list must be a tuple")
        self.push(self.sem.call(func, args, {}, f"REDUCE {short(func)}"))

    def op_build(self):
        state = self.pop()
        inst = self.top()
        self.set_loc()
        self.build(inst, state)
        if self.joblib and self.py:
            self._joblib_array()

    def op_persid(self):
        pid = self.readline_strip()
        try:
            pid = pid.decode("ascii")
        except UnicodeDecodeError:
            raise SimError("persistent IDs in protocol 0 must be ASCII strings") from None
        self._persistent(pid)

    def op_binpersid(self):
        self._persistent(self.pop())

    def _persistent(self, pid):
        if self.persistent is None:
            raise SimError("A load persistent id instruction was encountered, but no persistent_load "
                           "function was specified.")
        self.set_loc()
        self.push(self.persistent(pid))

    # ------------------------------------------------------------ BUILD
    def _setstate_of(self, inst):
        sem = self.sem
        if isinstance(inst, Instance):
            if inst.d is not None and "__setstate__" in inst.d.d:
                return inst.d.d["__setstate__"]
            cls = inst.cls
            if isinstance(cls, Ref) and cls.kind == "?":
                ov = self.ctx.class_overlay.get(cls.oid)
                if ov and "__setstate__" in ov:
                    return BoundMethodOf(inst, ov["__setstate__"])
                return None
            t = sem.class_lookup(cls, "__setstate__") if isinstance(cls, (Ref, ClassObj)) else NOTFOUND
            if t is NOTFOUND:
                return None
            tag, v = t
            if tag == "func":
                return BoundMethodOf(inst, v)
            if tag == "cm":
                return BoundMethodOf(cls, v)
            return v
        if isinstance(inst, Module):
            v = inst.overlay.get("__setstate__")
            return v
        if isinstance(inst, ClassObj):
            v = sem.classobj_raw(inst, "__setstate__")
            return None if v is NOTFOUND else v
        if isinstance(inst, Ref):
            if inst.kind == "c":
                t = sem.class_lookup(inst, "__setstate__")
                return None if t is NOTFOUND else t[1]
            fa = self.ctx.func_attrs.get(inst.oid)
            if fa is not None and "__setstate__" in fa.d:
                return fa.d["__setstate__"]
            return None
        if isinstance(inst, Opaque):
            if inst.attrs and "__setstate__" in inst.attrs:
                return inst.attrs["__setstate__"]
            return "opaque"
        if isinstance(inst, LazyIter):
            cls_oid = LAZY_TYPES.get(inst.kind)
            if cls_oid:
                t = sem.class_lookup(sem.ref(cls_oid), "__setstate__")
                if t is not NOTFOUND:
                    return BoundMethodOf(inst, t[1])
        return None

    def build(self, inst, state):
        sem = self.sem
        setstate = self._setstate_of(inst)
        if setstate == "opaque":
            if isinstance(state, PDict):
                for k, v in state.d.items():
                    if isinstance(k, str):
                        sem.setattr(inst, k, v)
            return
        if setstate is not None:
            sem.call(setstate, (state,), {}, "BUILD calls __setstate__")
            return
        slotstate = None
        has_slot = False
        if isinstance(state, tuple) and len(state) == 2:
            state, slotstate = state
            has_slot = True
        lenient = isinstance(inst, Instance) and isinstance(inst.cls, Ref) and inst.cls.kind == "?"
        sd = self._state_dict(state)
        if sd is not None:
            state = sd
        if self.py:
            if sem.truthy(state):
                if not isinstance(state, PDict):
                    if lenient or isinstance(state, (Instance, Opaque)):
                        return  # truthiness / items() of this object are unknown
                    raise SimError("state has no items()")
                d = self._inst_dict(inst, lenient)
                for k, v in state.d.items():
                    sem.setitem(d, k, v)
            if has_slot and sem.truthy(slotstate):
                if not isinstance(slotstate, PDict):
                    raise SimError("slot state has no items()")
                for k, v in slotstate.d.items():
                    sem.setattr(inst, k, v)
            return
        if state is not None:
            if not isinstance(state, PDict):
                if lenient:
                    return
                raise SimError("state is not a dictionary")
            d = self._inst_dict(inst, lenient)
            for k, v in state.d.items():
                if not is_hashable(k):
                    raise SimError("unhashable key in state")
                sem.setitem(d, k, v)
        if has_slot:
            if not isinstance(slotstate, PDict):
                if lenient:
                    return
                raise SimError("slot state is not a dictionary")
            for k, v in slotstate.d.items():
                sem.setattr(inst, k, v)

    def _state_dict(self, state):
        """A dict subclass instance (OrderedDict, defaultdict, ...) as a PDict."""
        if isinstance(state, Instance) and isinstance(state.cls, Ref):
            items = state.extra.get("items")
            dictlike = isinstance(items, dict)
            if not dictlike and state.cls.kind == "c":
                info = pyindex.class_info(state.cls.oid)
                dictlike = bool(info) and "builtins.dict" in info[2]
            if dictlike:
                return PDict(items if isinstance(items, dict) else {})
        return None

    def _inst_dict(self, inst, lenient):
        if isinstance(inst, Instance):
            if inst.d is None:
                if lenient:
                    inst.d = PDict()
                else:
                    raise SimError("object has no __dict__")
            return inst.d
        if isinstance(inst, Module):
            return ModuleDict(inst)
        if isinstance(inst, Ref) and inst.kind in ("f", "?"):
            return self.ctx.func_attrs.setdefault(inst.oid, PDict())
        if isinstance(inst, (Ref, ClassObj)) and (isinstance(inst, ClassObj) or inst.kind == "c"):
            return ClassDict(inst)  # a read-only mappingproxy: assigning into it fails
        if isinstance(inst, Opaque):
            return PDict()
        raise SimError(f"{describe(inst)} has no __dict__ to update")

    # ------------------------------------------------------------ joblib
    def _joblib_array(self):
        """joblib's NumpyUnpickler reads the array right after a wrapper's BUILD."""
        w = self.stack[-1] if self.stack else None
        if not (isinstance(w, Instance) and isinstance(w.cls, Ref) and w.cls.oid in JOBLIB_WRAPPERS):
            return
        d = w.d.d if w.d is not None else {}
        dtype = d.get("dtype")
        shape = d.get("shape", ())
        typestr, flags, elsize = None, 0, -1
        if isinstance(dtype, Instance):
            typestr = dtype.extra.get("typestr")
            st = dtype.extra.get("state")
            if isinstance(st, tuple) and len(st) >= 8:
                elsize = st[5] if isinstance(st[5], int) else -1
                flags = st[7] if isinstance(st[7], int) else 0
        start = max(self.pos, self.frame_end)
        hasobject = (isinstance(typestr, str) and typestr.lstrip("<>|=").startswith("O")) or bool(flags & 1)
        if hasobject:
            nested = Unpickler(self.sem, self.data, start, flavor="c", encoding="ASCII",
                               label=f"{self.label}[joblib object array @{start}]")
            value, end = nested.load()
            self.pos = end
        else:
            count = 1
            if isinstance(shape, tuple):
                for s in shape:
                    count *= s if isinstance(s, int) and s >= 0 else 0
            itemsize = _itemsize(typestr, elsize)
            pos = start
            if "numpy_array_alignment_bytes" in d and d.get("numpy_array_alignment_bytes") is not None:
                if pos >= len(self.data):
                    raise SimError("joblib array data truncated")
                pos += 1 + self.data[pos]
            end = pos + count * itemsize
            if end > len(self.data):
                raise SimError("joblib array data truncated")
            self.pos = end
        self.frame_end = 0
        self.stack[-1] = Opaque("numpy array (joblib)")


def BoundMethodOf(obj, f):
    from modelgate.values import BoundMethod
    return BoundMethod(obj, f)


def _itemsize(typestr, elsize):
    if isinstance(elsize, int) and elsize > 0:
        return elsize
    if isinstance(typestr, str):
        try:
            import numpy
            return numpy.dtype(typestr).itemsize
        except Exception:
            pass
    return 8


def _parse_int(s: bytes, base: int) -> int:
    try:
        txt = s.decode("ascii").strip()
    except UnicodeDecodeError:
        raise SimError("could not convert string to int") from None
    try:
        if base == 0:
            try:
                return int(txt, 0)
            except ValueError:
                # strtol(base 0) accepts legacy octal such as "010"
                neg = txt.startswith("-")
                t = txt.lstrip("+-")
                if t.startswith("0") and t.isdigit():
                    v = int(t, 8)
                    return -v if neg else v
                return int(txt, 10)
        return int(txt, base)
    except ValueError:
        raise SimError("could not convert string to int") from None


DISPATCH = [None] * 256


def _reg(code, name):
    DISPATCH[code if isinstance(code, int) else ord(code)] = getattr(Unpickler, name)


for _code, _name in [
    ("(", "op_mark"), (".", "op_stop"), ("0", "op_pop"), ("1", "op_pop_mark"), ("2", "op_dup"),
    ("F", "op_float"), ("I", "op_int"), ("J", "op_binint"), ("K", "op_binint1"), ("L", "op_long"),
    ("M", "op_binint2"), ("N", "op_none"), ("P", "op_persid"), ("Q", "op_binpersid"), ("R", "op_reduce"),
    ("S", "op_string"), ("T", "op_binstring"), ("U", "op_short_binstring"), ("V", "op_unicode"),
    ("X", "op_binunicode"), ("a", "op_append"), ("b", "op_build"), ("c", "op_global"), ("d", "op_dict"),
    ("}", "op_empty_dict"), ("e", "op_appends"), ("g", "op_get"), ("h", "op_binget"), ("i", "op_inst"),
    ("j", "op_long_binget"), ("l", "op_list"), ("]", "op_empty_list"), ("o", "op_obj"), ("p", "op_put"),
    ("q", "op_binput"), ("r", "op_long_binput"), ("s", "op_setitem"), ("t", "op_tuple"),
    (")", "op_empty_tuple"), ("u", "op_setitems"), ("G", "op_binfloat"), ("B", "op_binbytes"),
    ("C", "op_short_binbytes"),
    (0x80, "op_proto"), (0x81, "op_newobj"), (0x82, "op_ext1"), (0x83, "op_ext2"), (0x84, "op_ext4"),
    (0x85, "op_tuple1"), (0x86, "op_tuple2"), (0x87, "op_tuple3"), (0x88, "op_newtrue"),
    (0x89, "op_newfalse"), (0x8A, "op_long1"), (0x8B, "op_long4"), (0x8C, "op_short_binunicode"),
    (0x8D, "op_binunicode8"), (0x8E, "op_binbytes8"), (0x8F, "op_empty_set"), (0x90, "op_additems"),
    (0x91, "op_frozenset"), (0x92, "op_newobj_ex"), (0x93, "op_stack_global"), (0x94, "op_memoize"),
    (0x95, "op_frame"), (0x96, "op_bytearray8"), (0x97, "op_next_buffer"), (0x98, "op_readonly_buffer"),
]:
    _reg(_code, _name)
