"""An emulator of CPython's unpickler that never runs the pickle.

``Unpickler`` executes one pickle stream opcode by opcode with the semantics
of the C unpickler ``pickle.load`` uses (or of the pure-Python one that
joblib and pandas use), against the symbolic object model in ``pyrt``.  When
the real loader would call a capability-bearing callable, the emulator
records a finding instead.  When the real loader would raise, the emulator
stops at the same point, so nothing after it counts.
"""
from __future__ import annotations

import codecs
import struct
import time

from modelgate import kb as kbmod
from modelgate.pyrt import PickleStop, Runtime, ScanAbort
from modelgate.report import Finding
from modelgate.values import Dtype, Glob, Inst, Mod, Opaque, Sym, describe

HIGHEST_PROTOCOL = 5
TORCH_MAGIC = 0x1950A86A20F9469CFC6C
TORCH_PROTOCOL = 1001
TORCH_STORAGE_TYPES = {
    "DoubleStorage", "FloatStorage", "HalfStorage", "LongStorage", "IntStorage", "ShortStorage",
    "CharStorage", "ByteStorage", "BoolStorage", "BFloat16Storage", "ComplexDoubleStorage",
    "ComplexFloatStorage", "QUInt8Storage", "QInt8Storage", "QInt32Storage", "QUInt4x2Storage",
    "QUInt2x4Storage", "UntypedStorage", "Float8_e5m2Storage", "Float8_e4m3fnStorage",
    "Float8_e5m2fnuzStorage", "Float8_e4m3fnuzStorage", "UInt16Storage", "UInt32Storage",
    "UInt64Storage", "TypedStorage",
}


class Context:
    """State shared by every stream and nested load of one artifact."""

    def __init__(self, deadline, max_steps=200_000_000):
        self.kb = kbmod.get()
        self.findings = []
        self._seen = set()
        self.mod_overrides = {}
        self.cls_overrides = {}
        self.sysmods_override = {}
        self.ext_registry = {}
        self.ext_cache = {}
        self.deadline = deadline
        self.max_steps = max_steps
        self.steps = 0
        self.call_depth = 0
        self.depth = 0
        self.notes = []
        self.rt = Runtime(self)

    # -------------------------------------------------------------- results
    def report(self, kind, site, detail):
        key = (kind, site, detail)
        if key in self._seen or len(self.findings) >= 200:
            return
        self._seen.add(key)
        self.findings.append(Finding(kind, site, detail))

    def note(self, text):
        if len(self.notes) < 50:
            self.notes.append(text)

    def tick(self, n=1):
        before = self.steps
        self.steps = before + n
        if self.steps > self.max_steps:
            raise ScanAbort("pickle emulation step budget exhausted")
        if (before ^ self.steps) & ~0x3FF and time.monotonic() > self.deadline:
            raise ScanAbort("analysis time budget exhausted")

    def check_time(self):
        if time.monotonic() > self.deadline:
            raise ScanAbort("analysis time budget exhausted")

    # --------------------------------------------------------- nested loads
    def _enter(self):
        self.depth += 1
        if self.depth > 16:
            self.depth -= 1
            raise ScanAbort("nested loaders too deep")

    def load_pickle_bytes(self, data, python, encoding, site):
        """``pickle.loads(data)``: one pickle, trailing bytes ignored."""
        val, _end = self.load_pickle_stream(data, 0, python, encoding, site)
        return val

    def load_pickle_stream(self, data, pos, python, encoding, site, persistent=None):
        """``pickle.load(f)`` from offset ``pos``: returns (value, end offset)."""
        self._enter()
        try:
            up = Unpickler(self, data, pos, site, python=python, encoding=encoding, persistent=persistent)
            ok, value, end = up.run()
            if not ok:
                raise PickleStop(f"nested load failed: {up.stop_reason}")
            return value, end
        finally:
            self.depth -= 1

    def load_torch_bytes(self, data, site):
        from modelgate import loaders
        self._enter()
        try:
            return loaders.torch_load_bytes(self, data, site)
        finally:
            self.depth -= 1

    def load_numpy_bytes(self, data, allow_pickle, npz_ok, site):
        from modelgate import loaders
        self._enter()
        try:
            return loaders.numpy_load_bytes(self, data, allow_pickle, npz_ok, site)
        finally:
            self.depth -= 1

    def load_joblib_bytes(self, data, site):
        from modelgate import loaders
        self._enter()
        try:
            return loaders.joblib_load_bytes(self, data, site)
        finally:
            self.depth -= 1

    def npz_member(self, npz, key, site):
        from modelgate import loaders
        self._enter()
        try:
            return loaders.npz_member(self, npz, key, site)
        finally:
            self.depth -= 1


def _decode_str(raw, encoding, errors="strict"):
    if encoding == "bytes":
        return raw
    try:
        return raw.decode(encoding, errors)
    except (UnicodeDecodeError, LookupError) as exc:
        raise PickleStop(f"cannot decode py2 string: {exc}") from None


class Unpickler:
    """Execute one pickle stream starting at ``pos`` in ``data``."""

    def __init__(self, ctx, data, pos, location, *, python=False, encoding="ASCII",
                 persistent=None, storages=None, joblib=False, torch=False):
        self.ctx = ctx
        self.rt = ctx.rt
        self.data = data
        self.pos = pos
        self.start = pos
        self.location = location
        self.python = python
        self.encoding = encoding if isinstance(encoding, str) else "ASCII"
        self.persistent = persistent
        self.storages = storages
        self.joblib = joblib
        self.torch = torch
        self.stack = []
        self.marks = []
        self.memo = {}
        self.proto = 0
        self.stop_reason = None
        self.ops = 0
        self.op_pos = pos

    # ----------------------------------------------------------- reading
    def read(self, n):
        end = self.pos + n
        if n < 0 or end > len(self.data):
            raise PickleStop("pickle data was truncated")
        out = self.data[self.pos:end]
        self.pos = end
        return out

    def readline(self):
        i = self.data.find(b"\n", self.pos)
        if i < 0:
            raise PickleStop("pickle data was truncated")
        out = self.data[self.pos:i + 1]
        self.pos = i + 1
        return out

    # ------------------------------------------------------------ stack
    @property
    def fence(self):
        return self.marks[-1] if self.marks else 0

    def push(self, v):
        self.stack.append(v)

    def pop(self):
        if len(self.stack) <= self.fence:
            raise PickleStop("unpickling stack underflow")
        return self.stack.pop()

    def top(self):
        if len(self.stack) <= self.fence:
            raise PickleStop("unpickling stack underflow")
        return self.stack[-1]

    def marker(self):
        if not self.marks:
            raise PickleStop("could not find MARK")
        return self.marks.pop()

    def pop_mark(self):
        m = self.marker()
        items = self.stack[m:]
        del self.stack[m:]
        return items

    # ----------------------------------------------------------- driver
    @property
    def site(self):
        """Where the current opcode is, for findings (built only when needed)."""
        return f"{self.location}@{self.op_pos}"

    def run(self):
        """Returns (completed, value, end offset)."""
        dispatch = DISPATCH
        data = self.data
        size = len(data)
        ctx = self.ctx
        try:
            while True:
                pos = self.pos
                if pos >= size:
                    raise PickleStop("ran out of input")
                self.op_pos = pos
                code = data[pos]
                self.pos = pos + 1
                self.ops += 1
                if not self.ops & 0xFF:
                    ctx.tick(0x100)
                fn = dispatch.get(code)
                if fn is None:
                    raise PickleStop(f"invalid load key {code:#04x}")
                if fn(self) is STOP_MARK:
                    return True, self.result, self.pos
        except PickleStop as exc:
            self.stop_reason = str(exc)
            return False, None, self.pos
        except RecursionError:
            raise ScanAbort("pickle structure nested too deeply to emulate") from None
        except MemoryError:
            raise ScanAbort("out of memory while emulating a pickle") from None
        except ScanAbort:
            raise
        except Exception as exc:  # noqa: BLE001 - an emulator defect must not pass as clean
            raise ScanAbort(f"internal error at {self.site}: {type(exc).__name__}: {exc}") from None

    # ------------------------------------------------------- global lookup
    def find_class(self, module, name):
        rt = self.rt
        if not isinstance(module, str) or not isinstance(name, str):
            raise PickleStop("module and name must be strings")
        if self.proto < 3:
            key = (module, name)
            if key in kbmod.NAME_MAPPING:
                module, name = kbmod.NAME_MAPPING[key]
            elif module in kbmod.IMPORT_MAPPING:
                module = kbmod.IMPORT_MAPPING[module]
        if self.torch:
            if "Storage" in name and name in TORCH_STORAGE_TYPES:
                return Glob("torch." + name, "class")
            module = {"torch.tensor": "torch._tensor"}.get(module, module)
        mod = rt.import_module(module, self.site)
        if self.proto >= 4:
            parts = name.split(".")
            if "<locals>" in parts:
                raise PickleStop(f"Can't get local attribute {name!r}")
            obj = mod
            for part in parts:
                obj = rt.getattr(obj, part, self.site)
            return obj
        if "." in name or not name:
            ov = self.ctx.mod_overrides.get(mod.name, {}) if isinstance(mod, Mod) else {}
            if name not in ov:
                raise PickleStop(f"Can't get attribute {name!r} on module {module!r}")
        return rt.getattr(mod, name, self.site)

    def persistent_load(self, pid):
        mode = self.persistent
        if mode is None:
            raise PickleStop("persistent id encountered but no persistent_load is set")
        if not isinstance(pid, tuple) or not pid:
            raise PickleStop("persistent id is not a tuple")
        typename = pid[0]
        if isinstance(typename, (bytes, bytearray)):
            typename = bytes(typename).decode("ascii", "replace")
        if mode == "torch-zip":
            if typename != "storage" or len(pid) != 5:
                raise PickleStop(f"unknown persistent id type {describe(typename)}")
            key = pid[2]
            if self.storages is not None and str(key) not in self.storages:
                raise PickleStop(f"storage record data/{key} is missing")
            return Opaque("torch storage")
        if mode == "torch-legacy":
            if typename == "module":
                if len(pid) < 2:
                    raise PickleStop("bad module persistent id")
                return pid[1]
            if typename == "storage":
                if len(pid) != 6:
                    raise PickleStop("bad storage persistent id")
                return Opaque("torch storage")
            raise PickleStop(f"unknown saved id type {describe(typename)}")
        raise PickleStop("unsupported persistent id")

    # ---------------------------------------------------------- joblib
    def joblib_array(self, wrapper):
        attrs = wrapper.attrs
        cname = wrapper.cls.name
        if cname.endswith("NDArrayWrapper"):
            return Opaque("array stored in a side file")
        dtype = attrs.get("dtype")
        shape = attrs.get("shape")
        hasobject, itemsize = dtype_info(dtype)
        if hasobject:
            val, end = self.ctx.load_pickle_stream(self.data, self.pos, False, "ASCII",
                                                   f"{self.site} > joblib object array")
            self.pos = end
            return val
        if itemsize is None:
            raise ScanAbort("joblib array with a dtype the scanner cannot size")
        if attrs.get("numpy_array_alignment_bytes") is not None:
            pad = self.read(1)[0]
            self.read(pad)
        count = 1
        if isinstance(shape, tuple):
            for d in shape:
                if not isinstance(d, int) or d < 0:
                    raise ScanAbort("joblib array with an unknown shape")
                count *= d
        elif shape is not None:
            raise ScanAbort("joblib array with an unknown shape")
        self.read(count * itemsize)
        return Opaque("numpy array")


STOP_MARK = object()


def dtype_info(dtype):
    """(hasobject, itemsize) of a dtype built in a pickle, itemsize None if unknown."""
    spec = dtype.spec if isinstance(dtype, Dtype) else dtype
    if isinstance(spec, (bytes, bytearray)):
        spec = bytes(spec).decode("latin-1")
    if not isinstance(spec, str):
        return False, None
    s = spec.lstrip("<>|=!")
    if s.startswith("O"):
        return True, 8
    try:
        import numpy  # only needed for joblib arrays
        dt = numpy.dtype(spec)
        return bool(dt.hasobject), int(dt.itemsize)
    except Exception:  # noqa: BLE001
        return False, None


# ===================================================================
# Opcode implementations
# ===================================================================
DISPATCH = {}


def op(char):
    def deco(fn):
        DISPATCH[ord(char) if isinstance(char, str) else char] = fn
        return fn
    return deco


def _int_line(u, line, allow_bool=True):
    s = line[:-1]
    if len(line) < 2:
        raise PickleStop("pickle data was truncated")
    txt = s.decode("latin-1")
    if allow_bool and len(line) == 3 and txt in ("00", "01"):
        return txt == "01"
    try:
        val = int(txt, 0)
    except ValueError:
        try:
            val = int(txt.strip(), 10)
        except ValueError:
            try:
                st = txt.strip()
                sign = -1 if st.startswith("-") else 1
                st = st.lstrip("+-")
                val = sign * int(st, 8) if st.startswith("0") else int(st)
            except ValueError:
                raise PickleStop("could not convert string to int") from None
    if allow_bool and len(line) == 3 and val in (0, 1) and not isinstance(val, bool):
        return bool(val)
    return val


@op("(")
def op_mark(u):
    u.marks.append(len(u.stack))


@op(".")
def op_stop(u):
    u.result = u.pop()
    return STOP_MARK


@op("0")
def op_pop(u):
    if u.marks and u.marks[-1] == len(u.stack):
        u.marks.pop()
    else:
        u.pop()


@op("1")
def op_pop_mark(u):
    u.pop_mark()


@op("2")
def op_dup(u):
    u.push(u.top())


@op("F")
def op_float(u):
    line = u.readline()
    try:
        u.push(float(line[:-1].decode("latin-1")))
    except ValueError:
        raise PickleStop("could not convert string to float") from None


@op("I")
def op_int(u):
    u.push(_int_line(u, u.readline()))


@op("J")
def op_binint(u):
    u.push(struct.unpack("<i", u.read(4))[0])


@op("K")
def op_binint1(u):
    u.push(u.read(1)[0])


@op("M")
def op_binint2(u):
    u.push(struct.unpack("<H", u.read(2))[0])


@op("L")
def op_long(u):
    line = u.readline()
    s = line[:-1]
    if s.endswith(b"L"):
        s = s[:-1]
    try:
        u.push(int(s.decode("latin-1"), 0))
    except ValueError:
        try:
            u.push(int(s.decode("latin-1")))
        except ValueError:
            raise PickleStop("could not convert string to int") from None


@op("N")
def op_none(u):
    u.push(None)


@op(0x88)
def op_true(u):
    u.push(True)


@op(0x89)
def op_false(u):
    u.push(False)


@op(0x8a)
def op_long1(u):
    n = u.read(1)[0]
    u.push(int.from_bytes(u.read(n), "little", signed=True))


@op(0x8b)
def op_long4(u):
    n = struct.unpack("<i", u.read(4))[0]
    if n < 0:
        raise PickleStop("LONG pickle has negative byte count")
    u.push(int.from_bytes(u.read(n), "little", signed=True))


@op("G")
def op_binfloat(u):
    u.push(struct.unpack(">d", u.read(8))[0])


@op("S")
def op_string(u):
    line = u.readline()[:-1]
    if len(line) >= 2 and line[:1] == line[-1:] and line[:1] in (b"'", b'"'):
        body = line[1:-1]
    else:
        raise PickleStop("the STRING opcode argument must be quoted")
    try:
        raw = codecs.escape_decode(body)[0]
    except Exception:  # noqa: BLE001
        raise PickleStop("invalid escape in STRING") from None
    u.push(_decode_str(raw, u.encoding))


@op("T")
def op_binstring(u):
    n = struct.unpack("<i", u.read(4))[0]
    if n < 0:
        raise PickleStop("BINSTRING pickle has negative byte count")
    u.push(_decode_str(u.read(n), u.encoding))


@op("U")
def op_short_binstring(u):
    n = u.read(1)[0]
    u.push(_decode_str(u.read(n), u.encoding))


@op("V")
def op_unicode(u):
    line = u.readline()[:-1]
    try:
        u.push(line.decode("raw_unicode_escape"))
    except UnicodeDecodeError:
        raise PickleStop("invalid UNICODE argument") from None


def _utf8(raw):
    try:
        return raw.decode("utf-8", "surrogatepass")
    except UnicodeDecodeError:
        raise PickleStop("invalid utf-8 in BINUNICODE") from None


@op("X")
def op_binunicode(u):
    n = struct.unpack("<I", u.read(4))[0]
    u.push(_utf8(u.read(n)))


@op(0x8c)
def op_short_binunicode(u):
    n = u.read(1)[0]
    u.push(_utf8(u.read(n)))


@op(0x8d)
def op_binunicode8(u):
    n = struct.unpack("<Q", u.read(8))[0]
    u.push(_utf8(u.read(n)))


@op("B")
def op_binbytes(u):
    n = struct.unpack("<I", u.read(4))[0]
    u.push(u.read(n))


@op("C")
def op_short_binbytes(u):
    n = u.read(1)[0]
    u.push(u.read(n))


@op(0x8e)
def op_binbytes8(u):
    n = struct.unpack("<Q", u.read(8))[0]
    u.push(u.read(n))


@op(0x96)
def op_bytearray8(u):
    n = struct.unpack("<Q", u.read(8))[0]
    u.push(bytearray(u.read(n)))


@op(0x97)
def op_next_buffer(u):
    raise PickleStop("pickle stream refers to out-of-band data but no buffers were given")


@op(0x98)
def op_readonly_buffer(u):
    v = u.top()
    if not isinstance(v, (bytes, bytearray)) and not isinstance(v, Sym):
        raise PickleStop("READONLY_BUFFER needs a buffer")
    if isinstance(v, bytearray):
        u.stack[-1] = bytes(v)


@op(")")
def op_empty_tuple(u):
    u.push(())


@op("t")
def op_tuple(u):
    u.push(tuple(u.pop_mark()))


@op(0x85)
def op_tuple1(u):
    a = u.pop()
    u.push((a,))


@op(0x86)
def op_tuple2(u):
    b = u.pop()
    a = u.pop()
    u.push((a, b))


@op(0x87)
def op_tuple3(u):
    c = u.pop()
    b = u.pop()
    a = u.pop()
    u.push((a, b, c))


@op("]")
def op_empty_list(u):
    u.push([])


@op("l")
def op_list(u):
    u.push(list(u.pop_mark()))


@op("}")
def op_empty_dict(u):
    u.push({})


def _make_dict(u, items):
    if len(items) % 2:
        raise PickleStop("odd number of items for DICT")
    d = {}
    for i in range(0, len(items), 2):
        u.rt.hashable(items[i])
        d[items[i]] = items[i + 1]
    return d


@op("d")
def op_dict(u):
    u.push(_make_dict(u, u.pop_mark()))


@op(0x8f)
def op_empty_set(u):
    u.push(set())


@op(0x91)
def op_frozenset(u):
    items = u.pop_mark()
    for it in items:
        u.rt.hashable(it)
    u.push(frozenset(items))


@op(0x90)
def op_additems(u):
    m = u.marker()
    if m < 1 or m - 1 < (u.marks[-1] if u.marks else 0):
        raise PickleStop("unpickling stack underflow")
    items = u.stack[m:]
    del u.stack[m:]
    if items:
        u.rt.additems(u.stack[m - 1], items, u.site)


@op("a")
def op_append(u):
    value = u.pop()
    target = u.top()
    u.rt.append(target, [value], u.site)


@op("e")
def op_appends(u):
    m = u.marker()
    if m < 1 or m - 1 < (u.marks[-1] if u.marks else 0):
        raise PickleStop("unpickling stack underflow")
    items = u.stack[m:]
    del u.stack[m:]
    if items:
        u.rt.append(u.stack[m - 1], items, u.site)


def _setitems(u, target, items):
    if len(items) % 2:
        raise PickleStop("odd number of items for SETITEMS")
    for i in range(0, len(items), 2):
        u.rt.setitem(target, items[i], items[i + 1], u.site)


@op("s")
def op_setitem(u):
    value = u.pop()
    key = u.pop()
    target = u.top()
    _setitems(u, target, [key, value])


@op("u")
def op_setitems(u):
    m = u.marker()
    if m < 1 or m - 1 < (u.marks[-1] if u.marks else 0):
        raise PickleStop("unpickling stack underflow")
    items = u.stack[m:]
    del u.stack[m:]
    _setitems(u, u.stack[m - 1], items)


def _memo_get(u, idx):
    if idx not in u.memo:
        raise PickleStop(f"memo value not found at index {idx}")
    u.push(u.memo[idx])


@op("g")
def op_get(u):
    line = u.readline()
    idx = _int_line(u, line, allow_bool=False)
    _memo_get(u, idx)


@op("h")
def op_binget(u):
    _memo_get(u, u.read(1)[0])


@op("j")
def op_long_binget(u):
    _memo_get(u, struct.unpack("<I", u.read(4))[0])


def _memo_put(u, idx):
    if idx < 0:
        raise PickleStop("negative PUT argument")
    u.memo[idx] = u.top()


@op("p")
def op_put(u):
    _memo_put(u, _int_line(u, u.readline(), allow_bool=False))


@op("q")
def op_binput(u):
    _memo_put(u, u.read(1)[0])


@op("r")
def op_long_binput(u):
    _memo_put(u, struct.unpack("<I", u.read(4))[0])


@op(0x94)
def op_memoize(u):
    if len(u.stack) <= u.fence:
        raise PickleStop("unpickling stack underflow")
    u.memo[len(u.memo)] = u.stack[-1]


@op(0x80)
def op_proto(u):
    v = u.read(1)[0]
    if v > HIGHEST_PROTOCOL:
        raise PickleStop(f"unsupported pickle protocol: {v}")
    u.proto = v


@op(0x95)
def op_frame(u):
    n = struct.unpack("<Q", u.read(8))[0]
    if u.pos + n > len(u.data):
        raise PickleStop("pickle data was truncated (frame)")


def _line_text(u, encoding):
    line = u.readline()
    if len(line) < 2:
        raise PickleStop("pickle data was truncated")
    try:
        return line[:-1].decode(encoding)
    except UnicodeDecodeError:
        raise PickleStop("bad module or name encoding") from None


@op("c")
def op_global(u):
    module = _line_text(u, "utf-8")
    name = _line_text(u, "utf-8")
    u.push(u.find_class(module, name))


@op(0x93)
def op_stack_global(u):
    name = u.pop()
    module = u.pop()
    if type(name) is not str or type(module) is not str:
        raise PickleStop("STACK_GLOBAL requires str")
    u.push(u.find_class(module, name))


def _ext(u, nbytes):
    code = int.from_bytes(u.read(nbytes), "little")
    if code <= 0:
        raise PickleStop("EXT specifies code <= 0")
    ctx = u.ctx
    if code in ctx.ext_cache:
        u.push(ctx.ext_cache[code])
        return
    pair = ctx.ext_registry.get(code)
    if pair is None:
        raise PickleStop(f"unregistered extension code {code}")
    if not (isinstance(pair, tuple) and len(pair) == 2 and all(isinstance(x, str) for x in pair)):
        raise PickleStop("_inverted_registry entry must be a 2-tuple of str")
    obj = u.find_class(pair[0], pair[1])
    ctx.ext_cache[code] = obj
    u.push(obj)


@op(0x82)
def op_ext1(u):
    _ext(u, 1)


@op(0x83)
def op_ext2(u):
    _ext(u, 2)


@op(0x84)
def op_ext4(u):
    _ext(u, 4)


@op("R")
def op_reduce(u):
    args = u.pop()
    func = u.pop()
    if not isinstance(args, tuple):
        if not u.python:
            raise PickleStop("REDUCE argument list must be a tuple")
        args = tuple(u.rt.iterate(args, u.site))
    u.push(u.rt.call(func, args, {}, u.site, "REDUCE calls"))


@op(0x81)
def op_newobj(u):
    args = u.pop()
    cls = u.pop()
    if not u.python:
        if not u.rt.could_be_type(cls):
            raise PickleStop("NEWOBJ class argument isn't a type object")
        if not isinstance(args, tuple):
            raise PickleStop("NEWOBJ expected an arg tuple")
    elif not isinstance(args, tuple):
        args = tuple(u.rt.iterate(args, u.site))
    u.push(u.rt.new_instance(cls, args, {}, u.site, "NEWOBJ"))


@op(0x92)
def op_newobj_ex(u):
    kwargs = u.pop()
    args = u.pop()
    cls = u.pop()
    if not u.python:
        if not u.rt.could_be_type(cls):
            raise PickleStop("NEWOBJ_EX class argument must be a type")
        if not isinstance(args, tuple) or not isinstance(kwargs, dict):
            raise PickleStop("NEWOBJ_EX args must be a tuple and kwargs a dict")
    else:
        if not isinstance(args, tuple):
            args = tuple(u.rt.iterate(args, u.site))
        if not isinstance(kwargs, dict):
            raise PickleStop("NEWOBJ_EX kwargs must be a dict")
    for k in kwargs:
        if not isinstance(k, str):
            raise PickleStop("keywords must be strings")
    u.push(u.rt.new_instance(cls, args, kwargs, u.site, "NEWOBJ_EX"))


@op("i")
def op_inst(u):
    m = u.marker()
    module = _line_text(u, "ascii")
    name = _line_text(u, "ascii")
    cls = u.find_class(module, name)
    args = tuple(u.stack[m:])
    del u.stack[m:]
    u.push(u.rt.instantiate(cls, args, u.site, "INST calls"))


@op("o")
def op_obj(u):
    m = u.marker()
    if len(u.stack) - m < 1:
        raise PickleStop("unpickling stack underflow")
    cls = u.stack[m]
    args = tuple(u.stack[m + 1:])
    del u.stack[m:]
    u.push(u.rt.instantiate(cls, args, u.site, "OBJ calls"))


@op("b")
def op_build(u):
    state = u.pop()
    inst = u.top()
    u.rt.build(inst, state, u.site, python=u.python)
    if u.joblib and isinstance(inst, Inst) and isinstance(inst.cls, Glob) and \
            inst.cls.name.endswith(("NumpyArrayWrapper", "NDArrayWrapper")):
        u.stack.pop()
        u.push(u.joblib_array(inst))


@op("P")
def op_persid(u):
    line = u.readline()
    try:
        pid = line[:-1].decode("ascii")
    except UnicodeDecodeError:
        raise PickleStop("persistent IDs in protocol 0 must be ASCII strings") from None
    u.push(u.persistent_load(pid))


@op("Q")
def op_binpersid(u):
    pid = u.pop()
    u.push(u.persistent_load(pid))


# ===================================================================
# Stream drivers
# ===================================================================
def run_streams(ctx, data, location, *, python=False, encoding="ASCII", max_streams=None,
                persistent=None, joblib=False):
    """``pickle.load`` called repeatedly until end of file.

    Returns (streams completed, offset where parsing stopped, results).
    """
    pos = 0
    count = 0
    results = []
    while pos < len(data):
        if max_streams is not None and count >= max_streams:
            break
        ctx.check_time()
        loc = f"{location}[stream {count}]" if count or pos else location
        up = Unpickler(ctx, data, pos, loc, python=python, encoding=encoding,
                       persistent=persistent, joblib=joblib)
        ok, value, end = up.run()
        if not ok:
            if up.ops <= 1 and count == 0:
                ctx.note(f"{location}: not a pickle ({up.stop_reason})")
            else:
                ctx.note(f"{loc}: loading stops at offset {end}: {up.stop_reason}")
            return count, pos, results, False
        results.append(value)
        count += 1
        if end <= pos:
            break
        pos = end
    return count, pos, results, True


def is_torch_legacy(data):
    """A legacy torch.save file starts with the pickled magic number."""
    i = 0
    if data[i:i + 1] == b"\x80":
        i += 2
    if data[i:i + 1] == b"\x95":
        i += 9
    if data[i:i + 1] != b"\x8a" or len(data) < i + 2:
        return False
    n = data[i + 1]
    return int.from_bytes(data[i + 2:i + 2 + n], "little", signed=True) == TORCH_MAGIC and \
        data[i + 2 + n:i + 3 + n] == b"."


def run_torch_legacy(ctx, data, location):
    """torch.load on a legacy checkpoint: five streams, the 4th with persistent ids."""
    pos = 0
    for idx in range(5):
        ctx.check_time()
        persistent = "torch-legacy" if idx == 3 else None
        up = Unpickler(ctx, data, pos, f"{location}[stream {idx}]", encoding="utf-8",
                       persistent=persistent, torch=True)
        ok, value, end = up.run()
        if not ok:
            ctx.note(f"{location}[stream {idx}]: loading stops: {up.stop_reason}")
            return value
        if idx == 0 and value != TORCH_MAGIC:
            return value
        if idx == 1 and value != TORCH_PROTOCOL:
            return value
        if idx == 3:
            result = value
        pos = end
    return result
