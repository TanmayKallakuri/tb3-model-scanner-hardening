"""A symbolic re-implementation of CPython's C unpickler (``_pickle.c``).

The machine decodes opcodes exactly as ``pickle.load`` does (stack, mark
fences, memo, protocol-dependent ``find_class``, framing, persistent ids) and
hands every operation that would run Python code to ``Interp``, which records
a finding when a capability would execute.  Where the real unpickler would
raise, the machine raises ``LoadError`` and the stream stops: nothing after
that point executes.
"""
from __future__ import annotations

import codecs
import struct

from modelgate.interp import AttrDict
from modelgate.symbolic import Exhausted, Inst, LoadError, Opaque, Sym, Unres

HIGHEST_PROTOCOL = 5
TORCH_MAGIC = 0x1950A86A20F9469CFC6C
TORCH_PROTOCOL_VERSION = 1001


class EOFLoad(LoadError):
    """Ran out of input at an opcode boundary."""


def _decode_line_int(line: bytes, base=10):
    s = line.decode("ascii", "strict").strip()
    return int(s, base)


class Machine:
    """One ``Unpickler.load()`` call over ``data[pos:]``."""

    def __init__(self, interp, data, pos=0, *, location="pickle", persistent=None,
                 encoding="ASCII", errors="strict", fix_imports=True, joblib=False):
        self.I = interp
        self.data = data
        self.start = pos
        self.pos = pos
        self.end = len(data)
        self.stack = []
        self.marks = []
        self.memo = {}
        self.proto = 0
        self.persistent = persistent  # None | "torch-zip" | "torch-legacy"
        self.encoding = encoding
        self.errors = errors
        self.fix_imports = fix_imports
        self.joblib = joblib
        self.location = location
        self.op_pos = pos

    # ------------------------------------------------------------- reading
    def read(self, n):
        if n < 0 or self.pos + n > self.end:
            raise LoadError("pickle data was truncated")
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def readline(self):
        i = self.data.find(b"\n", self.pos, self.end)
        if i < 0:
            raise LoadError("pickle data was truncated")
        line = self.data[self.pos:i]
        self.pos = i + 1
        return line

    # --------------------------------------------------------------- stack
    def fence(self):
        return self.marks[-1] if self.marks else 0

    def push(self, v):
        self.stack.append(v)

    def pop(self):
        if len(self.stack) <= self.fence():
            raise LoadError("unpickling stack underflow")
        return self.stack.pop()

    def top(self):
        if len(self.stack) <= self.fence():
            raise LoadError("unpickling stack underflow")
        return self.stack[-1]

    def marker(self):
        if not self.marks:
            raise LoadError("could not find MARK")
        return self.marks.pop()

    def pop_from(self, start):
        if start < self.fence() or start > len(self.stack):
            raise LoadError("unpickling stack underflow")
        items = self.stack[start:]
        del self.stack[start:]
        return items

    # ---------------------------------------------------------------- run
    def run(self):
        """Execute until STOP. Returns the unpickled value; raises LoadError
        (EOFLoad at a clean end of input) when the load fails."""
        I = self.I
        data = self.data
        end = self.end
        table = DISPATCH_TABLE
        loc = self.location
        stop = _stop
        while True:
            pos = self.pos
            if pos >= end:
                raise EOFLoad("Ran out of input")
            self.op_pos = pos
            I._loc = loc
            I._pos = pos
            I.steps += 1
            if not (I.steps & 1023):
                I.check_budget()
            op = data[pos]
            self.pos = pos + 1
            fn = table[op]
            if fn is None:
                raise LoadError(f"invalid load key, {op:#x}")
            if fn is stop:
                return self.pop()
            try:
                fn(self)
            except (LoadError, Exhausted):
                raise
            except RecursionError:
                raise LoadError("maximum recursion depth exceeded") from None
            except Exception as exc:  # noqa: BLE001 - a scanner bug: stop this stream, remember it
                I.internal_errors.append(f"{loc}@{pos}: {type(exc).__name__}: {exc}"[:300])
                raise LoadError("internal error") from None

    # ------------------------------------------------------------ helpers
    def decode_string(self, b: bytes):
        if self.encoding == "bytes":
            return b
        try:
            return b.decode(self.encoding, self.errors)
        except (UnicodeDecodeError, LookupError):
            raise LoadError("cannot decode STRING") from None

    def find_class(self, module, name):
        if self.persistent in ("torch-zip", "torch-legacy"):
            # torch's UnpicklerWrapper.find_class
            if isinstance(name, str) and "Storage" in name and name in TORCH_STORAGE_TYPES:
                return Opaque(f"torch.{name}")
            if module == "torch.tensor":
                module = "torch._tensor"
        return self.I.find_class(module, name, self.proto, self.fix_imports)

    def persistent_load(self, pid):
        if self.persistent is None:
            raise LoadError("A load persistent id instruction was encountered, "
                            "but no persistent_load function was specified.")
        if not isinstance(pid, tuple) or not pid:
            raise LoadError("persistent id is not a tuple")
        typename = pid[0]
        if isinstance(typename, bytes):
            typename = typename.decode("ascii", "replace")
        if self.persistent == "torch-zip":
            if typename != "storage" or len(pid) != 5:
                raise LoadError("Unknown typename for persistent_load")
            return Opaque("torch storage")
        if typename == "module":
            return pid[1] if len(pid) > 1 else None
        if typename == "storage":
            if len(pid) != 6:
                raise LoadError("bad storage persistent id")
            return Opaque("torch storage")
        raise LoadError(f"Unknown saved id type: {typename!r}")


TORCH_STORAGE_TYPES = {
    "DoubleStorage", "FloatStorage", "HalfStorage", "LongStorage", "IntStorage", "ShortStorage",
    "CharStorage", "ByteStorage", "BoolStorage", "BFloat16Storage", "ComplexDoubleStorage",
    "ComplexFloatStorage", "QUInt8Storage", "QInt8Storage", "QInt32Storage", "QUInt4x2Storage",
    "QUInt2x4Storage", "UntypedStorage", "TypedStorage", "Float8_e5m2Storage", "Float8_e4m3fnStorage",
}

# ------------------------------------------------------------------ opcodes
DISPATCH = {}


def op(code):
    def deco(fn):
        DISPATCH[ord(code) if isinstance(code, str) else code] = fn
        return fn
    return deco


@op(".")
def _stop(m):  # handled in run()
    raise AssertionError


@op(0x80)
def _proto(m):
    p = m.read(1)[0]
    if p > HIGHEST_PROTOCOL:
        raise LoadError(f"unsupported pickle protocol: {p}")
    m.proto = p


@op(0x95)
def _frame(m):
    n = struct.unpack("<Q", m.read(8))[0]
    if n > 0x7FFFFFFFFFFFFFFF or m.pos + n > m.end:
        raise LoadError("pickle data was truncated")


@op("N")
def _none(m):
    m.push(None)


@op(0x88)
def _true(m):
    m.push(True)


@op(0x89)
def _false(m):
    m.push(False)


@op("I")
def _int(m):
    line = m.readline()
    if len(line) < 1:
        raise LoadError("pickle data was truncated")
    if line == b"00":
        m.push(False)
        return
    if line == b"01":
        m.push(True)
        return
    s = line.decode("latin-1")
    try:
        v = int(s.rstrip("L"), 0) if s.strip().lower().startswith(("0x", "0o", "0b", "-0x")) else int(s.rstrip("L"))
    except ValueError:
        try:
            v = int(s, 0)
        except ValueError:
            raise LoadError("could not convert string to int") from None
    m.push(v)


@op("J")
def _binint(m):
    m.push(struct.unpack("<i", m.read(4))[0])


@op("K")
def _binint1(m):
    m.push(m.read(1)[0])


@op("M")
def _binint2(m):
    m.push(struct.unpack("<H", m.read(2))[0])


@op("L")
def _long(m):
    line = m.readline()
    if len(line) < 1:
        raise LoadError("pickle data was truncated")
    s = line.decode("latin-1")
    if s.endswith("L"):
        s = s[:-1]
    try:
        m.push(int(s, 0) if s.strip().lower().startswith(("0x", "-0x", "0o", "0b")) else int(s))
    except ValueError:
        try:
            m.push(int(s, 0))
        except ValueError:
            raise LoadError("invalid literal for int()") from None


def _decode_long(b):
    return int.from_bytes(b, "little", signed=True) if b else 0


@op(0x8A)
def _long1(m):
    n = m.read(1)[0]
    m.push(_decode_long(m.read(n)))


@op(0x8B)
def _long4(m):
    n = struct.unpack("<i", m.read(4))[0]
    if n < 0:
        raise LoadError("LONG pickle has negative byte count")
    if n > 1 << 20:
        b = m.read(n)
        m.push(Opaque("huge int"))
        return
    m.push(_decode_long(m.read(n)))


@op("F")
def _float(m):
    line = m.readline()
    try:
        m.push(float(line.decode("latin-1")))
    except ValueError:
        raise LoadError("could not convert string to float") from None


@op("G")
def _binfloat(m):
    m.push(struct.unpack(">d", m.read(8))[0])


@op("S")
def _string(m):
    line = m.readline()
    if len(line) >= 2 and line[0] == line[-1] and line[:1] in (b"'", b'"'):
        body = line[1:-1]
    else:
        raise LoadError("the STRING opcode argument must be quoted")
    try:
        raw = codecs.escape_decode(body)[0]
    except Exception:  # noqa: BLE001
        raise LoadError("invalid \\x escape in STRING") from None
    m.push(m.decode_string(raw))


@op("T")
def _binstring(m):
    n = struct.unpack("<i", m.read(4))[0]
    if n < 0:
        raise LoadError("BINSTRING pickle has negative byte count")
    m.push(m.decode_string(m.read(n)))


@op("U")
def _short_binstring(m):
    n = m.read(1)[0]
    m.push(m.decode_string(m.read(n)))


@op("B")
def _binbytes(m):
    n = struct.unpack("<I", m.read(4))[0]
    m.push(m.read(n))


@op("C")
def _short_binbytes(m):
    n = m.read(1)[0]
    m.push(m.read(n))


@op(0x8E)
def _binbytes8(m):
    n = struct.unpack("<Q", m.read(8))[0]
    if n > 0x7FFFFFFFFFFFFFFF:
        raise LoadError("BINBYTES8 exceeds system's maximum size")
    m.push(m.read(n))


@op(0x96)
def _bytearray8(m):
    n = struct.unpack("<Q", m.read(8))[0]
    if n > 0x7FFFFFFFFFFFFFFF:
        raise LoadError("BYTEARRAY8 exceeds system's maximum size")
    m.push(bytearray(m.read(n)))


@op(0x97)
def _next_buffer(m):
    raise LoadError("pickle stream refers to out-of-band data but no *buffers* argument was given")


@op(0x98)
def _readonly_buffer(m):
    v = m.top()
    if isinstance(v, bytearray):
        m.stack[-1] = bytes(v)
    elif isinstance(v, bytes) or isinstance(v, Sym):
        pass
    else:
        raise LoadError("a bytes-like object is required")


@op("V")
def _unicode(m):
    line = m.readline()
    try:
        m.push(line.decode("raw_unicode_escape"))
    except UnicodeDecodeError:
        raise LoadError("cannot decode UNICODE") from None


def _utf8(b):
    try:
        return b.decode("utf-8", "surrogatepass")
    except UnicodeDecodeError:
        raise LoadError("cannot decode BINUNICODE") from None


@op("X")
def _binunicode(m):
    n = struct.unpack("<I", m.read(4))[0]
    m.push(_utf8(m.read(n)))


@op(0x8C)
def _short_binunicode(m):
    n = m.read(1)[0]
    m.push(_utf8(m.read(n)))


@op(0x8D)
def _binunicode8(m):
    n = struct.unpack("<Q", m.read(8))[0]
    if n > 0x7FFFFFFFFFFFFFFF:
        raise LoadError("BINUNICODE8 exceeds system's maximum size")
    m.push(_utf8(m.read(n)))


@op(")")
def _empty_tuple(m):
    m.push(())


@op(0x85)
def _tuple1(m):
    a = m.pop()
    m.push((a,))


@op(0x86)
def _tuple2(m):
    b = m.pop()
    a = m.pop()
    m.push((a, b))


@op(0x87)
def _tuple3(m):
    c = m.pop()
    b = m.pop()
    a = m.pop()
    m.push((a, b, c))


@op("t")
def _tuple(m):
    k = m.marker()
    m.push(tuple(m.pop_from(k)))


@op("]")
def _empty_list(m):
    m.push([])


@op("}")
def _empty_dict(m):
    m.push({})


@op(0x8F)
def _empty_set(m):
    m.push(set())


@op("l")
def _list(m):
    k = m.marker()
    m.push(list(m.pop_from(k)))


@op("d")
def _dict(m):
    k = m.marker()
    items = m.pop_from(k)
    if len(items) % 2:
        raise LoadError("odd number of items for DICT")
    d = {}
    for i in range(0, len(items), 2):
        try:
            d[items[i]] = items[i + 1]
        except TypeError:
            raise LoadError("unhashable type") from None
    m.push(d)


@op(0x91)
def _frozenset(m):
    k = m.marker()
    items = m.pop_from(k)
    try:
        m.push(frozenset(items))
    except TypeError:
        raise LoadError("unhashable type") from None


@op("a")
def _append(m):
    _do_append(m, len(m.stack) - 1)


@op("e")
def _appends(m):
    _do_append(m, m.marker())


def _do_append(m, x):
    n = len(m.stack)
    if x > n or x <= m.fence():
        raise LoadError("unpickling stack underflow")
    if n == x:
        return
    target = m.stack[x - 1]
    items = m.stack[x:]
    del m.stack[x:]
    m.I.extend(target, items)


@op("s")
def _setitem(m):
    _do_setitems(m, len(m.stack) - 2)


@op("u")
def _setitems(m):
    _do_setitems(m, m.marker())


def _do_setitems(m, x):
    n = len(m.stack)
    if x > n or x <= m.fence():
        raise LoadError("unpickling stack underflow")
    if n == x:
        return
    if (n - x) % 2:
        raise LoadError("odd number of items for SETITEMS")
    target = m.stack[x - 1]
    items = m.stack[x:]
    del m.stack[x:]
    for i in range(0, len(items), 2):
        m.I.setitem(target, items[i], items[i + 1])


@op(0x90)
def _additems(m):
    k = m.marker()
    n = len(m.stack)
    if k > n or k <= m.fence():
        raise LoadError("unpickling stack underflow")
    if n == k:
        return
    target = m.stack[k - 1]
    items = m.stack[k:]
    del m.stack[k:]
    m.I.add_items(target, items)


@op("(")
def _mark(m):
    m.marks.append(len(m.stack))


@op("0")
def _pop(m):
    if m.marks and m.marks[-1] == len(m.stack):
        m.marks.pop()
    else:
        m.pop()


@op("1")
def _pop_mark(m):
    k = m.marker()
    del m.stack[k:]


@op("2")
def _dup(m):
    m.push(m.top())


# ----------------------------------------------------------------- memo
def _memo_get(m, idx):
    try:
        return m.memo[idx]
    except KeyError:
        raise LoadError(f"Memo value not found at index {idx}") from None


@op("g")
def _get(m):
    line = m.readline()
    if not line:
        raise LoadError("pickle data was truncated")
    try:
        idx = _decode_line_int(line)
    except ValueError:
        raise LoadError("invalid GET argument") from None
    m.push(_memo_get(m, idx))


@op("h")
def _binget(m):
    m.push(_memo_get(m, m.read(1)[0]))


@op("j")
def _long_binget(m):
    m.push(_memo_get(m, struct.unpack("<I", m.read(4))[0]))


@op("p")
def _put(m):
    line = m.readline()
    if not line:
        raise LoadError("pickle data was truncated")
    try:
        idx = _decode_line_int(line)
    except ValueError:
        raise LoadError("invalid PUT argument") from None
    if idx < 0:
        raise LoadError("negative PUT argument")
    m.memo[idx] = m.top()


@op("q")
def _binput(m):
    idx = m.read(1)[0]
    m.memo[idx] = m.top()


@op("r")
def _long_binput(m):
    idx = struct.unpack("<I", m.read(4))[0]
    m.memo[idx] = m.top()


@op(0x94)
def _memoize(m):
    v = m.top()
    m.memo[len(m.memo)] = v


# ------------------------------------------------------------- globals
@op("c")
def _global(m):
    mod = m.readline()
    if not mod:
        raise LoadError("pickle data was truncated")
    name = m.readline()
    if not name:
        raise LoadError("pickle data was truncated")
    try:
        mod_s = mod.decode("utf-8")
        name_s = name.decode("utf-8")
    except UnicodeDecodeError:
        raise LoadError("cannot decode GLOBAL") from None
    m.push(m.find_class(mod_s, name_s))


@op(0x93)
def _stack_global(m):
    name = m.pop()
    mod = m.pop()
    if type(name) is not str or type(mod) is not str:
        raise LoadError("STACK_GLOBAL requires str")
    m.push(m.find_class(mod, name))


def _ext(m, code):
    if code <= 0:
        raise LoadError("EXT specifies code <= 0")
    key = m.I.ext_registry.get(code)
    if key is None:
        raise LoadError(f"unregistered extension code {code}")
    module, name = key
    if not isinstance(module, str) or not isinstance(name, str):
        raise LoadError("bad extension registry entry")
    m.push(m.find_class(module, name))


@op(0x82)
def _ext1(m):
    _ext(m, m.read(1)[0])


@op(0x83)
def _ext2(m):
    _ext(m, struct.unpack("<H", m.read(2))[0])


@op(0x84)
def _ext4(m):
    _ext(m, struct.unpack("<i", m.read(4))[0])


# --------------------------------------------------- object construction
@op("R")
def _reduce(m):
    args = m.pop()
    func = m.pop()
    m.push(call_with_argtuple(m.I, func, args))


def call_with_argtuple(I, func, args):
    if isinstance(args, tuple):
        return I.call(func, args)
    if isinstance(args, Opaque):
        return I.call(func, (Opaque("argument"),))
    if isinstance(args, Inst) and isinstance(args.cls, Unres):
        return I.call(func, (Opaque("argument"),))
    raise LoadError("argument list must be a tuple")


@op("b")
def _build(m):
    state = m.pop()
    inst = m.top()
    m.I.build(inst, state)
    if m.joblib and isinstance(inst, Inst) and isinstance(inst.cls, Unres) and \
            inst.cls.name.endswith(("numpy_pickle.NumpyArrayWrapper",)):
        _joblib_array(m, inst)


def _joblib_array(m, wrapper):
    """joblib's NumpyUnpickler reads the array bytes that follow the
    wrapper's BUILD directly from the stream."""
    st = wrapper.attrs
    shape = st.get("shape")
    dtype = st.get("dtype")
    itemsize, hasobject = _dtype_info(dtype)
    if hasobject:
        sub = Machine(m.I, m.data, m.pos, location=f"{m.location}>array", encoding=m.encoding)
        try:
            sub.run()
        finally:
            m.pos = max(m.pos, sub.pos)
        m.stack[-1] = Opaque("ndarray")
        return
    if st.get("numpy_array_alignment_bytes") is not None and "numpy_array_alignment_bytes" in st:
        pad = m.read(1)[0]
        m.read(pad)
    count = 1
    if isinstance(shape, tuple):
        for d in shape:
            if not isinstance(d, int):
                raise LoadError("bad array shape")
            count *= d
    if itemsize is None:
        raise LoadError("unknown dtype")
    m.read(count * itemsize)
    m.stack[-1] = Opaque("ndarray")


def _dtype_info(dtype):
    """(itemsize, hasobject) of a pickled numpy dtype."""
    descr = None
    if isinstance(dtype, Inst) and dtype.args:
        descr = dtype.args[0]
    if not isinstance(descr, str):
        return None, False
    try:
        import numpy as np

        dt = np.dtype(descr)
        if dt.itemsize == 0 and descr[:1] in ("U", "S", "V"):
            st = dtype.state if dtype.state is not None else None
            if isinstance(st, tuple) and len(st) >= 6 and isinstance(st[5], int) and st[5] > 0:
                return st[5], False
        return dt.itemsize, bool(dt.hasobject)
    except Exception:  # noqa: BLE001
        return None, descr.startswith("O")


@op("i")
def _inst(m):
    k = m.marker()
    mod = m.readline()
    if not mod:
        raise LoadError("pickle data was truncated")
    name = m.readline()
    if not name:
        raise LoadError("pickle data was truncated")
    try:
        mod_s = mod.decode("ascii")
        name_s = name.decode("ascii")
    except UnicodeDecodeError:
        raise LoadError("cannot decode INST") from None
    cls = m.find_class(mod_s, name_s)
    args = tuple(m.pop_from(k))
    m.push(_instantiate(m.I, cls, args))


@op("o")
def _obj(m):
    k = m.marker()
    if len(m.stack) - k < 1:
        raise LoadError("unpickling stack underflow")
    items = m.pop_from(k)
    cls, args = items[0], tuple(items[1:])
    m.push(_instantiate(m.I, cls, args))


def _instantiate(I, cls, args):
    if not args:
        return I.instantiate_noargs(cls)
    return I.call(cls, args)


@op(0x81)
def _newobj(m):
    args = m.pop()
    cls = m.pop()
    if not isinstance(args, tuple):
        if isinstance(args, Opaque):
            args = (Opaque("argument"),)
        else:
            raise LoadError("NEWOBJ args argument must be a tuple")
    m.push(m.I.new_object(cls, args, {}, "NEWOBJ"))


@op(0x92)
def _newobj_ex(m):
    kwargs = m.pop()
    args = m.pop()
    cls = m.pop()
    if not isinstance(args, tuple):
        raise LoadError("NEWOBJ_EX args argument must be a tuple")
    if not isinstance(kwargs, dict) or isinstance(kwargs, AttrDict) and False:
        raise LoadError("NEWOBJ_EX kwargs argument must be a dict")
    if not all(isinstance(k, str) for k in kwargs):
        raise LoadError("keywords must be strings")
    m.push(m.I.new_object(cls, args, dict(kwargs), "NEWOBJ_EX"))


@op("P")
def _persid(m):
    line = m.readline()
    try:
        pid = line.decode("ascii")
    except UnicodeDecodeError:
        raise LoadError("persistent IDs in protocol 0 must be ASCII strings") from None
    m.push(m.persistent_load(pid))


@op("Q")
def _binpersid(m):
    pid = m.pop()
    m.push(m.persistent_load(pid))


DISPATCH_TABLE = [DISPATCH.get(i) for i in range(256)]


# ------------------------------------------------------------- drivers
def load_one(interp, data, pos=0, **kw):
    """One ``pickle.load``; returns (value, end_position). Raises LoadError."""
    m = Machine(interp, data, pos, **kw)
    try:
        v = m.run()
    except LoadError as exc:
        exc.end_pos = m.pos
        raise
    return v, m.pos
