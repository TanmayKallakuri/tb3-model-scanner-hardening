"""Static pickle virtual machine. No Unpickler, find_class or artifact code runs."""
from __future__ import annotations

import io
import pickletools

from modelgate.abstract import World, Obj, Ref, Lazy, UNKNOWN, LoadStopped, Detected, concrete
from modelgate.limits import AnalysisLimit
from modelgate.report import Finding

TORCH_MAGIC = 0x1950A86A20F9469CFC6C
MARK = object()


class Machine:
    def __init__(self, world, location, *, persistent=False):
        self.world = world
        self.location = location
        self.persistent = persistent
        self.streams = 0

    def one(self, stream, memo=None):
        stack, memo = [], {} if memo is None else memo
        protocol = 0
        frame_end = None
        world = self.world
        self.streams += 1
        def marked():
            i = len(stack) - 1
            while i >= 0 and stack[i] is not MARK: i -= 1
            if i < 0: raise LoadStopped("missing pickle MARK")
            values = stack[i+1:]
            del stack[i:]
            return values
        for op, arg, pos in pickletools.genops(stream):
            world.budget.tick()
            world.location = f"{self.location}:stream[{self.streams}]@{pos} {op.name}"
            name = op.name
            if frame_end is not None:
                if pos >= frame_end:
                    frame_end = None
                elif stream.tell() > frame_end:
                    raise LoadStopped("opcode extends beyond pickle frame")
            if name == "PROTO":
                if arg > 5: raise LoadStopped("unsupported pickle protocol")
                protocol = arg
            elif name == "FRAME":
                if frame_end is not None:
                    raise LoadStopped("new FRAME before end of current frame")
                if arg > len(stream.getbuffer()) - stream.tell():
                    raise LoadStopped("truncated pickle frame")
                frame_end = stream.tell() + arg
            elif name == "MARK": stack.append(MARK)
            elif name == "STOP":
                if not stack: raise LoadStopped("empty pickle stack")
                if frame_end is not None:
                    # The real unpickler read the entire frame into its private
                    # buffer. Bytes following STOP inside it are discarded.
                    stream.seek(frame_end)
                return stack[-1]
            elif name == "POP": stack.pop()
            elif name == "POP_MARK": marked()
            elif name == "DUP": stack.append(stack[-1])
            elif name in ("INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT",
                          "STRING", "BINSTRING", "SHORT_BINSTRING", "UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8",
                          "BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8"):
                stack.append(arg)
            elif name == "NONE": stack.append(None)
            elif name == "NEWTRUE": stack.append(True)
            elif name == "NEWFALSE": stack.append(False)
            elif name == "EMPTY_TUPLE": stack.append(())
            elif name == "EMPTY_LIST": stack.append([])
            elif name == "EMPTY_DICT": stack.append({})
            elif name == "EMPTY_SET": stack.append(set())
            elif name in ("TUPLE", "LIST", "DICT", "FROZENSET"):
                values = marked()
                if name == "DICT":
                    if len(values) % 2: raise LoadStopped("odd dictionary item count")
                    stack.append(dict(zip(values[::2], values[1::2])))
                else: stack.append({"TUPLE": tuple, "LIST": list, "FROZENSET": frozenset}[name](values))
            elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
                count = int(name[-1])
                if len(stack) < count: raise LoadStopped("tuple stack underflow")
                values = tuple(stack[-count:])
                del stack[-count:]
                stack.append(values)
            elif name == "APPEND":
                value = stack.pop()
                world.method(stack[-1], "append", (value,), {})
            elif name == "APPENDS":
                values = marked()
                target = stack[-1]
                if isinstance(target, list): target.extend(values)
                else:
                    for value in values: world.method(target, "append", (value,), {})
            elif name == "ADDITEMS":
                values = marked()
                target = stack[-1]
                if isinstance(target, set): target.update(values)
                else:
                    for value in values: world.call(world.attr(target, "add"), (value,))
            elif name == "SETITEM":
                value, key = stack.pop(), stack.pop()
                world.setitem(stack[-1], key, value)
            elif name == "SETITEMS":
                values = marked()
                if len(values) % 2: raise LoadStopped("odd SETITEMS count")
                for key, value in zip(values[::2], values[1::2]): world.setitem(stack[-1], key, value)
            elif name in ("PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE"):
                memo[len(memo) if name == "MEMOIZE" else int(arg)] = stack[-1]
            elif name in ("GET", "BINGET", "LONG_BINGET"):
                if int(arg) not in memo: raise LoadStopped("missing pickle memo entry")
                stack.append(memo[int(arg)])
            elif name == "GLOBAL":
                module, attr = arg.split(" ", 1)
                stack.append(world.global_ref(module, attr, protocol))
            elif name == "STACK_GLOBAL":
                attr, module = stack.pop(), stack.pop()
                if not isinstance(module, str) or not isinstance(attr, str): raise LoadStopped("GLOBAL requires strings")
                stack.append(world.global_ref(module, attr, protocol))
            elif name == "REDUCE":
                args, fn = stack.pop(), stack.pop()
                if not isinstance(args, tuple): raise LoadStopped("REDUCE arguments must be tuple")
                stack.append(world.call(fn, args))
            elif name in ("NEWOBJ", "NEWOBJ_EX"):
                kwargs = stack.pop() if name == "NEWOBJ_EX" else None
                args, cls = stack.pop(), stack.pop()
                stack.append(world.call(cls, args, kwargs, new=True))
            elif name in ("INST", "OBJ"):
                values = marked()
                if name == "INST":
                    module, attr = arg.split(" ", 1)
                    cls = world.global_ref(module, attr, protocol)
                else:
                    if not values: raise LoadStopped("OBJ requires class")
                    cls, values = values[0], values[1:]
                stack.append(world.call(cls, tuple(values), new=not values and world.is_class(cls)))
            elif name == "BUILD":
                state = stack.pop()
                obj = stack[-1]
                world.build(obj, state)
                if isinstance(obj, Obj) and obj.type in ("joblib.numpy_pickle.NumpyArrayWrapper", "joblib.numpy_pickle_compat.NDArrayWrapper", "joblib.numpy_pickle_compat.ZNDArrayWrapper"):
                    stack[-1] = self.joblib_array(obj, stream)
            elif name in ("PERSID", "BINPERSID"):
                if name == "BINPERSID": stack.pop()
                if not self.persistent: raise LoadStopped("plain pickle has no persistent ID loader")
                stack.append(Obj("torch.storage"))
            elif name in ("EXT1", "EXT2", "EXT4"):
                pair = getattr(world, "extensions", {}).get(arg)
                if pair is None: raise LoadStopped("unregistered pickle extension")
                stack.append(world.global_ref(*pair, protocol))
            elif name == "NEXT_BUFFER": raise LoadStopped("no out-of-band buffers supplied")
            elif name == "READONLY_BUFFER":
                if not stack: raise LoadStopped("empty buffer stack")
            else:
                world.unsupported(f"unmodelled pickle opcode {name}")
        raise LoadStopped("pickle did not terminate")

    def joblib_array(self, obj, stream):
        if obj.type != "joblib.numpy_pickle.NumpyArrayWrapper":
            self.world.flag("file-access", "legacy joblib array wrapper reads a sidecar path")
        dtype = obj.attrs.get("dtype")
        if not isinstance(dtype, Obj) or dtype.type != "numpy.dtype" or not concrete(dtype.args):
            self.world.unsupported("joblib array dtype is not statically known")
        from modelgate.parser_env import prepare
        prepare()
        import numpy as np
        dt = np.dtype(*dtype.args)
        dtype_state = dtype.attrs.get("state")
        hasobject, itemsize = dt.hasobject, dt.itemsize
        if isinstance(dtype_state, tuple) and len(dtype_state) == 8:
            if isinstance(dtype_state[7], int): hasobject = bool(dtype_state[7] & 1)
            if isinstance(dtype_state[5], int) and dtype_state[5] >= 0: itemsize = dtype_state[5]
        shape = obj.attrs.get("shape")
        if not isinstance(shape, tuple) or not all(isinstance(v, int) and v >= 0 for v in shape):
            raise LoadStopped("invalid joblib array shape")
        if hasobject:
            return self.one(stream)
        size = itemsize
        for dim in shape: size *= dim
        if obj.attrs.get("numpy_array_alignment_bytes") is not None:
            padding = stream.read(1)
            if not padding: raise LoadStopped("truncated joblib padding")
            stream.seek(padding[0], 1)
        if size > len(stream.getbuffer()) - stream.tell(): raise LoadStopped("truncated joblib array")
        stream.seek(size, 1)
        return Obj("numpy.ndarray", attrs={"shape": shape, "dtype": dtype})


def scan_pickle(data: bytes, location: str, *, world=None, one=False, persistent=False, torch=False) -> list[Finding]:
    world = world or World()
    stream = io.BytesIO(data)
    machine = Machine(world, location, persistent=persistent or torch)
    oldloc = world.location
    try:
        count = 0
        while stream.tell() < len(data):
            result = machine.one(stream)
            count += 1
            if count == 1 and isinstance(result, int) and result == TORCH_MAGIC:
                torch = True
                machine.persistent = False
            if torch and count == 2 and result != 1001:
                break
            if torch:
                machine.persistent = count == 3
            if torch and count >= 5: break  # remaining bytes are storages, not pickles
            if one: break
        if not count: return [Finding("unsupported", location, "empty pickle stream")]
        return []
    except Detected as exc:
        return [exc.finding]
    except LoadStopped:
        # An actual unpickler error ends execution. In particular, don't scan
        # unreachable bytes after a bad REDUCE, a missing PID, or a bad global.
        return []
    except (ValueError, IndexError, KeyError, TypeError, OverflowError) as exc:
        return [Finding("unsupported", world.location, f"cannot parse pickle: {exc}")]
    finally:
        world.location = oldloc
