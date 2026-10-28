"""Interpret pickle opcodes on symbolic values, without unpickling anything."""
from __future__ import annotations
import io
import pickletools
from modelgate.symbolic import Runtime, Symbol, UNKNOWN, Stopped, Unsupported
from modelgate.report import Finding

TORCH_MAGIC = 0x1950a86a20f9469cfc6c
MARK = object()
LITERALS = {'INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4',
            'FLOAT', 'BINFLOAT', 'STRING', 'BINSTRING', 'SHORT_BINSTRING',
            'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE', 'BINUNICODE8',
            'BINBYTES', 'SHORT_BINBYTES', 'BINBYTES8', 'BYTEARRAY8'}


def _items(stack):
    for i in range(len(stack) - 1, -1, -1):
        if stack[i] is MARK:
            values = stack[i + 1:]
            del stack[i:]
            return values
    raise Stopped('missing MARK')


def _append(runtime, target, values):
    if isinstance(target, list):
        target.extend(values)
    elif isinstance(target, Symbol):
        # APPENDS uses extend where available, otherwise repeated append.
        custom = target.attrs.get('extend')
        if custom is not None:
            runtime.call(custom, (values,))
        else:
            for value in values:
                runtime.method(target, 'append', (value,), {})
    else:
        raise Stopped('APPEND target is not a list')


def _joblib_array(fh, obj, runtime, location):
    """Joblib's numeric arrays are raw blocks interleaved with pickle opcodes."""
    shape = obj.attrs.get('shape', ())
    dtype = obj.attrs.get('dtype')
    if not isinstance(shape, (list, tuple)) or not isinstance(dtype, Symbol):
        raise Unsupported('unknown joblib array layout')
    import numpy as np
    desc = dtype.args[0] if dtype.args else None
    try:
        dt = np.dtype(desc)
    except (TypeError, ValueError):
        raise Unsupported('unknown joblib dtype') from None
    state = dtype.attrs.get('_state')
    hasobject = dt.hasobject
    itemsize = dt.itemsize
    if isinstance(state, tuple) and len(state) >= 8:
        # Structured dtypes reconstruct from V<n>, then BUILD supplies fields
        # and flags. The object flag survives nested structured/subarray types.
        hasobject = hasobject or isinstance(state[-1], int) and bool(state[-1] & 1)
        if isinstance(state[5], int) and state[5] >= 0:
            itemsize = state[5]
    if hasobject:
        start = fh.tell()
        value, used = run_pickle(fh.getbuffer()[start:].tobytes(), location + '!object-array', runtime, mode='one')
        fh.seek(start + used)
        return value
    count = 1
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            raise Unsupported('unknown joblib array shape')
        count *= dim
    align = obj.attrs.get('numpy_array_alignment_bytes')
    if align is not None:
        padding = fh.read(1)
        if not padding:
            raise Unsupported('truncated joblib padding')
        fh.seek(padding[0], 1)
    end = fh.tell() + count * itemsize
    if end > len(fh.getbuffer()):
        raise Unsupported('truncated joblib array')
    fh.seek(end)
    return Symbol('array', 'numpy.ndarray')


def run_pickle(data, location, runtime, mode='sequence', persistent=False):
    """Return (last symbolic value, consumed bytes).

    `one` is pickle.loads/load semantics, `sequence` is registry pickle
    streams, and `torch` is a legacy five-stream checkpoint. Each load has a
    fresh memo; the runtime's module namespace persists across loads.
    """
    if not data and mode != 'sequence':
        raise Stopped('nested pickle is at end of file')
    fh = io.BytesIO(data)
    result = UNKNOWN
    stream_index = 0
    torch = mode == 'torch'
    while fh.tell() < len(data):
        stream_start = fh.tell()
        stack, memo = [], {}
        protocol = 0
        frame_end = None
        complete = False
        for op, arg, pos in pickletools.genops(fh):
            runtime.tick()
            runtime.location = f'{location}:pickle[{stream_index}]@{pos}'
            runtime.opcode = op.name
            name = op.name
            if frame_end is not None:
                if pos >= frame_end:
                    frame_end = None
                elif fh.tell() > frame_end:
                    raise Stopped('pickle opcode crosses frame boundary')
            if name in LITERALS:
                stack.append(arg)
            elif name == 'NONE':
                stack.append(None)
            elif name in ('NEWTRUE', 'NEWFALSE'):
                stack.append(name == 'NEWTRUE')
            elif name == 'PROTO':
                if not 0 <= arg <= 5:
                    raise Unsupported('unknown pickle protocol')
                protocol = arg
            elif name == 'FRAME':
                if frame_end is not None and pos < frame_end:
                    raise Stopped('nested pickle frame')
                frame_end = fh.tell() + arg
                if frame_end > len(data):
                    raise Unsupported('truncated pickle frame')
            elif name == 'MARK':
                stack.append(MARK)
            elif name == 'POP':
                stack.pop()
            elif name == 'POP_MARK':
                _items(stack)
            elif name == 'DUP':
                stack.append(stack[-1])
            elif name in ('PUT', 'BINPUT', 'LONG_BINPUT'):
                if arg < 0:
                    raise Stopped('negative memo index')
                memo[arg] = stack[-1]
            elif name == 'MEMOIZE':
                memo[len(memo)] = stack[-1]
            elif name in ('GET', 'BINGET', 'LONG_BINGET'):
                if arg not in memo:
                    raise Stopped('memo reference is missing')
                stack.append(memo[arg])
            elif name in ('EMPTY_LIST', 'EMPTY_TUPLE', 'EMPTY_DICT', 'EMPTY_SET'):
                stack.append({'EMPTY_LIST': list, 'EMPTY_TUPLE': tuple, 'EMPTY_DICT': dict, 'EMPTY_SET': set}[name]())
            elif name in ('LIST', 'TUPLE', 'DICT', 'FROZENSET'):
                values = _items(stack)
                if name == 'DICT':
                    if len(values) % 2:
                        raise Stopped('odd dictionary item count')
                    value = dict(zip(values[::2], values[1::2]))
                else:
                    value = {'LIST': list, 'TUPLE': tuple, 'FROZENSET': frozenset}[name](values)
                stack.append(value)
            elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
                count = int(name[-1])
                if len(stack) < count:
                    raise Stopped('tuple stack underflow')
                values = tuple(stack[-count:])
                del stack[-count:]
                stack.append(values)
            elif name == 'APPEND':
                value = stack.pop()
                _append(runtime, stack[-1], [value])
            elif name == 'APPENDS':
                values = _items(stack)
                _append(runtime, stack[-1], values)
            elif name == 'ADDITEMS':
                values = _items(stack)
                if isinstance(stack[-1], set):
                    stack[-1].update(values)
                else:
                    for value in values:
                        runtime.method(stack[-1], 'add', (value,), {})
            elif name == 'SETITEM':
                value, key = stack.pop(), stack.pop()
                runtime.setitem(stack[-1], key, value)
            elif name == 'SETITEMS':
                values = _items(stack)
                if len(values) % 2:
                    raise Stopped('odd dictionary item count')
                for i in range(0, len(values), 2):
                    runtime.setitem(stack[-1], values[i], values[i + 1])
            elif name == 'GLOBAL':
                module, qualname = arg.split(' ', 1)
                stack.append(runtime.global_ref(module, qualname, protocol))
            elif name == 'STACK_GLOBAL':
                qualname, module = stack.pop(), stack.pop()
                stack.append(runtime.global_ref(module, qualname, protocol))
            elif name in ('EXT1', 'EXT2', 'EXT4'):
                if arg not in runtime.extension_cache:
                    if arg not in runtime.extensions:
                        raise Unsupported('pickle extension code has no static registry')
                    module, qualname = runtime.extensions[arg]
                    runtime.extension_cache[arg] = runtime.global_ref(module, qualname, protocol)
                stack.append(runtime.extension_cache[arg])
            elif name == 'REDUCE':
                args, fn = stack.pop(), stack.pop()
                stack.append(runtime.call(fn, args))
            elif name == 'NEWOBJ':
                args, cls = stack.pop(), stack.pop()
                stack.append(runtime.newobj(cls, args))
            elif name == 'NEWOBJ_EX':
                kw, args, cls = stack.pop(), stack.pop(), stack.pop()
                stack.append(runtime.newobj(cls, args, kw))
            elif name == 'INST':
                args = tuple(_items(stack))
                module, qualname = arg.split(' ', 1)
                stack.append(runtime.construct(runtime.global_ref(module, qualname, protocol), args))
            elif name == 'OBJ':
                values = _items(stack)
                stack.append(runtime.construct(values[0], tuple(values[1:])))
            elif name == 'BUILD':
                state = stack.pop()
                obj = stack[-1]
                runtime.build(obj, state)
                if isinstance(obj, Symbol) and obj.name.endswith('.NumpyArrayWrapper'):
                    value = _joblib_array(fh, obj, runtime, location)
                    # The Unpickler's memo contains the wrapper itself, like
                    # joblib; only the stack entry is replaced by the array.
                    stack[-1] = value
                    frame_end = None
            elif name in ('PERSID', 'BINPERSID'):
                pid = stack.pop() if name == 'BINPERSID' else arg
                if not (persistent or torch):
                    raise Stopped('plain pickle has no persistent id handler')
                stack.append(Symbol('instance', 'torch.Storage', (pid,)))
            elif name == 'NEXT_BUFFER':
                raise Stopped('no external buffers supplied')
            elif name == 'READONLY_BUFFER':
                if not stack:
                    raise Stopped('missing buffer')
            elif name == 'STOP':
                if not stack or stack[-1] is MARK:
                    raise Stopped('empty pickle at STOP')
                result = stack[-1]
                # A C unpickler has already fetched an entire FRAME from its
                # input; a new pickle.load starts after that frame.
                if frame_end is not None:
                    fh.seek(frame_end)
                complete = True
                break
            else:
                raise Unsupported('unknown pickle opcode ' + name)
        if not complete:
            raise Unsupported('pickle is missing STOP')
        stream_index += 1
        if mode == 'one':
            return result, fh.tell()
        if stream_index == 1:
            if isinstance(result, int) and result == TORCH_MAGIC:
                torch = True
            elif mode == 'torch':
                raise Stopped('invalid torch legacy magic')
        if torch and stream_index == 2 and result != 1001:
            raise Stopped('invalid torch legacy protocol')
        if torch and stream_index >= 5:
            return result, fh.tell()
        if fh.tell() <= stream_start:
            raise Unsupported('pickle parser made no progress')
    if torch and stream_index < 5:
        raise Stopped('truncated legacy checkpoint')
    return result, fh.tell()


def scan_pickle(data, location, persistent=False):
    runtime = Runtime()
    try:
        run_pickle(data, location, runtime, persistent=persistent)
    except Stopped:
        pass
    except Exception as exc:
        runtime.findings.append(Finding('unsupported', runtime.location or location,
                                        f'could not analyze pickle: {type(exc).__name__}: {exc}'))
    return runtime.findings
