"""Static pickle virtual machine, protocols 0--5.

pickletools supplies opcode decoding only. No Unpickler, pickle.load, dynamic
import, reducer, constructor, or state hook from an artifact is ever invoked.
The stack/memo contain inert abstract values and calls use audited summaries.
"""
from __future__ import annotations

import io
import pickletools

from modelgate.limits import Budget, LimitExceeded
from modelgate.pickle_values import MARK, UNKNOWN, Obj, Ref, LoadError, Unresolved, FindingRaised
from modelgate.pickle_vm import VM

TORCH_MAGIC = 119547037146038801333356
VALUE_OPS = {'INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4',
             'FLOAT', 'BINFLOAT', 'STRING', 'BINSTRING', 'SHORT_BINSTRING',
             'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE', 'BINUNICODE8',
             'BINBYTES', 'SHORT_BINBYTES', 'BINBYTES8', 'BYTEARRAY8'}


def interpret(data, vm, *, sequence=True, persistent=False, torch=False, location='pickle'):
    """Return (abstract result, consumed bytes, loader halted).

    Each pickle.load gets a new stack and memo; the simulated imported modules
    survive across streams and nested loads, as they do in a Python process.
    Legacy torch checkpoints have exactly five streams followed by raw storage.
    """
    fh = io.BytesIO(data)
    stream = 0
    result = UNKNOWN
    torch_object = UNKNOWN
    legacy = torch
    outer_location = vm.location
    try:
        while fh.tell() < len(data):
            stream += 1
            if stream > 10000:
                raise LimitExceeded('too many pickle streams')
            stack, memo = [], {}
            protocol = 0
            stopped = False
            vm.location = f'{location}:stream[{stream}]'

            def pop_mark():
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i] is MARK:
                        values = stack[i + 1:]
                        del stack[i:]
                        return values
                raise LoadError('missing MARK')

            for op, arg, pos in pickletools.genops(fh):
                vm.budget.tick()
                vm.location = f'{location}:stream[{stream}]@{pos}'
                name = op.name
                if name in VALUE_OPS:
                    stack.append(arg)
                elif name == 'NONE':
                    stack.append(None)
                elif name == 'NEWTRUE':
                    stack.append(True)
                elif name == 'NEWFALSE':
                    stack.append(False)
                elif name == 'PROTO':
                    if arg > 5:
                        raise LoadError('unknown pickle protocol')
                    protocol = arg
                elif name == 'FRAME':
                    if arg > len(data) - fh.tell():
                        raise LoadError('truncated frame')
                elif name == 'MARK':
                    stack.append(MARK)
                elif name == 'POP':
                    stack.pop()
                elif name == 'POP_MARK':
                    pop_mark()
                elif name == 'DUP':
                    stack.append(stack[-1])
                elif name in ('BINPUT', 'LONG_BINPUT', 'PUT'):
                    if arg < 0:
                        raise LoadError('negative memo index')
                    memo[arg] = stack[-1]
                elif name == 'MEMOIZE':
                    memo[len(memo)] = stack[-1]
                elif name in ('BINGET', 'LONG_BINGET', 'GET'):
                    if arg not in memo:
                        raise LoadError('memo reference not defined in this stream')
                    stack.append(memo[arg])
                elif name == 'EMPTY_TUPLE':
                    stack.append(())
                elif name == 'EMPTY_LIST':
                    stack.append([])
                elif name == 'EMPTY_DICT':
                    stack.append({})
                elif name == 'EMPTY_SET':
                    stack.append(set())
                elif name == 'TUPLE':
                    stack.append(tuple(pop_mark()))
                elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
                    n = int(name[-1])
                    values = tuple(stack[-n:])
                    del stack[-n:]
                    if len(values) != n:
                        raise LoadError('tuple stack underflow')
                    stack.append(values)
                elif name == 'LIST':
                    stack.append(pop_mark())
                elif name == 'DICT':
                    values = pop_mark()
                    if len(values) % 2:
                        raise LoadError('odd dictionary item count')
                    stack.append(dict(zip(values[::2], values[1::2])))
                elif name == 'FROZENSET':
                    stack.append(frozenset(pop_mark()))
                elif name == 'APPEND':
                    value = stack.pop()
                    if isinstance(stack[-1], list):
                        stack[-1].append(value)
                    else:
                        vm.call(vm.attr(stack[-1], 'append'), (value,))
                elif name == 'APPENDS':
                    values = pop_mark()
                    if isinstance(stack[-1], list):
                        stack[-1].extend(values)
                    else:
                        vm.call(vm.attr(stack[-1], 'extend'), (values,))
                elif name == 'ADDITEMS':
                    values = pop_mark()
                    if isinstance(stack[-1], set):
                        stack[-1].update(values)
                    else:
                        for value in values:
                            vm.call(vm.attr(stack[-1], 'add'), (value,))
                elif name == 'SETITEM':
                    value, key = stack.pop(), stack.pop()
                    vm.setitem(stack[-1], key, value)
                elif name == 'SETITEMS':
                    values = pop_mark()
                    if len(values) % 2:
                        raise LoadError('odd dictionary item count')
                    for key, value in zip(values[::2], values[1::2]):
                        vm.setitem(stack[-1], key, value)
                elif name == 'GLOBAL':
                    module, attr = arg.split(' ', 1)
                    stack.append(vm.resolve(module, attr, protocol))
                elif name == 'STACK_GLOBAL':
                    attr, module = stack.pop(), stack.pop()
                    stack.append(vm.resolve(module, attr, protocol))
                elif name == 'REDUCE':
                    args, fn = stack.pop(), stack.pop()
                    stack.append(vm.call(fn, args))
                elif name in ('NEWOBJ', 'NEWOBJ_EX'):
                    kw = stack.pop() if name == 'NEWOBJ_EX' else {}
                    args, cls = stack.pop(), stack.pop()
                    stack.append(vm.new(cls, args, kw))
                elif name in ('INST', 'OBJ'):
                    values = pop_mark()
                    if name == 'INST':
                        module, attr = arg.split(' ', 1)
                        cls = vm.resolve(module, attr, protocol)
                    else:
                        cls, values = values[0], values[1:]
                    args = tuple(values)
                    stack.append(vm.new(cls, ()) if not args and vm.is_class(cls) else vm.call(cls, args))
                elif name == 'BUILD':
                    state = stack.pop()
                    vm.build(stack[-1], state)
                    if isinstance(stack[-1], Obj) and stack[-1].cls == 'joblib.numpy_pickle.NumpyArrayWrapper':
                        _joblib_array(stack, fh, data, vm, location)
                elif name in ('PERSID', 'BINPERSID'):
                    if name == 'BINPERSID':
                        stack.pop()
                    if not persistent and not (legacy and stream == 4):
                        raise LoadError('plain pickle has no persistent-id handler')
                    stack.append(Obj('torch.Storage'))
                elif name in ('EXT1', 'EXT2', 'EXT4'):
                    if arg in vm.extension_cache:
                        stack.append(vm.extension_cache[arg])
                    elif arg in vm.inverted_registry:
                        module, attr = vm.inverted_registry[arg]
                        value = vm.resolve(module, attr, protocol)
                        vm.extension_cache[arg] = value
                        stack.append(value)
                    else:
                        raise LoadError('extension code is not registered')
                elif name == 'NEXT_BUFFER':
                    raise LoadError('no out-of-band buffer supplied by loader')
                elif name == 'READONLY_BUFFER':
                    if not stack:
                        raise LoadError('no buffer')
                elif name == 'STOP':
                    if not stack:
                        raise LoadError('empty pickle result')
                    result = stack.pop()
                    stopped = True
                    break
                else:
                    raise Unresolved('unmodeled opcode ' + name)
            if not stopped:
                raise Unresolved('pickle stream has no STOP')
            if stream == 1:
                if torch and result != TORCH_MAGIC:
                    raise LoadError('invalid torch magic')
                if result == TORCH_MAGIC and sequence:
                    legacy = True
            if legacy and stream == 2 and result != 1001:
                raise LoadError('invalid torch serialization protocol')
            if legacy and stream == 4:
                torch_object = result
            if legacy and stream >= 5:
                return torch_object, fh.tell(), False
            if not sequence:
                return result, fh.tell(), False
        return result, fh.tell(), legacy and stream < 5
    except LoadError:
        return UNKNOWN, fh.tell(), True
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        # Decoder failures mean the artifact could not be analyzed, unlike a
        # modeled TypeError from the actual unpickler's execution semantics.
        vm.finding('unsupported', 'invalid pickle: ' + str(exc))
        return UNKNOWN, fh.tell(), True
    except (Unresolved, LimitExceeded, RecursionError) as exc:
        vm.finding('unsupported', str(exc))
        return UNKNOWN, fh.tell(), True
    finally:
        vm.location = outer_location


def _joblib_array(stack, fh, data, vm, location):
    """Joblib inserts raw NumPy data immediately after its wrapper's BUILD."""
    wrapper = stack[-1]
    shape = wrapper.attrs.get('shape', ())
    dtype = wrapper.attrs.get('dtype')
    if not isinstance(dtype, Obj) or not dtype.args:
        raise Unresolved('cannot resolve joblib array dtype')
    descr = dtype.args[0]
    if not isinstance(descr, str):
        raise Unresolved('cannot resolve joblib structured array dtype')
    if descr.lstrip('<>=|').startswith('O'):
        value, end, halted = interpret(data[fh.tell():], vm, sequence=False, location=location + ':joblib-array')
        fh.seek(end, 1)
        if halted:
            raise LoadError('invalid joblib object array')
        stack[-1] = value
        return
    import re
    match = re.fullmatch(r'[<>=|]?([a-zA-Z])(\d+)', descr)
    if not match:
        raise Unresolved('cannot resolve joblib array itemsize')
    size = int(match[2]) * (4 if match[1] == 'U' else 1)
    for dim in shape:
        size *= dim
    if wrapper.attrs.get('numpy_array_alignment_bytes') is not None:
        padding = fh.read(1)
        if not padding:
            raise LoadError('missing joblib padding')
        fh.seek(padding[0], 1)
    if fh.tell() + size > len(data):
        raise LoadError('truncated joblib array')
    fh.seek(size, 1)
    stack[-1] = Obj('numpy.ndarray', (), {}, wrapper.attrs)


def scan_pickle(data, location, *, budget=None, sequence=True, persistent=False):
    vm = VM(budget or Budget())
    try:
        interpret(data, vm, sequence=sequence, persistent=persistent, location=location)
    except FindingRaised:
        pass
    return vm.findings
