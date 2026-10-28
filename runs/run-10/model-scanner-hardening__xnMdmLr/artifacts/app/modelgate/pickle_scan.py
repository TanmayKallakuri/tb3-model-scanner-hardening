"""A non-executing pickle virtual machine for protocols 0 through 5.

Only pickletools' opcode readers touch the serialized bytes. Values on the
stack are builtin data or inert symbols; GLOBAL never imports anything and
REDUCE/BUILD are interpreted by our explicit call semantics.
"""
from __future__ import annotations

import io
import pickletools
from modelgate.abstract import Context, Value, UNKNOWN, MARK, LoadError, LimitError, Detected
from modelgate.capabilities import Semantics
from modelgate.report import Finding

TORCH_MAGIC = 0x1950a86a20f9469cfc6c


class Interpreter(Semantics):
    def __init__(self, context=None, location='pickle', persistent=False, *, encoding=None, errors='strict', fix_imports=True):
        super().__init__(context)
        self.base_location = location
        self.persistent = persistent
        self.encoding = encoding or ('utf-8' if persistent else 'ASCII')
        self.errors = errors
        self.fix_imports = fix_imports

    def one(self, data, start=0, *, memo=None):
        self.stack = []
        self.memo = {} if memo is None else memo
        self.protocol = 0
        stream = io.BytesIO(data)
        stream.seek(start)
        stack = self.stack
        try:
            for op, arg, pos in pickletools.genops(stream):
                self.ctx.tick()
                self.op = op.name
                self.location = f'{self.base_location}@{pos}'
                n = op.name
                if n == 'PROTO':
                    if arg > 5:
                        raise LimitError('unsupported pickle protocol')
                    self.protocol = arg
                elif n == 'FRAME':
                    if pos + 9 + arg > len(data):
                        raise LoadError('truncated pickle frame')
                elif n == 'MARK':
                    stack.append(MARK)
                elif n == 'STOP':
                    if not stack or stack[-1] is MARK:
                        raise LoadError('STOP without a value')
                    return stack[-1], stream.tell()
                elif n == 'POP':
                    stack.pop()
                elif n == 'POP_MARK':
                    self.pop_mark()
                elif n == 'DUP':
                    stack.append(stack[-1])
                elif n in ('NONE', 'NEWTRUE', 'NEWFALSE'):
                    stack.append({'NONE': None, 'NEWTRUE': True, 'NEWFALSE': False}[n])
                elif n in ('INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4',
                           'FLOAT', 'BINFLOAT', 'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE',
                           'BINUNICODE8', 'BINBYTES', 'SHORT_BINBYTES', 'BINBYTES8',
                           'STRING', 'BINSTRING', 'SHORT_BINSTRING', 'BYTEARRAY8'):
                    if n in ('STRING', 'BINSTRING', 'SHORT_BINSTRING') and isinstance(arg, str):
                        raw = arg.encode('latin1')
                        arg = raw if self.encoding == 'bytes' else raw.decode(self.encoding, self.errors)
                    stack.append(arg)
                elif n in ('EMPTY_LIST', 'EMPTY_TUPLE', 'EMPTY_DICT', 'EMPTY_SET'):
                    stack.append({'EMPTY_LIST': list, 'EMPTY_TUPLE': tuple,
                                  'EMPTY_DICT': dict, 'EMPTY_SET': set}[n]())
                elif n in ('LIST', 'TUPLE', 'DICT', 'FROZENSET'):
                    items = self.pop_mark()
                    if n == 'DICT':
                        if len(items) % 2:
                            raise LoadError('odd dictionary item count')
                        stack.append(dict(zip(items[::2], items[1::2])))
                    else:
                        stack.append({'LIST': list, 'TUPLE': tuple, 'FROZENSET': frozenset}[n](items))
                elif n in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
                    count = int(n[-1])
                    if len(stack) < count or any(v is MARK for v in stack[-count:]):
                        raise LoadError('invalid tuple stack')
                    items = tuple(stack[-count:])
                    del stack[-count:]
                    stack.append(items)
                elif n in ('PUT', 'BINPUT', 'LONG_BINPUT', 'MEMOIZE'):
                    self.memo[len(self.memo) if n == 'MEMOIZE' else int(arg)] = stack[-1]
                elif n in ('GET', 'BINGET', 'LONG_BINGET'):
                    stack.append(self.memo[int(arg)])
                elif n in ('GLOBAL', 'INST'):
                    module, name = arg.split(' ', 1)
                    func = self.global_ref(module, name, self.protocol, self.fix_imports)
                    if n == 'INST':
                        args = tuple(self.pop_mark())
                        stack.append(self.call(func, args, mode='new' if not args and self.is_class(func) else 'call'))
                    else:
                        stack.append(func)
                elif n == 'STACK_GLOBAL':
                    name, module = stack.pop(), stack.pop()
                    stack.append(self.global_ref(module, name, self.protocol, self.fix_imports))
                elif n == 'REDUCE':
                    args, func = stack.pop(), stack.pop()
                    stack.append(self.call(func, args))
                elif n == 'OBJ':
                    values = self.pop_mark()
                    func, args = values[0], tuple(values[1:])
                    stack.append(self.call(func, args, mode='new' if not args and self.is_class(func) else 'call'))
                elif n in ('NEWOBJ', 'NEWOBJ_EX'):
                    kwargs = stack.pop() if n == 'NEWOBJ_EX' else {}
                    args, func = stack.pop(), stack.pop()
                    stack.append(self.call(func, args, kwargs, mode='new'))
                elif n == 'BUILD':
                    state = stack.pop()
                    self.build(stack[-1], state)
                    self.joblib_array(stack[-1], stream)
                elif n in ('APPEND', 'APPENDS', 'ADDITEMS'):
                    values = [stack.pop()] if n == 'APPEND' else self.pop_mark()
                    target = stack[-1]
                    if isinstance(target, list):
                        target.extend(values)
                    elif isinstance(target, set):
                        target.update(values)
                    elif n == 'APPENDS':
                        self.call(self.attr(target, 'extend'), (values,))
                    else:
                        for value in values:
                            self.call(self.attr(target, 'add' if n == 'ADDITEMS' else 'append'), (value,))
                elif n in ('SETITEM', 'SETITEMS'):
                    values = [stack.pop(), stack.pop()][::-1] if n == 'SETITEM' else self.pop_mark()
                    if len(values) % 2:
                        raise LoadError('odd dictionary item count')
                    for k, v in zip(values[::2], values[1::2]):
                        self.setitem(stack[-1], k, v)
                elif n in ('PERSID', 'BINPERSID'):
                    pid = arg if n == 'PERSID' else stack.pop()
                    if not self.persistent:
                        raise LoadError('plain pickle has no persistent_load handler')
                    stack.append(Value('storage', 'torch.storage', (pid,)))
                elif n in ('EXT1', 'EXT2', 'EXT4'):
                    raise LoadError('extension code is not registered')
                elif n == 'NEXT_BUFFER':
                    raise LoadError('out-of-band buffers were not supplied')
                elif n == 'READONLY_BUFFER':
                    if not stack:
                        raise LoadError('empty buffer stack')
                else:
                    raise LimitError('unimplemented opcode ' + n)
        except (IndexError, KeyError, TypeError, UnicodeError) as exc:
            raise LoadError(str(exc)) from None
        except ValueError as exc:
            # Unknown bytes after a completed pickle can be raw storage data;
            # format dispatch, rather than this parser, decides when to stop.
            raise LoadError('invalid pickle: ' + str(exc)) from None
        raise LoadError('pickle has no STOP')

    def pop_mark(self):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index] is MARK:
                values = self.stack[index + 1:]
                del self.stack[index:]
                return values
        raise LoadError('MARK not found')

    def joblib_array(self, obj, stream):
        """Joblib stores ndarray bytes immediately after a wrapper's BUILD."""
        if not isinstance(obj, Value) or obj.name not in (
                'joblib.numpy_pickle.NumpyArrayWrapper', 'joblib.numpy_pickle.NDArrayWrapper'):
            return
        shape = obj.attrs.get('shape', ())
        dtype = obj.attrs.get('dtype')
        if not isinstance(shape, tuple) or not isinstance(dtype, Value) or not dtype.args:
            raise LimitError('unknown joblib array layout')
        descr = dtype.args[0]
        from modelgate.dispatch import dtype_info
        size, objects = dtype_info(descr)
        if objects:
            machine = Interpreter(self.ctx, self.location + '!joblib-array')
            data = stream.getvalue()
            value, end = machine.one(data, stream.tell())
            stream.seek(end)
            obj.name = 'numpy.ndarray'
            obj.attrs['data'] = value
            return
        total = size
        for n in shape:
            total *= n
        if obj.attrs.get('numpy_array_alignment_bytes') is not None:
            pad = stream.read(1)
            if not pad:
                raise LoadError('missing joblib alignment')
            stream.seek(pad[0], 1)
        if stream.tell() + total > len(stream.getbuffer()):
            raise LoadError('truncated joblib array')
        stream.seek(total, 1)
        obj.name = 'numpy.ndarray'


def analyze_pickle(data, location, context, *, single=False, persistent=False, legacy=False):
    offset, number, result = 0, 0, UNKNOWN
    while offset < len(data):
        machine = Interpreter(context, f'{location}:pickle[{number}]', persistent=persistent)
        result, end = machine.one(data, offset)
        if end <= offset:
            raise LoadError('pickle parser made no progress')
        offset, number = end, number + 1
        if number == 1 and not single and (legacy or result == TORCH_MAGIC):
            if result != TORCH_MAGIC:
                raise LoadError('invalid torch legacy magic')
            protocol, offset = Interpreter(context, location + ':pickle[1]').one(data, offset)
            if protocol != 1001:
                raise LoadError('invalid torch legacy protocol')
            _, offset = Interpreter(context, location + ':pickle[2]').one(data, offset)
            result, offset = Interpreter(context, location + ':pickle[3]', persistent=True).one(data, offset)
            Interpreter(context, location + ':pickle[4]').one(data, offset)
            return result
        if single:
            return result
    return result


def scan_pickle(data: bytes, location: str, **kwargs) -> list[Finding]:
    try:
        analyze_pickle(data, location, Context(), **kwargs)
        return []
    except Detected as exc:
        return [exc.finding]
    except LoadError:
        # The loader would abort here; unreachable calls have no effects.
        return []
    except Exception as exc:
        return [Finding('unsupported', location, f'cannot analyze pickle: {exc}')]
