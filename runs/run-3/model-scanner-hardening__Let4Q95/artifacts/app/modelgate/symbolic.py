"""Explicit, side-effect-free models of Python calls used in pickle recipes.

An artifact-selected callable is never looked up in the interpreter. All
operations below act only on our own symbolic values or literal primitives.
"""
from __future__ import annotations

import base64
import binascii
import codecs
import json
import operator

from modelgate.analysis import MAX_BYTES, Unsupported
from modelgate.pickle_scan import (Bound, Buffer, Getter, Lazy, Module, Object,
                                   Partial, Ref, UNKNOWN, Unknown, as_bytes,
                                   canonical, typename)
from modelgate.policy import CALLS, NESTED_LOADERS


def pure(value, seen=None):
    """Whether a value contains only inert, exact builtin types."""
    if type(value) in (str, bytes, bytearray, int, float, bool, complex, type(None), slice):
        return True
    if type(value) in (tuple, list, dict, set, frozenset):
        seen = seen or set()
        if id(value) in seen:
            return False
        seen.add(id(value))
        values = list(value.items()) if type(value) is dict else value
        return all(pure(v, seen) for v in values)
    return False


def bounded(value):
    if isinstance(value, (str, bytes, bytearray, list, tuple, dict, set, frozenset)) and len(value) > MAX_BYTES:
        raise Unsupported('symbolic value exceeds analysis limit')
    return value


def sequence(vm, value):
    if type(value) in (list, tuple, set, frozenset, dict, str, bytes, bytearray, range):
        if len(value) > 1_000_000:
            raise Unsupported('symbolic iteration limit exceeded')
        return list(value)
    if isinstance(value, Lazy):
        items = sequence(vm, value.items)
        if value.kind == 'map':
            return [vm.call(value.func, (x,)) for x in items]
        if value.kind == 'starmap':
            return [vm.call(value.func, tuple(x)) for x in items]
        if value.kind in ('filter', 'filterfalse'):
            return [x for x in items if bool(vm.call(value.func, (x,)) if value.func is not None else x)
                    == (value.kind == 'filter')]
        if value.kind == 'itercall':
            # Executing next() is sufficient to expose a capability; a pure
            # callable iteration with an unknown stopping point is bounded.
            result = vm.call(value.func, ())
            return [] if result == value.items else [result]
    raise Unsupported('cannot statically enumerate iterator')


def invoke(vm, func, args, kwargs):
    if isinstance(func, Partial):
        return vm.call(func.func, tuple(func.args) + args, {**(func.keywords or {}), **kwargs})
    if isinstance(func, Getter):
        if not args:
            raise Unsupported('getter called without a receiver')
        if func.kind == 'attrgetter':
            out = []
            for path in func.args:
                obj = args[0]
                for attr in path.split('.'):
                    obj = vm.attr(obj, attr)
                out.append(obj)
            return out[0] if len(out) == 1 else tuple(out)
        if func.kind == 'itemgetter':
            out = [vm.getitem(args[0], key) for key in func.args]
            return out[0] if len(out) == 1 else tuple(out)
        if func.kind == 'methodcaller':
            return vm.call(vm.attr(args[0], func.args[0]), func.args[1:], func.keywords)
    if isinstance(func, Bound):
        return bound_call(vm, func.obj, func.name, args, kwargs)
    if isinstance(func, Object):
        if func.cls == 'builtins.staticmethod':
            return vm.call(func.args[0], args, kwargs)
        if func.cls == 'types.MethodType':
            return vm.call(func.args[0], (func.args[1],) + args, kwargs)
        if func.cls == 'builtins.function':
            vm.finding('code-execution', f'{vm.op} serialized Python function')
            return UNKNOWN
        if func.cls.startswith('xmlrpc.client.'):
            vm.finding('network', f'{vm.op} XML-RPC method')
            return UNKNOWN
        custom = func.attrs.get('__call__')
        if custom is not None:
            return vm.call(custom, args, kwargs)
        return invoke(vm, Ref(func.cls + '.__call__'), args, kwargs)
    if not isinstance(func, Ref):
        raise Unsupported('executed callable cannot be resolved statically')
    name = canonical(func.name)
    arg = args[0] if args else None

    # Attribute access and import are reference construction, not execution.
    if name in ('builtins.getattr', 'builtins.hasattr'):
        value = vm.attr(arg, args[1], args[2] if len(args) > 2 else UNKNOWN)
        return not isinstance(value, Unknown) if name.endswith('hasattr') else value
    if name in ('builtins.__import__', 'importlib.import_module'):
        if not isinstance(arg, str):
            raise Unsupported('module name is not known')
        if name == 'builtins.__import__':
            fromlist = args[3] if len(args) > 3 else kwargs.get('fromlist')
            if not fromlist:
                arg = arg.split('.')[0]
        return Module(canonical(arg))
    if name in ('builtins.vars',):
        return vm.attr(arg, '__dict__')
    if name == 'builtins.globals':
        return Object('module.namespace', (Module('builtins'),))
    if name in ('builtins.setattr', 'builtins.delattr'):
        if isinstance(arg, (Object, Partial)) and len(args) > 1:
            if name.endswith('setattr'):
                arg.attrs[args[1]] = args[2]
            else:
                arg.attrs.pop(args[1], None)
        return None
    if name in ('operator.attrgetter', 'operator.itemgetter', 'operator.methodcaller'):
        return Getter(name.rsplit('.', 1)[1], args, kwargs)
    if name == 'functools.partial':
        return Partial(arg, args[1:], dict(kwargs))
    if name == 'functools.partial.__setstate__':
        vm.build(arg, args[1])
        return None
    if name in ('copyreg._reconstructor', 'copyreg.__newobj__', 'copyreg.__newobj_ex__'):
        return vm.newobj(arg, args[1:] if name != 'copyreg.__newobj_ex__' else args[1],
                         args[2] if name == 'copyreg.__newobj_ex__' else {})
    if name == 'builtins.type':
        if len(args) == 1:
            return Ref(typename(arg) or 'builtins.object')
        if len(args) == 3 and isinstance(args[2], dict):
            return Object('dynamic.class', args, dict(args[2]))
    if name in ('builtins.staticmethod', 'builtins.classmethod', 'types.MethodType', 'builtins.super'):
        return Object(name, args)

    if name in ('io.BytesIO', 'io.StringIO'):
        return Buffer(arg if args else (b'' if name.endswith('BytesIO') else ''))
    if name in ('pickle.Unpickler', 'pickle._Unpickler', 'dill.Unpickler'):
        return Object('pickle.Unpickler', args)
    if name in NESTED_LOADERS:
        return nested_load(vm, name, args, kwargs)
    if name in ('marshal.loads', 'marshal.load'):
        # Do not use marshal on attacker-supplied data. Merely producing a
        # code object is inert; FunctionType or a call is the capability.
        if name.endswith('.load') and not isinstance(arg, Buffer):
            vm.finding('file-access', f'{vm.op} {name} from external file')
        return Object('serialized.code', args)
    if name in ('types.CodeType', 'dill._dill._create_code'):
        return Object('serialized.code', args)
    if name == 'logging.config.dictConfig':
        return logging_config(vm, arg)
    if name == 'logging.config.DictConfigurator':
        return Object(name, args)
    if name == 'logging.config.fileConfig':
        vm.finding('code-execution' if isinstance(arg, Buffer) else 'file-access', f'{vm.op} {name}')
        return UNKNOWN
    if name in ('timeit.Timer', 'timeit.timeit', 'timeit.repeat'):
        statement = args[0] if args else kwargs.get('stmt', None)
        setup = args[1] if len(args) > 1 else kwargs.get('setup', None)
        if isinstance(statement, (str, bytes)) or isinstance(setup, (str, bytes)):
            vm.finding('code-execution', f'{vm.op} {name} compiles supplied source')
        if name != 'timeit.Timer':
            for callback in (setup, statement):
                if isinstance(callback, (Ref, Partial, Bound, Object, Getter)):
                    vm.call(callback, ())
        return Object('timeit.Timer', args, dict(kwargs)) if name == 'timeit.Timer' else 0.0

    # Only constructors that actually connect have a network capability.
    if name in ('ftplib.FTP', 'ftplib.FTP_TLS', 'smtplib.SMTP', 'smtplib.SMTP_SSL', 'telnetlib.Telnet'):
        host = arg if args else kwargs.get('host', '')
        if host:
            vm.finding('network', f'{vm.op} {name} with host')
        return Object(name, args, dict(kwargs))
    if name in ('socket.socket', 'http.client.HTTPConnection', 'http.client.HTTPSConnection',
                'xmlrpc.client.ServerProxy', 'urllib3.PoolManager', 'urllib3.HTTPConnectionPool',
                'requests.Session', 'requests.sessions.Session'):
        return Object('requests.sessions.Session' if name == 'requests.Session' else name, args, dict(kwargs))

    # Opening an in-memory file for compression/archive parsing is inert.
    if name in ('gzip.GzipFile', 'gzip.open', 'bz2.BZ2File', 'bz2.open', 'lzma.LZMAFile', 'lzma.open'):
        source = kwargs.get('fileobj', arg)
        if isinstance(source, Buffer):
            from modelgate.dispatch import decompress
            content = source.data[source.pos:]
            source.pos = len(source.data)
            return Buffer(decompress(content, vm.budget))
        vm.finding('file-access', f'{vm.op} {name} opens a path')
        return Object(name, args)
    if name in ('zipfile.ZipFile', 'tarfile.open', 'tarfile.TarFile.open'):
        source = kwargs.get('fileobj', arg)
        if isinstance(source, Buffer):
            return Object(name, args, {'source': source})
        vm.finding('file-access', f'{vm.op} {name} opens a path')
        return Object(name, args)
    if name in ('gzip.decompress', 'bz2.decompress', 'lzma.decompress', 'zlib.decompress'):
        from modelgate.dispatch import decompress
        if as_bytes(arg) is None:
            raise Unsupported('compressed data is not literal bytes')
        return decompress(bytes(arg), vm.budget, kind=name.split('.')[0])

    # Known eager capability calls. NEWOBJ never reaches this path.
    if name in CALLS:
        vm.finding(CALLS[name], f'{vm.op} {name}')
        if name in ('types.FunctionType', 'types.LambdaType', 'cloudpickle.cloudpickle._make_function', 'dill._dill._create_function'):
            return Object('builtins.function', args)
        return Object(name, args)
    if name.endswith('.__init__') and name[:-9] in CALLS:
        vm.finding(CALLS[name[:-9]], f'{vm.op} {name}')
        return None

    # Literal transformations let indirect calls keep their actual targets.
    if name in ('operator.getitem', 'operator.__getitem__'):
        return vm.getitem(arg, args[1])
    if name in ('operator.setitem', 'operator.__setitem__'):
        vm.setitem(arg, args[1], args[2])
        return None
    if name == 'operator.call':
        return vm.call(arg, args[1:], kwargs)
    if name in ('operator.add', 'operator.concat', 'operator.__add__', 'operator.mul', 'operator.__mul__',
                'operator.mod', 'operator.__mod__', 'operator.sub', 'operator.truediv', 'operator.floordiv',
                'operator.pow', 'operator.and_', 'operator.or_', 'operator.xor', 'operator.lshift', 'operator.rshift'):
        return literal_operator(name, args)
    if name in ('operator.index', 'operator.neg', 'operator.pos', 'operator.not_', 'operator.truth', 'operator.invert') and pure(args):
        return getattr(operator, name.split('.')[-1])(*args)
    if name in ('builtins.map', 'itertools.starmap', 'builtins.filter', 'itertools.filterfalse'):
        if name == 'builtins.map' and len(args) > 2:
            rows = list(zip(*(sequence(vm, v) for v in args[1:])))
            return Lazy('starmap', arg, rows)
        return Lazy(name.rsplit('.', 1)[1], arg, args[1])
    if name == 'builtins.iter':
        return Lazy('itercall', arg, args[1]) if len(args) == 2 else Lazy('map', Ref('modelgate.identity'), arg)
    if name == 'modelgate.identity':
        return arg
    if name == 'builtins.next':
        if isinstance(arg, Lazy) and arg.kind == 'itercall':
            return vm.call(arg.func, ())
        items = sequence(vm, arg)
        return items[0] if items else (args[1] if len(args) > 1 else UNKNOWN)
    if name in ('builtins.list', 'builtins.tuple', 'builtins.set', 'builtins.frozenset'):
        values = sequence(vm, arg) if args else []
        return {'builtins.list': list, 'builtins.tuple': tuple, 'builtins.set': set, 'builtins.frozenset': frozenset}[name](values)
    if name in ('builtins.dict', 'collections.OrderedDict', 'collections.UserDict'):
        if not args:
            return dict(kwargs)
        return {**(dict(arg) if isinstance(arg, dict) else dict(sequence(vm, arg))), **kwargs}
    if name == 'collections.defaultdict':
        return Object(name, args, {'_items': dict(args[1]) if len(args) > 1 else {}})
    if name == 'functools.reduce':
        items = sequence(vm, args[1])
        acc = args[2] if len(args) > 2 else items.pop(0)
        for item in items:
            acc = vm.call(arg, (acc, item))
        return acc
    if name in ('builtins.sorted', 'builtins.min', 'builtins.max'):
        items = sequence(vm, arg) if len(args) == 1 else list(args)
        key = kwargs.get('key')
        if key is not None:
            pairs = [(vm.call(key, (item,)), i, item) for i, item in enumerate(items)]
            if not pure([p[0] for p in pairs]):
                return items if name.endswith('sorted') else UNKNOWN
            items = [p[2] for p in sorted(pairs, reverse=bool(kwargs.get('reverse', False)))]
        elif pure(items):
            items = sorted(items, reverse=bool(kwargs.get('reverse', False)))
        return items if name.endswith('sorted') else (items[0 if name.endswith('min') else -1] if items else UNKNOWN)
    if name in ('builtins.any', 'builtins.all', 'builtins.sum'):
        items = sequence(vm, arg)
        if pure(items):
            return {'builtins.any': any, 'builtins.all': all, 'builtins.sum': sum}[name](items)
        return UNKNOWN
    if name == 'builtins.zip':
        return list(zip(*(sequence(vm, x) for x in args)))
    if name == 'builtins.enumerate':
        return list(enumerate(sequence(vm, arg), args[1] if len(args) > 1 else 0))
    if name == 'builtins.range' and pure(args):
        result = range(*args)
        if len(result) > 1_000_000:
            raise Unsupported('symbolic range too large')
        return result
    if name in ('builtins.str', 'builtins.bytes', 'builtins.bytearray', 'builtins.int', 'builtins.float',
                'builtins.bool', 'builtins.complex', 'builtins.chr', 'builtins.ord', 'builtins.repr',
                'builtins.len', 'builtins.slice', 'builtins.hex', 'builtins.oct', 'builtins.bin'):
        table = {'str': str, 'bytes': bytes, 'bytearray': bytearray, 'int': int, 'float': float,
                 'bool': bool, 'complex': complex, 'chr': chr, 'ord': ord, 'repr': repr,
                 'len': len, 'slice': slice, 'hex': hex, 'oct': oct, 'bin': bin}
        if name in ('builtins.bytes', 'builtins.bytearray') and isinstance(arg, int) and arg > MAX_BYTES:
            raise Unsupported('symbolic byte allocation too large')
        if pure(args):
            return bounded(table[name.split('.')[-1]](*args, **kwargs))
        return UNKNOWN
    if name in ('_codecs.encode', 'codecs.encode', '_codecs.decode', 'codecs.decode'):
        return literal_codec(name, args)
    if name.startswith('base64.') and name.split('.')[-1] in (
            'b64decode', 'standard_b64decode', 'urlsafe_b64decode', 'b32decode', 'b16decode', 'b85decode', 'a85decode',
            'b64encode', 'standard_b64encode', 'urlsafe_b64encode', 'encodebytes', 'decodebytes') and pure(args) and pure(kwargs):
        return bounded(getattr(base64, name.split('.')[-1])(*args, **kwargs))
    if name in ('binascii.a2b_base64', 'binascii.unhexlify', 'binascii.a2b_hex', 'binascii.hexlify') and pure(args):
        return bounded(getattr(binascii, name.split('.')[-1])(*args, **kwargs))
    if name == 'json.loads' and isinstance(arg, (str, bytes, bytearray)):
        value = json.loads(arg)
        hook = kwargs.get('object_hook')
        if hook:
            return json_hook(vm, value, hook)
        return value
    if name == 'os.fspath':
        if isinstance(arg, (str, bytes)):
            return arg
        if isinstance(arg, Object) and arg.cls.startswith('pathlib.'):
            return path_string(arg)
        return UNKNOWN
    if name in ('pathlib.Path', 'pathlib.PurePath', 'pathlib.PurePosixPath', 'pathlib.PureWindowsPath'):
        return Object(name, args)
    if name.startswith('builtins.') and '.' in name[len('builtins.'):]:
        # Unbound methods such as str.encode and dict.get.
        cls, method = name[len('builtins.'):].split('.', 1)
        if args and typename(arg) == 'builtins.' + cls:
            return bound_call(vm, arg, method, args[1:], kwargs)
    if name.startswith(('io.BytesIO.', 'io.StringIO.')) and isinstance(arg, Buffer):
        return bound_call(vm, arg, name.rsplit('.', 1)[1], args[1:], kwargs)

    # Data-only reconstruction of numerical/scientific objects. Unknown
    # reducers are retained symbolically, so subsequent method calls still
    # carry their module/class provenance.
    return Object(name, args, dict(kwargs))


def literal_operator(name, args):
    if not pure(args):
        return UNKNOWN
    op = name.split('.')[-1]
    if op in ('mul', '__mul__'):
        for a, b in (args, args[::-1]):
            if isinstance(a, (str, bytes, bytearray, tuple, list)) and isinstance(b, int) and len(a) * b > MAX_BYTES:
                raise Unsupported('symbolic repetition too large')
    if op in ('pow', 'lshift') and len(args) > 1 and isinstance(args[1], int) and abs(args[1]) > 100_000:
        raise Unsupported('symbolic integer computation too large')
    if op in ('mod', '__mod__') and isinstance(args[0], (str, bytes)):
        # Avoid enormous printf field widths while preserving ordinary data
        # formatting used to build attribute names and embedded payloads.
        import re
        if any(int(n) > 1_000_000 for n in re.findall(rb'\d{7,}' if isinstance(args[0], bytes) else r'\d{7,}', args[0])):
            raise Unsupported('symbolic format width too large')
    return bounded(getattr(operator, op)(*args))


def literal_codec(name, args):
    if not pure(args) or not args:
        return UNKNOWN
    encoding = args[1] if len(args) > 1 else 'utf-8'
    if not isinstance(encoding, str):
        return UNKNOWN
    normalized = encoding.lower().replace('_', '-').replace(' ', '')
    allowed = {'ascii', 'utf-8', 'utf8', 'utf-16', 'utf-16-le', 'utf-16-be', 'utf-32',
               'latin1', 'latin-1', 'iso-8859-1', 'unicode-escape', 'raw-unicode-escape',
               'base64', 'base64-codec', 'hex', 'hex-codec', 'rot13', 'rot-13'}
    if normalized not in allowed:
        raise Unsupported('unmodelled text encoding')
    return bounded((codecs.decode if name.endswith('decode') else codecs.encode)(*args))


def path_string(obj):
    if all(isinstance(x, str) for x in obj.args):
        import posixpath
        return posixpath.join(*obj.args) if obj.args else '.'
    return UNKNOWN


def bound_call(vm, obj, method, args, kwargs):
    if method == '__call__':
        return vm.call(obj, args, kwargs)
    if method == '__getattribute__':
        return vm.attr(obj, args[0])
    if method == '__getitem__':
        return vm.getitem(obj, args[0])
    if method == '__setitem__':
        vm.setitem(obj, args[0], args[1])
        return None
    if method == '__setstate__':
        vm.build(obj, args[0])
        return None
    if isinstance(obj, Buffer):
        if method in ('getvalue', 'getbuffer'):
            return obj.data
        if method == 'tell':
            return obj.pos
        if method == 'seek':
            offset = args[0]
            whence = args[1] if len(args) > 1 else 0
            obj.pos = (0 if whence == 0 else obj.pos if whence == 1 else len(obj.data)) + offset
            return obj.pos
        if method in ('read', 'read1', 'readline'):
            count = args[0] if args else -1
            end = len(obj.data) if count < 0 else min(len(obj.data), obj.pos + count)
            if method == 'readline':
                sep = b'\n' if isinstance(obj.data, bytes) else '\n'
                index = obj.data.find(sep, obj.pos, end)
                if index >= 0:
                    end = index + 1
            out = obj.data[obj.pos:end]
            obj.pos = end
            return out
        if method == 'write':
            content = args[0]
            obj.data = obj.data[:obj.pos] + content + obj.data[obj.pos + len(content):]
            obj.pos += len(content)
            return len(content)
        if method in ('close', 'flush'):
            return None
    if isinstance(obj, Partial) and method == '__setstate__':
        vm.build(obj, args[0])
        return None
    if isinstance(obj, Object):
        if obj.cls in ('module.namespace', 'class.namespace') and method == 'get':
            return vm.attr(obj.args[0], args[0], args[1] if len(args) > 1 else UNKNOWN)
        if obj.cls == 'pickle.Unpickler' and method == 'load':
            return nested_load(vm, 'pickle.load', obj.args, {})
        if obj.cls == 'logging.config.DictConfigurator' and method == 'configure':
            return logging_config(vm, obj.args[0])
        if obj.cls == 'timeit.Timer' and method in ('timeit', 'repeat', 'autorange'):
            for callback in obj.args[:2]:
                if not isinstance(callback, (str, bytes)):
                    vm.call(callback, ())
            return 0.0
        if obj.cls.startswith('xmlrpc.client.') and method not in ('__enter__', '__exit__', 'close'):
            vm.finding('network', f'{vm.op} {obj.cls}.{method}')
            return UNKNOWN
        if obj.cls.startswith('pathlib.'):
            if method in ('__str__', '__fspath__'):
                return path_string(obj)
            if method in ('joinpath', '__truediv__'):
                return Object(obj.cls, obj.args + args)
        if method == '__init__':
            return vm.call(Ref(obj.cls), args, kwargs)
        return vm.call(Ref(canonical(obj.cls + '.' + method)), (obj,) + args, kwargs)
    if type(obj) is dict:
        if method == 'get':
            return obj.get(*args)
        if method == 'setdefault':
            return obj.setdefault(*args)
        if method == 'update':
            obj.update(*args, **kwargs)
            return None
        if method == 'pop':
            return obj.pop(*args)
        if method in ('keys', 'values', 'items', 'copy'):
            out = getattr(obj, method)()
            return out if method == 'copy' else list(out)
    if type(obj) is list:
        if method in ('append', 'extend', 'insert', 'pop', 'reverse', 'clear', 'copy'):
            return getattr(obj, method)(*args)
        if method == 'sort':
            result = vm.call(Ref('builtins.sorted'), (obj,), kwargs)
            obj[:] = result
            return None
    if type(obj) in (str, bytes, bytearray):
        if method in ('encode', 'decode'):
            return literal_codec('codecs.' + method, (obj,) + args)
        allowed = {'join', 'replace', 'lower', 'upper', 'strip', 'lstrip', 'rstrip', 'split', 'rsplit',
                   'partition', 'rpartition', 'capitalize', 'swapcase', 'title', 'removeprefix',
                   'removesuffix', 'startswith', 'endswith', 'find', 'rfind', 'count', 'hex', 'fromhex'}
        if method in allowed and pure(args) and pure(kwargs):
            return bounded(getattr(obj, method)(*args, **kwargs))
        if method == 'format' and pure(args) and pure(kwargs):
            # The format machinery sees only literal builtin values here.
            if len(obj) > 100_000:
                raise Unsupported('symbolic format string too large')
            return bounded(obj.format(*args, **kwargs))
    raise Unsupported(f'unmodelled method {typename(obj)}.{method}')


def nested_load(vm, name, args, kwargs):
    from modelgate import dispatch, pickle_scan
    mode = NESTED_LOADERS[name]
    source = args[0] if args else kwargs.get('file', kwargs.get('f', kwargs.get('filepath_or_buffer', UNKNOWN)))
    if isinstance(source, Object) and source.cls.startswith('pathlib.'):
        source = path_string(source)
    embedded = mode in ('pickle-bytes', 'torch-bytes')
    if not embedded and isinstance(source, (str, bytes)):
        vm.finding('file-access', f'{vm.op} {name} reads an external path')
        return UNKNOWN
    if isinstance(source, Buffer):
        data = source.data[source.pos:]
    elif embedded:
        data = as_bytes(source)
    else:
        data = None
    if data is None or not isinstance(data, (bytes, bytearray)):
        raise Unsupported(f'{name} input cannot be resolved as embedded data')
    location = vm.site + ':' + name
    if mode.startswith('pickle'):
        findings, value, consumed = pickle_scan.read_pickle(data, location, vm.budget, vm.depth + 1)
        if isinstance(source, Buffer):
            source.pos += consumed
    else:
        allow_pickle = kwargs.get('allow_pickle', args[2] if mode == 'numpy' and len(args) > 2 else
                                  args[1] if mode == 'numpy-array' and len(args) > 1 else True)
        if mode in ('numpy', 'numpy-array') and allow_pickle is False:
            return Object('numpy.ndarray')
        findings, value = dispatch.analyze_bytes(bytes(data), location, vm.budget, vm.depth + 1)
        if isinstance(source, Buffer):
            source.pos = len(source.data)
    vm.findings.extend(findings)
    return value


def logging_config(vm, config):
    if not isinstance(config, dict):
        raise Unsupported('logging configuration is not a literal dictionary')
    if config.get('incremental'):
        return None

    def convert(value):
        if isinstance(value, str) and value.startswith('ext://'):
            return Ref(canonical(value[6:]))
        if isinstance(value, str) and value.startswith('cfg://'):
            current = config
            for piece in value[6:].split('.'):
                current = vm.getitem(current, piece)
            return current
        return value

    for section in ('formatters', 'filters', 'handlers'):
        entries = config.get(section, {})
        if not isinstance(entries, dict):
            continue
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            factory = entry.get('()')
            if factory is None and section == 'handlers':
                factory = entry.get('class')
            if factory is None:
                continue
            factory = convert(factory)
            if isinstance(factory, str):
                factory = Ref(canonical(factory))
            options = {k: convert(v) for k, v in entry.items()
                       if k not in ('()', 'class', '.', 'level', 'formatter', 'filters')}
            vm.call(factory, (), options)
    return None


def json_hook(vm, value, hook):
    if isinstance(value, list):
        return [json_hook(vm, x, hook) for x in value]
    if isinstance(value, dict):
        return vm.call(hook, ({k: json_hook(vm, v, hook) for k, v in value.items()},))
    return value
