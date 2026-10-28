"""Policy regressions built from bytes. No artifact in this suite is loaded."""
import base64
import bz2
import gzip
import io
import json
import lzma
import marshal
import os
import pickle
import struct
import sys
import tarfile
import unittest
import warnings
import zipfile
import zlib

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from modelgate.dispatch import analyze_data
from modelgate.pickle_scan import scan_pickle, TORCH_MAGIC
from modelgate.symbolic import Runtime, Stopped


class Expr:
    def __init__(self, code):
        self.code = code


def lit(value):
    if isinstance(value, Expr):
        return value.code
    if isinstance(value, tuple):
        return b'(' + b''.join(lit(x) for x in value) + b't'
    if isinstance(value, list):
        return b'](' + b''.join(lit(x) for x in value) + b'e'
    if isinstance(value, dict):
        return b'}(' + b''.join(lit(k) + lit(v) for k, v in value.items()) + b'u'
    return pickle.dumps(value, protocol=2)[2:-1]


def glob(module, name):
    return Expr(b'c' + module.encode() + b'\n' + name.encode() + b'\n')


def call(fn, *args):
    return Expr(lit(fn) + lit(args) + b'R')


def attr(obj, name):
    return call(glob('builtins', 'getattr'), obj, name)


def method(obj, name, *args):
    return call(attr(obj, name), *args)


def module(name):
    return call(glob('importlib', 'import_module'), name)


def partial(fn, *args, **kwargs):
    part = call(glob('functools', 'partial'), fn, *args)
    return Expr(lit(part) + lit((fn, args, kwargs, None)) + b'b')


def stream(expr, proto=4):
    return (bytes([128, proto]) if proto else b'') + lit(expr) + b'.'


def archive(items, prefix=b''):
    out = io.BytesIO(prefix)
    if prefix:
        out.seek(0, 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with zipfile.ZipFile(out, 'w') as z:
            for name, data in items:
                z.writestr(name, data)
    return out.getvalue()


def npy(payload, desc='|O', version=1):
    h = repr({'descr': desc, 'fortran_order': False, 'shape': (1,)}).encode() + b'\n'
    return b'\x93NUMPY' + bytes([version, 0]) + (struct.pack('<H', len(h)) if version == 1 else struct.pack('<I', len(h))) + h + payload


def findings(data):
    rt = Runtime()
    try:
        analyze_data(data, 'test', rt)
    except Stopped:
        pass
    return rt.findings


class PicklePolicy(unittest.TestCase):
    def check(self, expr, malicious, proto=4):
        result = scan_pickle(stream(expr, proto), 'test')
        self.assertEqual(bool(result), malicious, result)
        if malicious:
            self.assertNotIn('unsupported', {f.kind for f in result}, result)

    def test_capabilities_are_calls(self):
        targets = [('os', 'system', ('echo harmless-test-string',)),
                   ('posix', 'system', ('echo harmless-test-string',)),
                   ('nt', 'system', ('echo harmless-test-string',)),
                   ('builtins', 'eval', ('1+1',)),
                   ('builtins', 'compile', ('x=1', '<string>', 'exec')),
                   ('subprocess', 'Popen', (['command'],)),
                   ('subprocess', 'check_call', (['command'],)),
                   ('os', 'execve', ('command', [], {})),
                   ('builtins', 'open', ('/outside', 'r')),
                   ('pathlib', 'Path.write_text', (call(glob('pathlib', 'Path'), '/outside'), 'text')),
                   ('socket', 'create_connection', (('example.invalid', 123),)),
                   ('urllib.request', 'urlopen', ('https://example.invalid',)),
                   ('ssl', 'get_server_certificate', (('example.invalid', 443),)),
                   ('numpy.testing._private.utils', 'runstring', ('1+1', {})),
                   ('logging.config', 'dictConfig', ({'version': 1, 'handlers': {'h': {'class': 'logging.FileHandler', 'filename': '/outside'}}},))]
        for mod, name, args in targets:
            with self.subTest(target=mod + '.' + name):
                self.check(glob(mod, name), False)
                self.check([glob(mod, name), 'os.system eval( subprocess'], False)
                self.check(call(glob(mod, name), *args), True)

    def test_resolvers(self):
        system = glob('os', 'system')
        routes = [
            attr(module('os'), 'system'),
            call(call(glob('operator', 'attrgetter'), 'system'), module('os')),
            call(call(glob('operator', 'attrgetter'), 'os.system'), module('shutil')),
            attr(attr(module('shutil'), 'os'), 'system'),
            call(glob('pkgutil', 'resolve_name'), 'os:system'),
            call(glob('pkgutil', 'resolve_name'), 'os.system'),
            call(glob('operator', 'getitem'), attr(module('sys'), 'modules'), 'os'),
        ]
        routes[-1] = attr(routes[-1], 'system')
        namespace = call(glob('builtins', 'vars'), module('os'))
        routes += [call(glob('operator', 'getitem'), namespace, 'system'),
                   method(namespace, 'get', 'system'),
                   call(glob('copy', 'copy'), system),
                   call(glob('copy', 'deepcopy'), system),
                   call(call(glob('functools', 'lru_cache'), None), system),
                   call(glob('functools', 'cache'), system),
                   attr(module('subprocess'), 'os')]
        routes[-1] = attr(routes[-1], 'system')
        for i, fn in enumerate(routes):
            with self.subTest(route=i):
                self.check(fn, False)
                self.check(call(fn, 'echo test'), True)
        self.check(call(call(glob('operator', 'methodcaller'), 'system', 'echo test'), module('os')), True)
        self.check(call(glob('operator', 'call'), system, 'echo test'), True)

    def test_aliases_and_dotted(self):
        for mod, name, args in [('__builtin__', 'eval', ('1',)), ('commands', 'getoutput', ('x',)),
                                ('urllib2', 'urlopen', ('https://example.invalid',)),
                                ('httplib', 'HTTPConnection', ('example.invalid',))]:
            self.check(call(glob(mod, name), *args), name != 'HTTPConnection', proto=2)
        self.check(call(glob('shutil', 'os.system'), 'x'), True, 4)
        self.check(call(glob('shutil', 'os.system'), 'x'), False, 2)
        self.check(call(glob('bdb', 'Bdb.run'), None, 'x'), True, 4)
        self.check(call(glob('bdb', 'Bdb.run'), None, 'x'), False, 3)

    def test_memo_and_mutation(self):
        memo = lit(glob('os', 'system')) + b'p500\n0g500\n' + lit(('x',)) + b'R'
        self.check(Expr(memo), True)
        for fn in (glob('os', 'system'), glob('builtins', 'str')):
            part = partial(glob('builtins', 'str'))
            mutated = Expr(lit(part) + b'q\x010h\x01' + lit((fn, ('x',), {}, None)) + b'b0h\x01)R')
            self.check(mutated, fn.code == glob('os', 'system').code)
        rebinding = Expr(lit(module('math')) + lit({'sqrt': glob('os', 'system')}) + b'b0' + lit(call(glob('math', 'sqrt'), 'x')))
        self.check(rebinding, True)
        rebinding_safe = Expr(lit(module('os')) + lit({'system': glob('builtins', 'str')}) + b'b0' + lit(call(glob('os', 'system'), 'x')))
        self.check(rebinding_safe, False)
        a = stream(Expr(lit(module('math')) + lit({'sqrt': glob('os', 'system')}) + b'b'))
        b = stream(call(glob('math', 'sqrt'), 'x'))
        self.assertTrue(findings(a + b))

    def test_shared_partial_keywords(self):
        keywords = Expr(lit({'key': glob('builtins', 'str')}) + b'q\x20')
        part = Expr(lit(call(glob('functools', 'partial'), glob('builtins', 'sorted'))) +
                    lit((glob('builtins', 'sorted'), (), keywords, None)) + b'bq\x210')
        change = b'h\x20' + lit('key') + lit(glob('os', 'system')) + b's0'
        execute = b'h\x21' + lit((['x'],)) + b'R'
        self.check(Expr(lit(part) + change + execute), True)
        # A user attribute named func cannot shadow partial's read-only member.
        fake = Expr(lit(call(glob('functools', 'partial'), glob('builtins', 'str'))) +
                    lit((glob('builtins', 'str'), (), {}, {'func': glob('os', 'system')})) + b'b')
        self.check(call(attr(fake, 'func'), 'x'), False)

    def test_registered_extension_and_dynamic_methods(self):
        register = lit(call(glob('copyreg', 'add_extension'), 'os', 'system', 42)) + b'0'
        self.check(Expr(register + b'\x82*'), False)
        self.check(Expr(register + b'\x82*' + lit(('x',)) + b'R'), True)
        getter = method(module('os'), '__getattribute__', 'system')
        self.check(call(getter, 'x'), True)
        self.check(call(attr(partial(glob('os', 'system'), 'x'), '__call__')), True)
        self.check(call(glob('functools', 'partial.__call__'), partial(glob('os', 'system'), 'x')), True)
        cls = call(glob('builtins', 'type'), 'Data', (), {'__bool__': partial(glob('os', 'system'), 'x')})
        self.check(call(cls), False)
        self.check(call(glob('builtins', 'bool'), call(cls)), True)
        cls = call(glob('builtins', 'type'), 'Data', (), {'__getitem__': glob('os', 'system')})
        self.check(call(glob('operator', 'getitem'), call(cls), 'x'), True)

    def test_inst_obj_newobj(self):
        for fn in (glob('subprocess', 'Popen'), glob('ftplib', 'FTP')):
            self.check(Expr(lit(fn) + lit(('x',)) + b'\x81'), False)
            self.check(Expr(lit(fn) + lit(('x',)) + lit({}) + b'\x92'), False)
            self.check(Expr(b'(' + lit(fn) + b'o'), False)
            self.check(Expr(b'(' + lit(fn) + lit('x') + b'o'), True)
        self.check(Expr(b'(ios\nsystem\n'), False)  # function is called, but missing its argument
        self.check(Expr(b'(' + lit(partial(glob('os', 'system'), 'x')) + b'o'), True)
        self.check(Expr(lit(glob('os', 'system')) + lit(('x',)) + b'\x81'), False)
        self.check(Expr(lit(glob('os', 'system')) + lit(['x']) + b'R'), False)
        self.check(Expr(lit(glob('os', 'system')) + lit({'x': 1}) + b'R'), False)
        bad = stream(call(glob('os', 'system'), 'x'))
        # bytearray fills its buffer in __init__, whereas bytes does so in
        # __new__. NEWOBJ must preserve that distinction.
        empty = Expr(lit(glob('builtins', 'bytearray')) + lit((bad,)) + b'\x81')
        full = Expr(lit(glob('builtins', 'bytes')) + lit((bad,)) + b'\x81')
        self.check(call(glob('pickle', 'loads'), empty), False)
        self.check(call(glob('pickle', 'loads'), full), True)

    def test_network_lifecycle(self):
        for cls, args in [('socket.socket', ()), ('http.client.HTTPConnection', ('example.invalid', 80)),
                          ('ftplib.FTP', ()), ('smtplib.SMTP', ()), ('xmlrpc.client.ServerProxy', ('http://example.invalid',))]:
            mod, _, name = cls.rpartition('.')
            obj = call(glob(mod, name), *args)
            self.check(obj, False)
            self.check(method(obj, 'request' if cls.endswith('ServerProxy') else 'connect', ('example.invalid', 80)), True)
        self.check(call(glob('ftplib', 'FTP'), 'example.invalid'), True)
        self.check(call(glob('smtplib', 'SMTP'), 'example.invalid'), True)
        path = call(glob('pathlib', 'Path'), '/outside')
        self.check(path, False)
        self.check(method(path, 'write_text', 'x'), True)
        self.check(method(call(glob('io', 'BytesIO'), b'data'), 'write', b'more'), False)

    def test_iterators(self):
        for source in [call(glob('builtins', 'map'), glob('os', 'system'), ['x']),
                       call(glob('builtins', 'filter'), glob('os', 'system'), ['x']),
                       call(glob('itertools', 'starmap'), glob('os', 'system'), [('x',)])]:
            self.check(source, False)
            for consumer in [glob('builtins', 'list'), glob('builtins', 'tuple'), glob('builtins', 'set'),
                             glob('collections', 'deque'), glob('builtins', 'next'), glob('builtins', 'any')]:
                with self.subTest(consumer=consumer.code):
                    self.check(call(consumer, source), True)
        self.check(call(glob('builtins', 'list'), call(glob('builtins', 'map'), glob('os', 'system'), [])), False)
        self.check(call(glob('functools', 'reduce'), glob('builtins', 'eval'), ['1+1', {}]), True)
        for name in ('sorted', 'min', 'max'):
            key = partial(glob('builtins', name), key=glob('os', 'system'))
            self.check(call(key, ['x']), True)
            self.check(key, False)
        self.check(call(glob('builtins', 'any'), call(glob('itertools', 'chain'), [True], call(glob('builtins', 'map'), glob('os', 'system'), ['x']))), False)

    def test_nested_loaders(self):
        bad = stream(call(glob('os', 'system'), 'x'))
        good = stream({'x': 'os.system'})
        for data, expected in [(bad, True), (good, False), (good + bad, False)]:
            self.check(call(glob('pickle', 'loads'), data), expected)
            self.check(call(glob('_pickle', 'loads'), data), expected)
            self.check(call(glob('pickle', 'load'), call(glob('io', 'BytesIO'), data)), expected)
        self.check(call(call(glob('pickle', 'loads'), stream(glob('os', 'system'))), 'x'), True)
        for transform in [lambda b: call(glob('base64', 'b64decode'), base64.b64encode(b)),
                          lambda b: call(glob('zlib', 'decompress'), zlib.compress(b)),
                          lambda b: call(glob('builtins', 'bytes.fromhex'), b.hex()),
                          lambda b: call(glob('operator', 'add'), b[:10], b[10:]),
                          lambda b: call(glob('builtins', 'bytearray'), list(b)),
                          lambda b: call(glob('_codecs', 'encode'), b.decode('latin1'), 'latin1')]:
            for data, expected in [(bad, True), (good, False)]:
                self.check(call(glob('pickle', 'loads'), transform(data)), expected)
        for loader in [('pickle', 'load'), ('torch', 'load'), ('joblib', 'load'), ('numpy', 'load'), ('pandas', 'read_pickle')]:
            self.check(call(glob(*loader), '/outside'), True)
        self.check(call(partial(glob('numpy', 'load'), allow_pickle=True), call(glob('io', 'BytesIO'), npy(bad))), True)
        self.check(call(partial(glob('numpy', 'load'), allow_pickle=True), call(glob('io', 'BytesIO'), npy(good))), False)
        for payload, expected in [(bad, True), (good, False)]:
            legacy = pickle.dumps(TORCH_MAGIC, 2) + pickle.dumps(1001, 2) + pickle.dumps({}, 2) + payload + pickle.dumps([], 2) + b'raw-storage'
            self.check(call(glob('torch.storage', '_load_from_bytes'), legacy), expected)

    def test_threads(self):
        thread = call(partial(glob('threading', 'Thread'), target=glob('os', 'system'), args=('x',)))
        self.check(thread, False)
        self.check(method(thread, 'start'), True)
        safe = call(partial(glob('threading', 'Thread'), target=glob('builtins', 'str'), args=('x',)))
        self.check(method(safe, 'start'), False)
        pool = call(glob('concurrent.futures', 'ThreadPoolExecutor'))
        self.check(pool, False)
        self.check(method(pool, 'submit', glob('os', 'system'), 'x'), True)
        self.check(method(pool, 'submit', glob('builtins', 'str'), 'x'), False)

    def test_marshaled_code_is_not_a_function(self):
        code = call(glob('marshal', 'loads'), marshal.dumps((lambda x: x).__code__))
        self.check(code, False)
        self.check(call(glob('types', 'FunctionType'), code, {}), True)
        self.check(call(glob('types', 'FunctionType'), call(glob('marshal', 'loads'), marshal.dumps('data')), {}), False)
        self.check(call(glob('types', 'FunctionType'), call(glob('marshal', 'loads'), b'invalid'), {}), False)
        self.check(call(glob('torch._tensor', '_rebuild_from_type_v2'), glob('os', 'system'), glob('torch', 'Tensor'), ('x',), {}), True)

    def test_in_memory_archive_loaders_and_timer(self):
        bad = stream(call(glob('os', 'system'), 'x'))
        good = stream({'data': 'os.system'})
        for payload, expected in ((bad, True), (good, False)):
            archive_file = call(glob('zipfile', 'ZipFile'), call(glob('io', 'BytesIO'), archive([('model.pkl', payload)])))
            self.check(archive_file, False)
            self.check(call(glob('pickle', 'loads'), method(archive_file, 'read', 'model.pkl')), expected)
            compressed = call(partial(glob('gzip', 'GzipFile'), fileobj=call(glob('io', 'BytesIO'), gzip.compress(payload))))
            self.check(compressed, False)
            self.check(call(glob('pickle', 'load'), compressed), expected)
        self.check(call(partial(glob('timeit', 'timeit'), stmt=partial(glob('os', 'system'), 'x'), number=0)), False)
        self.check(call(partial(glob('timeit', 'timeit'), stmt=partial(glob('os', 'system'), 'x'), number=1)), True)
        self.check(call(partial(glob('timeit', 'timeit'), stmt='pass', number=0)), True)

    def test_joblib_inline_arrays(self):
        bad = stream(call(glob('os', 'system'), 'x'))
        for object_dtype in (False, True):
            dtype = call(glob('numpy', 'dtype'), 'O8' if object_dtype else 'u1')
            state = {'shape': (len(bad),), 'order': 'C', 'dtype': dtype, 'allow_mmap': False,
                     'numpy_array_alignment_bytes': 16}
            prefix = b'\x80\x04' + lit(glob('joblib.numpy_pickle', 'NumpyArrayWrapper')) + b')\x81' + lit(state) + b'b'
            if object_dtype:
                data = prefix + bad + b'.'
            else:
                padding = (16 - (len(prefix) + 1) % 16) % 16
                data = prefix + bytes([padding]) + b'\0' * padding + bad + b'.'
            result = scan_pickle(data, 'joblib')
            self.assertEqual(bool(result), object_dtype, result)
            self.assertNotIn('unsupported', {f.kind for f in result})


class Containers(unittest.TestCase):
    def test_wrappers(self):
        for payload, expected in [(stream(call(glob('os', 'system'), 'x')), True), (stream({'data': 'os.system'}), False)]:
            for wrap in (gzip.compress, bz2.compress, lzma.compress, zlib.compress):
                with self.subTest(wrap=wrap):
                    self.assertEqual(bool(findings(wrap(payload))), expected)
            self.assertEqual(bool(findings(archive([('odd.dat', payload)], prefix=b'not a torch zip\n'))), expected)
            self.assertEqual(bool(findings(npy(payload))), expected)
            self.assertEqual(bool(findings(archive([('arr.npy', npy(payload))]))), expected)
            self.assertEqual(bool(findings(gzip.compress(bz2.compress(archive([('model', payload)]))))), expected)

    def test_duplicates_and_data(self):
        bad = stream(call(glob('os', 'system'), 'x'))
        good = stream({'data': 'os.system'})
        self.assertFalse(findings(archive([('data.pkl', bad), ('data.pkl', good)])))
        self.assertTrue(findings(archive([('data.pkl', good), ('data.pkl', bad)])))
        self.assertFalse(findings(archive([('array.npy', npy(bad, '<i8'))])))
        self.assertTrue(findings(gzip.compress(good) + gzip.compress(bad)))
        self.assertTrue(findings(good + bad))


if __name__ == '__main__':
    unittest.main()
