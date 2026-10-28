"""Independent policy regressions. Payloads are assembled, never loaded."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import base64
import bz2
import gzip
import io
import json
import lzma
import marshal
import pickle
import struct
import tarfile
import tempfile
import unittest
import zipfile
import zlib

from modelgate.dispatch import scan_data
from modelgate.pickle_scan import scan_pickle, TORCH_MAGIC
from modelgate.policy import TF_OPS
from modelgate import keras_scan, savedmodel_scan


class Raw:
    def __init__(self, data):
        self.data = data


def global_(module, name):
    return Raw(b'c' + module.encode() + b'\n' + name.encode() + b'\n')


def encode(value):
    if isinstance(value, Raw):
        return value.data
    if value is None:
        return b'N'
    if value is True:
        return b'\x88'
    if value is False:
        return b'\x89'
    if isinstance(value, int):
        return b'L' + str(value).encode() + b'L\n'
    if isinstance(value, str):
        raw = value.encode()
        return b'X' + struct.pack('<I', len(raw)) + raw
    if isinstance(value, bytes):
        return b'B' + struct.pack('<I', len(value)) + value
    if isinstance(value, bytearray):
        return b'\x96' + struct.pack('<Q', len(value)) + value
    if isinstance(value, tuple):
        return b'(' + b''.join(encode(x) for x in value) + b't'
    if isinstance(value, list):
        return b'(' + b''.join(encode(x) for x in value) + b'l'
    if isinstance(value, dict):
        return b'(' + b''.join(encode(k) + encode(v) for k, v in value.items()) + b'd'
    raise TypeError(value)


def call(fn, *args, **kw):
    if isinstance(fn, str):
        module, name = fn.rsplit('.', 1)
        fn = global_(module, name)
    if kw:
        partial = call(global_('functools', 'partial'), fn)
        partial = Raw(encode(partial) + encode((fn, (), kw, None)) + b'b')
        return call(partial, *args)
    return Raw(encode(fn) + encode(args) + b'R')


def attr(obj, name):
    return call('builtins.getattr', obj, name)


def pkl(value, protocol=4):
    return b'\x80' + bytes([protocol]) + encode(value) + b'.'


def zip_bytes(entries, prefix=b''):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as z:
        for name, data in entries:
            z.writestr(name, data)
    return prefix + out.getvalue()


def npy(data, descr='|O', version=1):
    header = repr({'descr': descr, 'fortran_order': False, 'shape': (1,)}).encode()
    header += b' ' * (-(10 + len(header) + 1) % 64) + b'\n'
    return b'\x93NUMPY' + bytes((version, 0)) + struct.pack('<H' if version == 1 else '<I', len(header)) + header + data


class PolicyTests(unittest.TestCase):
    def check(self, value, malicious, *, raw=False, protocol=4, kinds=True):
        data = value if raw else pkl(value, protocol)
        findings = scan_data(data, 'artifact.bin')
        self.assertEqual(bool(findings), malicious, [(x.kind, x.detail, x.location) for x in findings])
        if malicious and kinds:
            self.assertTrue(any(x.kind != 'unsupported' for x in findings), [(x.kind, x.detail) for x in findings])

    def test_globals_and_strings_are_data(self):
        for value in ['os.system', 'subprocess', 'eval(', 'exec(', {'payload': 'ReadFile'},
                      global_('os', 'system'), [global_('builtins', 'eval')],
                      call('builtins.__import__', 'subprocess'), call('importlib.import_module', 'os')]:
            with self.subTest(value=value):
                self.check(value, False)

    def test_aliases_and_protocols(self):
        for protocol in range(6):
            for module, name in [('os', 'system'), ('posix', 'system'), ('nt', 'system')]:
                with self.subTest(protocol=protocol, module=module):
                    self.check(call(global_(module, name), 'echo test'), True, protocol=protocol)
        for module, name in [('__builtin__', 'eval'), ('commands', 'getoutput'), ('urllib2', 'urlopen'), ('httplib', 'HTTPConnection.request')]:
            if '.' not in name:
                self.check(call(global_(module, name), 'test'), True, protocol=2)
        for protocol in range(6):
            self.check(call(global_('shutil', 'os.system'), 'test'), protocol >= 4, protocol=protocol)

    def test_execution_opcode_semantics(self):
        system = encode(global_('os', 'system'))
        popen = encode(global_('subprocess', 'Popen'))
        arg = encode(('test',))
        self.check(b'\x80\x04' + system + encode('test') + b'R.', False, raw=True)
        self.check(b'\x80\x04' + system + arg + b'\x81.', False, raw=True)
        self.check(b'\x80\x04' + popen + arg + b'\x81.', False, raw=True)
        self.check(b'\x80\x04' + popen + arg + encode({}) + b'\x92.', False, raw=True)
        self.check(b'(isubprocess\nPopen\n.', False, raw=True)
        self.check(b'(' + encode('test') + b'isubprocess\nPopen\n.', True, raw=True)
        self.check(b'(' + popen + b'o.', False, raw=True)
        self.check(b'(' + popen + encode('test') + b'o.', True, raw=True)
        self.check(b'(' + system + encode('test') + b'o.', True, raw=True)
        self.check(b'(ios\nfork\n.', True, raw=True)
        # PERSID stops plain pickle; nothing after it is executed.
        self.check(b'Pstorage\n' + encode(call('os.system', 'test')) + b'.', False, raw=True)

    def test_memo_references(self):
        for put, get in [(b'p123\n', b'g123\n'), (b'q\x7b', b'h\x7b'),
                         (b'r' + struct.pack('<I', 1000), b'j' + struct.pack('<I', 1000)),
                         (b'\x94', b'h\x00')]:
            raw = b'\x80\x04' + encode(global_('os', 'system')) + put + b'0' + get + encode(('test',)) + b'R.'
            self.check(raw, True, raw=True)
        raw = b'\x80\x04' + encode('os') + b'q\x010' + encode('system') + b'q\x020h\x01h\x02\x93' + encode(('test',)) + b'R.'
        self.check(raw, True, raw=True)

    def test_extension_registry(self):
        for code, op in [(42, b'\x82*'), (1000, b'\x83' + struct.pack('<H', 1000)),
                         (100000, b'\x84' + struct.pack('<i', 100000))]:
            self.check(Raw(op), False)
            register = call('copyreg.add_extension', 'os', 'system', code)
            prefix = b'\x80\x04' + encode(register) + b'0' + op
            self.check(prefix + b'.', False, raw=True)
            self.check(prefix + encode(('test',)) + b'R.', True, raw=True)
            self.check(pkl(register) + b'\x80\x04' + op + encode(('test',)) + b'R.', True, raw=True)

    def test_partial_mutation(self):
        safe = call('functools.partial', global_('builtins', 'str'), 'test')
        self.check(safe, False)
        state = (global_('os', 'system'), ('test',), {}, None)
        self.check(Raw(encode(safe) + encode(state) + b'b'), False)
        raw = b'\x80\x04' + encode(safe) + b'q\x00' + encode(state) + b'b0h\x00)R.'
        self.check(raw, True, raw=True)
        # Argument-less OBJ invokes a partial (not a class).
        self.check(Raw(b'(' + encode(call('functools.partial', global_('os', 'system'), 'test')) + b'o'), True)

    def test_module_mutation(self):
        module = call('builtins.__import__', 'os')
        raw = b'\x80\x04' + encode(module) + encode({'getpid': global_('builtins', 'eval')}) + b'b0'
        raw += encode(call('posix.getpid', '1+1')) + b'.'
        self.check(raw, True, raw=True)
        raw = b'\x80\x04' + encode(module) + encode({'system': global_('builtins', 'str')}) + b'b0'
        raw += encode(call('nt.system', 'test')) + b'.'
        self.check(raw, False, raw=True)
        # A module's dictionary is live, including through vars and getitem.
        ns = call('builtins.vars', module)
        setter = call('operator.setitem', ns, 'system', global_('builtins', 'str'))
        self.check([setter, call('os.system', 'test')], False)

    def test_attribute_state_hooks(self):
        module = call('builtins.__import__', 'os')
        first_state = {'__setstate__': global_('builtins', 'exec')}
        self.check(Raw(encode(module) + encode(first_state) + b'b' + encode('x = 1') + b'b'), True)
        cls = global_('bdb', 'Bdb')
        setter = call('builtins.setattr', cls, '__setstate__', global_('builtins', 'exec'))
        instance = Raw(encode(cls) + b')\x81' + encode('x = 1') + b'b')
        self.check([setter, instance], True)
        self.check([call('builtins.setattr', cls, 'stored_data', global_('builtins', 'exec')),
                    Raw(encode(cls) + b')\x81')], False)

    def test_keyword_binding_and_invalid_calls(self):
        evil = pkl(call('os.system', 'test'))
        buf = call('io.BytesIO', initial_bytes=evil)
        self.check(call('pickle.load', buf), True)
        self.check(call(call('pkgutil.resolve_name', name='os.system'), 'test'), True)
        self.check(call('builtins.compile', '1+1', '<test>', 'eval'), True)
        self.check(call('builtins.compile', '1+1'), False)
        self.check(call('os.system', 'test', 'extra'), False)
        self.check(call('subprocess.Popen'), False)
        self.check(call('subprocess.Popen', args='test'), True)
        self.check(call('functools.reduce', global_('os', 'system'), []), False)
        self.check(call('functools.reduce', global_('os', 'system'), ['test']), False)
        self.check(call('os.system', 42), False)
        self.check(call('subprocess.Popen', []), False)

    def test_additional_callbacks_and_reconstructors(self):
        self.check(call('json.loads', '123', parse_int=global_('os', 'system')), True)
        self.check(call('json.loads', '123', parse_int=global_('builtins', 'str')), False)
        self.check(call('json.loads', '{"text": "os.system"}'), False)
        self.check(call('copyreg._reconstructor', global_('subprocess', 'Popen'),
                        global_('subprocess', 'Popen'), 'test'), True)
        self.check(call('copyreg._reconstructor', global_('builtins', 'str'),
                        global_('builtins', 'str'), 'os.system'), False)

    def test_copied_partial_independence(self):
        partial = call('functools.partial', global_('builtins', 'str'), 'text')
        code = b'\x80\x04' + encode(partial) + b'q\x000'
        code += encode(call('copy.copy', Raw(b'h\x00'))) + b'q\x010'
        code += b'h\x00' + encode((global_('os', 'system'), ('test',), {}, None)) + b'b0'
        self.check(code + b'h\x01)R.', False, raw=True)
        self.check(code + b'h\x00)R.', True, raw=True)

    def test_in_memory_files(self):
        clean, evil = pkl({'message': 'exec('}), pkl(call('os.system', 'test'))
        for data, expected in [(clean, False), (evil, True)]:
            z = call('zipfile.ZipFile', call('io.BytesIO', zip_bytes([('inside.pkl', data)])))
            self.check(call('pickle.loads', call(attr(z, 'read'), 'inside.pkl')), expected)
            for name, compress in [('gzip.GzipFile', gzip.compress), ('bz2.BZ2File', bz2.compress), ('lzma.LZMAFile', lzma.compress)]:
                stream = call(name, call('io.BytesIO', compress(data)))
                self.check(call('pickle.loads', call(attr(stream, 'read'))), expected)
            npz = call('numpy.load', call('io.BytesIO', zip_bytes([('array.npy', npy(data))])), None, True)
            self.check(npz, False)
            self.check(call('operator.getitem', npz, 'array'), expected)

    def test_logging_config(self):
        self.check(call('logging.config.dictConfig', {'version': 1, 'handlers': {
            'h': {'class': 'logging.StreamHandler'}}}), False)
        self.check(call('logging.config.dictConfig', {'version': 1, 'handlers': {
            'h': {'class': 'logging.FileHandler', 'filename': '/tmp/outside'}}}), True)
        self.check(call('logging.config.dictConfig', {'version': 1, 'handlers': {
            'h': {'()': 'subprocess.Popen', 'args': 'test'}}}), True)
        self.check(call('logging.config.dictConfig', {'version': 1, 'formatters': {
            'f': {'format': 'os.system %(message)s'}}}), False)

    def test_indirection_routes(self):
        mod = call('builtins.__import__', 'os')
        routes = [attr(mod, 'system'),
                  call(call('operator.attrgetter', 'system'), mod),
                  call('pkgutil.resolve_name', 'os:system'),
                  call('pkgutil.resolve_name', 'shutil.os.system'),
                  call('copy.copy', global_('os', 'system')),
                  call('copy.deepcopy', global_('os', 'system')),
                  call(call('functools.lru_cache', None), global_('os', 'system')),
                  call('functools.cache', global_('os', 'system')),
                  call('operator.getitem', call('builtins.vars', mod), 'system'),
                  attr(call('operator.getitem', global_('sys', 'modules'), 'os'), 'system')]
        for fn in routes:
            with self.subTest(fn=fn.data):
                self.check(fn, False)
                self.check(call(fn, 'test'), True)
        self.check(call(call('operator.methodcaller', 'system', 'test'), mod), True)
        self.check(call('operator.call', global_('os', 'system'), 'test'), True)

    def test_lazy_callbacks(self):
        system = global_('os', 'system')
        for lazy in [call('builtins.map', system, ['test']),
                     call('builtins.filter', system, ['test']),
                     call('itertools.starmap', system, [('test',)])]:
            self.check(lazy, False)
            for consumer in ('builtins.list', 'builtins.tuple', 'collections.deque', 'builtins.next'):
                self.check(call(consumer, lazy), True)
        for consumer in ('builtins.sorted', 'builtins.min', 'builtins.max'):
            self.check(call(consumer, ['test'], key=system), True)
            self.check(call(consumer, [], key=system, default=0) if consumer != 'builtins.sorted' else call(consumer, [], key=system), False)
        self.check(call('functools.reduce', global_('builtins', 'eval'), ['1+1', {}]), True)
        self.check(call('builtins.list', call('builtins.map', global_('builtins', 'str'), [1, 2])), False)
        self.check(call('builtins.any', call('builtins.map', global_('builtins', 'bool'), [1, 0])), False)
        self.check(call('builtins.list', call('itertools.islice', call('builtins.map', system, ['test']), 0)), False)

    def test_threads_and_executors(self):
        thread = call('threading.Thread', None, global_('os', 'system'), None, ('test',))
        safe_thread = call('threading.Thread', None, global_('builtins', 'str'), None, ('test',))
        self.check(thread, False)
        self.check(call(attr(thread, 'start')), True)
        self.check(call(attr(safe_thread, 'start')), False)
        self.check(call(global_('threading', 'Thread.start'), thread), True)
        self.check(call('_thread.start_new_thread', global_('os', 'system'), ('test',)), True)
        for ctor in ('concurrent.futures.ThreadPoolExecutor', 'concurrent.futures.ProcessPoolExecutor'):
            executor = call(ctor, 1)
            self.check(executor, False)
            self.check(call(attr(executor, 'submit'), global_('os', 'system'), 'test'), True)
        executor = call('concurrent.futures.ThreadPoolExecutor', 1)
        self.check(call(attr(executor, 'submit'), global_('builtins', 'str'), 'test'), False)

    def test_clients(self):
        for ctor, args in [('socket.socket', ()), ('http.client.HTTPConnection', ('localhost', 80)),
                           ('ftplib.FTP', ()), ('smtplib.SMTP', ()),
                           ('xmlrpc.client.ServerProxy', ('http://localhost',))]:
            value = call(ctor, *args)
            self.check(value, False)
            method = 'connect' if ctor != 'xmlrpc.client.ServerProxy' else 'test'
            connect_args = (() if ctor == 'http.client.HTTPConnection' else
                            (('localhost', 80),) if ctor == 'socket.socket' else ('localhost', 80))
            self.check(call(attr(value, method), *connect_args), True)
        self.check(call('ftplib.FTP', 'localhost'), True)
        self.check(call('smtplib.SMTP', 'localhost'), True)
        self.check(call('ftplib.FTP', ''), False)
        self.check(call(attr(call('pathlib.Path', '/tmp/foo'), 'write_text'), 'x'), True)
        self.check(call('pathlib.Path', '/tmp/foo'), False)

    def test_nested_loaders(self):
        evil = pkl(call('os.system', 'test'))
        clean = pkl({'text': 'os.system', 'f': global_('builtins', 'eval')})
        for value, expected in [(evil, True), (clean, False), (clean + evil, False), (evil + clean, True)]:
            for factory in [lambda d: d, lambda d: bytearray(d),
                            lambda d: call('base64.b64decode', base64.b64encode(d)),
                            lambda d: call('zlib.decompress', zlib.compress(d)),
                            lambda d: call('gzip.decompress', gzip.compress(d)),
                            lambda d: call('bz2.decompress', bz2.compress(d)),
                            lambda d: call('lzma.decompress', lzma.compress(d)),
                            lambda d: call('builtins.bytes.fromhex', d.hex()),
                            lambda d: call('operator.add', d[:8], d[8:]),
                            lambda d: call('builtins.bytearray', d),
                            lambda d: call(global_('builtins', 'str.encode'), d.decode('latin1'), 'latin1')]:
                decoded = factory(value)
                self.check(call('pickle.loads', decoded), expected)
                self.check(call('pickle.load', call('io.BytesIO', decoded)), expected)
            for wrapper in ('builtins.memoryview', 'pickle.PickleBuffer'):
                buffer = call(wrapper, value)
                self.check(buffer, False)
                self.check(call('pickle.loads', buffer), expected)
            self.check(call('torch.storage._load_from_bytes',
                            b''.join(pickle.dumps(x, 2) for x in [TORCH_MAGIC, 1001, {}]) + value + pickle.dumps([], 2)), evil in value)
        for loader in ('torch.load', 'numpy.load', 'joblib.load', 'pandas.read_pickle'):
            self.check(call(loader, '/tmp/outside'), True)
        self.check(call('numpy.load', call('io.BytesIO', npy(evil)), None, True), True)
        self.check(call('numpy.load', call('io.BytesIO', npy(clean)), None, True), False)
        self.check(call('numpy.load', call('io.BytesIO', npy(evil))), False)
        self.check(call('marshal.loads', marshal.dumps({'text': 'not executed'})), False)
        code = marshal.dumps(compile('1+1', '<test>', 'eval'))
        self.check(call('marshal.loads', code), False)
        self.check(call('types.FunctionType', call('marshal.loads', code), {}), True)
        self.check(call('pickle.loads', call('marshal.loads', marshal.dumps(evil))), True)
        self.check(call('types.FunctionType', call('marshal.loads', marshal.dumps(b'not code')), {}), False)

    def test_stream_sequences(self):
        clean = pkl({'safe': True})
        evil = pkl(call('os.system', 'test'))
        self.check(clean + evil, True, raw=True)
        self.check(clean * 8, False, raw=True)
        legacy = b''.join(pickle.dumps(x, 2) for x in [TORCH_MAGIC, 1001, {}])
        self.check(legacy + evil + pickle.dumps([], 2) + b'\0' * 100, True, raw=True)
        self.check(legacy + clean + pickle.dumps([], 2) + evil, False, raw=True)
        module = call('builtins.__import__', 'os')
        mutate = Raw(encode(module) + encode({'getpid': global_('builtins', 'eval')}) + b'b')
        self.check(pkl(mutate) + pkl(call('os.getpid', '1+1')), True, raw=True)

    def test_containers(self):
        for value, expected in [(pkl(call('os.system', 'test')), True), (pkl({'s': 'eval('}), False)]:
            wrappers = [gzip.compress, bz2.compress, lzma.compress, zlib.compress,
                        lambda d: zip_bytes([('file.notpkl', d)], prefix=b'not a local header'),
                        lambda d: zip_bytes([('array.npy', npy(d))]),
                        lambda d: zip_bytes([('archive/data.pkl', d), ('archive/version', b'3')]),
                        lambda d: gzip.compress(bz2.compress(zip_bytes([('data', d)])))]
            for wrapper in wrappers:
                with self.subTest(wrapper=wrapper, expected=expected):
                    self.check(wrapper(value), expected, raw=True)
            self.check(gzip.compress(pkl(None)) + gzip.compress(value), expected, raw=True)
            out = io.BytesIO()
            with tarfile.open(fileobj=out, mode='w') as t:
                info = tarfile.TarInfo('x.pkl')
                info.size = len(value)
                t.addfile(info, io.BytesIO(value))
            self.check(out.getvalue(), expected, raw=True)
        self.check(npy(b'os.system', '<f8'), False, raw=True)
        self.check(npy(pkl(call('os.system', 'test')), [('x', '<i4'), ('obj', '|O')]), True, raw=True)
        clean, evil = pkl(1), pkl(call('os.system', 'test'))
        self.check(gzip.compress(npy(b'\0' * 8, '<f8')) + gzip.compress(npy(evil)), True, raw=True)
        self.check(gzip.compress(clean[:3]) + gzip.compress(clean[3:]), False, raw=True)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.check(zip_bytes([('x.pkl', evil), ('x.pkl', clean)]), False, raw=True)
            self.check(zip_bytes([('x.pkl', clean), ('x.pkl', evil)]), True, raw=True)

    def test_keras(self):
        bytecode = {'class_name': '__lambda__', 'config': {'code': 'base64', 'defaults': None, 'closure': None}}
        named = {'module': 'builtins', 'class_name': 'function', 'config': 'relu'}
        for field in ('function', 'output_shape', 'activation'):
            for fn, expected in [(bytecode, True), (named, False), ('relu', False)]:
                layer = {'class_name': 'Lambda' if field != 'activation' else 'Dense', 'config': {field: fn}}
                for _ in range(8):
                    layer = {'class_name': 'TimeDistributed', 'config': {'layer': layer}}
                self.assertEqual(bool(keras_scan.scan_config(layer, 'cfg')), expected)
        self.assertFalse(keras_scan.scan_config({'class_name': 'Dense', 'config': {'name': 'Lambda os.system'}}, 'cfg'))

    def test_tf_closed_table(self):
        from modelgate.tfproto.saved_model_pb2 import SavedModel
        for op in list(TF_OPS) + ['SaveV2', 'RestoreV2', 'MergeV2Checkpoints', 'ShardedFilename', 'Identity', 'readfile', 'NotInPolicy']:
            sm = SavedModel()
            sm.meta_graphs.add()
            graph = sm.meta_graphs.add().graph_def
            fn = graph.library.function.add()
            fn.signature.name = 'hidden_in_attribute'
            node = fn.node_def.add()
            node.name = 'os.system ReadFile'
            node.op = op
            node.attr['text'].s = b'PyFunc'
            findings = savedmodel_scan.scan_saved_model(sm.SerializeToString(), 'saved_model.pb')
            self.assertEqual(bool(findings), op in TF_OPS, op)
        from google.protobuf import text_format
        self.assertEqual(bool(savedmodel_scan.scan_saved_model(text_format.MessageToString(sm).encode(), 'pbtxt', True)), False)


if __name__ == '__main__':
    unittest.main()
