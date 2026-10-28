"""Policy regression tests. Payload builders only emit bytes; never load them."""
import base64
import bz2
import gzip
import io
import json
import lzma
import marshal
import os
import pickle
import pickletools
import struct
import tarfile
import tempfile
import unittest
import zipfile
import zlib

from modelgate.abstract import Context, Detected, LoadError, LimitError
from modelgate.dispatch import analyze_bytes
from modelgate.keras_scan import scan_config
from modelgate.pickle_scan import TORCH_MAGIC, scan_pickle


def g(name):
    mod, name = name.rsplit(':', 1) if ':' in name else name.rsplit('.', 1)
    return b'c' + mod.encode() + b'\n' + name.encode() + b'\n'


def literal(value):
    data = pickle.dumps(value, protocol=4)
    ops = list(pickletools.genops(data))
    return b''.join(data[pos:ops[i + 1][2]] for i, (op, arg, pos) in enumerate(ops[:-1])
                    if op.name not in ('PROTO', 'FRAME'))


def tup(*values):
    return b'(' + b''.join(values) + b't'


def call(func, *args):
    return func + tup(*args) + b'R'


def attr(obj, name):
    return call(g('builtins.getattr'), obj, literal(name))


def imp(name):
    return call(g('importlib.import_module'), literal(name))


def partial(func, *args, **kw):
    value = call(g('functools.partial'), func, *args)
    if kw:
        # State values may be symbolic pickle expressions, emitted verbatim.
        fields = b'(' + b''.join(literal(k) + v for k, v in kw.items()) + b'd'
        value += tup(func, tup(*args), fields, literal(None)) + b'b'
    return value


def p(expr, proto=4):
    return b'\x80' + bytes([proto]) + expr + b'.'


def zip_bytes(entries, prefix=b''):
    f = io.BytesIO()
    with zipfile.ZipFile(f, 'w') as z:
        for name, data in entries:
            z.writestr(name, data)
    return prefix + f.getvalue()


def npy(data, descr='O'):
    header = repr({'descr': descr, 'fortran_order': False, 'shape': (1,)}).encode() + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + data


def legacy(obj, fifth=None):
    return (pickle.dumps(TORCH_MAGIC, 2) + pickle.dumps(1001, 2) +
            pickle.dumps({'protocol_version': 1001}, 2) + obj +
            (pickle.dumps([], 2) if fifth is None else fifth) + b'raw storage')


BAD = call(g('os.system'), literal('never-run-this-command'))
SAFE = literal({'text': 'os.system and subprocess are just data'})


class PolicyTests(unittest.TestCase):
    def assert_scan(self, expr, expected, *, proto=4, raw=False, name='test.bin'):
        data = expr if raw else p(expr, proto)
        try:
            analyze_bytes(data, name, Context())
            verdict = False
        except Detected:
            verdict = True
        except LoadError:
            verdict = False
        except Exception as exc:
            self.fail(f'unexpected unsupported result: {type(exc).__name__}: {exc}')
        self.assertEqual(verdict, expected)

    def test_data_is_inert(self):
        for expr in (SAFE, g('os.system'), tup(g('builtins.eval'), g('subprocess.Popen')),
                     partial(g('os.system'), literal('id')),
                     call(g('builtins.__import__'), literal('os'))):
            with self.subTest(expr=expr):
                self.assert_scan(expr, False)

    def test_protocols_and_memo(self):
        for proto in range(6):
            for module in ('os', 'posix', 'nt'):
                self.assert_scan(call(g(module + '.system'), literal('id')), True, proto=proto)
            self.assert_scan(g('os.system') + b'p999\n0g999\n' + tup(literal('id')) + b'R', True, proto=proto)
        for module in ('__builtin__',):
            self.assert_scan(call(g(module + '.eval'), literal('1+1')), True, proto=2)
        self.assert_scan(call(g('commands.getoutput'), literal('id')), True, proto=0)
        self.assert_scan(call(g('urllib2.urlopen'), literal('http://localhost/')), True, proto=2)

    def test_opcode_execution_boundaries(self):
        self.assert_scan(g('os.system') + literal(['id']) + b'R', False)
        self.assert_scan(b'(' + literal('id') + b'ios\nsystem\n', True, proto=0)
        self.assert_scan(b'(' + g('os.system') + literal('id') + b'o', True, proto=0)
        self.assert_scan(b'(' + partial(g('os.system'), literal('id')) + b'o', True)
        self.assert_scan(g('subprocess.Popen') + tup(literal(['id'])) + b'\x81', False)
        self.assert_scan(g('subprocess.Popen') + tup(literal(['id'])) + literal({}) + b'\x92', False)
        self.assert_scan(b'(isubprocess\nPopen\n', False, proto=0)
        self.assert_scan(b'(' + g('subprocess.Popen') + b'o', False)
        self.assert_scan(call(g('subprocess.Popen'), literal(['id'])), True)
        self.assert_scan(g('builtins.eval') + tup(literal('1')) + b'\x81', False)
        self.assert_scan(literal('storage') + b'Q' + BAD, False)
        self.assert_scan(call(g('shutil:os.system'), literal('id')), True, proto=4)
        self.assert_scan(call(g('shutil:os.system'), literal('id')), False, proto=2)

    def test_attribute_routes(self):
        refs = [attr(imp('os'), 'system'),
                call(call(g('operator.attrgetter'), literal('os.system')), imp('shutil')),
                call(g('pkgutil.resolve_name'), literal('os:system')),
                call(g('pkgutil.resolve_name'), literal('shutil.os.system')),
                call(g('operator.getitem'), call(g('builtins.vars'), imp('os')), literal('system')),
                attr(call(g('operator.getitem'), attr(imp('sys'), 'modules'), literal('os')), 'system'),
                call(g('copy.copy'), g('os.system')),
                call(g('functools.lru_cache'), g('os.system')),
                call(call(g('functools.lru_cache'), literal(None)), g('os.system'))]
        for ref in refs:
            with self.subTest(ref=ref):
                self.assert_scan(ref, False)
                self.assert_scan(call(ref, literal('id')), True)
        self.assert_scan(call(call(g('operator.methodcaller'), literal('system'), literal('id')), imp('os')), True)

    def test_partial_retarget_and_module_build(self):
        base = partial(g('builtins.str'), literal('x')) + b'p0\n0'
        state = tup(g('os.system'), tup(literal('id')), literal({}), literal(None))
        self.assert_scan(base + b'g0\n' + state + b'b0g0\n' + tup() + b'R', True)
        module = imp('builtins') + b'(' + literal('harmless_name') + g('os.system') + b'db0'
        self.assert_scan(module + call(g('builtins.harmless_name'), literal('id')), True)
        self.assert_scan(module + g('builtins.harmless_name'), False)
        streams = p(module + literal(None)) + p(call(g('builtins.harmless_name'), literal('id')))
        self.assert_scan(streams, True, raw=True)

    def test_lazy_and_eager_callbacks(self):
        mapped = call(g('builtins.map'), g('os.system'), literal(['id']))
        filtered = call(g('builtins.filter'), g('os.system'), literal(['id']))
        starred = call(g('itertools.starmap'), g('os.system'), literal([('id',)]))
        for iterator in (mapped, filtered, starred):
            self.assert_scan(iterator, False)
            self.assert_scan(call(g('builtins.list'), iterator), True)
            self.assert_scan(call(g('builtins.next'), iterator), True)
        self.assert_scan(call(g('builtins.list'), call(g('builtins.map'), g('os.system'), literal([]))), False)
        for consumer in ('builtins.sorted', 'builtins.min', 'builtins.max'):
            self.assert_scan(call(partial(g(consumer), key=g('os.system')), literal(['id'])), True)
            self.assert_scan(call(partial(g(consumer), key=g('builtins.str')), literal([2, 1])), False)
        self.assert_scan(call(g('operator.call'), g('os.system'), literal('id')), True)
        self.assert_scan(call(g('functools.reduce'), partial(g('builtins.eval')), literal(['1', {}])), True)

    def test_threads_and_clients(self):
        thread = call(partial(g('threading.Thread'), target=g('os.system'), args=literal(('id',))))
        self.assert_scan(thread, False)
        self.assert_scan(call(attr(thread, 'start')), True)
        executor = call(g('concurrent.futures.ThreadPoolExecutor'))
        self.assert_scan(call(attr(executor, 'submit'), g('os.system'), literal('id')), True)
        for cls, args, method in [('socket.socket', (), 'connect'),
                                  ('http.client.HTTPConnection', ('localhost', 80), 'connect'),
                                  ('ftplib.FTP', (), 'connect'),
                                  ('smtplib.SMTP', (), 'connect')]:
            obj = call(g(cls), *(literal(v) for v in args))
            self.assert_scan(obj, False)
            self.assert_scan(call(attr(obj, method), literal('localhost')), True)
        self.assert_scan(call(g('ftplib.FTP'), literal('localhost')), True)
        self.assert_scan(call(g('smtplib.SMTP'), literal('localhost')), True)
        self.assert_scan(call(g('xmlrpc.client.ServerProxy'), literal('http://localhost')), False)

    def test_nested_deserialization(self):
        bad, safe = p(BAD), p(SAFE)
        self.assert_scan(call(g('pickle.loads'), literal(bad)), True)
        self.assert_scan(call(g('pickle.loads'), literal(safe + bad)), False)
        self.assert_scan(call(g('pickle.load'), call(g('io.BytesIO'), literal(bad))), True)
        for encode, decode in [(base64.b64encode, 'base64.b64decode'), (zlib.compress, 'zlib.decompress'),
                               (bz2.compress, 'bz2.decompress'), (lzma.compress, 'lzma.decompress')]:
            self.assert_scan(call(g('pickle.loads'), call(g(decode), literal(encode(bad)))), True)
            self.assert_scan(call(g('pickle.loads'), call(g(decode), literal(encode(safe)))), False)
        self.assert_scan(call(g('pickle.loads'), call(g('builtins.bytes:fromhex'), literal(bad.hex()))), True)
        self.assert_scan(call(g('pickle.loads'), call(g('builtins.bytearray'), literal(bad))), True)
        self.assert_scan(call(g('pickle.loads'), call(g('operator.add'), literal(bad[:8]), literal(bad[8:]))), True)
        self.assert_scan(call(g('torch.storage._load_from_bytes'), literal(legacy(p(SAFE), p(BAD)))), True)
        self.assert_scan(call(g('torch.storage._load_from_bytes'), literal(legacy(p(SAFE)))), False)
        self.assert_scan(call(g('torch.load'), call(g('io.BytesIO'), literal(legacy(p(BAD))))), True)
        self.assert_scan(call(g('numpy.load'), literal('/outside/array.npy')), True)
        # A nested benign pickle can return a callable that the outer stream calls.
        self.assert_scan(call(call(g('pickle.loads'), literal(p(g('os.system')))), literal('id')), True)

    def test_streams_and_containers(self):
        bad, safe = p(BAD), p(SAFE)
        self.assert_scan(safe + bad, True, raw=True)
        self.assert_scan(legacy(safe, bad), True, raw=True)
        self.assert_scan(legacy(safe) + bad, False, raw=True)
        for compressor in (gzip.compress, bz2.compress, lzma.compress, zlib.compress):
            self.assert_scan(compressor(bad), True, raw=True)
            self.assert_scan(compressor(safe), False, raw=True)
        self.assert_scan(gzip.compress(safe) + gzip.compress(bad), True, raw=True)
        self.assert_scan(npy(bad), True, raw=True)
        self.assert_scan(npy(safe), False, raw=True)
        self.assert_scan(zip_bytes([('x.npy', npy(bad))]), True, raw=True)
        self.assert_scan(zip_bytes([('a.weird', bad)], b'arbitrary-prefix'), True, raw=True)
        self.assert_scan(zip_bytes([('x', bad), ('x', safe)]), False, raw=True)
        self.assert_scan(zip_bytes([('x', safe), ('x', bad)]), True, raw=True)
        torch = zip_bytes([('archive/data.pkl', bad), ('archive/version', b'3')])
        self.assert_scan(torch, True, raw=True)
        tar = io.BytesIO()
        with tarfile.open(fileobj=tar, mode='w') as tf:
            info = tarfile.TarInfo('nested.pt')
            info.size = len(torch)
            tf.addfile(info, io.BytesIO(torch))
        self.assert_scan(gzip.compress(tar.getvalue()), True, raw=True)

    def test_keras(self):
        bytecode = {'class_name': '__lambda__', 'config': {'code': 'abc'}}
        function = {'class_name': 'function', 'module': 'builtins', 'config': 'relu'}
        for field in ('function', 'output_shape'):
            for value, expected in ((bytecode, True), (function, False), ('relu', False)):
                config = {'class_name': 'Lambda', 'config': {field: value}}
                nested = {'class_name': 'Sequential', 'config': {'layers': [
                    {'class_name': 'TimeDistributed', 'config': {'layer': config}}]}}
                self.assertEqual(bool(scan_config(nested, 'keras')), expected)
        activation = {'class_name': '__lambda__', 'config': {'value': ['abc', None, None]}}
        self.assertTrue(scan_config({'class_name': 'Dense', 'config': {'activation': activation}}, 'keras'))
        self.assertFalse(scan_config({'class_name': 'Dense', 'config': {'activation': 'relu'}}, 'keras'))

    def test_legacy_string_decoding(self):
        high_bytes = b'T\x01\x00\x00\x00\xff'
        self.assert_scan(high_bytes + b'0' + BAD, False)
        payload = p(high_bytes + b'0' + BAD)
        self.assert_scan(call(partial(g('pickle.loads'), encoding=literal('latin1')), literal(payload)), True)
        alias_payload = p(call(g('__builtin__.eval'), literal('1+1')), 2)
        self.assert_scan(call(g('pickle.loads'), literal(alias_payload)), True)
        self.assert_scan(call(partial(g('pickle.loads'), fix_imports=literal(False)), literal(alias_payload)), False)

    def test_deferred_construction(self):
        compiler = call(g('codeop.CommandCompiler'))
        self.assert_scan(compiler, False)
        self.assert_scan(call(compiler, literal('x=1')), True)
        config = literal({'version': 1, 'handlers': {'h': {'class': 'logging.FileHandler', 'filename': '/outside/x'}}})
        configurator = call(g('logging.config.DictConfigurator'), config)
        self.assert_scan(configurator, False)
        self.assert_scan(call(attr(configurator, 'configure')), True)
        delayed = call(partial(g('logging.FileHandler'), delay=literal(True)), literal('/outside/x'))
        self.assert_scan(delayed, False)
        record = call(g('logging.LogRecord'), *(literal(v) for v in ('name', 20, 'source.py', 1, 'message', (), None)))
        self.assert_scan(record, False)
        self.assert_scan(call(attr(delayed, 'emit'), record), True)
        clients = [('logging.handlers.SocketHandler', ('localhost', 8000)),
                   ('logging.handlers.HTTPHandler', ('localhost:8000', '/')),
                   ('logging.handlers.SMTPHandler', (('localhost', 25), 'from@example.invalid', ['to@example.invalid'], 'subject'))]
        for cls, args in clients:
            handler = call(g(cls), *(literal(v) for v in args))
            self.assert_scan(handler, False)
            self.assert_scan(call(attr(handler, 'emit'), record), True)
        self.assert_scan(call(g('builtins.list'), call(g('itertools.accumulate'), literal([]))), False)
        self.assert_scan(call(g('builtins.list'), call(g('itertools.pairwise'), literal([]))), False)
        self.assert_scan(call(g('functools.reduce'), g('os.system'), literal([])), False)
        self.assert_scan(call(g('zlib.decompress'), literal(b'not compressed data')), False)
        it = call(g('builtins.map'), g('os.system'), literal(['id']))
        self.assert_scan(call(g('builtins.list'), call(attr(it, '__iter__'))), True)

    def test_joblib_arrays_and_iterator_factories(self):
        def wrapper(dtype, shape, data):
            attrs = {'shape': literal(shape), 'dtype': call(g('numpy.dtype'), literal(dtype)),
                     'order': literal('C'), 'numpy_array_alignment_bytes': literal(16)}
            state = b'(' + b''.join(literal(k) + v for k, v in attrs.items()) + b'd'
            return g('joblib.numpy_pickle.NumpyArrayWrapper') + tup() + b'\x81' + state + b'b' + data
        numeric = wrapper('u1', (len(p(BAD)),), b'\x00' + p(BAD))
        self.assert_scan(numeric, False)
        self.assert_scan(numeric + b'0' + BAD, True)
        self.assert_scan(wrapper('O8', (1,), p(SAFE)), False)
        self.assert_scan(wrapper('O8', (1,), p(BAD)), True)
        lazy = g('itertools.starmap') + tup(g('os.system'), literal([('id',)])) + b'\x81'
        self.assert_scan(lazy, False)
        self.assert_scan(call(g('builtins.list'), lazy), True)
        for consumer in ('itertools.product', 'itertools.permutations', 'itertools.combinations'):
            mapped = call(g('builtins.map'), g('os.system'), literal(['id']))
            extra = () if consumer.endswith('product') else (literal(1),)
            self.assert_scan(call(g(consumer), mapped, *extra), True)
        self.assert_scan(call(g('pickle.loads'), call(attr(call(partial(g('gzip.GzipFile'), fileobj=call(g('io.BytesIO'), literal(gzip.compress(p(BAD)))))), 'read'))), True)
        json_target = call(g('operator.getitem'), call(g('json.loads'), literal('{"name":"system"}')), literal('name'))
        self.assert_scan(call(call(g('builtins.getattr'), imp('os'), json_target), literal('id')), True)

    def test_marshaled_code_and_named_factories(self):
        code = marshal.dumps(compile('value = 1', '<fixture>', 'exec'))
        obj = call(g('marshal.loads'), literal(code))
        self.assert_scan(obj, False)
        self.assert_scan(call(g('types.FunctionType'), obj, literal({})), True)
        self.assert_scan(call(g('types.FunctionType'), obj, literal([])), False)
        self.assert_scan(call(g('types.FunctionType'), call(g('marshal.loads'), literal(marshal.dumps(123))), literal({})), False)
        bundled = marshal.dumps((compile('value = 1', '<fixture>', 'exec'), 'metadata'))
        extracted = call(g('operator.getitem'), call(g('marshal.loads'), literal(bundled)), literal(0))
        self.assert_scan(call(g('types.FunctionType'), extracted, literal({})), True)
        self.assert_scan(call(g('types.FunctionType'), call(g('marshal.loads'), literal(b'not marshal')), literal({})), False)
        self.assert_scan(call(g('cloudpickle.loads'), literal(p(SAFE) + p(BAD))), False)
        self.assert_scan(call(g('cloudpickle.loads'), literal(p(BAD))), True)
        self.assert_scan(call(g('pickle._loads'), literal(p(BAD))), True)
        self.assert_scan(call(g('builtins.eval'), literal(42)), False)
        self.assert_scan(call(g('os.system'), literal(42)), False)
        self.assert_scan(g('os.nonexistent') + b'0' + BAD, False)
        self.assert_scan(call(g('timeit.timeit'), literal('value = 1')), True)
        self.assert_scan(call(g('timeit.timeit'), g('builtins.str')), False)
        self.assert_scan(call(g('timeit.timeit'), partial(g('os.system'), literal('id'))), True)

    def test_loader_boundaries(self):
        bad, safe = p(BAD), p(SAFE)
        self.assert_scan(call(g('joblib.load'), call(g('io.BytesIO'), literal(safe + bad))), False)
        self.assert_scan(call(g('torch.load'), call(g('io.BytesIO'), literal(safe + bad))), False)
        self.assert_scan(call(g('torch.load'), call(g('io.BytesIO'), literal(gzip.compress(bad)))), False)
        zipped = zip_bytes([('p/data.pkl', bad), ('p/version', b'3')], b'prefix')
        self.assert_scan(call(g('torch.load'), call(g('io.BytesIO'), literal(zipped))), False)
        # read_array has a required NPY header; NumPy object loading is opt-in.
        self.assert_scan(call(g('numpy.lib.format.read_array'), call(g('io.BytesIO'), literal(bad))), False)
        self.assert_scan(call(g('numpy.load'), call(g('io.BytesIO'), literal(npy(bad)))), False)
        enabled = partial(g('numpy.load'), allow_pickle=literal(True))
        self.assert_scan(call(enabled, call(g('io.BytesIO'), literal(npy(bad)))), True)
        self.assert_scan(call(g('pickle.loads'), literal(safe + bad)), False)
        # Archive members are loaded independently, including after an error.
        self.assert_scan(zip_bytes([('broken.pkl', b'\x80\x04Pstorage\n.'), ('good.pkl', bad)]), True, raw=True)
        # Legacy loading returns stream 4 and still reads stream 5.
        ret = call(g('torch.load'), call(g('io.BytesIO'), literal(legacy(p(g('os.system'))))))
        self.assert_scan(call(ret, literal('id')), True)

    def test_unbound_data_methods(self):
        bad = p(BAD)
        decoded = call(g('builtins.bytes:join'), literal(b''), literal([bad[:10], bad[10:]]))
        self.assert_scan(call(g('pickle.loads'), decoded), True)
        self.assert_scan(call(g('pickle.loads'), call(g('builtins.memoryview'), literal(bad))), True)
        target = call(g('builtins.dict:get'), call(g('builtins.vars'), imp('os')), literal('system'))
        self.assert_scan(call(target, literal('id')), True)
        thread = call(partial(g('threading.Thread'), target=g('os.system'), args=literal(('id',))))
        self.assert_scan(call(g('threading:Thread.start'), thread), True)
        remote = attr(attr(call(g('xmlrpc.client.ServerProxy'), literal('http://localhost')), 'system'), 'listMethods')
        self.assert_scan(call(remote), True)

    def test_tf_closed_table(self):
        from modelgate.tfproto.saved_model_pb2 import SavedModel
        from modelgate.savedmodel_scan import scan_saved_model
        from modelgate.policy import TF_OPS
        from google.protobuf import text_format
        for op in list(TF_OPS) + ['SaveV2', 'RestoreV2', 'MergeV2Checkpoints', 'ShardedFilename', 'ShardedFilespec', 'Const', 'Identity', 'readfile']:
            model = SavedModel()
            model.meta_graphs.add().graph_def.node.add(name='ReadFile', op='Const')
            graph = model.meta_graphs.add().graph_def
            graph.library.function.add().node_def.add(name='innocent', op=op)
            graph.node.add(name='PyFunc', op='Const').attr['value'].s = b'ReadFile'
            for text in (False, True):
                data = text_format.MessageToString(model).encode() if text else model.SerializeToString()
                self.assertEqual(bool(scan_saved_model(data, 'model', text=text)), op in TF_OPS)


if __name__ == '__main__':
    unittest.main()
