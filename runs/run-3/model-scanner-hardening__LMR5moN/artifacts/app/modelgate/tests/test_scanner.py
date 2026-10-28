"""Static regression corpus, generated from policy semantics, never unpickled.

Run: python3 -B -m unittest discover -s modelgate/tests -v
"""
import bz2
import gzip
import io
import json
import lzma
import pickle
import struct
import tarfile
import unittest
import warnings
import zipfile

from modelgate.dispatch import scan_bytes
from modelgate.pickle_scan import scan_pickle, TORCH_MAGIC
from modelgate.policy import CALL_CAPABILITIES, TF_OPS
from modelgate.keras_scan import scan_config
from modelgate.savedmodel_scan import scan_saved_model


class Expr:
    def __init__(self, data):
        self.data = data


def g(name):
    module, attr = name.rsplit(':', 1) if ':' in name else name.rsplit('.', 1)
    return Expr(b'c' + module.encode() + b'\n' + attr.encode() + b'\n')


def enc(value):
    if isinstance(value, Expr):
        return value.data
    if value is None:
        return b'N'
    if isinstance(value, bool):
        return b'\x88' if value else b'\x89'
    if isinstance(value, str):
        data = value.encode('utf-8')
        return b'X' + struct.pack('<I', len(data)) + data
    if isinstance(value, bytes):
        return b'B' + struct.pack('<I', len(value)) + value
    if isinstance(value, int):
        return b'L' + str(value).encode() + b'L\n'
    if isinstance(value, tuple):
        return b'(' + b''.join(map(enc, value)) + b't'
    if isinstance(value, list):
        return b'](' + b''.join(map(enc, value)) + b'e'
    if isinstance(value, dict):
        return b'}(' + b''.join(enc(k) + enc(v) for k, v in value.items()) + b'u'
    raise TypeError(value)


def call(fn, *args):
    return Expr(enc(g(fn) if isinstance(fn, str) else fn) + enc(args) + b'R')


def partial(fn, *args, **kwargs):
    value = call('functools.partial', g(fn) if isinstance(fn, str) else fn, *args)
    return build(value, (g(fn) if isinstance(fn, str) else fn, args, kwargs, None))


def build(value, state):
    return Expr(enc(value) + enc(state) + b'b')


def new(fn, *args, **kwargs):
    return Expr(enc(g(fn)) + enc(args) + (enc(kwargs) + b'\x92' if kwargs else b'\x81'))


def attr(obj, name):
    return call('builtins.getattr', obj, name)


def dump(value, protocol=4):
    return b'\x80' + bytes([protocol]) + enc(value) + b'.'


def zip_bytes(entries, prefix=b''):
    out = io.BytesIO()
    out.write(prefix)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(out, 'w') as z:
            for name, data in entries:
                z.writestr(name, data)
    return out.getvalue()


def npy(payload, descr='|O', version=1):
    hdr = repr({'descr': descr, 'fortran_order': False, 'shape': (1,)}).encode()
    return b'\x93NUMPY' + bytes([version, 0]) + struct.pack('<H' if version == 1 else '<I', len(hdr) + 1) + hdr + b'\n' + payload


BAD = dump(call('os.system', 'modelgate-never-execute'))
GOOD = dump({'value': 'os.system', 'global': g('os.system')})


class PickleTests(unittest.TestCase):
    def verdict(self, expr, bad, protocol=4):
        findings = scan_pickle(dump(expr, protocol), 'test')
        self.assertEqual(bool(findings), bad, findings)
        if bad:
            self.assertNotIn('unsupported', [f.kind for f in findings], findings)

    def test_globals_are_inert(self):
        for name in CALL_CAPABILITIES:
            with self.subTest(name=name):
                self.verdict(g(name), False)
                self.verdict({'callable': g(name)}, False)
                self.verdict(new(name, 'ignored'), False)
                self.verdict(new(name, 'ignored', kw='ignored'), False)

    def test_direct_capabilities(self):
        for name in ('os.system', 'posix.system', 'nt.system', 'subprocess.Popen', 'subprocess.getoutput',
                     'builtins.eval', 'builtins.exec', 'builtins.compile', 'runpy.run_path',
                     'runpy._run_code', 'timeit.timeit', 'code:InteractiveInterpreter.runsource',
                     'bdb:Bdb.run', 'pdb.run', 'numpy.testing._private.utils.runstring',
                     'os.posix_spawn', 'os.execve', 'os.spawnv', 'platform.popen', 'pydoc.pipepager',
                     'builtins.open', 'io.open', '_io.open', 'os.remove', 'shutil.move',
                     'pathlib:Path.write_text', 'socket.create_connection', 'urllib.request.urlopen',
                     'http.client:HTTPConnection.connect', 'ssl.get_server_certificate', 'webbrowser.open'):
            with self.subTest(name=name):
                self.verdict(call(name, 'payload'), True)

    def test_compat_aliases(self):
        for name in ('__builtin__.eval', 'commands.getoutput', 'urllib2.urlopen', 'httplib:HTTPConnection.request',
                     'urllib.urlopen', 'os.system'):
            with self.subTest(name=name):
                self.verdict(call(name, 'payload'), True, 2)

    def test_inst_obj(self):
        for name in ('os.system', 'subprocess.Popen', 'builtins.open', 'socket.create_connection'):
            ref = enc(g(name))
            module, attrname = ref[1:-1].split(b'\n')
            for args, bad in (((), False), (('payload',), True)):
                inst = Expr(b'(' + b''.join(map(enc, args)) + b'i' + module + b'\n' + attrname + b'\n')
                obj = Expr(b'(' + ref + b''.join(map(enc, args)) + b'o')
                self.verdict(inst, bad, 0)
                self.verdict(obj, bad, 1)

    def test_indirection(self):
        module = call('importlib.import_module', 'os')
        fn = attr(module, 'system')
        self.verdict(module, False)
        self.verdict(fn, False)
        self.verdict(call(fn, 'payload'), True)
        self.verdict(call(call('operator.attrgetter', 'system'), module), False)
        self.verdict(call(call(call('operator.attrgetter', 'system'), module), 'payload'), True)
        self.verdict(call(call('operator.methodcaller', 'system', 'payload'), module), True)
        self.verdict(call(call('operator.itemgetter', 'fn'), {'fn': g('os.system')}), False)
        self.verdict(call(call(call('operator.itemgetter', 'fn'), {'fn': g('os.system')}), 'payload'), True)
        self.verdict(call('operator.call', g('os.system'), 'payload'), True)

    def test_partial_state(self):
        p = partial('os.system', 'payload')
        self.verdict(p, False)
        self.verdict(call(p), True)
        p = build(call('functools.partial', g('builtins.str')), (g('os.system'), ('payload',), {}, None))
        self.verdict(p, False)
        self.verdict(call(p), True)
        clean = build(p, (g('builtins.str'), ('payload',), {}, None))
        self.verdict(call(clean), False)
        # Mutating through one alias must affect the other alias.
        expr = Expr(enc(call('functools.partial', g('builtins.str'))) + b'q\x00' + enc((g('os.system'), ('payload',), {}, None)) + b'b0h\x00)R')
        self.verdict(expr, True)
        # A nested partial without an instance dictionary snapshots its target.
        # Later mutation of the inner partial must not retarget the outer one.
        for initial, later, bad in (('os.system', 'builtins.str', True), ('builtins.str', 'os.system', False)):
            inner = enc(call('functools.partial', g(initial), 'payload')) + b'q\x00'
            outer = enc(g('functools.partial')) + b'h\x00\x85Rq\x01'
            mutate = b'h\x00' + enc((g(later), ('payload',), {}, None)) + b'b0'
            self.verdict(Expr(inner + outer + mutate + b'h\x01)R'), bad)

    def test_memo(self):
        for put, get in ((b'p70\n', b'g70\n'), (b'q\x46', b'h\x46'), (b'r\x00\x01\x00\x00', b'j\x00\x01\x00\x00')):
            self.verdict(Expr(enc(g('os.system')) + put + b'0' + get + enc(('payload',)) + b'R'), True)
        # STACK_GLOBAL operands can be arbitrarily separated and memoized.
        expr = Expr(enc('os') + b'q\x00' + b'0' + enc('system') + b'q\x01' + b'0h\x00h\x01\x93' + enc(('payload',)) + b'R')
        self.verdict(expr, True)

    def test_network_constructor_distinction(self):
        for name, args in (('socket.socket', ()), ('http.client.HTTPConnection', ('localhost', 80)),
                           ('ftplib.FTP', ()), ('smtplib.SMTP', ()), ('xmlrpc.client.ServerProxy', ('http://example/',))):
            with self.subTest(name=name):
                self.verdict(call(name, *args), False)
        for name in ('ftplib.FTP', 'smtplib.SMTP', 'ftplib.FTP_TLS', 'smtplib.SMTP_SSL'):
            self.verdict(call(name, 'localhost'), True)
            self.verdict(call(attr(call(name), 'connect'), 'localhost'), True)
        self.verdict(call(attr(call('socket.socket'), 'connect'), ('localhost', 1)), True)
        self.verdict(call(attr(call('http.client.HTTPConnection', 'localhost'), 'request'), 'GET', '/'), True)
        self.verdict(call(attr(call('xmlrpc.client.ServerProxy', 'http://example/'), 'ping')), True)
        self.verdict(build(call('xmlrpc.client.ServerProxy', 'http://example/'), {}), True)
        self.verdict(call('urllib.request.urlopen', 'data:text/plain,inert-data'), False)

    def test_bound_methods(self):
        for cls, method in (('code.InteractiveInterpreter', 'runsource'), ('bdb.Bdb', 'run'), ('pdb.Pdb', 'run'),
                            ('pathlib.Path', 'write_text'), ('pathlib.PosixPath', 'unlink')):
            self.verdict(call(attr(call(cls), method), 'payload'), True)
        self.verdict(call(attr(call('io.BytesIO', b'abc'), 'read')), False)

    def test_serialized_functions(self):
        self.verdict(call('marshal.loads', b'opaque-data'), False)
        self.verdict(call('types.FunctionType', call('marshal.loads', b'opaque-bytecode'), {}), True)

    def test_nested_loads(self):
        for name in ('pickle.loads', '_pickle.loads', 'torch.storage._load_from_bytes'):
            with self.subTest(name=name):
                self.verdict(call(name, BAD), True)
                self.verdict(call(name, GOOD), False)
        for name in ('pickle.load', 'joblib.load', 'pandas.read_pickle', 'torch.load'):
            with self.subTest(name=name):
                self.verdict(call(name, call('io.BytesIO', BAD)), True)
                self.verdict(call(name, call('io.BytesIO', GOOD)), False)
                self.verdict(call(name, '/outside/file'), True)
        self.verdict(call('pickle.loads', GOOD + BAD), False)
        # The inner loader is harmless but returns a capability that is then called.
        self.verdict(call(call('pickle.loads', dump(g('os.system'))), 'payload'), True)
        self.verdict(call('pickle.loads', call('base64.b64decode', __import__('base64').b64encode(BAD))), True)
        self.verdict(call(attr(call('pickle.Unpickler', call('io.BytesIO', BAD)), 'load')), True)
        self.verdict(call('pickle.loads', call(attr(call('io.BytesIO', BAD), 'read'))), True)

    def test_numpy_nested(self):
        for name, args in (('numpy.load', (None, True)), ('numpy.lib.format.read_array', (True,))):
            self.verdict(call(name, call('io.BytesIO', npy(BAD)), *args), True)
            self.verdict(call(name, call('io.BytesIO', npy(GOOD)), *args), False)
            self.verdict(call(name, call('io.BytesIO', npy(BAD))), False)
        self.verdict(call(partial('numpy.load', allow_pickle=True), call('io.BytesIO', npy(BAD))), True)
        archive = zip_bytes([('x.npy', npy(BAD)), ('y.npy', npy(GOOD))])
        loaded = call(partial('numpy.load', allow_pickle=True), call('io.BytesIO', archive))
        self.verdict(loaded, False)
        self.verdict(call('operator.getitem', loaded, 'x'), True)
        self.verdict(call(attr(loaded, 'get'), 'y'), False)
        self.verdict(call('operator.getitem', call('numpy.load', call('io.BytesIO', archive)), 'x'), False)

    def test_compressed_stream_objects(self):
        for name, compressor in (('gzip.GzipFile', gzip.compress), ('bz2.BZ2File', bz2.compress), ('lzma.LZMAFile', lzma.compress)):
            for payload, bad in ((GOOD, False), (BAD, True)):
                buffer = call('io.BytesIO', compressor(payload))
                stream = call(partial(name, fileobj=buffer)) if name == 'gzip.GzipFile' else call(name, buffer)
                self.verdict(call('pickle.load', stream), bad)

    def test_joblib_arrays(self):
        for objects in (False, True):
            dtype = call('numpy.dtype', 'O8' if objects else 'f8')
            state = {'shape': (2,), 'order': 'C', 'dtype': dtype,
                     'subclass': g('numpy.ndarray'), 'allow_mmap': False,
                     'numpy_array_alignment_bytes': 16}
            wrapper = enc(build(new('joblib.numpy_pickle.NumpyArrayWrapper'), state))
            for bad in (False, True):
                # Numeric storage can contain bytes that look like opcodes.
                inline = (BAD if bad else GOOD) if objects else b'\x00' + (b'cos\nsystem\nR.' + bytes(16))[:16]
                suffix = b'0' + (enc(call('os.system', 'payload')) if bad and not objects else b'N')
                findings = scan_pickle(b'\x80\x04' + wrapper + inline + suffix + b'.', 'joblib')
                self.assertEqual(bool(findings), bad, findings)
                self.assertNotIn('unsupported', [f.kind for f in findings], findings)

    def test_lazy_callbacks(self):
        mapping = call('builtins.map', g('os.system'), ['payload'])
        self.verdict(mapping, False)
        self.verdict(call('builtins.list', mapping), True)
        self.verdict(call('builtins.next', mapping), True)
        self.verdict(call('builtins.list', call('builtins.map', g('builtins.str'), ['payload'])), False)

    def test_logging(self):
        self.verdict(call('logging.config.dictConfig', {'version': 1, 'handlers': {'h': {'class': 'logging.FileHandler', 'filename': '/outside'}}}), True)
        self.verdict(call('logging.config.dictConfig', {'version': 1, 'formatters': {'f': {'()': 'logging.Formatter', 'format': 'os.system'}}}), False)
        self.verdict(call('logging.config.dictConfig', {'version': 1, 'handlers': {'h': {'()': partial('os.system', 'payload')}}}), True)

    def test_protocols(self):
        for protocol in range(6):
            values = [None, True, 3, 1234567890123456789, 'exec(', b'subprocess', ('os.system',), [1, 2], {1, 2}, {'x': 'PyFunc'}]
            self.assertFalse(scan_pickle(pickle.dumps(values, protocol), 'safe'))

    def test_multiple_streams_legacy(self):
        self.assertFalse(scan_pickle(pickle.dumps(TORCH_MAGIC, 2), 'ordinary-integer'))
        self.assertTrue(scan_pickle(GOOD + BAD, 'test'))
        self.assertFalse(scan_pickle(GOOD + GOOD, 'test'))
        for payload, bad in ((GOOD, False), (BAD, True)):
            legacy = pickle.dumps(TORCH_MAGIC, 2) + pickle.dumps(1001, 2) + dump({'little_endian': True}) + payload + dump([]) + b'\x80malicious-looking raw storage'
            self.assertEqual(bool(scan_pickle(legacy, 'test')), bad)


class ContainerTests(unittest.TestCase):
    def check(self, data, bad, location='misleading.extension'):
        findings = scan_bytes(data, location)
        self.assertEqual(bool(findings), bad, findings)
        if bad:
            self.assertNotIn('unsupported', [f.kind for f in findings], findings)

    def test_compression(self):
        for compress in (gzip.compress, bz2.compress, lzma.compress, __import__('zlib').compress):
            self.check(compress(BAD), True)
            self.check(compress(GOOD), False)
        self.check(gzip.compress(GOOD) + gzip.compress(BAD), True)
        self.check(gzip.compress(GOOD) + gzip.compress(GOOD), False)
        self.check(gzip.compress(bz2.compress(lzma.compress(BAD))), True)

    def test_zip_directory(self):
        for prefix in (b'', b'arbitrary prefix PK\x03\x04 bytes', BAD):
            self.check(zip_bytes([('same.pkl', BAD), ('same.pkl', GOOD)], prefix), False)
            self.check(zip_bytes([('same.pkl', GOOD), ('same.pkl', BAD)], prefix), True)
            self.check(zip_bytes([('model.wrong', BAD), ('readme.txt', b'Model information')], prefix), True)
        self.check(zip_bytes([('model.wrong', GOOD), ('readme.txt', b'Model information')]), False)

    def test_torch_zip(self):
        for payload, bad in ((GOOD, False), (BAD, True)):
            self.check(zip_bytes([('weights/data.pkl', payload), ('weights/data/0', BAD), ('weights/version', b'3\n')]), bad)

    def test_tar(self):
        for payload, bad in ((GOOD, False), (BAD, True)):
            out = io.BytesIO()
            with tarfile.open(fileobj=out, mode='w') as tf:
                entry = tarfile.TarInfo('model.weird')
                entry.size = len(payload)
                tf.addfile(entry, io.BytesIO(payload))
            self.check(out.getvalue(), bad)
            self.check(gzip.compress(out.getvalue()), bad)

    def test_numpy(self):
        for version in (1, 2, 3):
            self.check(npy(BAD, version=version), True)
            self.check(npy(GOOD, version=version), False)
            self.check(npy(BAD, descr='<u1', version=version), False)
        self.check(npy(BAD, descr=[('name', '<i4'), ('stuff', '|O')]), True)
        self.check(zip_bytes([('array.npy', npy(BAD))]), True)
        self.check(zip_bytes([('array.npy', npy(GOOD))]), False)

    def test_keras(self):
        lam = {'class_name': 'Lambda', 'config': {'function': {'class_name': '__lambda__', 'config': {'code': 'c2VyaWFsaXplZA=='}}}}
        clean = {'class_name': 'Lambda', 'config': {'function': {'module': 'builtins', 'class_name': 'function', 'config': 'relu'}}}
        for layer, bad in ((lam, True), (clean, False), ({'class_name': 'Activation', 'config': {'activation': 'eval'}}, False)):
            for _ in range(5):
                layer = {'class_name': 'TimeDistributed', 'config': {'layer': {'class_name': 'Sequential', 'config': {'layers': [layer]}}}}
            self.assertEqual(bool(scan_config(layer, 'config')), bad)
            archive = zip_bytes([('config.json', json.dumps(layer).encode()), ('metadata.json', b'{}'), ('model.weights.h5', b'weight data')])
            self.check(archive, bad)

    def test_tensorflow(self):
        from modelgate.tfproto.saved_model_pb2 import SavedModel
        from google.protobuf.text_format import MessageToString
        for name in list(TF_OPS) + ['SaveV2', 'RestoreV2', 'MergeV2Checkpoints', 'ShardedFilename', 'ShardedFilespec', 'Const', 'readfile', 'NotReadFile']:
            for function in (False, True):
                sm = SavedModel()
                sm.meta_graphs.add()
                graph = sm.meta_graphs.add().graph_def
                if function:
                    fn = graph.library.function.add()
                    fn.signature.name = 'hidden_body'
                    node = fn.node_def.add()
                    graph.node.add(name='ref', op='PartitionedCall').attr['f'].func.name = fn.signature.name
                else:
                    node = graph.node.add()
                node.name, node.op = 'ReadFile PyFunc', name
                node.attr['text'].s = b'WriteFile EagerPyFunc'
                with self.subTest(name=name, function=function):
                    self.assertEqual(bool(scan_saved_model(sm.SerializeToString(), 'saved_model.pb')), name in TF_OPS)
                    self.assertEqual(bool(scan_saved_model(MessageToString(sm).encode(), 'saved_model.pbtxt', text=True)), name in TF_OPS)


if __name__ == '__main__':
    unittest.main()
