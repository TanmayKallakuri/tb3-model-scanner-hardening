"""Policy regressions. Payloads are assembled and scanned, never unpickled."""
import base64
import bz2
import gzip
import io
import json
import lzma
import pickle
import pickletools
import struct
import tarfile
import unittest
import warnings
import zipfile

from modelgate.dispatch import scan_bytes
from modelgate.keras_scan import scan_config
from modelgate.pickle_scan import scan_pickle
from modelgate.policy import TF_OPS
from modelgate.savedmodel_scan import scan_saved_model


class Raw:
    def __init__(self, data):
        self.data = data


def val(value):
    if isinstance(value, Raw):
        return value.data
    if isinstance(value, tuple):
        return b'(' + b''.join(val(x) for x in value) + b't'
    if isinstance(value, list):
        return b'](' + b''.join(val(x) for x in value) + b'e'
    if isinstance(value, dict):
        return b'}(' + b''.join(val(k) + val(v) for k, v in value.items()) + b'u'
    # Independent literals must not reuse the builder's memo slots.
    data = pickle.dumps(value, protocol=2)
    ops = list(pickletools.genops(data))
    return b''.join(data[pos:ops[i + 1][2]] for i, (op, _, pos) in enumerate(ops[:-1])
                    if op.name not in ('PROTO', 'BINPUT', 'LONG_BINPUT', 'PUT', 'MEMOIZE'))


def glob(module, name):
    return Raw(b'c' + module.encode() + b'\n' + name.encode() + b'\n')


def call(fn, *args):
    return Raw(val(fn) + val(args) + b'R')


def attr(obj, name):
    return call(glob('builtins', 'getattr'), obj, name)


def module(name):
    return call(glob('importlib', 'import_module'), name)


def stream(value, protocol=4):
    return b'\x80' + bytes([protocol]) + val(value) + b'.'


def archive(entries, prefix=b''):
    target = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(target, 'w') as z:
            for name, data in entries:
                z.writestr(name, data)
    return prefix + target.getvalue()


def tar(entries):
    target = io.BytesIO()
    with tarfile.open(fileobj=target, mode='w') as tf:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return target.getvalue()


def npy(payload, dtype='|O', version=1):
    header = repr({'descr': dtype, 'fortran_order': False, 'shape': (1,)}).encode() + b'\n'
    return b'\x93NUMPY' + bytes([version, 0]) + struct.pack('<H' if version == 1 else '<I', len(header)) + header + payload


class ScannerTests(unittest.TestCase):
    def check(self, data, malicious, kind=None, container=False):
        findings = scan_bytes(data, 'fixture') if container else scan_pickle(data, 'fixture')
        self.assertEqual(bool(findings), malicious, findings)
        if kind:
            self.assertIn(kind, {f.kind for f in findings}, findings)
        if malicious and kind != 'unsupported':
            self.assertNotIn('unsupported', {f.kind for f in findings}, findings)
        return findings

    def test_literals_and_data_globals(self):
        dangerous = [glob('os', 'system'), glob('builtins', 'eval'), glob('subprocess', 'Popen')]
        for proto in range(6):
            for data in ['os.system', 'subprocess', 'eval(', 'exec(', ['ReadFile', 'PyFunc'], {'danger': 'os.system'}]:
                self.check(pickle.dumps(data, protocol=proto), False)
            for fn in dangerous:
                self.check(stream([fn, {'callback': fn}], proto), False)

    def test_direct_calls_aliases_and_fix_imports(self):
        for mod, name, args, kind in [
            ('posix', 'system', ('echo inert',), 'process'),
            ('nt', 'spawnv', (0, 'dummy', []), 'process'),
            ('commands', 'getoutput', ('echo inert',), 'process'),
            ('__builtin__', 'eval', ('1+1',), 'code-execution'),
            ('_io', 'open', ('outside',), 'file-access'),
            ('urllib2', 'urlopen', ('http://invalid.test/',), 'network'),
            ('httplib', 'HTTPConnection.connect', (None,), 'network'),
            ('shutil', 'move', ('a', 'b'), 'file-access'),
            ('ssl', 'get_server_certificate', (('invalid.test', 443),), 'network'),
        ]:
            for proto in range(6):
                self.check(stream(call(glob(mod, name), *args), proto), True, kind)

    def test_opcode_stack_and_memo(self):
        # LONG_BINPUT/LONG_BINGET, POP, DUP and both forms of GLOBAL.
        fn = b'cos\nsystem\n'
        for put, get in [(b'p123\n', b'g123\n'), (b'q\xfe', b'h\xfe'), (b'r' + struct.pack('<I', 700), b'j' + struct.pack('<I', 700))]:
            self.check(fn + put + b'0' + get + val(('echo inert',)) + b'R.', True, 'process')
            self.check(fn + put + b'0' + get + b'.', False)
        self.check(val('os') + b'q\x00' + val('system') + b'q\x01' + b'00h\x00h\x01\x93' + val(('echo inert',)) + b'R.', True, 'process')
        self.check(fn + b'2' + val(('echo inert',)) + b'R0.', True, 'process')
        self.check(fn + b'(NN1' + val(('echo inert',)) + b'R.', True, 'process')

    def test_inst_obj_newobj(self):
        args = val('echo inert')
        self.check(b'(' + args + b'ios\nsystem\n.', True, 'process')
        self.check(b'(cos\nsystem\n' + args + b'o.', True, 'process')
        self.check(b'(ios\nsystem\n.', False)
        self.check(b'(cos\nsystem\no.', False)
        self.check(b'csubprocess\nPopen\n' + val((['echo', 'inert'],)) + b'\x81.', False)
        self.check(b'csubprocess\nPopen\n' + val((['echo', 'inert'],)) + b'}\x92.', False)
        self.check(b'Poutside\n.', False)
        self.check(val(('storage', 'outside')) + b'Q.', False)

    def test_getattr_and_operators(self):
        system = attr(module('os'), 'system')
        self.check(stream(call(system, 'echo inert')), True, 'process')
        self.check(stream(system), False)
        self.check(stream(call(call(glob('operator', 'attrgetter'), 'system'), module('os'))), False)
        self.check(stream(call(call(call(glob('operator', 'attrgetter'), 'system'), module('os')), 'echo inert')), True, 'process')
        self.check(stream(call(call(glob('operator', 'methodcaller'), 'system', 'echo inert'), module('os'))), True, 'process')
        self.check(stream(call(call(glob('operator', 'itemgetter'), 'f'), {'f': glob('os', 'system')})), False)
        self.check(stream(call(call(call(glob('operator', 'itemgetter'), 'f'), {'f': glob('os', 'system')}), 'echo inert')), True, 'process')
        namespace = attr(module('builtins'), '__dict__')
        self.check(stream(call(call(glob('operator', 'getitem'), namespace, 'eval'), '1+1')), True, 'code-execution')

    def test_partial_retarget_and_memo_identity(self):
        p = call(glob('functools', 'partial'), glob('builtins', 'str'), 'inert')
        self.check(stream(p), False)
        self.check(stream(call(call(glob('functools', 'partial'), glob('os', 'system'), 'echo inert'))), True, 'process')
        state = (glob('os', 'system'), ('echo inert',), {}, None)
        # A memo alias points to the same object after BUILD.
        payload = val(p) + b'q\x002' + val(state) + b'b0h\x00)R.'
        self.check(payload, True, 'process')
        self.check(val(p) + val(state) + b'b.', False)
        # Build can also replace a dangerous function by an inert one.
        p = call(glob('functools', 'partial'), glob('os', 'system'), 'echo inert')
        self.check(val(p) + val((glob('builtins', 'str'), ('data',), {}, None)) + b'b)R.', False)

    def test_concatenated_and_legacy(self):
        benign = stream({'a': 1})
        evil = stream(call(glob('os', 'system'), 'echo inert'))
        self.check(benign + evil, True, 'process')
        self.check(benign + stream({'os.system': 'data'}), False)
        magic = pickle.dumps(119547037146038801333356, protocol=2)
        header = magic + pickle.dumps(1001, protocol=2) + pickle.dumps({}, protocol=2)
        self.check(header + evil + stream([]) + b'raw storage', True, 'process')
        self.check(header + benign + stream([]) + evil, False)

    def test_nested_pickle_loaders(self):
        good = stream({'f': glob('os', 'system'), 'data': 'eval('})
        bad = stream(call(glob('os', 'system'), 'echo inert'))
        for mod in ('pickle', '_pickle', 'cPickle'):
            self.check(stream(call(glob(mod, 'loads'), good)), False)
            self.check(stream(call(glob(mod, 'loads'), bad)), True, 'process')
            self.check(stream(call(glob(mod, 'loads'), good + bad)), False)
        fn = call(glob('pickle', 'loads'), stream(glob('os', 'system')))
        self.check(stream(call(fn, 'echo inert')), True, 'process')
        for payload, expected in ((good, False), (bad, True)):
            buffer = call(glob('io', 'BytesIO'), payload)
            self.check(stream(call(glob('pickle', 'load'), buffer)), expected, 'process' if expected else None)
            self.check(stream(call(attr(call(glob('pickle', 'Unpickler'), buffer), 'load'))), expected, 'process' if expected else None)
            encoded = call(glob('base64', 'b64decode'), base64.b64encode(payload))
            self.check(stream(call(glob('pickle', 'loads'), encoded)), expected, 'process' if expected else None)

    def test_network_constructors(self):
        for mod, cls, args in [('socket', 'socket', ()), ('http.client', 'HTTPConnection', ('invalid.test', 80)),
                               ('ftplib', 'FTP', ()), ('smtplib', 'SMTP', ()), ('xmlrpc.client', 'ServerProxy', ('http://invalid.test/',))]:
            obj = call(glob(mod, cls), *args)
            self.check(stream(obj), False)
            method = 'request' if cls == 'HTTPConnection' else 'connect'
            if cls != 'ServerProxy':
                self.check(stream(call(attr(obj, method), 'host')), True, 'network')
        for mod, cls in [('ftplib', 'FTP'), ('smtplib', 'SMTP')]:
            self.check(stream(call(glob(mod, cls), 'invalid.test')), True, 'network')

    def test_code_materialization(self):
        code = call(glob('marshal', 'loads'), b'inert representation')
        self.check(stream(code), False)
        self.check(stream(call(glob('types', 'FunctionType'), code, {})), True, 'code-execution')
        for mod, cls, method in [('code', 'InteractiveInterpreter', 'runsource'), ('bdb', 'Bdb', 'run'), ('pdb', 'Pdb', 'runeval')]:
            instance = call(glob(mod, cls))
            self.check(stream(instance), False)
            self.check(stream(call(attr(instance, method), '1+1')), True, 'code-execution')

    def test_logging_config(self):
        config = {'version': 1, 'handlers': {'h': {'class': 'logging.StreamHandler'}}, 'root': {'handlers': ['h']}}
        self.check(stream(call(glob('logging.config', 'dictConfig'), config)), False)
        config = {'version': 1, 'handlers': {'h': {'()': 'subprocess.Popen', 'args': ['echo', 'inert']}}}
        self.check(stream(call(glob('logging.config', 'dictConfig'), config)), True, 'process')

    def test_lazy_and_decorated_calls(self):
        callback = glob('os', 'system')
        lazy = call(glob('builtins', 'map'), callback, ['echo inert'])
        self.check(stream(lazy), False)
        self.check(stream(call(glob('builtins', 'tuple'), lazy)), True, 'process')
        chain = call(glob('itertools', 'chain'), lazy)
        self.check(stream(chain), False)
        self.check(stream(call(glob('builtins', 'tuple'), chain)), True, 'process')
        wrapped = call(call(glob('functools', 'lru_cache'), None), callback)
        self.check(stream(wrapped), False)
        self.check(stream(call(wrapped, 'echo inert')), True, 'process')
        dynamic = call(glob('builtins', 'type'), 'Fixture', (), {'__init__': callback})
        self.check(stream(dynamic), False)
        self.check(stream(call(dynamic, 'echo inert')), True, 'process')

    def test_memory_files_and_transforms(self):
        good = stream({'a': 1})
        bad = stream(call(glob('os', 'system'), 'echo inert'))
        for data, expected in [(good, False), (bad, True)]:
            hex_bytes = call(attr(glob('builtins', 'bytes'), 'fromhex'), data.hex())
            self.check(stream(call(glob('pickle', 'loads'), hex_bytes)), expected, 'process' if expected else None)
            z = call(glob('zipfile', 'ZipFile'), call(glob('io', 'BytesIO'), archive([('x.pkl', data)])))
            self.check(stream(z), False)
            self.check(stream(call(glob('pickle', 'loads'), call(attr(z, 'read'), 'x.pkl'))), expected, 'process' if expected else None)
            kwargs = {'fileobj': call(glob('io', 'BytesIO'), gzip.compress(data))}
            partial = Raw(val(glob('functools', 'partial')) + val((glob('gzip', 'GzipFile'),)) + b'R'
                          + val((glob('gzip', 'GzipFile'), (), kwargs, None)) + b'b')
            opened = call(partial)
            self.check(stream(opened), False)
            self.check(stream(call(glob('pickle', 'load'), opened)), expected, 'process' if expected else None)

    def test_joblib_inline_arrays(self):
        bad = stream(call(glob('os', 'system'), 'echo inert'))
        good = stream([1, 2, 3])
        for dtype, raw, object_dtype in [('i8', b'\x00' * 24, False), ('O8', good, True), ('O8', bad, True)]:
            state = {'shape': (3,), 'dtype': call(glob('numpy', 'dtype'), dtype),
                     'order': 'C', 'allow_mmap': True, 'numpy_array_alignment_bytes': 16}
            for proto in (0, 1, 2, 3, 4, 5):
                head = (b'\x80' + bytes([proto]) if proto >= 2 else b'')
                head += val(glob('joblib.numpy_pickle', 'NumpyArrayWrapper')) + b')\x81' + val(state) + b'b'
                padding = b'' if object_dtype else b'\x03\xff\xff\xff'
                self.check(head + padding + raw + b'.', raw == bad, 'process' if raw == bad else None, container=True)

    def test_content_containers_and_duplicates(self):
        good = stream({'hello': 'os.system'})
        bad = stream(call(glob('os', 'system'), 'echo inert'))
        for data, expected in [(good, False), (bad, True)]:
            transforms = [gzip.compress, bz2.compress, lzma.compress,
                          lambda d: archive([('model.unknown', d), ('notes.txt', b'just data')]),
                          lambda d: tar([('model.unknown', d)]),
                          lambda d: gzip.compress(tar([('model.dat', archive([('x.unknown', d)]))]))]
            for transform in transforms:
                self.check(transform(data), expected, 'process' if expected else None, container=True)
            self.check(archive([('model.bin', data)], prefix=b'arbitrary prefix'), expected, 'process' if expected else None, container=True)
        self.check(archive([('x.pkl', bad), ('x.pkl', good)]), False, container=True)
        self.check(archive([('x.pkl', good), ('x.pkl', bad)]), True, 'process', container=True)
        self.check(gzip.compress(good) + gzip.compress(bad), True, 'process', container=True)
        # A zip in an ordinary pickle byte string has no capability.
        self.check(stream(archive([('x.pkl', bad)])), False, container=True)
        self.check(stream(call(glob('os', 'system'), 'echo inert')) + archive([('x.pkl', good)]), False, container=True)

    def test_numpy_and_nested_container_loaders(self):
        good = stream([1, 2, 3])
        bad = stream(call(glob('os', 'system'), 'echo inert'))
        for v in (1, 2, 3):
            self.check(npy(good, version=v), False, container=True)
            self.check(npy(bad, version=v), True, 'process', container=True)
            self.check(npy(bad, dtype='<f8', version=v), False, container=True)
        self.check(npy(bad, dtype=[('x', '<i4'), ('object', '|O')]), True, 'process', container=True)
        self.check(archive([('x.npy', npy(bad))]), True, 'process', container=True)
        for loader in [('numpy', 'load'), ('numpy.lib.format', 'read_array'), ('joblib', 'load'), ('pandas', 'read_pickle'), ('torch', 'load')]:
            for payload, expected in [(good, False), (bad, True)]:
                if loader[0].startswith('numpy'):
                    payload = npy(payload)
                source = call(glob('io', 'BytesIO'), payload)
                args = (source, True) if loader[1] == 'read_array' else (source, None, True) if loader[0] == 'numpy' else (source,)
                self.check(stream(call(glob(*loader), *args)), expected, 'process' if expected else None)
            self.check(stream(call(glob(*loader), '/outside')), True, 'file-access')
        source = call(glob('io', 'BytesIO'), npy(bad))
        self.check(stream(call(glob('numpy', 'load'), source)), False)
        self.check(stream(call(glob('numpy.lib.format', 'read_array'), source)), False)
        for loader in [('joblib', 'load'), ('pandas', 'read_pickle'), ('torch', 'load')]:
            self.check(stream(call(glob(*loader), call(glob('io', 'BytesIO'), good + bad))), False)

    def test_torch_container(self):
        bad = stream(call(glob('os', 'system'), 'echo inert'))
        good = stream({'a': 1})
        self.check(archive([('weights/data.pkl', bad), ('weights/version', b'3'), ('weights/data/0', b'raw')]), True, 'process', container=True)
        self.check(archive([('weights/data.pkl', good), ('weights/version', b'3'), ('weights/data/0', bad)]), False, container=True)

    def test_keras_nested_layers(self):
        bad = {'class_name': 'Lambda', 'config': {'function': {'class_name': '__lambda__', 'config': {'code': 'base64'}}}}
        good = {'class_name': 'Lambda', 'config': {'function': {'module': 'builtins', 'class_name': 'function', 'config': 'relu'}}}
        for layer, expected in [(bad, True), (good, False)]:
            config = {'class_name': 'Functional', 'config': {'layers': [{'class_name': 'Bidirectional', 'config': {'layer': layer}}]}}
            self.assertEqual(bool(scan_config(config, 'fixture')), expected)
            self.check(archive([('config.json', json.dumps(config)), ('metadata.json', '{}'), ('model.weights.h5', b'weights')]), expected, 'code-execution' if expected else None, container=True)
        self.assertFalse(scan_config({'class_name': 'Activation', 'config': {'activation': '__lambda__', 'code': 'data'}}, 'fixture'))

    def test_tensorflow_all_graphs_and_function_bodies(self):
        from modelgate.tfproto.saved_model_pb2 import SavedModel
        from google.protobuf import text_format
        for op, expected_kind in list(TF_OPS.items()) + [(x, None) for x in ('SaveV2', 'RestoreV2', 'MergeV2Checkpoints', 'ShardedFilename', 'ShardedFilespec', 'NoOp', 'readfile')]:
            for where in ('graph', 'function'):
                model = SavedModel()
                model.meta_graphs.add()
                graph = model.meta_graphs.add().graph_def
                if where == 'function':
                    f = graph.library.function.add()
                    f.signature.name = 'called_from_attribute'
                    node = f.node_def.add()
                    caller = graph.node.add()
                    caller.op = 'PartitionedCall'
                    caller.attr['f'].func.name = f.signature.name
                else:
                    node = graph.node.add()
                node.name = 'ReadFile_PyFunc'
                node.op = op
                node.attr['text'].s = b'ReadFile PyFunc'
                for data, text in [(model.SerializeToString(), False), (text_format.MessageToString(model).encode(), True)]:
                    findings = scan_saved_model(data, 'fixture', text=text)
                    self.assertEqual(bool(findings), expected_kind is not None, findings)
                    if expected_kind: self.assertEqual(findings[0].kind, expected_kind)


if __name__ == '__main__':
    unittest.main()
