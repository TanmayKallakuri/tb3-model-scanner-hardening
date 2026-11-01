"""Policy regressions built from opcode bytes, never unpickled or executed.

Run with python3 -B -m unittest discover -s /app/modelgate/tests -v
"""
import base64
import bz2
import gzip
import io
import json
import lzma
import marshal
import os
import pathlib
import pickle
import struct
import sys
import tarfile
import tempfile
import unittest
import zipfile
import zlib

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from modelgate import dispatch, keras_scan, pickle_scan, savedmodel_scan
from modelgate.report import Result


class Global:
    def __init__(self, module, name): self.module, self.name = module, name


class Call:
    def __init__(self, func, *args): self.func, self.args = func, args


class Build:
    def __init__(self, obj, state): self.obj, self.state = obj, state


class New:
    def __init__(self, func, *args): self.func, self.args = func, args


class KwCall:
    def __init__(self, func, args, kw): self.func, self.args, self.kw = func, args, kw


def G(name):
    module, _, attr = name.rpartition(':')
    if not module: module, _, attr = name.rpartition('.')
    return Global(module, attr)


def emit(v):
    if isinstance(v, Global): return b'c' + v.module.encode() + b'\n' + v.name.encode() + b'\n'
    if isinstance(v, Call): return emit(v.func) + emit(v.args) + b'R'
    if isinstance(v, KwCall):
        return emit(G('functools.partial')) + emit((v.func,) + v.args) + b'R' + emit((v.func, v.args, v.kw, None)) + b'b)R'
    if isinstance(v, Build): return emit(v.obj) + emit(v.state) + b'b'
    if isinstance(v, New): return emit(v.func) + emit(v.args) + b'\x81'
    if isinstance(v, tuple): return b'(' + b''.join(map(emit, v)) + b't'
    if isinstance(v, list): return b'](' + b''.join(map(emit, v)) + b'e'
    if isinstance(v, dict): return b'}(' + b''.join(emit(k) + emit(x) for k, x in v.items()) + b'u'
    return pickle.dumps(v, protocol=4)[2:-1]  # frames have no semantic effect


def stream(v): return b'\x80\x04' + emit(v) + b'.'

def getter(obj, name): return Call(G('builtins.getattr'), obj, name)

def module(name): return Call(G('importlib.import_module'), name)

def method(obj, name, *args): return Call(getter(obj, name), *args)


def zipbytes(entries, prefix=b''):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED) as f:
        for name, data in entries: f.writestr(name, data)
    return prefix + out.getvalue()


def tarbytes(entries):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w') as f:
        for name, data in entries:
            info = tarfile.TarInfo(name); info.size = len(data)
            f.addfile(info, io.BytesIO(data))
    return out.getvalue()


def npy(payload, descr='|O'):
    header = repr({'descr': descr, 'fortran_order': False, 'shape': (1,)}) + '\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header.encode() + payload


class PicklePolicy(unittest.TestCase):
    def check(self, value, malicious, raw=False):
        b = value if raw else stream(value)
        findings = pickle_scan.scan_pickle(b, 'test')
        self.assertEqual(bool(findings), malicious, repr(value) + ': ' + repr(findings))
        return findings

    def test_literals_and_globals_are_data(self):
        for v in [None, {'os.system': 'eval('}, [G('os.system'), G('builtins.eval')],
                  G('subprocess.Popen'), G('socket.socket'), G('builtins.open'),
                  Call(G('functools.partial'), G('os.system'), 'echo never')]:
            with self.subTest(v=v): self.check(v, False)

    def test_protocols(self):
        for proto in range(6):
            # Pickler is only used for scanner-owned literals.
            self.check(pickle.dumps(['os.system', 'subprocess', {'eval(': 'exec('}], protocol=proto), False, True)
            payload = b'cos\nsystem\n(Vecho never\ntR.'
            self.check((bytes([128, proto]) if proto else b'') + payload, True, True)

    def test_seed_capabilities(self):
        targets = {
            'builtins.eval': ('1+1',), 'builtins.exec': ('pass',), 'builtins.compile': ('pass', 'x', 'exec'),
            'os.system': ('echo never',), 'posix.system': ('echo never',),
            'os.execve': ('/no', [], {}), 'os.spawnv': (0, '/no', []),
            'subprocess.Popen': (['never'],), 'pty.spawn': (['never'],),
            'pydoc.pipepager': ('never', 'never'), 'os.remove': ('/no',),
            'io.open': ('/no',), '_io.open': ('/no',), 'shutil.move': ('/no', '/no2'),
            'pathlib:Path.write_text': (New(G('pathlib.Path'), '/no'), 'never'),
            'socket.create_connection': (('example.invalid', 80),),
            'socket:socket.connect': (Call(G('socket.socket')), ('example.invalid', 80)),
            'urllib.request.urlopen': ('http://example.invalid',),
            'http.client:HTTPConnection.request': (Call(G('http.client.HTTPConnection'), 'example.invalid'), 'GET', '/'),
            'ftplib.FTP': ('example.invalid',), 'smtplib.SMTP': ('example.invalid',),
            'ssl.get_server_certificate': (('example.invalid', 443),),
            'webbrowser.open': ('http://example.invalid',),
            'runpy.run_path': ('/no',), 'timeit.timeit': ('pass',),
            'code:InteractiveInterpreter.runsource': (Call(G('code.InteractiveInterpreter')), 'pass'),
            'bdb:Bdb.run': (Call(G('bdb.Bdb')), 'pass'), 'pdb.run': ('pass',),
            'numpy.testing._private.utils.runstring': ('pass', {}),
        }
        for target, args in targets.items():
            with self.subTest(target=target): self.check(Call(G(target), *args), True)

    def test_clients_do_not_connect_at_construction(self):
        for v in [Call(G('socket.socket')), Call(G('http.client.HTTPConnection'), 'example.invalid'),
                  Call(G('ftplib.FTP')), Call(G('smtplib.SMTP')),
                  Call(G('xmlrpc.client.ServerProxy'), 'http://example.invalid')]:
            with self.subTest(v=v): self.check(v, False)
        self.check(method(Call(G('ftplib.FTP')), 'connect', 'example.invalid'), True)

    def test_construct_only_new(self):
        for target in ('subprocess.Popen', 'io.FileIO', 'smtplib.SMTP', 'ftplib.FTP'):
            self.check(New(G(target), 'example.invalid'), False)
        self.check(b'\x80\x04(csubprocess\nPopen\no.', False, True)
        self.check(b'\x80\x04(ios\nsystem\n.', False, True)  # no argument -> TypeError
        self.check(stream(New(G('builtins.eval'), 'pass')), False, True)
        self.check(b'cos\nsystem\n]R.', False, True)  # non-tuple REDUCE
        self.check(Call(G('timeit.Timer')), False)
        self.check(Call(G('timeit.timeit')), False)
        self.check(Call(G('timeit.timeit'), Call(G('functools.partial'), G('builtins.str'), 'data')), False)
        self.check(Call(G('timeit.timeit'), Call(G('functools.partial'), G('os.system'), 'pass')), True)

    def test_memo_and_stack_global(self):
        self.check(b'\x80\x04cos\nsystem\nq\x060h\x06(Vecho never\ntR.', True, True)
        self.check(b'\x80\x04\x8c\x02os\x94\x8c\x06system\x94h\x00h\x01\x93(Vecho never\ntR.', True, True)
        self.check(b'cos\nsystem\np1000\n0g1000\n(Vecho never\ntR.', True, True)

    def test_python2_aliases_and_protocol_dotted_names(self):
        for mod, name in [('__builtin__', 'eval'), ('commands', 'getoutput'), ('urllib2', 'urlopen')]:
            self.check(b'\x80\x02' + emit(Call(Global(mod, name), 'pass')) + b'.', True, True)
            self.check(b'\x80\x04' + emit(Call(Global(mod, name), 'pass')) + b'.', False, True)
        for proto in range(6):
            self.check(bytes([128, proto]) + emit(Call(Global('shutil', 'os.system'), 'pass')) + b'.', proto >= 4, True)

    def test_reachable_attributes(self):
        routes = [
            getter(module('os'), 'system'),
            Call(Call(G('operator.attrgetter'), 'os.system'), module('shutil')),
            Call(Call(G('operator.itemgetter'), 'system'), Call(G('builtins.vars'), module('os'))),
            method(getter(module('sys'), 'modules'), 'get', 'os'),
            Call(G('pkgutil.resolve_name'), 'os:system'),
            Call(G('pkgutil.resolve_name'), 'os.system'),
            Call(G('copy.copy'), G('os.system')),
            Call(G('functools.lru_cache'), G('os.system')),
        ]
        for i, fn in enumerate(routes):
            if i == 3: fn = getter(fn, 'system')
            with self.subTest(route=i): self.check(Call(fn, 'pass'), True)
        self.check(Call(Call(G('operator.methodcaller'), 'system', 'pass'), module('os')), True)

    def test_partial_build_retargets(self):
        safe = Call(G('functools.partial'), G('builtins.str'))
        poisoned = Build(safe, (G('os.system'), ('pass',), {}, None))
        self.check(poisoned, False)
        self.check(Call(poisoned), True)
        self.check(Call(Call(G('functools.partial'), G('builtins.getattr'), module('os'), 'system')), False)

    def test_module_mutation_shared_across_streams(self):
        patch = Build(module('os'), {'innocent': G('os.system')})
        self.check(stream(patch) + stream(Call(G('os.innocent'), 'pass')), True, True)
        self.check(stream(patch) + stream(G('os.innocent')), False, True)
        self.check(stream(Call(G('builtins.setattr'), module('os'), 'innocent', G('os.system'))) + stream(Call(G('os.innocent'), 'pass')), True, True)

    def test_lazy_and_eager_callbacks(self):
        for maker in ('builtins.map', 'itertools.starmap', 'builtins.filter'):
            items = [('pass',)] if maker.endswith('starmap') else ['pass']
            lazy = Call(G(maker), G('os.system'), items)
            self.check(lazy, False)
            self.check(Call(G('builtins.list'), lazy), True)
        self.check(Call(G('builtins.list'), Call(G('builtins.map'), G('os.system'), [])), False)
        for consumer in ('builtins.list', 'builtins.tuple', 'builtins.set', 'builtins.next', 'builtins.sum', 'builtins.any', 'builtins.all'):
            self.check(Call(G(consumer), Call(G('builtins.map'), G('os.system'), ['pass'])), True)
        for fn in ('builtins.sorted', 'builtins.min', 'builtins.max'):
            self.check(KwCall(G(fn), (['pass'],), {'key': G('os.system')}), True)
            self.check(KwCall(G(fn), ([],), {'key': G('os.system'), 'default': None}), False)
        self.check(Call(G('functools.reduce'), G('os.system'), ['pass']), False)
        self.check(Call(G('functools.reduce'), G('builtins.eval'), ['1+1', {}]), True)
        self.check(Call(G('operator.call'), G('os.system'), 'pass'), True)

    def test_thread_callbacks(self):
        thread = KwCall(G('threading.Thread'), (), {'target': G('os.system'), 'args': ('pass',)})
        self.check(thread, False)
        self.check(method(thread, 'start'), True)
        self.check(Call(G('threading:Thread.start'), thread), True)
        self.check(method(Call(G('concurrent.futures.ThreadPoolExecutor')), 'submit', G('os.system'), 'pass'), True)
        self.check(Call(G('concurrent.futures:ThreadPoolExecutor.submit'), Call(G('concurrent.futures.ThreadPoolExecutor')), G('os.system'), 'pass'), True)
        self.check(Call(G('concurrent.futures.ProcessPoolExecutor')), False)
        self.check(method(Call(G('concurrent.futures.ProcessPoolExecutor')), 'submit', G('builtins.str'), 'data'), True)

    def test_nested_pickle_one_stream(self):
        benign = stream({'os.system': G('os.system')})
        bad = stream(Call(G('os.system'), 'pass'))
        for loader in ('pickle.loads', '_pickle.loads'):
            self.check(Call(G(loader), benign), False)
            self.check(Call(G(loader), bad), True)
            self.check(Call(G(loader), benign + bad), False)
            self.check(Call(G(loader), bad + benign), True)
        self.check(Call(G('pickle.load'), Call(G('io.BytesIO'), bad)), True)
        self.check(Call(Call(G('pickle.loads'), stream(G('os.system'))), 'pass'), True)
        self.check([Call(G('pickle.loads'), b'invalid'), Call(G('os.system'), 'pass')], False)
        for loader in ('joblib.load', 'pandas.read_pickle'):
            self.check(Call(G(loader), Call(G('io.BytesIO'), benign + bad)), False)

    def test_decoded_nested_data(self):
        bad = stream(Call(G('os.system'), 'pass'))
        encoded = [
            Call(G('base64.b64decode'), base64.b64encode(bad)),
            Call(G('zlib.decompress'), zlib.compress(bad)),
            Call(getter(G('builtins.bytes'), 'fromhex'), bad.hex()),
            Call(G('builtins.bytearray'), list(bad)),
            Call(G('operator.add'), bad[:5], bad[5:]),
            Call(G('_codecs.encode'), bad.decode('latin1'), 'latin1'),
        ]
        for value in encoded:
            with self.subTest(value=value): self.check(Call(G('pickle.loads'), value), True)
        self.check(Call(G('pickle.loads'), Call(G('base64.b64decode'), base64.b64encode(stream({'safe': 'os.system'})))), False)

    def test_nested_loader_paths_and_buffers(self):
        bad = stream(Call(G('os.system'), 'pass'))
        for loader in ('joblib.load', 'pandas.read_pickle', 'torch.load', 'numpy.load'):
            self.check(Call(G(loader), '/outside'), True)
        for loader in ('joblib.load', 'pandas.read_pickle'):
            self.check(Call(G(loader), Call(G('io.BytesIO'), stream([1, 2]))), False)
            self.check(Call(G(loader), Call(G('io.BytesIO'), bad)), True)
        self.check(KwCall(G('numpy.load'), (Call(G('io.BytesIO'), npy(bad)),), {'allow_pickle': True}), True)
        self.check(Call(G('numpy.load'), Call(G('io.BytesIO'), npy(bad))), False)
        self.check([Call(G('numpy.load'), Call(G('io.BytesIO'), npy(bad))), Call(G('os.system'), 'pass')], False)
        self.check(KwCall(G('numpy.load'), (Call(G('io.BytesIO'), stream([1]) + bad),), {'allow_pickle': True}), False)
        npz = KwCall(G('numpy.load'), (Call(G('io.BytesIO'), zipbytes([('a.npy', npy(bad))])),), {'allow_pickle': True})
        self.check(npz, False)
        self.check(Call(G('operator.getitem'), npz, 'a'), True)
        checkpoint = b''.join(stream(v) for v in (pickle_scan.TORCH_MAGIC, 1001, {}, Call(G('os.system'), 'pass'), [])) + b'raw storage'
        self.check(Call(G('torch.storage._load_from_bytes'), checkpoint), True)
        clean_checkpoint = b''.join(stream(v) for v in (pickle_scan.TORCH_MAGIC, 1001, {}, {}, [])) + bad
        self.check(Call(G('torch.storage._load_from_bytes'), clean_checkpoint), False)
        for fn, verdict in [('builtins.str', False), ('os.system', True)]:
            checkpoint_zip = zipbytes([('x/data.pkl', stream(G(fn))), ('x/version', b'3')])
            self.check(Call(Call(G('torch.load'), Call(G('io.BytesIO'), checkpoint_zip)), 'pass'), verdict)

    def test_code_materialization(self):
        code = Call(G('marshal.loads'), marshal.dumps(compile('pass', 'controlled-test', 'exec')))
        self.check(code, False)
        self.check(Call(G('types.FunctionType'), code, {}), True)
        self.check(Call(G('types.FunctionType'), Call(G('marshal.loads'), b'invalid'), {}), False)
        self.check(Call(G('types.FunctionType'), Call(G('marshal.loads'), marshal.dumps('data')), {}), False)

    def test_namespace_copies_and_descriptors(self):
        for namespace in [Call(G('builtins.dict'), Call(G('builtins.vars'), module('os'))),
                          method(Call(G('builtins.vars'), module('os')), 'copy')]:
            self.check(Call(method(namespace, 'get', 'system'), 'pass'), True)
        self.check(Call(Call(G('pkgutil.resolve_name'), 'shutil.os.system'), 'pass'), True)
        self.check(Call(Call(G('types.MethodType'), G('builtins.eval'), '1+1'), {}), True)
        self.check(Call(New(G('operator.attrgetter'), 'system'), module('os')), False)
        self.check(Call(Call(New(G('operator.attrgetter'), 'system'), module('os')), 'pass'), True)
        self.check(Call(G('os.system'), 'pass', 'too many'), False)
        self.check(KwCall(G('os.system'), (), {'command': 'pass'}), False)
        self.check(Call(G('builtins.eval'), {'only': 'data'}), False)
        self.check(Call(Call(G('builtins.getattr'), module('os'), 'absent_attribute', G('os.system')), 'pass'), True)
        self.check(Call(Call(G('builtins.getattr'), module('os'), 'fspath', G('os.system')), 'pass'), False)
        self.check(Call(getter(module('os'), Call(G('posixpath.basename'), '/prefix/system')), 'pass'), True)

    def test_additional_iterators(self):
        for fn in ('itertools.groupby',):
            self.check(Call(G(fn), ['pass'], G('os.system')), False)
            self.check(Call(G('builtins.next'), Call(G(fn), ['pass'], G('os.system'))), True)
            self.check(Call(G('builtins.list'), Call(G(fn), ['a', 'a', 'b'], G('builtins.str'))), False)
        self.check(Call(G('builtins.list'), Call(G('itertools.accumulate'), ['1+1', {}], G('builtins.eval'))), True)
        self.check(Call(G('builtins.next'), Call(G('itertools.accumulate'), ['1+1', {}], G('builtins.eval'))), False)

    def test_memory_wrappers_are_not_file_access(self):
        good, bad = stream([1, 2]), stream(Call(G('os.system'), 'pass'))
        for value, verdict in [(good, False), (bad, True)]:
            for wrapper in [lambda b: Call(G('io.BufferedReader'), Call(G('io.BytesIO'), b)),
                            lambda b: KwCall(G('gzip.GzipFile'), (), {'fileobj': Call(G('io.BytesIO'), gzip.compress(b))}),
                            lambda b: method(Call(G('zipfile.ZipFile'), Call(G('io.BytesIO'), zipbytes([('a', b)]))), 'open', 'a')]:
                self.check(Call(G('pickle.load'), wrapper(value)), verdict)

    def test_joblib_inline_arrays(self):
        bad = stream(Call(G('os.system'), 'pass'))
        wrapper = New(G('joblib.numpy_pickle.NumpyArrayWrapper'))
        state = {'shape': (len(bad),), 'dtype': Call(G('numpy.dtype'), 'u1'),
                 'numpy_array_alignment_bytes': 16, 'allow_mmap': True}
        # Raw numeric bytes may look exactly like a pickle. They are not
        # opcodes; only code after the array is interpreted.
        prefix = b'\x80\x04]' + emit(Build(wrapper, state)) + b'\x03xxx' + bad + b'a'
        self.check(prefix + b'.', False, True)
        self.check(prefix + emit(Call(G('os.system'), 'pass')) + b'a.', True, True)


class FormatPolicy(unittest.TestCase):
    def check(self, data, malicious):
        findings = dispatch.scan_bytes(data, 'artifact')
        self.assertEqual(bool(findings), malicious, repr(findings))

    def test_content_and_containers(self):
        bad = stream(Call(G('os.system'), 'pass'))
        good = stream({'os.system': 'subprocess'})
        wrappers = [lambda b: b, gzip.compress, bz2.compress, lzma.compress,
                    zlib.compress,
                    lambda b: zipbytes([('model.unknown', b)]),
                    lambda b: zipbytes([('model.pkl', b)], prefix=b'JUNK-STUB'),
                    lambda b: tarbytes([('model.unknown', b)]),
                    lambda b: gzip.compress(tarbytes([('a', zipbytes([('b', b)]))]))]
        for wrapper in wrappers:
            self.check(wrapper(bad), True)
            self.check(wrapper(good), False)
        self.check(gzip.compress(good) + gzip.compress(bad), True)
        self.check(zipbytes([('same.pkl', bad), ('same.pkl', good)]), False)
        self.check(zipbytes([('same.pkl', good), ('same.pkl', bad)]), True)
        self.check(zipbytes([('a', bad)], prefix=stream(None)), True)
        self.check(zipbytes([('a', good)], prefix=bad), False)

    def test_torch_storages_are_not_scanned(self):
        bad = stream(Call(G('os.system'), 'pass'))
        for value, expected in [(bad, True), (stream([1, 2]), False)]:
            self.check(zipbytes([('x/data.pkl', value), ('x/version', b'3'), ('x/data/0', bad)]), expected)
        checkpoint = b''.join(stream(v) for v in (pickle_scan.TORCH_MAGIC, 1001, {}, {}, [])) + bad
        self.check(checkpoint, False)
        self.check(stream({}) + bad, True)

    def test_numpy_object_and_numeric(self):
        bad = stream(Call(G('os.system'), 'pass'))
        self.check(npy(bad), True)
        self.check(npy(stream(['os.system', 'eval('])), False)
        self.check(npy(bad, '<f8'), False)
        self.check(npy(bad, [('x', '<f8'), ('obj', '|O')]), True)
        self.check(zipbytes([('a.npy', npy(stream([1]))), ('b.npy', npy(bad))]), True)

    def test_savedmodel_table_and_nesting(self):
        from modelgate.tfproto import saved_model_pb2
        from modelgate.policy import TF_OPS
        from google.protobuf import text_format
        for op in list(TF_OPS) + ['SaveV2', 'RestoreV2', 'MergeV2Checkpoints', 'ShardedFilename', 'ShardedFilespec', 'Const', 'readfile']:
            sm = saved_model_pb2.SavedModel()
            sm.meta_graphs.add()
            graph = sm.meta_graphs.add().graph_def
            fun = graph.library.function.add(); fun.signature.name = 'branch'
            n = fun.node_def.add(); n.name = 'ReadFile'; n.op = op
            n.attr['data'].s = b'PyFunc os.system'
            self.assertEqual(bool(savedmodel_scan.scan_saved_model(sm.SerializeToString(), 'pb')), op in TF_OPS, op)
            self.assertEqual(bool(savedmodel_scan.scan_saved_model(text_format.MessageToString(sm).encode(), 'txt', text=True)), op in TF_OPS, op)

    def test_keras_nested_fields(self):
        bytecode = {'class_name': '__lambda__', 'config': {'code': 'ZmFrZQ==', 'defaults': None, 'closure': None}}
        activation = {'class_name': '__lambda__', 'config': {'value': ['ZmFrZQ==', None, None]}}
        named = {'module': 'builtins', 'class_name': 'function', 'config': 'relu'}
        for field in ('function', 'output_shape'):
            for fn, malicious in [(bytecode, True), (named, False)]:
                layer = {'class_name': 'Lambda', 'config': {field: fn}}
                for wrap in [lambda v: v, lambda v: {'config': {'layers': [v]}},
                             lambda v: {'class_name': 'TimeDistributed', 'config': {'layer': v}},
                             lambda v: {'config': {'layers': [{'class_name': 'Model', 'config': {'layers': [v]}}]}}]:
                    self.assertEqual(bool(keras_scan.scan_config(wrap(layer), 'json')), malicious)
        self.assertTrue(keras_scan.scan_config({'class_name': 'Dense', 'config': {'activation': activation}}, 'json'))
        self.assertFalse(keras_scan.scan_config({'class_name': 'Dense', 'config': {'activation': 'relu'}}, 'json'))
        self.assertFalse(keras_scan.scan_config({'description': bytecode}, 'json'))


if __name__ == '__main__': unittest.main()
