"""Policy regressions. Fixtures are assembled, never deserialized or run.

Run: python3 -B -m unittest discover -s modelgate/tests -v
"""
from __future__ import annotations

import base64
import bz2
import gzip
import io
import json
import lzma
import os
import pickle
import pickletools
import struct
import sys
import tarfile
import tempfile
import unittest
import warnings
import zipfile
import zlib

sys.dont_write_bytecode = True
from modelgate import dispatch, keras_scan, pickle_scan, savedmodel_scan
from modelgate.analysis import Budget
from modelgate.policy import CALLS, TF_OPS


class Expr:
    def __init__(self, data):
        self.data = data

    def __call__(self, *args):
        return Expr(self.data + enc(tuple(args)) + b'R')


def enc(value):
    if isinstance(value, Expr):
        return value.data
    if isinstance(value, tuple):
        return b'(' + b''.join(enc(x) for x in value) + b't'
    if isinstance(value, list):
        return b'](' + b''.join(enc(x) for x in value) + b'e'
    if isinstance(value, dict):
        return b'}(' + b''.join(enc(k) + enc(v) for k, v in value.items()) + b'u'
    # Literal pickle fragments have no memo, so composing them is safe.
    return pickletools.optimize(pickle.dumps(value, protocol=2))[2:-1]


def g(module, name):
    return Expr(b'c' + module.encode() + b'\n' + name.encode() + b'\n')


def stream(value, protocol=4):
    return b'\x80' + bytes([protocol]) + enc(value) + b'.'


def get(obj, name):
    return g('builtins', 'getattr')(obj, name)


def imported(name):
    return g('importlib', 'import_module')(name)


def zip_bytes(entries, prefix=b''):
    out = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(out, 'w') as zf:
            for name, data in entries:
                zf.writestr(name, data)
    return prefix + out.getvalue()


def tar_bytes(entries):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w') as tf:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return out.getvalue()


def npy(payload, descr='|O', version=1):
    header = repr({'descr': descr, 'fortran_order': False, 'shape': (1,)}).encode() + b'\n'
    return b'\x93NUMPY' + bytes([version, 0]) + struct.pack('<H' if version == 1 else '<I', len(header)) + header + payload


BAD = g('os', 'system')('never executed')
GOOD = {'text': 'os.system subprocess eval( exec(', 'callback': g('os', 'system')}


class PickleTests(unittest.TestCase):
    def verdict(self, value, malicious, protocol=4):
        data = value if isinstance(value, bytes) else stream(value, protocol)
        findings = pickle_scan.scan_pickle(data, 'fixture')
        self.assertEqual(bool(findings), malicious, findings)
        if malicious:
            self.assertTrue(any(f.kind != 'unsupported' for f in findings), findings)
        return findings

    def test_data_is_not_execution(self):
        for module, name in [('os', 'system'), ('builtins', 'eval'), ('socket', 'create_connection'),
                             ('pickle', 'loads'), ('functools', 'partial')]:
            with self.subTest(module=module, name=name):
                self.verdict({'f': g(module, name), 's': f'{module}.{name}'}, False)
        self.verdict(GOOD, False)

    def test_protocols_and_memo(self):
        for protocol in range(6):
            with self.subTest(protocol=protocol):
                self.verdict(stream(BAD, protocol), True)
                self.verdict(stream(GOOD, protocol), False)
        for put, getcode in [(b'p3\n', b'g3\n'), (b'q\x03', b'h\x03'),
                             (b'r\x03\x01\0\0', b'j\x03\x01\0\0')]:
            data = b'\x80\x02' + enc(g('os', 'system')) + put + b'0' + getcode + enc(('x',)) + b'R.'
            self.verdict(data, True)
        # STACK_GLOBAL takes its strings from the stack, even if memoized
        # long before and interspersed with unrelated values.
        data = b'\x80\x04' + enc('os') + b'q\x000' + enc('system') + b'q\x010h\x00h\x01\x93' + enc(('x',)) + b'R.'
        self.verdict(data, True)

    def test_aliases(self):
        for module, name in [('posix', 'system'), ('nt', 'system'), ('__builtin__', 'eval'),
                             ('commands', 'getoutput'), ('urllib2', 'urlopen'), ('httplib', 'HTTPConnection.request'),
                             ('_io', 'open'), ('urllib', 'urlopen')]:
            with self.subTest(module=module):
                self.verdict(g(module, name)('x'), True, protocol=2)
                self.verdict(g(module, name), False, protocol=2)

    def test_inert_construction(self):
        for ref in [g('subprocess', 'Popen'), g('builtins', 'eval'), g('socket', 'create_connection')]:
            self.verdict(Expr(enc(ref) + enc(('x',)) + b'\x81'), False)
            self.verdict(Expr(enc(ref) + enc(('x',)) + enc({}) + b'\x92'), False)
        self.verdict(b'(ios\nsystem\n.', False)
        self.verdict(b'(cos\nsystem\no.', False)
        self.verdict(b'(Vx\nios\nsystem\n.', True)
        self.verdict(b'(cos\nsystem\nVx\no.', True)
        self.verdict(b'\x80\x02' + enc(('storage', 'os.system')) + b'Q.', False)
        self.verdict(b'Poutside storage\n.', False)

    def test_indirect_calls(self):
        variants = [get(imported('os'), 'system')('x'),
                    get(g('builtins', '__import__')('posix'), 'system')('x'),
                    g('operator', 'attrgetter')('system')(imported('os'))('x'),
                    g('operator', 'attrgetter')('pathlib.Path.write_text')(imported('builtins'))('x'),
                    g('operator', 'methodcaller')('system', 'x')(imported('os')),
                    g('functools', 'partial')(g('os', 'system'), 'x')(),
                    g('operator', 'itemgetter')('callback')({'callback': g('os', 'system')})('x'),
                    get(get(imported('builtins'), '__dict__'), 'get')('eval')('1+2'),
                    get(get(imported('os'), 'system'), '__call__')('x')]
        for i, expr in enumerate(variants):
            if i == 3:  # not a real Python path; test valid dotted path separately
                continue
            with self.subTest(i=i):
                self.verdict(expr, True)
        self.verdict(g('operator', 'attrgetter')('pathlib.Path.write_text'), False)
        self.verdict(g('operator', 'methodcaller')('system', 'x'), False)
        self.verdict(g('functools', 'partial')(g('os', 'system'), 'x'), False)
        self.verdict(g('builtins', '__import__')('os'), False)
        self.verdict(get(imported('os'), 'system'), False)

    def test_partial_retargeting_shared_memo(self):
        initial = enc(g('functools', 'partial')(g('builtins', 'str')))
        state = enc((g('os', 'system'), ('x',), {}, {}))
        # The same object is memoized before BUILD and retrieved afterwards.
        self.verdict(b'\x80\x04' + initial + b'q\x00' + state + b'b0h\x00)R.', True)
        self.verdict(b'\x80\x04' + initial + b'q\x00' + state + b'b.', False)
        # Retargeting a dangerous partial to an inert callable is clean.
        initial = enc(g('functools', 'partial')(g('os', 'system')))
        state = enc((g('builtins', 'str'), ('x',), {}, {}))
        self.verdict(b'\x80\x04' + initial + state + b'b)R.', False)

    def test_concatenated_and_legacy_streams(self):
        self.verdict(stream(GOOD) + stream(BAD), True)
        self.verdict(stream(GOOD) + stream(GOOD), False)
        prefix = stream(pickle_scan.TORCH_MAGIC, 2) + stream(1001, 2) + stream({}, 2)
        self.verdict(prefix + stream(BAD, 2) + stream([], 2) + b'raw bytes', True)
        self.verdict(prefix + stream(GOOD, 2) + stream([], 2) + stream(BAD), False)

    def test_nested_loaders(self):
        for module, name, wrap in [('pickle', 'loads', False), ('_pickle', 'loads', False),
                                   ('pickle', 'load', True), ('torch', 'load', True),
                                   ('joblib', 'load', True), ('pandas', 'read_pickle', True),
                                   ('torch.storage', '_load_from_bytes', False)]:
            for bad in (False, True):
                with self.subTest(loader=(module, name), bad=bad):
                    data = stream(BAD if bad else GOOD)
                    source = g('io', 'BytesIO')(data) if wrap else data
                    self.verdict(g(module, name)(source), bad)
        # The result of a nested load must preserve callable identity.
        self.verdict(g('pickle', 'loads')(stream(g('os', 'system')))('x'), True)
        # pickle.loads consumes one stream, not unrelated trailing bytes.
        self.verdict(g('pickle', 'loads')(stream(GOOD) + stream(BAD)), False)
        for module, name in [('torch', 'load'), ('joblib', 'load'), ('numpy', 'load'), ('pandas', 'read_pickle')]:
            self.verdict(g(module, name)('/outside/model'), True)
        self.verdict(g('pickle', 'loads')(g('base64', 'b64decode')(base64.b64encode(stream(BAD)))), True)
        self.verdict(g('pickle', 'loads')(g('_codecs', 'encode')(stream(BAD).decode('latin1'), 'latin1')), True)

    def test_network_connection_not_constructor(self):
        for module, name, args in [('socket', 'socket', ()), ('http.client', 'HTTPConnection', ('host', 80)),
                                   ('http.client', 'HTTPSConnection', ('host',)), ('ftplib', 'FTP', ()),
                                   ('smtplib', 'SMTP', ()), ('xmlrpc.client', 'ServerProxy', ('http://host',))]:
            with self.subTest(module=module, name=name):
                obj = g(module, name)(*args)
                self.verdict(obj, False)
                self.verdict(get(obj, 'connect')('host'), True)
        for module, name in [('ftplib', 'FTP'), ('smtplib', 'SMTP')]:
            self.verdict(g(module, name)('host'), True)
        self.verdict(get(g('http.client', 'HTTPConnection')('host'), 'request')('GET', '/'), True)

    def test_methods_and_file_access(self):
        self.verdict(get(g('pathlib', 'Path')('/outside'), 'write_text')('hello'), True)
        self.verdict(get(g('code', 'InteractiveInterpreter')(), 'runsource')('pass'), True)
        self.verdict(get(g('bdb', 'Bdb')(), 'run')('pass'), True)
        self.verdict(g('operator', 'methodcaller')('runsource', 'pass')(g('code', 'InteractiveInterpreter')()), True)
        self.verdict(g('pathlib', 'Path')('/outside'), False)
        self.verdict(get(g('io', 'BytesIO')(b'text'), 'read')(), False)
        self.verdict(get(g('io', 'BytesIO')(), 'write')(b'text'), False)
        self.verdict(g('types', 'FunctionType')(g('marshal', 'loads')(b'not executed'), {}), True)
        self.verdict(g('marshal', 'loads')(b'not executed'), False)

    def test_logging(self):
        self.verdict(g('logging.config', 'dictConfig')({'version': 1, 'handlers': {'h': {'class': 'logging.StreamHandler'}}}), False)
        self.verdict(g('logging.config', 'dictConfig')({'version': 1, 'handlers': {'h': {'class': 'logging.FileHandler', 'filename': '/outside'}}}), True)
        self.verdict(g('logging.config', 'dictConfig')({'version': 1, 'filters': {'f': {'()': 'os.system', 'command': 'x'}}}), True)


class FormatTests(unittest.TestCase):
    def check(self, data, malicious):
        findings, _ = dispatch.analyze_bytes(data, 'misleading.extension')
        self.assertEqual(bool(findings), malicious, findings)
        if malicious:
            self.assertTrue(any(f.kind != 'unsupported' for f in findings), findings)

    def test_compression_and_nesting(self):
        for compress in (gzip.compress, bz2.compress, lzma.compress, zlib.compress):
            for bad in (False, True):
                with self.subTest(compress=compress.__name__, bad=bad):
                    self.check(compress(stream(BAD if bad else GOOD)), bad)
                    self.check(compress(zip_bytes([('model.any', stream(BAD if bad else GOOD)), ('card.json', b'{}')])), bad)
        self.check(gzip.compress(stream(GOOD)) + gzip.compress(stream(BAD)), True)
        self.check(gzip.compress(stream(GOOD)) + gzip.compress(stream(GOOD)), False)
        self.check(lzma.compress(tar_bytes([('inner', zip_bytes([('model', stream(BAD))]))])), True)

    def test_archive_precedence(self):
        for prefix in (b'', b'#!/bin/sh\n', stream(BAD)):
            self.check(zip_bytes([('same.pkl', stream(BAD)), ('same.pkl', stream(GOOD))], prefix), False)
            self.check(zip_bytes([('same.pkl', stream(GOOD)), ('same.pkl', stream(BAD))], prefix), True)
        self.check(tar_bytes([('same.pkl', stream(BAD)), ('same.pkl', stream(GOOD))]), False)
        self.check(tar_bytes([('ckpt/byteorder', b'little'), ('ckpt/.format_version', b'1'),
                              ('ckpt/.data/serialization_id', b'012345678'), ('ckpt/data.pkl', stream(GOOD))]), False)

    def test_torch_zip(self):
        self.check(zip_bytes([('a/data.pkl', stream(BAD)), ('a/version', b'3\n'), ('a/data/0', b'raw')]), True)
        self.check(zip_bytes([('a/data.pkl', stream(GOOD)), ('a/version', b'3\n'), ('a/data/0', stream(BAD))]), False)

    def test_numpy(self):
        for version in (1, 2, 3):
            for bad in (False, True):
                self.check(npy(stream(BAD if bad else GOOD), version=version), bad)
                self.check(npy(stream(BAD if bad else GOOD), descr=[('value', '|O')], version=version), bad)
        self.check(npy(stream(BAD), descr='<f8'), False)
        self.check(zip_bytes([('a.npy', npy(stream(GOOD))), ('b.npy', npy(stream(BAD)))]), True)
        self.check(zip_bytes([('a.npy', npy(stream(GOOD))), ('b.npy', npy(stream(GOOD)))]), False)
        for name in [('numpy', 'load'), ('numpy.lib.format', 'read_array')]:
            expr = g(*name)(g('io', 'BytesIO')(npy(stream(BAD))))
            self.check(stream(expr), True)

    def test_keras(self):
        bad = {'class_name': 'Lambda', 'config': {'function': {'class_name': '__lambda__', 'config': {'code': 'bytecode'}}}}
        good = {'class_name': 'Lambda', 'config': {'function': {'module': 'builtins', 'class_name': 'function', 'config': 'relu'}}}
        for layer, expected in ((bad, True), (good, False), ({'class_name': 'Activation', 'config': {'activation': 'relu'}}, False)):
            config = {'class_name': 'Functional', 'config': {'layers': [{'class_name': 'TimeDistributed', 'config': {'layer': {'class_name': 'Sequential', 'config': {'layers': [layer]}}}}]}}
            self.assertEqual(bool(keras_scan.scan_config(config, 'config')), expected)
            self.check(zip_bytes([('config.json', json.dumps(config).encode()), ('metadata.json', b'{}'), ('model.weights.h5', b'')]), expected)

    def test_savedmodel_closed_ops(self):
        from modelgate.tfproto import saved_model_pb2
        from google.protobuf import text_format
        for op in list(TF_OPS) + ['SaveV2', 'RestoreV2', 'MergeV2Checkpoints', 'ShardedFilename', 'ShardedFilespec', 'Identity', 'readfile']:
            for group in ('graph', 'function'):
                sm = saved_model_pb2.SavedModel()
                sm.meta_graphs.add()
                graph = sm.meta_graphs.add().graph_def
                nodes = graph.node if group == 'graph' else graph.library.function.add().node_def
                nodes.add(name='PyFunc_ReadFile', op=op)
                with self.subTest(op=op, group=group):
                    self.assertEqual(bool(savedmodel_scan.scan_saved_model(sm.SerializeToString(), 'saved_model.pb')), op in TF_OPS)
                    self.assertEqual(bool(savedmodel_scan.scan_saved_model(text_format.MessageToString(sm).encode(), 'saved_model.pbtxt', text=True)), op in TF_OPS)
        sm = saved_model_pb2.SavedModel()
        node = sm.meta_graphs.add().graph_def.node.add(name='ReadFile', op='Const', device='PyFunc')
        node.attr['value'].tensor.string_val.append(b'PyFunc')
        self.assertFalse(savedmodel_scan.scan_saved_model(sm.SerializeToString(), 'saved_model.pb'))


if __name__ == '__main__':
    unittest.main()
