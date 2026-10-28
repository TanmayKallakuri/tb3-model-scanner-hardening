"""Container, graph, layer, and scanner-boundary regressions."""
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from test_semantics import archive, stream, call, glob, lit, Expr, npy, findings, partial, method, module, attr
from modelgate import keras_scan, savedmodel_scan
from modelgate.tfproto import saved_model_pb2


class Graphs(unittest.TestCase):
    def test_closed_table_in_every_scope(self):
        dangerous = '''PyFunc PyFuncStateless EagerPyFunc ReadFile WriteFile MatchingFiles
            MatchingFilesDataset ImmutableConst WholeFileReader WholeFileReaderV2 TextLineReader
            TextLineReaderV2 FixedLengthRecordReader FixedLengthRecordReaderV2 TFRecordReader
            TFRecordReaderV2 LMDBReader LMDBDataset ExperimentalLMDBDataset TextLineDataset
            TFRecordDataset FixedLengthRecordDataset DatasetToTFRecord ExperimentalDatasetToTFRecord
            SaveDataset SaveDatasetV2 LoadDataset SnapshotDataset SnapshotDatasetV2 LoadAndRemapMatrix
            GenerateVocabRemapping Save SaveSlices Restore RestoreSlice DataServiceDataset
            DataServiceDatasetV2 DataServiceDatasetV3 DataServiceDatasetV4 RegisterDataset
            RegisterDatasetV2 DistributedSave'''.split()
        safe = '''SaveV2 RestoreV2 MergeV2Checkpoints ShardedFilename ShardedFilespec Const Identity
            StatefulPartitionedCall PartitionedCall If StatelessIf While StatelessWhile Case MapDefun For
            readfile Readfile ReadFileV2 SomePyFunc'''.split()
        for op in dangerous + safe:
            for scope in ('graph', 'function', 'second_meta_graph'):
                with self.subTest(op=op, scope=scope):
                    sm = saved_model_pb2.SavedModel()
                    meta = sm.meta_graphs.add()
                    if scope == 'second_meta_graph':
                        meta = sm.meta_graphs.add()
                    if scope == 'function':
                        fn = meta.graph_def.library.function.add()
                        fn.signature.name = 'hidden_function'
                        node = fn.node_def.add()
                        root = meta.graph_def.node.add()
                        root.name, root.op = 'caller', 'StatelessIf'
                        root.attr['then_branch'].func.name = 'hidden_function'
                    else:
                        node = meta.graph_def.node.add()
                    node.name, node.op = 'ordinary_name', op
                    result = savedmodel_scan.scan_saved_model(sm.SerializeToString(), 'model.pb')
                    self.assertEqual(bool(result), op in dangerous)

    def test_strings_and_pbtxt(self):
        sm = saved_model_pb2.SavedModel()
        node = sm.meta_graphs.add().graph_def.node.add()
        node.name, node.op, node.device = 'ReadFile', 'Const', 'PyFunc'
        node.attr['value'].tensor.string_val.append(b'PyFunc WriteFile')
        self.assertFalse(savedmodel_scan.scan_saved_model(sm.SerializeToString(), 'model.pb'))
        from google.protobuf import text_format
        text = text_format.MessageToString(sm).encode()
        self.assertFalse(savedmodel_scan.scan_saved_model(text, 'model.pbtxt', text=True))
        node.op = 'PyFunc'
        text = text_format.MessageToString(sm).encode()
        self.assertTrue(savedmodel_scan.scan_saved_model(text, 'model.pbtxt', text=True))


class Keras(unittest.TestCase):
    def test_layers(self):
        bytecode = {'class_name': '__lambda__', 'config': {'code': 'serialized', 'defaults': None, 'closure': None}}
        activation = {'class_name': '__lambda__', 'config': {'value': ['serialized', None, None]}}
        named = {'module': 'builtins', 'class_name': 'function', 'config': 'relu'}
        cases = [({'class_name': 'Lambda', 'config': {'function': bytecode}}, True),
                 ({'class_name': 'Lambda', 'config': {'function': named}}, False),
                 ({'class_name': 'Lambda', 'config': {'function': named, 'output_shape': bytecode}}, True),
                 ({'class_name': 'Lambda', 'config': {'function': named, 'output_shape': [3, 4]}}, False),
                 ({'class_name': 'Dense', 'config': {'activation': activation}}, True),
                 ({'class_name': 'Dense', 'config': {'activation': 'relu'}}, False),
                 ({'class_name': 'Dense', 'config': {'name': '__lambda__', 'activation': named}}, False)]
        for layer, expected in cases:
            wrappers = [layer, {'class_name': 'TimeDistributed', 'config': {'layer': layer}},
                        {'class_name': 'Bidirectional', 'config': {'layer': layer}},
                        {'class_name': 'Functional', 'config': {'layers': [layer]}}]
            for wrapped in wrappers:
                config = {'class_name': 'Sequential', 'config': {'layers': [wrapped]}}
                for _ in range(4):
                    config = {'class_name': 'Functional', 'config': {'layers': [config]}}
                with self.subTest(layer=layer, wrapper=wrapped['class_name']):
                    self.assertEqual(bool(keras_scan.scan_config(config, 'config')), expected)
                    raw = archive([('config.json', json.dumps(config).encode()), ('metadata.json', b'{}'), ('model.weights.h5', b'weights')])
                    self.assertEqual(bool(findings(raw)), expected)

    def test_hdf5_in_memory(self):
        import h5py
        for fn, expected in [({'class_name': '__lambda__', 'config': {'code': 'serialized'}}, True),
                             ({'class_name': 'function', 'config': 'relu', 'module': 'builtins'}, False)]:
            data = io.BytesIO()
            with h5py.File(data, 'w') as f:
                f.attrs['model_config'] = json.dumps({'class_name': 'Lambda', 'config': {'function': fn}})
            self.assertEqual(bool(findings(data.getvalue())), expected)
            self.assertEqual(bool(findings(archive([('nested.h5', data.getvalue())]))), expected)


class LoaderBoundaries(unittest.TestCase):
    def test_numpy_nested(self):
        bad = stream(call(glob('os', 'system'), 'x'))
        source = call(glob('io', 'BytesIO'), npy(bad))
        self.assertFalse(findings(stream(call(glob('numpy', 'load'), source))))
        self.assertTrue(findings(stream(call(partial(glob('numpy', 'load'), allow_pickle=True), source))))
        zipped = call(glob('io', 'BytesIO'), archive([('a.npy', npy(bad))]))
        loader = call(partial(glob('numpy', 'load'), allow_pickle=True), zipped)
        self.assertFalse(findings(stream(loader)))
        self.assertTrue(findings(stream(call(glob('operator', 'getitem'), loader, 'a'))))

    def test_nested_trailing_and_torch(self):
        bad = stream(call(glob('os', 'system'), 'x'))
        good = stream({'clean': True})
        for mod in ('pickle', 'joblib', 'pandas'):
            name = 'read_pickle' if mod == 'pandas' else 'load'
            self.assertFalse(findings(stream(call(glob(mod, name), call(glob('io', 'BytesIO'), good + bad)))))
        self.assertFalse(findings(stream(call(glob('torch', 'load'), call(glob('io', 'BytesIO'), good + bad)))))
        # Torch ZIP uses its storage resolver; raw pickle stops at BINPERSID.
        after_pid = b'\x80\x04' + lit(('storage',)) + b'Q0' + lit(call(glob('os', 'system'), 'x')) + b'.'
        self.assertFalse(findings(after_pid))
        torch = archive([('weights/data.pkl', after_pid), ('weights/version', b'3\n'), ('weights/data/0', b'\0\0')])
        self.assertTrue(findings(torch))
        self.assertFalse(findings(stream(call(glob('torch', 'load'), call(glob('io', 'BytesIO'), b'prefix' + torch)))))

    def test_tar_containing_zip(self):
        for payload, expected in [(stream(call(glob('os', 'system'), 'x')), True), (stream('data'), False)]:
            raw = archive([('weights/data.pkl', payload), ('weights/version', b'3\n'), ('weights/data/0', b'raw storage')])
            out = io.BytesIO()
            with tarfile.open(fileobj=out, mode='w') as t:
                info = tarfile.TarInfo('model.pt')
                info.size = len(raw)
                t.addfile(info, io.BytesIO(raw))
            self.assertEqual(bool(findings(out.getvalue())), expected)
        # Unrelated members named version/data.pkl do not turn a general
        # archive into a checkpoint and hide its other members.
        decoy = archive([('a/data.pkl', stream('data')), ('b/version', b'3\n'),
                         ('payload.pkl', stream(call(glob('os', 'system'), 'x')))])
        result = findings(decoy)
        self.assertTrue(result)
        self.assertEqual(result[0].kind, 'process')

    def test_callback_short_circuit(self):
        side_effects = call(glob('builtins', 'map'), glob('os', 'system'), ['x'])
        for name in ('map', 'zip'):
            args = (glob('builtins', 'str'), [], side_effects) if name == 'map' else ([], side_effects)
            self.assertFalse(findings(stream(call(glob('builtins', 'list'), call(glob('builtins', name), *args)))))
        for cfg, expected in [({'version': 1}, False),
                              ({'version': 1, 'handlers': {'console': {'class': 'logging.StreamHandler'}}}, False),
                              ({'version': 1, 'handlers': {'file': {'class': 'logging.FileHandler', 'filename': '/outside'}}}, True),
                              ({'version': 1, 'handlers': {'file': {'class': 'logging.FileHandler', 'filename': '/outside', 'delay': True}}}, False),
                              ({'version': 1, 'handlers': {'process': {'()': 'subprocess.Popen', 'args': ['command']}}}, True)]:
            self.assertEqual(bool(findings(stream(call(glob('logging.config', 'dictConfig'), cfg)))), expected)


class ScannerBoundary(unittest.TestCase):
    def test_isolated_cli_and_no_side_effects(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        cli = str(root / 'cli.py')
        # The wrapper installs an audit barrier before importing the scanner.
        # This is an executable test of the scanner, not of any artifact.
        wrapper = r'''
import os, sys
sys.dont_write_bytecode = True
report, cli, artifact = sys.argv[1:]
def guard(event, args):
    if event == 'open':
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            if path != report:
                raise RuntimeError('forbidden write: ' + str(path))
    if event in ('os.system', 'os.fork', 'os.posix_spawn', 'subprocess.Popen',
                 'socket.__new__', 'socket.connect', 'socket.bind',
                 'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.link',
                 'os.symlink', 'os.chmod', 'os.chown', 'os.truncate', 'os.utime'):
        raise RuntimeError('forbidden side effect: ' + event)
sys.addaudithook(guard)
import runpy
sys.argv = [cli, 'scan', '--output', report, artifact]
runpy.run_path(cli, run_name='__main__')
'''
        with tempfile.TemporaryDirectory() as tmp:
            data = archive([('model.pkl', stream(call(glob('os', 'system'), 'never-execute-this')))])
            artifact = os.path.join(tmp, 'artifact.odd')
            report = os.path.join(tmp, 'report.json')
            import h5py
            h5 = io.BytesIO()
            with h5py.File(h5, 'w') as f:
                f.attrs['model_config'] = json.dumps({'class_name': 'Dense', 'config': {'activation': 'relu'}})
            sm = saved_model_pb2.SavedModel()
            node = sm.meta_graphs.add().graph_def.node.add()
            node.name, node.op = 'ReadFile', 'SaveV2'
            inputs = [(data, 'malicious'), (npy(stream({'data': 'subprocess'})), 'clean'),
                      (h5.getvalue(), 'clean'), (archive([('saved_model.pb', sm.SerializeToString())]), 'clean')]
            for raw, expected in inputs:
                pathlib.Path(artifact).write_bytes(raw)
                p = subprocess.run([sys.executable, '-I', '-c', wrapper, report, cli, artifact], cwd=tmp,
                                   env={'PATH': os.environ.get('PATH', ''), 'HOME': tmp, 'TMPDIR': tmp},
                                   capture_output=True, timeout=60)
                self.assertEqual(p.returncode, 0, p.stderr)
                doc = json.loads(pathlib.Path(report).read_text())
                self.assertEqual(doc['results'][0]['path'], artifact)
                self.assertEqual(doc['results'][0]['verdict'], expected, doc)
                self.assertNotIn('unsupported', {f['kind'] for f in doc['results'][0]['findings']})
            self.assertEqual(sorted(os.listdir(tmp)), ['artifact.odd', 'report.json'])

    def test_argument_and_report_contract(self):
        cli = str(pathlib.Path(__file__).resolve().parents[1] / 'cli.py')
        with tempfile.TemporaryDirectory() as tmp:
            pathlib.Path(tmp, 'clean.pkl').write_bytes(stream({'text': 'os.system'}))
            pathlib.Path(tmp, 'bad.pkl').write_bytes(stream(call(glob('os', 'system'), 'never-execute-this')))
            command = [sys.executable, '-I', cli, 'scan', '--output', 'report.json', './bad.pkl', 'clean.pkl']
            proc = subprocess.run(command, cwd=tmp, capture_output=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            doc = json.loads(pathlib.Path(tmp, 'report.json').read_text())
            self.assertEqual([r['path'] for r in doc['results']], ['./bad.pkl', 'clean.pkl'])
            self.assertEqual([r['verdict'] for r in doc['results']], ['malicious', 'clean'])
            own = pathlib.Path(tmp, 'own.pkl')
            own.write_bytes(stream(call(glob('builtins', 'open'), str(own), 'r')))
            proc = subprocess.run([sys.executable, '-I', cli, 'scan', '--output', 'report.json', str(own)], cwd=tmp, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(pathlib.Path(tmp, 'report.json').read_text())['results'][0]['verdict'], 'clean')
            proc = subprocess.run([sys.executable, '-I', cli, 'scan', '--unknown'], cwd=tmp, capture_output=True)
            self.assertEqual(proc.returncode, 2)


if __name__ == '__main__':
    unittest.main()
