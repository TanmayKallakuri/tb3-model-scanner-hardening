"""Exercise deployment isolation and audit scanner side effects."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_scanner import stream, Call, G, zipbytes, tarbytes, npy

CLI = str(Path(__file__).resolve().parents[1] / 'cli.py')
AUDITED_RUN = r'''
import os, runpy, sys
sys.dont_write_bytecode = True
cli, output, *paths = sys.argv[1:]
allowed_output = os.path.abspath(output)
def audit(event, args):
    if event == 'open':
        path, mode, flags = args
        if (isinstance(mode, str) and any(x in mode for x in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            if not isinstance(path, str) or os.path.abspath(path) != allowed_output:
                raise RuntimeError('scanner attempted an unexpected write: ' + repr(path))
    if (event.startswith('subprocess.') or event.startswith('os.exec') or event.startswith('os.spawn') or
        event in ('os.system', 'os.fork', 'os.forkpty', 'socket.connect', 'socket.bind', 'socket.getaddrinfo',
                  'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.chmod', 'os.link', 'os.symlink')):
        raise RuntimeError('scanner attempted forbidden side effect: ' + event)
sys.addaudithook(audit)
sys.argv = [cli, 'scan', '--output', output, *paths]
runpy.run_path(cli, run_name='__main__')
'''


class CommandLineSafety(unittest.TestCase):
    def test_isolated_cli_and_side_effects(self):
        from modelgate.tfproto import saved_model_pb2
        import h5py
        bad = stream(Call(G('os.system'), 'must-never-run'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            expected = []
            def add(name, data, malicious):
                path = root / name; path.write_bytes(data)
                paths.append(str(path)); expected.append(malicious)
            add('raw.wrong', bad, True)
            add('numeric.npy', npy(bad, '<f8'), False)
            add('object.npy', npy(bad), True)
            add('nested.zip', zipbytes([('a.tar', tarbytes([('b', bad)]))]), True)
            add('clean.pkl', stream([G('os.system'), 'os.system', 'eval(']), False)
            sm = saved_model_pb2.SavedModel()
            sm.meta_graphs.add().graph_def.node.add(name='side_effect', op='PyFunc')
            directory = root / 'savedmodel'; directory.mkdir()
            (directory / 'saved_model.pb').write_bytes(sm.SerializeToString())
            paths.append(str(directory)); expected.append(True)
            h5 = root / 'model.h5'
            with h5py.File(h5, 'w') as f:
                f.attrs['model_config'] = json.dumps({'class_name': 'Lambda', 'config': {'function':
                    {'class_name': '__lambda__', 'config': {'code': 'base64-bytecode'}}}})
            paths.append(str(h5)); expected.append(True)
            output = str(root / 'report.json')
            proc = subprocess.run([sys.executable, '-I', '-c', AUDITED_RUN, CLI, output, *paths],
                                  cwd=tmp, env={k: os.environ[k] for k in ('PATH', 'HOME', 'TMPDIR') if k in os.environ},
                                  text=True, capture_output=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            doc = json.loads(Path(output).read_text())
            self.assertEqual(doc['scanner'], 'modelgate')
            self.assertEqual(len(doc['results']), len(paths))
            for entry, path, malicious in zip(doc['results'], paths, expected):
                self.assertEqual(set(entry), {'path', 'verdict', 'findings'})
                self.assertEqual(entry['path'], path)
                self.assertEqual(entry['verdict'], 'malicious' if malicious else 'clean')
                self.assertEqual(bool(entry['findings']), malicious)
                # An audit rejection must not be hidden as unsupported.
                for finding in entry['findings']:
                    self.assertNotIn('scanner attempted', finding['detail'])
                    self.assertEqual(set(finding), {'kind', 'location', 'detail'})

    def test_usage_errors(self):
        for args in [[], ['scan'], ['scan', '--output', '/unused']]:
            proc = subprocess.run([sys.executable, '-I', CLI, *args], capture_output=True, timeout=5)
            self.assertEqual(proc.returncode, 2)
