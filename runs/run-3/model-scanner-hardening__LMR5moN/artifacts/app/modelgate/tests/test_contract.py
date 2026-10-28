import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from test_scanner import BAD, GOOD, npy, zip_bytes

PACKAGE = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_isolated_deployment_and_no_side_effects(self):
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            deployment = work / 'deployed' / 'modelgate'
            shutil.copytree(PACKAGE, deployment, ignore=shutil.ignore_patterns('__pycache__'))
            payloads = {
                'clean.pkl': GOOD,
                'malicious.bin': BAD,
                'clean.npy': npy(GOOD),
                'malicious.npy': npy(BAD),
                'clean.zip': zip_bytes([('model.pkl', GOOD)]),
                'malicious.zip': zip_bytes([('model.pkl', BAD)]),
            }
            for name, data in payloads.items():
                (work / name).write_bytes(data)
            import h5py
            for bad in (False, True):
                layer = {'class_name': 'Lambda', 'config': {'function': {'class_name': '__lambda__', 'config': {'code': 'Ynl0ZWNvZGU='}}}} if bad else {'class_name': 'Dense', 'config': {'name': 'os.system'}}
                with h5py.File(work / ('malicious.h5' if bad else 'clean.h5'), 'w') as fh:
                    fh.attrs['model_config'] = json.dumps(layer)
            from modelgate.tfproto.saved_model_pb2 import SavedModel
            for bad in (False, True):
                folder = work / ('malicious.savedmodel' if bad else 'clean.savedmodel')
                folder.mkdir()
                model = SavedModel()
                node = model.meta_graphs.add().graph_def.node.add()
                node.op = 'PyFunc' if bad else 'RestoreV2'
                node.name = 'test'
                (folder / 'saved_model.pb').write_bytes(model.SerializeToString())
            paths = ['./' + p.name for p in work.iterdir() if p.name != 'deployed']
            report = work / 'report.json'
            env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(work), 'TMPDIR': str(work)}
            cli = deployment / 'cli.py'
            for audited in (False, True):
                if audited:
                    cmd = [sys.executable, '-I', str(deployment / 'tests' / 'audit_scan.py'), str(cli), str(report), *paths]
                else:
                    cmd = [sys.executable, '-I', str(cli), 'scan', '--output', str(report), *paths]
                before = {str(p.relative_to(work)) for p in work.rglob('*')}
                proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                document = json.loads(report.read_text())
                self.assertEqual(document['scanner'], 'modelgate')
                self.assertEqual([r['path'] for r in document['results']], paths)
                for result in document['results']:
                    self.assertEqual(set(result), {'path', 'verdict', 'findings'})
                    bad = result['path'].startswith('./malicious')
                    self.assertEqual(result['verdict'], 'malicious' if bad else 'clean', result)
                    self.assertEqual(bool(result['findings']), bad)
                    for finding in result['findings']:
                        self.assertEqual(set(finding), {'kind', 'location', 'detail'})
                        self.assertTrue(all(isinstance(v, str) for v in finding.values()))
                        self.assertNotEqual(finding['kind'], 'unsupported', finding)
                after = {str(p.relative_to(work)) for p in work.rglob('*')}
                self.assertEqual(after - before, {'report.json'} if not audited else set())

    def test_usage_and_unsupported(self):
        with tempfile.TemporaryDirectory() as work:
            report = Path(work) / 'report.json'
            cli = str(PACKAGE / 'cli.py')
            proc = subprocess.run([sys.executable, '-I', cli, 'scan', '--output', str(report)], cwd=work, capture_output=True)
            self.assertEqual(proc.returncode, 2)
            unknown = Path(work) / 'unknown'
            unknown.write_bytes(b'not a model format')
            proc = subprocess.run([sys.executable, '-I', cli, 'scan', '--output', str(report), str(unknown)], cwd=work, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(report.read_text())['results'][0]
            self.assertEqual(result['verdict'], 'malicious')
            self.assertEqual(result['findings'][0]['kind'], 'unsupported')


if __name__ == '__main__':
    unittest.main()
