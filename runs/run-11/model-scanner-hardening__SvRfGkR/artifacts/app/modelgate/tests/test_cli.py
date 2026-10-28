"""Deployment and report contract tests in isolated Python processes."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import os
import shutil
import subprocess
import tempfile
import unittest

from test_scanner import pkl, call, global_


class DeploymentTests(unittest.TestCase):
    def test_isolated_deployment_and_report_contract(self):
        package = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='modelgate-deployment-') as tmp:
            root = Path(tmp)
            deployed = root / 'deployed' / 'modelgate'
            shutil.copytree(package, deployed, ignore=shutil.ignore_patterns('tests', '__pycache__', '*.pyc'))
            cwd = root / 'other working directory'
            cwd.mkdir()
            (cwd / 'clean λ.pkl').write_bytes(pkl({'text': 'eval(', 'fn': global_('os', 'system')}))
            (cwd / 'bad.pkl').write_bytes(pkl(call('os.system', 'must never execute')))
            (cwd / 'unknown.data').write_bytes(b'not a supported model format')
            env = {'PATH': os.environ['PATH'], 'HOME': str(cwd), 'TMPDIR': str(cwd)}
            cli = str(deployed / 'cli.py')
            report = str(cwd / 'report.json')
            paths = ['./clean λ.pkl', 'bad.pkl', './clean λ.pkl', 'unknown.data']
            before = {p.relative_to(root) for p in root.rglob('*')}
            proc = subprocess.run([sys.executable, '-I', cli, 'scan', '--output', report] + paths,
                                  cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            doc = json.loads(Path(report).read_text())
            self.assertEqual(doc['scanner'], 'modelgate')
            entries = doc['results']
            self.assertEqual([x['path'] for x in entries], paths)
            self.assertEqual([x['verdict'] for x in entries], ['clean', 'malicious', 'clean', 'malicious'])
            self.assertEqual(entries[-1]['findings'][0]['kind'], 'unsupported')
            self.assertEqual({p.relative_to(root) for p in root.rglob('*')} - before,
                             {Path('other working directory/report.json')})
            for entry in entries:
                self.assertEqual(set(entry), {'path', 'verdict', 'findings'})
                self.assertEqual(bool(entry['findings']), entry['verdict'] == 'malicious')
                for finding in entry['findings']:
                    self.assertEqual(set(finding), {'kind', 'location', 'detail'})
                    self.assertTrue(all(isinstance(v, str) for v in finding.values()))
            for args in [[], ['scan'], ['scan', '--output', report], ['scan', '--unknown', 'bad.pkl']]:
                proc = subprocess.run([sys.executable, '-I', cli] + args,
                                      cwd=cwd, env=env, capture_output=True, timeout=60)
                self.assertEqual(proc.returncode, 2)
            audit = str(Path(__file__).resolve().with_name('audit_scan.py'))
            proc = subprocess.run([sys.executable, '-I', audit, cli, report, str(cwd / 'bad.pkl')],
                                  cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == '__main__':
    unittest.main()
