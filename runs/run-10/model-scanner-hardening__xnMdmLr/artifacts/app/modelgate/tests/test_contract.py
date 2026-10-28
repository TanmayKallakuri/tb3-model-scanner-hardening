"""Exercise the deployed entry point with a sterile environment and audit hook."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from modelgate.tests.test_scanner import p, g, literal, call, npy, zip_bytes

CLI = str(Path(__file__).resolve().parents[1] / 'cli.py')
AUDIT = r'''
import os, sys
sys.dont_write_bytecode = True
report, cli, *paths = sys.argv[1:]
allowed = os.path.abspath(report)
violations = []
def audit(event, args):
    if event == 'open':
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            if not isinstance(path, (str, bytes)) or os.path.abspath(os.fsdecode(path)) != allowed:
                violations.append(event)
                raise RuntimeError('unexpected file write')
    elif event in ('subprocess.Popen', 'os.system', 'os.fork', 'os.forkpty', 'os.posix_spawn',
                   'os.exec', 'socket.connect', 'socket.bind', 'socket.getaddrinfo',
                   'os.remove', 'os.rename', 'os.mkdir'):
        violations.append(event)
        raise RuntimeError('unexpected side effect: ' + event)
sys.addaudithook(audit)
sys.path.insert(0, os.path.dirname(os.path.dirname(cli)))
from modelgate.cli import main
status = main(['scan', '--output', report, *paths])
if violations:
    print(violations)
    sys.exit(9)
sys.exit(status)
'''


class ContractTests(unittest.TestCase):
    def run_cli(self, cwd, args, audited=False):
        env = {'PATH': os.environ['PATH'], 'HOME': cwd, 'TMPDIR': cwd}
        cmd = ([sys.executable, '-I', '-c', AUDIT, args[0], CLI, *args[1:]] if audited else
               [sys.executable, '-I', CLI, *args])
        return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=20)

    def test_cli_and_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixtures = {
                'plain.dat': (p(literal({'note': 'os.system'})), 'clean'),
                'code.dat': (p(call(g('builtins.eval'), literal('1+1'))), 'malicious'),
                'process.dat': (p(call(g('os.system'), literal('must never execute'))), 'malicious'),
                'file.dat': (p(call(g('builtins.open'), literal('/outside/artifact'), literal('w'))), 'malicious'),
                'net.dat': (p(call(g('socket.create_connection'), literal(('127.0.0.1', 1)))), 'malicious'),
                'array.dat': (npy(p(literal([1, 2, 3]))), 'clean'),
            }
            fixtures['archive.dat'] = (zip_bytes([('model.bin', fixtures['process.dat'][0])], b'prefix'), 'malicious')
            # Prepare HDF5 in memory. h5py's platform query must not spawn uname
            # either during fixture construction or during the audited scan.
            import platform
            cached = platform.uname_result(*os.uname())
            cached.__dict__['processor'] = ''
            platform._uname_cache = cached
            import h5py
            buffer = io.BytesIO()
            with h5py.File(buffer, 'w') as fh:
                fh.attrs['model_config'] = json.dumps({'class_name': 'Sequential', 'config': {'layers': []}})
            fixtures['hdf.dat'] = (buffer.getvalue(), 'clean')
            for name, (data, _) in fixtures.items():
                Path(tmp, name).write_bytes(data)
            paths = ['./' + name for name in fixtures]
            proc = self.run_cli(tmp, ['report.json', *paths], audited=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            report = json.loads(Path(tmp, 'report.json').read_text())
            self.assertEqual(report['scanner'], 'modelgate')
            self.assertEqual([r['path'] for r in report['results']], paths)
            self.assertEqual([r['verdict'] for r in report['results']], [v for _, v in fixtures.values()])
            for result in report['results']:
                self.assertEqual(set(result), {'path', 'verdict', 'findings'})
                self.assertEqual(bool(result['findings']), result['verdict'] == 'malicious')
                for finding in result['findings']:
                    self.assertEqual(set(finding), {'kind', 'location', 'detail'})
                    self.assertTrue(all(isinstance(v, str) for v in finding.values()))
            self.assertEqual(set(os.listdir(tmp)), set(fixtures) | {'report.json'})

    def test_usage_and_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            for args in ([], ['scan'], ['scan', '--unknown'], ['unknown']):
                self.assertEqual(self.run_cli(tmp, args).returncode, 2)
            Path(tmp, 'unknown').write_bytes(b'\x00unrecognized')
            proc = self.run_cli(tmp, ['scan', '--output', 'report.json', 'unknown'])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(Path(tmp, 'report.json').read_text())['results'][0]
            self.assertEqual(result['verdict'], 'malicious')
            self.assertEqual(result['findings'][0]['kind'], 'unsupported')


if __name__ == '__main__':
    unittest.main()
