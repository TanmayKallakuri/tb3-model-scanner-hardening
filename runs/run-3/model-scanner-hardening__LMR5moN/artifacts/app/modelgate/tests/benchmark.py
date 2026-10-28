"""Generate a policy-derived 300-artifact corpus and time isolated CLI scans.
This is a development harness, not a loader. It never unpickles an artifact.
"""
import bz2
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))
sys.path.insert(0, str(PACKAGE / 'tests'))
from test_scanner import call, attr, partial, g, new, build, dump, GOOD, BAD, npy, zip_bytes
from modelgate.policy import TF_OPS
from modelgate.tfproto.saved_model_pb2 import SavedModel


def main():
    cases = []
    for name in ('os.system', 'posix.system', 'subprocess.run', 'subprocess.Popen', 'builtins.eval',
                 'builtins.exec', 'builtins.compile', 'runpy.run_path', 'timeit.timeit', 'builtins.open',
                 'io.open', 'os.remove', 'shutil.copyfile', 'socket.create_connection',
                 'urllib.request.urlopen', 'webbrowser.open'):
        cases.extend([(dump(call(name, 'payload')), True), (dump(g(name)), False), (dump(new(name, 'payload')), False)])
    cases.extend([
        (dump(call(attr(call('importlib.import_module', 'os'), 'system'), 'payload')), True),
        (dump(partial('os.system', 'payload')), False),
        (dump(call(partial('os.system', 'payload'))), True),
        (dump(call('pickle.loads', BAD)), True),
        (dump(call('pickle.loads', GOOD)), False),
        (dump(call('pickle.load', call('io.BytesIO', BAD))), True),
        (dump(call('pickle.load', call('io.BytesIO', GOOD))), False),
        (dump(call('socket.socket')), False),
        (dump(call('http.client.HTTPConnection', 'localhost')), False),
        (dump(call('ftplib.FTP')), False),
        (dump(call('ftplib.FTP', 'localhost')), True),
        (dump(call('smtplib.SMTP')), False),
        (dump(call('smtplib.SMTP', 'localhost')), True),
    ])
    with tempfile.TemporaryDirectory(prefix='modelgate-benchmark-') as tmp:
        root = Path(tmp)
        artifacts = []
        # Exercise content detection independently of extensions at each layer.
        for i, (payload, bad) in enumerate(cases):
            for wrap in (lambda b: b, gzip.compress, bz2.compress, lambda b: npy(b)):
                path = root / f'artifact_{len(artifacts):03d}.bin'
                path.write_bytes(wrap(payload))
                artifacts.append((path, bad))
        for name in list(TF_OPS) + ['RestoreV2', 'SaveV2', 'Const', 'MergeV2Checkpoints']:
            path = root / f'artifact_{len(artifacts):03d}'
            path.mkdir()
            sm = SavedModel()
            sm.meta_graphs.add()
            function = sm.meta_graphs.add().graph_def.library.function.add()
            function.signature.name = 'body'
            function.node_def.add(name='data_not_capability', op=name)
            (path / 'saved_model.pb').write_bytes(sm.SerializeToString())
            artifacts.append((path, name in TF_OPS))
        import h5py
        while len(artifacts) < 300:
            bad = len(artifacts) % 2 == 0
            layer = {'class_name': 'Lambda', 'config': {'function': {'class_name': '__lambda__', 'config': {'code': 'Ynl0ZWNvZGU='}}}} if bad else {'class_name': 'Lambda', 'config': {'function': {'class_name': 'function', 'module': 'builtins', 'config': 'relu'}}}
            for _ in range(4):
                layer = {'class_name': 'TimeDistributed', 'config': {'layer': layer}}
            path = root / f'artifact_{len(artifacts):03d}.h5'
            with h5py.File(path, 'w') as fh:
                fh.attrs['model_config'] = json.dumps(layer)
            artifacts.append((path, bad))
        artifacts = artifacts[:300]
        cli = str(PACKAGE / 'cli.py')
        report = root / 'report.json'
        env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': tmp, 'TMPDIR': tmp}
        total = time.monotonic()
        durations, errors = [], []
        for i, (path, bad) in enumerate(artifacts):
            started = time.monotonic()
            proc = subprocess.run([sys.executable, '-I', cli, 'scan', '--output', str(report), str(path)],
                                  cwd=tmp, env=env, capture_output=True, text=True, timeout=60)
            duration = time.monotonic() - started
            durations.append(duration)
            if proc.returncode:
                errors.append((str(path), proc.stderr))
            else:
                doc = json.loads(report.read_text())
                result = doc['results'][0]
                if result['verdict'] != ('malicious' if bad else 'clean') or result['path'] != str(path):
                    errors.append((str(path), result))
            if (i + 1) % 50 == 0:
                print(f'{i + 1}/300 scans; elapsed {time.monotonic() - total:.2f}s; mismatches {len(errors)}', flush=True)
        print(json.dumps({'artifacts': len(artifacts), 'seconds': round(time.monotonic() - total, 3),
                          'slowest_seconds': round(max(durations), 3), 'errors': errors}, indent=2))
        return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
