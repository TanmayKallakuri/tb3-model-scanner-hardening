"""Exercise isolated CLI invocations on independent fixtures, not sample labels.

python3 -B modelgate/tests/benchmark.py --count 650
Use --audit to additionally reject scanner process/network/write side effects.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import base64
import gzip
import io
import json
import marshal
import os
import platform
import subprocess
import tempfile
import time
import zipfile

from test_scanner import pkl, call, global_, attr, zip_bytes, npy, TORCH_MAGIC
from modelgate.tfproto.saved_model_pb2 import SavedModel
from google.protobuf import text_format


def fixtures():
    cases = []
    for bad in (False, True):
        suffix = 'bad' if bad else 'good'
        fn = global_('os', 'system') if bad else global_('builtins', 'str')
        value = call(fn, 'test')
        raw = pkl(value)
        cases += [(suffix + '-raw.pkl', raw, bad),
                  (suffix + '-gzip.bin', gzip.compress(raw), bad),
                  (suffix + '-npz.bin', zip_bytes([('x.npy', npy(raw))]), bad),
                  (suffix + '-prefixed.zip', zip_bytes([('x', raw)], prefix=pkl(42)), bad),
                  (suffix + '-torch.pt', zip_bytes([('archive/data.pkl', raw), ('archive/version', b'3')]), bad),
                  (suffix + '-legacy.bin', b''.join(pkl(x) for x in [TORCH_MAGIC, 1001, {}]) + raw + pkl([]) + b'\0' * 64, bad),
                  (suffix + '-memo.pkl', pkl(call('builtins.list', call('builtins.map', fn, ['test']))), bad),
                  (suffix + '-nested.pkl', pkl(call('pickle.load', call('io.BytesIO', raw))), bad)]
        sm = SavedModel()
        graph = sm.meta_graphs.add().graph_def
        node = graph.library.function.add().node_def.add()
        node.name = 'ReadFile PyFunc os.system'
        node.op = 'EagerPyFunc' if bad else 'RestoreV2'
        for fmt in ('pb', 'pbtxt'):
            data = sm.SerializeToString() if fmt == 'pb' else text_format.MessageToString(sm).encode()
            cases.append((suffix + '-tf-' + fmt, {'saved_model.' + fmt: data, 'variables/variables.index': b''}, bad))
        function = ({'class_name': '__lambda__', 'config': {
            'code': base64.b64encode(marshal.dumps(compile('1+1', '<fixture>', 'eval'))).decode(),
            'defaults': None, 'closure': None}} if bad else
            {'module': 'builtins', 'class_name': 'function', 'config': 'relu'})
        config = {'class_name': 'Sequential', 'config': {'layers': [
            {'class_name': 'TimeDistributed', 'config': {'layer':
             {'class_name': 'Lambda', 'config': {'function': function}}}}]}}
        cases.append((suffix + '-keras.bin', zip_bytes([
            ('config.json', json.dumps(config).encode()), ('metadata.json', b'{}'), ('model.weights.h5', b'')]), bad))
        platform.uname().__dict__.setdefault('processor', platform.uname().machine)
        import h5py
        target = io.BytesIO()
        with h5py.File(target, 'w') as fh:
            fh.attrs['model_config'] = json.dumps(config)
        cases.append((suffix + '-hdf5.bin', target.getvalue(), bad))
    # An embedded ZIP is data until its own lazy loader is consumed.
    embedded = zip_bytes([('array.npy', npy(pkl(call('os.system', 'test'))))])
    cases.append(('unused-zip.pkl', pkl(embedded), False))
    cases.append(('lazy-npz.pkl', pkl(call('numpy.load', call('io.BytesIO', embedded), None, True)), False))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=650)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    cli = str(Path(__file__).resolve().parents[1] / 'cli.py')
    audit = str(Path(__file__).resolve().with_name('audit_scan.py'))
    cases = fixtures()
    failures, durations = [], []
    with tempfile.TemporaryDirectory(prefix='modelgate-benchmark-') as tmp:
        root = Path(tmp)
        cwd = root / 'cwd'
        cwd.mkdir()
        paths = []
        for i in range(args.count):
            name, data, expected = cases[i % len(cases)]
            target = root / f'{i:04}-{name}'
            if isinstance(data, dict):
                for name, content in data.items():
                    member = target / name
                    member.parent.mkdir(parents=True, exist_ok=True)
                    member.write_bytes(content)
            else:
                target.write_bytes(data)
            paths.append((str(target), expected))
        report = str(cwd / 'report.json')
        env = {'PATH': os.environ['PATH'], 'HOME': str(cwd), 'TMPDIR': str(cwd)}
        started = time.monotonic()
        for i, (path, expected) in enumerate(paths):
            cmd = ([sys.executable, '-I', audit, cli, report, path] if args.audit else
                   [sys.executable, '-I', cli, 'scan', '--output', report, path])
            before = time.monotonic()
            proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
            durations.append(time.monotonic() - before)
            try:
                assert proc.returncode == 0, proc.stderr
                doc = json.loads(Path(report).read_text())
                assert doc['scanner'] == 'modelgate' and len(doc['results']) == 1
                entry = doc['results'][0]
                assert set(entry) == {'path', 'verdict', 'findings'}
                assert entry['path'] == path
                assert entry['verdict'] == ('malicious' if expected else 'clean'), entry
                assert bool(entry['findings']) == expected
                for finding in entry['findings']:
                    assert set(finding) == {'kind', 'location', 'detail'}
                    assert all(isinstance(v, str) for v in finding.values())
                    assert finding['kind'] in ('process', 'file-access', 'network', 'code-execution'), finding
            except Exception as exc:
                failures.append((Path(path).name, str(exc)))
            if (i + 1) % 100 == 0:
                print(f'{i+1}/{len(paths)} scans; {len(failures)} failures; {time.monotonic()-started:.2f}s', flush=True)
        elapsed = time.monotonic() - started
        assert sorted(p.name for p in cwd.iterdir()) == ['report.json']
    print(json.dumps({'scans': len(paths), 'failures': failures, 'seconds': round(elapsed, 3),
                      'slowest_seconds': round(max(durations), 3), 'audited': args.audit}, indent=2))
    return bool(failures) or elapsed > 900 or max(durations) > 60


if __name__ == '__main__':
    sys.exit(main())
