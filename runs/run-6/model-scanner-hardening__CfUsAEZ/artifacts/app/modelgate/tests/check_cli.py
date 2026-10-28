#!/usr/bin/env python3
"""Integration, side-effect audit, and sequential registry-size timing check.

This is a test harness, not imported by the scanner. It creates fixtures and
launches the scanner; no fixture is loaded by a model or pickle loader.
"""
import argparse
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_scanner import G, Call, New, Build, imp, attr, partial, method, stream, npy, archive, EVIL, SAFE
from modelgate.tfproto.saved_model_pb2 import SavedModel
from modelgate.policy import TF_OPS


AUDIT_RUNNER = r'''
import os, runpy, sys
sys.dont_write_bytecode = True
cli, report, artifact = sys.argv[1:]
report = os.path.abspath(report)
violations = []
def audit(event, args):
    bad = False
    if event == 'open':
        path, mode, flags = args
        write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        bad = write and (not isinstance(path, (str, bytes)) or os.path.abspath(os.fsdecode(path)) != report)
    elif event in ('subprocess.Popen', 'os.system', 'os.fork', 'os.forkpty', 'os.posix_spawn', 'os.exec', 'socket.connect', 'socket.connect_ex', 'socket.bind', 'socket.getaddrinfo', 'socket.gethostbyname', 'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.link', 'os.symlink'):
        bad = True
    if bad:
        violations.append((event, repr(args)))
        raise RuntimeError('forbidden scan event: ' + event)
sys.addaudithook(audit)
sys.argv = [cli, 'scan', '--output', report, artifact]
try:
    runpy.run_path(cli, run_name='__main__')
except SystemExit as exc:
    if exc.code: raise
if violations:
    raise RuntimeError('scanner attempted forbidden events: ' + repr(violations))
'''


def fixtures(root):
    result = []
    pairs = [
        (SAFE, EVIL),
        (G('os','system'), Call(attr(imp('shutil'),'os.system'),('true',))),
        (Call(G('socket','socket')), method(Call(G('socket','socket')),'connect',('example.invalid',80))),
        (partial(G('os','system'),'true'), Call(partial(G('os','system'),'true'))),
        (Call(G('builtins','map'),(G('os','system'),['true'])), Call(G('builtins','list'),(Call(G('builtins','map'),(G('os','system'),['true'])),))),
        (Call(G('pickle','loads'),(stream(SAFE)+stream(EVIL),)), Call(G('pickle','loads'),(stream(EVIL),))),
        (New(G('subprocess','Popen'),(['echo'],)), Call(G('subprocess','Popen'),(['echo'],))),
    ]
    # Use a valid nested attribute chain, not a single getattr containing a dot.
    pairs[1] = (G('os','system'), Call(attr(attr(imp('shutil'),'os'),'system'),('true',)))
    for protocol in range(6):
        for pair in pairs:
            for evil, value in enumerate(pair):
                raw=stream(value,protocol)
                for kind in range(4):
                    payload=raw if kind==0 else gzip.compress(raw) if kind==1 else archive([('weights.npy',npy(raw))]) if kind==2 else archive([('model.pkl',raw)],prefix=b'header\n')
                    p=root/f'model-{len(result):04d}.bin';p.write_bytes(payload)
                    result.append((p,'malicious' if evil else 'clean'))
    for op in list(TF_OPS)+['Identity','Const','SaveV2','RestoreV2','Readfile']:
        model=SavedModel();model.meta_graphs.add();graph=model.meta_graphs.add().graph_def
        node=graph.library.function.add().node_def.add();node.op=op;node.name='ReadFile'
        p=root/f'sm-{len(result)}';p.mkdir();(p/'saved_model.pb').write_bytes(model.SerializeToString())
        result.append((p,'malicious' if op in TF_OPS else 'clean'))
    import h5py
    for i in range(8):
        code={'class_name':'__lambda__','config':{'code':'encoded bytecode'}}
        named={'class_name':'function','config':'relu','module':'builtins'}
        config={'class_name':'Functional','config':{'layers':[{'class_name':'TimeDistributed','config':{'layer':{'class_name':'Lambda','config':{'function':code if i%2 else named}}}}]}}
        p=root/f'keras-{i}.h5'
        with h5py.File(p,'w') as h5:h5.attrs['model_config']=json.dumps(config)
        result.append((p,'malicious' if i%2 else 'clean'))
        p=root/f'keras-{i}.keras';p.write_bytes(archive([('config.json',json.dumps(config)),('metadata.json','{}'),('model.weights.h5',b'')]))
        result.append((p,'malicious' if i%2 else 'clean'))
    return result


def validate(report, arg, expected):
    doc=json.loads(report.read_text())
    assert doc['scanner']=='modelgate' and len(doc['results'])==1,doc
    item=doc['results'][0]
    assert set(item)=={'path','verdict','findings'} and item['path']==arg,item
    assert item['verdict']==expected,item
    assert bool(item['findings'])==(expected=='malicious'),item
    for f in item['findings']:
        assert set(f)=={'kind','location','detail'} and all(isinstance(v,str) for v in f.values()),f
        assert f['kind'] in ('code-execution','process','file-access','network','unsupported'),f
        assert f['kind']!='unsupported',f


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--count',type=int,default=650)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='modelgate-validation-') as tmp:
        root=Path(tmp);inputs=root/'inputs';inputs.mkdir()
        models=fixtures(inputs)
        deploy=root/'deploy';deploy.mkdir()
        shutil.copytree(ROOT/'modelgate',deploy/'modelgate',ignore=shutil.ignore_patterns('__pycache__','tests'))
        cli=str(deploy/'modelgate'/'cli.py')
        runner=root/'audit.py';runner.write_text(AUDIT_RUNNER)
        report=root/'report.json'
        env={k:os.environ[k] for k in ('PATH','HOME','TMPDIR') if k in os.environ}
        # Include each parser and symbolic state mutation in the audit.
        selected=models[:8]+models[40:48]+models[-18:]
        for artifact,expected in selected:
            proc=subprocess.run([sys.executable,'-I',str(runner),cli,str(report),str(artifact)],cwd=root,env=env,capture_output=True,text=True,timeout=60)
            assert proc.returncode==0,proc.stderr
            validate(report,str(artifact),expected)
        assert not list(deploy.rglob('__pycache__')),'scanner wrote bytecode caches'
        print(f'audit: {len(selected)} scans, no process/network/extra-write events',flush=True)
        # Argument order and spelling are part of the external contract.
        names=[os.path.relpath(models[0][0],root),'./'+os.path.relpath(models[4][0],root)]
        proc=subprocess.run([sys.executable,'-I',cli,'scan','--output','report.json',*names],cwd=root,env=env,capture_output=True,text=True,timeout=60)
        assert proc.returncode==0,proc.stderr
        doc=json.loads(report.read_text())
        assert [r['path'] for r in doc['results']]==names,doc
        assert [r['verdict'] for r in doc['results']]==[models[0][1],models[4][1]],doc
        proc=subprocess.run([sys.executable,'-I',cli,'scan'],cwd=root,env=env,capture_output=True,timeout=60)
        assert proc.returncode==2,proc.returncode
        started=time.monotonic();slowest=0
        for i in range(args.count):
            artifact,expected=models[i%len(models)]
            before=time.monotonic()
            proc=subprocess.run([sys.executable,'-I',cli,'scan','--output',str(report),str(artifact)],cwd=root,env=env,capture_output=True,text=True,timeout=60)
            slowest=max(slowest,time.monotonic()-before)
            assert proc.returncode==0,proc.stderr
            validate(report,str(artifact),expected)
            if (i+1)%100==0:print(f'{i+1}/{args.count} scans passed ({time.monotonic()-started:.1f}s)',flush=True)
        elapsed=time.monotonic()-started
        assert elapsed<900 and slowest<60,(elapsed,slowest)
        assert not list(deploy.rglob('__pycache__')),'scanner wrote bytecode caches'
        print(f'{args.count} sequential scans passed in {elapsed:.2f}s; slowest {slowest:.3f}s; {len(models)} distinct fixtures')
    return 0

if __name__=='__main__':sys.exit(main())
