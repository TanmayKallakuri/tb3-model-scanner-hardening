"""Generate 560 policy cases and run fresh isolated CLIs with side-effect auditing.

Run from the repository root: python3 -m modelgate.tests.integration
The generated fixtures and reports are temporary; payloads are never loaded.
"""
import base64
import bz2
import gzip
import io
import json
import lzma
import os
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
import zlib
from pathlib import Path
from modelgate.tests.builders import *
from modelgate.tests.test_scanner import archive


def cases():
    paths=[('os.system','echo never'),('builtins.eval','1+1'),('builtins.open','/outside'),
           ('socket.create_connection',('localhost',9))]
    for proto in range(6):
        for path,arg in paths:
            yield dump(Call(G(path),(arg,)),proto),True
            yield dump({'callable':G(path),'text':str(arg)},proto),False
            yield dump([Memo(G(path),7),Call(Get(7),(arg,))],proto),True
            yield dump([Memo(G(path),7),Get(7)],proto),False
    for getter in (lambda p:G(p),lambda p:attr(C('importlib.import_module',p.rsplit('.',1)[0]),p.rsplit('.',1)[1]),
                   lambda p:C('pkgutil.resolve_name',p),lambda p:Call(C('operator.attrgetter',p.split('.',1)[1]),(C('importlib.import_module',p.split('.',1)[0]),)),
                   lambda p:C('copy.copy',G(p)),lambda p:C('functools.cache',G(p))):
        for path,arg in paths:
            yield dump(Call(getter(path),(arg,))),True
            yield dump(getter(path)),False
    bad=dump(C('os.system','echo never'));good=dump({'text':'os.system eval( subprocess'})
    for transform in (lambda b:b,lambda b:C('base64.b64decode',base64.b64encode(b)),
                      lambda b:C('zlib.decompress',zlib.compress(b)),lambda b:C('operator.add',b[:5],b[5:]),
                      lambda b:Call(Global('builtins','bytes.fromhex'),(b.hex(),)),lambda b:bytearray(b)):
        for payload,mal in ((bad,True),(good,False)):
            yield dump(C('pickle.loads',transform(payload))),mal
            yield dump(C('pickle.load',C('io.BytesIO',transform(payload)))),mal
    base=list([(bad,True),(good,False),(good+bad,True)])
    for compress in (gzip.compress,bz2.compress,lzma.compress,zlib.compress):
        for payload,mal in base:
            yield compress(payload),mal
            yield compress(compress(payload)),mal
            yield archive([('nested',compress(payload))]),mal
    for i in range(24):
        mal=i%2==0
        payload=bad if mal else good
        yield archive([('weights.pkl',payload)],b'prefix\0'*(i+1)),mal
        yield archive([('weights.pkl',good if mal else bad),('weights.pkl',payload)]),mal
    for i in range(36):
        mal=i%2==0
        header=repr({'descr':'|O' if i%3 else [('x','<i4'),('y','|O')],
                     'shape':(1,),'fortran_order':False}).encode()+b'\n'
        v=i%3+1
        npy=b'\x93NUMPY'+bytes([v,0])+struct.pack('<H' if v==1 else '<I',len(header))+header+(bad if mal else good)
        yield npy,mal
        yield archive([('data.npy',npy),('description.txt',b'os.system')]),mal
    for i in range(24):
        mal=i%2==0
        f='builtins.eval' if mal else 'builtins.str'
        part=Build(Memo(C('functools.partial',G('builtins.str')),3),(G(f),('1+1',),{},None))
        yield dump([part,Call(Get(3))],i%6),mal
        yield dump(Build(C('importlib.import_module','math'),{'sqrt':G(f)}))+dump(C('math.sqrt','1+1')),mal
    for i in range(24):
        mal=i%2==0
        code={'class_name':'__lambda__','config':{'value':[CODE_BASE64,None,None]}}
        named={'class_name':'function','module':'builtins','config':'relu'}
        key=('function','output_shape','activation')[i%3]
        layer={'class_name':'Dense' if key=='activation' else 'Lambda','config':{key:code if mal else named}}
        for j in range(i%8):
            layer={'class_name':'TimeDistributed','config':{'layer':layer}} if j%2 else {'class_name':'Sequential','config':{'layers':[layer]}}
        yield archive([('config.json',json.dumps(layer)),('metadata.json','{}'),('model.weights.h5',b'')]),mal
    for i in range(24):
        mal=i%2==0
        lazy=C(('builtins.map','builtins.filter','itertools.starmap')[i%3],G('builtins.eval'),
               [('1+1',)] if i%3==2 else ['1+1'])
        yield dump(C('builtins.list',lazy) if mal else lazy),mal
    for i in range(12):
        builtin,slot,source=(('builtins.len','__len__','1'),('builtins.bool','__bool__','True'),
                             ('builtins.str','__str__',"'text'"))[i%3]
        cls=C('builtins.type',f'Object{i}',(),{slot:C('functools.partial',G('builtins.eval'),source)})
        obj=New(cls)
        yield dump(obj,i%6),False
        yield dump(C(builtin,obj),i%6),True
    for i in range(4):
        scalar=lambda:C('numpy.int64',i+1)
        fn=C('operator.getitem',C('builtins.dict',[(scalar(),G('os.system'))]),scalar())
        yield dump(fn),False
        yield dump(Call(fn,('echo never',))),True
    for data,mal in ((bad,True),(good,False)):
        yield dump(Call(Global('numpy.lib._npyio_impl','pickle.loads'),(data,))),mal
    yield dump(C('numpy.lib.npyio.DataSource')),False
    yield dump(C('numpy.lib.npyio.DataSource',None)),True
    yield dump(C('numpy.lib._datasource.open','/outside')),True
    cls=C('builtins.type','Finalized',(),{'__del__':C('functools.partial',G('builtins.eval'),'1')})
    obj=New(cls)
    yield dump(obj),False
    yield dump(Raw(emit(obj)+b'0N')),True
    yield dump(Raw(emit(Memo(obj,9))+b'0N')),True
    yield dump(C('ftplib.FTP',C('numpy.bool_',False))),False
    yield dump(C('ftplib.FTP',C('numpy.str_','example.com'))),True
    # Real HDF5 and SavedModel cases are generated by main below.


def main():
    import warnings
    warnings.filterwarnings('ignore',message='Duplicate name:')
    from modelgate.tfproto import saved_model_pb2
    from google.protobuf import text_format
    import h5py
    runner=str(Path(__file__).with_name('audit_runner.py'))
    with tempfile.TemporaryDirectory(prefix='modelgate-regression-') as temp:
        root=Path(temp); fixtures=[]
        for i,(data,mal) in enumerate(cases()):
            p=root/f'case-{i}.artifact';p.write_bytes(data);fixtures.append((p,mal))
        for i in range(40):
            mal=i%2==0
            p=root/f'keras-{i}.h5'
            cfg={'class_name':'Lambda','config':{'function':{'class_name':'__lambda__','config':{'code':CODE_BASE64}} if mal else {'class_name':'function','config':'relu'}}}
            with h5py.File(p,'w') as f:f.attrs['model_config']=json.dumps(cfg)
            fixtures.append((p,mal))
        while len(fixtures)<560:
            i=len(fixtures);mal=i%2==0
            p=root/f'savedmodel-{i}';p.mkdir();(p/'variables').mkdir()
            sm=saved_model_pb2.SavedModel();sm.meta_graphs.add()
            m=sm.meta_graphs.add()
            f=m.graph_def.library.function.add();f.signature.name='hidden'
            op=('PyFunc','ReadFile','DataServiceDatasetV4')[i%3] if mal else ('SaveV2','RestoreV2','Const')[i%3]
            f.node_def.add(op=op,name='ReadFile')
            txt=i%3==0
            (p/('saved_model.pbtxt' if txt else 'saved_model.pb')).write_bytes(text_format.MessageToString(sm).encode() if txt else sm.SerializeToString())
            fixtures.append((p,mal))
        assert len(fixtures)==560,len(fixtures)
        start=time.monotonic();slowest=0;wrong=[]
        for i,(path,mal) in enumerate(fixtures):
            t=time.monotonic()
            proc=subprocess.run([sys.executable,'-I',runner,'scan','--output',str(root/'report.json'),str(path)],
                                cwd=root,env={k:os.environ.get(k,'') for k in ('PATH','HOME','TMPDIR')},capture_output=True,text=True,timeout=60)
            elapsed=time.monotonic()-t;slowest=max(slowest,elapsed)
            error=None
            if proc.returncode: error=f'exit {proc.returncode}: {proc.stderr[-1000:]}'
            else:
                from modelgate.report import KINDS
                try:
                    doc=json.loads((root/'report.json').read_text());r=doc['results'][0]
                    assert doc['scanner']=='modelgate' and len(doc['results'])==1
                    assert set(r)=={'path','verdict','findings'} and r['path']==str(path)
                    assert r['verdict']==('malicious' if mal else 'clean'),r
                    assert bool(r['findings'])==mal
                    for f in r['findings']:
                        assert set(f)=={'kind','location','detail'} and f['kind'] in KINDS
                        assert all(isinstance(v,str) for v in f.values())
                except Exception as exc:error=str(exc)
            if error:wrong.append((str(path),mal,error))
            if (i+1)%50==0:print(f'{i+1}/560: {len(wrong)} failures, {time.monotonic()-start:.1f}s',flush=True)
        duration=time.monotonic()-start
        print(json.dumps({'artifacts':len(fixtures),'failures':wrong,'seconds':round(duration,2),'slowest_seconds':round(slowest,2)},indent=2))
        return bool(wrong) or duration>900

if __name__=='__main__':sys.exit(main())
