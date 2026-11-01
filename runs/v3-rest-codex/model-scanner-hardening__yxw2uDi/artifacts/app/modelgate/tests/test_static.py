"""Regression tests construct pickle *bytes*. They never load a pickle/model."""
import sys
sys.dont_write_bytecode = True
import base64
import bz2
import gzip
import io
import json
import lzma
import os
import pickle
import struct
import tarfile
import tempfile
import unittest
import zipfile
import zlib

from modelgate.context import Context
from modelgate.dispatch import scan_bytes
from modelgate.pickle_scan import scan_pickle


class Global:
    def __init__(self, name): self.module, self.name = name.rsplit(':', 1) if ':' in name else name.rsplit('.', 1)
class Call:
    def __init__(self, func, *args): self.func, self.args = func, args
class New:
    def __init__(self, func, *args): self.func, self.args = func, args
class Build:
    def __init__(self, obj, state): self.obj, self.state = obj, state
class Raw:
    def __init__(self, data): self.data = data


def enc(x):
    if isinstance(x, Global): return enc(x.module) + enc(x.name) + b'\x93'
    if isinstance(x, Call): return enc(x.func) + enc(x.args) + b'R'
    if isinstance(x, New): return enc(x.func) + enc(x.args) + b'\x81'
    if isinstance(x, Build): return enc(x.obj) + enc(x.state) + b'b'
    if isinstance(x, Raw): return x.data
    if isinstance(x, tuple): return b'(' + b''.join(enc(v) for v in x) + b't'
    if isinstance(x, list): return b'](' + b''.join(enc(v) for v in x) + b'e'
    if isinstance(x, dict): return b'}(' + b''.join(enc(k)+enc(v) for k,v in x.items()) + b'u'
    if isinstance(x, str):
        value = x.encode('utf-8'); return b'X' + struct.pack('<I', len(value)) + value
    if isinstance(x, bytes): return b'B' + struct.pack('<I', len(x)) + x
    if isinstance(x, bytearray): return b'\x96' + struct.pack('<Q',len(x)) + x
    return pickle.dumps(x, protocol=2)[2:-1]


def stream(value, proto=4): return b'\x80' + bytes([proto]) + enc(value) + b'.'
def G(x): return Global(x)
def C(name, *args): return Call(G(name), *args)
def getattr_(obj, attr): return C('builtins.getattr', obj, attr)
def partial(func,*args): return C('functools.partial',func,*args)
def zip_bytes(entries, prefix=b''):
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as z:
        for name, value in entries: z.writestr(name,value)
    return prefix+data.getvalue()


class StaticTests(unittest.TestCase):
    def check(self, value, malicious=True, data=False, name='test.pkl'):
        ctx=Context()
        scan_bytes(value if data else stream(value),name,ctx)
        self.assertEqual(bool(ctx.findings), malicious, [(f.kind,f.detail,f.location) for f in ctx.findings])
        return ctx.findings

    def test_data_and_constructors(self):
        for value in [G('os.system'), {'eval(': ['os.system','subprocess', G('builtins.eval')]},
                      partial(G('os.system'),'x'), C('socket.socket'), C('ftplib.FTP'),
                      C('http.client.HTTPConnection','example.com'), C('xmlrpc.client.ServerProxy','http://example.com'),
                      New(G('subprocess.Popen'),'x'), New(G('builtins.eval'),'x'),
                      C('builtins.map',G('os.system'),['x'])]:
            with self.subTest(value=value): self.check(value,False)

    def test_capabilities(self):
        for name,args in [('os.system',('x',)),('os.execve',('x',[],{})),('subprocess.Popen',(['x'],)),
                          ('builtins.eval',('1+1',)),('builtins.compile',('x','<x>','exec')),
                          ('io.open',('/outside',)),('shutil.move',('/a','/b')),
                          ('socket.create_connection',(('host',80),)),('ftplib.FTP',('host',)),
                          ('smtplib.SMTP',('host',)),('ssl.get_server_certificate',(('host',443),)),
                          ('webbrowser.open',('http://example.com',))]:
            with self.subTest(name=name): self.check(C(name,*args))

    def test_indirection(self):
        osmod=C('importlib.import_module','os')
        variants=[getattr_(osmod,'system'), C('operator.attrgetter','system'),
                  C('operator.getitem',C('builtins.vars',osmod),'system'),
                  getattr_(C('operator.getitem',getattr_(C('builtins.__import__','sys'),'modules'),'os'),'system'),
                  C('pkgutil.resolve_name','os:system'), C('pkgutil.resolve_name','shutil.os.system'),
                  C('copy.copy',G('os.system')), C('functools.lru_cache',G('os.system')),
                  partial(G('os.system'))]
        # attrgetter needs its object, the other expressions already return a callable.
        variants[1]=Call(variants[1],osmod)
        for func in variants:
            with self.subTest(func=func): self.check(Call(func,'x'))
        self.check(Call(C('operator.methodcaller','system','x'),osmod))
        self.check(Call(Call(C('operator.attrgetter','os.system'),C('importlib.import_module','shutil')),'x'))

    def test_memo_retarget_and_module_state(self):
        p=Build(partial(G('builtins.str')), (G('os.system'),('x',),{},None))
        self.check(Call(p))
        self.check(p,False)
        module=Build(C('builtins.__import__','os'),{'quiet':G('builtins.eval')})
        self.check([module,C('os.quiet','1+1')])
        # Module state lasts beyond STOP; memo indices do not.
        self.check(stream(module)+stream(C('os.quiet','1+1')),data=True)
        memo=enc(G('os.system'))+b'q\x00' + b'0' + b'h\x00'+enc(('x',))+b'R'
        self.check(Raw(memo))

    def test_iteration(self):
        for cons in ('builtins.list','builtins.tuple','builtins.set','collections.deque','builtins.any','builtins.all','builtins.sum','builtins.next'):
            self.check(C(cons,C('builtins.map',G('os.system'),['x'])))
        self.check(C('builtins.list',C('itertools.starmap',G('os.system'),[('x',)])))
        self.check(C('builtins.list',C('builtins.filter',G('os.system'),['x'])))
        self.check(C('builtins.list',C('builtins.map',G('os.system'),[])),False)
        self.check(C('builtins.list',C('itertools.islice',C('builtins.map',G('os.system'),['x']),0)),False)
        self.check(C('functools.reduce',G('builtins.eval'),['a',{}]))
        self.check(C('operator.call',G('os.system'),'x'))
        self.check(Call(partial(G('builtins.sorted'),),['x']),False)
        self.check(Call(Build(partial(G('builtins.sorted')), (G('builtins.sorted'),(),{'key':G('os.system')},None)),['x']))

    def test_nested_pickle(self):
        bad=stream(C('os.system','x')); good=stream({'os.system':'eval('})
        for value in [bad,bytearray(bad),C('base64.b64decode',base64.b64encode(bad)),
                      C('zlib.decompress',zlib.compress(bad)),C('builtins.bytes.fromhex',bad.hex()),
                      C('operator.add',bad[:10],bad[10:])]:
            self.check(C('pickle.loads',value))
        self.check(C('pickle.loads',good),False)
        self.check(C('pickle.loads',good+bad),False)
        self.check(Call(C('pickle.loads',stream(G('os.system'))),'x'))
        self.check(C('pickle.load',C('io.BytesIO',bad)))
        self.check(C('pickle.loads',stream(G('os.system'))),False)
        self.check(C('torch.load','/outside'))

    def test_stream_semantics(self):
        bad=stream(C('os.system','x')); good=stream({})
        self.check(good+bad,data=True)
        self.check(b'\x80\x04'+enc(G('os.system'))+enc(['x'])+b'R.'+bad,False,data=True)
        self.check(b'cos\nsystem\n'+enc(('x',))+b'R.',data=True)
        self.check(b'(Vx\nios\nsystem\n.',data=True)
        self.check(b'(cos\nsystem\nVx\no.',data=True)
        self.check(b'\x80\x03cshutil\nos.system\n'+enc(('x',))+b'R.',False,data=True)
        self.check(b'\x80\x04cshutil\nos.system\n'+enc(('x',))+b'R.',data=True)
        for proto, mod in [(0,'__builtin__'),(2,'__builtin__'),(4,'builtins')]:
            self.check(b'\x80'+bytes([proto])+b'c'+mod.encode()+b'\neval\n'+enc(('1+1',))+b'R.',data=True)

    def test_legacy_torch(self):
        for badindex in range(5):
            streams=[stream(119547037146038801333356),stream(1001),stream({}),stream({}),stream([])]
            streams[badindex]=stream(C('os.system','x'))
            self.check(b''.join(streams),data=True)
        self.check(b''.join([stream(119547037146038801333356),stream(1001),stream({}),stream({}),stream([])])+stream(C('os.system','x')),False,data=True)

    def test_containers(self):
        bad=stream(C('os.system','x')); good=stream({})
        for wrapper in [gzip.compress,bz2.compress,lzma.compress,zlib.compress]:
            self.check(wrapper(bad),data=True,name='artifact')
            self.check(wrapper(good),False,data=True,name='artifact')
        self.check(gzip.compress(good)+gzip.compress(bad),data=True)
        self.check(zip_bytes([('weights.pkl',bad)],b'prefix'),data=True)
        self.check(zip_bytes([('same.pkl',bad),('same.pkl',good)]),False,data=True)
        self.check(zip_bytes([('same.pkl',good),('same.pkl',bad)]),data=True)
        torch=zip_bytes([('model/data.pkl',bad),('model/version',b'3\n'),('model/data/0',b'storage')])
        self.check(torch,data=True)
        t=io.BytesIO()
        with tarfile.open(fileobj=t,mode='w') as tar:
            info=tarfile.TarInfo('a.pt');info.size=len(torch);tar.addfile(info,io.BytesIO(torch))
        self.check(gzip.compress(t.getvalue()),data=True)

    def test_tensorflow(self):
        from modelgate.tfproto.saved_model_pb2 import SavedModel
        from modelgate.policy import TF_OPS
        from modelgate.savedmodel_scan import scan_saved_model
        from google.protobuf import text_format
        for op in list(TF_OPS)+['SaveV2','RestoreV2','ReadFile_','Const']:
            sm=SavedModel()
            sm.meta_graphs.add().graph_def.node.add(name='ReadFile',op='Const')
            meta=sm.meta_graphs.add()
            fn=meta.graph_def.library.function.add(); fn.signature.name='branch'
            fn.node_def.add(name='x',op=op)
            for data,text in [(sm.SerializeToString(),False),(text_format.MessageToString(sm).encode(),True)]:
                with self.subTest(op=op,text=text): self.assertEqual(bool(scan_saved_model(data,'sm',text)),op in TF_OPS)

    def test_keras(self):
        from modelgate.keras_scan import scan_config
        lam={'class_name':'__lambda__','config':{'code':'abc','defaults':None,'closure':None}}
        for field in ('function','output_shape','activation'):
            cfg={'class_name':'Lambda' if field!='activation' else 'Dense','config':{field:lam}}
            for _ in range(5): cfg={'class_name':'TimeDistributed','config':{'layer':cfg}}
            self.assertTrue(scan_config(cfg,'keras'))
        self.assertFalse(scan_config({'class_name':'Lambda','config':{'function':{'class_name':'function','config':'relu'}}},'keras'))
        self.check(zip_bytes([('config.json',json.dumps({'class_name':'Lambda','config':{'function':lam}})),('metadata.json','{}')]),data=True)


if __name__=='__main__': unittest.main()
