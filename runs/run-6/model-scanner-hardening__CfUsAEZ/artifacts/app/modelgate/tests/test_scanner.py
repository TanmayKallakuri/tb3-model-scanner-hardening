"""Policy regressions. Payloads are assembled as bytes and NEVER unpickled."""
import base64
import bz2
import gzip
import io
import json
import lzma
import marshal
import os
import pickle
import struct
import tarfile
import unittest
import warnings
import zipfile
import zlib
from dataclasses import dataclass

from modelgate.dispatch import scan_bytes
from modelgate.pickle_scan import scan_pickle, TORCH_MAGIC
from modelgate.policy import TF_OPS
from modelgate.savedmodel_scan import scan_saved_model
from modelgate.keras_scan import scan_config


@dataclass
class G:
    module: str
    name: str


@dataclass
class Call:
    fn: object
    args: object = ()


@dataclass
class New:
    cls: object
    args: object = ()
    kwargs: object = None


@dataclass
class Build:
    obj: object
    state: object


@dataclass
class Memo:
    obj: object
    index: int = 0


@dataclass
class Get:
    index: int = 0


def encode(value):
    if isinstance(value, G): return b'c' + value.module.encode() + b'\n' + value.name.encode() + b'\n'
    if isinstance(value, Call): return encode(value.fn) + encode(value.args) + b'R'
    if isinstance(value, New): return encode(value.cls) + encode(value.args) + (encode(value.kwargs) + b'\x92' if value.kwargs is not None else b'\x81')
    if isinstance(value, Build): return encode(value.obj) + encode(value.state) + b'b'
    if isinstance(value, Memo): return encode(value.obj) + b'q' + bytes([value.index])
    if isinstance(value, Get): return b'h' + bytes([value.index])
    if value is None: return b'N'
    if type(value) is bool: return b'\x88' if value else b'\x89'
    if isinstance(value, int): return b'I' + str(value).encode() + b'\n'
    if isinstance(value, float): return b'G' + struct.pack('>d', value)
    if isinstance(value, str):
        data = value.encode()
        return b'X' + struct.pack('<I', len(data)) + data
    if isinstance(value, bytes): return b'B' + struct.pack('<I', len(value)) + value
    if isinstance(value, bytearray): return b'\x96' + struct.pack('<Q', len(value)) + value
    if isinstance(value, (tuple, list, dict)):
        data = b''.join(encode(k) + encode(v) for k, v in value.items()) if isinstance(value, dict) else b''.join(map(encode, value))
        return b'(' + data + (b't' if isinstance(value, tuple) else b'l' if isinstance(value, list) else b'd')
    raise TypeError(value)


def stream(value, protocol=4): return b'\x80' + bytes([protocol]) + encode(value) + b'.'
def imp(module): return Call(G('importlib', 'import_module'), (module,))
def attr(obj, name): return Call(G('builtins', 'getattr'), (obj, name))
def partial(fn, *args, **kw):
    if kw: return Build(Call(G('functools', 'partial'), (fn,)), (fn, args, kw, None))
    return Call(G('functools', 'partial'), (fn,) + args)
def method(obj, name, *args): return Call(attr(obj, name), args)


def archive(entries, prefix=b'', compression=zipfile.ZIP_STORED):
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with zipfile.ZipFile(buf, 'w', compression=compression) as z:
            for name, value in entries: z.writestr(name, value)
    return prefix + buf.getvalue()


def npy(payload, dtype='|O', version=1):
    header = repr({'descr': dtype, 'fortran_order': False, 'shape': (1,)}).encode() + b'\n'
    return b'\x93NUMPY' + bytes([version, 0]) + struct.pack('<H' if version == 1 else '<I', len(header)) + header + payload


EVIL = Call(G('os', 'system'), ('echo modelgate-test',))
SAFE = {'note': 'os.system eval( exec( subprocess', 'weights': [1, 2, 3]}


class ScannerTests(unittest.TestCase):
    def check(self, value, malicious, *, protocol=4, kind=None):
        data = stream(value, protocol)
        findings = scan_pickle(data, 'test.pkl')
        self.assertEqual(bool(findings), malicious, findings)
        if malicious:
            self.assertNotEqual(findings[0].kind, 'unsupported', findings)
            if kind: self.assertEqual(findings[0].kind, kind, findings)

    def test_data_and_globals(self):
        self.check(SAFE, False)
        for module, name in [('os','system'),('subprocess','Popen'),('builtins','eval'),('socket','create_connection'),('io','open')]:
            with self.subTest(module=module, name=name):
                self.check({'callable': G(module, name)}, False)
        self.check([Memo(G('os','system')), Get()], False)
        self.check([Memo(G('os','system')), Call(Get(), ('true',))], True)
        self.check(Call(G('builtins','str'), ('os.system',)), False)
        self.check(Call(G('builtins','__import__'), ('subprocess',)), False)

    def test_protocols(self):
        for protocol in range(6):
            with self.subTest(protocol=protocol): self.check(EVIL, True, protocol=protocol)
        for protocol in range(6):
            with self.subTest(dotted=protocol):
                self.check(Call(G('shutil','os.system'), ('true',)), protocol >= 4, protocol=protocol)
        self.assertTrue(scan_pickle(b"(S'true'\nios\nsystem\n.", 'inst'))
        self.assertTrue(scan_pickle(b"(cos\nsystem\nS'true'\no.", 'obj'))
        self.assertFalse(scan_pickle(b"(isubprocess\nPopen\n.", 'inst-noarg'))
        self.assertFalse(scan_pickle(b"(csubprocess\nPopen\no.", 'obj-noarg'))
        self.assertTrue(scan_pickle(b'\x80\x04(' + encode(partial(G('os','system'),'true')) + b'o.', 'partial-obj'))
        self.check(Call(G('os','system'), ['true']), False)
        self.check(Call(G('os','system'), 'true'), False)
        self.check(Call(G('os','system'), (123,)), False)
        self.check(New(G('subprocess','Popen'), (['echo'],)), False)
        self.check(New(G('subprocess','Popen'), (['echo'],), {'shell': True}), False)
        self.check(New(G('os','system'), ('true',)), False)
        self.assertFalse(scan_pickle(b'Pstorage\n' + encode(EVIL) + b'.', 'pid'))

    def test_global_routes(self):
        routes = [
            G('posix','system'), G('nt','system'), attr(imp('os'),'system'),
            Call(Call(G('operator','attrgetter'), ('os.system',)), (imp('shutil'),)),
            Call(Call(G('operator','itemgetter'), ('system',)), (Call(G('builtins','vars'), (imp('os'),)),)),
            Call(G('operator','getitem'), (attr(imp('sys'),'modules'), 'os')),
            Call(G('pkgutil','resolve_name'), ('os:system',)),
            Call(G('pkgutil','resolve_name'), ('shutil.os.system',)),
            Call(G('copy','copy'), (G('os','system'),)),
            Call(Call(G('functools','lru_cache'), (None,)), (G('os','system'),)),
            Call(G('functools','cache'), (G('os','system'),)),
        ]
        routes[5] = attr(routes[5], 'system')
        for i, fn in enumerate(routes):
            with self.subTest(i=i):
                self.check(fn, False)
                self.check(Call(fn, ('true',)), True)
        self.check(Call(Call(G('operator','methodcaller'), ('system','true')), (imp('os'),)), True)
        self.check(Call(G('operator','call'), (G('os','system'), 'true')), True)
        self.check(Call(G('__builtin__','eval'), ('1+1',)), True, protocol=2)
        self.check(Call(G('commands','getoutput'), ('true',)), True, protocol=2)
        self.check(Call(G('urllib2','urlopen'), ('https://example.invalid',)), True, protocol=2)
        self.check(Call(G('__builtin__','unicode'), ('eval(',)), False, protocol=2)

    def test_module_rebinding_and_partial(self):
        self.check(Call(Build(Call(G('functools','partial'), (G('builtins','str'),)), (G('os','system'), ('true',), {}, None))), True)
        self.check(Build(Call(G('functools','partial'), (G('os','system'),)), (G('builtins','str'), ('true',), {}, None)), False)
        self.check(Call(Build(Call(G('functools','partial'), (G('os','system'),)), (G('builtins','str'), ('true',), {}, None))), False)
        self.check([Build(imp('builtins'), {'str': G('os','system')}), Call(G('builtins','str'), ('true',))], True)
        self.check([Build(imp('os'), {'system': G('builtins','str')}), Call(G('os','system'), ('true',))], False)
        self.check([Call(G('operator','setitem'), (Call(G('builtins','vars'), (imp('os'),)), 'fspath', G('os','system'))), Call(G('os','fspath'), ('true',))], True)
        self.check([Build(imp('shutil'), {'os': imp('builtins')}), Call(G('shutil','os.eval'), ('1+1',))], True)
        blob = stream(Build(imp('builtins'), {'str': G('os','system')})) + stream(Call(G('builtins','str'), ('true',)))
        self.assertTrue(scan_pickle(blob, 'streams'))

    def test_lazy_and_eager_callbacks(self):
        for module, name, args in [('builtins','map',(G('os','system'),['true'])),('builtins','filter',(G('os','system'),['true'])),('itertools','starmap',(G('os','system'),[('true',)]))]:
            value = Call(G(module,name),args)
            with self.subTest(name=name):
                self.check(value, False)
                self.check(Call(G('builtins','list'),(value,)),True)
                self.check(Call(G('builtins','next'),(value,)),True)
        self.check(Call(G('builtins','list'), (Call(G('builtins','map'), (G('builtins','str'),[1,2,3])),)),False)
        self.check(Call(G('builtins','list'), (Call(G('builtins','map'), (G('os','system'),[])),)),False)
        for name in ('sorted','min','max'):
            with self.subTest(name=name):
                self.check(Call(partial(G('builtins',name), ['true'], key=G('os','system'))),True)
                self.check(Call(partial(G('builtins',name), [], key=G('os','system'), default=None)) if name != 'sorted' else Call(partial(G('builtins',name), [], key=G('os','system'))),False)
        self.check(Call(G('functools','reduce'), (G('builtins','eval'), ['1', {}])),True)
        self.check(Call(G('functools','reduce'), (G('os','system'), ['true'])),False)
        self.check(Call(G('builtins','next'), (Call(G('builtins','iter'), (partial(G('os','system'),'true'), None)),)),True)
        self.check(Call(G('collections','deque'),(Call(G('builtins','map'),(G('os','system'),['true'])),0)),True)

    def test_nested_loading(self):
        for value, malicious in [(stream(SAFE),False),(stream(EVIL),True),(stream(SAFE)+stream(EVIL),False)]:
            with self.subTest(malicious=malicious,value=value[:20]):
                self.check(Call(G('pickle','loads'), (value,)),malicious)
                self.check(Call(G('_pickle','loads'), (bytearray(value),)),malicious)
                self.check(Call(G('pickle','load'), (Call(G('_io','BytesIO'),(value,)),)),malicious)
                for decoder, data in [(G('base64','b64decode'),base64.b64encode(value)),(G('zlib','decompress'),zlib.compress(value)),(G('bz2','decompress'),bz2.compress(value)),(G('lzma','decompress'),lzma.compress(value))]:
                    self.check(Call(G('pickle','loads'),(Call(decoder,(data,)),)),malicious)
                self.check(Call(G('pickle','loads'),(Call(G('builtins','bytes.fromhex'),(value.hex(),)),)),malicious)
                self.check(Call(G('pickle','loads'),(Call(G('operator','add'),(value[:5],value[5:])),)),malicious)
        for loader in [G('numpy','load'),G('joblib','load'),G('pandas','read_pickle'),G('torch','load')]:
            self.check(Call(loader,('/outside/artifact',)),True,kind='file-access')
        for evil in (False, True):
            legacy = b''.join(stream(v) for v in (TORCH_MAGIC,1001,{},EVIL if evil else SAFE,[])) + b'raw storage'
            self.check(Call(G('torch.storage','_load_from_bytes'),(legacy,)),evil)
        self.check(Call(G('pickle','loads'), (b'not a pickle',)),False)

    def test_clients_and_threads(self):
        for cls, args, methodname, methodargs in [
            (G('socket','socket'),(), 'connect', (('example.invalid',80),)),
            (G('http.client','HTTPConnection'),('example.invalid',80),'request',('GET','/')),
            (G('ftplib','FTP'),(),'connect',('example.invalid',)),
            (G('xmlrpc.client','ServerProxy'),('http://example.invalid',),'ping',()),
        ]:
            obj = Call(cls,args)
            self.check(obj,False)
            self.check(method(obj,methodname,*methodargs),True,kind='network')
        for module, cls in [('ftplib','FTP'),('smtplib','SMTP')]:
            self.check(Call(G(module,cls)),False)
            self.check(Call(G(module,cls),('example.invalid',)),True)
            self.check(New(G(module,cls),('example.invalid',)),False)
        thread = Call(partial(G('threading','Thread'),target=G('os','system'),args=('true',)))
        self.check(thread,False)
        self.check(method(thread,'start'),True)
        self.check(method(Call(G('concurrent.futures','ThreadPoolExecutor')),'submit',G('os','system'),'true'),True)
        self.check(Call(G('concurrent.futures','ThreadPoolExecutor')),False)
        self.check(method(Call(G('concurrent.futures','ThreadPoolExecutor')),'submit',G('builtins','str'),123),False)

    def test_code_and_file(self):
        code = marshal.dumps(compile('1+1','<test>','eval'))
        self.check(Call(G('marshal','loads'),(code,)),False)
        self.check(Call(G('types','FunctionType'),(Call(G('marshal','loads'),(code,)),{})),True)
        for cls, methodname in [(G('code','InteractiveInterpreter'),'runsource'),(G('bdb','Bdb'),'run')]:
            self.check(Call(cls),False)
            self.check(method(Call(cls),methodname,'1+1'),True)
        path = Call(G('pathlib','Path'),('/outside/artifact',))
        self.check(path,False)
        self.check(method(path,'write_text','data'),True)
        self.check(method(Call(G('io','BytesIO'),(b'data',)),'read'),False)
        self.check(Call(G('logging.config','dictConfig'),({'version':1,'handlers':{}},)),False)
        self.check(Call(G('logging.config','dictConfig'),({'version':1,'handlers':{'f': {'class':'logging.FileHandler','filename':'/outside'}}},)),True)

    def test_containers(self):
        for evil in (False,True):
            raw = stream(EVIL if evil else SAFE)
            cases = [raw, gzip.compress(raw),bz2.compress(raw),lzma.compress(raw),zlib.compress(raw),
                     archive([('model.pkl',raw)]), archive([('model.pkl',raw)],prefix=b'prefix bytes'),
                     npy(raw),npy(raw,version=2),npy(raw,version=3),archive([('array.npy',npy(raw))])]
            for i,data in enumerate(cases):
                with self.subTest(evil=evil,i=i):
                    findings=scan_bytes(data,'model.misleading')
                    self.assertEqual(bool(findings),evil,findings)
                    if evil:self.assertNotEqual(findings[0].kind,'unsupported',findings)
            buf=io.BytesIO()
            with tarfile.open(fileobj=buf,mode='w') as t:
                info=tarfile.TarInfo('nested/model.zip'); v=archive([('model.pkl',raw)]);info.size=len(v);t.addfile(info,io.BytesIO(v))
            self.assertEqual(bool(scan_bytes(gzip.compress(buf.getvalue()),'model.tgz')),evil)
        self.assertFalse(scan_bytes(npy(stream(EVIL),dtype='<f4'),'numeric.npy'))
        self.assertFalse(scan_bytes(npy(stream(EVIL),dtype='i4,f8'),'structured-numeric.npy'))
        self.assertTrue(scan_bytes(npy(stream(EVIL),dtype='i4,O'),'structured-object.npy'))
        for evil in (False,True):
            entries=[('m.pkl',stream(SAFE if evil else EVIL)),('m.pkl',stream(EVIL if evil else SAFE))]
            self.assertEqual(bool(scan_bytes(archive(entries),'duplicate.zip')),evil)
        self.assertTrue(scan_bytes(gzip.compress(stream(SAFE))+gzip.compress(stream(EVIL)),'multi.gz'))
        self.assertTrue(scan_bytes(stream(EVIL)+stream(SAFE),'multi.pkl'))
        self.assertTrue(scan_bytes(stream(SAFE)+stream(EVIL),'multi.pkl'))
        self.assertFalse(scan_bytes(stream(SAFE)+archive([('m.pkl',stream(SAFE))]),'prefixed.zip'))

    def test_tensorflow_closed_table(self):
        from modelgate.tfproto.saved_model_pb2 import SavedModel
        from google.protobuf import text_format
        for opname in list(TF_OPS) + ['SaveV2','RestoreV2','MergeV2Checkpoints','ShardedFilename','ShardedFilespec','Readfile','Const','Identity']:
            for place in ('graph','function'):
                with self.subTest(op=opname,place=place):
                    sm=SavedModel();sm.meta_graphs.add();g=sm.meta_graphs.add().graph_def
                    node=(g.node if place=='graph' else g.library.function.add().node_def).add()
                    node.name='ReadFile';node.op=opname;node.attr['note'].s=b'PyFunc os.system'
                    for binary in (True,False):
                        blob=sm.SerializeToString() if binary else text_format.MessageToString(sm).encode()
                        self.assertEqual(bool(scan_saved_model(blob,'model',text=not binary)),opname in TF_OPS)

    def test_keras(self):
        named={'module':'builtins','class_name':'function','config':'relu'}
        code={'class_name':'__lambda__','config':{'code':'data','defaults':None,'closure':None}}
        for function in (named,code):
            for field in ('function','output_shape','activation'):
                for wrapper in (None,'TimeDistributed','Bidirectional','Functional'):
                    cfg={'class_name':'Lambda' if field!='activation' else 'Dense','config':{field:function}}
                    if wrapper:cfg={'class_name':wrapper,'config':{'layer':cfg,'layers':[cfg]}}
                    self.assertEqual(bool(scan_config(cfg,'config')),function is code)
        self.assertFalse(scan_config({'note':{'class_name':'__lambda__','config':{'code':'text'}}},'config'))


if __name__=='__main__':unittest.main()
