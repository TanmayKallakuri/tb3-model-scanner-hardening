"""Policy regression tests. Payloads are assembled as bytes, never loaded."""
import sys
sys.dont_write_bytecode = True
import base64
import bz2
import gzip
import io
import json
import lzma
import pickle
import struct
import tarfile
import unittest
import zipfile
import zlib
from pathlib import Path

from modelgate import dispatch, pickle_scan, keras_scan, savedmodel_scan
from modelgate.symbolic import Context


class Expr:
    def __init__(self, data): self.data = data


def e(value):
    if isinstance(value, Expr): return value.data
    if value is None: return b'N'
    if value is True: return b'\x88'
    if value is False: return b'\x89'
    if isinstance(value, int): return b'I' + str(value).encode() + b'\n'
    if isinstance(value, float): return b'G' + struct.pack('>d', value)
    if isinstance(value, str):
        b = value.encode('utf-8'); return b'X' + struct.pack('<I', len(b)) + b
    if isinstance(value, bytes): return b'B' + struct.pack('<I', len(value)) + value
    if isinstance(value, bytearray): return b'\x96' + struct.pack('<Q', len(value)) + value
    if isinstance(value, tuple): return b'(' + b''.join(e(v) for v in value) + b't'
    if isinstance(value, list): return b'(' + b''.join(e(v) for v in value) + b'l'
    if isinstance(value, dict): return b'(' + b''.join(e(k) + e(v) for k, v in value.items()) + b'd'
    raise TypeError(type(value))


def g(module, name): return Expr(b'c' + module.encode() + b'\n' + name.encode() + b'\n')
def call(fn, *args): return Expr(e(fn) + e(tuple(args)) + b'R')
def attr(obj, name): return call(g('builtins', 'getattr'), obj, name)
def mod(name): return call(g('importlib', 'import_module'), name)
def build(obj, state): return Expr(e(obj) + e(state) + b'b')
def kwcall(fn, args=(), kw=None):
    p = build(call(g('functools','partial'), fn), (fn, args, kw or {}, None))
    return call(p)
def payload(obj, protocol=4): return b'\x80' + bytes([protocol]) + e(obj) + b'.'
def zipbytes(entries, prefix=b''):
    f=io.BytesIO()
    with zipfile.ZipFile(f, 'w') as z:
        for name, data in entries: z.writestr(name, data)
    return prefix + f.getvalue()
def scan(data, loader=None):
    ctx = Context()
    dispatch.analyze_bytes(data, 'test', ctx, loader=loader)
    return ctx.findings


class PicklePolicy(unittest.TestCase):
    def verdict(self, obj, malicious, protocol=4, kinds=None):
        findings = scan(payload(obj,protocol))
        self.assertEqual(bool(findings), malicious, findings)
        if kinds is not None:
            self.assertEqual({f.kind for f in findings}, set(kinds), findings)
        return findings

    def test_direct_and_stored(self):
        cases = [
            ('os','system',('true',),'process'), ('posix','system',('true',),'process'),
            ('nt','system',('true',),'process'), ('builtins','eval',('1+1',),'code-execution'),
            ('builtins','compile',('1+1','x','eval'),'code-execution'),
            ('subprocess','Popen',(['true'],),'process'),
            ('io','open',('/tmp/x',),'file-access'), ('shutil','copy',('/tmp/a','/tmp/b'),'file-access'),
            ('socket','create_connection',(('example.com',80),),'network'),
            ('urllib.request','urlopen',('https://example.com',),'network'),
            ('timeit','timeit',('1+1',),'code-execution'),
            ('pydoc','pipepager',('text','cat'),'process'),
            ('numpy.testing._private.utils','runstring',('1+1',),'code-execution'),
        ]
        for module,name,args,kind in cases:
            for protocol in (0,1,2,3,4,5):
                with self.subTest(module=module,name=name,protocol=protocol):
                    self.verdict([g(module,name), module+'.'+name], False, protocol)
                    self.verdict(call(g(module,name),*args), True, protocol, [kind])

    def test_aliases(self):
        for m,n,args in [('__builtin__','eval',('1+1',)),('commands','getoutput',('true',)),('urllib2','urlopen',('http://x',)),('httplib','HTTPConnection.request',(call(g('httplib','HTTPConnection'),'x'),'GET','/'))]:
            if '.' not in n: self.verdict(call(g(m,n),*args),True,2)
        for m in ('cPickle','_pickle'):
            self.verdict(call(g(m,'loads'),payload(call(g('os','system'),'true'))),True,2)
            self.verdict(call(g(m,'loads'),payload({'os.system':'subprocess'})),False,2)

    def test_protocol_dotted(self):
        expr=call(g('shutil','os.system'),'true')
        self.verdict(expr,True,4,['process'])
        self.verdict(expr,False,2)
        self.verdict(call(g('bdb','Bdb.run'),call(g('bdb','Bdb')),'1+1'),True,4,['code-execution'])

    def test_memo_stack_globals_and_containers(self):
        raw=b'\x80\x04'+e('os')+b'q\x01'+e('system')+b'q\x02'+b'0' + b'0h\x01h\x02\x93q\x03'+e(('true',))+b'R.'
        self.assertEqual([x.kind for x in scan(raw)],['process'])
        target=call(g('operator','getitem'),{'fn':g('os','system')},'fn')
        self.verdict(call(target,'true'),True,kinds=['process'])

    def test_resolvers(self):
        targets=[attr(mod('os'),'system'),
                 call(call(g('operator','attrgetter'),'system'),mod('os')),
                 attr(attr(mod('shutil'),'os'),'system'),
                 call(g('pkgutil','resolve_name'),'os:system'),
                 call(g('operator','getitem'),attr(mod('sys'),'modules'),'os')]
        for fn in targets[:-1]: self.verdict(call(fn,'true'),True,kinds=['process'])
        self.verdict(call(attr(targets[-1],'system'),'true'),True,kinds=['process'])
        for namespace in [call(g('builtins','vars'),mod('os')),attr(mod('os'),'__dict__')]:
            fn=call(g('operator','getitem'),namespace,'system')
            self.verdict(call(fn,'true'),True,kinds=['process'])
        self.verdict(call(call(g('operator','methodcaller'),'system','true'),mod('os')),True,kinds=['process'])

    def test_partial_and_wrappers(self):
        partial=call(g('functools','partial'),g('os','system'),'true')
        self.verdict(partial,False)
        self.verdict(call(partial),True,kinds=['process'])
        retarget=build(call(g('functools','partial'),g('builtins','str')), (g('os','system'),('true',),{},None))
        self.verdict(call(retarget),True,kinds=['process'])
        safe=build(partial,(g('builtins','str'),('hello',),{},None))
        self.verdict(call(safe),False)
        for fn in [call(g('copy','copy'),g('os','system')),call(g('functools','lru_cache'),g('os','system')),call(call(g('functools','lru_cache'),8),g('os','system'))]:
            self.verdict(fn,False)
            self.verdict(call(fn,'true'),True,kinds=['process'])

    def test_module_mutation(self):
        mutation=build(mod('os'),{'benign':g('os','system')})
        raw=payload(mutation)+payload(call(g('os','benign'),'true'))
        self.assertEqual([f.kind for f in scan(raw)],['process'])
        raw=payload(build(mod('os'),{'system':g('builtins','str')}))+payload(call(g('os','system'),'true'))
        self.assertFalse(scan(raw))

    def test_calling_opcodes(self):
        for obj in [Expr(e(g('subprocess','Popen'))+e((['true'],))+b'\x81'),Expr(e(g('subprocess','Popen'))+e((['true'],))+e({})+b'\x92'),Expr(b'(csubprocess\nPopen\no'),Expr(b'(isubprocess\nPopen\n')]:
            self.verdict(obj,False)
        self.verdict(Expr(b'('+e('true')+b'ios\nsystem\n'),True,kinds=['process'])
        self.verdict(Expr(b'('+e(g('os','system'))+e('true')+b'o'),True,kinds=['process'])
        partial=call(g('functools','partial'),g('os','system'),'true')
        self.verdict(Expr(b'('+e(partial)+b'o'),True,kinds=['process'])
        self.verdict(Expr(e(g('os','system'))+e(['true'])+b'R'),False)
        self.verdict(Expr(e(g('os','system'))+e(('true',))+b'\x81'),False)
        self.assertFalse(scan(b'Pmissing\n'+e(call(g('os','system'),'true'))+b'.'))

    def test_lazy_callbacks(self):
        for iterator in [call(g('builtins','map'),g('os','system'),['true']),call(g('builtins','filter'),g('os','system'),['true']),call(g('itertools','starmap'),g('os','system'),[('true',)])]:
            self.verdict(iterator,False)
            for consumer in ('list','tuple','next'):
                self.verdict(call(g('builtins',consumer),iterator),True,kinds=['process'])
        self.verdict(call(g('builtins','list'),call(g('builtins','map'),g('os','system'),[])),False)
        self.verdict(call(g('builtins','list'),call(g('builtins','map'),g('builtins','str'),[1,2])),False)
        for consumer in ('sorted','min','max'):
            self.verdict(kwcall(g('builtins',consumer),(['true'],),{'key':g('os','system')}),True,kinds=['process'])
        self.verdict(call(g('functools','reduce'),g('builtins','eval'),['1+1',{}]),True,kinds=['code-execution'])
        self.verdict(call(g('operator','call'),g('os','system'),'true'),True,kinds=['process'])

    def test_network_clients_and_files(self):
        for cls,args in [('socket.socket',()),('http.client.HTTPConnection',('host',80)),('ftplib.FTP',()),('smtplib.SMTP',()),('xmlrpc.client.ServerProxy',('http://host',))]:
            module,name=cls.rsplit('.',1)
            obj=call(g(module,name),*args)
            self.verdict(obj,False)
            method='request' if 'HTTPConnection' in cls else 'system' if 'ServerProxy' in cls else 'connect'
            self.verdict(call(attr(obj,method),'host'),True,kinds=['network'])
        for cls in ('ftplib.FTP','smtplib.SMTP'):
            module,name=cls.rsplit('.',1);self.verdict(call(g(module,name),'host'),True,kinds=['network'])
        path=call(g('pathlib','Path'),'/tmp/x')
        self.verdict(path,False)
        self.verdict(call(attr(path,'write_text'),'hello'),True,kinds=['file-access'])

    def test_threads(self):
        thread=kwcall(g('threading','Thread'),(),{'target':g('os','system'),'args':('true',)})
        self.verdict(thread,False)
        self.verdict(call(attr(thread,'start')),True,kinds=['process'])
        executor=call(g('concurrent.futures','ThreadPoolExecutor'),1)
        self.verdict(executor,False)
        self.verdict(call(attr(executor,'submit'),g('os','system'),'true'),True,kinds=['process'])

    def test_nested_loaders(self):
        evil=payload(call(g('os','system'),'true'))
        safe=payload({'fn':g('os','system'),'label':'eval('})
        for data,mal in [(evil,True),(safe,False),(safe+evil,False),(evil+safe,True)]:
            for module,name,arg in [('pickle','loads',data),('pickle','load',call(g('io','BytesIO'),data)),('pandas','read_pickle',call(g('io','BytesIO'),data)),('joblib','load',call(g('io','BytesIO'),data))]:
                self.verdict(call(g(module,name),arg),mal)
        for data in (safe,evil):
            for decoder in [call(g('base64','b64decode'),base64.b64encode(data)),call(g('zlib','decompress'),zlib.compress(data)),call(g('builtins','bytes.fromhex'),data.hex()),call(g('operator','add'),data[:5],data[5:]),call(g('builtins','bytes'),bytearray(data))]:
                self.verdict(call(g('pickle','loads'),decoder),data==evil)
        self.verdict(call(g('numpy','load'),'/tmp/x'),True,kinds=['file-access'])
        self.verdict(call(g('pickle','loads'),payload(g('os','system'))),False)
        self.verdict(call(call(g('pickle','loads'),payload(g('os','system'))),'true'),True,kinds=['process'])


class Containers(unittest.TestCase):
    def test_content_and_compression(self):
        for evil in (False,True):
            data=payload(call(g('os','system'),'true') if evil else {'label':'os.system'})
            for compress in (gzip.compress,bz2.compress,lzma.compress,zlib.compress):
                self.assertEqual(bool(scan(compress(data))),evil)
            self.assertEqual(bool(scan(gzip.compress(payload(None))+gzip.compress(data))),evil)
            for prefix in (b'',b'prefixed bytes here\x00'):
                self.assertEqual(bool(scan(zipbytes([('weird.txt',data)],prefix))),evil)
            f=io.BytesIO()
            with tarfile.open(fileobj=f,mode='w') as tf:
                wrapped=zipbytes([('model.bin',data)])
                info=tarfile.TarInfo('model.zip');info.size=len(wrapped);tf.addfile(info,io.BytesIO(wrapped))
            self.assertEqual(bool(scan(gzip.compress(f.getvalue()))),evil)

    def test_zip_duplicates(self):
        import warnings
        safe=payload(None);evil=payload(call(g('os','system'),'true'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertFalse(scan(zipbytes([('m',evil),('m',safe)],b'prefix')))
            self.assertTrue(scan(zipbytes([('m',safe),('m',evil)],b'prefix')))

    def test_torch_and_npy(self):
        for evil in (False,True):
            data=payload(call(g('os','system'),'true') if evil else {'weight':[1,2]})
            legacy=payload(pickle_scan.TORCH_MAGIC)+payload(1001)+payload({})+data+payload([])+b'raw storage bytes'
            self.assertEqual(bool(scan(legacy)),evil)
            self.assertEqual(bool(scan(payload(call(g('torch.storage','_load_from_bytes'),legacy)))),evil)
            self.assertEqual(bool(scan(zipbytes([('weights/data.pkl',data),('weights/version',b'3\n'),('weights/data/0',b'raw')]))),evil)
            hdr=repr({'descr':'|O','shape':(1,),'fortran_order':False}).encode()+b'\n'
            npy=b'\x93NUMPY\x01\x00'+struct.pack('<H',len(hdr))+hdr+data
            self.assertEqual(bool(scan(npy)),evil)
            self.assertEqual(bool(scan(zipbytes([('a.npy',npy)]))),evil)


class Graphs(unittest.TestCase):
    def test_tf_closed_table_and_scopes(self):
        from modelgate.tfproto.saved_model_pb2 import SavedModel
        from modelgate.policy import TF_OPS
        from google.protobuf import text_format
        for op in list(TF_OPS)+['SaveV2','RestoreV2','MergeV2Checkpoints','Const','pyfunc','ReadFileSuffix']:
            sm=SavedModel()
            sm.meta_graphs.add()
            meta=sm.meta_graphs.add()
            fn=meta.graph_def.library.function.add();fn.signature.name='inner'
            node=fn.node_def.add();node.name='ReadFile';node.op=op
            node.attr['payload'].s=b'PyFunc'
            for data,text in [(sm.SerializeToString(),False),(text_format.MessageToString(sm).encode(),True)]:
                fs=savedmodel_scan.scan_saved_model(data,'test',text=text)
                self.assertEqual(bool(fs),op in TF_OPS,(op,fs))

    def test_keras(self):
        for field in ('function','output_shape','activation'):
            for bytecode in (False,True):
                fn={'class_name':'__lambda__','config':{'code':'abc'}} if bytecode else {'module':'builtins','class_name':'function','config':'relu'}
                layer={'class_name':'Lambda' if field!='activation' else 'Dense','config':{field:fn}}
                cfg={'class_name':'Sequential','config':{'layers':[{'class_name':'TimeDistributed','config':{'layer':layer}}]}}
                self.assertEqual(bool(keras_scan.scan_config(cfg,'test')),bytecode)
        self.assertFalse(keras_scan.scan_config({'class_name':'Dense','config':{'name':'__lambda__','activation':'relu'}},'test'))


if __name__=='__main__': unittest.main()
