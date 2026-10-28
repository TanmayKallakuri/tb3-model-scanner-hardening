"""Policy regressions. Payloads are assembled as bytes, never unpickled."""
import sys
sys.dont_write_bytecode = True
import base64
import bz2
import gzip
import io
import json
import lzma
import os
import pickle  # dumps of literal test data only; never loads
import struct
import tarfile
import unittest
import warnings
import zipfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from modelgate.dispatch import scan_data
from modelgate.keras_scan import scan_config
from modelgate.pickle_scan import scan_pickle
from modelgate.policy import TF_OPS
from modelgate.savedmodel_scan import scan_saved_model


def g(path):
    module, name = path.rsplit(':', 1)
    return b'c' + module.encode() + b'\n' + name.encode() + b'\n'


def lit(value):
    # Protocol 3 produces no FRAME/MEMOIZE; each literal fragment has no
    # memo lookups into any other fragment.
    stream = io.BytesIO()
    writer = pickle.Pickler(stream, protocol=3)
    writer.fast = True
    writer.dump(value)
    data = stream.getvalue()
    return data[2:-1]


def tup(*parts): return b'(' + b''.join(parts) + b't'
def call(fn, *args): return fn + tup(*args) + b'R'
def p(fragment, proto=4): return b'\x80' + bytes([proto]) + fragment + b'.'
def attr(obj, name): return call(g('builtins:getattr'), obj, lit(name))
def imp(name): return call(g('importlib:import_module'), lit(name))
def effect(path='os:system', arg='echo never-executed'): return call(g(path), lit(arg))


def archive(entries, prefix=b''):
    out = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, data in entries: z.writestr(name, data)
    return prefix + out.getvalue()


def tar(entries):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w') as z:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            z.addfile(info, io.BytesIO(data))
    return out.getvalue()


def npy(payload, descr='|O', version=1, shape=(1,)):
    header = repr({'descr': descr, 'fortran_order': False, 'shape': shape}).encode('latin1') + b'\n'
    size = struct.pack('<H' if version == 1 else '<I', len(header))
    return b'\x93NUMPY' + bytes([version, 0]) + size + header + payload


class PolicyTests(unittest.TestCase):
    def assert_verdict(self, data, malicious, kind=None):
        findings = scan_data(data, 'artifact.bin')
        self.assertEqual(bool(findings), malicious, findings)
        if kind:
            self.assertIn(kind, {f.kind for f in findings}, findings)
            self.assertNotIn('unsupported', {f.kind for f in findings}, findings)
        return findings

    def test_inert_references_and_data(self):
        for proto in range(6):
            for value in [{'code': 'os.system', 'subprocess': 'eval(compile(...))'}, [], (), None, {'nested': [1, 'ReadFile']}]:
                with self.subTest(proto=proto, value=value): self.assert_verdict(pickle.dumps(value, proto), False)
        for path in ['os:system', 'builtins:eval', 'socket:socket', 'subprocess:Popen', 'pickle:loads']:
            self.assert_verdict(p(tup(g(path))), False)
            self.assert_verdict(p(g(path) + b'q\x00' + b'0' + lit('safe')), False)

    def test_calls_and_allocations(self):
        for path, arg, kind in [('posix:system', 'x', 'process'), ('_io:open', '/external', 'file-access'),
                                ('builtins:eval', '1', 'code-execution'), ('socket:create_connection', ('host',80), 'network')]:
            with self.subTest(path=path):
                self.assert_verdict(p(effect(path,arg)), True, kind)
                self.assert_verdict(p(g(path) + tup(lit(arg)) + b'\x81'), False)
                self.assert_verdict(p(g(path) + tup(lit(arg)) + lit({}) + b'\x92'), False)
                self.assert_verdict(p(b'(' + g(path) + b'o'), False)
                module, name = path.split(':')
                self.assert_verdict(p(b'(i' + module.encode() + b'\n' + name.encode() + b'\n'), False)
                self.assert_verdict(p(b'(' + g(path) + lit(arg) + b'o'), True, kind)
                self.assert_verdict(p(b'(' + lit(arg) + b'i' + module.encode() + b'\n' + name.encode() + b'\n'), True, kind)
        self.assert_verdict(p(g('subprocess:Popen') + b')R'), True, 'process')

    def test_memo_and_stack_globals(self):
        for put, get in [(b'q\x07',b'h\x07'),(b'r'+struct.pack('<I',999),b'j'+struct.pack('<I',999)),(b'p12\n',b'g12\n')]:
            self.assert_verdict(p(g('os:system')+put+b'0'+call(get,lit('x'))),True,'process')
        self.assert_verdict(p(lit('os')+b'q\x0c0'+lit('system')+b'q\x0d0'+b'h\x0ch\x0d\x93'+tup(lit('x'))+b'R'),True,'process')
        self.assert_verdict(p(g('os:system') + b'q\x000' + lit('data') + b'q\x000' + b'h\x00'),False)
        # MEMOIZE uses len(memo), not last explicit index + 1.
        self.assert_verdict(p(lit('unused')+b'q\x090'+g('os:system')+b'\x940'+call(b'h\x01',lit('x'))),True,'process')
        self.assert_verdict(p(g('os:system')+b'2'+tup(lit('x'))+b'R'),True,'process')

    def test_python2_mappings(self):
        for path, arg, kind in [('__builtin__:eval','1','code-execution'),('commands:getoutput','x','process'),
                                ('urllib:urlopen','http://example.invalid','network'),('urllib2:urlopen','http://example.invalid','network'),
                                ('httplib:HTTPConnection.connect','x','network'),('cPickle:loads',p(effect()),'process')]:
            self.assert_verdict(p(effect(path,arg),2),True,kind)
        self.assert_verdict(p(call(g('__builtin__:xrange'), lit(10)),2),False)

    def test_indirect_calls(self):
        for fn in [attr(imp('os'),'system'),call(call(g('operator:attrgetter'),lit('system')),imp('os')),
                   call(g('functools:partial'),g('os:system'),lit('x'))]:
            self.assert_verdict(p(fn),False)
            self.assert_verdict(p(call(fn,lit('x'))),True,'process')
        fn=call(g('operator:methodcaller'),lit('system'),lit('x'))
        self.assert_verdict(p(call(fn,imp('os'))),True,'process')
        self.assert_verdict(p(call(attr(call(g('builtins:__import__'),lit('os')),'system'),lit('x'))),True,'process')
        target=call(g('operator:getitem'),attr(imp('builtins'),'__dict__'),lit('eval'))
        self.assert_verdict(p(call(target,lit('1+1'))),True,'code-execution')
        self.assert_verdict(p(call(call(call(g('operator:itemgetter'),lit('f')),b'}'+lit('f')+g('os:system')+b's'),lit('x'))),True,'process')

    def test_partial_build_and_alias(self):
        for initial, final, expected in [('builtins:len','os:system',True),('os:system','builtins:len',False)]:
            obj = call(g('functools:partial'),g(initial)) + b'p99\n'
            state = tup(g(final),tup(lit('x')),lit({}),lit({}))
            data = p(obj + state + b'b0' + call(b'g99\n'))
            self.assert_verdict(data,expected,'process' if expected else None)
            self.assert_verdict(p(obj + state + b'b'),False)
        obj = g('functools:partial') + tup(g('os:system')) + b'\x81'
        self.assert_verdict(p(call(obj,lit('x'))),True,'process')

    def test_network_constructors(self):
        for path, args in [('socket:socket',()),('http.client:HTTPConnection',('localhost',80)),('ftplib:FTP',()),
                           ('ftplib:FTP_TLS',()),('smtplib:SMTP',()),('xmlrpc.client:ServerProxy',('http://localhost',))]:
            value=call(g(path),*(lit(a) for a in args))
            self.assert_verdict(p(value),False)
        for path in ['ftplib:FTP','ftplib:FTP_TLS','smtplib:SMTP','smtplib:SMTP_SSL']:
            self.assert_verdict(p(call(g(path),lit('localhost'))),True,'network')
        for cls, method, ctor in [('socket:socket','connect',()),('http.client:HTTPConnection','request',('localhost',)),
                                  ('ftplib:FTP','connect',()),('xmlrpc.client:ServerProxy','ping',('http://localhost',))]:
            obj=call(g(cls),*(lit(a) for a in ctor))
            self.assert_verdict(p(call(attr(obj,method),lit('target'))),True,'network')

    def test_serialized_code(self):
        code=call(g('marshal:loads'),lit(b'opaque code data'))
        self.assert_verdict(p(code),False)
        self.assert_verdict(p(call(g('types:FunctionType'),code,lit({}))),True,'code-execution')

    def test_nested_loaders(self):
        for malicious in [False,True]:
            nested=p(effect()) if malicious else pickle.dumps({'safe':'os.system'},4)
            self.assert_verdict(p(call(g('pickle:loads'),lit(nested))),malicious,'process' if malicious else None)
            encoded=call(g('base64:b64decode'),lit(base64.b64encode(nested).decode()))
            self.assert_verdict(p(call(g('_pickle:loads'),encoded)),malicious,'process' if malicious else None)
            for loader in ['pickle:load','joblib:load','pandas:read_pickle','torch:load']:
                buf=call(g('io:BytesIO'),lit(nested))
                self.assert_verdict(p(call(g(loader),buf)),malicious,'process' if malicious else None)
            self.assert_verdict(p(call(g('numpy:load'),call(g('io:BytesIO'),lit(npy(nested))),lit(None),lit(True))),malicious,'process' if malicious else None)
            self.assert_verdict(p(call(g('numpy.lib.format:read_array'),call(g('io:BytesIO'),lit(npy(nested))),lit(True))),malicious,'process' if malicious else None)
        for loader in ['pickle:load','joblib:load','pandas:read_pickle','numpy:load','torch:load']:
            self.assert_verdict(p(call(g(loader),lit('/external/file'))),True,'file-access')
        # pickle.loads loads one object, not trailing bytes.
        self.assert_verdict(p(call(g('pickle:loads'),lit(p(lit(0))+p(effect())))),False)
        # A returned global remains inert until the outer pickle calls it.
        self.assert_verdict(p(call(call(g('pickle:loads'),lit(p(g('os:system')))),lit('x'))),True,'process')

    def test_stream_sequence_and_legacy(self):
        self.assert_verdict(p(lit(0))+p(effect()),True,'process')
        self.assert_verdict(pickle.dumps({'first':'thing'},4)+pickle.dumps({'second':'other'},4),False)
        prefix=b''.join(pickle.dumps(v,2) for v in [0x1950A86A20F9469CFC6C,1001,{'x':'y'}])
        self.assert_verdict(prefix+p(lit({'tensor':'data'}))+pickle.dumps(['storage'],2)+p(effect()),False)
        self.assert_verdict(prefix+p(effect())+pickle.dumps([],2),True,'process')

    def test_containers(self):
        for malicious in [False,True]:
            raw=p(effect()) if malicious else p(lit({'text':'exec('}))
            wrappers=[gzip.compress(raw),bz2.compress(raw),lzma.compress(raw),zlib.compress(raw),
                      archive([('model.unknown',raw)]),tar([('model.bin',raw)]),
                      gzip.compress(tar([('model.dat',bz2.compress(raw))])),
                      archive([('model.pkl',raw),('card.json',b'{"data":"os.system"}')],prefix=p(lit(3)))]
            for data in wrappers: self.assert_verdict(data,malicious,'process' if malicious else None)
        self.assert_verdict(gzip.compress(p(lit(0)))+gzip.compress(p(effect())),True,'process')
        self.assert_verdict(archive([('same.pkl',p(effect())),('same.pkl',p(lit(0)))]),False)
        self.assert_verdict(archive([('same.pkl',p(lit(0))),('same.pkl',p(effect()))]),True,'process')
        # The central directory overrides an executable-looking ZIP prefix.
        self.assert_verdict(archive([('data.pkl',p(lit(0)))],prefix=p(effect())),False)
        # In a real checkpoint the numeric storage is not a model member.
        self.assert_verdict(archive([('x/data.pkl',p(lit(0))),('x/version',b'3'),('x/data/0',p(effect()))]),False)

    def test_numpy(self):
        for version in [1,2,3]:
            for dtype in ['|O',[('text','<U3'),('nested',[('value','|O')])]]:
                self.assert_verdict(npy(p(effect()),dtype,version),True,'process')
                self.assert_verdict(npy(p(lit(['os.system'])),dtype,version),False)
            self.assert_verdict(npy(p(effect()),'|u1',version,shape=(len(p(effect())),)),False)
        self.assert_verdict(archive([('one.npy',npy(p(lit(0)))),('two.npy',npy(p(effect())))]),True,'process')

    def test_keras_recursive_and_named(self):
        code={'class_name':'Lambda','config':{'function':{'class_name':'__lambda__','config':{'code':'base64','defaults':None,'closure':None}}}}
        named={'class_name':'Lambda','config':{'function':{'module':'builtins','class_name':'function','config':'relu'}}}
        for layer,expected in [(code,True),(named,False),({'class_name':'Activation','config':{'activation':'relu'}},False)]:
            config={'class_name':'Sequential','config':{'layers':[{'class_name':'TimeDistributed','config':{'layer':{'class_name':'Sequential','config':{'layers':[layer]}}}}]}}
            self.assertEqual(bool(scan_config(config,'config')),expected)
            self.assert_verdict(archive([('config.json',json.dumps(config).encode()),('metadata.json',b'{}'),('model.weights.h5',b'')]),expected,'code-execution' if expected else None)


    def test_additional_indirection(self):
        # A builtin used as getattr's default, and a function-global dictionary.
        default = call(g('builtins:getattr'), imp('builtins'), lit('absent_attribute'), g('builtins:eval'))
        self.assert_verdict(p(call(default, lit('1+1'))), True, 'code-execution')
        namespace = attr(g('collections:namedtuple'), '__globals__')
        builtin_dict = call(g('operator:getitem'), namespace, lit('__builtins__'))
        fn = call(g('operator:getitem'), builtin_dict, lit('eval'))
        self.assert_verdict(p(call(fn, lit('1+1'))), True, 'code-execution')
        # Calling __init__ explicitly on an allocated Popen executes it.
        obj = g('subprocess:Popen') + b')\x81'
        self.assert_verdict(p(call(attr(obj, '__init__'), lit(['echo', 'never-run']))), True, 'process')
        # A state hook assembled as data; type() itself executes no payload.
        fields = b'}' + lit('__setstate__') + g('builtins:eval') + b's'
        cls = call(g('builtins:type'), lit('Stateful'), tup(g('builtins:object')), fields)
        self.assert_verdict(p(cls), False)
        self.assert_verdict(p(cls + b')\x81' + lit('1+1') + b'b'), True, 'code-execution')

    def test_loader_stopping_rules_and_return_values(self):
        source = p(lit(0)) + p(effect())
        for loader in ['pickle:load', 'joblib:load', 'pandas:read_pickle']:
            buf = call(g('io:BytesIO'), lit(source))
            self.assert_verdict(p(call(g(loader), buf)), False)
        for loader in ['joblib:load', 'pandas:read_pickle']:
            inner = call(g(loader), call(g('io:BytesIO'), lit(p(g('os:system')))))
            self.assert_verdict(p(call(inner, lit('x'))), True, 'process')
        for loader, suffix in [('numpy:load',(lit(None),lit(False))), ('numpy.lib.format:read_array',(lit(False),))]:
            self.assert_verdict(p(call(g(loader), call(g('io:BytesIO'), lit(npy(p(effect())))), *suffix)), False)
        zipped = archive([('x.npy', npy(p(effect())))])
        lazy = call(g('numpy:load'), call(g('io:BytesIO'),lit(zipped)),lit(None),lit(True))
        self.assert_verdict(p(lazy), False)
        self.assert_verdict(p(call(g('operator:getitem'), lazy, lit('x'))), True, 'process')
        # A loader reads at the current position of a memoized memory stream.
        buf = call(g('io:BytesIO'),lit(source)) + b'p99\n0'
        first = call(g('pickle:load'),b'g99\n') + b'0'
        second = call(g('pickle:load'),b'g99\n')
        self.assert_verdict(p(buf + first + second), True, 'process')
        # The torch loader requires ZIP's local-header magic at offset zero.
        cleanzip = archive([('x/data.pkl',p(effect())),('x/version',b'3')],prefix=p(lit(0)))
        self.assert_verdict(p(call(g('torch:load'),call(g('io:BytesIO'),lit(cleanzip)))),False)

    def test_embedded_archives_are_data(self):
        zipped = archive([('evil.pkl',p(effect()))])
        self.assert_verdict(p(lit(zipped)),False)
        self.assert_verdict(p(lit(zipped)+b'0'+effect()),True,'process')
        self.assert_verdict(npy(p(lit(zipped))),False)
        self.assert_verdict(npy(zipped,'u1',shape=(len(zipped),)),False)
        for compress in [lambda x:gzip.compress(x,compresslevel=0),bz2.compress,lzma.compress]:
            self.assert_verdict(compress(p(lit(zipped))),False)
        self.assert_verdict(archive([('data.npy',npy(zipped,'u1',shape=(len(zipped),)))]),False)

    def test_joblib_inline_arrays(self):
        def wrapper(dtype, size, alignment=None):
            fields = b'}' + lit('shape') + lit((size,)) + b's'
            fields += lit('dtype') + call(g('numpy:dtype'),lit(dtype)) + b's'
            if alignment is not None: fields += lit('numpy_array_alignment_bytes') + lit(alignment) + b's'
            return g('joblib.numpy_pickle:NumpyArrayWrapper') + b')\x81' + fields + b'b'
        # Raw numerical data can contain bytes that are valid malicious pickles.
        raw = p(effect())
        for align in [None,16]:
            padding = b'\x03\xff\xff\xff' if align else b''
            obj = wrapper('u1',len(raw),align) + padding + raw
            self.assert_verdict(p(obj),False)
            self.assert_verdict(p(obj+b'0'+effect()),True,'process')
        self.assert_verdict(p(wrapper('O8',1)+p(effect())),True,'process')
        self.assert_verdict(p(wrapper('O8',1)+p(lit(['safe']))),False)

    def test_callable_factory_is_not_execution(self):
        for fn in [call(g('timeit:Timer'),g('builtins:len')), call(g('timeit:Timer'))]:
            self.assert_verdict(p(fn),False)
        self.assert_verdict(p(call(g('timeit:Timer'),lit('pass'))),True,'code-execution')
        timer = call(g('timeit:Timer'),call(g('functools:partial'),g('os:system'),lit('x')))
        self.assert_verdict(p(timer),False)
        self.assert_verdict(p(call(attr(timer,'timeit'))),True,'process')
        safe = {'version':1,'handlers':{'sink':{'class':'logging.NullHandler'}},'root':{'handlers':['sink']}}
        self.assert_verdict(p(call(g('logging.config:dictConfig'),lit(safe))),False)
        bad = {'version':1,'handlers':{'sink':{'class':'logging.FileHandler','filename':'/external'}}}
        self.assert_verdict(p(call(g('logging.config:dictConfig'),lit(bad))),True,'file-access')

    def test_tensorflow_closed_table(self):
        from modelgate.tfproto.saved_model_pb2 import SavedModel
        from google.protobuf import text_format
        for op, kind in list(TF_OPS.items()) + [(n,None) for n in ['Const','ReadFileV2','readfile','SaveV2','RestoreV2','MergeV2Checkpoints','ShardedFilename','ShardedFilespec']]:
            for placement in ['graph','function']:
                sm=SavedModel(saved_model_schema_version=1)
                sm.meta_graphs.add()
                mg=sm.meta_graphs.add()
                if placement=='graph': node=mg.graph_def.node.add()
                else:
                    fn=mg.graph_def.library.function.add()
                    fn.signature.name='branch'
                    node=fn.node_def.add()
                    callnode=mg.graph_def.node.add(name='if',op='StatelessIf')
                    callnode.attr['then_branch'].func.name='branch'
                node.name='ReadFile'
                node.op=op
                node.attr['value'].tensor.string_val.append(b'PyFunc')
                for text in [False,True]:
                    raw=text_format.MessageToString(sm).encode() if text else sm.SerializeToString()
                    findings=scan_saved_model(raw,'saved_model.pbtxt' if text else 'saved_model.pb',text)
                    self.assertEqual({f.kind for f in findings},{kind} if kind else set(),(op,placement,text,findings))


if __name__ == '__main__': unittest.main()
