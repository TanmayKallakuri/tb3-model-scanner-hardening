"""Boundaries where a capability exists but does not execute, and vice versa."""
import datetime
import functools
import io
import marshal
import pickle
import re
import struct
import unittest
import zlib

from modelgate.pickle_scan import scan_pickle, TORCH_MAGIC
from modelgate.dispatch import scan_bytes
from test_scanner import G, Call, New, Build, Memo, Get, stream, encode, imp, attr, partial, method, archive, npy, EVIL, SAFE


class ReduceFixture:
    def __init__(self, fn, args): self.fn, self.args = fn, args
    def __reduce__(self): return self.fn, self.args


class EdgeTests(unittest.TestCase):
    def check(self, value, evil, kind=None):
        result=scan_pickle(stream(value),'test')
        self.assertEqual(bool(result),evil,result)
        if evil:
            self.assertNotEqual(result[0].kind,'unsupported',result)
            if kind:self.assertEqual(result[0].kind,kind,result)

    def test_stream_buffers_serialized_normally(self):
        for evil in (False,True):
            data=stream(EVIL if evil else SAFE)
            obj=ReduceFixture(pickle.load,(io.BytesIO(data),))
            for proto in range(2,6):
                result=scan_pickle(pickle.dumps(obj,protocol=proto),'io-buffer')
                self.assertEqual(bool(result),evil,result)
                if evil:self.assertEqual(result[0].kind,'process',result)
        self.check(Call(G('pickle','load'),(Build(New(G('io','BytesIO')), (stream(EVIL),0,None)),)),True)

    def test_partial_identity(self):
        # A partial with an empty instance dict is flattened at construction.
        self.check([Memo(partial(G('builtins','str')),0),Memo(partial(Get(0)),1),Build(Get(0),(G('os','system'),('true',),{},None)),Call(Get(1))],False)
        self.check([Memo(partial(G('os','system'),'true'),0),Memo(partial(Get(0)),1),Build(Get(0),(G('builtins','str'),(),{},None)),Call(Get(1))],True)
        # A partial with user attributes is retained by reference instead.
        self.check([Memo(Build(partial(G('builtins','str')),(G('builtins','str'),(),{}, {'note':1})),0),Memo(partial(Get(0)),1),Build(Get(0),(G('os','system'),('true',),{},None)),Call(Get(1))],True)
        # Keywords and namespace passed by BUILD retain identity.
        self.check([Memo({},0),Memo(Build(partial(G('builtins','sorted')), (G('builtins','sorted'),(['true'],),Get(0),None)),1),Call(G('operator','setitem'),(Get(0),'key',G('os','system'))),Call(Get(1))],True)
        self.check([Memo(Build(partial(G('builtins','str')),(G('builtins','str'),(),{}, {'__setstate__':G('os','system')})),0),Build(Get(0),'true')],True)

    def test_copied_module_dicts(self):
        for fn in (G('builtins','dict'), G('copy','copy')):
            module_dict=Call(G('builtins','vars'),(imp('os'),))
            self.check(Call(Call(G('operator','getitem'),(Call(fn,(module_dict,)),'system')),('true',)),True)
        self.check(Call(method(Call(G('builtins','vars'),(imp('builtins'),)),'copy'),()),False)

    def test_frames(self):
        body=encode(SAFE)+b'.'+stream(EVIL)
        framed=b'\x80\x04\x95'+struct.pack('<Q',len(body))+body
        self.assertFalse(scan_pickle(framed,'frame'))
        self.assertTrue(scan_pickle(framed+stream(EVIL),'frame'))
        self.assertFalse(scan_pickle(b'\x80\x04\x95'+struct.pack('<Q',len(body)+100)+body,'truncated-frame'))

    def test_unpickler_memo(self):
        data=stream(Memo(G('os','system'),0))+stream(Call(Get(0),('true',)))
        unpickler=Call(G('pickle','Unpickler'),(Call(G('io','BytesIO'),(data,)),))
        self.check([Memo(unpickler),method(Get(),'load'),method(Get(),'load')],True)
        # Separate pickle.load calls do not share a memo.
        self.check([Memo(Call(G('io','BytesIO'),(data,))),Call(G('pickle','load'),(Get(),)),Call(G('pickle','load'),(Get(),))],False)

    def test_nested_loader_boundaries(self):
        evil=stream(EVIL)
        self.check(Call(G('joblib','load'),(Call(G('io','BytesIO'),(stream(SAFE)+evil,)),)),False)
        compressed=zlib.compress(stream(SAFE)+b'ignored trailing bytes'*100)
        self.check([Call(G('joblib','load'),(Call(G('io','BytesIO'),(compressed,)),)),EVIL],True)
        self.check(Call(G('numpy','load'),(Call(G('io','BytesIO'),(npy(evil),)),)),False)
        self.check(Call(partial(G('numpy','load'),Call(G('io','BytesIO'),(npy(evil),)),allow_pickle=True)),True)
        self.check(Call(G('numpy.lib.format','read_array'),(Call(G('io','BytesIO'),(npy(evil),)),True)),True)
        self.check(Call(G('numpy.lib.format','read_array'),(Call(G('io','BytesIO'),(npy(evil),)),False)),False)
        self.check(Call(G('torch','load'),(Call(G('io','BytesIO'),(stream(SAFE)+evil,)),)),False)
        legacy=stream(TORCH_MAGIC)+stream(1000)+stream(EVIL)
        self.check(Call(G('torch.storage','_load_from_bytes'),(legacy,)),False)
        npz=archive([('a.npy',npy(evil))])
        value=Call(partial(G('numpy','load'),Call(G('io','BytesIO'),(npz,)),allow_pickle=True))
        self.check(value,False)
        self.check(Call(G('operator','getitem'),(value,'a')),True)
        for evil in (False,True):
            payload=stream(EVIL if evil else SAFE)
            self.check(Call(G('pickle','loads'),(Call(G('marshal','loads'),(marshal.dumps(payload),)),)),evil)

    def test_iterator_boundaries(self):
        mapped=Call(G('builtins','map'),(G('os','system'),['true']))
        tee=Call(G('itertools','tee'),(mapped,))
        self.check(tee,False)
        self.check(Call(G('builtins','next'),(Call(G('operator','getitem'),(tee,0)),)),True)
        self.check(Call(G('itertools','product'),(mapped,)),True)
        self.check(Call(G('itertools','permutations'),(mapped,)),True)
        self.check(Call(G('builtins','list'),(Call(G('itertools','islice'),(mapped,0)),)),False)
        self.check(Call(G('builtins','list'),(Call(G('itertools','islice'),(mapped,1,0)),)),True)
        self.check(Call(G('builtins','all'),([False,mapped],)),False)
        self.check(New(G('builtins','list'),(mapped,)),False)
        self.check(New(G('builtins','tuple'),(mapped,)),True)
        self.check(Call(G('builtins','tuple.__new__'),(G('builtins','tuple'),mapped)),True)
        self.check(Call(G('functools','partial.__new__'),(G('functools','partial'),G('os','system'),'true')),False)
        self.check(Call(Call(G('functools','partial.__new__'),(G('functools','partial'),G('os','system'),'true'))),True)

    def test_stdlib_metadata(self):
        values=[datetime.datetime(2020,1,1),datetime.timezone.utc,re.compile('os.system'),io.BytesIO(b'os.system'),io.StringIO('eval(')]
        for value in values:
            for proto in range(2,6):
                with self.subTest(type=type(value).__name__,protocol=proto):
                    self.assertFalse(scan_pickle(pickle.dumps(value,proto),'stdlib'))
        self.check(Call(G('timeit','Timer')),False)
        self.check(Call(G('timeit','Timer'),(partial(G('os','system'),'true'),)),False)
        self.check(method(Call(G('timeit','Timer'),(partial(G('os','system'),'true'),)),'timeit'),True)
        self.check(Call(G('copyreg','_reconstructor'),(G('subprocess','Popen'),G('subprocess','Popen'),['echo'])),True)
        self.check(Call(G('types','MethodType'),(G('os','system'),'true')),False)
        self.check(Call(Call(G('types','MethodType'),(G('os','system'),'true'))),True)
        self.check(Call(Call(G('types','MethodType'),(G('builtins','str'),'text'))),False)

    def test_policy_sinks(self):
        calls={
            'process': [('os','execve',('/bin/echo',['echo'],{})),('os','spawnv',(0,'/bin/echo',['echo'])),('os','posix_spawn',('/bin/echo',['echo'],{})),('subprocess','check_call',(['echo'],)),('pty','spawn',(['echo'],)),('pydoc','pipepager',('text','cat'))],
            'file-access':[('io','open',('/outside',)),('os','rename',('/a','/b')),('shutil','copy',('/a','/b')),('shutil','move',('/a','/b')),('os','makedirs',('/outside',)),('os','rmdir',('/outside',)),('numpy._core.multiarray','fromfile',('/outside',))],
            'network':[('socket','create_connection',(('example.invalid',80),)),('urllib.request','urlopen',('https://example.invalid',)),('ssl','get_server_certificate',(('example.invalid',443),)),('webbrowser','open',('https://example.invalid',))],
            'code-execution':[('builtins','compile',('1+1','<test>','eval')),('runpy','run_path',('/outside',)),('runpy','_run_code',('1+1',{})),('timeit','timeit',('1+1',)),('pdb','run',('1+1',)),('numpy.testing._private.utils','runstring',('1+1',{}))]
        }
        for kind, entries in calls.items():
            for module,name,args in entries:
                with self.subTest(module=module,name=name):
                    self.check(Call(G(module,name),args),True,kind)
                    self.check(G(module,name),False)

    def test_archive_metadata(self):
        for text in (b'Model description',b'Contains os.system text',b'Graph metadata',b'ImmutableConst is an op name',b'Saved model'):
            self.assertFalse(scan_bytes(archive([('README.md',text),('m.pkl',stream(SAFE))]),'notes.zip'))

    def test_python2_fix_imports(self):
        for proto in range(6):
            for module,name in [('__builtin__','eval'),('commands','getoutput')]:
                result=scan_pickle(stream(Call(G(module,name),('1+1',)),proto),'python2')
                self.assertEqual(bool(result),proto<3,result)
            value=Call(G('builtins','list'),(Call(G('itertools','imap'),(G('os','system'),['true'])),))
            self.assertEqual(bool(scan_pickle(stream(value,proto),'imap')),proto<3)

    def test_joblib_array_buffers(self):
        for object_array in (False,True):
            for evil in (False,True):
                state={'subclass':G('numpy','ndarray'),'shape':(10,), 'order':'C',
                       'dtype':Call(G('numpy','dtype'),('O' if object_array else 'u1',)),
                       'allow_mmap':True,'numpy_array_alignment_bytes':16}
                data=b'\x80\x04'+encode(Build(New(G('joblib.numpy_pickle','NumpyArrayWrapper')),state))
                data += stream(EVIL if evil else SAFE) if object_array else b'\x02\xff\xffrawnumeric'
                if evil and not object_array:data+=b'0'+encode(EVIL)
                result=scan_pickle(data+b'.','joblib')
                self.assertEqual(bool(result),evil,result)
                if evil:self.assertEqual(result[0].kind,'process')

    def test_numpy_metadata(self):
        import numpy as np
        values=[np.arange(12),np.array([{'os.system':'subprocess'}],dtype=object),
                np.dtype([('label','O'),('weight','f8')]),np.float32(1),np.random.default_rng(3),
                np.ma.array([1,2,3],mask=[0,1,0])]
        for value in values:
            for proto in range(6):
                with self.subTest(type=type(value).__name__,protocol=proto):
                    self.assertFalse(scan_pickle(pickle.dumps(value,proto),'numpy'))


if __name__=='__main__':unittest.main()
