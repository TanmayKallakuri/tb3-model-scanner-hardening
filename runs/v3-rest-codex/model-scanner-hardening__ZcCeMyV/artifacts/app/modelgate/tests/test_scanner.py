import base64
import bz2
import gzip
import io
import json
import lzma
import os
import struct
import tarfile
import tempfile
import unittest
import zipfile
import zlib

from modelgate import dispatch,keras_scan,savedmodel_scan
from modelgate.symbolic import Runtime,Detected,Halt,Unsupported
from modelgate.tests.builders import *


def findings(data,loader=None,**opts):
    rt=Runtime();rt.load_callback=dispatch.load_content
    try:
        dispatch.load_content(data,'artifact',rt,loader,opts)
    except Detected as exc: return [exc.finding]
    except Halt: return []
    except Unsupported as exc: raise AssertionError('unexpected unsupported result: '+str(exc)) from exc
    return []


def archive(members,prefix=b''):
    buf=io.BytesIO(prefix)
    buf.seek(0,2)
    with zipfile.ZipFile(buf,'w') as z:
        for name,data in members: z.writestr(name,data)
    return buf.getvalue()


class Pickles(unittest.TestCase):
    def verdict(self,obj,malicious=True,proto=4,loader=None):
        result=findings(dump(obj,proto),loader)
        self.assertEqual(bool(result),malicious,(obj,result))

    def test_data(self):
        for p in range(6):
            self.verdict({'os.system':G('os.system'),'subprocess':[G('builtins.eval'),'exec(']},False,p)

    def test_direct_and_memo(self):
        for p in range(6):
            for fn,arg in [('os.system','echo no'),('builtins.eval','1+1'),('builtins.open','/outside'),('socket.create_connection',('localhost',80))]:
                self.verdict(Call(Memo(G(fn)),(arg,)),True,p)
                self.verdict([Memo(G(fn)),Call(Get(),(arg,))],True,p)

    def test_stack_global(self):
        self.verdict(Raw(emit(Memo('os',1))+emit(Memo('system',2))+b'00'+emit(Get(1))+emit(Get(2))+b'\x93'+emit(('x',))+b'R'))

    def test_indirections(self):
        funcs=[attr(C('builtins.__import__','os'),'system'),
               attr(C('importlib.import_module','os'),'system'),
               Call(C('operator.attrgetter','system'),(C('importlib.import_module','os'),)),
               Call(C('operator.attrgetter','os.system'),(C('importlib.import_module','shutil'),)),
               C('pkgutil.resolve_name','os:system'),C('pkgutil.resolve_name','os.system'),
               C('operator.getitem',C('builtins.vars',C('importlib.import_module','os')),'system'),
               C('operator.getitem',attr(C('importlib.import_module','sys'),'modules'),'os')]
        funcs[-1]=attr(funcs[-1],'system')
        for fn in funcs:
            self.verdict(fn,False)
            self.verdict(Call(fn,('x',)))
        for wrapper in ('copy.copy','copy.deepcopy','functools.cache'):
            self.verdict(Call(C(wrapper,G('os.system')),('x',)))
        self.verdict(Call(Call(C('functools.lru_cache',None),(G('os.system'),)),('x',)))

    def test_partial_and_retarget(self):
        partial=C('functools.partial',G('os.system'),'x')
        self.verdict(partial,False)
        self.verdict(Call(partial))
        self.verdict(Call(C('operator.methodcaller','system','x'),(C('importlib.import_module','os'),)))
        for fn in ('builtins.str','os.system'):
            p=Build(Memo(C('functools.partial',G('builtins.str')),9),(G(fn),('x',),{},None))
            self.verdict([p,Call(Get(9))],fn=='os.system')

    def test_module_build(self):
        mod=C('importlib.import_module','math')
        for fn in ('builtins.str','os.system'):
            self.verdict([Build(mod,{'sqrt':G(fn)}),C('math.sqrt','x')],fn=='os.system')
        # A state setter installed by one BUILD is called by the next BUILD.
        self.verdict(Build(Build(C('types.SimpleNamespace'),{'__setstate__':G('builtins.eval')}),'1+1'))

    def test_streams(self):
        clean=dump({'data':'os.system'})
        bad=dump(C('os.system','x'))
        self.assertTrue(findings(clean+bad))
        self.assertFalse(findings(clean+bad,'pickle.loads'))
        # Imported module mutation survives a new Unpickler and memo.
        first=dump(Build(C('importlib.import_module','math'),{'sqrt':G('os.system')}))
        self.assertTrue(findings(first+dump(C('math.sqrt','x'))))

    def test_bad_reduce_arguments(self):
        for arg in ('x',['x'],{'x':1},None):
            self.verdict(Call(G('os.system'),arg),False)
        for fn in ('builtins.eval','os.system','subprocess.Popen'):
            self.verdict(New(G(fn)),False)
        # A deterministic error prevents a later stream from being loaded.
        self.assertFalse(findings(dump(Call(G('os.system'),[]))+dump(C('os.system','x'))))

    def test_inst_obj_new(self):
        for is_obj in (False,True):
            self.verdict(Inst(G('subprocess.Popen'),(),is_obj),False,0)
            self.verdict(Inst(G('subprocess.Popen'),(['x'],),is_obj),True,0)
            self.verdict(Inst(G('os.system'),('x',),is_obj),True,0)
            self.verdict(Inst(G('os.system'),(),is_obj),False,0)
        self.verdict(Inst(C('functools.partial',G('os.system'),'x'),(),True),True,0)
        self.verdict(Call(New(G('subprocess.Popen')),()),False)
        self.verdict(New(G('subprocess.Popen'),(['x'],)),False)

    def test_dotted_globals(self):
        for p in range(6):
            self.verdict(Call(Global('shutil','os.system'),('x',)),p>=4,p)
            self.verdict(Call(Global('bdb','Bdb.run'),(None,'1+1')),p>=4,p)

    def test_aliases(self):
        for mod,name in [('posix','system'),('nt','system'),('__builtin__','eval'),('commands','getoutput')]:
            self.verdict(Call(Global(mod,name),('x',)),True,2)
        self.verdict(Call(Global('urllib2','urlopen'),('http://example',)),True,2)
        self.verdict(C('_io.open','/outside'))
        self.verdict(C('_pickle.loads',dump(C('os.system','x'))))

    def test_iterators(self):
        for fn in ('builtins.map','builtins.filter','itertools.filterfalse','itertools.starmap'):
            seq=[('x',)] if fn.endswith('starmap') else ['x']
            lazy=C(fn,G('os.system'),seq)
            self.verdict(lazy,False)
            self.verdict(C('builtins.list',lazy))
            self.verdict(C('builtins.next',lazy))
        self.verdict(C('builtins.list',C('builtins.map',G('os.system'),[])),False)
        for fn in ('builtins.sorted','builtins.min','builtins.max'):
            self.verdict(Call(C('functools.partial',G(fn)),(['x'],)),False)
            self.verdict(Call(Build(C('functools.partial',G(fn)),(G(fn),(),{'key':G('os.system')},None)),(['x'],)))
        self.verdict(C('functools.reduce',G('builtins.eval'),[{}],'1+1'))
        self.verdict(C('operator.call',G('os.system'),'x'))
        self.verdict(C('builtins.list',C('itertools.chain',[1],C('builtins.map',G('os.system'),['x']))))

    def test_nested(self):
        bad=dump(C('os.system','x'))
        good=dump({'data':'eval('})
        transforms=[lambda b:b,lambda b:C('base64.b64decode',base64.b64encode(b)),
                    lambda b:C('zlib.decompress',zlib.compress(b)),
                    lambda b:Call(Global('builtins','bytes.fromhex'),(b.hex(),)),
                    lambda b:C('operator.add',b[:9],b[9:]),lambda b:bytearray(b)]
        for trans in transforms:
            for payload,mal in ((bad,True),(good,False)):
                self.verdict(C('pickle.loads',trans(payload)),mal)
                self.verdict(C('pickle.load',C('io.BytesIO',trans(payload))),mal)
        self.verdict(Call(C('pickle.loads',dump(G('os.system'))),('x',)))
        self.verdict(C('pickle.loads',good+bad),False)
        for loader in ('torch.load','torch.storage._load_from_bytes','joblib.load','pandas.read_pickle'):
            buf=bad if loader.endswith('_from_bytes') else C('io.BytesIO',bad)
            self.verdict(C(loader,buf))
            self.verdict(C(loader,'/outside'),not loader.endswith('_from_bytes'))

    def test_clients(self):
        for cls,args,method in [('socket.socket',(),'connect'),('http.client.HTTPConnection',('host',),'request'),
                                ('ftplib.FTP',(),'connect'),('smtplib.SMTP',(),'connect')]:
            client=C(cls,*args)
            self.verdict(client,False)
            self.verdict(Call(attr(client,method),(('GET','/') if method=='request' else (('localhost',80),) if cls=='socket.socket' else ('host',))))
        for cls in ('ftplib.FTP','smtplib.SMTP'): self.verdict(C(cls,'host'))
        self.verdict(C('xmlrpc.client.ServerProxy','http://host'),False)

    def test_native_handle_is_deferred(self):
        library=C('ctypes.CDLL',None)
        self.verdict(library,False)
        self.verdict(attr(library,'system'),False)
        self.verdict(C('operator.getitem',library,'system'),False)
        self.verdict(Call(C('operator.getitem',library,'system'),(b'echo never',)))
        self.verdict(Call(attr(library,'system'),(b'echo never',)))

    def test_deferred(self):
        for cls in ('threading.Thread','multiprocessing.Process'):
            t=Build(New(G(cls)),{'target':G('os.system'),'args':('x',),'kwargs':{}})
            self.verdict(t,False)
            self.verdict(Call(attr(t,'start')))
        pool=C('concurrent.futures.ThreadPoolExecutor',1)
        self.verdict(pool,False)
        self.verdict(Call(attr(pool,'submit'),(G('os.system'),'x')))

    def test_executor_initializers_and_empty_maps(self):
        factory=C('functools.partial',G('os.system'),'x')
        for cls in ('concurrent.futures.ThreadPoolExecutor','concurrent.futures.ProcessPoolExecutor'):
            executor=C(cls,1)
            self.verdict(Call(attr(executor,'map'),(G('builtins.str'),[])),False)
            self.verdict(Call(attr(executor,'submit')),False)
        executor=Call(Build(C('functools.partial',G('concurrent.futures.ThreadPoolExecutor')),
                            (G('concurrent.futures.ThreadPoolExecutor'),(),{'initializer':factory},None)))
        self.verdict(executor,False)
        self.verdict(Call(attr(executor,'map'),(G('builtins.str'),[1])))
        self.verdict(C('multiprocessing.pool.ThreadPool',1),False)
        self.verdict(Call(attr(C('multiprocessing.pool.ThreadPool',1),'apply'),(factory,)))

    def test_dynamic_classes(self):
        cls=C('builtins.type','Model',(),{'__setstate__':G('builtins.eval')})
        self.verdict(Build(New(cls),'1+1'))
        cls=C('builtins.type','Model',(),{'__init__':G('os.system')})
        self.verdict(New(cls),False)
        self.verdict(Call(cls,('x',)))


    def test_frames_stop_at_loader_boundary(self):
        good=dump(123)
        bad=dump(C('os.system','x'))
        framed=b'\x80\x04\x95'+struct.pack('<Q',len(good+bad))+good+bad
        self.assertFalse(findings(framed))
        framed=b'\x80\x04\x95'+struct.pack('<Q',len(good))+good+bad
        self.assertTrue(findings(framed))

    def test_nested_loader_boundaries(self):
        good=dump({'ok':True}); bad=dump(C('os.system','x'))
        for loader in ('pickle.loads','pickle.load','joblib.load','pandas.read_pickle'):
            source=good+bad if loader.endswith('loads') else C('io.BytesIO',good+bad)
            self.verdict(C(loader,source),False)
        magic=119547037146038801333356
        self.verdict(C('pickle.loads',dump(magic)+bad),False)
        for i in range(5):
            streams=[magic,1001,{},[],[]];streams[i]=C('os.system','x')
            data=b''.join(dump(x,2) for x in streams)
            self.verdict(C('torch.load',C('io.BytesIO',data)))
        self.verdict(C('torch.load',C('io.BytesIO',gzip.compress(bad))),False)
        load=Build(C('functools.partial',G('numpy.load')),(G('numpy.load'),(),{'allow_pickle':True},None))
        self.verdict(Call(load,(C('io.BytesIO',good+bad),)),False)

    def test_c_arity_and_types(self):
        for fn in ('os.system','builtins.eval','builtins.exec','builtins.compile','builtins.open','subprocess.Popen'):
            self.verdict(C(fn),False)
        for data in (None,1,[] ,{}):
            self.verdict(C('os.system',data),False)
            self.verdict(C('builtins.eval',data),False)
        self.verdict(C('types.FunctionType','not a code object',{}),False)

    def test_more_callback_lifetimes(self):
        for fn in ('itertools.takewhile','itertools.dropwhile'):
            obj=C(fn,G('builtins.eval'),['1+1'])
            self.verdict(obj,False);self.verdict(C('builtins.next',obj))
        lazy=C('builtins.map',G('builtins.eval'),['0','1+1'])
        # A zero-length islice does not advance its map.
        self.verdict(C('builtins.list',C('itertools.islice',lazy,0)),False)
        self.verdict(C('builtins.next',C('itertools.cycle',lazy)))
        self.verdict(C('timeit.timeit',C('functools.partial',G('os.system'),'x')))
        self.verdict(C('timeit.Timer',C('functools.partial',G('os.system'),'x')),False)
        self.verdict(Call(attr(C('timeit.Timer',C('functools.partial',G('os.system'),'x')),'timeit')))
        self.verdict(C('timeit.timeit',G('builtins.str')),False)
        self.verdict(Call(C('numpy.vectorize',G('builtins.eval')),(['1+1'],)))
        self.verdict(C('numpy.fromiter',C('builtins.map',G('builtins.eval'),['1+1']),G('builtins.object')))
        arr=C('numpy.array',[G('os.system')])
        self.verdict(Call(C('operator.getitem',arr,0),('x',)))

    def test_more_attribute_paths(self):
        module=C('importlib.import_module','os')
        path=Call(attr(module,'__getattribute__'),('system',))
        self.verdict(Call(path,('x',)))
        self.verdict(Call(attr(attr(G('builtins.__import__'),'__self__'),'eval'),('1+1',)))
        self.verdict(Call(C('operator.getitem',Global('builtins','__dict__'),'eval'),('1+1',)))
        self.verdict([C('copyreg.add_extension','os','system',517),Call(Raw(b'\x83'+struct.pack('<H',517)),('x',))])
        self.verdict([C('copyreg.add_extension','os','system',517),Raw(b'\x83'+struct.pack('<H',517))],False)

    def test_logging(self):
        self.verdict(C('logging.config.dictConfig',{'version':1,'handlers':{}}),False)
        factory=C('functools.partial',G('os.system'),'x')
        cfg={'version':1,'filters':{'custom':{'()':factory}}}
        self.verdict(C('logging.config.dictConfig',cfg))
        cfg['incremental']=True
        self.verdict(C('logging.config.dictConfig',cfg),False)
        self.verdict(C('logging.config.dictConfig',{'filters':{'custom':{'()':factory}}}),False)

    def test_serialized_bytecode(self):
        code=C('marshal.loads',CODE_BYTES)
        self.verdict(code,False)
        self.verdict(C('types.FunctionType',code,{}))
        self.verdict(C('types.FunctionType',attr(G('copy.copy'),'__code__'),{}))
        self.verdict(attr(G('copy.copy'),'__code__'),False)
        self.verdict(C('types.FunctionType',attr(G('builtins.eval'),'__code__'),{}),False)

    def test_joblib_inline_arrays(self):
        state={'shape':(2,), 'dtype':C('numpy.dtype','f8'), 'order':'C',
               'allow_mmap':True,'numpy_array_alignment_bytes':16,'subclass':G('numpy.ndarray')}
        array=Build(New(Global('joblib.numpy_pickle','NumpyArrayWrapper')),state)
        raw=emit(array)+b'\x02\xff\xff'+struct.pack('<dd',1.,2.)
        bad=emit(C('os.system','x'))
        self.assertTrue(findings(b'\x80\x04'+raw+b'0'+bad+b'.'))
        self.verdict(C('joblib.load',C('io.BytesIO',b'\x80\x04'+raw+b'0'+bad+b'.')))
        self.verdict(C('pickle.loads',b'\x80\x04'+raw+b'0'+bad+b'.'),False)
        state['dtype']=C('numpy.dtype','O8')
        raw=emit(Build(New(Global('joblib.numpy_pickle','NumpyArrayWrapper')),state))
        self.assertTrue(findings(b'\x80\x04'+raw+dump(C('os.system','x'))+b'.'))


    def test_new_only_and_buffers(self):
        bad=dump(C('os.system','x'))
        self.verdict(C('pickle.load',New(G('io.BytesIO'),(bad,))),False)
        self.verdict(C('pickle.loads',Raw(emit(bytearray(bad),5)+b'\x98')),True,5)
        self.verdict(New(G('types.FunctionType'),(C('marshal.loads',CODE_BYTES),{})))
        self.verdict(New(G('numpy.memmap'),('/outside',)))
        self.verdict(New(G('numpy.memmap')),False)
        self.verdict(New(G('zoneinfo.ZoneInfo'),('UTC',)))
        for loader in ('pickle.load','numpy.lib.format.read_array'):
            self.verdict(C(loader,'/outside'),False)
        self.verdict(C('pickle.Unpickler','/outside'),False)

    def test_partial_flattening_and_caching(self):
        inner=Memo(C('functools.partial',G('builtins.str')),3)
        outer=Memo(C('functools.partial',inner),4)
        retarget=Build(Get(3),(G('os.system'),(),{},None))
        self.verdict([outer,retarget,Call(Get(4),('x',))],False)
        inner=Memo(C('functools.partial',G('builtins.str')),3)
        cache=Memo(Call(C('functools.lru_cache',None),(inner,)),4)
        self.verdict([cache,Call(Get(4),('x',)),retarget,Call(Get(4),('x',))],False)
        self.verdict([cache,Call(Get(4),('x',)),retarget,Call(Get(4),('y',))])

    def test_fix_imports(self):
        legacy=dump(Call(Global('__builtin__','eval'),('1+1',)),2)
        for fix in (True,False):
            load=Build(C('functools.partial',G('pickle.loads')),(G('pickle.loads'),(),{'fix_imports':fix},None))
            self.verdict(Call(load,(legacy,)),fix)
        self.verdict(Call(Global('__builtin__','eval'),('1+1',)),False,4)

    def test_in_memory_containers(self):
        for data,mal in ((dump(C('os.system','x')),True),(dump({'data':'os.system'}),False)):
            for kind,compress in (('zlib',zlib.compress),('bz2',bz2.compress),('lzma',lzma.compress)):
                decoder=C({'zlib':'zlib.decompressobj','bz2':'bz2.BZ2Decompressor','lzma':'lzma.LZMADecompressor'}[kind])
                self.verdict(C('pickle.loads',Call(attr(decoder,'decompress'),(compress(data),))),mal)
            self.verdict(C('pickle.load',C('gzip.open',C('io.BytesIO',gzip.compress(data)))),mal)
            zipobj=C('zipfile.ZipFile',C('io.BytesIO',archive([('data.pkl',data)])))
            self.verdict(zipobj,False)
            self.verdict(C('pickle.loads',Call(attr(zipobj,'read'),('data.pkl',))),mal)
            encoded=C('codecs.getdecoder','base64_codec')
            decoded=C('operator.getitem',Call(encoded,(base64.b64encode(data),)),0)
            self.verdict(C('pickle.loads',decoded),mal)

    def test_module_callbacks_and_class_hooks(self):
        self.verdict([Build(C('importlib.import_module','math'),{'__getattr__':G('os.system')}),G('math.command')])
        self.verdict([Build(C('importlib.import_module','math'),{'__getattr__':G('os.system')}),G('math.sqrt')],False)
        base=C('builtins.type','Base',(),{'__init_subclass__':C('builtins.staticmethod',C('functools.partial',G('os.system'),'x'))})
        self.verdict(base,False)
        self.verdict(C('builtins.type','Derived',(base,),{}))

    def test_array_and_tensor_subclass_callbacks(self):
        hook=C('builtins.staticmethod',C('functools.partial',G('builtins.exec'),'pass'))
        subtype=C('builtins.type','Array',(G('numpy.ndarray'),),{'__array_finalize__':hook})
        self.verdict(C('numpy._core.multiarray._reconstruct',subtype,(0,),b'b'))
        self.verdict(New(subtype,((1,),)))
        self.verdict(New(subtype),False)
        self.verdict(C('numpy.apply_along_axis',G('subprocess.Popen'),0,['echo','never']))
        hook=C('builtins.staticmethod',C('functools.partial',G('builtins.exec'),'pass',{}))
        subtype=C('builtins.type','Tensor',(G('torch.Tensor'),),{'__setstate__':hook})
        self.verdict(C('torch._tensor._rebuild_from_type_v2',G('torch.empty'),subtype,((0,),),{}))
        self.verdict(C('functools.partial',123),False)

    def test_implicit_object_protocol_callbacks(self):
        for builtin,slot,source in [('builtins.len','__len__','1'),('builtins.bool','__bool__','True'),
                                    ('builtins.str','__str__',"'text'"),('builtins.int','__int__','1')]:
            hook=C('functools.partial',G('builtins.eval'),source)
            cls=C('builtins.type','Model',(),{slot:hook})
            obj=New(cls)
            self.verdict(obj,False)
            self.verdict(C(builtin,obj))
        hook=C('functools.partial',G('builtins.eval'),'1')
        cls=C('builtins.type','Key',(),{'__hash__':hook})
        key=New(cls)
        self.verdict(C('builtins.dict',[(key,1)]))
        self.verdict(C('builtins.set',[key]))
        factory=Memo(C('functools.partial',G('builtins.int')),3)
        cls=C('builtins.type','Key',(),{'__hash__':factory})
        key=Memo(New(cls),4)
        mapping=Memo(C('builtins.dict',[(key,1)]),5)
        retarget=Build(Get(3),(G('os.system'),('echo never',),{},None))
        self.verdict([mapping,retarget],False)
        self.verdict([mapping,retarget,C('operator.getitem',Get(5),Get(4))])
        self.verdict(C('builtins.eval','1+1',7),False)
        self.verdict(C('builtins.exec','pass',{},7),False)
        # An instance attribute named __call__ never makes an object callable.
        self.verdict(Call(Build(C('types.SimpleNamespace'),{'__call__':G('os.system')}),('x',)),False)

    def test_finalization_during_unpickling(self):
        cls=C('builtins.type','Model',(),{'__del__':C('functools.partial',G('builtins.eval'),'1')})
        obj=New(cls)
        self.verdict(obj,False)  # a retained model is not destroyed during load
        self.verdict(Raw(emit(obj)+b'0N'))
        self.verdict(Raw(emit(Memo(obj,9))+b'0N'))  # memo is discarded at STOP
        self.verdict([Memo(obj,9),Get(9)],False)
        self.verdict(Raw(emit(Memo(obj,9))+b'0R'))  # exception cleanup also drops the memo

    def test_numeric_scalar_dataflow(self):
        dtype=C('numpy.dtype','i8')
        scalar=lambda n:C('numpy._core.multiarray.scalar',dtype,struct.pack('<q',n))
        mapping=C('builtins.dict',[(scalar(1),G('os.system'))])
        fn=C('operator.getitem',mapping,scalar(1))
        self.verdict(fn,False)
        self.verdict(Call(fn,('echo never',)))
        false=C('numpy._core.multiarray.scalar',C('numpy.dtype','b1'),b'\0')
        self.verdict(C('ftplib.FTP',false),False)
        self.verdict(C('ftplib.FTP',C('numpy.bool_',False)),False)
        self.verdict(C('ftplib.FTP',C('numpy.bool',False)),False)
        self.verdict(C('builtins.bool',false),False)
        data=dump(C('os.system','echo never'))
        value=C('numpy._core.multiarray.scalar',C('numpy.dtype','S'+str(len(data))),data)
        self.verdict(C('pickle.loads',C('builtins.bytes',value)))
        self.verdict(C('pickle.loads',New(G('numpy.bytes_'),(data,))))
        cls=C('builtins.type','Number',(),{'__float__':C('functools.partial',G('builtins.eval'),'1.0')})
        self.verdict(C('numpy.float64',New(cls)))

    def test_numpy_aliases_and_embedded_loaders(self):
        bad=dump(C('os.system','echo never'));good=dump({'data':'os.system'})
        for data,mal in ((bad,True),(good,False)):
            self.verdict(Call(Global('numpy.lib._npyio_impl','pickle.loads'),(data,)),mal)
        self.verdict(C('numpy.lib._npyio_impl.loadtxt','/outside'))
        self.verdict(C('numpy.lib._datasource.open','/outside'))
        self.verdict(C('numpy.lib.npyio.DataSource'),False)
        self.verdict(C('numpy.lib.npyio.DataSource',None))

    def test_data_wrappers(self):
        self.verdict(Call(C('numpy.object_',G('os.system')),('x',)))
        factory=C('functools.partial',G('os.system'),'x')
        self.verdict(C('collections.defaultdict',factory),False)
        self.verdict(C('operator.getitem',C('collections.defaultdict',factory),'missing'))
        wrapped=C('collections.UserDict',{'call':G('os.system')})
        self.verdict(Call(C('operator.getitem',wrapped,'call'),('x',)))
        transformer=C('sklearn.preprocessing.FunctionTransformer',G('os.system'))
        self.verdict(transformer,False)
        self.verdict(Call(attr(transformer,'transform'),('x',)))


class Containers(unittest.TestCase):
    def test_wrappers(self):
        bad=dump(C('os.system','x')); good=dump({'data':'os.system'})
        for compress in (gzip.compress,bz2.compress,lzma.compress,zlib.compress):
            self.assertTrue(findings(compress(bad)))
            self.assertFalse(findings(compress(good)))
        self.assertTrue(findings(gzip.compress(good)+gzip.compress(bad)))
        for prefix in (b'',b'prefix junk',bad,b'\x1f\x8bgarbage',gzip.compress(good),b'BZhbad'):
            self.assertTrue(findings(archive([('model.pkl',bad)],prefix)))
            self.assertFalse(findings(archive([('model.pkl',good)],prefix)))
        self.assertFalse(findings(archive([('model.pkl',bad),('model.pkl',good)])))
        self.assertTrue(findings(archive([('model.pkl',good),('model.pkl',bad)])))

    def test_torch_and_tar(self):
        data=b'\x80\x02'+emit(('storage',G('torch.FloatStorage'),'0','cpu',1),2)+b'Q0'+emit(C('os.system','x'),2)+b'.'
        torch=archive([('a/data.pkl',data),('a/data/0',b'1234'),('a/version',b'3')])
        self.assertTrue(findings(torch))
        good=dump({'data':'os.system'});bad=dump(C('os.system','x'))
        for nested,main,mal in ((good,bad,True),(bad,good,False)):
            z=archive([('a/extra/data.pkl',nested),('a/data.pkl',main),('a/version',b'3')])
            self.assertEqual(bool(findings(z)),mal)
        t=io.BytesIO()
        with tarfile.open(fileobj=t,mode='w') as tf:
            info=tarfile.TarInfo('a/model.pt');info.size=len(torch);tf.addfile(info,io.BytesIO(torch))
        self.assertTrue(findings(t.getvalue()))
        self.assertTrue(findings(gzip.compress(t.getvalue())))
        legacy=b''.join(dump(v,2) for v in (119547037146038801333356,1001,{},C('os.system','x'),[]))
        self.assertTrue(findings(legacy))
        # A malicious stream beyond the five torch pickles is storage data.
        legacy=b''.join(dump(v,2) for v in (119547037146038801333356,1001,{},[],[]))+dump(C('os.system','x'))
        self.assertFalse(findings(legacy))
        raw=b''.join(dump(v,2) for v in (119547037146038801333356,9999,{},[],[],C('os.system','x')))
        self.assertTrue(findings(raw))

    def test_large_numeric_weights_use_metadata(self):
        from unittest.mock import patch
        header=repr({'descr':'|u1','shape':(4096,),'fortran_order':False}).encode()+b'\n'
        npy=b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+b'\0'*4096
        torch=archive([('a/data.pkl',dump({'weight':'data'})),('a/version',b'3'),('a/data/0',b'\0'*4096)])
        keras=archive([('config.json',json.dumps({'class_name':'Dense','config':{'activation':'relu'}})),
                       ('metadata.json','{}'),('model.weights.h5',b'\0'*4096)])
        cases=[npy,torch,keras,archive([('x.npy',npy)])]
        cases += [compress(npy) for compress in (gzip.compress,bz2.compress,lzma.compress)]
        with tempfile.TemporaryDirectory() as temp:
            for i,data in enumerate(cases):
                p=os.path.join(temp,str(i))
                with open(p,'wb') as f:f.write(data)
                with patch.object(dispatch,'MAX_EXPANDED',512):
                    result=dispatch.scan_path(p)
                self.assertEqual(result.verdict,'clean',result.findings)

    def test_numpy(self):
        def npy(payload,dtype='|O',version=1):
            header=repr({'descr':dtype,'shape':(1,),'fortran_order':False}).encode()+b'\n'
            return b'\x93NUMPY'+bytes([version,0])+struct.pack('<H' if version==1 else '<I',len(header))+header+payload
        bad=dump(C('os.system','x'));good=dump(['os.system'])
        for v in (1,2,3):
            self.assertTrue(findings(npy(bad,version=v)))
            self.assertFalse(findings(npy(good,version=v)))
            self.assertFalse(findings(npy(bad,'|u1',v)))
        self.assertTrue(findings(archive([('a.npy',npy(good)),('b.npy',npy(bad))])))
        self.assertTrue(findings(npy(bad,[('x','i4'),('y','|O')])) )
        self.assertTrue(findings(gzip.compress(npy(good))+gzip.compress(npy(bad))))
        z=archive([('bad.npy',npy(bad))])
        loader=Build(C('functools.partial',G('numpy.load')),(G('numpy.load'),(),{'allow_pickle':True},None))
        self.assertFalse(findings(dump(Call(loader,(C('io.BytesIO',z),)))))
        self.assertTrue(findings(dump(C('operator.getitem',Call(loader,(C('io.BytesIO',z),)),'bad'))))
        self.assertTrue(findings(dump(C('numpy.load',C('io.BytesIO',npy(bad)),None,True))))
        self.assertFalse(findings(dump(C('numpy.load',C('io.BytesIO',npy(bad)),'r',True))))



class Graphs(unittest.TestCase):
    def test_tf(self):
        from modelgate.tfproto import saved_model_pb2
        from google.protobuf import text_format
        from modelgate.policy import TF_OPS
        sm=saved_model_pb2.SavedModel()
        meta=sm.meta_graphs.add()
        for op in ('SaveV2','RestoreV2','Identity','Const','MergeV2Checkpoints'):
            n=meta.graph_def.node.add(name='ReadFile',op=op)
            n.attr['PyFunc'].s=b'os.system'
        self.assertFalse(savedmodel_scan.scan_saved_model(sm.SerializeToString(),'x'))
        for op in TF_OPS:
            meta2=sm.meta_graphs.add()
            f=meta2.graph_def.library.function.add();f.signature.name='hidden'
            f.node_def.add(name='nothing',op=op)
            self.assertTrue(savedmodel_scan.scan_saved_model(sm.SerializeToString(),'x'))
            self.assertTrue(savedmodel_scan.scan_saved_model(text_format.MessageToString(sm).encode(),'x',True))
            del sm.meta_graphs[-1]

    def test_keras(self):
        bytecode={'class_name':'__lambda__','config':{'code':CODE_BASE64}}
        generic={'class_name':'__lambda__','config':{'value':[CODE_BASE64,None,None]}}
        named={'module':'builtins','class_name':'function','config':'relu'}
        data={'class_name':'Lambda','config':{'function':bytecode}}
        model={'class_name':'Sequential','config':{'layers':[],'metadata':data},'metadata':data}
        self.assertFalse(keras_scan.scan_config(model,'x'))
        for field in ('function','output_shape','activation'):
            for fn,mal in ((bytecode,True),(generic,True),(named,False),('relu',False)):
                layer={'class_name':'Lambda' if field!='activation' else 'Dense','config':{field:fn}}
                layer={'class_name':'TimeDistributed','config':{'layer':layer}}
                for _ in range(8): layer={'class_name':'Sequential','config':{'layers':[layer]}}
                self.assertEqual(bool(keras_scan.scan_config(layer,'x')),mal)
                z=archive([('config.json',json.dumps(layer)),('metadata.json','{}'),('model.weights.h5',b'')])
                self.assertEqual(bool(findings(z)),mal)


if __name__=='__main__':unittest.main()
