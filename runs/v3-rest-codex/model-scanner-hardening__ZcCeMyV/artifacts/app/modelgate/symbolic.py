"""A small, non-executing model of Python values used by pickle.

Only literal operations on exact builtin data types are evaluated. Serialized
callables are represented by Ref/Instance objects and are *never* imported or
called by Python. Effects are reported at the point the model calls them.
"""
from __future__ import annotations

import ast
import base64
import builtins
import json
import binascii
import codecs
import math
import operator
import os
import posixpath
import sys
from dataclasses import dataclass, field
from modelgate.policy import CALL_CAPABILITIES, REQUIRED_ARGUMENTS, MAX_POSITIONAL_ARGUMENTS
from modelgate.report import Finding


class Halt(Exception):
    """The real loader deterministically raises before executing anything else."""


class Unsupported(Exception):
    pass


class Detected(Exception):
    def __init__(self, finding):
        self.finding = finding


@dataclass(eq=False)
class Unknown:
    label: str = field(default='unknown',init=False)


UNKNOWN = Unknown()
MISSING = Unknown()


@dataclass(eq=False)
class Ref:
    name: str
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Module:
    name: str
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Namespace:
    owner: object


@dataclass(eq=False)
class Instance:
    cls: object
    args: tuple = ()
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class DynamicClass:
    name: str
    bases: tuple
    attrs: dict


@dataclass(eq=False)
class Bound(Unknown):
    obj: object
    name: str


@dataclass(eq=False)
class Partial:
    fn: object
    args: tuple
    kw: dict
    attrs: dict = field(default_factory=dict)
    namespace_initialized: bool = False


@dataclass(eq=False)
class Wrapper:
    kind: str
    args: tuple = ()
    kw: dict = field(default_factory=dict)


@dataclass(eq=False)
class Buffer:
    data: object = b''
    pos: int = 0
    text: bool = False


@dataclass(eq=False)
class Lazy:
    kind: str
    args: tuple
    index: int = 0
    source: object = None
    done: bool = False


@dataclass(eq=False)
class Decoder:
    kind: str
    engine: object


ALIASES = {'posix': 'os', 'nt': 'os', '_pickle': 'pickle', '_io': 'io',
           '_operator': 'operator', '_functools': 'functools', '_socket': 'socket', 'os.path': 'posixpath',
           }
# Module-valued attributes found along common Python module paths.
MODULE_EDGES = {
    'os': {'path': 'posixpath'},
    'shutil': {'os': 'os', 'sys': 'sys', 'stat': 'stat', 'collections': 'collections'},
    'subprocess': {'os': 'os', 'sys': 'sys', 'time': 'time', 'threading': 'threading'},
    'pathlib': {'os': 'os', 'sys': 'sys', 'io': 'io', 'posixpath': 'posixpath'},
    'pickle': {'sys': 'sys', 'io': 'io', 'codecs': 'codecs', 're': 're'},
    'functools': {'types': 'types'},
    'copy': {'types': 'types', 'copyreg': 'copyreg'},
    'pydoc': {'os': 'os', 'sys': 'sys', 'subprocess': 'subprocess', 'inspect': 'inspect', 'builtins': 'builtins'},
    'tempfile': {'_os': 'os', '_io': 'io', 'os': 'os'},
    'logging': {'os': 'os', 'sys': 'sys', 'io': 'io'},
    'logging.config': {'os': 'os', 'sys': 'sys', 'logging': 'logging', 'types': 'types'},
    'urllib.request': {'os': 'os', 'sys': 'sys', 'io': 'io', 'socket': 'socket', 'http': 'http'},
    'http.client': {'socket': 'socket', 'ssl': 'ssl', 'io': 'io'},
    'socket': {'os': 'os', 'sys': 'sys', 'io': 'io'},
    'ssl': {'os': 'os', 'sys': 'sys', 'socket': 'socket'},
    'importlib': {'sys': 'sys'}, 'pkgutil': {'sys': 'sys', 'importlib': 'importlib'},
    'numpy': {'ctypeslib': 'numpy.ctypeslib', 'lib': 'numpy.lib', 'testing': 'numpy.testing'},
    'numpy.lib': {'format': 'numpy.lib.format'},
    'torch': {'storage': 'torch.storage', '_utils': 'torch._utils', 'serialization': 'torch.serialization'},
    'torch.serialization': {'pickle': 'pickle', 'os': 'os', 'io': 'io', 'sys': 'sys'},
}
KNOWN_MODULES = set(MODULE_EDGES) | set(ALIASES.values()) | {
    'builtins', 'sys', 'types', 'io', 'operator', 'functools', 'itertools', 'collections',
    'collections.abc', 'copy', 'copyreg', 'base64', 'codecs', '_codecs', 'binascii',
    'zlib', 'gzip', 'bz2', 'lzma', 'marshal', 'importlib', 'importlib.util',
    'importlib.machinery', 'pkgutil', 'timeit', 'code', 'bdb', 'pdb', 'runpy',
    'logging.config', 'logging.handlers', 'threading', 'concurrent', 'concurrent.futures',
    'concurrent.futures.thread', 'concurrent.futures.process', 'multiprocessing',
    'multiprocessing.pool', 'multiprocessing.context', 'multiprocessing.process',
    'http', 'http.client', 'urllib', 'urllib.request', 'urllib.parse', 'xmlrpc',
    'xmlrpc.client', 'ftplib', 'smtplib', 'ssl', 'webbrowser', 'pty', 'pydoc',
    'posixpath', 'ntpath', 'numpy.lib.format', 'numpy.core.multiarray', 'numpy._core.multiarray',
    'numpy.core.numeric', 'numpy._core.numeric', 'numpy.testing._private.utils',
    'torch', 'torch.storage', 'torch.serialization', 'torch._utils', 'joblib',
    'joblib.numpy_pickle', 'joblib.numpy_pickle_compat', 'pandas',
}
CLASS_NAMES = {
    'builtins.object', 'builtins.type', 'builtins.str', 'builtins.bytes', 'builtins.bytearray',
    'builtins.int', 'builtins.float', 'builtins.bool', 'builtins.list', 'builtins.tuple',
    'builtins.dict', 'builtins.set', 'builtins.frozenset', 'builtins.complex',
    'builtins.slice', 'builtins.range', 'builtins.property', 'builtins.staticmethod',
    'builtins.classmethod', 'types.SimpleNamespace', 'types.ModuleType', 'types.FunctionType',
    'types.CodeType', 'io.BytesIO', 'io.StringIO', 'io.FileIO', 'io.TextIOWrapper',
    'io.BufferedReader', 'io.BufferedWriter', 'io.BufferedRandom',
    'socket.socket', 'http.client.HTTPConnection', 'http.client.HTTPSConnection',
    'ftplib.FTP', 'ftplib.FTP_TLS', 'smtplib.SMTP', 'smtplib.SMTP_SSL', 'xmlrpc.client.ServerProxy',
    'pathlib.Path', 'pathlib.PosixPath', 'pathlib.WindowsPath', 'pathlib.PurePath',
    'pathlib.PurePosixPath', 'pathlib.PureWindowsPath', 'subprocess.Popen',
    'threading.Thread', 'threading.Timer', 'multiprocessing.Process',
    'multiprocessing.context.Process', 'concurrent.futures.ThreadPoolExecutor',
    'concurrent.futures.ProcessPoolExecutor', 'pickle.Unpickler',
    'functools.partial', 'operator.attrgetter', 'operator.itemgetter', 'operator.methodcaller',
    'timeit.Timer', 'bdb.Bdb', 'pdb.Pdb', 'code.InteractiveInterpreter', 'code.InteractiveConsole',
    'collections.OrderedDict', 'collections.defaultdict', 'collections.Counter', 'collections.deque',
    'numpy.dtype', 'numpy.ndarray', 'numpy.memmap','zoneinfo.ZoneInfo','joblib.numpy_pickle.NumpyArrayWrapper',
}


NUMPY_SCALARS = {'numpy.'+n for n in (
    'bool bool_ int8 int16 int32 int64 uint8 uint16 uint32 uint64 int_ uint '
    'float16 float32 float64 longdouble complex64 complex128 clongdouble bytes_ str_ void').split()}
CLASS_NAMES.update(NUMPY_SCALARS)


def refname(value):
    return value.name if isinstance(value, Ref) else ''


def classname(value):
    return refname(value.cls) if isinstance(value, Instance) else ''


def literal(value):
    if type(value) in (str, bytes, bytearray, int, float, bool, type(None)):
        return True
    np=sys.modules.get('numpy')
    if np is not None and isinstance(value,np.generic) and not value.dtype.hasobject:
        return True
    if type(value) in (list, tuple, set, frozenset):
        return all(literal(x) for x in value)
    if type(value) is dict:
        return all(literal(k) and literal(v) for k, v in value.items())
    return False


class Runtime:
    MAX_ITEMS = 20000
    MAX_BYTES = 128 * 1024 * 1024

    def __init__(self, artifact_path=None, artifact_data=None, directory=False):
        self.modules = {}
        self.refs = {}
        self.location = ''
        self.steps = 0
        self.depth = 0
        self.artifact_path = os.path.abspath(artifact_path) if artifact_path else None
        self.artifact_data = artifact_data
        self.directory = directory
        self.load_callback = None
        self.vm_roots=[]
        self.call_roots=[]
        self.returned_roots=[]
        self.finalizers={}
        self.finalized=set()

    @staticmethod
    def children(value):
        if isinstance(value,dict): return list(value.keys())+list(value.values())
        if isinstance(value,(list,tuple,set,frozenset)): return list(value)
        if isinstance(value,(Module,Ref)): return [value.attrs]
        if isinstance(value,Instance): return [value.cls,value.args,value.attrs]
        if isinstance(value,DynamicClass): return [value.bases,value.attrs]
        if isinstance(value,Partial): return [value.fn,value.args,value.kw,value.attrs]
        if isinstance(value,Wrapper): return [value.args,value.kw]
        if isinstance(value,Bound): return [value.obj]
        if isinstance(value,Namespace): return [value.owner]
        if isinstance(value,Lazy): return [value.args,value.source]
        if isinstance(value,Buffer): return [value.data]
        return ()

    def collect_finalizers(self,extra=(),ignore=None):
        if not self.finalizers: return
        roots=list(extra)+list(self.modules.values())+list(self.refs.values())+self.returned_roots+list(self.call_roots)
        for context in self.vm_roots:
            if context is ignore: continue
            stack,memo=context
            roots.extend(stack); roots.extend(memo.values())
        seen=set()
        while roots:
            node=roots.pop(); ident=id(node)
            if ident in seen: continue
            seen.add(ident); roots.extend(self.children(node))
        for ident,obj in list(self.finalizers.items()):
            if ident in seen or ident in self.finalized: continue
            # Refcount cleanup is deterministic. Cycles require a separate GC
            # event, so do not invent an immediate finalization of a cycle.
            pending=list(self.children(obj)); visited=set(); cyclic=False
            while pending:
                node=pending.pop(); node_id=id(node)
                if node is obj: cyclic=True; break
                if node_id in visited: continue
                visited.add(node_id); pending.extend(self.children(node))
            if cyclic: continue
            self.finalized.add(ident)
            method=self.special(obj,'__del__')
            if method is not MISSING:
                try: self.call(method)
                except Halt: pass  # CPython ignores exceptions in __del__.

    def tick(self):
        self.steps += 1
        if self.steps > 1500000:
            raise Unsupported('static evaluation work limit exceeded')

    def fail(self, kind, detail):
        raise Detected(Finding(kind, self.location, detail))

    def ref(self, name):
        if name not in self.refs:
            self.refs[name] = Ref(name)
        return self.refs[name]

    def module(self, name):
        if not isinstance(name, str):
            raise Halt('module name must be a string')
        name = ALIASES.get(name, name)
        if name in {'__builtin__','copy_reg','cPickle','commands','urllib2','httplib','cStringIO','StringIO','exceptions','xmlrpclib','urlparse','UserDict','UserList','UserString','ConfigParser','Queue','SocketServer'}:
            raise Halt('Python 2 module name requires fix_imports below protocol 3')
        if name not in self.modules:
            self.modules[name] = Module(name)
        return self.modules[name]

    def global_ref(self, module, name, proto, fix_imports=True):
        if proto < 3 and fix_imports:
            import _compat_pickle
            module, name = _compat_pickle.NAME_MAPPING.get((module, name), (module, name))
            module = _compat_pickle.IMPORT_MAPPING.get(module, module)
        obj = self.module(module)
        if proto < 4 and '.' in name:
            # find_class below protocol 4 does one getattr, not dotted lookup.
            if name not in obj.attrs:
                raise Halt('dotted global below protocol 4')
            return obj.attrs[name]
        for part in name.split('.'):
            obj = self.getattr(obj, part)
        return obj

    def isclass(self, obj):
        if isinstance(obj, DynamicClass):
            return True
        name = refname(obj)
        return name in CLASS_NAMES or (bool(name) and name.rsplit('.', 1)[-1][:1].isupper())

    def getattr(self, obj, name, default=MISSING):
        self.tick()
        if isinstance(name,Unknown): raise Unsupported('unresolved attribute name')
        if not isinstance(name, str):
            raise Halt('attribute name must be a string')
        attrs = getattr(obj, 'attrs', None)
        deleted=isinstance(attrs,dict) and attrs.get(name) is MISSING
        if isinstance(attrs, dict) and name in attrs and not deleted:
            return attrs[name]
        if name in ('__getattribute__', '__setattr__', '__delattr__'):
            return Bound(obj, name)
        if name == '__dict__' and isinstance(obj, (Module, Ref, Instance, Partial, DynamicClass)):
            if isinstance(obj,Partial): obj.namespace_initialized=True
            return Namespace(obj)
        if isinstance(obj, Module):
            if deleted and '__getattr__' not in obj.attrs:
                if default is not MISSING: return default
                raise Halt('deleted module attribute')
            trusted={'os','sys','math','builtins','operator','functools','types','io',
                     'copy','copyreg','collections','base64','codecs','pickle',
                     'json','ast','posixpath','zlib','gzip','bz2','lzma'}
            real=sys.modules.get(obj.name) if obj.name in trusted else None
            if real is not None and name not in vars(real) and name not in MODULE_EDGES.get(obj.name,{}) and '__getattr__' not in obj.attrs:
                if default is not MISSING: return default
                raise Halt('module attribute does not exist')
            if obj.name == 'sys' and name == 'modules':
                return Wrapper('modules')
            if name in ('__builtins__', 'builtins'):
                return self.module('builtins')
            if name == '__name__':
                return obj.name
            if name == '__class__': return self.ref('types.ModuleType')
            if name in MODULE_EDGES.get(obj.name, {}):
                return self.module(MODULE_EDGES[obj.name][name])
            if '__getattr__' in obj.attrs:
                # Module.__getattr__ is itself an executed callback. Existing
                # attributes bypass it. Inspect only already imported trusted
                # stdlib dictionaries; no artifact module is ever imported.
                trusted={'os','sys','math','builtins','operator','functools','types','io',
                         'copy','copyreg','collections','base64','codecs','pickle',
                         'json','ast','posixpath','zlib','gzip','bz2','lzma'}
                real=sys.modules.get(obj.name) if obj.name in trusted else None
                if deleted or real is not None and name not in vars(real):
                    return self.call(obj.attrs['__getattr__'],(name,))
                if real is None: raise Unsupported('cannot resolve a module attribute with a custom __getattr__')
            fullname = obj.name + '.' + name
            if obj.name=='numpy' or obj.name.startswith('numpy.'):
                self.numpy_module()
                module_name=obj.name.replace('numpy.core.','numpy._core.',1)
                native=sys.modules.get(module_name)
                if native is not None:
                    actual=vars(native).get(name,MISSING)
                    if isinstance(actual,type(sys)): return self.module(actual.__name__)
                    if actual is not MISSING and callable(actual):
                        mod=getattr(actual,'__module__',None)
                        qual=getattr(actual,'__qualname__',getattr(actual,'__name__',None))
                        if isinstance(mod,str) and isinstance(qual,str):
                            return self.ref(ALIASES.get(mod,mod)+'.'+qual)
            if fullname in KNOWN_MODULES:
                return self.module(fullname)
            # Common re-exports share identity, including their mutable attrs.
            exports = {
                'os._exit': 'os._exit', 'io.open': 'builtins.open',
                'posixpath.os': 'os', 'ntpath.os': 'os',
                'operator.__call__': 'operator.call',
                'numpy.load': 'numpy.load', 'torch.serialization.load': 'torch.load',
                'pickle._Unpickler': 'pickle.Unpickler',
                'pickle._loads': 'pickle.loads', 'pickle._load': 'pickle.load',
                'joblib.numpy_pickle.load':'joblib.load',
                'pandas.io.pickle.read_pickle':'pandas.read_pickle',
                'numpy.lib.npyio.load':'numpy.load', 'numpy.lib._npyio_impl.load':'numpy.load',
            }
            fullname = exports.get(fullname, fullname)
            if fullname in KNOWN_MODULES:
                return self.module(fullname)
            return self.ref(fullname)
        if isinstance(obj, Ref):
            if name in ('__code__','__globals__','__self__'):
                # Distinguish Python functions from builtin functions without
                # importing an artifact-selected module or evaluating a getter.
                trusted={'builtins','os','math','operator','functools','copy','copyreg',
                         'types','io','base64','codecs','pickle','json','ast','posixpath'}
                parts=obj.name.split('.')
                real=MISSING
                for split in range(len(parts)-1,0,-1):
                    module='.'.join(parts[:split])
                    if module not in trusted or module not in sys.modules: continue
                    real=sys.modules[module]
                    for part in parts[split:]:
                        try: real=vars(real).get(part,MISSING)
                        except TypeError: real=MISSING
                        if real is MISSING: break
                    break
                if real is not MISSING:
                    import types
                    is_python=isinstance(real,types.FunctionType)
                    if name in ('__code__','__globals__') and not is_python:
                        raise Halt('builtin or class has no Python function state')
                    if name=='__self__' and is_python:
                        raise Halt('unbound Python function has no __self__')
                if name=='__code__': return Instance(self.ref('types.CodeType'),(obj,))
            if name == '__self__':
                return self.module(obj.name.rsplit('.', 1)[0])
            if name == '__module__':
                return obj.name.rsplit('.', 1)[0]
            if name == '__name__':
                return obj.name.rsplit('.', 1)[-1]
            if name == '__globals__':
                return Namespace(self.module(obj.name.rsplit('.', 1)[0]))
            if name == '__builtins__':
                return Namespace(self.module('builtins'))
            if name in ('__func__', '__wrapped__'):
                return obj
            if name == '__class__':
                return self.ref('builtins.type' if self.isclass(obj) else 'types.FunctionType')
            if name in ('__mro__', '__bases__'):
                return (obj, self.ref('builtins.object')) if name == '__mro__' else (self.ref('builtins.object'),)
            # Descriptors and class methods remain symbolic too.
            return self.ref(obj.name + '.' + name)
        if isinstance(obj, Partial):
            if name == 'func': return obj.fn
            if name == 'args': return obj.args
            if name == 'keywords': return obj.kw
            if name == '__call__': return obj
            return Bound(obj, name)
        if isinstance(obj, Wrapper):
            if name=='cache_clear' and obj.kind=='cache': return Bound(obj,name)
            if name in ('__wrapped__', '__func__') and obj.kind in ('forward','cache'):
                return obj.args[0]
            if name == '__call__': return obj
            return Bound(obj, name)
        if isinstance(obj, DynamicClass):
            for base in obj.bases:
                if isinstance(base, DynamicClass) and name in base.attrs:
                    return base.attrs[name]
            if default is not MISSING:
                return default
            return self.ref(obj.name + '.' + name)
        if isinstance(obj, Instance):
            if name == '__class__': return obj.cls
            if isinstance(obj.cls, DynamicClass):
                method = self.getattr(obj.cls, name, default)
                return self.bind_descriptor(method, obj)
            return Bound(obj, name)
        np=sys.modules.get('numpy')
        if np is not None and isinstance(obj,np.generic):
            if name=='__class__': return self.ref('numpy.'+type(obj).__name__)
            return Bound(obj,name)
        if isinstance(obj,Decoder):
            if name in ('unused_data','unconsumed_tail','eof','needs_input'): return getattr(obj.engine,name,UNKNOWN)
            return Bound(obj,name)
        if isinstance(obj, (Buffer, Lazy, Namespace)) or type(obj) in (str, bytes, bytearray, list, dict, tuple, set, frozenset):
            if name == '__class__': return self.ref('builtins.' + type(obj).__name__)
            return Bound(obj, name)
        if isinstance(obj, Unknown):
            return UNKNOWN if default is MISSING else default
        if default is not MISSING:
            return default
        return Bound(obj, name)

    def special(self,obj,name):
        if not isinstance(obj,Instance) or not isinstance(obj.cls,DynamicClass): return MISSING
        def lookup(cls):
            if not isinstance(cls,DynamicClass): return MISSING
            if name in cls.attrs: return cls.attrs[name]
            for base in cls.bases:
                value=lookup(base)
                if value is not MISSING: return value
            return MISSING
        fn=lookup(obj.cls)
        return self.bind_descriptor(fn,obj) if fn is not MISSING else MISSING

    def truth(self,obj):
        fn=self.special(obj,'__bool__')
        if fn is not MISSING: return bool(self.call(fn))
        fn=self.special(obj,'__len__')
        if fn is not MISSING: return bool(self.call(fn))
        return bool(obj)

    def hash_key(self,obj):
        if isinstance(obj,(tuple,frozenset)):
            for element in obj: self.hash_key(element)
            return
        fn=self.special(obj,'__hash__')
        if fn is None: raise Halt('unhashable key')
        if fn is not MISSING: self.call(fn)

    def bind_descriptor(self, value, obj):
        if isinstance(value, Wrapper) and value.kind == 'staticmethod': return value.args[0]
        if isinstance(value, Wrapper) and value.kind == 'classmethod': return Partial(value.args[0], (obj.cls,), {})
        if isinstance(value,(Partial,Bound)) or isinstance(value,Wrapper) and value.kind not in ('cache','forward'):
            return value
        # Builtin functions do not bind self when stored on a Python class.
        if isinstance(value, Ref) and value.name.startswith(('builtins.', 'os.', 'operator.')):
            return value
        return Partial(value, (obj,), {})

    def getitem(self, obj, key, default=MISSING):
        self.tick()
        if isinstance(obj, Namespace):
            owner = obj.owner
            if isinstance(owner, Module):
                if key == '__builtins__': return Namespace(self.module('builtins'))
                return self.getattr(owner, key, default)
            return owner.attrs.get(key, UNKNOWN if default is MISSING else default)
        if isinstance(obj, Wrapper) and obj.kind == 'modules':
            return self.module(key)
        if isinstance(obj, Instance):
            if classname(obj) in ('ctypes.CDLL','ctypes.PyDLL'):
                return Bound(obj,key if isinstance(key,str) else str(key))
            if classname(obj)=='collections.defaultdict':
                mapping=obj.attrs['_mapping']
                if key not in mapping:
                    factory=obj.attrs.get('default_factory')
                    if factory is None: raise Halt('defaultdict missing key')
                    mapping[key]=self.call(factory)
                return mapping[key]
            if '_arrays' in obj.attrs:
                raw=obj.attrs['_arrays'].get(key,obj.attrs['_arrays'].get(str(key).removesuffix('.npy')))
                if raw is None: raise Halt('array not found')
                value,_=self.load_callback(raw,self.location+'!'+str(key),self,'numpy.load',{'allow_pickle':obj.attrs['_allow_pickle']})
                return value
            if '_elements' in obj.attrs:
                return self.getitem(obj.attrs['_elements'],key,default)
            if isinstance(obj.cls,DynamicClass):
                method=self.special(obj,'__getitem__')
                if method is MISSING: raise Halt('object is not subscriptable')
                return self.call(method,(key,))
            raise Unsupported('unmodelled subscript on '+classname(obj))
        if isinstance(obj,dict): self.hash_key(key)
        if isinstance(obj, Unknown) or isinstance(key, Unknown):
            return UNKNOWN
        try:
            return obj[key]
        except (KeyError, IndexError, TypeError):
            if default is not MISSING: return default
            raise Halt('invalid subscript')

    def setitem(self, obj, key, value):
        if isinstance(obj, Namespace):
            obj.owner.attrs[key] = value
        elif isinstance(obj, Wrapper) and obj.kind == 'modules':
            self.modules[ALIASES.get(key, key)] = value
        elif isinstance(obj, (dict, list)):
            if isinstance(obj,dict): self.hash_key(key)
            try: obj[key] = value
            except (KeyError, IndexError, TypeError): raise Halt('invalid item assignment')
        elif isinstance(obj, Instance):
            if classname(obj)=='collections.defaultdict':
                obj.attrs['_mapping'][key]=value
            elif '_elements' in obj.attrs:
                self.setitem(obj.attrs['_elements'],key,value)
            elif isinstance(obj.cls, DynamicClass):
                self.call(self.getattr(obj, '__setitem__'), (key, value))
            else:
                obj.attrs.setdefault('_items', {})[key] = value

    def build(self, obj, state):
        # BUILD looks up __setstate__ on the instance, not only on its type.
        attrs = getattr(obj, 'attrs', {})
        if '__setstate__' in attrs:
            self.call(attrs['__setstate__'], (state,))
            return
        if isinstance(obj, Partial):
            if not isinstance(state, tuple) or len(state) != 4:
                raise Halt('invalid partial state')
            fn, args, kw, namespace = state
            if not isinstance(args, tuple) or not isinstance(kw, (dict, type(None))):
                raise Halt('invalid partial state')
            if not self.is_callable(fn): raise Halt('partial target must be callable')
            obj.fn, obj.args, obj.kw = fn, args, kw or {}
            obj.namespace_initialized=isinstance(namespace,dict)
            if isinstance(namespace, dict): obj.attrs.update(namespace)
            return
        if isinstance(obj, Instance) and isinstance(obj.cls, DynamicClass) and '__setstate__' in obj.cls.attrs:
            self.call(self.getattr(obj, '__setstate__'), (state,))
            return
        slots = None
        if isinstance(state, tuple) and len(state) == 2:
            state, slots = state
        if isinstance(state, dict):
            attrs.update(state)
        if isinstance(slots, dict):
            attrs.update(slots)
        if isinstance(obj, Instance):
            obj.attrs['_pickle_state'] = state
            if classname(obj) in ('numpy.ndarray','numpy._core.multiarray._reconstruct','numpy.core.multiarray._reconstruct'):
                if isinstance(state,tuple) and len(state)>4 and isinstance(state[4],list): obj.attrs['_elements']=state[4]

    def file_effect(self, args, name, kw):
        paths = args
        if not paths:
            params=REQUIRED_ARGUMENTS.get(name,())
            paths=tuple(kw.get(k,UNKNOWN) for k in params)
            if not paths: paths=(kw.get('filename',kw.get('path',UNKNOWN)),)
        if name.startswith('pathlib.') and args and isinstance(args[0], Instance):
            paths = (self.path_string(args[0]),) + args[1:]
        # Calls operating entirely on an in-memory file are harmless.
        if paths and isinstance(paths[0], Buffer):
            return False
        if paths and self.inside_artifact(paths[0]):
            if name in ('os.rename', 'os.replace', 'shutil.copy', 'shutil.copyfile', 'shutil.copy2', 'shutil.move'):
                return len(paths) < 2 or not self.inside_artifact(paths[1])
            return False
        return True

    def path_string(self, obj):
        if isinstance(obj, Instance) and classname(obj).startswith('pathlib.'):
            if all(isinstance(x, (str, bytes)) for x in obj.args):
                return os.path.join(*obj.args) if obj.args else '.'
        return obj

    def inside_artifact(self, path):
        path = self.path_string(path)
        if not self.artifact_path or not isinstance(path, (str, bytes)):
            return False
        try:
            path = os.path.abspath(os.fsdecode(path))
            return path == self.artifact_path or (self.directory and path.startswith(self.artifact_path + os.sep))
        except (ValueError, TypeError):
            return False

    def call(self, fn, args=(), kw=None, new_only=False):
        self.tick()
        if not isinstance(kw,(dict,type(None))) or (kw and not all(isinstance(k,str) for k in kw)):
            raise Halt('keyword names must be strings')
        self.depth += 1
        if self.depth > 100:
            self.depth -= 1
            raise Unsupported('call/loader nesting limit exceeded')
        self.call_roots.append((fn,args,kw))
        try:
            result=self._call(fn, args, kw or {}, new_only)
            if isinstance(result,Instance) and self.special(result,'__del__') is not MISSING:
                self.finalizers[id(result)]=result
            return result
        except (ValueError, TypeError, IndexError, KeyError, OverflowError, UnicodeError, ZeroDivisionError) as exc:
            raise Halt(str(exc)) from None
        finally:
            self.call_roots.pop()
            self.depth -= 1

    def _call(self, fn, args, kw, new_only):
        if isinstance(args,Unknown): raise Unsupported('unresolved REDUCE arguments')
        if not isinstance(args, tuple):
            raise Halt('REDUCE arguments must be a tuple')
        if new_only:
            if isinstance(fn,Unknown): raise Unsupported('unresolved NEWOBJ class')
            if not self.isclass(fn):
                raise Halt('NEWOBJ requires a class')
            if isinstance(fn, DynamicClass) and '__new__' in fn.attrs:
                return self.call(fn.attrs['__new__'], (fn,) + args, kw)
            if refname(fn) in ('io.BytesIO','io.StringIO'):
                return Buffer('' if refname(fn).endswith('StringIO') else b'',text=refname(fn).endswith('StringIO'))
            if refname(fn) in ('builtins.list','builtins.dict','builtins.set','builtins.bytearray'):
                return {'builtins.list':list,'builtins.dict':dict,'builtins.set':set,'builtins.bytearray':bytearray}[fn.name]()
            # C types with custom __new__ parse their values without __init__.
            if refname(fn) in ('functools.partial', 'operator.attrgetter', 'operator.itemgetter',
                               'operator.methodcaller', 'builtins.bytes', 'builtins.str',
                               'builtins.tuple', 'builtins.frozenset', 'builtins.int',
                               'builtins.float', 'builtins.complex','builtins.type','types.FunctionType','types.LambdaType','types.CodeType','numpy.memmap','zoneinfo.ZoneInfo') or refname(fn) in NUMPY_SCALARS:
                return self._call(fn, args, kw, False)
            if isinstance(fn,DynamicClass):
                def derives(cls,name):
                    if isinstance(cls,Ref): return cls.name==name
                    return isinstance(cls,DynamicClass) and any(derives(b,name) for b in cls.bases)
                if derives(fn,'numpy.ndarray'):
                    if not args: raise Halt('ndarray.__new__ requires a shape')
                    obj=Instance(fn,args)
                    hook=self.special(obj,'__array_finalize__')
                    if hook is not MISSING: self.call(hook,(None,))
                    return obj
            return Instance(fn, args)
        if isinstance(fn, Partial):
            return self.call(fn.fn, fn.args + args, {**fn.kw, **kw})
        if isinstance(fn, Bound):
            return self.method(fn.obj, fn.name, args, kw)
        if isinstance(fn, Wrapper):
            return self.wrapper(fn, args, kw)
        if isinstance(fn, DynamicClass):
            if '__new__' in fn.attrs:
                obj = self.call(fn.attrs['__new__'], (fn,) + args, kw)
            else:
                obj = self.call(fn,args,kw,new_only=True)
            if '__init__' in fn.attrs and isinstance(obj,Instance) and obj.cls is fn:
                self.call(self.bind_descriptor(fn.attrs['__init__'], obj), args, kw)
            return obj
        if isinstance(fn, Instance):
            if isinstance(fn.cls, DynamicClass):
                method=self.special(fn,'__call__')
                if method is MISSING: raise Halt('instance is not callable')
                return self.call(method,args,kw)
            return UNKNOWN
        if isinstance(fn, Unknown):
            raise Unsupported('cannot resolve executed callable')
        if not isinstance(fn, Ref):
            raise Halt('value is not callable')
        name = fn.name
        # Explicit descriptor invocation: type.__call__, f.__call__, C.__new__.
        if name == 'builtins.type.__call__':
            return self.call(args[0], args[1:], kw)
        if name.endswith('.__call__') and name != 'xmlrpc.client.ServerProxy.__call__':
            return self.call(self.ref(name[:-9]), args, kw)
        if name=='builtins.object.__new__':
            if not args or not self.isclass(args[0]): raise Halt('object.__new__ needs a class')
            return Instance(args[0])
        if name.endswith('.__new__'):
            cls = args[0] if args else self.ref(name[:-8])
            return self.call(cls, args[1:], kw, new_only=True)
        if name.endswith('.__init__'):
            if not args: raise Halt('unbound init needs instance')
            return self.call(self.ref(name[:-9]), args[1:], kw)
        clsname, _, method = name.rpartition('.')
        if clsname in CLASS_NAMES and (name not in CALL_CAPABILITIES or clsname=='timeit.Timer') and args and isinstance(args[0], Instance) and method not in ('__new__','__init__'):
            return self.method(args[0],method,args[1:],kw)
        if name in ('gzip.GzipFile','gzip.open','bz2.BZ2File','bz2.open','lzma.LZMAFile','lzma.open'):
            target=kw.get('fileobj',args[0] if args else kw.get('filename'))
            if target is None: raise Halt('compressed file needs a file object or path')
            if isinstance(target,Buffer):
                from modelgate.dispatch import decompress
                kind={'gzip':'gzip','bz2':'bz2','lzma':'lzma'}[name.split('.')[0]]
                mode=args[1] if len(args)>1 else kw.get('mode','rb')
                if isinstance(mode,str) and any(c in mode for c in 'wa'): return Buffer()
                return Buffer(decompress(target.data[target.pos:],kind))
            if not self.inside_artifact(target): self.fail('file-access',name+' opens a path')
            return Buffer(self.artifact_data or b'')
        if name in ('zipfile.ZipFile','tarfile.open','tarfile.TarFile.open'):
            target=args[0] if args else kw.get('fileobj',kw.get('name',kw.get('file')))
            if isinstance(target,Buffer):
                import io
                if name=='zipfile.ZipFile':
                    import zipfile
                    with zipfile.ZipFile(io.BytesIO(target.data)) as zf:
                        infos={i.filename:i for i in zf.infolist() if not i.is_dir()}
                        if sum(i.file_size for i in infos.values())>self.MAX_BYTES: raise Unsupported('expanded zip size limit')
                        entries={n:zf.read(i) for n,i in infos.items()}
                    return Instance(fn,(),{'_zip_entries':entries})
                import tarfile
                with tarfile.open(fileobj=io.BytesIO(target.data),mode='r:*') as tf:
                    infos={i.name:i for i in tf if i.isfile()}
                    if sum(i.size for i in infos.values())>self.MAX_BYTES: raise Unsupported('expanded tar size limit')
                    entries={n:tf.extractfile(i).read() for n,i in infos.items()}
                return Instance(fn,(),{'_tar_entries':entries})
            if target is None: raise Halt('archive needs a path or file object')
            self.fail('file-access',name+' opens a path')
        if name in ('io.BufferedReader','io.BufferedWriter','io.BufferedRandom','io.TextIOWrapper') and args and isinstance(args[0],Buffer):
            if name=='io.TextIOWrapper':
                encoding=args[1] if len(args)>1 else kw.get('encoding','utf-8')
                return Buffer(args[0].data.decode(encoding),text=True)
            return args[0]
        if name=='functools._lru_cache_wrapper':
            return Wrapper('cache',(args[0],),{'maxsize':args[1] if len(args)>1 else 128,'_cache':{}})
        if name in ('builtins.memoryview',):
            if not isinstance(args[0],(str,bytes,bytearray)): raise Halt('not a buffer')
            return bytes(args[0])
        if name in ('codecs.getdecoder','codecs.getencoder'):
            return Wrapper('codec',(args[0],'decode' if name.endswith('decoder') else 'encode'))
        if name in ('zlib.decompressobj','bz2.BZ2Decompressor','lzma.LZMADecompressor'):
            if not literal(args) or not literal(kw): raise Halt('invalid decompressor options')
            if name.startswith('zlib'):
                import zlib
                engine=zlib.decompressobj(*args,**kw)
            elif name.startswith('bz2'):
                import bz2
                engine=bz2.BZ2Decompressor(*args,**kw)
            else:
                import lzma
                engine=lzma.LZMADecompressor(*args,**kw)
            return Decoder(name.split('.')[0],engine)
        if name in ('base64.decode','base64.encode'):
            if len(args)<2 or not isinstance(args[0],Buffer) or not isinstance(args[1],Buffer):
                raise Halt('base64 file codec needs file objects')
            data=self.method(args[0],'read',(),{})
            value=base64.decodebytes(data) if name.endswith('decode') else base64.encodebytes(data)
            self.method(args[1],'write',(value,),{})
            return None
        if name in ('copy._reconstruct',):
            # copy._reconstruct(x, memo, func, args, state, listiter, dictiter)
            obj=self.call(args[2],args[3])
            if len(args)>4 and args[4] is not None: self.build(obj,args[4])
            if len(args)>5 and args[5] is not None:
                for value in self.consume(args[5]): self.method(obj,'append',(value,),{})
            if len(args)>6 and args[6] is not None:
                for key,value in self.consume(args[6]): self.setitem(obj,key,value)
            return obj
        if name == 'torch._tensor._rebuild_from_type_v2':
            obj=self.call(args[0],args[2])
            if isinstance(obj,Instance) and isinstance(args[1],DynamicClass): obj.cls=args[1]
            if len(args)>3: self.build(obj,args[3])
            return obj
        if name in ('builtins.abs','builtins.round','builtins.divmod','builtins.pow',
                    'builtins.repr','builtins.ascii','builtins.chr','builtins.ord','builtins.hex',
                    'builtins.oct','builtins.bin','builtins.format','builtins.hash'):
            if literal(args) and literal(kw):
                if name.endswith('pow') and len(args)>1 and isinstance(args[1],int) and abs(args[1])>10000:
                    raise Unsupported('integer exponent limit')
                return getattr(builtins,name.split('.')[-1])(*args,**kw)
            return UNKNOWN
        if name in ('builtins.callable',): return self.is_callable(args[0])
        if name in ('ast.literal_eval',):
            if not args or not isinstance(args[0],(str,bytes)): raise Halt('literal_eval needs source')
            return ast.literal_eval(args[0])
        if name == 'ast.parse': return Instance(fn,args,dict(kw))
        if name in ('json.loads',):
            if not args or not isinstance(args[0],(str,bytes,bytearray)): raise Halt('json needs text')
            value=json.loads(args[0])
            def walk(x):
                if isinstance(x,list): return [walk(v) for v in x]
                if isinstance(x,dict):
                    x={k:walk(v) for k,v in x.items()}
                    if kw.get('object_pairs_hook'): return self.call(kw['object_pairs_hook'],(list(x.items()),))
                    if kw.get('object_hook'): return self.call(kw['object_hook'],(x,))
                return x
            return walk(value)
        if name == 'json.dumps':
            return json.dumps(*args,**kw) if literal(args) and literal(kw) else UNKNOWN
        if name in ('builtins.bytes.fromhex', 'builtins.bytearray.fromhex'):
            if not args or not isinstance(args[0], str): raise Halt('fromhex needs a string')
            value=bytes.fromhex(args[0])
            return bytearray(value) if name.startswith('builtins.bytearray.') else value
        if name.startswith('builtins.'):
            parts=name.split('.')
            if len(parts)==3 and parts[1] in ('dict','list','tuple','str','bytes','bytearray','set','object'):
                if not args: raise Halt('unbound method needs an object')
                return self.method(args[0],parts[2],args[1:],kw)
        if name in ('_thread.start_new_thread', 'threading._start_new_thread'):
            return self.call(args[0],args[1] if len(args)>1 else (),args[2] if len(args)>2 else {})
        if name in NUMPY_SCALARS:
            if not literal(args) or not literal(kw):
                if args:
                    short=name.split('.')[-1]
                    slot='__bool__' if short in ('bool','bool_') else '__str__' if short=='str_' else '__bytes__' if short=='bytes_' else '__float__' if 'float' in short or short=='longdouble' else '__int__'
                    method=self.special(args[0],slot)
                    if method is not MISSING:
                        args=(self.call(method),)+args[1:]
                if not literal(args) or not literal(kw): return UNKNOWN
            if name=='numpy.void' and args and isinstance(args[0],int) and args[0]>self.MAX_BYTES:
                raise Unsupported('void scalar size limit')
            return getattr(self.numpy_module(),name.split('.')[-1])(*args,**kw)
        if name in ('numpy.lib.npyio.DataSource','numpy.lib._datasource.DataSource'):
            destination=args[0] if args else kw.get('destpath','.')
            if isinstance(destination,Unknown): raise Unsupported('unresolved DataSource destination')
            if not self.truth(destination): self.fail('file-access','DataSource creates a temporary directory')
            return Instance(self.ref('numpy.lib.npyio.DataSource'),args,dict(kw))
        if name in ('numpy.object_', 'numpy.object0'):
            return args[0] if args else None
        if name in ('numpy.core.multiarray.scalar','numpy._core.multiarray.scalar'):
            dtype=args[0] if args else UNKNOWN
            if isinstance(dtype,Instance) and dtype.args and isinstance(dtype.args[0],str) and dtype.args[0].lstrip('<>=|').startswith('O'):
                return args[1] if len(args)>1 else None
            if len(args)>1: return self.numeric_scalar(dtype,args[1])
            return Instance(fn,args)
        if name.endswith('.FunctionTransformer') and name.startswith('sklearn.'):
            attrs={'func':args[0] if args else None,'inverse_func':args[1] if len(args)>1 else None}
            attrs.update(kw)
            return Instance(fn,args,attrs)
        if name in ('collections.UserDict','collections.UserList','collections.UserString'):
            value=args[0] if args else ({} if name.endswith('Dict') else [] if name.endswith('List') else '')
            attrs={'data':value,'_elements':value}
            return Instance(fn,args,attrs)
        if name in ('numpy.core.multiarray._reconstruct','numpy._core.multiarray._reconstruct'):
            subtype=args[0] if args else None
            obj=Instance(subtype if self.isclass(subtype) else self.ref('numpy.ndarray'),args)
            if isinstance(subtype,DynamicClass) and '__array_finalize__' in subtype.attrs:
                self.call(self.bind_descriptor(subtype.attrs['__array_finalize__'],obj),(None,))
            return obj
        if name in ('numpy.apply_along_axis','numpy.apply_over_axes'):
            if len(args)<3: raise Halt('array callback needs function, array and axis')
            if name.endswith('apply_along_axis'):
                array=args[2]
                if isinstance(array,(list,tuple)) and not array: raise Halt('empty array callback axis')
                return self.call(args[0],(array,)+args[3:],kw)
            axes=args[2] if isinstance(args[2],(list,tuple)) else (args[2],)
            array=args[1]
            for axis in axes: array=self.call(args[0],(array,axis))
            return array
        if name in ('torch.empty','torch.zeros','torch.ones','torch.tensor','torch.from_numpy'):
            return Instance(self.ref('torch.Tensor'),args,dict(kw))
        if name in ('numpy.vectorize', 'numpy.frompyfunc'):
            return Wrapper('vectorize',(args[0],))
        if name in ('numpy.array', 'numpy.asarray', 'numpy.asanyarray'):
            return Instance(self.ref('numpy.ndarray'),args,{'_elements':args[0] if args else []})
        if name in ('numpy.fromiter',):
            return Instance(self.ref('numpy.ndarray'),args,{'_elements':self.consume(args[0])})
        if name == 'builtins.reversed':
            if not isinstance(args[0], (list,tuple,str,bytes,bytearray,range)):
                raise Halt('reversed requires a sequence')
            return Lazy('sequence',(args[0][::-1],))
        if name == 'itertools.tee':
            # Independent iterator state with a shared, lazily filled cache.
            count=args[1] if len(args)>1 else 2
            shared=[self.iterator(args[0]),[]]
            return tuple(Lazy('tee',(),source=shared) for _ in range(count))
        if name == 'functools.partial':
            if not args or not self.is_callable(args[0]): raise Halt('partial needs a callable')
            if isinstance(args[0],Partial) and not args[0].namespace_initialized:
                inner=args[0]
                return Partial(inner.fn,inner.args+args[1:],{**inner.kw,**kw})
            return Partial(args[0], args[1:], dict(kw))
        if name in ('builtins.getattr', 'inspect.getattr_static'):
            return self.getattr(args[0], args[1], args[2] if len(args) > 2 else MISSING)
        if name == 'builtins.setattr':
            obj, key, value = args
            if hasattr(obj, 'attrs'): obj.attrs[key] = value
            return None
        if name == 'builtins.delattr':
            if hasattr(args[0], 'attrs'): args[0].attrs[args[1]]=MISSING
            return None
        if name == 'builtins.hasattr':
            return self.getattr(args[0], args[1], MISSING) is not MISSING
        if name in ('builtins.vars',):
            return Namespace(args[0]) if args and hasattr(args[0], 'attrs') else {}
        if name in ('builtins.globals', 'builtins.locals'):
            return Namespace(self.module('builtins'))
        if name in ('builtins.__import__', 'importlib.import_module', 'importlib.__import__'):
            module = args[0]
            if isinstance(module,Unknown): raise Unsupported('unresolved import name')
            if not isinstance(module, str): raise Halt('invalid module name')
            if name == 'builtins.__import__' and '.' in module and not (len(args) > 3 and args[3]) and not kw.get('fromlist'):
                module = module.split('.')[0]
            return self.module(module)
        if name in ('pkgutil.resolve_name', 'logging.config._resolve', 'pydoc.locate'):
            path = args[0]
            if isinstance(path,Unknown): raise Unsupported('unresolved qualified name')
            if not isinstance(path, str): raise Halt('invalid qualified name')
            if ':' in path:
                mod, path = path.split(':', 1)
                obj = self.module(mod)
                for part in path.split('.') if path else []: obj = self.getattr(obj, part)
                return obj
            parts = path.split('.')
            split = max((i for i in range(1, len(parts) + 1) if '.'.join(parts[:i]) in KNOWN_MODULES), default=1)
            obj = self.module('.'.join(parts[:split]))
            for part in parts[split:]: obj = self.getattr(obj, part)
            return obj
        if name in ('operator.attrgetter', 'operator.itemgetter', 'operator.methodcaller'):
            return Wrapper(name.split('.')[1], args, kw)
        if name in ('operator.call', 'functools.apply', 'builtins.apply'):
            return self.call(args[0], args[1:], kw)
        if name in ('copy.copy', 'copy.deepcopy'):
            obj = args[0]
            if isinstance(obj, Partial): return Partial(obj.fn, obj.args, dict(obj.kw), dict(obj.attrs),obj.namespace_initialized)
            if isinstance(obj, Instance): return Instance(obj.cls, obj.args, dict(obj.attrs))
            return obj
        if name in ('functools.lru_cache', 'functools.cache'):
            size=None if name=='functools.cache' else kw.get('maxsize',128)
            if args and self.is_callable(args[0]): return Wrapper('cache',(args[0],),{'maxsize':size,'_cache':{}})
            if args: size=args[0]
            return Wrapper('cache-decorator',(),{'maxsize':size})
        if name in ('functools.update_wrapper', 'functools.wraps'):
            if name.endswith('wraps'): return Wrapper('decorator')
            return args[0]
        if name in ('builtins.staticmethod', 'builtins.classmethod', 'builtins.property'):
            return Wrapper(name.split('.')[1], args, kw)
        if name in ('types.MethodType',):
            return Partial(args[0], (args[1],), {})
        if name == 'builtins.type':
            if len(args) == 3:
                if not isinstance(args[0],str) or not isinstance(args[1],tuple) or not isinstance(args[2],dict):
                    raise Halt('invalid type constructor arguments')
                attrs=dict(args[2])
                if '__eq__' in attrs and '__hash__' not in attrs: attrs['__hash__']=None
                cls=DynamicClass(args[0],args[1],attrs)
                for base in cls.bases:
                    if isinstance(base,DynamicClass) and '__init_subclass__' in base.attrs:
                        hook=base.attrs['__init_subclass__']
                        if isinstance(hook,Wrapper) and hook.kind=='staticmethod': self.call(hook.args[0],(),kw)
                        elif isinstance(hook,Wrapper) and hook.kind=='classmethod': self.call(hook.args[0],(cls,),kw)
                        else: self.call(hook,(cls,),kw)
                        break
                return cls
            if len(args) == 1:
                if isinstance(args[0], Instance): return args[0].cls
                return self.getattr(args[0], '__class__')
            raise Halt('invalid type arguments')
        if name in ('types.ModuleType',): return Module(args[0] if args else '')
        if name == 'types.SimpleNamespace': return Instance(fn, args, dict(kw))
        if name == 'copyreg.add_extension':
            if not hasattr(self,'extensions'): self.extensions={}
            self.extensions[args[2]]=(args[0],args[1]); return None
        if name in ('copyreg.remove_extension','copyreg.clear_extension_cache'):
            if name.endswith('remove_extension') and hasattr(self,'extensions'): self.extensions.pop(args[2],None)
            return None
        if name in ('copyreg._reconstructor',):
            return self.call(args[0], (), new_only=True)
        if name in ('copyreg.__newobj__', 'copyreg.__newobj_ex__'):
            cls = args[0]
            return self.call(cls, args[1] if name.endswith('_ex__') else args[1:],
                             args[2] if name.endswith('_ex__') else {}, new_only=True)
        if name in ('io.BytesIO', 'io.StringIO'):
            return Buffer(args[0] if args else ('' if name.endswith('StringIO') else b''), text=name.endswith('StringIO'))
        if name in ('pickle.loads', 'pickle.load', 'pickle.Unpickler.load', 'pickle._Unpickler.load',
                    'torch.load', 'torch.storage._load_from_bytes', 'joblib.load', 'pandas.read_pickle',
                    'numpy.load', 'numpy.lib.format.read_array'):
            return self.load(name, args, kw)
        if name in ('marshal.loads', 'marshal.load'):
            # No marshal or code object is materialized by the scanner.
            if not args: raise Halt('marshal requires data')
            raw=args[0].data[args[0].pos:] if isinstance(args[0],Buffer) else args[0]
            if not isinstance(raw,(bytes,bytearray)) or not raw: raise Halt('invalid marshal input')
            return Instance(self.ref('marshal.code' if raw[0]&0x7f==ord('c') else 'marshal.value'), args)
        if name in ('pickle.Unpickler',):
            if not args or not isinstance(args[0],(Buffer,Instance)): raise Halt('unpickler needs a file object')
            if len(args)>1: raise Halt('unpickler options are keyword only')
            return Instance(fn, args, dict(kw))
        if name in ('base64.b64decode', 'base64.standard_b64decode', 'base64.urlsafe_b64decode',
                    'base64.decodebytes', 'base64.b32decode', 'base64.b16decode',
                    'base64.b85decode', 'base64.a85decode', 'binascii.a2b_base64',
                    'binascii.unhexlify', 'binascii.a2b_hex'):
            if not args or not isinstance(args[0], (bytes, bytearray, str)): return UNKNOWN
            module, func = name.split('.')
            return getattr(base64 if module == 'base64' else binascii, func)(*args, **kw)
        if name in ('base64.b64encode', 'binascii.hexlify'):
            return base64.b64encode(args[0]) if name.startswith('base64') else binascii.hexlify(args[0])
        if name in ('zlib.decompress', 'gzip.decompress', 'bz2.decompress', 'lzma.decompress'):
            from modelgate.dispatch import decompress
            return decompress(args[0], name.split('.')[0], args[1] if len(args) > 1 else kw.get('wbits'))
        if name in ('_codecs.encode', 'codecs.encode', '_codecs.decode', 'codecs.decode'):
            if not literal(args) or not literal(kw): return UNKNOWN
            encoding = args[1] if len(args) > 1 else kw.get('encoding', 'utf-8')
            # These builtin codecs transform data; never consult an artifact's codec hook.
            if encoding.lower().replace('_', '-').strip() not in {
                'latin1', 'latin-1', 'iso-8859-1', 'utf8', 'utf-8', 'ascii',
                'utf-16', 'utf-16-le', 'utf-16-be', 'utf-32', 'raw-unicode-escape',
                'unicode-escape', 'base64', 'base64-codec', 'hex', 'hex-codec', 'rot13', 'rot-13', 'zlib', 'bz2'}:
                raise Unsupported('unmodelled codec')
            if name.endswith('decode') and encoding in ('zlib','bz2'):
                from modelgate.dispatch import decompress
                return decompress(args[0],encoding)
            return (codecs.encode if name.endswith('encode') else codecs.decode)(*args, **kw)
        if name in ('logging.config.dictConfig', 'logging.config.fileConfig'):
            return self.logging_config(name, args, kw)
        if name in ('builtins.map', 'builtins.filter', 'itertools.starmap', 'itertools.filterfalse',
                    'itertools.takewhile', 'itertools.dropwhile', 'itertools.accumulate'):
            return Lazy(name.split('.')[-1], args)
        if name in ('itertools.chain', 'itertools.chain.from_iterable', 'itertools.repeat',
                    'itertools.cycle', 'itertools.islice', 'itertools.zip_longest',
                    'builtins.zip', 'builtins.enumerate', 'builtins.reversed'):
            return Lazy(name.split('itertools.')[-1].replace('builtins.', ''), args)
        if name == 'builtins.iter':
            return Lazy('calliter', args) if len(args) == 2 else self.iterator(args[0])
        if name == 'builtins.next':
            value = self.next(args[0])
            if value is MISSING:
                if len(args) > 1: return args[1]
                raise Halt('StopIteration')
            return value
        if name in ('functools.reduce',):
            seq = self.consume(args[1]); initial = args[2] if len(args) > 2 else MISSING
            if initial is MISSING:
                if not seq: raise Halt('reduce of empty sequence')
                initial, seq = seq[0], seq[1:]
            for x in seq: initial = self.call(args[0], (initial, x))
            return initial
        if name in ('builtins.sorted', 'builtins.min', 'builtins.max'):
            seq = self.consume(args[0]) if len(args) == 1 else list(args)
            key = kw.get('key')
            keys = [self.call(key, (x,)) for x in seq] if key is not None else seq
            if len(keys)>1 and any(isinstance(x,Instance) and isinstance(x.cls,DynamicClass) for x in keys):
                import functools
                def compare(a,b):
                    fn=self.special(a,'__lt__')
                    if fn is not MISSING and self.truth(self.call(fn,(b,))): return -1
                    fn=self.special(b,'__lt__')
                    if fn is not MISSING and self.truth(self.call(fn,(a,))): return 1
                    return 0
                order=sorted(range(len(keys)),key=functools.cmp_to_key(lambda i,j:compare(keys[i],keys[j])))
                if name.endswith('sorted'):
                    if kw.get('reverse'): order.reverse()
                    return [seq[i] for i in order]
                return seq[order[0] if name.endswith('min') else order[-1]]
            if name.endswith('sorted'):
                if literal(keys): return [x for _, x in sorted(zip(keys, seq), key=lambda p:p[0], reverse=kw.get('reverse', False))]
                return seq
            if not seq: return kw.get('default', UNKNOWN)
            if literal(keys): return seq[(min if name.endswith('min') else max)(range(len(seq)), key=lambda i:keys[i])]
            return seq[0]
        if name in ('builtins.any', 'builtins.all'):
            it = self.iterator(args[0]); is_all = name.endswith('all')
            for _ in range(self.MAX_ITEMS):
                x = self.next(it)
                if x is MISSING: return is_all
                if self.truth(x) != is_all: return not is_all
            raise Unsupported('iterator work limit exceeded')
        if name in ('builtins.list', 'builtins.tuple', 'builtins.set', 'builtins.frozenset', 'collections.deque'):
            values = self.consume(args[0]) if args else []
            if name in ('builtins.set','builtins.frozenset'):
                for value in values: self.hash_key(value)
            return {'builtins.list':list, 'builtins.tuple':tuple, 'builtins.set':set,
                    'builtins.frozenset':frozenset, 'collections.deque':list}[name](values)
        if name == 'collections.defaultdict':
            factory=args[0] if args else None
            mapping=dict(args[1]) if len(args)>1 else {}
            mapping.update(kw)
            return Instance(fn,(),{'_mapping':mapping,'default_factory':factory})
        if name == 'collections.Counter':
            mapping={}
            if args:
                if isinstance(args[0],dict): mapping.update(args[0])
                else:
                    for x in self.consume(args[0]): mapping[x]=mapping.get(x,0)+1
            mapping.update(kw); return mapping
        if name in ('builtins.dict', 'collections.OrderedDict', 'collections.defaultdict', 'collections.Counter'):
            if name == 'collections.defaultdict': args = args[1:]
            result = {}
            if args:
                if isinstance(args[0], dict): result.update(args[0])
                elif isinstance(args[0], Namespace): result.update(args[0].owner.attrs)
                else:
                    for pair in self.consume(args[0]): self.setitem(result,pair[0],pair[1])
            result.update(kw)
            return result
        if name == 'builtins.sum':
            values = self.consume(args[0]); total = args[1] if len(args)>1 else 0
            for x in values: total = self.binary('add', total, x)
            return total
        if name=='builtins.len' and args:
            method=self.special(args[0],'__len__')
            if method is not MISSING: return self.call(method)
        if name == 'builtins.len':
            return len(args[0]) if isinstance(args[0], (str,bytes,bytearray,list,tuple,dict,set,frozenset)) else UNKNOWN
        if name in ('builtins.bool','builtins.str','builtins.bytes','builtins.int','builtins.float','builtins.complex') and args:
            kind=name.split('.')[-1]
            if kind=='bool': return self.truth(args[0])
            methods={'str':('__str__','__repr__'),'bytes':('__bytes__',),'int':('__int__','__index__'),
                     'float':('__float__','__index__'),'complex':('__complex__','__float__','__index__')}
            for method in methods.get(kind,()):
                fn=self.special(args[0],method)
                if fn is not MISSING: return self.call(fn)
        if name in ('builtins.str', 'builtins.bytes', 'builtins.bytearray', 'builtins.int',
                    'builtins.float', 'builtins.bool', 'builtins.complex'):
            kind = name.split('.')[1]
            if kind in ('bytes','bytearray') and args and isinstance(args[0], Lazy):
                args = (self.consume(args[0]),) + args[1:]
            if literal(args) and literal(kw):
                if args and isinstance(args[0], int) and kind in ('bytes','bytearray') and args[0] > self.MAX_BYTES:
                    raise Unsupported('byte string size limit')
                return {'str':str,'bytes':bytes,'bytearray':bytearray,'int':int,'float':float,'bool':bool,'complex':complex}[kind](*args, **kw)
            if kind == 'str' and args and classname(args[0]).startswith('pathlib.'):
                return self.path_string(args[0])
            return UNKNOWN
        if name in ('builtins.range', 'builtins.slice'):
            if not literal(args): return UNKNOWN
            return range(*args) if name.endswith('range') else slice(*args)
        if name in ('operator.getitem', 'operator.__getitem__'):
            return self.getitem(args[0], args[1])
        if name in ('operator.setitem', 'operator.__setitem__'):
            self.setitem(*args); return None
        if name.startswith('operator.'):
            op = name.split('.')[1].strip('_')
            if op in ('add','concat','iadd','iconcat','mul','imul','or','xor','and','sub','truediv','floordiv','mod','pow','rshift','lshift'):
                return self.binary(op, args[0], args[1])
            if op in ('index','length_hint','truth','not','neg','pos','inv','invert','eq','ne','lt','le','gt','ge') and literal(args):
                func = getattr(operator, op + '_' if op in ('not','and','or') else op, None)
                return func(*args) if func else UNKNOWN
        if name in ('os.fspath', 'posixpath.normpath', 'posixpath.abspath', 'posixpath.basename',
                    'posixpath.dirname', 'posixpath.join', 'ntpath.join', 'ntpath.normpath'):
            if name == 'os.fspath': return self.path_string(args[0])
            if literal(args): return getattr(posixpath, name.split('.')[-1])(*args)
            return UNKNOWN
        if name.startswith('pathlib.') and name.rsplit('.',1)[-1] in ('Path','PosixPath','WindowsPath','PurePath','PurePosixPath','PureWindowsPath'):
            return Instance(fn, args)
        if name in ('ftplib.FTP', 'ftplib.FTP_TLS', 'smtplib.SMTP', 'smtplib.SMTP_SSL', 'telnetlib.Telnet'):
            host = args[0] if args else kw.get('host')
            if self.truth(host): self.fail('network', f'call {name} with a host')
            return Instance(fn, args, dict(kw))
        if name in ('http.client.HTTPConnection', 'http.client.HTTPSConnection', 'socket.socket',
                    'xmlrpc.client.ServerProxy', 'ssl.SSLContext'):
            return Instance(fn, args, dict(kw))
        if name in ('threading.Thread', 'threading.Timer', 'multiprocessing.Process', 'multiprocessing.context.Process'):
            attrs = dict(kw)
            if name == 'threading.Timer':
                attrs.update(target=args[1] if len(args)>1 else kw.get('function'),
                             args=args[2] if len(args)>2 else kw.get('args',()),
                             kwargs=args[3] if len(args)>3 else kw.get('kwargs',{}))
            else:
                if len(args)>1: attrs['target']=args[1]
                if len(args)>3: attrs['args']=args[3]
                if len(args)>4: attrs['kwargs']=args[4]
            return Instance(fn, args, attrs)
        if name in ('multiprocessing.pool.ThreadPool','multiprocessing.dummy.Pool'):
            initializer=args[1] if len(args)>1 else kw.get('initializer')
            initargs=args[2] if len(args)>2 else kw.get('initargs',())
            if initializer is not None: self.call(initializer,tuple(initargs))
            return Instance(self.ref('multiprocessing.pool.ThreadPool'),args,dict(kw))
        if name in ('logging.handlers.SocketHandler','logging.handlers.DatagramHandler',
                    'logging.handlers.HTTPHandler','logging.handlers.SMTPHandler','logging.NullHandler'):
            return Instance(fn,args,dict(kw))
        if name in ('logging.FileHandler','logging.handlers.RotatingFileHandler',
                    'logging.handlers.TimedRotatingFileHandler','logging.handlers.WatchedFileHandler') and kw.get('delay'):
            return Instance(fn,args,dict(kw))
        if name in ('concurrent.futures.ThreadPoolExecutor', 'concurrent.futures.ProcessPoolExecutor',
                    'concurrent.futures.thread.ThreadPoolExecutor', 'concurrent.futures.process.ProcessPoolExecutor'):
            return Instance(fn, args, dict(kw))
        if name in ('timeit.Timer','timeit.timeit','timeit.repeat'):
            stmt=args[0] if args else kw.get('stmt',MISSING)
            setup=args[1] if len(args)>1 else kw.get('setup',MISSING)
            if any(isinstance(x,(str,bytes)) for x in (stmt,setup)):
                self.fail('code-execution', name+' compiles supplied Python source')
            if name=='timeit.Timer':
                return Instance(fn,args,{'stmt':stmt,'setup':setup})
            for cb in (setup,stmt):
                if cb is not MISSING: self.call(cb)
            return 0.0
        if name in ('weakref.finalize', 'atexit.register'):
            # The registry also observes effects during interpreter shutdown.
            cb = args[1] if name.startswith('weakref') else args[0]
            cbargs = args[2:] if name.startswith('weakref') else args[1:]
            return self.call(cb, cbargs, kw)
        if name in ('numpy.dtype',):
            return Instance(fn, args)
        maximum=MAX_POSITIONAL_ARGUMENTS.get(name)
        if maximum is not None and len(args)>maximum: raise Halt('too many callable arguments')
        required=REQUIRED_ARGUMENTS.get(name,())
        if any(i>=len(args) and key not in kw for i,key in enumerate(required)):
            raise Halt('missing required callable arguments')
        if name in ('builtins.eval','builtins.exec','builtins.compile'):
            src=args[0] if args else kw.get('source')
            if not isinstance(src,(str,bytes,bytearray,Unknown,Ref,Instance)):
                raise Halt('source must be a string or code')
            if name in ('builtins.eval','builtins.exec'):
                if len(args)>1 and args[1] is not None and not isinstance(args[1],dict):
                    raise Halt('eval/exec globals must be a dict')
                if len(args)>2 and args[2] is not None and not isinstance(args[2],(dict,Namespace,Unknown)):
                    obj=args[2]
                    if not isinstance(obj,Instance) or self.special(obj,'__getitem__') is MISSING:
                        raise Halt('eval/exec locals must be a mapping')
            if name=='builtins.eval' and kw or name=='builtins.exec' and any(k!='closure' for k in kw):
                raise Halt('eval/exec positional-only argument passed by keyword')
        if name in ('os.system','os.popen'):
            command=args[0] if args else kw.get('command',kw.get('cmd'))
            if not isinstance(command,(str,bytes,Unknown)):
                raise Halt('command must be a string')
        if name in ('types.FunctionType','types.LambdaType'):
            code=args[0] if args else kw.get('code')
            if not isinstance(code,Unknown) and classname(code) not in ('marshal.code','types.CodeType'):
                raise Halt('function needs a code object')
        if name in ('os.walk','os.fwalk','glob.iglob','pathlib.Path.iterdir','pathlib.Path.glob','pathlib.Path.rglob'):
            return Lazy('file-iterator',(name,args,kw))
        kind = CALL_CAPABILITIES.get(name)
        if kind:
            if kind != 'file-access' or self.file_effect(args, name, kw):
                self.fail(kind, f'call {name}')
            if name in ('builtins.open','io.open'):
                return Buffer(self.artifact_data or b'')
            return UNKNOWN
        if name in ('ctypes.CDLL', 'ctypes.PyDLL', 'ctypes.cdll.LoadLibrary', 'ctypes.pydll.LoadLibrary'):
            library=args[0] if args else kw.get('name',MISSING)
            if library is MISSING: raise Halt('library name is required')
            if library is None: return Instance(self.ref('ctypes.CDLL'),args,dict(kw))
            self.fail('file-access', f'load external native library via {name}')
        if name in ('ctypes.CFUNCTYPE', 'ctypes.PYFUNCTYPE', 'ctypes.WINFUNCTYPE'):
            return Wrapper('native-function-type', args)
        if name.startswith('math.') or name.startswith('cmath.'):
            if literal(args) and literal(kw):
                if name in ('math.factorial','math.comb','math.perm') and any(isinstance(x,int) and abs(x)>100000 for x in args):
                    raise Unsupported('large integer numerical operation')
                module=math
                if name.startswith('cmath.'):
                    import cmath
                    module=cmath
                func=getattr(module,name.split('.')[-1],None)
                if func is None: raise Halt('numerical function does not exist')
                return func(*args,**kw)
            return UNKNOWN
        safe_prefixes=('numpy.','sklearn.','collections.','datetime.','decimal.','array.',
                       '_decimal.','fractions.','uuid.','enum.','random.','statistics.',
                       'torch._utils._rebuild_', 'torch._tensor._rebuild_')
        safe_names={'re._compile','re.compile','builtins.object','builtins.isinstance','builtins.issubclass',
                    'builtins.enumerate','types.MappingProxyType','types.GenericAlias',
                    'bdb.Bdb','pdb.Pdb','code.InteractiveInterpreter','code.InteractiveConsole',
                    'logging.StreamHandler','logging.Formatter','logging.Filter',
                    'torch.Tensor','torch.Size','torch.device','torch.dtype',
                    'joblib.numpy_pickle.NumpyArrayWrapper'}
        if name.startswith(safe_prefixes) or name in safe_names:
            return Instance(fn,args,dict(kw))
        raise Unsupported('unmodelled callable '+name)

    def numpy_module(self):
        # As with h5py, CPU version discovery must not launch uname -p.
        import platform
        uname=platform.uname(); uname.processor=uname.machine
        import numpy
        return numpy

    def numeric_scalar(self,dtype,data):
        if not isinstance(dtype,Instance) or refname(dtype.cls)!='numpy.dtype' or not dtype.args:
            return Instance(self.ref('numpy.scalar'),(dtype,data))
        spec=dtype.args[0]
        state=dtype.attrs.get('_pickle_state')
        if not isinstance(spec,str) or not isinstance(data,(bytes,bytearray)):
            return Instance(self.ref('numpy.scalar'),(dtype,data))
        if isinstance(state,tuple) and len(state)>4 and state[4] is not None:
            return Instance(self.ref('numpy.scalar'),(dtype,data))
        np=self.numpy_module()
        native=np.dtype(spec)
        if native.hasobject: return data
        if isinstance(state,tuple) and len(state)>1 and state[1] in ('<','>','=','|'):
            native=native.newbyteorder(state[1])
        return np._core.multiarray.scalar(native,bytes(data))

    def is_callable(self, value):
        return isinstance(value, (Ref, Partial, Bound, Wrapper, DynamicClass)) or (isinstance(value,Instance) and isinstance(value.cls,DynamicClass) and '__call__' in value.cls.attrs)

    def binary(self, op, a, b):
        if op in ('add','concat','iadd','iconcat'):
            if type(a) is type(b) and type(a) in (str,bytes,bytearray,tuple,list):
                if len(a)+len(b)>self.MAX_BYTES: raise Unsupported('assembled data size limit')
                if op in ('iadd','iconcat') and isinstance(a,(list,bytearray)):
                    a+=b; return a
                return a+b
        if op in ('mul','imul') and isinstance(a, (str,bytes,bytearray,tuple,list)) and isinstance(b,int):
            if len(a)*b>self.MAX_BYTES: raise Unsupported('assembled data size limit')
            if op=='imul' and isinstance(a,(list,bytearray)):
                a*=b; return a
            return a*b
        if literal(a) and literal(b):
            if op in ('pow','lshift','rshift') and isinstance(b,int) and abs(b)>10000:
                raise Unsupported('integer operation limit')
            if isinstance(a,int) and isinstance(b,int) and op in ('pow','mul','imul'):
                bits=a.bit_length()*max(0,b) if op=='pow' else a.bit_length()+b.bit_length()
                if bits>self.MAX_BYTES*8: raise Unsupported('integer result size limit')
            aliases={'concat':'add','iadd':'add','iconcat':'add','imul':'mul','or':'or_','and':'and_'}
            return getattr(operator, aliases.get(op,op))(a,b)
        return UNKNOWN

    def wrapper(self, fn, args, kw):
        if fn.kind=='cache-decorator':
            return Wrapper('cache',(args[0],),{'maxsize':fn.kw.get('maxsize',128),'_cache':{}})
        if fn.kind=='cache':
            key=(args,tuple(sorted(kw.items())))
            self.hash_key(key)
            try: hash(key)
            except TypeError: raise Halt('unhashable cache argument')
            cache=fn.kw['_cache']
            if key in cache: return cache[key]
            value=self.call(fn.args[0],args,kw)
            size=fn.kw.get('maxsize')
            if size is None or size>0:
                if size is not None and len(cache)>=size: cache.pop(next(iter(cache)))
                cache[key]=value
            return value
        if fn.kind in ('forward','staticmethod','classmethod'):
            return self.call(fn.args[0], args, kw)
        if fn.kind == 'decorator': return Wrapper('forward', (args[0],))
        if fn.kind == 'attrgetter':
            def get(path):
                obj = args[0]
                for part in path.split('.'): obj=self.getattr(obj,part)
                return obj
            result = tuple(get(p) for p in fn.args)
            return result[0] if len(result)==1 else result
        if fn.kind == 'itemgetter':
            result = tuple(self.getitem(args[0], key) for key in fn.args)
            return result[0] if len(result)==1 else result
        if fn.kind == 'methodcaller':
            return self.call(self.getattr(args[0], fn.args[0]), fn.args[1:], fn.kw)
        if fn.kind == 'codec':
            value=self.call(self.ref('codecs.'+fn.args[1]),(args[0],fn.args[0])+args[1:],kw)
            return value,len(args[0])
        if fn.kind == 'vectorize':
            seqs=[self.consume(x) if isinstance(x,(list,tuple,Lazy)) else [x] for x in args]
            return [self.call(fn.args[0],tuple(values),kw) for values in zip(*seqs)]
        if fn.kind == 'native-function-type':
            self.fail('code-execution', 'materialize native function pointer')
        raise Halt('wrapper is not callable')

    def method(self, obj, name, args, kw):
        if isinstance(obj,Wrapper) and obj.kind=='cache' and name=='cache_clear':
            obj.kw['_cache'].clear(); return None
        if name == '__call__': return self.call(obj, args, kw)
        if name == '__getattribute__': return self.getattr(obj, args[0])
        if name == '__setattr__':
            if hasattr(obj,'attrs'): obj.attrs[args[0]]=args[1]
            return None
        if name == '__delattr__':
            if hasattr(obj,'attrs'): obj.attrs[args[0]]=MISSING
            return None
        if name == '__get__': return self.bind_descriptor(obj, args[0])
        if name == '__setstate__': self.build(obj, args[0]); return None
        if name in ('__getitem__', 'get'):
            return self.getitem(obj,args[0], args[1] if len(args)>1 else (None if name=='get' else MISSING))
        if name == '__setitem__': self.setitem(obj,*args); return None
        np=sys.modules.get('numpy')
        if np is not None and isinstance(obj,np.generic) and not obj.dtype.hasobject:
            if name in ('item','tolist','tobytes','tostring') and literal(args) and literal(kw):
                return getattr(obj,name)(*args,**kw)
        if isinstance(obj,Decoder):
            if name=='decompress':
                if not args or not isinstance(args[0],(bytes,bytearray)): raise Halt('decompress needs bytes')
                limit=min(args[1],self.MAX_BYTES+1) if len(args)>1 and args[1]>0 else self.MAX_BYTES+1
                value=obj.engine.decompress(args[0],limit)
                if len(value)>self.MAX_BYTES: raise Unsupported('expanded data size limit')
                return value
            if name=='flush':
                if obj.kind!='zlib': return b''
                result=bytearray()
                while obj.engine.unconsumed_tail:
                    previous=obj.engine.unconsumed_tail
                    result.extend(obj.engine.decompress(previous,self.MAX_BYTES-len(result)+1))
                    if len(result)>self.MAX_BYTES: raise Unsupported('expanded data size limit')
                    if obj.engine.unconsumed_tail==previous: break
                result.extend(obj.engine.flush())
                if len(result)>self.MAX_BYTES: raise Unsupported('expanded data size limit')
                return bytes(result)
        if isinstance(obj, Namespace):
            obj = obj.owner.attrs
        if isinstance(obj, Buffer):
            if name in ('read','readline','readlines','getvalue','getbuffer'):
                if name in ('getvalue','getbuffer'): return obj.data
                if name=='readlines': return obj.data[obj.pos:].splitlines(True)
                end = len(obj.data) if not args or args[0]<0 else min(len(obj.data),obj.pos+args[0])
                if name=='readline':
                    i=obj.data.find('\n' if obj.text else b'\n',obj.pos,end)
                    if i>=0: end=i+1
                value=obj.data[obj.pos:end]; obj.pos=end; return value
            if name=='seek':
                obj.pos=(0 if len(args)<2 or args[1]==0 else obj.pos if args[1]==1 else len(obj.data))+args[0]
                return obj.pos
            if name=='tell': return obj.pos
            if name=='write':
                b=args[0]; obj.data=obj.data[:obj.pos]+b+obj.data[obj.pos+len(b):]; obj.pos+=len(b); return len(b)
            if name in ('close','flush'): return None
        if isinstance(obj, Lazy):
            if name in ('__next__','next'): return self.call(self.ref('builtins.next'),(obj,))
            if name=='__iter__': return obj
        if isinstance(obj, Instance):
            cls=classname(obj)
            if '_zip_entries' in obj.attrs or '_tar_entries' in obj.attrs:
                entries=obj.attrs.get('_zip_entries',obj.attrs.get('_tar_entries'))
                if name in ('read','open','extractfile'):
                    key=args[0] if args else ''
                    if not isinstance(key,str): raise Halt('archive member needs a name')
                    if key not in entries: raise Halt('archive member does not exist')
                    return entries[key] if name=='read' else Buffer(entries[key])
                if name in ('namelist','getnames'): return list(entries)
                if name=='close': return None
            if cls.endswith('.FunctionTransformer') and name in ('transform','fit_transform','inverse_transform'):
                fn=obj.attrs.get('inverse_func' if name=='inverse_transform' else 'func')
                kwds=obj.attrs.get('inv_kw_args' if name=='inverse_transform' else 'kw_args') or {}
                return self.call(fn,args,kwds) if fn is not None else args[0] if args else None
            if cls=='collections.defaultdict' and name=='__missing__':
                return self.getitem(obj,args[0])
            if cls=='numpy.ndarray' and name in ('tobytes','tostring'):
                state=obj.attrs.get('_pickle_state')
                if isinstance(state,tuple) and len(state)>4 and isinstance(state[4],(bytes,bytearray)):
                    if not state[3] or len(state[1])<=1 or kw.get('order') in ('F','A'):
                        return bytes(state[4])
                raise Unsupported('cannot resolve ndarray byte order/data')
            if cls=='numpy.ndarray' and '_elements' in obj.attrs:
                elements=obj.attrs['_elements']
                if name=='tolist': return elements
                if name=='item': return self.getitem(elements,args[0] if args else 0)
                if name in ('flatten','ravel','reshape'): return obj
            if name in obj.attrs: return self.call(obj.attrs[name],args,kw)
            if cls.startswith('pathlib.'):
                if name in ('joinpath','__truediv__'):
                    return Instance(obj.cls, obj.args+args)
                if name in ('__str__','__fspath__','as_posix'): return self.path_string(obj)
                return self.call(self.ref('pathlib.Path.'+name),(obj,)+args,kw)
            if cls=='timeit.Timer' and name in ('timeit','repeat','autorange'):
                for cb in (obj.attrs.get('setup',MISSING),obj.attrs.get('stmt',MISSING)):
                    if cb is not MISSING: self.call(cb)
                return 0.0
            if cls=='pickle.Unpickler' and name=='load': return self.load('pickle.load',obj.args, obj.attrs)
            if cls.startswith(('threading.','multiprocessing.')) and name in ('start','run'):
                if name=='start' and cls.startswith('multiprocessing.'):
                    self.fail('process', f'{cls}.start')
                target=obj.attrs.get('target',obj.attrs.get('_target'))
                return self.call(target,tuple(obj.attrs.get('args',obj.attrs.get('_args',()))), obj.attrs.get('kwargs',obj.attrs.get('_kwargs',{}))) if target else None
            if cls=='multiprocessing.pool.ThreadPool':
                if name in ('close','join','terminate'): return None
                if name in ('apply','apply_async'):
                    return self.call(args[0],tuple(args[1]) if len(args)>1 else (),args[2] if len(args)>2 else {})
                if name in ('map','map_async','imap','imap_unordered','starmap','starmap_async'):
                    return self.consume(Lazy('starmap' if name.startswith('starmap') else 'map',args[:2]))
            if cls.startswith('logging.') and 'FileHandler' in cls and name in ('emit','_open'):
                self.fail('file-access',cls+'.'+name)
            if 'Executor' in cls and name in ('submit','map'):
                if not args: raise Halt('executor needs a callable')
                def start_worker():
                    if 'ProcessPool' in cls: self.fail('process',cls+'.'+name)
                    initializer=obj.attrs.get('initializer')
                    if initializer and not obj.attrs.get('_initialized'):
                        self.call(initializer,tuple(obj.attrs.get('initargs',())))
                        obj.attrs['_initialized']=True
                if name=='submit':
                    start_worker()
                    result=self.call(args[0],args[1:],kw)
                    return Instance(self.ref('concurrent.futures.Future'),(),{'_result':result})
                if len(args)<2: raise Halt('executor.map needs iterables')
                sources=[self.iterator(s) for s in args[1:]]
                result=[]
                for _ in range(self.MAX_ITEMS):
                    values=tuple(self.next(s) for s in sources)
                    if any(v is MISSING for v in values): return result
                    start_worker()
                    result.append(self.call(args[0],values))
                raise Unsupported('executor map work limit exceeded')
            if cls=='concurrent.futures.Future' and name=='result': return obj.attrs.get('_result',UNKNOWN)
            if cls=='xmlrpc.client.ServerProxy' and not name.startswith('_'):
                self.fail('network', f'XML-RPC request {name}')
            if cls in ('ctypes.CDLL','ctypes.PyDLL'):
                if name in ('system','popen','fork','execv','execve','execl','spawnv'):
                    self.fail('process','native '+name)
                if name.startswith(('PyRun_','Py_CompileString','PyEval_')):
                    self.fail('code-execution','native Python '+name)
                if name in ('open','fopen','freopen','remove','unlink','rename','mkdir','rmdir'):
                    self.fail('file-access','native '+name)
                if name in ('connect','getaddrinfo'): self.fail('network','native '+name)
                raise Unsupported('unmodelled native function '+name)
            if cls=='marshal.code': return UNKNOWN
            fullname=cls+'.'+name
            if fullname in CALL_CAPABILITIES:
                return self.call(self.ref(fullname),(obj,)+args,kw)
            if cls in ('socket.socket','http.client.HTTPConnection','http.client.HTTPSConnection',
                       'ftplib.FTP','ftplib.FTP_TLS','smtplib.SMTP','smtplib.SMTP_SSL'):
                return self.call(self.ref(fullname),(obj,)+args,kw)
            if isinstance(obj.cls,DynamicClass):
                return self.call(self.getattr(obj,name),args,kw)
            return UNKNOWN
        if isinstance(obj, dict):
            if name=='update':
                if args:
                    other=args[0].owner.attrs if isinstance(args[0],Namespace) else args[0]
                    obj.update(other)
                obj.update(kw); return None
            if name in ('items','keys','values'): return list(getattr(obj,name)())
            if name=='setdefault':
                self.hash_key(args[0]); return obj.setdefault(*args)
            if name=='pop':
                self.hash_key(args[0]); return obj.pop(*args)
            if name=='copy': return dict(obj)
        if isinstance(obj,list):
            if name in ('__add__','__iadd__','__mul__','__imul__'): return self.binary(name.strip('_'),obj,args[0])
            if name=='append': obj.append(args[0]); return None
            if name=='extend': obj.extend(self.consume(args[0])); return None
            if name=='pop': return obj.pop(*args)
            if name=='insert': obj.insert(*args); return None
            if name=='sort':
                result=self.call(self.ref('builtins.sorted'),(obj,),kw); obj[:]=result; return None
        if isinstance(obj,(str,bytes,bytearray)):
            safe={'encode','decode','join','replace','split','rsplit','strip','lstrip','rstrip',
                  'lower','upper','capitalize','title','swapcase','hex','fromhex','startswith','endswith',
                  'removeprefix','removesuffix','splitlines','find','rfind','partition','rpartition',
                  'zfill','ljust','rjust','center','__add__','__mul__','__mod__'}
            if name=='join' and args:
                args=(self.consume(args[0]),)+args[1:]
            if name in safe and literal(args) and literal(kw): return getattr(obj,name)(*args,**kw)
        if name in ('__iter__',): return self.iterator(obj)
        return UNKNOWN

    def iterator(self, obj):
        if isinstance(obj, Lazy): return obj
        if isinstance(obj, dict): return Lazy('sequence',(list(obj),))
        if isinstance(obj, (list,tuple,set,frozenset,str,bytes,bytearray,range)):
            return Lazy('sequence',(obj if not isinstance(obj,(set,frozenset)) else list(obj),))
        if isinstance(obj,Instance) and classname(obj)=='collections.defaultdict':
            return self.iterator(obj.attrs['_mapping'])
        if isinstance(obj, Instance) and '_elements' in obj.attrs:
            return self.iterator(obj.attrs['_elements'])
        if isinstance(obj, Instance) and isinstance(obj.cls,DynamicClass):
            return self.call(self.getattr(obj,'__iter__'))
        if isinstance(obj, Unknown): raise Unsupported('cannot resolve consumed iterator')
        raise Halt('object is not iterable')

    def consume(self, obj):
        if isinstance(obj,(list,tuple,set,frozenset,dict,str,bytes,bytearray,range)):
            if len(obj)>self.MAX_ITEMS: raise Unsupported('iterator size limit')
            return list(obj)
        it=self.iterator(obj); result=[]
        for _ in range(self.MAX_ITEMS):
            value=self.next(it)
            if value is MISSING: return result
            result.append(value)
        raise Unsupported('iterator work limit exceeded')

    def next(self, it):
        self.tick()
        if not isinstance(it,Lazy):
            if isinstance(it,Instance): return self.call(self.getattr(it,'__next__'))
            raise Halt('not an iterator')
        if it.done: return MISSING
        kind,args=it.kind,it.args
        if kind=='file-iterator':
            name,callargs,callkw=args
            if self.file_effect(callargs,name,callkw): self.fail('file-access','consume '+name)
            return MISSING
        if kind=='sequence':
            if it.index>=len(args[0]): it.done=True; return MISSING
            value=args[0][it.index]; it.index+=1; return value
        if kind=='calliter':
            value=self.call(args[0])
            if value==args[1]: it.done=True; return MISSING
            return value
        if kind in ('map','zip','zip_longest','starmap','filter','filterfalse','takewhile','dropwhile','enumerate','accumulate'):
            if it.source is None:
                seqs=args[1:] if kind in ('map','starmap','filter','filterfalse','takewhile','dropwhile') else args[:1] if kind in ('enumerate','accumulate') else args
                it.source=[self.iterator(s) for s in seqs]
            if kind in ('filter','filterfalse','takewhile','dropwhile'):
                for _ in range(self.MAX_ITEMS):
                    value=self.next(it.source[0])
                    if value is MISSING: it.done=True; return MISSING
                    passed=self.truth(self.call(args[0],(value,))) if args[0] is not None else self.truth(value)
                    if kind=='takewhile' and not passed: it.done=True; return MISSING
                    if kind=='dropwhile':
                        if it.index or not passed: it.index=1; return value
                    elif passed != (kind=='filterfalse'): return value
                raise Unsupported('iterator work limit exceeded')
            values=tuple(self.next(s) for s in it.source)
            if any(x is MISSING for x in values): it.done=True; return MISSING
            if kind=='map': return self.call(args[0],values)
            if kind=='starmap': return self.call(args[0],tuple(values[0]))
            if kind=='enumerate':
                value=((args[1] if len(args)>1 else 0)+it.index,values[0]); it.index+=1; return value
            if kind=='accumulate':
                val=values[0]
                if it.index:
                    fn=args[1] if len(args)>1 else self.ref('operator.add')
                    val=self.call(fn,(it.args[-1],val))
                it.index+=1
                # Store accumulator separately from the immutable source tuple.
                it.args=(args[0], args[1] if len(args)>1 else self.ref('operator.add'), val)
                return val
            return values
        if kind=='repeat':
            if len(args)>1 and it.index>=args[1]: it.done=True; return MISSING
            it.index+=1; return args[0]
        if kind=='cycle':
            if it.source is None: it.source=[self.iterator(args[0]), [], False]
            source,cache,complete=it.source
            if not complete:
                value=self.next(source)
                if value is not MISSING:
                    cache.append(value); return value
                it.source[2]=True
            if not cache: return MISSING
            value=cache[it.index%len(cache)]; it.index+=1; return value
        if kind=='tee':
            source,cache=it.source
            if it.index==len(cache):
                value=self.next(source)
                if value is MISSING: return MISSING
                cache.append(value)
            value=cache[it.index]; it.index+=1; return value
        if kind=='reversed':
            if it.source is None: it.source=self.consume(args[0])[::-1]
            return self.next_sequence(it,it.source)
        if kind in ('chain','chain.from_iterable'):
            if it.source is None:
                it.source=[self.iterator(args[0]) if kind.endswith('from_iterable') else self.iterator(args), None]
            while True:
                outer,inner=it.source
                if inner is None:
                    seq=self.next(outer)
                    if seq is MISSING: it.done=True; return MISSING
                    inner=self.iterator(seq); it.source[1]=inner
                value=self.next(inner)
                if value is not MISSING: return value
                it.source[1]=None
        if kind=='islice':
            if it.source is None: it.source=self.iterator(args[0])
            start,stop,step=(0,args[1],1) if len(args)==2 else (args[1],args[2],args[3] if len(args)>3 else 1)
            while stop is None or it.index<stop:
                i=it.index; it.index+=1; value=self.next(it.source)
                if value is MISSING: it.done=True; return MISSING
                if i>=start and (i-start)%step==0: return value
            it.done=True; return MISSING
        raise Unsupported('unmodelled lazy iterator '+kind)

    def next_sequence(self,it,seq):
        if it.index>=len(seq): return MISSING
        value=seq[it.index]; it.index+=1; return value

    def load(self, name, args, kw):
        if not args: raise Halt('loader needs input')
        data=args[0]
        if isinstance(data,Unknown): raise Unsupported('unresolved nested loader input')
        if name in ('pickle.loads','pickle.load') and len(args)>1:
            raise Halt('pickle loader options are keyword only')
        if name in ('pickle.load','pickle.Unpickler.load','pickle._Unpickler.load','numpy.lib.format.read_array') and not isinstance(data,(Buffer,Instance)):
            raise Halt('loader requires a file object, not a path')
        if (name.endswith('loads') or name=='torch.storage._load_from_bytes') and not isinstance(data,(bytes,bytearray)):
            raise Halt('loads needs bytes')
        if isinstance(data, Buffer):
            raw=data.data[data.pos:]
        elif isinstance(data,(bytes,bytearray)) and (name.endswith('loads') or name=='torch.storage._load_from_bytes'):
            raw=bytes(data)
        elif isinstance(data,(str,bytes,Instance)):
            if self.inside_artifact(data) and self.artifact_data is not None:
                raw=self.artifact_data
            else:
                self.fail('file-access', f'{name} reads a path')
        else:
            raise Halt('loader needs a path or file object')
        if not isinstance(raw,(bytes,bytearray)): raise Halt('binary loader needs bytes')
        if name=='pandas.read_pickle' and len(args)>1:
            kw={**kw,'compression':args[1]}
        if name=='numpy.load':
            kw=dict(kw)
            if len(args)>1: kw.setdefault('mmap_mode',args[1])
            if len(args)>2: kw.setdefault('allow_pickle',args[2])
        elif name=='numpy.lib.format.read_array' and len(args)>1:
            kw={**kw,'allow_pickle':args[1]}
        if self.load_callback is None: raise Unsupported('nested loader unavailable')
        value,used=self.load_callback(bytes(raw), self.location+'!'+name, self, name, kw)
        if isinstance(data,Buffer): data.pos+=used
        return value

    def logging_config(self,name,args,kw):
        if name.endswith('fileConfig'):
            if args and not isinstance(args[0],Buffer): self.fail('file-access', 'logging.config.fileConfig reads a path')
            self.fail('code-execution','logging.config.fileConfig evaluates configuration expressions')
        cfg=args[0] if args else {}
        if not isinstance(cfg,dict) or cfg.get('version')!=1: raise Halt('invalid logging config version')
        if cfg.get('incremental'): return None
        def resolve(value):
            if isinstance(value,str):
                return self.call(self.ref('pkgutil.resolve_name'),(value.removeprefix('ext://'),))
            return value
        # dictConfig calls factories in formatters/filters/handlers; ordinary
        # logger configuration does not execute Python source.
        for section in ('formatters','filters','handlers'):
            for spec in cfg.get(section,{}).values():
                if not isinstance(spec,dict): continue
                factory=spec.get('()')
                if factory is None and section=='handlers': factory=spec.get('class')
                if factory is not None:
                    params={k:(resolve(v) if isinstance(v,str) and v.startswith('ext://') else v)
                            for k,v in spec.items() if k not in ('()','class','level','formatter','filters','.')}
                    self.call(resolve(factory),(),params)
        return None
