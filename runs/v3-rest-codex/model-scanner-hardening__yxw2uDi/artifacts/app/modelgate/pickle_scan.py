"""A non-executing interpreter for pickle's stack machine.

No GLOBAL is imported, no REDUCE callable is invoked, and no object from the
pickle is deserialized by Python. Values below are constants or symbolic
objects. Calls are interpreted by their semantics; only an executed capability
is a finding. Module namespaces and mutable wrappers are shared by streams.
"""
from __future__ import annotations

import _compat_pickle
import ast
import base64
import binascii
import bz2
import codecs
import io
import itertools
import lzma
import math
import operator
import os
import pickletools
import struct
import zlib
from dataclasses import dataclass, field

from modelgate.context import Context, LimitError


class LoadError(Exception):
    """A known unpickler/runtime error stops loading, without a capability."""


class FrameReader:
    """The C unpickler reads a complete FRAME before interpreting its opcodes.

    Keeping that read-ahead matters when STOP precedes the end of a frame:
    later bytes in that frame are not a second pickle.load stream.
    """
    def __init__(self, source):
        self.source = source
        self.frame = None
        self.frame_start = 0

    def tell(self):
        return self.frame_start + self.frame.tell() if self.frame is not None else self.source.tell()

    def read(self, size=-1):
        if self.frame is not None:
            result = self.frame.read(size)
            if not result and size != 0:
                self.frame = None
            elif size < 0 or len(result) == size:
                return result
            else:
                raise LoadError('opcode exceeds FRAME boundary')
        return self.source.read(size)

    def readline(self):
        if self.frame is not None:
            result = self.frame.readline()
            if not result:
                self.frame = None
            elif result.endswith(b'\n'):
                return result
            else:
                raise LoadError('line exceeds FRAME boundary')
        return self.source.readline()

    def load_frame(self, size):
        if self.frame is not None and self.frame.read():
            raise LoadError('FRAME began before previous frame ended')
        self.frame_start = self.source.tell()
        if size > len(self.source.getbuffer()) - self.source.tell():
            raise LoadError('truncated FRAME')
        self.frame = io.BytesIO(self.source.read(size))


@dataclass(eq=False)
class Symbol:
    kind: str
    name: str = ''
    args: tuple = ()
    attrs: dict = field(default_factory=dict)


UNKNOWN = Symbol('unknown')
MARK = object()

ALIASES = {
    'posix': 'os', 'nt': 'os', '_pickle': 'pickle', 'cPickle': 'pickle',
    '_io': 'io', '__builtin__': 'builtins', 'copy_reg': 'copyreg',
    'commands': 'subprocess', 'urllib2': 'urllib.request', 'httplib': 'http.client',
    '_operator': 'operator', '_functools': 'functools', '_socket': 'socket',
    '_thread': '_thread', 'numpy.core': 'numpy._core',
    'numpy.core.multiarray': 'numpy._core.multiarray',
    'numpy.core.numeric': 'numpy._core.numeric',
}
MODULE_ATTRIBUTES = {
    'os.path': 'posixpath', 'shutil.os': 'os', 'subprocess.os': 'os',
    'pathlib.os': 'os', 'glob.os': 'os', 'tempfile._os': 'os',
    'threading._os': 'os', 'random._os': 'os', 'pydoc.os': 'os',
    'http.client.socket': 'socket', 'http.client.ssl': 'ssl',
    'urllib.request.http': 'http', 'urllib.request.socket': 'socket',
    'urllib.request.os': 'os', 'urllib.request.subprocess': 'subprocess',
    'pickle.io': 'io', 'pickle.sys': 'sys', 'pickle.codecs': 'codecs',
    'pickle._compat_pickle': '_compat_pickle', 'copy.types': 'types',
    'copy.copyreg': 'copyreg', 'pkgutil.importlib': 'importlib',
    'importlib.sys': 'sys', 'numpy.ctypeslib.ctypes': 'ctypes',
}
MODULE_NAMES = set('''
builtins os sys io pickle subprocess socket shutil pathlib functools operator
itertools importlib pkgutil collections types marshal base64 zlib bz2 lzma
codecs _codecs binascii runpy timeit code bdb pdb logging logging.config
numpy numpy._core numpy._core.multiarray numpy._core.numeric numpy.lib
numpy.lib.format numpy.testing numpy.testing._private numpy.testing._private.utils
urllib urllib.request urllib.parse http http.client ftplib smtplib ssl
webbrowser pty pydoc xmlrpc xmlrpc.client threading _thread multiprocessing
multiprocessing.context multiprocessing.pool concurrent concurrent.futures
concurrent.futures.thread concurrent.futures.process copy copyreg torch
torch.storage torch.serialization torch._utils pandas joblib joblib.numpy_pickle
posixpath ntpath json ast struct math array datetime decimal fractions
'''.split())

CODE_CALLS = set('''
builtins.eval builtins.exec builtins.compile runpy.run_path runpy.run_module
runpy._run_code runpy._run_module_code timeit.timeit timeit.repeat
timeit.Timer.timeit timeit.Timer.repeat code.InteractiveInterpreter.runsource
code.InteractiveInterpreter.runcode code.InteractiveConsole.runsource
code.InteractiveConsole.runcode bdb.Bdb.run bdb.Bdb.runeval bdb.Bdb.runctx
pdb.run pdb.runeval pdb.runctx numpy.testing._private.utils.runstring
cloudpickle.cloudpickle._make_function cloudpickle.cloudpickle_fast._make_function
dill._dill._create_function types.FunctionType types.LambdaType
'''.split())
PROCESS_CALLS = set('''
os.system os.popen os.execv os.execvp os.execve os.execvpe os.execl os.execlp
os.execlpe os.execle os.spawnl os.spawnle os.spawnlp os.spawnlpe os.spawnv
os.spawnve os.spawnvp os.spawnvpe os.posix_spawn os.posix_spawnp os.fork
os.forkpty os.startfile subprocess.Popen subprocess.call subprocess.run
subprocess.check_call subprocess.check_output subprocess.getoutput
subprocess.getstatusoutput subprocess._fork_exec pty.spawn pty.fork
pydoc.pipepager pydoc.tempfilepager multiprocessing.util.spawnv_passfds
asyncio.create_subprocess_exec asyncio.create_subprocess_shell
asyncio.subprocess.create_subprocess_exec asyncio.subprocess.create_subprocess_shell
'''.split())
FILE_CALLS = set('''
builtins.open io.open io.open_code io.FileIO os.open os.fdopen os.read os.write
os.remove os.unlink os.rename os.renames os.replace os.mkdir os.makedirs
os.rmdir os.removedirs os.listdir os.scandir os.walk os.fwalk os.stat os.lstat
os.readlink os.symlink os.link os.truncate os.chmod os.chown os.utime
os.access os.mknod os.mkfifo os.chdir os.chroot os.chsize shutil.copy
shutil.copy2 shutil.copyfile shutil.copytree shutil.move shutil.rmtree
shutil.chown shutil.copymode shutil.copystat shutil.make_archive shutil.unpack_archive
pathlib.Path.open pathlib.Path.read_text pathlib.Path.read_bytes pathlib.Path.write_text
pathlib.Path.write_bytes pathlib.Path.unlink pathlib.Path.rename pathlib.Path.replace
pathlib.Path.mkdir pathlib.Path.rmdir pathlib.Path.iterdir pathlib.Path.glob
pathlib.Path.rglob pathlib.Path.stat pathlib.Path.lstat pathlib.Path.touch
pathlib.Path.chmod pathlib.Path.symlink_to pathlib.Path.hardlink_to
pathlib.Path.exists pathlib.Path.is_file pathlib.Path.is_dir pathlib.Path.readlink pathlib.Path.resolve
fileinput.input fileinput.FileInput tempfile.NamedTemporaryFile tempfile.TemporaryFile
tempfile.mkstemp tempfile.mkdtemp tempfile.TemporaryDirectory shelve.open
dbm.open sqlite3.connect gzip.open gzip.GzipFile bz2.open bz2.BZ2File
lzma.open lzma.LZMAFile zipfile.ZipFile tarfile.open tarfile.TarFile.open
numpy.fromfile numpy.memmap numpy.save numpy.savez numpy.savez_compressed
numpy.savetxt numpy.loadtxt numpy.genfromtxt torch.save joblib.dump pandas.read_csv
pandas.read_json pandas.read_parquet pandas.read_hdf pandas.read_excel
'''.split())
NETWORK_CALLS = set('''
socket.create_connection socket.socket.connect socket.socket.connect_ex
socket.socket.sendto socket.socket.sendmsg socket.socket.send socket.socket.sendall
socket.getaddrinfo socket.gethostbyname socket.gethostbyname_ex socket.gethostbyaddr
urllib.request.urlopen urllib.request.urlretrieve urllib.request.OpenerDirector.open
http.client.HTTPConnection.connect http.client.HTTPConnection.request
http.client.HTTPConnection.send http.client.HTTPConnection.endheaders http.client.HTTPConnection.getresponse
http.client.HTTPSConnection.connect http.client.HTTPSConnection.request
ftplib.FTP.connect ftplib.FTP.login ftplib.FTP.retrbinary ftplib.FTP.storbinary
ftplib.FTP_TLS.connect smtplib.SMTP.connect smtplib.SMTP.sendmail
smtplib.SMTP.send_message smtplib.SMTP_SSL.connect ssl.get_server_certificate
webbrowser.open webbrowser.open_new webbrowser.open_new_tab requests.get
requests.post requests.put requests.delete requests.request requests.head
requests.sessions.Session.request requests.sessions.Session.get
httpx.get httpx.post httpx.request xmlrpc.client.ServerProxy.__call__
'''.split())
NESTED_CALLS = set('''
pickle.loads pickle.load torch.load torch.storage._load_from_bytes
numpy.load numpy.lib.format.read_array joblib.load pandas.read_pickle
'''.split())
CLASS_NAMES = set('''
functools.partial functools.partialmethod functools.cached_property functools.lru_cache
socket.socket socket.SocketType io.BytesIO io.StringIO io.FileIO io.TextIOWrapper
io.BufferedReader io.BufferedWriter pathlib.Path pathlib.PosixPath pathlib.WindowsPath
pathlib.PurePath pathlib.PurePosixPath pathlib.PureWindowsPath types.ModuleType
types.SimpleNamespace types.FunctionType types.CodeType collections.defaultdict
collections.OrderedDict collections.deque builtins.type builtins.object builtins.list
builtins.tuple builtins.dict builtins.set builtins.frozenset builtins.str builtins.bytes
builtins.bytearray builtins.int builtins.float builtins.bool builtins.complex builtins.range
builtins.slice builtins.map builtins.filter builtins.zip builtins.enumerate builtins.reversed
'''.split())


def canonical(module):
    return ALIASES.get(module, module)



def normalize_call(name):
    aliases = {
        'pickle._loads': 'pickle.loads', 'pickle._load': 'pickle.load',
        'cloudpickle.loads': 'pickle.loads', 'cloudpickle.load': 'pickle.load',
        'joblib.numpy_pickle.load': 'joblib.load',
        'pandas.io.pickle.read_pickle': 'pandas.read_pickle',
        'numpy.lib.npyio.load': 'numpy.load', 'numpy.lib._npyio_impl.load': 'numpy.load',
        'concurrent.futures.thread.ThreadPoolExecutor': 'concurrent.futures.ThreadPoolExecutor',
        'concurrent.futures.process.ProcessPoolExecutor': 'concurrent.futures.ProcessPoolExecutor',
    }
    if name in aliases: return aliases[name]
    for subclass, base in (
        ('pathlib.PosixPath.', 'pathlib.Path.'), ('pathlib.WindowsPath.', 'pathlib.Path.'),
        ('socket.SocketType.', 'socket.socket.'), ('ssl.SSLSocket.', 'socket.socket.'),
        ('ftplib.FTP_TLS.', 'ftplib.FTP.'), ('smtplib.SMTP_SSL.', 'smtplib.SMTP.'),
        ('smtplib.LMTP.', 'smtplib.SMTP.'),
    ):
        if name.startswith(subclass): return base + name[len(subclass):]
    return name

def ref(name):
    return Symbol('ref', name)


def known(value):
    if isinstance(value, Symbol):
        return False
    if isinstance(value, (list, tuple)):
        return all(known(x) for x in value)
    return isinstance(value, (str, bytes, bytearray, int, float, bool, complex, type(None), slice))


class Interpreter:
    def __init__(self, ctx, location, mode='pickle', depth=0):
        self.ctx = ctx
        self.location = location
        self.mode = mode
        self.depth = depth
        self.where = location
        self.call_depth = 0

    def module(self, name):
        name = canonical(name)
        if name not in self.ctx.modules:
            self.ctx.modules[name] = Symbol('module', name)
        return self.ctx.modules[name]

    def namespace(self, name):
        name = canonical(name)
        if name not in self.ctx.module_dicts:
            self.ctx.module_dicts[name] = Symbol('namespace', name, attrs=self.module(name).attrs)
        return self.ctx.module_dicts[name]

    def global_value(self, module, name, protocol):
        if protocol < 3:
            module, name = _compat_pickle.NAME_MAPPING.get((module, name),
                (_compat_pickle.IMPORT_MAPPING.get(module, module), name))
        if protocol < 4 and '.' in name:
            # find_class only follows dotted attributes in protocols >= 4.
            raise LoadError('dotted GLOBAL name below protocol 4')
        value = self.module(module)
        for part in name.split('.'):
            value = self.attr(value, part)
        return value

    def attr(self, value, name, default=UNKNOWN):
        self.ctx.tick()
        if not isinstance(name, str):
            raise LoadError('attribute name must be a string')
        if isinstance(value, Symbol):
            if name in value.attrs:
                return value.attrs[name]
            if name == '__dict__':
                if value.kind == 'module':
                    return self.namespace(value.name)
                return value.attrs
            if name == '__class__':
                if value.kind == 'module':
                    return ref('types.ModuleType')
                if value.kind == 'object':
                    return value.args[0] if value.args and isinstance(value.args[0], Symbol) else ref(value.name)
            if value.kind == 'module':
                key = value.name + '.' + name
                if key == 'sys.modules':
                    return Symbol('modules', attrs=self.ctx.modules)
                if key in MODULE_ATTRIBUTES:
                    return self.module(MODULE_ATTRIBUTES[key])
                if key in MODULE_NAMES:
                    return self.module(key)
                if value.name == 'builtins' and name == '__builtins__':
                    return self.namespace('builtins')
                if key == 'ssl.socket':
                    return ref('socket.socket')
                return ref(key)
            if value.kind == 'ref':
                if name == '__call__':
                    return value
                if name == '__globals__':
                    return self.namespace(value.name.rsplit('.', 1)[0])
                return ref(value.name + '.' + name)
            if value.kind == 'partial':
                if name == 'func': return value.args[0]
                if name == 'args': return value.args[1]
                if name == 'keywords': return value.attrs.setdefault('keywords', {})
            if value.kind == 'wrapper' and name in ('__wrapped__', 'func'):
                return value.args[0]
            if name in ('__func__', '__self__') and value.kind == 'bound':
                return ref(value.name) if name == '__func__' else value.args[0]
            return Symbol('bound', name, (value,))
        if name == '__class__':
            return ref('builtins.' + type(value).__name__)
        return Symbol('bound', name, (value,))

    def item(self, value, key, default=UNKNOWN):
        if isinstance(value, Symbol):
            if value.kind == 'modules' and isinstance(key, str):
                return value.attrs.get(key, self.module(key))
            if value.kind == 'namespace' and isinstance(key, str):
                return self.attr(self.module(value.name), key, default)
            if value.kind == 'object' and key in value.attrs:
                return value.attrs[key]
            return default
        try:
            return value[key]
        except (KeyError, IndexError, TypeError):
            return default

    def setitem(self, value, key, item):
        if isinstance(value, Symbol):
            value.attrs[key] = item
        else:
            try: value[key] = item
            except (TypeError, IndexError): raise LoadError('invalid item assignment')

    def is_class(self, value):
        if not isinstance(value, Symbol): return False
        if value.kind == 'class': return True
        if value.kind != 'ref': return False
        return value.name in CLASS_NAMES or value.name.rsplit('.', 1)[-1][:1].isupper()

    def construct(self, value, args, kwargs=None, new_only=False):
        if not isinstance(args, tuple) or (kwargs is not None and not isinstance(kwargs, dict)):
            raise LoadError('invalid constructor arguments')
        if not new_only:
            return self.call(value, args, kwargs)
        if not self.is_class(value):
            raise LoadError('NEWOBJ requires a class')
        if value.kind == 'class':
            new = value.attrs.get('__new__')
            if new is not None:
                return self.call(new, (value,) + args, kwargs)
            return Symbol('object', value.name, (value,))
        name = value.name
        # For these types the capability/construction is implemented in __new__.
        if name in ('types.FunctionType', 'types.LambdaType', 'functools.partial',
                    'builtins.type', 'pathlib.Path', 'pathlib.PosixPath', 'pathlib.WindowsPath'):
            return self.call(value, args, kwargs)
        return Symbol('object', name, (value,) + args)

    def build(self, obj, state, stream):
        if not isinstance(obj, Symbol):
            # Native dict/list do not gain arbitrary attributes.
            return
        if obj.kind == 'partial':
            if not isinstance(state, tuple) or len(state) != 4:
                raise LoadError('invalid partial state')
            func, args, kwargs, attrs = state
            if not isinstance(func, Symbol) or func.kind in ('module', 'namespace', 'stream', 'code', 'path', 'iterator'):
                raise LoadError('partial target is not callable')
            if not isinstance(args, tuple) or not isinstance(kwargs, (dict, type(None))):
                raise LoadError('invalid partial state')
            obj.args = (func, args)
            obj.attrs['keywords'] = kwargs or {}
            if isinstance(attrs, dict): obj.attrs.update(attrs)
            return
        if obj.kind == 'object':
            cls = obj.args[0] if obj.args else None
            if isinstance(cls, Symbol) and cls.kind == 'class' and '__setstate__' in cls.attrs:
                self.call(cls.attrs['__setstate__'], (obj, state))
                return
            if '__setstate__' in obj.attrs:
                self.call(obj.attrs['__setstate__'], (state,))
                return
        slots = None
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            state, slots = state
        if isinstance(state, dict): obj.attrs.update(state)
        if slots: obj.attrs.update(slots)
        # joblib's NumpyArrayWrapper replaces itself with array data immediately
        # after BUILD. Numeric arrays are raw aligned bytes, object arrays are
        # another pickle, read by a separate (plain) unpickler.
        if obj.name in ('joblib.numpy_pickle.NumpyArrayWrapper', 'joblib.numpy_pickle_compat.NDArrayWrapper'):
            self.joblib_array(obj, stream)

    def joblib_array(self, obj, stream):
        if self.mode != 'joblib':
            return
        dtype = obj.attrs.get('dtype')
        descr = dtype.args[1] if isinstance(dtype, Symbol) and len(dtype.args) > 1 else None
        import numpy as np
        try: dt = np.dtype(descr)
        except Exception: raise LimitError('unknown joblib array dtype')
        if dt.hasobject:
            self.run_one(stream, 'pickle')
        else:
            shape = obj.attrs.get('shape', ())
            if not isinstance(shape, (list, tuple)) or not all(isinstance(n, int) and n >= 0 for n in shape):
                raise LoadError('invalid array shape')
            size = math.prod(shape) * dt.itemsize
            alignment = obj.attrs.get('numpy_array_alignment_bytes')
            if alignment is not None:
                pad = stream.read(1)
                if not pad: raise LoadError('truncated array alignment')
                stream.seek(pad[0], 1)
            stream.seek(size, 1)

    def call(self, func, args=(), kwargs=None):
        self.ctx.tick()
        if not isinstance(args, tuple):
            raise LoadError('REDUCE arguments must be a tuple')
        kwargs = kwargs or {}
        if not isinstance(kwargs, dict) or not all(isinstance(k, str) for k in kwargs):
            raise LoadError('call keywords must be strings')
        self.call_depth += 1
        if self.call_depth > 100:
            self.call_depth -= 1
            raise LimitError('recursive symbolic calls')
        try:
            return self._call(func, args, kwargs)
        finally:
            self.call_depth -= 1

    def _call(self, func, args, kw):
        if not isinstance(func, Symbol):
            raise LoadError('object is not callable')
        if func.kind == 'partial':
            kwargs = dict(func.attrs.get('keywords') or {})
            kwargs.update(kw)
            return self.call(func.args[0], func.args[1] + args, kwargs)
        if func.kind == 'wrapper':
            return self.call(func.args[0], args, kw)
        if func.kind == 'weakref':
            return func.args[0] if func.args else None
        if func.kind == 'vectorize':
            if not args: raise LoadError('vectorized function missing input')
            sources = [self.consume(a) if isinstance(a, (list, tuple, Symbol)) else [a] for a in args]
            if not sources or any(not a for a in sources): return []
            return [self.call(func.args[0], tuple(items), kw) for items in zip(*sources)]
        if func.kind == 'attrgetter':
            if not args: raise LoadError('attrgetter missing object')
            values = []
            for path in func.args:
                value = args[0]
                for name in path.split('.'):
                    value = self.attr(value, name)
                values.append(value)
            return values[0] if len(values) == 1 else tuple(values)
        if func.kind == 'itemgetter':
            if not args: raise LoadError('itemgetter missing object')
            values = tuple(self.item(args[0], k) for k in func.args)
            return values[0] if len(values) == 1 else values
        if func.kind == 'methodcaller':
            if not args: raise LoadError('methodcaller missing object')
            return self.call(self.attr(args[0], func.name), func.args, func.attrs)
        if func.kind == 'bound':
            return self.bound(func.args[0], func.name, args, kw)
        if func.kind == 'class':
            obj = self.construct(func, args, kw, True)
            init = func.attrs.get('__init__')
            if init is not None: self.call(init, (obj,) + args, kw)
            return obj
        if func.kind == 'object':
            cls = func.args[0] if func.args else None
            target = func.attrs.get('__call__')
            if target is None and isinstance(cls, Symbol): target = cls.attrs.get('__call__')
            if target is not None: return self.call(target, (func,) + args, kw)
            if func.name == 'timeit.Timer':
                return UNKNOWN
            if func.name == 'xmlrpc.client.ServerProxy':
                self.ctx.add('network', self.where, 'XML-RPC request')
            return UNKNOWN
        if func.kind != 'ref':
            if func.kind == 'unknown':
                raise LimitError('cannot resolve called value')
            raise LoadError('object is not callable')
        name = normalize_call(func.name)
        if name == 'copyreg.add_extension':
            if len(args) == 3 and isinstance(args[0], str) and isinstance(args[1], str) and isinstance(args[2], int):
                self.ctx.modules.setdefault('_extensions', Symbol('registry')).attrs[args[2]] = args[:2]
            return None
        if name in ('timeit.Timer.timeit', 'timeit.Timer.repeat'):
            if not args: raise LoadError('timer method missing self')
            return self.bound(args[0], name.rsplit('.', 1)[-1], args[1:], kw)
        if name in ('timeit.timeit', 'timeit.repeat'):
            timer = self.call(ref('timeit.Timer'), args[:3], {k:v for k,v in kw.items() if k in ('stmt', 'setup', 'timer', 'globals')})
            options = {k:v for k,v in kw.items() if k in ('number', 'repeat')}
            if len(args) > 3: options['number' if name == 'timeit.timeit' else 'repeat'] = args[3]
            if len(args) > 4 and name == 'timeit.repeat': options['number'] = args[4]
            return self.bound(timer, name.split('.')[-1], (), options)
        if name in CODE_CALLS:
            self.validate_capability(name, args, kw)
            self.ctx.add('code-execution', self.where, f'call {name}')
            return UNKNOWN
        if name in PROCESS_CALLS:
            self.validate_capability(name, args, kw)
            self.ctx.add('process', self.where, f'call {name}')
            return UNKNOWN
        if name in FILE_CALLS:
            self.validate_capability(name, args, kw)
            path = args[0] if args else kw.get('file', kw.get('filename', kw.get('name', kw.get('fileobj'))))
            memory_file = kw.get('fileobj', path)
            if isinstance(memory_file, Symbol) and memory_file.kind == 'stream':
                return Symbol('reader', name, (memory_file,), dict(kw))
            if name == 'gzip.GzipFile' and path is None: raise LoadError('GzipFile needs a filename or fileobj')
            if isinstance(path, Symbol) and path.kind == 'path': path = path.name
            paths = (path,)
            if name in ('os.rename', 'os.replace', 'os.renames', 'shutil.copy', 'shutil.copyfile', 'shutil.move', 'shutil.copy2'):
                destination = args[1] if len(args) > 1 else kw.get('dst')
                if isinstance(destination, Symbol) and destination.kind == 'path': destination = destination.name
                paths = (path, destination)
            if any(self.ctx.outside(p) for p in paths): self.ctx.add('file-access', self.where, f'call {name}')
            return Symbol('object', name, (func,) + args)
        if name in NETWORK_CALLS:
            self.validate_capability(name, args, kw)
            self.ctx.add('network', self.where, f'call {name}')
            return UNKNOWN
        if name in NESTED_CALLS:
            return self.nested(name, args, kw)

        if name in ('builtins.getattr', 'builtins.hasattr'):
            if len(args) < 2: raise LoadError('missing getattr arguments')
            result = self.attr(args[0], args[1], args[2] if len(args) > 2 else UNKNOWN)
            return True if name.endswith('hasattr') else result
        if name in ('builtins.setattr', 'builtins.delattr'):
            if len(args) < 2: raise LoadError('missing setattr arguments')
            obj, attr = args[:2]
            if isinstance(obj, Symbol):
                if name.endswith('setattr') and len(args) > 2: obj.attrs[attr] = args[2]
                else: obj.attrs.pop(attr, None)
            return None
        if name in ('builtins.__import__', 'importlib.import_module'):
            if not args or not isinstance(args[0], str): raise LoadError('invalid module name')
            module = args[0]
            if name == 'builtins.__import__':
                fromlist = args[3] if len(args) > 3 else kw.get('fromlist')
                if not fromlist: module = module.split('.')[0]
            return self.module(module)
        if name == 'pkgutil.resolve_name':
            if not args or not isinstance(args[0], str): raise LoadError('invalid name')
            path = args[0]
            if ':' in path:
                module, path = path.split(':', 1)
                value = self.module(module)
                for part in filter(None, path.split('.')): value = self.attr(value, part)
                return value
            parts = path.split('.')
            cut = len(parts)
            while cut > 1 and '.'.join(parts[:cut]) not in MODULE_NAMES: cut -= 1
            value = self.module('.'.join(parts[:cut]))
            for part in parts[cut:]: value = self.attr(value, part)
            return value
        if name == 'builtins.vars':
            if not args: return self.namespace('__main__')
            return self.attr(args[0], '__dict__')
        if name == 'builtins.globals': return self.namespace('__main__')
        if name == 'builtins.locals': return {}
        if name in ('operator.attrgetter', 'operator.itemgetter'):
            return Symbol(name.split('.')[-1], args=args)
        if name == 'operator.methodcaller':
            if not args: raise LoadError('missing method name')
            return Symbol('methodcaller', args[0], args[1:], dict(kw))
        if name in ('operator.call', 'functools.apply'):
            if not args: raise LoadError('missing callable')
            return self.call(args[0], args[1:], kw)
        if name in ('functools.partial', 'functools.partialmethod'):
            if not args: raise LoadError('missing partial callable')
            if not isinstance(args[0], Symbol) or args[0].kind in ('module', 'namespace', 'stream', 'code', 'path', 'iterator'):
                raise LoadError('partial target is not callable')
            return Symbol('partial', args=(args[0], args[1:]), attrs={'keywords': dict(kw)})
        if name in ('functools.singledispatch', 'functools.singledispatchmethod', 'weakref.proxy'):
            if not args: raise LoadError('wrapper missing callable')
            return Symbol('wrapper', args=(args[0],))
        if name == 'weakref.ref': return Symbol('weakref', args=args)
        if name in ('numpy.vectorize', 'numpy.frompyfunc'):
            if not args: raise LoadError('wrapper missing callable')
            return Symbol('vectorize', args=(args[0],))
        if name in ('functools.lru_cache', 'functools.cache'):
            if args and isinstance(args[0], Symbol): return Symbol('wrapper', args=(args[0],))
            return Symbol('ref', '_modelgate.wrap')
        if name == '_modelgate.wrap': return Symbol('wrapper', args=args[:1])
        if name in ('functools.wraps',):
            return Symbol('partial', args=(ref('functools.update_wrapper'), ()), attrs={'keywords': {'wrapped': args[0]}})
        if name == 'functools.update_wrapper': return args[0] if args else kw.get('wrapper', UNKNOWN)
        if name in ('copy.copy', 'copy.deepcopy'):
            if not args: raise LoadError('missing copy object')
            # Modules/functions are identities; partial.copy retains its target.
            value = args[0]
            if isinstance(value, Symbol) and value.kind in ('partial', 'object'):
                return Symbol(value.kind, value.name, value.args, dict(value.attrs))
            return value
        if name == 'functools.reduce':
            if len(args) < 2: raise LoadError('missing reduce args')
            values = self.consume(args[1])
            if len(args) > 2: acc = args[2]
            elif values: acc, values = values[0], values[1:]
            else: raise LoadError('reduce of empty sequence')
            for value in values: acc = self.call(args[0], (acc, value))
            return acc

        if name in ('builtins.map', 'builtins.filter', 'itertools.starmap', 'itertools.filterfalse',
                    'itertools.takewhile', 'itertools.dropwhile'):
            return Symbol('iterator', name, args)
        if name in ('builtins.zip', 'builtins.enumerate', 'builtins.reversed', 'builtins.iter',
                    'itertools.chain', 'itertools.chain.from_iterable', 'itertools.islice',
                    'itertools.repeat', 'itertools.cycle', 'itertools.accumulate',
                    'itertools.compress', 'itertools.product'):
            return Symbol('iterator', name, args, dict(kw))
        if name == 'builtins.next':
            if not args: raise LoadError('next missing iterator')
            values = self.consume(args[0], 1)
            if values: return values[0]
            if len(args) > 1: return args[1]
            raise LoadError('StopIteration')
        if name in ('builtins.list', 'builtins.tuple', 'builtins.set', 'builtins.frozenset', 'collections.deque'):
            values = self.consume(args[0]) if args else []
            if name.endswith('.tuple'): return tuple(values)
            if name.endswith(('.set', '.frozenset')):
                try: return set(values)
                except TypeError: raise LoadError('unhashable set item')
            return values
        if name in ('builtins.sorted', 'builtins.min', 'builtins.max'):
            if not args: raise LoadError('missing iterable')
            values = self.consume(args[0]) if len(args) == 1 or name.endswith('.sorted') else list(args)
            key = kw.get('key')
            keys = [self.call(key, (v,)) for v in values] if key is not None else values
            if name.endswith('.sorted'):
                if all(known(k) for k in keys):
                    try: return [v for k, v in sorted(zip(keys, values), key=lambda x:x[0], reverse=kw.get('reverse', False))]
                    except (TypeError, ValueError): pass
                return values
            if not values: return kw.get('default', UNKNOWN)
            if all(known(k) for k in keys):
                try: return values[(min if name.endswith('.min') else max)(range(len(keys)), key=keys.__getitem__)]
                except (TypeError, ValueError): pass
            return values[0]
        if name in ('builtins.any', 'builtins.all'):
            if not args: raise LoadError('missing iterable')
            iterator = self.as_iterator(args[0])
            for _ in range(10000):
                value = self.next_value(iterator)
                if value is _END: return name.endswith('.all')
                if isinstance(value, Symbol): continue
                if bool(value) == name.endswith('.any'): return bool(value)
            raise LimitError('iterator limit')
        if name == 'builtins.sum':
            values = self.consume(args[0]) if args else []
            if all(known(v) for v in values):
                try: return sum(values, args[1] if len(args) > 1 else 0)
                except (TypeError, ValueError): raise LoadError('invalid sum')
            return UNKNOWN
        if name in ('builtins.dict', 'collections.OrderedDict', 'collections.defaultdict'):
            result = {}
            items = args[1:] if name.endswith('defaultdict') else args
            if items:
                if isinstance(items[0], dict): result.update(items[0])
                else:
                    for pair in self.consume(items[0]):
                        if isinstance(pair, (list, tuple)) and len(pair) == 2: result[pair[0]] = pair[1]
            result.update(kw)
            return result
        if name in ('types.ModuleType',):
            return Symbol('module', args[0] if args and isinstance(args[0], str) else '_anonymous')
        if name == 'types.SimpleNamespace': return Symbol('object', name, (func,), dict(kw))
        if name == 'builtins.type':
            if len(args) == 1: return self.attr(args[0], '__class__')
            if len(args) == 3 and isinstance(args[2], dict): return Symbol('class', str(args[0]), (args[1],), dict(args[2]))
            raise LoadError('invalid type arguments')
        if name in ('copyreg._reconstructor', 'copyreg.__newobj__', 'copyreg.__newobj_ex__'):
            if not args: raise LoadError('missing reconstruction class')
            if name.endswith('__newobj_ex__'): return self.construct(args[0], args[1], args[2], True)
            if name.endswith('__newobj__'): return self.construct(args[0], args[1:], new_only=True)
            return Symbol('object', args[0].name if isinstance(args[0], Symbol) else '', (args[0],))

        if name == 'builtins.memoryview':
            return Symbol('buffer', args=args)
        if name in ('io.BytesIO', 'io.StringIO'):
            data = args[0] if args else (b'' if name.endswith('BytesIO') else '')
            return Symbol('stream', name, (data,), {'position': 0})
        if name in ('pickle.Unpickler', 'pickle._Unpickler'):
            return Symbol('unpickler', name, args)
        if name in ('marshal.loads', 'marshal.load'):
            if not args: raise LoadError('marshal missing input')
            value = args[0]
            if name == 'marshal.load':
                if isinstance(value, Symbol) and value.kind == 'stream': value = value.args[0][value.attrs.get('position', 0):]
                else: raise LoadError('marshal.load requires file')
            if not isinstance(value, (bytes, bytearray)): raise LoadError('marshal requires bytes')
            if value and value[0] & 0x7f == ord('c'): return Symbol('code', name, (bytes(value),))
            return Symbol('marshalled', name, args)
        if name in ('types.CodeType', 'dill._dill._create_code'):
            return Symbol('code', name, args)
        if name in ('pathlib.Path', 'pathlib.PosixPath', 'pathlib.WindowsPath',
                    'pathlib.PurePath', 'pathlib.PurePosixPath', 'pathlib.PureWindowsPath'):
            if all(isinstance(a, str) for a in args): return Symbol('path', os.path.join(*args) if args else '.')
            return Symbol('path', '')
        if name in ('socket.socket', 'socket.SocketType', 'http.client.HTTPConnection',
                    'http.client.HTTPSConnection', 'xmlrpc.client.ServerProxy', 'ftplib.FTP',
                    'ftplib.FTP_TLS', 'smtplib.SMTP', 'smtplib.SMTP_SSL', 'smtplib.LMTP'):
            if name.startswith(('ftplib.', 'smtplib.')) and (args[0] if args else kw.get('host')):
                self.ctx.add('network', self.where, f'connect in {name}')
            return Symbol('object', name.replace('socket.SocketType', 'socket.socket'), (func,) + args, dict(kw))
        if name == 'timeit.Timer':
            statement = args[0] if args else kw.get('stmt')
            setup = args[1] if len(args) > 1 else kw.get('setup')
            if isinstance(statement, str) or isinstance(setup, str):
                self.ctx.add('code-execution', self.where, 'timeit.Timer compiles supplied source')
            return Symbol('object', name, (func,) + args, {'stmt': statement, 'setup': setup})
        if name == 'logging.config.dictConfig':
            if args: self.logging_config(args[0])
            return None
        if name in ('threading.Thread', 'multiprocessing.Process', 'multiprocessing.context.Process'):
            attrs = dict(kw)
            if len(args) > 1: attrs['target'] = args[1]
            if len(args) > 3: attrs['args'] = args[3]
            if len(args) > 4: attrs['kwargs'] = args[4]
            return Symbol('object', name, (func,) + args, attrs)
        if name in ('multiprocessing.Pool', 'multiprocessing.pool.Pool'):
            self.ctx.add('process', self.where, f'process pool {name}')
            return Symbol('object', name, (func,) + args, dict(kw))
        if name.startswith(('concurrent.futures.', 'multiprocessing.pool.')) and name.endswith(('Executor', 'Pool')):
            return Symbol('object', name, (func,) + args, dict(kw))
        if name in ('_thread.start_new_thread', '_thread.start_new'):
            if args: self.call(args[0], args[1] if len(args) > 1 and isinstance(args[1], tuple) else (),
                               args[2] if len(args) > 2 else {})
            return UNKNOWN
        if name in ('builtins.staticmethod', 'builtins.classmethod'):
            return args[0] if args else UNKNOWN
        return self.pure(name, args, kw, func)

    def validate_capability(self, name, args, kw):
        # Call binding happens before the dangerous operation. This matters for
        # invalid reducers and callable objects which merely look dangerous.
        required = {
            'builtins.eval': (1, 3, ('source',)), 'builtins.exec': (1, 3, ('source',)),
            'builtins.compile': (3, 6, ('source', 'filename', 'mode')),
            'os.system': (1, 1, ('command',)), 'os.popen': (1, 4, ('cmd',)),
            'subprocess.Popen': (1, 32, ('args',)), 'subprocess.call': (1, 32, ('args',)),
            'subprocess.run': (1, 32, ('args',)), 'subprocess.check_call': (1, 32, ('args',)),
            'subprocess.check_output': (1, 32, ('args',)),
            'subprocess.getoutput': (1, 1, ('cmd',)), 'subprocess.getstatusoutput': (1, 1, ('cmd',)),
            'builtins.open': (1, 8, ('file',)), 'io.open': (1, 8, ('file',)),
            'io.FileIO': (1, 3, ('file',)), 'os.open': (2, 3, ('path', 'flags')),
            'os.remove': (1, 1, ('path',)), 'os.unlink': (1, 1, ('path',)),
            'os.rename': (2, 2, ('src', 'dst')), 'os.replace': (2, 2, ('src', 'dst')),
            'os.mkdir': (1, 2, ('path',)), 'os.makedirs': (1, 3, ('name',)),
            'shutil.copy': (2, 2, ('src', 'dst')), 'shutil.copyfile': (2, 2, ('src', 'dst')),
            'shutil.move': (2, 3, ('src', 'dst')), 'shutil.rmtree': (1, 3, ('path',)),
            'socket.create_connection': (1, 3, ('address',)),
            'socket.socket.connect': (2, 2, ()), 'socket.socket.connect_ex': (2, 2, ()),
            'urllib.request.urlopen': (1, 3, ('url',)),
            'webbrowser.open': (1, 3, ('url',)),
            'ssl.get_server_certificate': (1, 4, ('addr',)),
            'types.FunctionType': (2, 5, ('code', 'globals')),
            'types.LambdaType': (2, 5, ('code', 'globals')),
            'runpy.run_path': (1, 3, ('path_name',)),
            'runpy.run_module': (1, 4, ('mod_name',)),
            'runpy._run_code': (2, 9, ('code', 'run_globals')),
            'code.InteractiveInterpreter.runsource': (2, 4, ('self', 'source')),
            'code.InteractiveInterpreter.runcode': (2, 2, ('self', 'code')),
            'code.InteractiveConsole.runsource': (2, 4, ('self', 'source')),
            'code.InteractiveConsole.runcode': (2, 2, ('self', 'code')),
            'bdb.Bdb.run': (2, 4, ('self', 'cmd')),
            'bdb.Bdb.runeval': (2, 4, ('self', 'expr')),
            'bdb.Bdb.runctx': (4, 4, ('self', 'cmd', 'globals', 'locals')),
            'pdb.run': (1, 3, ('statement',)), 'pdb.runeval': (1, 3, ('expression',)),
        }
        rule = required.get(name)
        if rule is not None:
            minimum, maximum, names = rule
            for index in range(minimum):
                if len(args) <= index and (index >= len(names) or names[index] not in kw):
                    raise LoadError('missing capability argument')
            if len(args) > maximum: raise LoadError('too many capability arguments')
        if name in ('types.FunctionType', 'types.LambdaType'):
            code = args[0] if args else kw.get('code')
            globals_ = args[1] if len(args) > 1 else kw.get('globals')
            if not isinstance(code, Symbol) or code.kind not in ('code', 'unknown'):
                raise LoadError('FunctionType requires a code object')
            if not isinstance(globals_, dict) and not (isinstance(globals_, Symbol) and globals_.kind == 'namespace'):
                raise LoadError('FunctionType requires globals dict')
        if name in ('builtins.eval', 'builtins.exec', 'builtins.compile'):
            source = args[0] if args else kw.get('source')
            if not isinstance(source, (str, bytes, bytearray, Symbol)):
                raise LoadError('source is not code or text')
        if name in ('os.system', 'os.popen') and args and not isinstance(args[0], (str, bytes, Symbol)):
            raise LoadError('command is not text')

    def bound(self, obj, method, args, kw):
        if method in ('__getattribute__', '__getattr__'):
            return self.attr(obj, args[0]) if args else UNKNOWN
        if method == '__call__': return self.call(obj, args, kw)
        if method in ('__getitem__', 'get'):
            return self.item(obj, args[0], args[1] if len(args) > 1 else UNKNOWN) if args else UNKNOWN
        if method == '__setitem__':
            if len(args) >= 2: self.setitem(obj, args[0], args[1])
            return None
        if isinstance(obj, Symbol):
            if obj.kind == 'reader':
                source = obj.args[0]
                data = source.args[0][source.attrs.get('position', 0):]
                if method in ('read', 'read1', 'readall'):
                    from modelgate.dispatch import _expand
                    if obj.name.startswith(('gzip.', 'bz2.', 'lzma.')):
                        if 'data' not in obj.attrs:
                            obj.attrs['data'] = _expand(data, obj.name.split('.')[0], self.ctx)
                            obj.attrs['position'] = 0
                        start = obj.attrs['position']
                        end = start + args[0] if args and isinstance(args[0], int) and args[0] >= 0 else len(obj.attrs['data'])
                        obj.attrs['position'] = min(end, len(obj.attrs['data']))
                        return obj.attrs['data'][start:end]
                    if obj.name == 'zipfile.ZipFile' and args and isinstance(args[0], str):
                        import zipfile
                        with zipfile.ZipFile(io.BytesIO(data)) as zf:
                            info = zf.getinfo(args[0]); self.ctx.account(info.file_size)
                            return zf.read(info)
                if method == 'close': return None
                return UNKNOWN
            if obj.kind == 'buffer' and method in ('tobytes', 'toreadonly'):
                return bytes(obj.args[0]) if method == 'tobytes' and obj.args and isinstance(obj.args[0], (bytes, bytearray)) else obj
            if obj.kind == 'unpickler' and method == 'load':
                if not obj.args: raise LoadError('Unpickler missing file')
                return self.nested('pickle.load', obj.args, {})
            if obj.kind == 'stream':
                data = obj.args[0]
                pos = obj.attrs.get('position', 0)
                if method == 'getvalue': return data
                if method == 'tell': return pos
                if method == 'seek' and args and isinstance(args[0], int):
                    whence = args[1] if len(args) > 1 else 0
                    obj.attrs['position'] = args[0] + (pos if whence == 1 else len(data) if whence == 2 else 0)
                    return obj.attrs['position']
                if method in ('read', 'read1', 'readline'):
                    end = len(data) if not args or args[0] < 0 else pos + args[0]
                    if method == 'readline':
                        index = data.find(b'\n' if isinstance(data, bytes) else '\n', pos, end)
                        if index >= 0: end = index + 1
                    obj.attrs['position'] = min(end, len(data))
                    return data[pos:end]
                if method == 'write' and args:
                    if isinstance(data, (bytes, str)) and isinstance(args[0], type(data)):
                        self.ctx.account(len(args[0]))
                        obj.args = (data[:pos] + args[0] + data[pos + len(args[0]):],)
                        obj.attrs['position'] = pos + len(args[0])
                        return len(args[0])
                return UNKNOWN
            if obj.kind == 'iterator' and method in ('__iter__',): return obj
            if obj.kind == 'iterator' and method in ('__next__', 'next'):
                result = self.consume(obj, 1)
                if not result: raise LoadError('StopIteration')
                return result[0]
            if obj.kind in ('partial',) and method == '__setstate__':
                if args: self.build(obj, args[0], io.BytesIO())
                return None
            if obj.kind == 'path':
                if method in ('__fspath__', '__str__', 'as_posix'): return obj.name
                if method == 'joinpath':
                    return Symbol('path', os.path.join(obj.name, *args)) if all(isinstance(a, str) for a in args) else UNKNOWN
                if method in ('with_name', 'with_suffix') and args and isinstance(args[0], str):
                    path = (os.path.join(os.path.dirname(obj.name), args[0]) if method == 'with_name'
                            else os.path.splitext(obj.name)[0] + args[0])
                    return Symbol('path', path)
                return self.call(ref('pathlib.Path.' + method), (obj,) + args, kw)
            if obj.kind in ('namespace', 'modules'):
                if method in ('keys', 'values', 'items'):
                    values = obj.attrs
                    return list(getattr(values, method)())
                if method == 'update':
                    if args and isinstance(args[0], dict): obj.attrs.update(args[0])
                    obj.attrs.update(kw)
                    return None
            if obj.kind == 'object':
                cls = obj.args[0] if obj.args else None
                if isinstance(cls, Symbol) and cls.kind == 'class' and method in cls.attrs:
                    return self.call(cls.attrs[method], (obj,) + args, kw)
                if obj.name == 'timeit.Timer' and method in ('timeit', 'repeat'):
                    number = (args[1] if method == 'repeat' and len(args) > 1 else args[0]
                              if method == 'timeit' and args else kw.get('number', 1000000))
                    repeat = args[0] if method == 'repeat' and args else kw.get('repeat', 5)
                    if method == 'repeat' and isinstance(repeat, int) and repeat <= 0: return []
                    setup, stmt = obj.attrs.get('setup'), obj.attrs.get('stmt')
                    if isinstance(setup, Symbol): self.call(setup, ())
                    if isinstance(stmt, Symbol) and (not isinstance(number, int) or number > 0): self.call(stmt, ())
                    return 0.0 if method == 'timeit' else [0.0]
                if obj.name in ('threading.Thread', 'multiprocessing.Process', 'multiprocessing.context.Process'):
                    if method in ('start', 'run'):
                        if method == 'start' and obj.name.startswith('multiprocessing.'):
                            self.ctx.add('process', self.where, f'{obj.name}.start')
                        target = obj.attrs.get('target', obj.attrs.get('_target'))
                        call_args = obj.attrs.get('args', obj.attrs.get('_args', ()))
                        call_kw = obj.attrs.get('kwargs', obj.attrs.get('_kwargs', {}))
                        if target is not None and isinstance(call_args, tuple): self.call(target, call_args, call_kw)
                        return None
                if 'Executor' in obj.name or obj.name.startswith('multiprocessing.pool.'):
                    if method in ('submit', 'map', 'apply', 'apply_async', 'starmap', 'starmap_async', 'map_async'):
                        if obj.name.endswith('ProcessPoolExecutor') or obj.name.startswith('multiprocessing.pool.Pool'):
                            self.ctx.add('process', self.where, f'{obj.name}.{method}')
                        if args:
                            if method in ('apply', 'apply_async'):
                                call_args = args[1] if len(args) > 1 else kw.get('args', ())
                                self.call(args[0], call_args, args[2] if len(args) > 2 else kw.get('kwds', {}))
                            elif method in ('map', 'map_async', 'starmap', 'starmap_async'):
                                if len(args) > 1:
                                    for value in self.consume(args[1]):
                                        self.call(args[0], tuple(value) if method.startswith('starmap') and isinstance(value, (tuple, list)) else (value,))
                            else: self.call(args[0], args[1:], kw)
                        return Symbol('object', 'concurrent.futures.Future')
                if obj.name == 'xmlrpc.client.ServerProxy' and not method.startswith('_'):
                    self.ctx.add('network', self.where, f'XML-RPC method {method}')
                    return UNKNOWN
                return self.call(ref(obj.name + '.' + method), (obj,) + args, kw)
            return UNKNOWN
        if method in ('append', 'extend', 'insert', 'add', 'update', 'setdefault', 'pop', 'clear', 'copy'):
            try:
                if method == 'append' and isinstance(obj, list): obj.append(args[0]); return None
                if method == 'extend' and isinstance(obj, list): obj.extend(self.consume(args[0])); return None
                if method == 'insert' and isinstance(obj, list): obj.insert(*args); return None
                if method == 'add' and isinstance(obj, set): obj.add(args[0]); return None
                if method == 'update' and isinstance(obj, dict):
                    if args and isinstance(args[0], dict): obj.update(args[0])
                    obj.update(kw); return None
                if method == 'setdefault' and isinstance(obj, dict): return obj.setdefault(args[0], args[1] if len(args) > 1 else None)
                if method == 'pop' and isinstance(obj, (dict, list, set)): return obj.pop(*args)
                if method == 'copy': return obj.copy()
                if method == 'clear': obj.clear(); return None
            except (KeyError, IndexError, TypeError, AttributeError): raise LoadError('invalid container method')
        if isinstance(obj, dict) and method in ('keys', 'values', 'items'):
            return list(getattr(obj, method)())
        if method == '__iter__': return Symbol('iterator', 'builtins.iter', (obj,))
        if method in ('__add__', '__radd__', '__iadd__'):
            return self.pure('operator.add', (obj, args[0]), {}, ref('operator.add'))
        if method == '__mul__': return self.pure('operator.mul', (obj, args[0]), {}, ref('operator.mul'))
        if method in ('__getitem__',): return self.item(obj, args[0])
        # Only methods of literal, inert built-in values can be evaluated here.
        safe_methods = {
            str: {'encode', 'join', 'format', 'replace', 'strip', 'lstrip', 'rstrip', 'split',
                  'rsplit', 'lower', 'upper', 'capitalize', 'swapcase', 'removeprefix', 'removesuffix'},
            bytes: {'decode', 'join', 'replace', 'strip', 'lstrip', 'rstrip', 'split', 'hex'},
            bytearray: {'decode', 'hex'},
        }
        if type(obj) in safe_methods and method in safe_methods[type(obj)] and all(known(a) for a in args) and all(known(v) for v in kw.values()):
            try:
                if method == 'join' and args and isinstance(args[0], Symbol): return UNKNOWN
                value = getattr(obj, method)(*args, **kw)
                if isinstance(value, (str, bytes, bytearray)): self.ctx.account(len(value))
                return value
            except (ValueError, TypeError, UnicodeError, LookupError): raise LoadError('invalid literal operation')
        return UNKNOWN

    def consume(self, iterable, limit=None):
        """Consume at most the requested number of iterator elements.

        Positions live on symbolic iterators, so next() and partial consumption
        preserve laziness. In particular next(map(...)) never calls its second
        element's function.
        """
        full = limit is None
        limit = 100000 if full else min(limit, 100000)
        result = []
        iterable = self.as_iterator(iterable)
        for _ in range(limit):
            value = self.next_value(iterable)
            if value is _END: return result
            result.append(value)
        if full:
            raise LimitError('iterator materialisation limit')
        return result

    def next_value(self, iterable):
        self.ctx.tick()
        if isinstance(iterable, Symbol):
            if iterable.kind == 'iterator': return self.iterator_next(iterable)
            if iterable.kind in ('namespace', 'modules'):
                return self.next_value(self.as_iterator(list(iterable.attrs)))
            if iterable.kind == 'object':
                cls = iterable.args[0] if iterable.args else None
                if isinstance(cls, Symbol) and cls.kind == 'class' and '__iter__' in cls.attrs:
                    it = iterable.attrs.get('_iterator')
                    if it is None:
                        it = self.call(cls.attrs['__iter__'], (iterable,))
                        iterable.attrs['_iterator'] = it
                    if it is not iterable: return self.next_value(it)
                return _END
            return _END
        # A source gets its position in its wrapping iterator, never in the list.
        if isinstance(iterable, (list, tuple, set, frozenset, dict, str, bytes, bytearray, range)):
            # Calls to consume always need a fresh iter(sequence). This branch
            # is only used via an iterator's owned source wrapper.
            return _END
        return _END

    def as_iterator(self, value):
        if isinstance(value, Symbol) and value.kind == 'iterator': return value
        return Symbol('iterator', '_sequence', (value,), {'index': 0})

    def iterator_next(self, it):
        name, args, attrs = it.name, it.args, it.attrs
        if name == '_sequence':
            source = args[0]
            index = attrs.get('index', 0)
            if isinstance(source, Symbol):
                if source.kind == 'object': return self.next_value(source)
                return _END
            if isinstance(source, (set, frozenset, dict)): source = list(source)
            if not isinstance(source, (list, tuple, str, bytes, bytearray, range)) or index >= len(source): return _END
            attrs['index'] = index + 1
            return source[index]
        if name == 'builtins.iter':
            if len(args) == 2:
                value = self.call(args[0], ())
                return _END if value == args[1] else value
            source = attrs.setdefault('source', self.as_iterator(args[0] if args else ()))
            return self.next_value(source)
        if name == 'builtins.map':
            sources = attrs.setdefault('sources', [self.as_iterator(a) for a in args[1:]])
            if not sources: raise LoadError('map requires iterable')
            values = []
            for source in sources:
                value = self.next_value(source)
                if value is _END: return _END
                values.append(value)
            return self.call(args[0], tuple(values))
        if name in ('builtins.filter', 'itertools.filterfalse', 'itertools.takewhile', 'itertools.dropwhile'):
            source = attrs.setdefault('source', self.as_iterator(args[1] if len(args) > 1 else ()))
            if attrs.get('ended'): return _END
            for _ in range(10000):
                value = self.next_value(source)
                if value is _END: return _END
                test = self.call(args[0], (value,)) if args[0] is not None else value
                truth = bool(test) if not isinstance(test, Symbol) else True
                if name == 'itertools.takewhile':
                    if not truth: attrs['ended'] = True; return _END
                    return value
                if name == 'itertools.dropwhile':
                    if not attrs.get('dropped') and truth: continue
                    attrs['dropped'] = True
                    return value
                if truth != (name == 'itertools.filterfalse'): return value
            raise LimitError('iterator predicate limit')
        if name == 'itertools.starmap':
            source = attrs.setdefault('source', self.as_iterator(args[1] if len(args) > 1 else ()))
            value = self.next_value(source)
            if value is _END: return _END
            return self.call(args[0], tuple(self.consume(value)))
        if name == 'builtins.zip':
            sources = attrs.setdefault('sources', [self.as_iterator(a) for a in args])
            if not sources: return _END
            values = []
            for source in sources:
                value = self.next_value(source)
                if value is _END: return _END
                values.append(value)
            return tuple(values)
        if name == 'builtins.enumerate':
            source = attrs.setdefault('source', self.as_iterator(args[0] if args else ()))
            value = self.next_value(source)
            if value is _END: return _END
            index = attrs.get('index', args[1] if len(args) > 1 else 0)
            attrs['index'] = index + 1
            return (index, value)
        if name == 'builtins.reversed':
            source = attrs.setdefault('source', self.as_iterator(list(reversed(args[0])) if args and isinstance(args[0], (list, tuple, str, bytes)) else []))
            return self.next_value(source)
        if name == 'itertools.repeat':
            index = attrs.get('index', 0)
            if len(args) > 1 and isinstance(args[1], int) and index >= args[1]: return _END
            attrs['index'] = index + 1
            return args[0] if args else UNKNOWN
        if name == 'itertools.islice':
            if len(args) < 2: raise LoadError('islice missing args')
            source = attrs.setdefault('source', self.as_iterator(args[0]))
            start, stop, step = ((0, args[1], 1) if len(args) == 2 else
                                 (args[1] or 0, args[2], args[3] if len(args) > 3 else 1))
            index = attrs.get('index', 0)
            target = attrs.get('target', start)
            if stop is not None and target >= stop: return _END
            while index <= target:
                value = self.next_value(source)
                if value is _END: return _END
                index += 1
            attrs['index'], attrs['target'] = index, target + step
            return value
        if name in ('itertools.chain', 'itertools.chain.from_iterable'):
            if name.endswith('from_iterable'):
                outer = attrs.setdefault('outer', self.as_iterator(args[0] if args else ()))
            else:
                outer = attrs.setdefault('outer', self.as_iterator(args))
            while True:
                source = attrs.get('source')
                if source is not None:
                    value = self.next_value(source)
                    if value is not _END: return value
                source = self.next_value(outer)
                if source is _END: return _END
                attrs['source'] = self.as_iterator(source)
        if name == 'itertools.accumulate':
            source = attrs.setdefault('source', self.as_iterator(args[0] if args else ()))
            value = self.next_value(source)
            if value is _END: return _END
            if 'acc' not in attrs: attrs['acc'] = value
            else: attrs['acc'] = self.call(args[1] if len(args) > 1 and args[1] is not None else ref('operator.add'), (attrs['acc'], value))
            return attrs['acc']
        return _END

    def pure(self, name, args, kw, func):
        # These operations run only on literal data, never on attacker-supplied
        # Python objects. This is an allowlist, not getattr/import-and-call.
        try:
            if name in ('builtins.bytes', 'builtins.bytearray') and args and isinstance(args[0], Symbol) and args[0].kind == 'buffer':
                return bytes(args[0].args[0])
            if name in ('operator.getitem',): return self.item(args[0], args[1])
            if name == 'operator.setitem': self.setitem(args[0], args[1], args[2]); return None
            if name in ('operator.delitem',):
                if isinstance(args[0], dict): args[0].pop(args[1], None)
                return None
            if name in ('operator.methodcaller.__call__', 'operator.attrgetter.__call__', 'functools.partial.__call__'):
                return self.call(args[0], args[1:], kw)
            if name in ('operator.add', 'operator.concat', 'operator.iadd', 'operator.iconcat'):
                if len(args) == 2 and all(known(a) for a in args):
                    result = args[0] + args[1]
                    if isinstance(result, (bytes, str, list, tuple, bytearray)): self.ctx.account(len(result))
                    return result
            if name in ('operator.mul', 'operator.imul'):
                if len(args) == 2 and all(known(a) for a in args):
                    if isinstance(args[1], int) and isinstance(args[0], (str, bytes, list, tuple, bytearray)):
                        if len(args[0]) * max(args[1], 0) > self.ctx.max_bytes: raise LimitError('oversized literal repetition')
                    return args[0] * args[1]
            if name == 'builtins.len' and args and isinstance(args[0], (str, bytes, list, tuple, dict, set, bytearray)):
                return len(args[0])
            if name in ('builtins.str', 'builtins.bytes', 'builtins.bytearray', 'builtins.int',
                        'builtins.float', 'builtins.bool', 'builtins.complex', 'builtins.range', 'builtins.slice'):
                if all(known(a) for a in args):
                    constructors = {'str': str, 'bytes': bytes, 'bytearray': bytearray, 'int': int,
                                    'float': float, 'bool': bool, 'complex': complex, 'range': range, 'slice': slice}
                    if name in ('builtins.bytes', 'builtins.bytearray') and args and isinstance(args[0], int) and args[0] > self.ctx.max_bytes:
                        raise LimitError('oversized byte literal')
                    return constructors[name.split('.')[-1]](*args, **kw)
            if name in ('builtins.chr', 'builtins.ord') and args and known(args[0]):
                return (chr if name.endswith('chr') else ord)(args[0])
            if name in ('builtins.bytes.fromhex', 'builtins.bytearray.fromhex') and args and isinstance(args[0], str):
                return (bytearray if 'bytearray' in name else bytes).fromhex(args[0])
            if name in ('_codecs.encode', 'codecs.encode', '_codecs.decode', 'codecs.decode') and args and known(args[0]):
                encoding = args[1] if len(args) > 1 else 'utf-8'
                allowed = {'utf-8', 'utf8', 'utf-16', 'utf-16-le', 'utf-16-be', 'utf-32', 'latin1', 'latin-1',
                           'iso8859-1', 'ascii', 'raw_unicode_escape', 'unicode_escape', 'base64',
                           'base64_codec', 'hex', 'hex_codec', 'zlib', 'zlib_codec', 'bz2', 'bz2_codec', 'rot_13'}
                if name.endswith('decode') and encoding in ('zlib', 'zlib_codec', 'bz2', 'bz2_codec') and isinstance(args[0], (bytes, bytearray)):
                    from modelgate.dispatch import _expand
                    return _expand(bytes(args[0]), encoding.split('_')[0], self.ctx)
                if encoding in allowed:
                    return (codecs.decode if name.endswith('decode') else codecs.encode)(*args, **kw)
            if name in ('base64.b64decode', 'base64.standard_b64decode', 'base64.urlsafe_b64decode',
                        'base64.decodebytes', 'base64.b85decode', 'base64.a85decode',
                        'binascii.a2b_base64', 'binascii.unhexlify', 'binascii.a2b_hex') and args and known(args[0]):
                decoders = {'base64.b64decode': base64.b64decode, 'base64.standard_b64decode': base64.standard_b64decode,
                            'base64.urlsafe_b64decode': base64.urlsafe_b64decode, 'base64.decodebytes': base64.decodebytes,
                            'base64.b85decode': base64.b85decode, 'base64.a85decode': base64.a85decode,
                            'binascii.a2b_base64': binascii.a2b_base64, 'binascii.unhexlify': binascii.unhexlify,
                            'binascii.a2b_hex': binascii.unhexlify}
                result = decoders[name](*args, **kw)
                self.ctx.account(len(result)); return result
            if name in ('zlib.decompress', 'gzip.decompress', 'bz2.decompress', 'lzma.decompress') and args and isinstance(args[0], (bytes, bytearray)):
                from modelgate.dispatch import _expand
                if name == 'zlib.decompress':
                    decoder = zlib.decompressobj(args[1] if len(args) > 1 else kw.get('wbits', 15))
                    result = decoder.decompress(bytes(args[0]), self.ctx.max_bytes - self.ctx.expanded + 1)
                    self.ctx.account(len(result)); return result
                return _expand(bytes(args[0]), name.split('.')[0], self.ctx)
            if name in ('base64.b64encode', 'base64.standard_b64encode', 'binascii.hexlify') and args and isinstance(args[0], (bytes, bytearray)):
                return (binascii.hexlify if name.endswith('hexlify') else base64.b64encode)(args[0])
            if name in ('ast.literal_eval',) and args and isinstance(args[0], str): return ast.literal_eval(args[0])
            if name == 'json.loads' and args and isinstance(args[0], (str, bytes, bytearray)):
                import json
                return json.loads(args[0])
            if name in ('os.fspath', 'os.fsencode', 'os.fsdecode') and args:
                value = args[0].name if isinstance(args[0], Symbol) and args[0].kind == 'path' else args[0]
                if isinstance(value, (str, bytes)):
                    return {'os.fspath': os.fspath, 'os.fsencode': os.fsencode, 'os.fsdecode': os.fsdecode}[name](value)
            if name in ('posixpath.join', 'ntpath.join') and all(isinstance(a, str) for a in args): return os.path.join(*args)
            if name in ('posixpath.basename', 'posixpath.dirname', 'posixpath.normpath', 'posixpath.splitext') and args and isinstance(args[0], str):
                return getattr(os.path, name.split('.')[-1])(args[0])
            if name == 'struct.pack' and all(known(a) for a in args): return struct.pack(*args)
            if name in ('numpy.dtype',): return Symbol('object', name, (func,) + args)
            if name == 'numpy._core.multiarray._reconstruct': return Symbol('object', 'numpy.ndarray', args)
            if name == 'builtins.object': return Symbol('object', name, (func,))
            if name.endswith('.__new__') and args and self.is_class(args[0]):
                return self.construct(args[0], args[1:], kw, True)
            if name.endswith('.__setstate__') and args:
                self.build(args[0], args[1] if len(args) > 1 else None, io.BytesIO()); return None
        except (ValueError, TypeError, IndexError, KeyError, UnicodeError, LookupError, OverflowError, zlib.error, lzma.LZMAError):
            raise LoadError('invalid literal operation')
        return Symbol('object', name, (func,) + args, dict(kw))

    def nested(self, name, args, kw):
        if not args: raise LoadError('loader missing input')
        value = args[0]
        data = None
        stream = None
        if isinstance(value, Symbol) and value.kind == 'stream':
            stream = value
            pos = value.attrs.get('position', 0)
            data = value.args[0][pos:]
        elif name in ('pickle.loads', 'torch.storage._load_from_bytes'):
            if isinstance(value, (bytes, bytearray)): data = bytes(value)
            else: raise LoadError('loader requires bytes')
        elif isinstance(value, (str, bytes)) or (isinstance(value, Symbol) and value.kind == 'path'):
            if name in ('pickle.load', 'numpy.lib.format.read_array'):
                raise LoadError('loader requires a readable file object')
            path = value.name if isinstance(value, Symbol) else value
            if self.ctx.outside(path): self.ctx.add('file-access', self.where, f'{name} reads an external path')
            return UNKNOWN
        if data is None:
            self.ctx.add('unsupported', self.where, f'cannot statically resolve input to {name}')
            return UNKNOWN
        if not isinstance(data, (bytes, bytearray)): raise LoadError('loader requires binary data')
        data = bytes(data)
        if self.depth > 18: raise LimitError('nested deserialization limit')
        nested_location = self.where + '!' + name
        child = Interpreter(self.ctx, nested_location, self.mode, self.depth + 1)
        if name.startswith('pickle.'):
            source = io.BytesIO(data)
            result = child.run_one(source, 'pickle')
            if stream is not None: stream.attrs['position'] += source.tell()
            return result
        if name.startswith('torch.'):
            import zipfile
            if data.startswith(b'PK\x03\x04') and zipfile.is_zipfile(io.BytesIO(data)):
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    files = [n for n in zf.namelist() if n.endswith('/data.pkl') or n == 'data.pkl']
                    if files:
                        child.mode = 'torch'
                        return child.run_one(io.BytesIO(zf.read(files[0])), 'torch')
                return UNKNOWN
            return child.run(data, streams=True, torch_only=True)
        if name.startswith('numpy.'):
            from modelgate.dispatch import scan_bytes
            allow = kw.get('allow_pickle', args[2] if len(args) > 2 and name == 'numpy.load' else
                           args[1] if len(args) > 1 and name.endswith('read_array') else False)
            if allow:
                if not data.startswith((b'\x93NUMPY', b'PK')):
                    return child.run_one(io.BytesIO(data), 'pickle')
                scan_bytes(data, nested_location, self.ctx, self.depth + 1)
            return UNKNOWN
        if name == 'joblib.load':
            from modelgate.dispatch import _expand
            for magic, kind in ((b'\x1f\x8b', 'gzip'), (b'BZh', 'bz2'), (b'\xfd7zXZ\x00', 'xz'),
                                (b'\x78', 'zlib')):
                if data.startswith(magic): data = _expand(data, kind, self.ctx); break
            child.mode = 'joblib'
            return child.run_one(io.BytesIO(data), 'joblib')
        if name == 'pandas.read_pickle':
            from modelgate.dispatch import _expand
            for magic, kind in ((b'\x1f\x8b', 'gzip'), (b'BZh', 'bz2'), (b'\xfd7zXZ\x00', 'xz')):
                if data.startswith(magic): data = _expand(data, kind, self.ctx); break
            return child.run_one(io.BytesIO(data), 'pickle')
        return UNKNOWN

    def logging_config(self, config):
        if not isinstance(config, dict): return
        # dictConfig's custom factory syntax calls the factory, not the spelling
        # of strings in ordinary log records/configuration.
        for group in ('formatters', 'filters', 'handlers'):
            for value in (config.get(group) or {}).values():
                if not isinstance(value, dict): continue
                factory = value.get('()')
                if factory is not None:
                    if isinstance(factory, str): factory = self.call(ref('pkgutil.resolve_name'), (factory,))
                    kwargs = {k:v for k,v in value.items() if k not in ('()', '.', 'level', 'formatter', 'filters')}
                    self.call(factory, (), kwargs)
                elif group == 'handlers' and isinstance(value.get('class'), str):
                    factory = self.call(ref('pkgutil.resolve_name'), (value['class'],))
                    if isinstance(factory, Symbol) and factory.name.endswith(('FileHandler', 'WatchedFileHandler', 'RotatingFileHandler', 'TimedRotatingFileHandler')):
                        self.ctx.add('file-access', self.where, 'logging file handler')
                    elif isinstance(factory, Symbol) and factory.name.endswith(('SMTPHandler', 'HTTPHandler', 'SocketHandler', 'DatagramHandler', 'SysLogHandler')):
                        # These handlers generally defer connection until emit.
                        pass
                    else: self.call(factory, (), {k:v for k,v in value.items() if k != 'class'})

    def run(self, data, streams=True, torch_only=False):
        source = io.BytesIO(data)
        value = UNKNOWN
        count = 0
        legacy = False
        object_value = UNKNOWN
        while source.tell() < len(data):
            self.ctx.tick()
            count += 1
            self.location_stream = f'{self.location}:stream[{count}]'
            value = self.run_one(source, 'torch' if legacy or torch_only else self.mode)
            if count == 1:
                if value == 119547037146038801333356:
                    legacy = True
                elif torch_only:
                    raise LoadError('invalid torch legacy magic')
            if count == 4: object_value = value
            if (legacy or torch_only) and count >= 5:
                return object_value
            if not streams: return value
        return value

    def run_one(self, source, mode=None):
        mode = mode or self.mode
        stack = []
        memo = {}
        protocol = 0
        base = source.tell()
        location = getattr(self, 'location_stream', self.location)

        def marked():
            index = len(stack) - 1
            while index >= 0 and stack[index] is not MARK: index -= 1
            if index < 0: raise LoadError('missing MARK')
            values = stack[index + 1:]
            del stack[index:]
            return values

        reader = FrameReader(source)
        for op, arg, pos in pickletools.genops(reader):
            self.ctx.tick()
            self.where = f'{location}@{pos}'
            name = op.name
            if name == 'PROTO':
                if arg > 5: raise LimitError('unsupported pickle protocol')
                protocol = arg
            elif name == 'FRAME':
                reader.load_frame(arg)
            elif name == 'MARK': stack.append(MARK)
            elif name == 'STOP':
                if not stack: raise LoadError('empty pickle stack')
                return stack.pop()
            elif name in ('NONE', 'NEWTRUE', 'NEWFALSE'):
                stack.append({'NONE': None, 'NEWTRUE': True, 'NEWFALSE': False}[name])
            elif name in ('INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4', 'FLOAT', 'BINFLOAT',
                          'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE', 'BINUNICODE8', 'STRING', 'BINSTRING',
                          'SHORT_BINSTRING', 'BINBYTES', 'SHORT_BINBYTES', 'BINBYTES8', 'BYTEARRAY8'):
                stack.append(arg)
            elif name == 'EMPTY_TUPLE': stack.append(())
            elif name == 'EMPTY_LIST': stack.append([])
            elif name == 'EMPTY_DICT': stack.append({})
            elif name == 'EMPTY_SET': stack.append(set())
            elif name in ('TUPLE', 'LIST', 'DICT', 'FROZENSET'):
                values = marked()
                if name == 'TUPLE': stack.append(tuple(values))
                elif name == 'LIST': stack.append(values)
                elif name == 'DICT': stack.append(dict(zip(values[::2], values[1::2])))
                else: stack.append(frozenset(values))
            elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
                size = int(name[-1])
                values = tuple(stack[-size:]); del stack[-size:]; stack.append(values)
            elif name in ('BINPUT', 'LONG_BINPUT', 'PUT'): memo[int(arg)] = stack[-1]
            elif name == 'MEMOIZE': memo[len(memo)] = stack[-1]
            elif name in ('BINGET', 'LONG_BINGET', 'GET'):
                if int(arg) not in memo: raise LoadError('missing memo reference')
                stack.append(memo[int(arg)])
            elif name == 'POP': stack.pop()
            elif name == 'POP_MARK': marked()
            elif name == 'DUP': stack.append(stack[-1])
            elif name in ('GLOBAL', 'INST'):
                module, global_name = arg.split(' ', 1)
                target = self.global_value(module, global_name, protocol)
                if name == 'GLOBAL': stack.append(target)
                else:
                    args = tuple(marked())
                    stack.append(self.construct(target, args, new_only=not args and self.is_class(target)))
            elif name == 'STACK_GLOBAL':
                global_name = stack.pop(); module = stack.pop()
                if not isinstance(module, str) or not isinstance(global_name, str): raise LoadError('non-string GLOBAL')
                stack.append(self.global_value(module, global_name, protocol))
            elif name == 'REDUCE':
                args = stack.pop(); target = stack.pop()
                stack.append(self.call(target, args))
            elif name == 'NEWOBJ':
                args = stack.pop(); cls = stack.pop()
                stack.append(self.construct(cls, args, new_only=True))
            elif name == 'NEWOBJ_EX':
                kwargs = stack.pop(); args = stack.pop(); cls = stack.pop()
                stack.append(self.construct(cls, args, kwargs, True))
            elif name == 'OBJ':
                values = marked()
                if not values: raise LoadError('OBJ missing target')
                target, args = values[0], tuple(values[1:])
                stack.append(self.construct(target, args, new_only=not args and self.is_class(target)))
            elif name == 'BUILD':
                state = stack.pop()
                self.build(stack[-1], state, source)
            elif name in ('APPEND', 'APPENDS', 'ADDITEMS'):
                values = [stack.pop()] if name == 'APPEND' else marked()
                obj = stack[-1]
                if isinstance(obj, list): obj.extend(values)
                elif isinstance(obj, set): obj.update(values)
                else:
                    for value in values: self.call(self.attr(obj, 'add' if name == 'ADDITEMS' else 'append'), (value,))
            elif name in ('SETITEM', 'SETITEMS'):
                if name == 'SETITEM':
                    item = stack.pop(); key = stack.pop(); self.setitem(stack[-1], key, item)
                else:
                    values = marked()
                    for i in range(0, len(values), 2): self.setitem(stack[-1], values[i], values[i + 1])
            elif name in ('PERSID', 'BINPERSID'):
                pid = stack.pop() if name == 'BINPERSID' else arg
                if mode != 'torch': raise LoadError('plain pickle has no persistent_load')
                stack.append(Symbol('storage', args=(pid,)))
            elif name in ('EXT1', 'EXT2', 'EXT4'):
                # No artifact can register an extension in a fresh loader without
                # an earlier explicit copyreg.add_extension call.
                registry = self.ctx.modules.get('_extensions')
                if registry is None or arg not in registry.attrs: raise LoadError('unregistered extension')
                module, global_name = registry.attrs[arg]
                stack.append(self.global_value(module, global_name, protocol))
            elif name == 'NEXT_BUFFER': raise LoadError('loader has no out-of-band buffers')
            elif name == 'READONLY_BUFFER': pass
            else: raise LimitError(f'unsupported opcode {name}')
        raise LoadError('pickle missing STOP')


_END = object()


def scan_pickle(data: bytes, location: str, ctx=None, *, mode='pickle', depth=0, streams=True):
    owned = ctx is None
    if owned: ctx = Context()
    machine = Interpreter(ctx, location, mode, depth)
    result = UNKNOWN
    try:
        result = machine.run(data, streams=streams)
    except LoadError:
        # A loader error prevents all later opcodes/streams from running.
        pass
    except Exception as exc:
        if not ctx.findings: ctx.add('unsupported', machine.where, f'cannot analyse pickle: {type(exc).__name__}: {exc}')
    return ctx.findings if owned else result
