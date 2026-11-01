"""A static pickle virtual machine.

The VM interprets stack/memo/container operations and models Python call
semantics. It never imports a pickle GLOBAL, unpickles data, or calls a value
obtained from an artifact. Literal operations below are a small, explicit
allowlist over scanner-owned primitive values.
"""
from __future__ import annotations

import _compat_pickle
import base64
import binascii
import codecs
import io
import marshal
import math
import operator
import pickletools
import posixpath
import struct
import types
import zlib
from dataclasses import dataclass, field

from modelgate.report import Finding
from modelgate.module_exports import MODULE_EXPORTS

MAX_STEPS = 1000000
MAX_ITEMS = 100000
MAX_DATA = 256 * 1024 * 1024
TORCH_MAGIC = 119547037146038801333356
MARK = object()
MISSING = object()


class LoadError(Exception):
    """The real loader stops here before executing any later opcode."""


class AnalysisLimit(Exception):
    pass


@dataclass(eq=False)
class Ref:
    name: str
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Module:
    name: str
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Instance:
    cls: str
    args: tuple = ()
    attrs: dict = field(default_factory=dict)
    state: object = None


@dataclass(eq=False)
class Partial:
    func: object
    args: tuple = ()
    kw: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Bound:
    obj: object
    name: str


@dataclass(eq=False)
class Adapter:
    kind: str
    args: tuple
    kw: dict = field(default_factory=dict)


@dataclass(eq=False)
class Lazy:
    kind: str
    func: object
    sources: tuple
    index: int = 0
    state: dict = field(default_factory=dict)


@dataclass(eq=False)
class Unknown:
    description: str = 'unknown value'


@dataclass(eq=False)
class Modules:
    pass


@dataclass(eq=False)
class Buffer:
    data: object
    offset: int = 0


ALIASES = {'posix': 'os', 'nt': 'os', '_pickle': 'pickle', '_io': 'io',
           '_operator': 'operator', '_functools': 'functools', '_socket': 'socket',
           '_thread': '_thread'}
MODULE_ATTRS = {
    'shutil.os': 'os', 'shutil.sys': 'sys', 'shutil.collections': 'collections',
    'os.path': 'posixpath', 'os.sys': 'sys', 'os.posix': 'os',
    'subprocess.os': 'os', 'subprocess.sys': 'sys', 'subprocess.signal': 'signal',
    'pathlib.os': 'os', 'pathlib.io': 'io', 'pathlib.sys': 'sys',
    'pickle.io': 'io', 'pickle.sys': 'sys', 'pickle.codecs': 'codecs',
    'pickle.struct': 'struct', 'pickle.re': 're', 'pickle._compat_pickle': '_compat_pickle',
    'copy.copyreg': 'copyreg', 'copy.types': 'types',
    'functools.types': 'types', 'functools.sys': 'sys',
    'operator._operator': 'operator', 'io._io': 'io', 'codecs.sys': 'sys',
    'code.sys': 'sys', 'code.builtins': 'builtins',
    'pdb.os': 'os', 'pdb.sys': 'sys', 'pdb.bdb': 'bdb',
    'bdb.os': 'os', 'bdb.sys': 'sys', 'bdb.types': 'types',
    'timeit.sys': 'sys', 'timeit.itertools': 'itertools',
    'pkgutil.sys': 'sys', 'pkgutil.importlib': 'importlib',
    'importlib.sys': 'sys', 'importlib._bootstrap.sys': 'sys',
    'urllib.request.os': 'os', 'urllib.request.socket': 'socket',
    'urllib.request.http': 'http', 'http.client.socket': 'socket',
    'ftplib.socket': 'socket', 'smtplib.socket': 'socket',
    'socket.os': 'os', 'socket.sys': 'sys', 'socket._socket': 'socket',
    'socket.io': 'io', 'ssl.socket': 'socket',
    'logging.os': 'os', 'logging.sys': 'sys', 'logging.threading': 'threading',
    'logging.config.os': 'os', 'logging.config.sys': 'sys',
    'numpy.os': 'os', 'numpy.sys': 'sys',
    'numpy.ctypeslib.ctypes': 'ctypes', 'torch.serialization.pickle': 'pickle',
    'torch.serialization.os': 'os', 'torch.serialization.io': 'io',
    'joblib.numpy_pickle.pickle': 'pickle', 'joblib.numpy_pickle.np': 'numpy',
}

# Exact callable capabilities, plus families with the same documented effect.
CAPS = {}
def _caps(kind, module, names):
    CAPS.update((module + '.' + name, kind) for name in names.split())

_caps('code-execution', 'builtins', 'eval exec compile')
_caps('code-execution', 'runpy', 'run_path run_module _run_code _run_module_code')
_caps('code-execution', 'timeit', 'timeit repeat Timer')
_caps('code-execution', 'code', 'InteractiveInterpreter.runsource InteractiveInterpreter.runcode InteractiveConsole.push interact')
_caps('code-execution', 'bdb', 'Bdb.run Bdb.runeval Bdb.runctx Bdb.runcall')
_caps('code-execution', 'pdb', 'run runeval runctx runcall Pdb.run Pdb.runeval Pdb.runctx Pdb.runcall')
_caps('code-execution', 'numpy.testing._private.utils', 'runstring')
_caps('code-execution', 'types', 'FunctionType LambdaType')
_caps('process', 'os', '''system popen execv execve execvp execvpe execl execle execlp execlpe
spawnl spawnle spawnlp spawnlpe spawnv spawnve spawnvp spawnvpe posix_spawn posix_spawnp
fork forkpty startfile''')
_caps('process', 'subprocess', 'Popen run call check_call check_output getoutput getstatusoutput')
_caps('process', 'pty', 'spawn fork')
_caps('process', 'pydoc', 'pipepager tempfilepager')
_caps('process', 'multiprocessing', 'Process.start Pool')
_caps('process', 'multiprocessing.pool', 'Pool')
_caps('process', 'concurrent.futures', 'ProcessPoolExecutor')
_caps('file-access', 'builtins', 'open')
_caps('file-access', 'io', 'open open_code FileIO BufferedReader BufferedWriter BufferedRandom TextIOWrapper')
_caps('file-access', 'os', '''open fdopen remove unlink rename renames replace mkdir makedirs rmdir
removedirs listdir scandir walk fwalk stat lstat chmod chown lchown fchmod fchown truncate
utime link symlink readlink chdir chroot access mknod mkfifo read write readv writev pread pwrite
sendfile fsync fdatasync ftruncate close dup dup2''')
_caps('file-access', 'posixpath', 'getsize getmtime getatime getctime exists lexists isfile isdir islink ismount samefile realpath')
_caps('file-access', 'shutil', 'copy copy2 copyfile copyfileobj copymode copystat copytree move rmtree chown disk_usage make_archive unpack_archive')
_caps('file-access', 'pathlib', '''Path.open Path.read_text Path.read_bytes Path.write_text Path.write_bytes
Path.unlink Path.rename Path.replace Path.mkdir Path.rmdir Path.touch Path.stat Path.lstat
Path.iterdir Path.glob Path.rglob Path.exists Path.is_file Path.is_dir Path.is_symlink Path.chmod
Path.lchmod Path.chown Path.symlink_to Path.hardlink_to Path.link_to Path.readlink Path.resolve
PosixPath.open PosixPath.read_text PosixPath.read_bytes PosixPath.write_text PosixPath.write_bytes
PosixPath.unlink PosixPath.rename PosixPath.replace PosixPath.mkdir PosixPath.rmdir PosixPath.touch''')
_caps('file-access', 'tempfile', 'mkstemp mkdtemp NamedTemporaryFile TemporaryFile TemporaryDirectory SpooledTemporaryFile')
_caps('file-access', 'glob', 'glob iglob')
_caps('file-access', 'linecache', 'getline getlines updatecache')
_caps('file-access', 'tokenize', 'open')
_caps('file-access', 'fileinput', 'input FileInput')
_caps('file-access', 'sqlite3', 'connect')
_caps('file-access', 'dbm', 'open')
_caps('file-access', 'shelve', 'open')
_caps('file-access', 'h5py', 'File')
_caps('file-access', 'numpy', 'save savez savez_compressed savetxt loadtxt genfromtxt fromfile memmap')
_caps('file-access', 'gzip', 'open GzipFile')
_caps('file-access', 'bz2', 'open BZ2File')
_caps('file-access', 'lzma', 'open LZMAFile')
_caps('file-access', 'zipfile', 'ZipFile PyZipFile')
_caps('file-access', 'tarfile', 'open TarFile.open')
_caps('network', 'socket', 'create_connection create_server socket.connect socket.connect_ex socket.sendto socket.sendmsg getaddrinfo gethostbyname gethostbyname_ex gethostbyaddr')
_caps('network', 'urllib.request', 'urlopen urlretrieve OpenerDirector.open AbstractHTTPHandler.do_open')
_caps('network', 'http.client', 'HTTPConnection.connect HTTPConnection.request HTTPConnection.endheaders HTTPSConnection.connect HTTPSConnection.request')
_caps('network', 'ftplib', 'FTP.connect FTP_TLS.connect FTP.login FTP.retrbinary FTP.storbinary')
_caps('network', 'smtplib', 'SMTP.connect SMTP_SSL.connect LMTP.connect')
_caps('network', 'ssl', 'get_server_certificate')
_caps('network', 'webbrowser', 'open open_new open_new_tab')
_caps('network', 'requests', 'get post put delete head options patch request Session.request Session.get Session.post')
_caps('network', 'urllib3', 'request PoolManager.request HTTPConnectionPool.urlopen')
_caps('network', 'urllib.request', 'ProxyHandler.proxy_open HTTPHandler.http_open HTTPSHandler.https_open FTPHandler.ftp_open')

NESTED = {
    'pickle.loads': 'pickle-one', 'pickle.load': 'pickle-file',
    'pickle._loads': 'pickle-one', 'pickle._load': 'pickle-file',
    'torch.load': 'torch', 'torch.serialization.load': 'torch',
    'torch.storage._load_from_bytes': 'torch-bytes',
    'numpy.load': 'numpy', 'numpy.lib.npyio.load': 'numpy',
    'numpy.lib._npyio_impl.load': 'numpy',
    'numpy.lib.format.read_array': 'numpy-array',
    'joblib.load': 'joblib', 'joblib.numpy_pickle.load': 'joblib',
    'pandas.read_pickle': 'pandas', 'pandas.io.pickle.read_pickle': 'pandas',
}


def canonical_module(module):
    return ALIASES.get(module, module)


def looks_like_pickle(data):
    if not data:
        return False
    # Probe opcode syntax, not values or strings. genops stops at the first
    # STOP, and never invokes any callable.
    if chr(data[0]) not in pickletools.code2op:
        return False
    try:
        for i, _ in enumerate(pickletools.genops(data)):
            if i > MAX_STEPS:
                return True
        return True
    except Exception:
        return data[:1] == b'\x80' and len(data) > 1 and data[1] <= 5


class Context:
    def __init__(self):
        self.modules = {}
        self.refs = {}
        self.steps = 0
        self.depth = 0
        self.findings = []
        self.modules_view = Modules()
        self.extensions = {}

    def tick(self):
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise AnalysisLimit('pickle instruction/callback budget exceeded')

    def module(self, name):
        name = canonical_module(name)
        if name in _compat_pickle.IMPORT_MAPPING and name not in MODULE_EXPORTS:
            raise LoadError('Python 2 module is unavailable without fix_imports: ' + name)
        if name not in self.modules:
            self.modules[name] = Module(name)
        return self.modules[name]

    def ref(self, name):
        if name not in self.refs:
            self.refs[name] = Ref(name)
        return self.refs[name]


class VM:
    def __init__(self, context, location, persistent=False):
        self.ctx = context
        self.location = location
        self.pos = 0
        self.persistent = persistent
        self.proto = 0
        self.memo = {}

    def finding(self, kind, detail):
        if len(self.ctx.findings) < 100:
            self.ctx.findings.append(Finding(kind, f'{self.location}@{self.pos}', detail))

    def resolve(self, module, name):
        if isinstance(module, Unknown) or isinstance(name, Unknown):
            raise AnalysisLimit('cannot resolve symbolic global name')
        if not isinstance(module, str) or not isinstance(name, str):
            raise LoadError('global names must be strings')
        if self.proto < 3:
            module, name = _compat_pickle.NAME_MAPPING.get((module, name), (module, name))
            module = _compat_pickle.IMPORT_MAPPING.get(module, module)
        if self.proto < 4 and '.' in name:
            raise LoadError('dotted GLOBAL requires protocol 4')
        obj = self.ctx.module(module)
        for part in name.split('.'):
            obj = self.attr(obj, part)
        return obj

    def attr(self, obj, name, default=MISSING):
        if isinstance(name, Unknown):
            raise AnalysisLimit('cannot resolve symbolic attribute name')
        if not isinstance(name, str):
            raise LoadError('attribute name is not a string')
        if isinstance(obj, (Module, Ref, Instance, Partial)):
            if name in obj.attrs:
                return obj.attrs[name]
            if name == '__dict__':
                if isinstance(obj, Module):
                    return Adapter('namespace', (obj,))
                return obj.attrs
        if isinstance(obj, Module):
            full = obj.name + '.' + name
            if full in MODULE_ATTRS:
                return self.ctx.module(MODULE_ATTRS[full])
            if full == 'sys.modules':
                return self.ctx.modules_view
            if full == 'socket.SocketType':
                return self.ctx.ref('socket.socket')
            constants = {'os.name': 'posix', 'os.sep': '/', 'os.pathsep': ':', 'os.linesep': '\n',
                         'os.devnull': '/dev/null', 'sys.platform': 'linux', 'sys.maxsize': 2**63 - 1,
                         'posixpath.sep': '/', 'posixpath.curdir': '.', 'posixpath.pardir': '..'}
            if full in constants: return constants[full]
            if name == '__builtins__':
                return self.ctx.module('builtins')
            if full in ('os.environ',):
                return {}
            if full in ('os.O_RDONLY', 'os.O_WRONLY', 'os.O_RDWR', 'os.O_CREAT', 'os.O_TRUNC'):
                return {'O_RDONLY': 0, 'O_WRONLY': 1, 'O_RDWR': 2, 'O_CREAT': 64, 'O_TRUNC': 512}[name]
            if obj.name in MODULE_EXPORTS and name not in MODULE_EXPORTS[obj.name]:
                if default is not MISSING: return default
                raise LoadError('module has no attribute ' + name)
            return self.ctx.ref(full)
        if isinstance(obj, Ref):
            if name == '__module__':
                return obj.name.rsplit('.', 1)[0]
            if name in ('__name__', '__qualname__'):
                return obj.name.rsplit('.', 1)[-1]
            if name in ('__call__', '__get__'):
                return Bound(obj, name)
            if name == '__globals__':
                return Adapter('namespace', (self.ctx.module(obj.name.rsplit('.', 1)[0]),))
            return self.ctx.ref(obj.name + '.' + name)
        if isinstance(obj, Partial):
            if name == 'func': return obj.func
            if name == 'args': return obj.args
            if name == 'keywords': return obj.kw
            return Bound(obj, name)
        if isinstance(obj, Instance):
            if name == '__class__': return self.ctx.ref(obj.cls)
            if obj.cls == 'numpy.NpzFile' and name == 'files':
                import zipfile
                with zipfile.ZipFile(io.BytesIO(obj.attrs['_data'])) as archive:
                    return [n[:-4] if n.endswith('.npy') else n for n in archive.namelist()]
            if obj.cls == 'numpy.dtype':
                if name == 'hasobject': return self.dtype_info(obj)[1]
                if name == 'itemsize': return self.dtype_info(obj)[0]
            return Bound(obj, name)
        if isinstance(obj, (str, bytes, bytearray, tuple, list, dict, set, Buffer, Lazy, Adapter, Modules)):
            return Bound(obj, name)
        if default is not MISSING:
            return default
        return Unknown(f'attribute {name}')

    def item(self, obj, key, default=MISSING):
        if isinstance(key, Unknown):
            raise AnalysisLimit('cannot resolve symbolic item key')
        try:
            if isinstance(obj, Modules):
                return self.ctx.module(key)
            if isinstance(obj, Adapter) and obj.kind == 'namespace':
                return self.attr(obj.args[0], key, default)
            if isinstance(obj, (dict, list, tuple, str, bytes, bytearray)):
                return obj[key]
            if isinstance(obj, Instance):
                if obj.cls == 'numpy.NpzFile':
                    import zipfile
                    with zipfile.ZipFile(io.BytesIO(obj.attrs['_data'])) as archive:
                        name = key if key in archive.namelist() else key + '.npy'
                        info = archive.getinfo(name)
                        if info.file_size > MAX_DATA: raise AnalysisLimit('NumPy member size')
                        data = archive.read(info)
                    return self.numpy_data(data, obj.attrs['_allow_pickle'], self.location + '!npz-member')
                if '_values' in obj.attrs:
                    return obj.attrs['_values'][key]
                if obj.cls in ('collections.OrderedDict', 'collections.defaultdict', 'builtins.dict'):
                    return obj.attrs.get('_items', {})[key]
                return self.call(self.attr(obj, '__getitem__'), (key,))
        except (KeyError, IndexError, TypeError):
            if default is not MISSING: return default
            raise LoadError('invalid item access')
        return Unknown('item')

    def setitem(self, obj, key, value):
        if isinstance(obj, Adapter) and obj.kind == 'namespace':
            obj.args[0].attrs[key] = value
        elif isinstance(obj, Modules):
            self.ctx.modules[canonical_module(key)] = value
        elif isinstance(obj, (dict, list)):
            obj[key] = value
        elif isinstance(obj, Instance):
            obj.attrs.setdefault('_items', {})[key] = value
        else:
            raise LoadError('object does not support item assignment')

    def build(self, obj, state):
        if isinstance(obj, Partial):
            if not isinstance(state, tuple) or len(state) != 4:
                raise LoadError('invalid partial state')
            func, args, kw, attrs = state
            if not isinstance(args, tuple) or not isinstance(kw, dict):
                raise LoadError('invalid partial state')
            obj.func, obj.args, obj.kw = func, args, kw
            if isinstance(attrs, dict): obj.attrs.update(attrs)
            return
        if isinstance(obj, Unknown):
            # Numerical reconstruction results can have inert state, e.g.
            # tensors. This must not hide later opcodes in the stream.
            return
        if isinstance(obj, (Instance, Ref, Module)):
            setter = obj.attrs.get('__setstate__')
            if setter is not None:
                self.call(setter, (state,))
                return
            if isinstance(obj, Instance): obj.state = state
            if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
                state, slots = state
                obj.attrs.update(slots)
            if isinstance(state, dict): obj.attrs.update(state)
            if isinstance(obj, Instance) and obj.cls == 'numpy.ndarray' and isinstance(state, tuple) and state:
                if isinstance(state[-1], list): obj.attrs['_values'] = state[-1]
            return
        raise LoadError('invalid BUILD target')

    def run(self, stream):
        stack = []
        self.memo = {}
        self.proto = 0
        def marked():
            items = []
            while stack:
                value = stack.pop()
                if value is MARK:
                    return items[::-1]
                items.append(value)
            raise LoadError('missing MARK')
        for op, arg, pos in pickletools.genops(stream):
            self.ctx.tick()
            self.pos = pos
            n = op.name
            if n in ('PROTO', 'FRAME'):
                if n == 'PROTO':
                    if arg > 5: raise LoadError('unsupported pickle protocol')
                    self.proto = arg
            elif n == 'MARK': stack.append(MARK)
            elif n == 'STOP':
                if not stack: raise LoadError('empty pickle stack')
                return stack[-1]
            elif n == 'POP': stack.pop()
            elif n == 'POP_MARK': marked()
            elif n == 'DUP': stack.append(stack[-1])
            elif n == 'NONE': stack.append(None)
            elif n == 'NEWTRUE': stack.append(True)
            elif n == 'NEWFALSE': stack.append(False)
            elif n in ('INT', 'BININT', 'BININT1', 'BININT2', 'LONG', 'LONG1', 'LONG4',
                       'FLOAT', 'BINFLOAT', 'STRING', 'BINSTRING', 'SHORT_BINSTRING',
                       'UNICODE', 'BINUNICODE', 'SHORT_BINUNICODE', 'BINUNICODE8',
                       'BINBYTES', 'SHORT_BINBYTES', 'BINBYTES8'):
                stack.append(arg)
            elif n == 'BYTEARRAY8': stack.append(bytearray(arg))
            elif n in ('PUT', 'BINPUT', 'LONG_BINPUT'): self.memo[arg] = stack[-1]
            elif n == 'MEMOIZE': self.memo[len(self.memo)] = stack[-1]
            elif n in ('GET', 'BINGET', 'LONG_BINGET'):
                if arg not in self.memo: raise LoadError('missing memo entry')
                stack.append(self.memo[arg])
            elif n == 'EMPTY_TUPLE': stack.append(())
            elif n == 'EMPTY_LIST': stack.append([])
            elif n == 'EMPTY_DICT': stack.append({})
            elif n == 'EMPTY_SET': stack.append(set())
            elif n in ('TUPLE', 'LIST', 'DICT', 'FROZENSET'):
                items = marked()
                if n == 'TUPLE': stack.append(tuple(items))
                elif n == 'LIST': stack.append(items)
                elif n == 'FROZENSET': stack.append(frozenset(items))
                else:
                    obj = {}
                    for k, v in zip(items[::2], items[1::2]): self.setitem(obj, k, v)
                    stack.append(obj)
            elif n in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
                count = int(n[-1]); items = stack[-count:]; del stack[-count:]
                stack.append(tuple(items))
            elif n == 'APPEND':
                value = stack.pop(); obj = stack[-1]
                self.append(obj, [value])
            elif n == 'APPENDS':
                items = marked(); self.append(stack[-1], items)
            elif n == 'ADDITEMS':
                items = marked()
                if isinstance(stack[-1], set): stack[-1].update(items)
                else:
                    for item in items: self.call(self.attr(stack[-1], 'add'), (item,))
            elif n == 'SETITEM':
                value = stack.pop(); key = stack.pop(); self.setitem(stack[-1], key, value)
            elif n == 'SETITEMS':
                items = marked()
                for k, v in zip(items[::2], items[1::2]): self.setitem(stack[-1], k, v)
            elif n == 'GLOBAL':
                module, name = arg.split(' ', 1); stack.append(self.resolve(module, name))
            elif n == 'STACK_GLOBAL':
                name = stack.pop(); module = stack.pop(); stack.append(self.resolve(module, name))
            elif n in ('EXT1', 'EXT2', 'EXT4'):
                if arg not in self.ctx.extensions:
                    raise LoadError('unregistered pickle extension')
                module, name = self.ctx.extensions[arg]
                stack.append(self.resolve(module, name))
            elif n == 'REDUCE':
                args = stack.pop(); func = stack.pop()
                if isinstance(args, Unknown): raise AnalysisLimit('symbolic REDUCE arguments')
                if not isinstance(args, tuple): raise LoadError('REDUCE arguments must be a tuple')
                stack.append(self.call(func, args))
            elif n in ('INST', 'OBJ'):
                args = marked()
                if n == 'INST':
                    module, name = arg.split(' ', 1); func = self.resolve(module, name)
                else:
                    if not args: raise LoadError('OBJ has no callable')
                    func, args = args[0], args[1:]
                stack.append(self.call(func, tuple(args), construct='inst'))
            elif n in ('NEWOBJ', 'NEWOBJ_EX'):
                kw = stack.pop() if n == 'NEWOBJ_EX' else {}
                args = stack.pop(); cls = stack.pop()
                if not isinstance(args, tuple) or not isinstance(kw, dict):
                    raise LoadError('invalid NEWOBJ arguments')
                stack.append(self.call(cls, args, kw, construct='new'))
            elif n == 'BUILD':
                state = stack.pop(); obj = stack[-1]; self.build(obj, state)
                if isinstance(obj, Instance) and obj.cls.endswith('.NumpyArrayWrapper'):
                    self.joblib_array(obj, stream)
                elif isinstance(obj, Instance) and obj.cls.endswith('.NDArrayWrapper'):
                    self.finding('file-access', 'legacy joblib array wrapper loads an external file')
            elif n in ('PERSID', 'BINPERSID'):
                pid = stack.pop() if n == 'BINPERSID' else arg
                if not self.persistent: raise LoadError('no persistent-id handler')
                stack.append(Instance('torch.Storage', (pid,)))
            elif n == 'NEXT_BUFFER':
                raise LoadError('no out-of-band buffers supplied')
            elif n == 'READONLY_BUFFER': pass
            else: raise AnalysisLimit('unmodelled opcode ' + n)
            if len(stack) > MAX_ITEMS or len(self.memo) > MAX_ITEMS:
                raise AnalysisLimit('pickle stack/memo limit')
        raise LoadError('missing STOP')

    def append(self, obj, items):
        if isinstance(obj, list): obj.extend(items)
        else:
            # The unpickler uses extend if available, falling back to append.
            if isinstance(obj, Instance) and 'extend' not in obj.attrs:
                obj.attrs.setdefault('_items_list', []).extend(items)
            else:
                self.call(self.attr(obj, 'extend'), (items,))

    def dtype_info(self, dtype):
        """Return (itemsize, hasobject) without importing or constructing NumPy."""
        if isinstance(dtype, Instance) and dtype.cls == 'numpy.dtype':
            state = dtype.state
            if isinstance(state, tuple) and len(state) >= 6:
                if isinstance(state[4], dict):
                    fields = [v[0] for v in state[4].values() if isinstance(v, tuple) and v]
                    return max(0, state[5]), any(self.dtype_info(v)[1] for v in fields)
                if state[2] and isinstance(state[2], tuple):
                    base, shape = state[2]
                    size, obj = self.dtype_info(base)
                    for dim in shape: size *= dim
                    return size, obj
            return self.dtype_info(dtype.args[0] if dtype.args else '')
        if isinstance(dtype, str):
            code = dtype.lstrip('<>|=')
            if code.startswith('O'): return 8, True
            names = {'bool': 1, 'int8': 1, 'uint8': 1, 'int16': 2, 'uint16': 2,
                     'int32': 4, 'uint32': 4, 'float32': 4, 'int64': 8,
                     'uint64': 8, 'float64': 8, 'complex64': 8, 'complex128': 16}
            if code in names: return names[code], False
            if len(code) > 1 and code[1:].isdigit():
                return int(code[1:]) * (4 if code[0] == 'U' else 1), False
            if code in ('b', 'B', '?'): return 1, False
        raise AnalysisLimit('unmodelled NumPy dtype')

    def joblib_array(self, obj, stream):
        size, hasobject = self.dtype_info(obj.attrs.get('dtype'))
        if hasobject:
            start = stream.tell()
            data = stream.getbuffer()[start:].tobytes()
            value, used = interpret(data, self.location + '!joblib-array', mode='one',
                                    context=self.ctx, strict=True)
            stream.seek(start + used)
            if isinstance(value, Instance):
                obj.attrs.update(value.attrs)
            obj.cls = 'numpy.ndarray'
            return
        count = 1
        for dim in obj.attrs.get('shape', ()):
            if not isinstance(dim, int) or dim < 0: raise LoadError('invalid array shape')
            count *= dim
        alignment = obj.attrs.get('numpy_array_alignment_bytes')
        if alignment is not None:
            padding = stream.read(1)
            if not padding: raise LoadError('missing joblib alignment padding')
            stream.seek(padding[0], 1)
        end = stream.tell() + count * size
        if end > len(stream.getbuffer()): raise LoadError('truncated joblib array')
        stream.seek(end)
        obj.cls = 'numpy.ndarray'

    def call(self, func, args=(), kw=None, construct=None):
        self.ctx.tick()
        self.ctx.depth += 1
        if self.ctx.depth > 80:
            self.ctx.depth -= 1
            raise AnalysisLimit('nested call limit')
        try:
            result = self._call(func, args, {} if kw is None else kw, construct)
            if isinstance(result, (str, bytes, bytearray, list, tuple, dict, set)) and len(result) > MAX_DATA:
                raise AnalysisLimit('literal result exceeds size limit')
            return result
        finally:
            self.ctx.depth -= 1

    def _call(self, func, args, kw, construct):
        if isinstance(func, Partial):
            if construct == 'new': raise LoadError('NEWOBJ requires a class')
            return self.call(func.func, func.args + args, {**func.kw, **kw})
        if isinstance(func, Bound):
            if construct == 'new': raise LoadError('NEWOBJ requires a class')
            return self.bound(func.obj, func.name, args, kw)
        if isinstance(func, Adapter):
            if construct == 'new': raise LoadError('NEWOBJ requires a class')
            if func.kind == 'attrgetter':
                values = []
                for name in func.args:
                    obj = args[0]
                    for part in name.split('.'): obj = self.attr(obj, part)
                    values.append(obj)
                return values[0] if len(values) == 1 else tuple(values)
            if func.kind == 'itemgetter':
                values = [self.item(args[0], k) for k in func.args]
                return values[0] if len(values) == 1 else tuple(values)
            if func.kind == 'methodcaller':
                return self.call(self.attr(args[0], func.args[0]), func.args[1:], func.kw)
            if func.kind in ('cache', 'wrap'):
                return self.call(func.args[0], args, kw)
            if func.kind == 'cache-decorator':
                return Adapter('cache', (args[0],))
            if func.kind == 'compose':
                value = self.call(func.args[-1], args, kw)
                for inner in reversed(func.args[:-1]): value = self.call(inner, (value,))
                return value
            raise LoadError('noncallable adapter')
        if isinstance(func, Instance):
            target = func.attrs.get('__call__')
            if target is not None:
                return self.call(target, (func,) + args, kw)
            if func.cls == 'types.FunctionType':
                self.finding('code-execution', 'call serialized Python function')
                return Unknown()
            raise LoadError('object is not callable')
        if not isinstance(func, Ref):
            if isinstance(func, Unknown):
                raise AnalysisLimit('cannot resolve callable: ' + func.description)
            raise LoadError('object is not callable')
        name = func.name
        # Attribute/global rebinding is already resolved before this point.
        if name in ('functools.partial.__call__', 'operator.attrgetter.__call__',
                    'operator.itemgetter.__call__', 'operator.methodcaller.__call__'):
            if not args: raise LoadError('descriptor requires an instance')
            return self.call(args[0], args[1:], kw)
        method_owner, _, method_name = name.rpartition('.')
        unbound_methods = {
            'threading.Thread': {'start', 'run'}, 'threading.Timer': {'start', 'run'},
            'concurrent.futures.ThreadPoolExecutor': {'submit', 'map'},
            'concurrent.futures.ProcessPoolExecutor': {'submit', 'map'},
            'concurrent.futures.thread.ThreadPoolExecutor': {'submit', 'map'},
            'concurrent.futures.process.ProcessPoolExecutor': {'submit', 'map'},
            'multiprocessing.pool.ThreadPool': {'apply', 'apply_async', 'map', 'starmap'},
            'pickle.Unpickler': {'load'}, 'pickle._Unpickler': {'load'},
            'timeit.Timer': {'timeit', 'repeat', 'autorange'},
            'logging.config.DictConfigurator': {'configure'},
        }
        if method_name in unbound_methods.get(method_owner, ()):
            if not args: raise LoadError('unbound method requires an instance')
            return self.bound(args[0], method_name, args[1:], kw)
        if name == 'builtins.type.__call__':
            if not args: raise LoadError('type.__call__ arity')
            return self.call(args[0], args[1:], kw)
        if name.endswith('.__call__'):
            return self.call(self.ctx.ref(name[:-9]), args, kw)
        if name.endswith('.__new__'):
            if not args: raise LoadError('__new__ requires class')
            return self.call(args[0], args[1:], kw, construct='new')
        if name in ('sys.exit', 'os._exit', 'builtins.exit', 'builtins.quit'):
            raise LoadError('loader terminates')
        is_class = self.is_class(func)
        new_only = construct == 'new' or (construct == 'inst' and not args and is_class)
        if construct == 'new' and not is_class:
            raise LoadError('NEWOBJ requires a class')
        if new_only and name in ('io.BytesIO', 'io.StringIO'):
            return Buffer(b'' if name.endswith('BytesIO') else '')
        if new_only and name not in ('functools.partial', 'types.FunctionType', 'types.LambdaType',
                                     'builtins.bytes', 'builtins.bytearray', 'builtins.str',
                                     'builtins.tuple', 'pathlib.Path', 'pathlib.PosixPath',
                                     'pathlib.PurePath', 'pathlib.PurePosixPath', 'builtins.type',
                                     'operator.attrgetter', 'operator.itemgetter', 'operator.methodcaller'):
            return Instance(name, args, {'_new_only': True})
        if name.startswith('dynamic.'):
            obj = Instance(name, args, {'_class_ref': func})
            if '__new__' in func.attrs:
                obj = self.call(func.attrs['__new__'], (func,) + args, kw)
            if not new_only and '__init__' in func.attrs:
                self.call(func.attrs['__init__'], (obj,) + args, kw)
            if isinstance(obj, Instance):
                for k, v in func.attrs.items():
                    if k.startswith('__'): obj.attrs[k] = v
            return obj
        if name in ('builtins.getattr',):
            if len(args) < 2: raise LoadError('getattr arity')
            return self.attr(args[0], args[1], args[2] if len(args) > 2 else MISSING)
        if name == 'builtins.hasattr':
            try:
                self.attr(args[0], args[1])
                return True
            except LoadError:
                return False
        if name in ('builtins.setattr', 'builtins.delattr'):
            if len(args) < 2: raise LoadError('setattr arity')
            obj, attr = args[:2]
            if not isinstance(obj, (Module, Ref, Instance, Partial)): raise LoadError('no attributes')
            if name.endswith('setattr'): obj.attrs[attr] = args[2]
            else: obj.attrs.pop(attr, None)
            return None
        if name in ('builtins.__import__', 'importlib.import_module', 'importlib.__import__'):
            if args and isinstance(args[0], Unknown): raise AnalysisLimit('symbolic module name')
            if not args or not isinstance(args[0], str): raise LoadError('module name must be text')
            module = args[0]
            if name == 'builtins.__import__' and '.' in module:
                fromlist = args[3] if len(args) > 3 else kw.get('fromlist', ())
                if not fromlist: module = module.split('.')[0]
            return self.ctx.module(module)
        if name == 'pkgutil.resolve_name':
            text = args[0]
            if ':' in text:
                module, _, path = text.partition(':')
            else:
                module, _, path = text.rpartition('.')
                if not module: return self.ctx.module(path)
                module = MODULE_ATTRS.get(module, module)
            obj = self.ctx.module(module)
            for part in path.split('.') if path else (): obj = self.attr(obj, part)
            return obj
        if name in ('pydoc.locate', 'logging.config._resolve'):
            text = args[0]
            module, _, path = text.rpartition('.')
            module = MODULE_ATTRS.get(module, module)
            return self.attr(self.ctx.module(module), path)
        if name in ('builtins.vars', 'builtins.globals', 'builtins.locals'):
            if not args: return Adapter('namespace', (self.ctx.module('builtins'),))
            return self.attr(args[0], '__dict__')
        if name in ('types.MethodType',):
            if len(args) < 2: raise LoadError('MethodType arity')
            return Partial(args[0], (args[1],))
        if name in ('builtins.staticmethod', 'builtins.classmethod'):
            return args[0]
        if name == 'functools.partial':
            if not args: raise LoadError('partial requires a callable')
            if isinstance(args[0], Partial):
                inner = args[0]
                return Partial(inner.func, inner.args + args[1:], {**inner.kw, **kw})
            return Partial(args[0], args[1:], kw.copy())
        if name in ('functools.partial.__setstate__', 'functools.partial.__setstate'):
            self.build(args[0], args[1]); return None
        if name in ('operator.attrgetter', 'operator.itemgetter', 'operator.methodcaller'):
            return Adapter(name.split('.')[-1], args, kw.copy())
        if name in ('operator.getitem', 'operator.__getitem__'):
            return self.item(*args)
        if name in ('operator.setitem', 'operator.__setitem__'):
            self.setitem(*args); return None
        if name in ('operator.delitem', 'operator.__delitem__'):
            obj, key = args
            if isinstance(obj, dict): obj.pop(key, None)
            return None
        if name == 'operator.call':
            return self.call(args[0], args[1:], kw)
        if name in ('functools.lru_cache', 'functools.cache'):
            if args and isinstance(args[0], (Ref, Partial, Bound, Adapter)):
                return Adapter('cache', (args[0],))
            return Adapter('cache-decorator', ())
        if name in ('functools.update_wrapper', 'functools.wraps'):
            if name.endswith('wraps'): return Adapter('cache-decorator', ())
            return args[0]
        if name in ('copy.copy', 'copy.deepcopy'):
            obj = args[0]
            if isinstance(obj, (Ref, Module, Adapter, Bound)): return obj
            if isinstance(obj, Partial): return Partial(obj.func, obj.args, obj.kw.copy(), obj.attrs.copy())
            if isinstance(obj, (list, dict, set, bytearray)): return obj.copy() if not isinstance(obj, bytearray) else obj[:]
            return obj
        if name in ('copyreg._reconstructor',):
            cls, base, state = args
            if isinstance(base, Ref) and base.name != 'builtins.object':
                self.call(self.attr(base, '__new__'), (cls, state))
            return self.call(cls, (), construct='new')
        if name in ('copyreg.add_extension', 'copyreg.remove_extension', 'copyreg.clear_extension_cache'):
            if name.endswith('add_extension'):
                self.ctx.extensions[args[2]] = (args[0], args[1])
            elif name.endswith('remove_extension'):
                self.ctx.extensions.pop(args[2], None)
            return None
        if name in ('copyreg.__newobj__', 'copyreg.__newobj_ex__'):
            if name.endswith('_ex__'): return self.call(args[0], args[1], args[2], construct='new')
            return self.call(args[0], args[1:], construct='new')
        if name in ('builtins.object.__getattribute__', 'builtins.type.__getattribute__'):
            return self.attr(*args)
        if name == 'builtins.type':
            if len(args) == 3:
                clsname, bases, attrs = args
                return Ref('dynamic.' + str(clsname), dict(attrs))
            if len(args) == 1:
                obj = args[0]
                if isinstance(obj, Instance): return self.ctx.ref(obj.cls)
                return self.ctx.ref('builtins.' + type(obj).__name__)
        if name in ('builtins.map', 'builtins.filter', 'itertools.starmap', 'itertools.filterfalse',
                    'itertools.takewhile', 'itertools.dropwhile'):
            if len(args) < 2: raise LoadError('iterator requires callable and iterable')
            return Lazy(name.rsplit('.', 1)[-1], args[0], args[1:])
        if name == 'itertools.groupby':
            return Lazy('groupby', args[1] if len(args) > 1 else kw.get('key'), (args[0],))
        if name in ('builtins.iter', 'builtins.reversed', 'builtins.enumerate', 'builtins.zip',
                    'itertools.chain', 'itertools.chain.from_iterable', 'itertools.islice',
                    'itertools.repeat', 'itertools.cycle', 'itertools.accumulate', 'itertools.tee'):
            if name == 'builtins.iter' and len(args) == 2:
                return Lazy('sentinel', args[0], (args[1],))
            if name == 'itertools.accumulate':
                return Lazy('accumulate', args[1] if len(args) > 1 else kw.get('func'), (args[0],),
                            state={'initial': kw.get('initial', MISSING)})
            if name == 'builtins.enumerate':
                return Lazy('enumerate', None, (args[0],), state={'count': args[1] if len(args) > 1 else kw.get('start', 0)})
            if name == 'itertools.tee':
                count = args[1] if len(args) > 1 else 2
                if not isinstance(count, int) or count > 100: raise AnalysisLimit('tee size')
                return tuple(Lazy('iter', None, (args[0],)) for _ in range(count))
            return Lazy(name.rsplit('.', 1)[-1], None, args)
        if name == 'builtins.next':
            try: return self.next_value(args[0])
            except StopIteration:
                if len(args) > 1: return args[1]
                raise LoadError('exhausted iterator')
        if name in ('builtins.list', 'builtins.tuple', 'builtins.set', 'builtins.frozenset',
                    'collections.deque'):
            values = self.consume(args[0]) if args else []
            return {'builtins.list': list, 'builtins.tuple': tuple, 'builtins.set': set,
                    'builtins.frozenset': frozenset, 'collections.deque': list}[name](values)
        if name in ('builtins.dict', 'collections.OrderedDict', 'collections.defaultdict'):
            if name == 'collections.defaultdict':
                factory = args[0] if args else None
                rest = args[1:]
                obj = Instance(name, args, {'default_factory': factory, '_items': {}})
                if rest: obj.attrs['_items'].update(self.to_dict(rest[0]))
                return obj
            result = self.to_dict(args[0]) if args else {}
            result.update(kw)
            if name == 'collections.OrderedDict':
                return Instance(name, args, {'_items': result})
            return result
        if name in ('builtins.sorted', 'builtins.min', 'builtins.max'):
            values = self.consume(args[0]) if len(args) == 1 else list(args)
            key = kw.get('key')
            keyed = [self.call(key, (v,)) for v in values] if key is not None else values
            if name == 'builtins.sorted':
                try:
                    return [values[i] for i in sorted(range(len(values)), key=lambda i: keyed[i], reverse=bool(kw.get('reverse')))]
                except TypeError: return values
            if not values: return kw.get('default', Unknown('empty min/max'))
            try:
                index = (min if name.endswith('min') else max)(range(len(values)), key=lambda i: keyed[i])
                return values[index]
            except TypeError: return values[0]
        if name == 'functools.reduce':
            values = self.consume(args[1])
            if len(args) > 2: accum = args[2]
            elif values: accum = values.pop(0)
            else: raise LoadError('empty reduce')
            for value in values: accum = self.call(args[0], (accum, value))
            return accum
        if name in ('builtins.sum', 'builtins.any', 'builtins.all'):
            iterator = self.iterator(args[0])
            accum = args[1] if len(args) > 1 else 0
            while True:
                try: value = self.next_value(iterator)
                except StopIteration: break
                if name.endswith('any') and value: return True
                if name.endswith('all') and not value: return False
                if name.endswith('sum'): accum = self.literal(operator.add, (accum, value))
            return accum if name.endswith('sum') else name.endswith('all')
        if name in NESTED:
            return self.nested(name, args, kw)
        if name in ('pickle.Unpickler', 'pickle._Unpickler'):
            return Instance('pickle.Unpickler', args)
        if name in ('io.BytesIO', 'io.StringIO'):
            return Buffer(args[0] if args else (b'' if name.endswith('BytesIO') else ''))
        if name in ('io.BufferedReader', 'io.BufferedWriter', 'io.BufferedRandom', 'io.TextIOWrapper'):
            if not args: raise LoadError('stream wrapper needs stream')
            if isinstance(args[0], Buffer): return args[0]
            return Instance(name, args)
        if name in ('gzip.open', 'gzip.GzipFile', 'bz2.open', 'bz2.BZ2File', 'lzma.open', 'lzma.LZMAFile'):
            source = kw.get('fileobj', args[0] if args else kw.get('filename'))
            if isinstance(source, Buffer):
                from modelgate.dispatch import _decompress
                kind = name.split('.')[0].replace('lzma', 'xz')
                return Buffer(b''.join(_decompress(source.data[source.offset:], kind)))
            self.finding('file-access', 'call ' + name)
            return Instance(name, args)
        if name in ('zipfile.ZipFile', 'zipfile.PyZipFile', 'tarfile.open', 'tarfile.TarFile.open'):
            source = kw.get('fileobj', args[0] if args else kw.get('file', kw.get('name')))
            if isinstance(source, Buffer):
                return Instance('zipfile.ZipFile' if name.startswith('zipfile') else 'tarfile.TarFile',
                                (), {'_buffer': source})
            if source is not None:
                self.finding('file-access', 'call ' + name)
            return Instance(name, args)
        if name in ('marshal.loads', 'marshal.load'):
            if name.endswith('.load'):
                if not args or not isinstance(args[0], Buffer):
                    raise LoadError('marshal.load requires file')
                data = args[0].data[args[0].offset:]
            else: data = args[0] if args else b''
            if not isinstance(data, (bytes, bytearray)): raise LoadError('marshal requires bytes')
            # Marshal creates inert constants/code objects and has no object
            # hooks. Convert code to a symbolic value immediately; never make
            # a real Python function or execute the decoded code.
            try:
                raw = marshal.loads(data)
            except (EOFError, ValueError, TypeError):
                raise LoadError('invalid marshal data')
            def symbolic(value):
                if isinstance(value, types.CodeType): return Instance('marshal.code')
                if isinstance(value, tuple): return tuple(map(symbolic, value))
                if isinstance(value, list): return list(map(symbolic, value))
                if isinstance(value, dict): return {symbolic(k): symbolic(v) for k, v in value.items()}
                return value
            return symbolic(raw)
        if name == 'types.CodeType':
            return Instance('marshal.code', args)
        if name in ('types.FunctionType', 'types.LambdaType'):
            if len(args) < 2 or not isinstance(args[1], dict): raise LoadError('FunctionType arguments')
            if not isinstance(args[0], Instance) or args[0].cls != 'marshal.code':
                raise LoadError('FunctionType requires a code object')
            self.finding('code-execution', 'materialize serialized bytecode as a Python function')
            return Instance('types.FunctionType', args)
        if name in ('threading.Thread', 'threading.Timer'):
            # Positional Thread(group, target, name, args, kwargs).
            if name.endswith('Timer'):
                target = args[1] if len(args) > 1 else kw.get('function')
                targs = args[2] if len(args) > 2 else kw.get('args', ())
                tkw = args[3] if len(args) > 3 else kw.get('kwargs', {})
            else:
                target = args[1] if len(args) > 1 else kw.get('target')
                targs = args[3] if len(args) > 3 else kw.get('args', ())
                tkw = args[4] if len(args) > 4 else kw.get('kwargs', {})
            return Instance(name, args, {'_target': target, '_args': targs or (), '_kwargs': tkw or {}})
        if name in ('_thread.start_new_thread', '_thread.start_new', 'threading._start_new_thread'):
            if args: self.call(args[0], args[1] if len(args) > 1 else (), args[2] if len(args) > 2 else {})
            return 1
        if name in ('concurrent.futures.ThreadPoolExecutor', 'concurrent.futures.ProcessPoolExecutor',
                    'concurrent.futures.thread.ThreadPoolExecutor', 'concurrent.futures.process.ProcessPoolExecutor',
                    'multiprocessing.pool.ThreadPool'):
            return Instance(name, args)
        if name in ('logging.config.dictConfig', 'logging.config.dictConfigClass'):
            if args: self.logging_config(args[0])
            return None
        if name in ('timeit.timeit', 'timeit.repeat', 'timeit.Timer'):
            supplied_stmt = args[0] if args else kw.get('stmt', MISSING)
            supplied_setup = args[1] if len(args) > 1 else kw.get('setup', MISSING)
            timer = args[2] if len(args) > 2 else kw.get('timer', MISSING)
            for source in (supplied_stmt, supplied_setup):
                if isinstance(source, (str, bytes)):
                    self.finding('code-execution', 'timeit compiles supplied Python source')
            obj = Instance('timeit.Timer', (), {'_stmt': supplied_stmt, '_setup': supplied_setup, '_timer': timer})
            if name == 'timeit.Timer': return obj
            self.bound(obj, 'timeit', (), {'number': kw.get('number', args[3] if len(args) > 3 else 1000000)})
            return 0.0 if name.endswith('timeit') else [0.0]
        if name in ('pdb.runcall', 'bdb.Bdb.runcall', 'pdb.Pdb.runcall'):
            index = 0 if name == 'pdb.runcall' else 1
            if len(args) <= index: raise LoadError('runcall needs a callable')
            return self.call(args[index], args[index + 1:], kw)
        if name in ('logging.FileHandler', 'logging.handlers.RotatingFileHandler',
                    'logging.handlers.TimedRotatingFileHandler', 'logging.handlers.WatchedFileHandler'):
            delay_index = 3 if name == 'logging.FileHandler' else 5 if 'Rotating' in name else 3
            delay = args[delay_index] if len(args) > delay_index else kw.get('delay', False)
            if not delay and (args or kw.get('filename')):
                self.finding('file-access', 'call ' + name)
            return Instance(name, args, {'_delay': delay})
        if name == 'logging.config.fileConfig':
            self.finding('file-access', 'logging configuration reads file')
            return None
        if name in ('ftplib.FTP', 'ftplib.FTP_TLS', 'smtplib.SMTP', 'smtplib.SMTP_SSL', 'smtplib.LMTP',
                    'telnetlib.Telnet'):
            host = args[0] if args else kw.get('host', '')
            if host: self.finding('network', name + '(host)')
            return Instance(name, args)
        if name in ('socket.socket', 'http.client.HTTPConnection', 'http.client.HTTPSConnection',
                    'xmlrpc.client.ServerProxy', 'xmlrpc.client.Server'):
            return Instance(name, args)
        if name in ('pathlib.Path', 'pathlib.PosixPath', 'pathlib.PurePath', 'pathlib.PurePosixPath',
                    'pathlib.WindowsPath', 'pathlib.PureWindowsPath'):
            return Instance('pathlib.Path' if 'Pure' not in name else 'pathlib.PurePath', args)
        if name in CAPS:
            # Calling an unbound method needs a receiver; an invalid REDUCE
            # arity has no capability. Constructor-only __init__ effects were
            # excluded by new_only above.
            if not self.enough_args(name, args, kw): raise LoadError('capability call arity')
            self.finding(CAPS[name], 'call ' + name)
            return Instance(name, args) if is_class else Unknown('result of ' + name)
        if name in ('builtins.bytes', 'builtins.bytearray', 'builtins.str', 'builtins.int',
                    'builtins.float', 'builtins.complex', 'builtins.bool', 'builtins.range',
                    'builtins.slice'):
            if name in ('builtins.bytes', 'builtins.bytearray') and args and isinstance(args[0], Lazy):
                args = (self.consume(args[0]),) + args[1:]
            fn = {'bytes': bytes, 'bytearray': bytearray, 'str': str, 'int': int, 'float': float,
                  'complex': complex, 'bool': bool, 'range': range, 'slice': slice}[name[9:]]
            return self.literal(fn, args, kw)
        if name in ('builtins.len', 'builtins.abs', 'builtins.round', 'builtins.repr', 'builtins.ascii',
                    'builtins.chr', 'builtins.ord', 'builtins.divmod', 'builtins.pow'):
            fn = {'len': len, 'abs': abs, 'round': round, 'repr': repr, 'ascii': ascii,
                  'chr': chr, 'ord': ord, 'divmod': divmod, 'pow': pow}[name[9:]]
            return self.literal(fn, args, kw)
        if name == 'os.fspath':
            if args and isinstance(args[0], (str, bytes)): return args[0]
            if args and isinstance(args[0], Instance): return '/'.join(map(str, args[0].args))
            return Unknown('path')
        if name.startswith('operator.'):
            opname = name[9:]
            opname = opname.strip('_')
            if opname in ('add', 'concat', 'iadd', 'iconcat', 'sub', 'mul', 'imul', 'truediv', 'floordiv',
                          'mod', 'neg', 'pos', 'not', 'and', 'or', 'xor', 'lshift', 'rshift',
                          'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'index', 'truth', 'contains', 'countOf', 'indexOf'):
                fn = getattr(operator, opname + '_' if opname in ('not', 'and', 'or') else opname)
                return self.literal(fn, args, kw)
        if name.startswith('posixpath.') and name[10:] in (
                'join', 'split', 'splitext', 'splitdrive', 'basename', 'dirname', 'normcase', 'normpath',
                'commonprefix', 'commonpath', 'isabs', 'abspath'):
            return self.literal(getattr(posixpath, name[10:]), args, kw)
        if name.startswith('math.'):
            fn = getattr(math, name[5:], None)
            if callable(fn): return self.literal(fn, args, kw)
        decoded = self.decode(name, args, kw)
        if decoded is not MISSING: return decoded
        if name in ('numpy.core.multiarray._reconstruct', 'numpy._core.multiarray._reconstruct',
                    'numpy._core.numeric._frombuffer', 'numpy.core.numeric._frombuffer'):
            return Instance('numpy.ndarray', args)
        if name in ('builtins.object', 'types.SimpleNamespace'):
            return Instance(name, args, kw.copy())
        if name.endswith('.__init__'):
            if args and isinstance(args[0], Instance):
                cls = name[:-9]
                self.call(self.ctx.ref(cls), args[1:], kw)
            return None
        # Numerical reconstruction routines and model class state do not run
        # model methods. A reference remains symbolic if used as a callable
        # later, rather than being invoked by the scanner.
        return Instance(name, args) if is_class else Unknown('return from ' + name)

    @staticmethod
    def is_class(ref):
        name = ref.name
        simple = name.rsplit('.', 1)[-1]
        module = name.rsplit('.', 1)[0]
        if (module not in MODULE_EXPORTS and name not in CAPS and name not in NESTED and
                not simple.startswith('_') and (simple.endswith(('_matrix', '_array')) or
                module.startswith(('numpy', 'scipy', 'sklearn', 'pandas', 'torch')))):
            return True
        if module == 'datetime' and simple in ('date', 'datetime', 'time', 'timedelta', 'timezone'):
            return True
        return (simple[:1].isupper() or name.startswith('dynamic.') or name in {
            'builtins.object', 'builtins.type', 'builtins.bytes', 'builtins.bytearray',
            'builtins.str', 'builtins.int', 'builtins.float', 'builtins.bool', 'builtins.complex',
            'builtins.list', 'builtins.tuple', 'builtins.dict', 'builtins.set', 'builtins.frozenset',
            'builtins.range', 'builtins.slice', 'builtins.map', 'builtins.filter', 'builtins.zip',
            'builtins.enumerate', 'builtins.reversed', 'functools.partial', 'socket.socket',
            'operator.attrgetter', 'operator.itemgetter', 'operator.methodcaller',
            'builtins.staticmethod', 'builtins.classmethod',
            'numpy.dtype', 'numpy.ndarray', 'numpy.float64', 'numpy.float32', 'numpy.int64',
            'collections.defaultdict', 'collections.deque',
        })

    @staticmethod
    def enough_args(name, args, kw):
        signatures = {
            'builtins.eval': (('source', 'globals', 'locals'), 1, True),
            'builtins.exec': (('source', 'globals', 'locals'), 1, True),
            'builtins.compile': (('source', 'filename', 'mode', 'flags', 'dont_inherit', 'optimize'), 3, False),
            'os.system': (('command',), 1, True),
            'os.popen': (('cmd', 'mode', 'buffering'), 1, False),
            'os.execv': (('path', 'argv'), 2, True),
            'os.execve': (('path', 'argv', 'env'), 3, True),
            'os.execvp': (('file', 'args'), 2, False),
            'os.execvpe': (('file', 'args', 'env'), 3, False),
            'socket.socket.connect': (('self', 'address'), 2, True),
            'socket.socket.connect_ex': (('self', 'address'), 2, True),
        }
        if name in signatures:
            params, required, positional = signatures[name]
            if len(args) > len(params): return False
            extras = {'closure'} if name == 'builtins.exec' else {'_feature_version'} if name == 'builtins.compile' else set()
            if positional and set(kw) - extras: return False
            if not positional and set(kw) - set(params) - extras: return False
            if any(p in kw for p in params[:len(args)]): return False
            if not all(i < len(args) or p in kw for i, p in enumerate(params[:required])): return False
            if name in ('builtins.eval', 'builtins.exec'):
                source = args[0]
                if not isinstance(source, (str, bytes, Unknown)) and not (isinstance(source, Instance) and source.cls == 'marshal.code'):
                    return False
                if len(args) > 1 and args[1] is not None and not isinstance(args[1], (dict, Unknown)):
                    return False
            if name in ('os.system', 'os.popen'):
                cmd = args[0] if args else kw.get(params[0])
                if not isinstance(cmd, (str, bytes, Unknown)): return False
            return True
        if args or kw:
            return True
        return name in {'os.fork', 'os.forkpty', 'pty.fork', 'os.getcwd', 'os.listdir', 'os.scandir',
                        'tempfile.mkstemp', 'tempfile.mkdtemp', 'tempfile.TemporaryFile',
                        'tempfile.NamedTemporaryFile', 'tempfile.TemporaryDirectory',
                        'tempfile.SpooledTemporaryFile', 'timeit.timeit', 'timeit.repeat', 'timeit.Timer'}

    def bound(self, obj, name, args, kw):
        if name == '__call__': return self.call(obj, args, kw)
        if name == '__getattribute__': return self.attr(obj, args[0])
        if name == '__getattr__': return self.attr(obj, args[0])
        if name == '__setattr__':
            if isinstance(obj, (Instance, Module, Ref, Partial)):
                obj.attrs[args[0]] = args[1]
            return None
        if name == '__get__':
            if not args or args[0] is None: return obj
            return Partial(obj, (args[0],))
        if name in ('__getitem__', 'get'):
            return self.item(obj, args[0], args[1] if len(args) > 1 else (None if name == 'get' else MISSING))
        if name == '__setitem__':
            self.setitem(obj, *args); return None
        if name == '__setstate__':
            self.build(obj, args[0]); return None
        if isinstance(obj, Buffer):
            if name in ('read', 'read1', 'readline', 'getvalue', 'getbuffer'):
                if name in ('getvalue', 'getbuffer'): return obj.data
                n = args[0] if args else -1
                if name == 'readline':
                    end = obj.data.find(b'\n' if isinstance(obj.data, bytes) else '\n', obj.offset)
                    end = len(obj.data) if end < 0 else end + 1
                else: end = len(obj.data) if n is None or n < 0 else obj.offset + n
                result = obj.data[obj.offset:end]; obj.offset += len(result)
                return result
            if name == 'seek':
                offset = args[0]; whence = args[1] if len(args) > 1 else 0
                obj.offset = offset + (obj.offset if whence == 1 else len(obj.data) if whence == 2 else 0)
                return obj.offset
            if name == 'tell': return obj.offset
            if name == 'write':
                value = args[0]
                obj.data = obj.data[:obj.offset] + value + obj.data[obj.offset + len(value):]
                obj.offset += len(value)
                return len(value)
            if name in ('close', 'flush', '__enter__', '__exit__'): return obj
        if isinstance(obj, (dict, Modules)) or isinstance(obj, Adapter) and obj.kind == 'namespace':
            if name == 'update':
                values = self.to_dict(args[0]) if args else {}
                values.update(kw)
                for key, value in values.items(): self.setitem(obj, key, value)
                return None
            if name in ('items', 'values', 'keys', 'copy'):
                if isinstance(obj, Adapter):
                    module = obj.args[0]
                    values = self.namespace(module)
                elif isinstance(obj, Modules): values = dict(self.ctx.modules)
                else: values = obj
                return dict(values) if name == 'copy' else list(getattr(values, name)())
            if name == 'setdefault':
                value = self.item(obj, args[0], MISSING)
                if value is MISSING:
                    value = args[1] if len(args) > 1 else None
                    self.setitem(obj, args[0], value)
                return value
            if name == 'pop' and isinstance(obj, dict): return obj.pop(*args)
        if isinstance(obj, list):
            if name == 'append': obj.append(args[0]); return None
            if name == 'extend': obj.extend(self.consume(args[0])); return None
            if name == 'insert': obj.insert(*args); return None
            if name == 'pop': return obj.pop(*args)
            if name == 'clear': obj.clear(); return None
            if name == 'reverse': obj.reverse(); return None
            if name == 'copy': return obj[:]
            if name == 'sort':
                result = self.call(self.ctx.ref('builtins.sorted'), (obj,), kw)
                obj[:] = result; return None
        if isinstance(obj, (str, bytes, bytearray, tuple, list)):
            if name == '__add__': return self.literal(operator.add, (obj, args[0]))
            if name == '__mul__': return self.literal(operator.mul, (obj, args[0]))
            if name == '__iter__': return self.iterator(obj)
            if name == '__len__': return len(obj)
            if name in ('join',):
                return self.literal(obj.join, (self.consume(args[0]),))
            if name in ('encode', 'decode'):
                codec = args[0] if args else kw.get('encoding', 'utf-8')
                errors = args[1] if len(args) > 1 else kw.get('errors', 'strict')
                return self.text_codec(obj, codec, errors, encode=name == 'encode')
            if name in ('format', 'format_map'):
                # Formatting can traverse attacker-specified attributes. Do
                # not let Python's formatter see symbolic objects.
                if not all(self.primitive(v) for v in args) or not self.primitive(kw):
                    raise AnalysisLimit('symbolic string formatting')
            if name in ('replace', 'split', 'rsplit', 'strip', 'lstrip', 'rstrip', 'lower', 'upper',
                        'capitalize', 'title', 'swapcase', 'casefold', 'removeprefix', 'removesuffix',
                        'format', 'format_map', 'zfill', 'ljust', 'rjust', 'center', 'hex',
                        'startswith', 'endswith', 'find', 'rfind', 'index', 'count', 'translate',
                        'maketrans', 'partition', 'rpartition'):
                method = getattr(obj, name, None)
                if method is not None: return self.literal(method, args, kw)
        if isinstance(obj, Lazy) and name == '__next__': return self.next_value(obj)
        if isinstance(obj, Instance):
            cls = obj.cls
            if '_items' in obj.attrs and name in ('items', 'values', 'keys', 'copy', 'update', 'pop', 'setdefault'):
                return self.bound(obj.attrs['_items'], name, args, kw)
            if cls == 'timeit.Timer' and name in ('timeit', 'repeat', 'autorange'):
                setup = obj.attrs.get('_setup', MISSING)
                stmt = obj.attrs.get('_stmt', MISSING)
                timer = obj.attrs.get('_timer', MISSING)
                if setup is not MISSING and not isinstance(setup, (str, bytes)):
                    self.call(setup, ())
                if timer is not MISSING: self.call(timer, ())
                number = args[0] if args and name == 'timeit' else kw.get('number', 1000000)
                if number and stmt is not MISSING and not isinstance(stmt, (str, bytes)):
                    self.call(stmt, ())
                return 0.0
            if cls == 'logging.config.DictConfigurator' and name == 'configure':
                if obj.args: self.logging_config(obj.args[0])
                return None
            if cls in ('threading.Thread', 'threading.Timer') and name in ('start', 'run'):
                target = obj.attrs.get('_target')
                if target is not None:
                    self.call(target, tuple(obj.attrs.get('_args', ())), obj.attrs.get('_kwargs', {}))
                return None
            if cls.startswith('multiprocessing.') and cls.endswith('Process') and name == 'start':
                self.finding('process', 'start multiprocessing process')
                return None
            if name in ('submit', 'apply', 'apply_async', 'map', 'starmap', 'imap', 'map_async') and (
                'Executor' in cls or 'Pool' in cls):
                process = 'Process' in cls or cls == 'multiprocessing.pool.Pool'
                if process and name in ('submit', 'apply', 'apply_async'):
                    self.finding('process', 'start process executor')
                if name == 'submit': value = self.call(args[0], args[1:], kw)
                elif name.startswith('apply'):
                    value = self.call(args[0], tuple(args[1]) if len(args) > 1 else (), args[2] if len(args) > 2 else {})
                else:
                    it = Lazy('starmap' if name == 'starmap' else 'map', args[0], args[1:])
                    value = self.consume(it)
                    if process and value: self.finding('process', 'start process executor')
                return Instance('concurrent.futures.Future', (), {'_result': value})
            if name in ('result', 'get') and '_result' in obj.attrs: return obj.attrs['_result']
            if cls == 'pickle.Unpickler' and name == 'load':
                return self.nested('pickle.load', obj.args, {})
            if cls in ('zipfile.ZipFile', 'tarfile.TarFile') and '_buffer' in obj.attrs:
                import tarfile
                import zipfile
                data = obj.attrs['_buffer'].data
                if name in ('extract', 'extractall', 'write'):
                    self.finding('file-access', 'archive extraction or write')
                    return None
                if cls == 'zipfile.ZipFile':
                    with zipfile.ZipFile(io.BytesIO(data)) as archive:
                        if name == 'namelist': return archive.namelist()
                        if name in ('read', 'open'):
                            info = archive.getinfo(args[0])
                            if info.file_size > MAX_DATA: raise AnalysisLimit('zip member size')
                            raw = archive.read(info)
                            return Buffer(raw) if name == 'open' else raw
                else:
                    with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as archive:
                        if name == 'getnames': return archive.getnames()
                        if name == 'extractfile':
                            info = archive.getmember(args[0])
                            if info.size > MAX_DATA: raise AnalysisLimit('tar member size')
                            return Buffer(archive.extractfile(info).read())
            if cls in ('pathlib.Path', 'pathlib.PurePath'):
                if name in ('joinpath', '__truediv__', 'with_name', 'with_suffix'):
                    return Instance(cls, obj.args + args)
                if name in ('__str__', '__fspath__', 'as_posix'): return '/'.join(map(str, obj.args))
            if cls == 'xmlrpc.client.ServerProxy':
                self.finding('network', 'call XML-RPC method ' + name)
                return Unknown('RPC result')
            if cls == 'collections.defaultdict' and name == '__missing__':
                factory = obj.attrs.get('default_factory')
                if factory is not None: return self.call(factory, ())
            key = cls + '.' + name
            if key in CAPS:
                return self.call(self.ctx.ref(key), (obj,) + args, kw)
            if name == '__init__': return self.call(self.ctx.ref(cls), args, kw)
            if name == '__enter__': return obj
            if name in ('close', 'join', 'shutdown', '__exit__'): return None
            if cls in ('numpy.ndarray', 'numpy.dtype'):
                if '_values' in obj.attrs:
                    values = obj.attrs['_values']
                    if name == 'tolist': return values
                    if name == 'item': return values[args[0] if args else 0]
                return Unknown('numerical method result')
            return self.call(self.ctx.ref(key), (obj,) + args, kw)
        return Unknown('bound method ' + name)

    def to_dict(self, obj):
        if isinstance(obj, dict): return obj.copy()
        if isinstance(obj, Adapter) and obj.kind == 'namespace':
            return self.namespace(obj.args[0])
        if isinstance(obj, Instance) and '_items' in obj.attrs: return obj.attrs['_items'].copy()
        return dict(self.consume(obj))

    def namespace(self, module):
        values = {}
        prefix = module.name + '.'
        for name in MODULE_EXPORTS.get(module.name, ()):
            values[name] = self.attr(module, name)
        for full in CAPS.keys() | NESTED.keys() | MODULE_ATTRS.keys() | self.ctx.refs.keys():
            if full.startswith(prefix) and '.' not in full[len(prefix):]:
                attr = full[len(prefix):]
                if module.name not in MODULE_EXPORTS or attr in MODULE_EXPORTS[module.name] or attr in module.attrs:
                    values[attr] = self.attr(module, attr)
        for attr in ('__import__', 'getattr', 'setattr', 'vars', 'map', 'filter', 'list', 'tuple', 'dict') if module.name == 'builtins' else ():
            values[attr] = self.attr(module, attr)
        values.update(module.attrs)
        return values

    @staticmethod
    def primitive(value, seen=None):
        if type(value) in (str, bytes, bytearray, int, float, bool, complex, type(None), range, slice):
            return True
        if type(value) in (list, tuple, dict, set, frozenset):
            seen = set() if seen is None else seen
            if id(value) in seen: return False
            seen.add(id(value))
            vals = list(value.items()) if isinstance(value, dict) else value
            result = all(VM.primitive(v, seen) for v in vals)
            seen.remove(id(value))
            return result
        return False

    def literal(self, fn, args, kw=None):
        kw = {} if kw is None else kw
        if not self.primitive(args) or not self.primitive(kw):
            return Unknown('symbolic literal operation')
        # Bound allocations before performing amplification operations.
        if fn in (operator.mul, operator.imul) and len(args) == 2:
            a, b = args
            if isinstance(a, int): a, b = b, a
            if isinstance(a, (bytes, str, list, tuple, bytearray)) and isinstance(b, int):
                if len(a) * max(0, b) > MAX_DATA: raise AnalysisLimit('literal multiplication size')
        if fn in (bytes, bytearray) and args and isinstance(args[0], int) and args[0] > MAX_DATA:
            raise AnalysisLimit('byte allocation size')
        if fn in (pow, operator.lshift) and len(args) > 1 and isinstance(args[1], int) and args[1] > 100000:
            raise AnalysisLimit('integer computation size')
        try:
            return fn(*args, **kw)
        except (ValueError, TypeError, IndexError, KeyError, OverflowError, UnicodeError, ZeroDivisionError):
            raise LoadError('invalid literal operation')

    def iterator(self, value):
        if isinstance(value, Lazy): return value
        if isinstance(value, (tuple, list, str, bytes, bytearray, range, dict, set, frozenset)):
            if isinstance(value, (dict, set, frozenset)): value = list(value)
            return Lazy('seq', None, (value,))
        if isinstance(value, Instance):
            if '__iter__' in value.attrs: return self.call(value.attrs['__iter__'], (value,))
            if '_items_list' in value.attrs: return self.iterator(value.attrs['_items_list'])
            if '_items' in value.attrs: return self.iterator(value.attrs['_items'])
        raise LoadError('not iterable')

    def next_value(self, it):
        self.ctx.tick()
        if not isinstance(it, Lazy): raise LoadError('not an iterator')
        if it.state.get('exhausted'): raise StopIteration
        kind = it.kind
        if kind == 'seq':
            if it.index >= len(it.sources[0]): raise StopIteration
            value = it.sources[0][it.index]; it.index += 1; return value
        if kind == 'sentinel':
            value = self.call(it.func, ())
            if value == it.sources[0]: raise StopIteration
            return value
        if kind == 'accumulate':
            if not it.index:
                it.sources = (self.iterator(it.sources[0]),)
                it.index = 1
                initial = it.state.pop('initial', MISSING)
                if initial is not MISSING and initial is not None:
                    it.state['total'] = initial
                    return initial
            value = self.next_value(it.sources[0])
            if 'total' in it.state:
                value = (self.call(it.func, (it.state['total'], value)) if it.func is not None
                         else self.literal(operator.add, (it.state['total'], value)))
            it.state['total'] = value
            return value
        if kind == 'groupby':
            if not it.index:
                it.sources = (self.iterator(it.sources[0]),)
                it.index = 1
            if 'group' in it.state:
                group = it.state.pop('group')
                self.consume(group)
            if 'pending' in it.state:
                key, value = it.state.pop('pending')
            else:
                value = self.next_value(it.sources[0])
                key = self.call(it.func, (value,)) if it.func is not None else value
            group = Lazy('group', None, (it,), state={'key': key, 'first': value})
            it.state['group'] = group
            return (key, group)
        if kind == 'group':
            if 'first' in it.state: return it.state.pop('first')
            parent = it.sources[0]
            value = self.next_value(parent.sources[0])
            key = self.call(parent.func, (value,)) if parent.func is not None else value
            if key != it.state['key']:
                parent.state['pending'] = (key, value)
                it.state['exhausted'] = True
                raise StopIteration
            return value
        if kind == 'repeat':
            if len(it.sources) > 1 and it.index >= it.sources[1]: raise StopIteration
            it.index += 1; return it.sources[0]
        if kind in ('map', 'filter', 'filterfalse', 'starmap', 'takewhile', 'dropwhile', 'zip',
                    'iter', 'enumerate'):
            if not it.index:
                sources = it.sources
                if kind == 'enumerate': sources = sources[:1]
                it.sources = tuple(self.iterator(s) for s in sources)
                it.index = 1
            if kind in ('map', 'zip'):
                values = []
                for source in it.sources:
                    values.append(self.next_value(source))
                values = tuple(values)
                return self.call(it.func, values) if kind == 'map' else values
            value = self.next_value(it.sources[0])
            if kind == 'iter': return value
            if kind == 'enumerate':
                count = it.state.get('count', 0)
                it.state['count'] = count + 1
                return (count, value)
            if kind == 'starmap': return self.call(it.func, tuple(value))
            while True:
                test = self.call(it.func, (value,)) if it.func is not None else bool(value)
                if kind == 'filter' and test or kind == 'filterfalse' and not test: return value
                if kind == 'takewhile':
                    if test: return value
                    it.state['exhausted'] = True
                    raise StopIteration
                if kind == 'dropwhile' and not test:
                    it.kind = 'iter'; return value
                value = self.next_value(it.sources[0])
        if kind in ('chain', 'from_iterable'):
            if kind == 'from_iterable':
                if not it.index:
                    it.sources = (self.iterator(it.sources[0]),)
                    it.index = 1
                while True:
                    if 'inner' not in it.state:
                        it.state['inner'] = self.iterator(self.next_value(it.sources[0]))
                    try: return self.next_value(it.state['inner'])
                    except StopIteration: it.state.pop('inner')
            while it.index < len(it.sources):
                source = it.sources[it.index]
                if not isinstance(source, Lazy):
                    source = self.iterator(source)
                    it.sources = it.sources[:it.index] + (source,) + it.sources[it.index + 1:]
                try: return self.next_value(source)
                except StopIteration: it.index += 1
            raise StopIteration
        if kind in ('reversed', 'cycle'):
            if it.index == 0:
                values = self.consume(it.sources[0])
                if kind == 'reversed': values.reverse()
                it.sources = (values,)
                if kind == 'reversed': it.kind = 'seq'; return self.next_value(it)
            values = it.sources[0]
            if not values: raise StopIteration
            value = values[it.index % len(values)]; it.index += 1; return value
        if kind == 'islice':
            if it.index == 0:
                source, *bounds = it.sources
                if len(bounds) == 1: start, stop, step = 0, bounds[0], 1
                else: start, stop, step = bounds[0] or 0, bounds[1], (bounds[2] if len(bounds) > 2 else 1) or 1
                it.sources = (self.iterator(source), start, stop, step, 0)
                it.index = 1
            source, start, stop, step, current = it.sources
            if stop is not None and start >= stop: raise StopIteration
            value = None
            while current <= start:
                value = self.next_value(source); current += 1
            it.sources = (source, start + step, stop, step, current)
            return value
        raise AnalysisLimit('unmodelled lazy iterator ' + kind)

    def consume(self, value):
        it = self.iterator(value)
        values = []
        while len(values) <= MAX_ITEMS:
            try: values.append(self.next_value(it))
            except StopIteration: return values
        raise AnalysisLimit('iterator consumption limit')

    def text_codec(self, data, encoding, errors='strict', *, encode=False):
        if not isinstance(encoding, str) or not isinstance(errors, str):
            raise LoadError('invalid codec')
        normal = encoding.lower().replace('-', '_')
        if normal in ('base64', 'base64_codec', 'hex', 'hex_codec', 'zlib', 'zlib_codec', 'bz2', 'bz2_codec'):
            if normal.startswith('base64'): return base64.b64encode(data) if encode else base64.b64decode(data)
            if normal.startswith('hex'): return binascii.hexlify(data) if encode else binascii.unhexlify(data)
            if normal.startswith('zlib'): return zlib.compress(data) if encode else self.inflate(data)
            import bz2
            if encode: return bz2.compress(data)
            result = bz2.BZ2Decompressor().decompress(data, max_length=MAX_DATA + 1)
            if len(result) > MAX_DATA: raise AnalysisLimit('decoded byte limit')
            return result
        allowed = ('utf8', 'utf_8', 'utf_8_sig', 'utf16', 'utf_16', 'utf_16_le', 'utf_16_be',
                   'utf32', 'utf_32', 'utf_32_le', 'utf_32_be', 'latin1', 'latin_1',
                   'iso8859_1', 'iso_8859_1', 'ascii', 'us_ascii', 'unicode_escape',
                   'raw_unicode_escape', 'rot13', 'rot_13', 'cp1252')
        if normal not in allowed: raise AnalysisLimit('unmodelled codec ' + encoding)
        try:
            return codecs.encode(data, encoding, errors) if encode else codecs.decode(data, encoding, errors)
        except (ValueError, TypeError, UnicodeError): raise LoadError('invalid encoded data')

    @staticmethod
    def inflate(data, wbits=zlib.MAX_WBITS):
        obj = zlib.decompressobj(wbits)
        result = obj.decompress(data, MAX_DATA + 1)
        if len(result) > MAX_DATA or not obj.eof: raise AnalysisLimit('decoded byte limit or truncated zlib')
        return result

    def decode(self, name, args, kw):
        if name in ('_codecs.encode', 'codecs.encode', '_codecs.decode', 'codecs.decode'):
            return self.text_codec(args[0], args[1] if len(args) > 1 else 'utf-8',
                                   args[2] if len(args) > 2 else 'strict', encode=name.endswith('.encode'))
        funcs = {
            'base64.b64decode': base64.b64decode, 'base64.standard_b64decode': base64.standard_b64decode,
            'base64.urlsafe_b64decode': base64.urlsafe_b64decode, 'base64.decodebytes': base64.decodebytes,
            'base64.b85decode': base64.b85decode, 'base64.a85decode': base64.a85decode,
            'base64.b16decode': base64.b16decode, 'base64.b32decode': base64.b32decode,
            'base64.b64encode': base64.b64encode, 'base64.urlsafe_b64encode': base64.urlsafe_b64encode,
            'binascii.a2b_base64': binascii.a2b_base64, 'binascii.unhexlify': binascii.unhexlify,
            'binascii.a2b_hex': binascii.unhexlify,
            'builtins.bytes.fromhex': bytes.fromhex, 'builtins.bytearray.fromhex': bytearray.fromhex,
            'struct.pack': struct.pack, 'struct.unpack': struct.unpack,
        }
        if name in funcs: return self.literal(funcs[name], args, kw)
        if name == 'zlib.decompress':
            return self.inflate(args[0], args[1] if len(args) > 1 else kw.get('wbits', zlib.MAX_WBITS))
        if name in ('gzip.decompress', 'bz2.decompress', 'lzma.decompress'):
            from modelgate.dispatch import _decompress
            return b''.join(_decompress(args[0], name.split('.')[0].replace('lzma', 'xz')))
        if name == 'zlib.compress': return self.literal(zlib.compress, args, kw)
        if name == 'json.loads':
            import json
            return self.literal(json.loads, args, kw)
        if name == 'ast.literal_eval':
            import ast
            return self.literal(ast.literal_eval, args, kw)
        if name in ('builtins.dict.get', 'builtins.dict.__getitem__', 'builtins.dict.update',
                    'builtins.list.append', 'builtins.list.extend', 'builtins.list.sort',
                    'builtins.str.encode', 'builtins.bytes.decode', 'builtins.bytearray.decode',
                    'io.BytesIO.read', 'io.BytesIO.getvalue'):
            return self.bound(args[0], name.rsplit('.', 1)[-1], args[1:], kw)
        return MISSING

    def nested(self, name, args, kw):
        if not args: raise LoadError('loader needs input')
        value = args[0]
        if isinstance(value, Unknown): raise AnalysisLimit('cannot resolve nested loader input')
        kind = NESTED[name]
        buffer = value if isinstance(value, Buffer) else None
        if kind == 'pickle-one':
            if not isinstance(value, (bytes, bytearray)): raise LoadError('pickle.loads requires bytes')
            data = bytes(value)
        elif kind == 'pickle-file':
            if buffer is None: raise LoadError('pickle.load requires a file object')
            data = buffer.data[buffer.offset:]
        elif kind == 'torch-bytes':
            if not isinstance(value, (bytes, bytearray)): raise LoadError('torch storage requires bytes')
            data = bytes(value)
        elif buffer is not None:
            data = buffer.data[buffer.offset:]
        elif isinstance(value, (str, bytes, bytearray, Instance)):
            self.finding('file-access', 'nested loader ' + name + ' reads a path')
            return Unknown('loaded file')
        else: raise LoadError('invalid nested loader input')
        if not isinstance(data, (bytes, bytearray)): raise LoadError('binary loader needs bytes')
        data = bytes(data)
        nested_location = f'{self.location}@{self.pos}!{name}'
        if kind in ('pickle-one', 'pickle-file'):
            result, used = interpret(data, nested_location, mode='one', context=self.ctx, strict=True)
            if buffer is not None: buffer.offset += used
            return result
        if kind in ('torch', 'torch-bytes'):
            if data.startswith(b'PK\x03\x04'):
                import zipfile
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    names = archive.namelist()
                    candidates = [n for n in names if n == 'data.pkl' or n.endswith('/data.pkl')]
                    if not candidates: raise LoadError('torch zip has no data.pkl')
                    info = archive.getinfo(candidates[-1])
                    if info.file_size > MAX_DATA: raise AnalysisLimit('torch pickle size')
                    payload = archive.read(info)
                result, _ = interpret(payload, nested_location, mode='one', persistent=True,
                                      context=self.ctx, strict=True)
                return result
            result, used = interpret(data, nested_location, mode='torch', persistent=True, context=self.ctx, strict=True)
            if buffer is not None: buffer.offset += used
            return result
        if kind in ('joblib', 'pandas'):
            if data.startswith((b'\x1f\x8b', b'BZh', b'\xfd7zXZ\x00')):
                from modelgate.dispatch import _decompress
                compressor = 'gzip' if data.startswith(b'\x1f\x8b') else 'bz2' if data.startswith(b'BZh') else 'xz'
                data = b''.join(_decompress(data, compressor))
            elif data.startswith(b'x'):
                data = self.inflate(data)
            result, used = interpret(data, nested_location, mode='one', context=self.ctx, strict=True)
            if buffer is not None: buffer.offset += used
            return result
        if kind in ('numpy', 'numpy-array'):
            allow = args[2] if kind == 'numpy' and len(args) > 2 else args[1] if kind == 'numpy-array' and len(args) > 1 else kw.get('allow_pickle', False)
            return self.numpy_data(data, allow, nested_location, array_only=kind == 'numpy-array')
        from modelgate.dispatch import scan_bytes
        findings = scan_bytes(data, nested_location, context=self.ctx)
        for f in findings:
            if f not in self.ctx.findings: self.ctx.findings.append(f)
        return Unknown('nested loader result')

    def numpy_data(self, data, allow_pickle, location, *, array_only=False):
        if data.startswith(b'\x93NUMPY'):
            from modelgate.dispatch import npy_info
            start, hasobject = npy_info(data)
            if not hasobject: return Instance('numpy.ndarray')
            if not allow_pickle: raise LoadError('NumPy object arrays require allow_pickle')
            result, _ = interpret(data[start:], location, mode='one', context=self.ctx, strict=True)
            return result
        if array_only: raise LoadError('not a NumPy array')
        if data.startswith((b'PK\x03\x04', b'PK\x05\x06')):
            # NpzFile reads individual members lazily. Returning one does not
            # deserialize any array; __getitem__ above performs that step.
            return Instance('numpy.NpzFile', (), {'_data': data, '_allow_pickle': allow_pickle})
        if not allow_pickle: raise LoadError('NumPy pickle fallback is disabled')
        result, _ = interpret(data, location, mode='one', context=self.ctx, strict=True)
        return result

    def logging_config(self, config):
        if not isinstance(config, dict): raise LoadError('invalid logging configuration')
        for section in ('formatters', 'filters', 'handlers'):
            for entry in config.get(section, {}).values():
                if not isinstance(entry, dict): continue
                factory = entry.get('()')
                if isinstance(factory, str): factory = self.resolve_text(factory)
                kwargs = {k: v for k, v in entry.items() if k not in ('()', 'class', 'level', 'formatter', 'filters', '.')}
                if factory is None and section == 'handlers':
                    factory = entry.get('class')
                    if isinstance(factory, str): factory = self.resolve_text(factory)
                if factory is not None: self.call(factory, (), kwargs)

    def resolve_text(self, text):
        module, _, name = text.rpartition('.')
        return self.attr(self.ctx.module(module), name)


def interpret(data, location, *, mode='sequence', persistent=False, context=None, strict=False):
    ctx = Context() if context is None else context
    stream = io.BytesIO(data)
    vm = VM(ctx, location, persistent)
    result = None
    stream_count = 0
    torch = mode == 'torch'
    torch_result = None
    ctx.depth += 1
    if ctx.depth > 40:
        ctx.depth -= 1
        raise AnalysisLimit('nested deserialization limit')
    try:
        while stream.tell() < len(data):
            vm.location = f'{location}:pickle[{stream_count}]'
            if torch:
                # Legacy torch uses a persistent-id handler for the object
                # pickle only, not its magic/protocol/header/storage-key data.
                vm.persistent = stream_count == 3
            result = vm.run(stream)
            stream_count += 1
            if stream_count == 1 and result == TORCH_MAGIC:
                torch = True
            if mode == 'one': break
            if mode == 'torch' and stream_count == 1 and result != TORCH_MAGIC:
                raise LoadError('invalid torch checkpoint magic')
            if torch and stream_count == 4: torch_result = result
            if torch and stream_count == 5:
                result = torch_result
                break  # remaining bytes are storages, never pickle code
            if stream_count > MAX_ITEMS: raise AnalysisLimit('too many pickle streams')
        if not stream_count:
            raise LoadError('empty pickle')
        if mode == 'torch' and stream_count < 5:
            raise LoadError('truncated torch checkpoint')
    except (LoadError, IndexError, KeyError, TypeError, ValueError, AttributeError, StopIteration, struct.error) as exc:
        # This is a statically known loader failure. Later opcodes/streams do
        # not run; capabilities already reached still count.
        if strict:
            raise LoadError(str(exc)) from None
    finally:
        ctx.depth -= 1
    return result, stream.tell()


def scan_pickle(data: bytes, location: str, *, mode='sequence', persistent=False, context=None) -> list[Finding]:
    ctx = Context() if context is None else context
    first = len(ctx.findings)
    try:
        interpret(data, location, mode=mode, persistent=persistent, context=ctx)
    except (AnalysisLimit, RecursionError, MemoryError) as exc:
        if not ctx.findings[first:]:
            ctx.findings.append(Finding('unsupported', location, str(exc) or type(exc).__name__))
    return ctx.findings[first:]
