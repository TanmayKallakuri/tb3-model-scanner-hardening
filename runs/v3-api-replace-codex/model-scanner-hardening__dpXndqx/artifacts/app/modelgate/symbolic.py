"""An inert value model for CPython unpickling.

There is deliberately no import-by-name, eval, unpickling, or invocation of a
value supplied by an artifact. Calls below implement small, explicit semantic
models. Native operations are limited to concrete built-in data values.
"""
from __future__ import annotations

import base64
import binascii
import bz2
import codecs
import io
import lzma
import json
import os
import operator
import posixpath
import struct
import sys
import zlib
from dataclasses import dataclass, field

from modelgate.limits import AnalysisLimit, Budget, MAX_BYTES, MAX_ITEMS
from modelgate.report import Finding


class CapabilityFound(Exception):
    """A concrete finding fixes the verdict; further interpretation is needless."""


class LoadError(Exception):
    """The standard loader stops here, without executing later opcodes."""


@dataclass(eq=False)
class Ref:
    name: str
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Module:
    name: str
    attrs: dict = field(default_factory=dict)
    imported: bool = False


@dataclass(eq=False)
class Instance:
    name: str
    args: tuple = ()
    attrs: dict = field(default_factory=dict)
    state: object = None
    data: object = None


@dataclass(eq=False)
class Partial:
    func: object
    args: tuple = ()
    keywords: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)
    has_dict: bool = False


@dataclass(eq=False)
class Accessor:
    name: str
    args: tuple
    keywords: dict = field(default_factory=dict)


@dataclass(eq=False)
class Wrapped:
    func: object
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Namespace:
    target: object


@dataclass(eq=False)
class DictView:
    target: object
    kind: str


@dataclass(eq=False)
class Memory:
    data: object = b''
    pos: int = 0
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Lazy:
    name: str
    args: tuple
    index: int = 0
    sources: list | None = None
    done: bool = False
    keywords: dict = field(default_factory=dict)


@dataclass(eq=False)
class TeeState:
    source: object
    cache: list = field(default_factory=list)
    iterator: object = None
    done: bool = False


@dataclass(eq=False)
class DynamicClass:
    name: str
    bases: tuple
    attrs: dict


@dataclass(eq=False)
class DynamicInstance:
    cls: DynamicClass
    attrs: dict = field(default_factory=dict)
    value: object = None


@dataclass(eq=False)
class Bound:
    target: object
    name: str


@dataclass(eq=False)
class Unknown:
    reason: str = 'opaque model value'


UNKNOWN = Unknown()
END = object()
DELETED = object()
# Frozen descriptions of trusted Python namespaces, generated without loading
# artifacts. These resolve aliases and missing attributes without importing a
# module named by an upload (which would itself be unsafe).
with open(os.path.join(os.path.dirname(__file__), 'stdlib.json'), encoding='utf-8') as _catalog_file:
    CATALOG = json.load(_catalog_file)


def normal_ref(name):
    for old, new in (
        ('concurrent.futures.thread.ThreadPoolExecutor', 'concurrent.futures.ThreadPoolExecutor'),
        ('concurrent.futures.process.ProcessPoolExecutor', 'concurrent.futures.ProcessPoolExecutor'),
        ('pathlib.PosixPath.', 'pathlib.Path.'), ('pathlib.WindowsPath.', 'pathlib.Path.'),
        ('socket.SocketType.', 'socket.socket.'),
    ):
        if name.startswith(old): name = new + name[len(old):]
    return name


CATALOG['classes'] = {normal_ref(k): v for k, v in CATALOG['classes'].items()}
CATALOG['functions'] = {normal_ref(k): v for k, v in CATALOG['functions'].items()}
ALIASES = {'posix': 'os', 'nt': 'os', '_pickle': 'pickle', 'cPickle': 'pickle',
           '__builtin__': 'builtins', 'copy_reg': 'copyreg', '_io': 'io',
           '_operator': 'operator', '_functools': 'functools', '_socket': 'socket'}
# These are references to modules, not arbitrary dotted callable names.
MODULE_LINKS = {
    'shutil': {'os', 'sys', 'stat', 'collections', 'functools'},
    'subprocess': {'os', 'sys', 'io', 'signal', 'contextlib'},
    'pathlib': {'os', 'sys', 'io', 'fnmatch'},
    'pickle': {'sys', 'io', 're', 'codecs'},
    'copy': {'types', 'copyreg'},
    'pkgutil': {'os', 'sys', 'importlib'},
    'importlib': {'sys'},
    'importlib.util': {'sys', 'types'},
    'threading': {'os', 'sys', 'functools'},
    'concurrent.futures.thread': {'os', 'threading', 'weakref'},
    'multiprocessing': {'os', 'sys', 'context'},
    'tempfile': {'os', 'io', 'shutil'},
    'pydoc': {'os', 'sys', 'subprocess', 'inspect'},
    'inspect': {'os', 'sys', 'types', 'functools'},
    'runpy': {'os', 'sys', 'io', 'types', 'importlib'},
    'logging': {'os', 'sys', 'io', 'threading'},
    'logging.config': {'os', 'sys', 'logging', 'types'},
    'logging.handlers': {'os', 'io', 'socket', 'pickle', 'struct'},
    'urllib.request': {'os', 'sys', 'socket', 'io', 'base64', 'http'},
    'http.client': {'os', 'socket', 'io', 'ssl'},
    'ftplib': {'os', 'sys', 'socket'},
    'smtplib': {'socket', 'io', 'sys'},
    'socket': {'os', 'sys', 'io'},
    'ssl': {'os', 'sys', 'socket'},
    'os': {'sys', 'stat'},
    'numpy': {'os', 'sys'},
    'torch': {'os', 'sys', 'pickle', 'types'},
    'torch.storage': {'io', 'torch'},
    'torch.serialization': {'os', 'sys', 'io', 'pickle', 'torch'},
}
MODULE_RENAMES = {('os', 'path'): 'posixpath', ('threading', '_os'): 'os',
                  ('tempfile', '_os'): 'os', ('tempfile', '_io'): 'io',
                  ('importlib', '_bootstrap'): 'importlib._bootstrap',
                  ('builtins', '__builtins__'): 'builtins'}
MODULE_NAMES = set(MODULE_LINKS) | set(ALIASES.values()) | {
    'types', 'sys', 'io', 'operator', 'functools', 'collections', 'collections.abc',
    'base64', 'zlib', 'gzip', 'bz2', 'lzma', 'binascii', 'marshal', 'timeit',
    'code', 'bdb', 'pdb', 'pty', 'webbrowser', 'xmlrpc.client', 'urllib',
    'urllib.request', 'urllib.parse', 'http', 'http.client', 'concurrent',
    'concurrent.futures', 'multiprocessing.context', 'numpy.lib',
    'numpy.lib.format', 'numpy.core', 'numpy._core', 'numpy.core.multiarray',
    'numpy._core.multiarray', 'numpy.lib.npyio', 'numpy.lib._npyio_impl',
    'joblib', 'pandas', 'json', 'ast', 'codecs', '_codecs', 'builtins',
    'importlib', 'importlib.util', 'importlib.machinery', 'posixpath', 'ntpath',
}
CLASS_NAMES = set('''
subprocess.Popen pathlib.Path pathlib.PosixPath pathlib.WindowsPath
pathlib.PurePath pathlib.PurePosixPath pathlib.PureWindowsPath
socket.socket socket.SocketType http.client.HTTPConnection http.client.HTTPSConnection
ftplib.FTP ftplib.FTP_TLS smtplib.SMTP smtplib.SMTP_SSL smtplib.LMTP
xmlrpc.client.ServerProxy xmlrpc.client.Transport xmlrpc.client.SafeTransport
io.FileIO io.TextIOWrapper io.BufferedReader io.BufferedWriter io.BufferedRandom
io.BytesIO io.StringIO functools.partial functools.partialmethod
threading.Thread threading.Timer multiprocessing.Process multiprocessing.context.Process
concurrent.futures.ThreadPoolExecutor concurrent.futures.ProcessPoolExecutor
pickle.Unpickler pickle.Pickler types.FunctionType types.CodeType types.MethodType
code.InteractiveInterpreter code.InteractiveConsole bdb.Bdb pdb.Pdb
timeit.Timer builtins.type builtins.object builtins.bytes builtins.bytearray
builtins.str builtins.int builtins.float builtins.bool builtins.list builtins.tuple
builtins.set builtins.frozenset builtins.dict builtins.range builtins.slice
builtins.map builtins.filter builtins.zip builtins.enumerate builtins.reversed
collections.OrderedDict collections.defaultdict collections.deque
numpy.ndarray numpy.dtype
'''.split())

CODE_CALLS = set('''
builtins.eval builtins.exec builtins.compile builtins.execfile
runpy.run_path runpy.run_module runpy._run_code runpy._run_module_code
runpy._run_module_as_main timeit.timeit timeit.repeat timeit.Timer
code.InteractiveInterpreter.runsource code.InteractiveInterpreter.runcode
code.InteractiveConsole.runsource code.InteractiveConsole.runcode
code.InteractiveConsole.push bdb.Bdb.run bdb.Bdb.runctx bdb.Bdb.runeval
pdb.run pdb.runctx pdb.runeval pdb.Pdb.run pdb.Pdb.runctx pdb.Pdb.runeval
numpy.testing._private.utils.runstring ctypes.CFUNCTYPE ctypes.PYFUNCTYPE
ctypes.CDLL ctypes.PyDLL ctypes.cdll.LoadLibrary ctypes.pydll.LoadLibrary
imp.load_source imp.load_module imp.load_compiled
importlib.machinery.SourceFileLoader.exec_module
importlib.machinery.SourceFileLoader.load_module
importlib.machinery.SourcelessFileLoader.exec_module
importlib.machinery.SourcelessFileLoader.load_module
'''.split())
PROCESS_CALLS = set('''
os.system os.popen os.fork os.forkpty os.posix_spawn os.posix_spawnp
subprocess.Popen subprocess.call subprocess.run subprocess.check_call
subprocess.check_output subprocess.getoutput subprocess.getstatusoutput
subprocess._fork_exec pty.spawn pty.fork pydoc.pipepager pydoc.tempfilepager
asyncio.create_subprocess_exec asyncio.create_subprocess_shell
multiprocessing.Pool multiprocessing.pool.Pool multiprocessing.context.Pool
'''.split())
FILE_CALLS = set('''
builtins.input builtins.print gzip.open gzip.GzipFile bz2.open bz2.BZ2File
lzma.open lzma.LZMAFile zipfile.ZipFile tarfile.open tarfile.TarFile.open
tarfile.TarFile.add tarfile.TarFile.extract tarfile.TarFile.extractall
zipfile.ZipFile.extract zipfile.ZipFile.extractall zipfile.ZipFile.write
shutil.which shutil.disk_usage os.statvfs os.fstatvfs ssl.create_default_context
ssl.SSLContext.load_verify_locations ssl.SSLContext.load_cert_chain
ssl.SSLContext.set_default_verify_paths
builtins.open io.open io.open_code io.FileIO
os.open os.read os.write os.fdopen os.remove os.unlink os.rename os.replace
os.mkdir os.makedirs os.rmdir os.removedirs os.listdir os.scandir os.walk os.fwalk
os.stat os.lstat os.chmod os.chown os.lchown os.truncate os.utime os.link
os.symlink os.readlink os.access os.chroot os.chdir os.mkfifo os.mknod
shutil.copy shutil.copy2 shutil.copyfile shutil.copyfileobj shutil.copymode
shutil.copystat shutil.copytree shutil.move shutil.rmtree shutil.chown
shutil.make_archive shutil.unpack_archive
pathlib.Path.open pathlib.Path.read_text pathlib.Path.read_bytes
pathlib.Path.write_text pathlib.Path.write_bytes pathlib.Path.unlink
pathlib.Path.rename pathlib.Path.replace pathlib.Path.mkdir pathlib.Path.rmdir
pathlib.Path.touch pathlib.Path.iterdir pathlib.Path.glob pathlib.Path.rglob
pathlib.Path.stat pathlib.Path.lstat pathlib.Path.chmod pathlib.Path.lchmod
pathlib.Path.symlink_to pathlib.Path.hardlink_to pathlib.Path.link_to
pathlib.Path.readlink pathlib.Path.exists pathlib.Path.is_file pathlib.Path.is_dir
pathlib.Path.resolve pathlib.Path.samefile pathlib.Path.owner pathlib.Path.group
os.path.exists os.path.isfile os.path.isdir os.path.islink os.path.getsize
os.path.getmtime os.path.getatime os.path.getctime os.path.samefile
posixpath.exists posixpath.isfile posixpath.isdir posixpath.islink posixpath.getsize
posixpath.getmtime posixpath.getatime posixpath.getctime posixpath.samefile
ntpath.exists ntpath.isfile ntpath.isdir ntpath.islink ntpath.getsize
ntpath.getmtime ntpath.getatime ntpath.getctime ntpath.samefile
sqlite3.connect _sqlite3.connect shelve.open dbm.open dbm.gnu.open dbm.ndbm.open
fileinput.input linecache.getline linecache.getlines tokenize.open
numpy.fromfile numpy.memmap numpy.savetxt numpy.save numpy.savez numpy.savez_compressed
pandas.read_csv pandas.read_json pandas.read_excel pandas.read_parquet pandas.read_feather
logging.FileHandler logging.handlers.RotatingFileHandler logging.handlers.TimedRotatingFileHandler
logging.handlers.WatchedFileHandler logging.basicConfig
'''.split())
NETWORK_CALLS = set('''
socket.create_connection socket.create_server socket.socket.connect socket.socket.connect_ex
socket.socket.bind socket.socket.listen socket.socket.send socket.socket.sendall
socket.socket.sendto socket.socket.sendmsg socket.getaddrinfo socket.gethostbyname
socket.gethostbyname_ex socket.gethostbyaddr socket.getnameinfo
urllib.request.urlopen urllib.request.urlretrieve urllib.request.OpenerDirector.open
http.client.HTTPConnection.connect http.client.HTTPConnection.request
http.client.HTTPConnection.endheaders http.client.HTTPConnection.send
http.client.HTTPSConnection.connect http.client.HTTPSConnection.request
http.client.HTTPSConnection.endheaders http.client.HTTPSConnection.send
ftplib.FTP.connect ftplib.FTP_TLS.connect ftplib.FTP.login ftplib.FTP.retrbinary
smtplib.SMTP.connect smtplib.SMTP_SSL.connect smtplib.LMTP.connect
smtplib.SMTP.sendmail smtplib.SMTP.send_message smtplib.SMTP.login
ssl.get_server_certificate webbrowser.open webbrowser.open_new webbrowser.open_new_tab
requests.get requests.post requests.put requests.delete requests.request
requests.sessions.Session.request requests.Session.request
xmlrpc.client._Method.__call__
'''.split())
NESTED_CALLS = set('''
pickle.loads pickle.load torch.load torch.serialization.load torch.storage._load_from_bytes
numpy.load numpy.lib.npyio.load numpy.lib._npyio_impl.load numpy.lib.format.read_array
joblib.load joblib.numpy_pickle.load pandas.read_pickle
'''.split())
PURE_METHODS = set('''
encode decode fromhex hex join split rsplit splitlines partition rpartition
replace strip lstrip rstrip removeprefix removesuffix lower upper casefold title
capitalize swapcase translate maketrans ljust rjust center zfill expandtabs
startswith endswith find rfind index rindex count isalpha isalnum isascii isdecimal
isdigit isidentifier islower isnumeric isprintable isspace istitle isupper
format format_map __add__ __radd__ __mul__ __rmul__ __getitem__ __len__
__str__ __repr__ __bytes__ __int__ __float__ __bool__
'''.split())

# Opaque numeric/model reducers have no Python execution or I/O semantics.
# Everything else without an explicit model is fail-closed at its call site.
PURE_OPAQUE_PREFIXES = (
    'math.', 'cmath.', 'decimal.', '_decimal.', 'fractions.', 'datetime.', '_datetime.',
    're.', '_sre.', 'sre_parse.', 'sre_compile.', 'array.',
    'numpy._core.', 'numpy.core.', 'numpy.linalg.', 'numpy.fft.', 'numpy.random.',
    'numpy.ma.', 'numpy.polynomial.',
    'scipy.sparse.', 'scipy.linalg.', 'scipy.special.', 'scipy.spatial.',
    'pandas._libs.', 'pandas.core.indexes.', 'pandas.core.arrays.',
    'torch._utils._rebuild', 'torch._tensor._rebuild', 'torch._tensor._convert',
)
PURE_OPAQUE_NAMES = set('''
numpy.dtype numpy.ndarray numpy.array numpy.asarray numpy.asanyarray numpy.matrix
numpy.log1p numpy.expm1 numpy.add numpy.multiply numpy.subtract numpy.divide
numpy.sum numpy.mean numpy.zeros numpy.ones numpy.empty numpy.arange numpy.reshape
numpy.float16 numpy.float32 numpy.float64 numpy.int8 numpy.int16 numpy.int32 numpy.int64
numpy.uint8 numpy.uint16 numpy.uint32 numpy.uint64 numpy.bool_ numpy.str_ numpy.bytes_
collections.Counter collections.ChainMap collections.UserDict collections.UserList
collections.UserString types.SimpleNamespace ssl.SSLContext
code.InteractiveInterpreter code.InteractiveConsole bdb.Bdb pdb.Pdb
urllib.request.Request urllib.request.OpenerDirector urllib.request.ProxyHandler
urllib.request.HTTPHandler urllib.request.HTTPSHandler
logging.StreamHandler logging.Formatter logging.Filter logging.NullHandler
logging.handlers.SocketHandler logging.handlers.DatagramHandler
logging.handlers.SMTPHandler logging.handlers.HTTPHandler
joblib.numpy_pickle.NumpyArrayWrapper
'''.split())


def concrete(value, seen=None):
    """Only exact builtin data may cross into a real Python operation.

    Visit shared containers once and bound depth; a memoized data DAG must
    not turn this safety check into exponential work.
    """
    pending = [(value, 0)]
    visited = set()
    while pending:
        obj, depth = pending.pop()
        if type(obj) in (str, bytes, bytearray, int, float, bool, complex, type(None)):
            continue
        if type(obj) is slice:
            pending.extend((v, depth + 1) for v in (obj.start, obj.stop, obj.step))
            continue
        if type(obj) not in (tuple, list, dict, set, frozenset): return False
        if id(obj) in visited: continue
        visited.add(id(obj))
        if depth > 100 or len(visited) > MAX_ITEMS or len(obj) > MAX_ITEMS: return False
        vals = list(obj.keys()) + list(obj.values()) if type(obj) is dict else obj
        pending.extend((v, depth + 1) for v in vals)
    return True


def class_name(name):
    if name in CLASS_NAMES:
        return True
    tail = name.rsplit('.', 1)[-1]
    return bool(tail and tail[0].isupper() and not tail.isupper())


class World:
    def __init__(self, budget=None, artifact=None, stop_on_finding=False):
        self.budget = budget or Budget()
        self.artifact = artifact
        self.stop_on_finding = stop_on_finding
        self.modules = {}
        self.refs = {}
        self.extensions = {}
        self.forward_extensions = {}
        self.extension_cache = {}
        self.hash_tables = {}
        self.hash_containers = {}
        self.findings = []
        self.location = ''
        self.depth = 0
        self.call_depth = 0
        self.embedded = None  # dispatcher installs an inert nested-format reader

    def finding(self, kind, detail):
        if len(self.findings) < 100:
            self.findings.append(Finding(kind, self.location, detail))
        if self.stop_on_finding: raise CapabilityFound()

    def module(self, name):
        # Alias names can export the same callable without sharing a module
        # dictionary (os/posix, pickle/_pickle, io/_io). Normalize callable
        # identities in ref(), while retaining each namespace for BUILD.
        return self.modules.setdefault(name, Module(name, imported=True))

    def ref(self, name):
        module, _, rest = name.partition('.')
        name = ALIASES.get(module, module) + ('.' + rest if rest else '')
        name = normal_ref(name)
        return self.refs.setdefault(name, Ref(name))

    def is_class(self, value):
        if isinstance(value, DynamicClass): return True
        if not isinstance(value, Ref): return False
        if value.name in CATALOG['classes']: return True
        if value.name in CATALOG['functions']: return False
        return class_name(value.name)

    def catalog_value(self, item):
        kind, value = item
        if kind == 'm': return self.module(value)
        if kind == 'n': return Namespace(self.module(value))
        if kind == 'r': return self.ref(value)
        if kind == 'd': return value
        if kind == 't': return tuple(value)
        return Instance(value)

    def namespace_data(self, obj):
        if isinstance(obj, Module):
            base = {k: self.catalog_value(v) for k, v in CATALOG['modules'].get(obj.name, {}).items()} if obj.imported else {}
            if obj.imported and obj.name not in CATALOG['modules']: base['__builtins__'] = Namespace(self.module('builtins'))
        elif isinstance(obj, Ref):
            base = {k: self.catalog_value(v) for k, v in (CATALOG['classes'].get(obj.name) or {}).items()}
        else: base = {}
        base.update(getattr(obj, 'attrs', {}))
        return {k: v for k, v in base.items() if v is not DELETED}

    def bind_signature(self, name, args, kw):
        signature = CATALOG['functions'].get(name, {}).get('signature')
        if signature is None: return
        positional = [p for p in signature if p[1] in ('POSITIONAL_ONLY', 'POSITIONAL_OR_KEYWORD')]
        has_varargs = any(p[1] == 'VAR_POSITIONAL' for p in signature)
        has_varkw = any(p[1] == 'VAR_KEYWORD' for p in signature)
        if len(args) > len(positional) and not has_varargs: raise LoadError('too many arguments to ' + name)
        bound = {p[0] for p in positional[:len(args)]}
        for key in kw:
            param = next((p for p in signature if p[0] == key and p[1] in ('POSITIONAL_OR_KEYWORD', 'KEYWORD_ONLY')), None)
            if param is None:
                if not has_varkw: raise LoadError('unexpected keyword to ' + name)
            elif key in bound: raise LoadError('duplicate argument to ' + name)
            else: bound.add(key)
        if any(not default and kind not in ('VAR_POSITIONAL', 'VAR_KEYWORD') and key not in bound
               for key, kind, default in signature):
            raise LoadError('missing argument to ' + name)

    def global_ref(self, module, name, proto):
        from _compat_pickle import IMPORT_MAPPING, NAME_MAPPING
        if proto >= 3 and module in IMPORT_MAPPING and module.split('.')[0] not in sys.stdlib_module_names:
            raise LoadError('Python 2 module name is not mapped at this protocol')
        if proto < 3:
            if (module, name) in NAME_MAPPING:
                module, name = NAME_MAPPING[module, name]
            else:
                module = IMPORT_MAPPING.get(module, module)
        if '.' in name and proto < 4:
            raise LoadError('dotted GLOBAL requires protocol 4')
        value = self.module(module)
        for part in name.split('.'):
            value = self.attr(value, part)
        return value

    def bind_dynamic(self, obj, val):
        if isinstance(val, Instance) and val.name == 'functools.partialmethod':
            return Partial(val.args[0], (obj,) + val.args[1], val.args[2])
        if isinstance(val, Instance) and val.name in ('builtins.staticmethod', 'builtins.classmethod'):
            return val.args[0] if val.name.endswith('staticmethod') else Partial(val.args[0], (obj.cls,))
        meta = CATALOG['functions'].get(val.name, {}) if isinstance(val, Ref) else {}
        signature = meta.get('signature', ())
        descriptor = bool(meta.get('globals')) or bool(signature and signature[0][0] == 'self')
        return Partial(val, (obj,)) if descriptor or isinstance(val, Wrapped) else val

    def special(self, obj, name, args=()):
        if isinstance(obj, DynamicInstance) and name in obj.cls.attrs:
            return self.call(self.bind_dynamic(obj, obj.cls.attrs[name]), args)
        return END

    def truth(self, value):
        if isinstance(value, DynamicInstance):
            result = self.special(value, '__bool__')
            if result is not END:
                if isinstance(result, Unknown) and self.findings: return True
                if type(result) is not bool: raise LoadError('__bool__ must return bool')
                return result
            result = self.special(value, '__len__')
            if result is not END:
                if not isinstance(result, int) or result < 0: raise LoadError('invalid __len__ result')
                return bool(result)
            return self.truth(value.value) if value.value is not None else True
        if isinstance(value, Unknown):
            if self.findings: return True
            raise AnalysisLimit('cannot resolve truth of an opaque value')
        if isinstance(value, Instance):
            if value.name in ('collections.OrderedDict', 'collections.defaultdict', 'collections.Counter'): return bool(value.data)
            known = CATALOG['classes'].get(value.name)
            if known is not None and '__bool__' not in known and '__len__' not in known: return True
            raise AnalysisLimit('cannot resolve truth of ' + value.name)
        if isinstance(value, (Ref, Module, Partial, Wrapped, Accessor, Bound, Memory)): return True
        return bool(value)

    def attr(self, obj, name, default=END):
        if isinstance(obj, Unknown) or isinstance(name, Unknown): return UNKNOWN
        if not isinstance(name, str):
            raise LoadError('attribute name must be a string')
        if isinstance(obj, DynamicInstance) and '__getattribute__' in obj.cls.attrs:
            getter = obj.cls.attrs['__getattribute__']
            if not (isinstance(getter, Ref) and getter.name == 'builtins.object.__getattribute__'):
                return self.call(self.bind_dynamic(obj, getter), (name,))
        attrs = getattr(obj, 'attrs', {})
        if name in attrs and name != '__dict__' and not (isinstance(obj, Partial) and name in ('func', 'args', 'keywords')) and not (isinstance(obj, Module) and name == '__class__'):
            value = attrs[name]
            if value is DELETED:
                if default is not END: return default
                raise LoadError('attribute was deleted')
            if isinstance(obj, DynamicClass) and isinstance(value, Instance):
                if value.name == 'builtins.staticmethod': return value.args[0]
                if value.name == 'builtins.classmethod': return Partial(value.args[0], (obj,))
            return value
        if name == '__dict__' and isinstance(obj, (Ref, Module, Instance, Partial, Wrapped, DynamicClass, DynamicInstance, Memory)):
            if isinstance(obj, Partial): obj.has_dict = True
            if isinstance(obj, Ref) and obj.name in CATALOG['functions'] and not CATALOG['functions'][obj.name].get('globals'):
                if default is not END: return default
                raise LoadError('builtin function has no instance dictionary')
            return Namespace(obj)
        if isinstance(obj, Module):
            if name == '__setstate__':
                return None if default is END else default
            if obj.name == 'sys' and name == 'modules':
                return Namespace('sys.modules')
            if obj.imported and obj.name in ('copyreg', 'pickle'):
                if name == '_inverted_registry': return self.extensions
                if name == '_extension_registry': return self.forward_extensions
                if name == '_extension_cache': return self.extension_cache
            if name == '__builtins__':
                entry = CATALOG['modules'].get(obj.name, {}).get(name) if obj.imported else None
                if entry: return self.catalog_value(entry)
                if obj.imported and obj.name not in CATALOG['modules']: return Namespace(self.module('builtins'))
                if default is not END: return default
                raise LoadError('module has no __builtins__')
            if name == '__class__': return self.ref('types.ModuleType')
            if not obj.imported:
                getter = obj.attrs.get('__getattr__', DELETED)
                if getter is not DELETED: return self.call(getter, (name,))
                if default is not END: return default
                raise LoadError('new module attribute does not exist')
            renamed = MODULE_RENAMES.get((obj.name, name))
            if renamed:
                return self.module(renamed)
            if obj.name not in CATALOG['modules'] and name in MODULE_LINKS.get(obj.name, ()):
                full = obj.name + '.' + name
                return self.module(full if full in MODULE_NAMES else name)
            full = obj.name + '.' + name
            if full in MODULE_NAMES and obj.name not in CATALOG['modules']:
                return self.module(full)
            catalog = CATALOG['modules'].get(obj.name)
            if catalog is not None:
                if name in catalog: return self.catalog_value(catalog[name])
                if name == '__class__': return self.ref('types.ModuleType')
                if name in ('__getattribute__', '__setattr__', '__delattr__'): return Bound(obj, name)
                getter = obj.attrs.get('__getattr__', DELETED)
                if getter is not DELETED: return self.call(getter, (name,))
                if default is not END: return default
                raise LoadError('module attribute does not exist: ' + full)
            getter = obj.attrs.get('__getattr__', DELETED)
            if getter is not DELETED: return self.call(getter, (name,))
            return self.ref(full)
        if isinstance(obj, Partial):
            if name == 'func': return obj.func
            if name == 'args': return obj.args
            if name == 'keywords': return obj.keywords
            if name == '__setstate__': return Bound(obj, name)
        if isinstance(obj, Wrapped):
            if name == '__wrapped__': return obj.func
            return self.attr(obj.func, name, default)
        if isinstance(obj, Ref):
            if name == '__call__': return obj
            if name in ('__module__', '__name__', '__qualname__'):
                meta = CATALOG['functions'].get(obj.name, {})
                module = meta.get('globals') or max((m for m in CATALOG['modules'] if obj.name.startswith(m + '.')), key=len, default=obj.name.rsplit('.', 1)[0])
                if name == '__module__': return meta.get('self', module)
                if name == '__qualname__': return obj.name[len(module) + 1:]
                return obj.name.rsplit('.', 1)[-1]
            if name == '__globals__':
                meta = CATALOG['functions'].get(obj.name)
                if meta is not None and not meta.get('globals'):
                    if default is not END: return default
                    raise LoadError('builtin callable has no __globals__')
                return Namespace(self.module(meta.get('globals') if meta else obj.name.rsplit('.', 1)[0]))
            if name == '__builtins__': return Namespace(self.module('builtins'))
            if name == '__self__':
                meta = CATALOG['functions'].get(obj.name, {})
                return self.module(meta['self']) if 'self' in meta else default if default is not END else UNKNOWN
            if name == '__setstate__': return None if default is END else default
            catalog = CATALOG['classes'].get(obj.name)
            if catalog is not None:
                if name in catalog: return self.catalog_value(catalog[name])
                if default is not END: return default
                raise LoadError('class attribute does not exist: ' + obj.name + '.' + name)
            return self.ref(obj.name + '.' + name)
        if isinstance(obj, DynamicInstance):
            if name in obj.cls.attrs:
                return self.bind_dynamic(obj, obj.cls.attrs[name])
            if obj.value is not None: return self.attr(obj.value, name, default)
            if '__getattr__' in obj.cls.attrs: return self.special(obj, '__getattr__', (name,))
            return default if default is not END else UNKNOWN
        if isinstance(obj, DynamicClass):
            for base in obj.bases:
                value = self.attr(base, name, END)
                if value is not END:
                    return value
            return default if default is not END else UNKNOWN
        if isinstance(obj, Instance):
            if name == '__setstate__':
                # Opaque ML objects have inert state; no artifact-defined
                # Python class is imported into the scanner.
                return None if default is END else default
            if obj.name == 'types.MethodType' and name in ('__func__', '__self__'):
                return obj.args[0 if name == '__func__' else 1]
            if obj.name == 'pathlib.Path' and name in ('name', 'suffix', 'stem', 'parent'):
                path = self.path_value(obj)
                if isinstance(path, str):
                    if name == 'name': return posixpath.basename(path)
                    if name == 'suffix': return posixpath.splitext(path)[1]
                    if name == 'stem': return posixpath.splitext(posixpath.basename(path))[0]
                    return Instance('pathlib.Path', (posixpath.dirname(path),))
            catalog = CATALOG['classes'].get(obj.name)
            if catalog is not None:
                entry = catalog.get(name)
                if entry is None: return UNKNOWN
                if entry[0] in ('d', 't', 'm', 'n'): return self.catalog_value(entry)
                if entry[0] == 'u': return UNKNOWN
                candidate = self.catalog_value(entry)
                if isinstance(candidate, Ref) and self.is_class(candidate): return candidate
            return Bound(obj, name)
        if isinstance(obj, (Namespace, Memory, Lazy)):
            return Bound(obj, name)
        if type(obj) in (str, bytes, bytearray, list, tuple, dict, set, frozenset, int, float, bool, range):
            if name == '__class__': return self.ref('builtins.' + type(obj).__name__)
            if not hasattr(type(obj), name):
                if default is not END: return default
                raise LoadError('builtin value has no attribute ' + name)
            value = getattr(obj, name)
            return Bound(obj, name) if callable(value) else value if concrete(value) else UNKNOWN
        return default if default is not END else UNKNOWN

    def getitem(self, obj, key, default=END):
        if isinstance(obj, Unknown) or isinstance(key, Unknown): return UNKNOWN
        if isinstance(obj, DynamicInstance):
            result = self.special(obj, '__getitem__', (key,))
            if result is not END: return result
        if isinstance(obj, DynamicInstance) and obj.value is not None:
            return self.getitem(obj.value, key, default)
        if isinstance(obj, Namespace):
            if obj.target == 'sys.modules':
                return self.module(key) if isinstance(key, str) else UNKNOWN
            data = self.namespace_data(obj.target)
            if isinstance(obj.target, Module) and obj.target.imported and obj.target.name not in CATALOG['modules'] and key not in ('__dict__', '__class__') and key not in data:
                return self.attr(obj.target, key, default)
            if key in data: return data[key]
            if default is not END: return default
            raise LoadError('dictionary key does not exist')
        if isinstance(obj, Instance) and obj.name == 'numpy.ndarray':
            if isinstance(obj.state, tuple) and len(obj.state) >= 5 and isinstance(obj.state[4], list):
                return self.getitem(obj.state[4], key, default)
            return UNKNOWN
        if isinstance(obj, Instance) and obj.name == 'numpy.NpzFile':
            # Avoid the generic Bound __getitem__ shortcut for this lazy format.
            from modelgate.dispatch import _npy, _read_zip
            import zipfile
            with zipfile.ZipFile(io.BytesIO(obj.args[0])) as zf:
                member = key if key in zf.namelist() else key + '.npy'
                return _npy(_read_zip(zf, zf.getinfo(member), self), obj.args[1] + '!' + member,
                            self, obj.attrs.get('allow_pickle', False))
        if isinstance(obj, Instance) and obj.name in ('collections.OrderedDict', 'collections.defaultdict', 'collections.Counter'):
            data = obj.data if isinstance(obj.data, dict) else {}
            if obj.name == 'collections.Counter' and key not in data and default is END: return 0
            if obj.name == 'collections.defaultdict' and key not in data and default is END:
                factory = obj.attrs.get('default_factory')
                if factory is None: raise LoadError('missing dictionary key')
                value = self.call(factory, ())
                self.dict_set(data, key, value)
                obj.data = data
                return value
            return self.dict_get(data, key, default)
        if type(obj) is dict: return self.dict_get(obj, key, default)
        if type(obj) in (list, tuple, str, bytes, bytearray, range):
            try:
                return obj[key]
            except (KeyError, IndexError, TypeError):
                if default is not END: return default
                raise LoadError('invalid subscript')
        return UNKNOWN

    def setitem(self, obj, key, value):
        if isinstance(obj, DynamicInstance):
            result = self.special(obj, '__setitem__', (key, value))
            if result is not END: return result
            if obj.value is not None: return self.setitem(obj.value, key, value)
        if isinstance(obj, Namespace):
            if obj.target == 'sys.modules':
                self.modules[key] = value
            else:
                if self.sensitive_key(key): raise AnalysisLimit('dynamic namespace key is not modelled')
                obj.target.attrs[key] = value
        elif isinstance(obj, dict): self.dict_set(obj, key, value)
        elif isinstance(obj, list):
            obj[key] = value
        elif isinstance(obj, Instance):
            if obj.name in ('collections.OrderedDict', 'collections.defaultdict', 'collections.Counter'):
                if obj.data is None: obj.data = {}
                self.dict_set(obj.data, key, value)
            else:
                func = obj.attrs.get('__setitem__')
                if func: self.call(func, (key, value))
                else: raise AnalysisLimit('cannot model item assignment on ' + obj.name)
        else:
            raise LoadError('object does not support item assignment')

    def sensitive_key(self, value):
        if isinstance(value, DynamicInstance): return True
        if isinstance(value, (tuple, frozenset)): return any(self.sensitive_key(v) for v in value)
        return False

    def check_finalizer(self, attrs):
        finalizer = attrs.get('__del__')
        if finalizer is None: return
        args, kw = (), {}
        if isinstance(finalizer, Partial):
            args, kw, finalizer = finalizer.args, finalizer.keywords, finalizer.func
        safe = {'builtins.str', 'builtins.int', 'builtins.float', 'builtins.bool', 'builtins.complex',
                'builtins.list', 'builtins.dict', 'builtins.set', 'builtins.tuple', 'builtins.object'}
        if isinstance(finalizer, Ref) and finalizer.name in safe and not args and not kw: return
        raise AnalysisLimit('cannot prove object finalization has no capability')

    def copy_value(self, obj, deep=False, memo=None):
        memo = {} if memo is None else memo
        if id(obj) in memo: return memo[id(obj)]
        if isinstance(obj, (Ref, Wrapped, Accessor)): return obj
        if type(obj) in (str, bytes, int, float, bool, complex, type(None), range): return obj
        if type(obj) in (list, dict, set, bytearray):
            if not deep:
                result = obj.copy()
                self.copy_hashes(obj, result)
                return result
            if type(obj) is bytearray: return obj.copy()
            result = [] if type(obj) is list else {} if type(obj) is dict else set()
            memo[id(obj)] = result
            if type(obj) is list: result.extend(self.copy_value(v, True, memo) for v in obj)
            elif type(obj) is dict:
                for k, v in obj.items(): self.dict_set(result, self.copy_value(k, True, memo), self.copy_value(v, True, memo))
            else:
                for v in obj: self.set_add(result, self.copy_value(v, True, memo))
            return result
        if type(obj) is tuple:
            if not deep: return obj
            result = tuple(self.copy_value(v, True, memo) for v in obj)
            memo[id(obj)] = result
            return result
        if isinstance(obj, DynamicInstance) and not deep and '__copy__' in obj.cls.attrs:
            return self.call(self.attr(obj.cls, '__copy__'), (obj,))
        attrs = getattr(obj, 'attrs', {})
        clsattrs = obj.cls.attrs if isinstance(obj, DynamicInstance) else {}
        if deep and ('__deepcopy__' in attrs or '__deepcopy__' in clsattrs or '__getattribute__' in clsattrs or '__getattr__' in clsattrs or isinstance(obj, Module) and '__getattr__' in attrs):
            copier = self.attr(obj, '__deepcopy__', None)
            if copier is not None: return self.call(copier, (memo,))
        reducer = None
        reduce_args = ()
        for name in ('__reduce_ex__', '__reduce__'):
            if name in attrs or name in clsattrs or isinstance(obj, DynamicInstance) and '__getattribute__' in clsattrs:
                reducer = self.attr(obj, name, None)
                if reducer is not None:
                    reduce_args = (4,) if name == '__reduce_ex__' else ()
                    break
        if reducer is not None:
            reduction = self.call(reducer, reduce_args)
            if isinstance(reduction, str): return obj
            if not isinstance(reduction, tuple) or not 2 <= len(reduction) <= 5: raise LoadError('invalid copy reduction')
            args = self.copy_value(reduction[1], True, memo) if deep else reduction[1]
            result = self.call(reduction[0], args)
            memo[id(obj)] = result
            if len(reduction) > 2 and reduction[2] is not None:
                state = self.copy_value(reduction[2], True, memo) if deep else reduction[2]
                self.build(result, state)
            if len(reduction) > 3 and reduction[3] is not None:
                for item in self.consume(reduction[3]):
                    self.call(self.attr(result, 'append'), (self.copy_value(item, True, memo) if deep else item,))
            if len(reduction) > 4 and reduction[4] is not None:
                for key, val in self.consume(reduction[4]):
                    self.setitem(result, self.copy_value(key, True, memo) if deep else key, self.copy_value(val, True, memo) if deep else val)
            return result
        if isinstance(obj, Module): raise LoadError('modules cannot be copied')
        if isinstance(obj, Partial):
            result = Partial(obj.func, obj.args, obj.keywords.copy(), obj.attrs.copy(), obj.has_dict)
            memo[id(obj)] = result
            if deep:
                result.args = self.copy_value(obj.args, True, memo)
                result.keywords = self.copy_value(obj.keywords, True, memo)
                result.attrs = self.copy_value(obj.attrs, True, memo)
            return result
        if isinstance(obj, DynamicInstance):
            result = DynamicInstance(obj.cls, obj.attrs.copy(), obj.value)
            memo[id(obj)] = result
            if deep:
                result.attrs = self.copy_value(obj.attrs, True, memo)
                result.value = self.copy_value(obj.value, True, memo)
            return result
        return obj

    def hash_value(self, value):
        if isinstance(value, Unknown):
            if self.findings: return object.__hash__(value)
            raise AnalysisLimit('cannot resolve hash of opaque value')
        if isinstance(value, DynamicInstance):
            result = self.special(value, '__hash__')
            if result is not END:
                if not isinstance(result, int): raise LoadError('__hash__ must return an integer')
                result = result if -(1 << 63) <= result < (1 << 63) else hash(result)
                return -2 if result == -1 else result
            if '__eq__' in value.cls.attrs: raise LoadError('unhashable dynamic object')
            return self.hash_value(value.value) if value.value is not None else object.__hash__(value)
        if isinstance(value, Instance) and value.name in ('numpy.ndarray', 'collections.OrderedDict', 'collections.defaultdict', 'collections.Counter'):
            raise LoadError('unhashable data object')
        if isinstance(value, tuple) and self.sensitive_key(value):
            for item in value: self.hash_value(item)
            raise AnalysisLimit('compound dynamic hash is not modelled')
        if isinstance(value, frozenset) and self.sensitive_key(value):
            raise AnalysisLimit('compound dynamic hash is not modelled')
        return hash(value)

    def equal(self, left, right):
        if left is right: return True
        result = self.special(left, '__eq__', (right,))
        if result is not END: return self.truth(result)
        result = self.special(right, '__eq__', (left,))
        if result is not END: return self.truth(result)
        if isinstance(left, DynamicInstance): left = left.value if left.value is not None else left
        if isinstance(right, DynamicInstance): right = right.value if right.value is not None else right
        if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
            return len(left) == len(right) and all(self.equal(a, b) for a, b in zip(left, right))
        if isinstance(left, Unknown) or isinstance(right, Unknown): raise AnalysisLimit('cannot resolve equality')
        return left == right

    def key_records(self, container):
        records = self.hash_tables.get(id(container))
        if records is None:
            records = []
            for key in container:
                if self.sensitive_key(key): raise AnalysisLimit('cannot recover a dynamic key hash')
                records.append((key, self.hash_value(key)))
            if len(self.hash_tables) >= MAX_ITEMS: raise AnalysisLimit('dynamic container count limit exceeded')
            self.hash_tables[id(container)] = records
            self.hash_containers[id(container)] = container
        else:
            records[:] = [(key, h) for key, h in records if key in container]
        return records

    def find_key(self, container, key):
        h = self.hash_value(key)
        records = self.key_records(container)
        for stored, stored_hash in records:
            if stored_hash == h and (stored is key or self.equal(stored, key)):
                return stored, h
        return END, h

    def dict_set(self, container, key, value):
        if not self.sensitive_key(key) and id(container) not in self.hash_tables:
            self.hash_value(key)
            container[key] = value
            return
        stored, h = self.find_key(container, key)
        if stored is END:
            if any(old is key and old_hash != h for old, old_hash in self.hash_tables[id(container)]):
                raise AnalysisLimit('a mutable key changed hash during insertion')
            container[key] = value
            self.hash_tables[id(container)].append((key, h))
        else: container[stored] = value

    def dict_get(self, container, key, default=END):
        if not self.sensitive_key(key) and id(container) not in self.hash_tables:
            self.hash_value(key)
            try: return container[key]
            except KeyError:
                if default is not END: return default
                raise LoadError('dictionary key does not exist')
        stored, _ = self.find_key(container, key)
        if stored is not END: return container[stored]
        if default is not END: return default
        raise LoadError('dictionary key does not exist')

    def set_add(self, container, value):
        if not self.sensitive_key(value) and id(container) not in self.hash_tables:
            self.hash_value(value)
            container.add(value)
            return
        stored, h = self.find_key(container, value)
        if stored is END:
            if any(old is value and old_hash != h for old, old_hash in self.hash_tables[id(container)]):
                raise AnalysisLimit('a mutable key changed hash during insertion')
            container.add(value)
            self.hash_tables[id(container)].append((value, h))

    def copy_hashes(self, source, target):
        if id(source) in self.hash_tables:
            self.hash_tables[id(target)] = list(self.hash_tables[id(source)])
            self.hash_containers[id(target)] = target

    def delete_attr(self, obj, name):
        if isinstance(obj, Partial) and name in ('func', 'args', 'keywords'):
            raise LoadError('partial attribute is read-only')
        if isinstance(obj, Module) and name in ('__dict__', '__class__'):
            raise LoadError('module attribute is read-only')
        missing = object()
        if self.attr(obj, name, missing) is missing: raise LoadError('attribute does not exist')
        if not hasattr(obj, 'attrs'): raise LoadError('object has no writable attributes')
        obj.attrs[name] = DELETED

    def delitem(self, obj, key):
        if isinstance(obj, Namespace):
            if obj.target == 'sys.modules':
                if key not in self.modules: raise LoadError('module is not cached')
                del self.modules[key]
            else: self.delete_attr(obj.target, key)
        elif isinstance(obj, dict):
            if self.sensitive_key(key) or id(obj) in self.hash_tables:
                stored, _ = self.find_key(obj, key)
                if stored is END: raise LoadError('dictionary key does not exist')
                del obj[stored]
            else: del obj[key]
        elif isinstance(obj, list): del obj[key]
        else: raise LoadError('object does not support item deletion')

    def build(self, obj, state):
        if isinstance(obj, Unknown): raise AnalysisLimit('cannot resolve BUILD target')
        if isinstance(obj, Lazy):
            if isinstance(state, int): obj.index = max(0, state)
            return
        if isinstance(obj, Partial):
            if not (isinstance(state, tuple) and len(state) == 4 and isinstance(state[1], tuple)
                    and (state[2] is None or isinstance(state[2], dict))
                    and (state[3] is None or isinstance(state[3], dict))):
                raise LoadError('invalid functools.partial state')
            obj.func, obj.args, keywords, attrs = state
            obj.keywords = {} if keywords is None else keywords
            obj.attrs = {} if attrs is None else attrs
            obj.has_dict = attrs is not None
            return
        setstate = self.attr(obj, '__setstate__', None)
        if setstate is not None:
            self.call(setstate, (state,))
            return
        if isinstance(obj, Instance):
            if obj.name == 'numpy.dtype' and isinstance(state, tuple):
                for i in (0, 1, 5, 6, 7):
                    if i < len(state) and isinstance(state[i], (DynamicInstance, Unknown)): raise AnalysisLimit('opaque NumPy dtype state field')
            if obj.name == 'numpy.ndarray' and isinstance(state, tuple) and len(state) >= 5:
                controls = (state[0], state[3]) + (tuple(state[1]) if isinstance(state[1], (tuple, list)) else (state[1],))
                if any(isinstance(v, (DynamicInstance, Unknown)) for v in controls): raise AnalysisLimit('opaque NumPy array state field')
                if isinstance(state[4], (DynamicInstance, Unknown)): raise AnalysisLimit('opaque NumPy buffer state')
            obj.state = state
        slotstate = None
        if isinstance(state, tuple) and len(state) == 2:
            state, slotstate = state
        if state and isinstance(obj, Ref) and obj.name in CATALOG['functions'] and not CATALOG['functions'][obj.name].get('globals'):
            raise LoadError('BUILD target has no instance dictionary')
        if state and not hasattr(obj, 'attrs'):
            raise LoadError('BUILD target has no instance dictionary')
        if isinstance(state, dict) and hasattr(obj, 'attrs'):
            obj.attrs.update(state)
        if isinstance(slotstate, dict) and hasattr(obj, 'attrs'):
            obj.attrs.update(slotstate)

    def path_value(self, obj):
        if isinstance(obj, Instance) and obj.name.startswith('pathlib.'):
            args = tuple(self.path_value(v) for v in obj.args)
            if all(isinstance(a, (str, bytes)) for a in args):
                return posixpath.join(*args) if args else '.'
        return obj

    def iterator(self, obj):
        if isinstance(obj, DictView):
            data = self.namespace_data(obj.target.target) if isinstance(obj.target, Namespace) and hasattr(obj.target.target, 'attrs') else self.modules if isinstance(obj.target, Namespace) else obj.target
            keys = list(data)
            return Lazy('dict_view', (obj, keys, len(keys)))
        if isinstance(obj, DynamicInstance):
            if '__iter__' in obj.cls.attrs: return self.iterator(self.call(self.attr(obj, '__iter__'), ()))
            if obj.value is not None: return self.iterator(obj.value)
        if isinstance(obj, Lazy): return obj
        if isinstance(obj, Instance):
            if obj.name in ('collections.OrderedDict', 'collections.defaultdict', 'collections.Counter'):
                obj = obj.data if isinstance(obj.data, dict) else {}
            elif obj.name == 'numpy.ndarray' and isinstance(obj.state, tuple) and len(obj.state) >= 5 and isinstance(obj.state[4], list):
                obj = obj.state[4]
            elif isinstance(obj.state, (list, tuple, dict)):
                obj = obj.state
            else:
                return Lazy('values', ([],))
        if isinstance(obj, Namespace):
            obj = self.namespace_data(obj.target) if hasattr(obj.target, 'attrs') else self.modules
        if type(obj) in (dict, list, tuple, set, frozenset, range, str, bytes, bytearray):
            if len(obj) > MAX_ITEMS: raise AnalysisLimit('iterator size limit exceeded')
            if isinstance(obj, (dict, set, frozenset)):
                return Lazy('values', (list(obj),), keywords={'original': obj, 'size': len(obj)})
            # List iterators retain a live reference: appending to a memoized
            # list after constructing map/filter must be visible at next().
            return Lazy('values', (obj,))
        if isinstance(obj, Unknown): return Lazy('values', ([],))
        raise LoadError('object is not iterable')

    def next(self, it):
        self.budget.tick()
        if not isinstance(it, Lazy): raise LoadError('not an iterator')
        if it.done: return END
        name, args = it.name, it.args
        if name == 'dict_view':
            view, keys, size = args
            target = view.target
            data = self.namespace_data(target.target) if isinstance(target, Namespace) and hasattr(target.target, 'attrs') else self.modules if isinstance(target, Namespace) else target
            if len(data) != size: raise LoadError('dictionary changed size during iteration')
            if it.index >= len(keys):
                it.done = True
                return END
            key = keys[it.index]
            it.index += 1
            if key not in data: raise AnalysisLimit('dictionary keys changed during iteration')
            if view.kind == 'keys': return key
            value = data[key]
            return value if view.kind == 'values' else (key, value)
        if name == 'values':
            if 'original' in it.keywords and len(it.keywords['original']) != it.keywords['size']:
                raise LoadError('container changed size during iteration')
            if it.index >= len(args[0]):
                it.done = True
                return END
            value = args[0][it.index]
            it.index += 1
            return value
        if name in ('map', 'zip', 'starmap', 'filter', 'filterfalse', 'takewhile', 'dropwhile', 'enumerate', 'reversed'):
            if name == 'zip' and not args:
                it.done = True
                return END
            if it.sources is None:
                sources = args[1:] if name in ('map', 'starmap', 'filter', 'filterfalse', 'takewhile', 'dropwhile') else args[:1] if name in ('enumerate', 'reversed') else args
                it.sources = [self.iterator(v) for v in sources]
            while True:
                vals = []
                for i, source in enumerate(it.sources):
                    value = self.next(source)
                    if value is END:
                        it.done = True
                        if name == 'zip' and it.keywords.get('strict'):
                            if i: raise LoadError('zip input lengths differ')
                            for later in it.sources[1:]:
                                if self.next(later) is not END: raise LoadError('zip input lengths differ')
                        return END
                    vals.append(value)
                if name == 'map': return self.call(args[0], tuple(vals))
                if name == 'starmap': return self.call(args[0], tuple(self.consume(vals[0])))
                if name == 'zip': return tuple(vals)
                if name == 'enumerate':
                    index = it.index + (args[1] if len(args) > 1 else it.keywords.get('start', 0))
                    it.index += 1
                    return (index, vals[0])
                if name == 'reversed': return vals[0]
                if name == 'dropwhile' and it.index: return vals[0]
                pred = self.call(args[0], (vals[0],)) if args[0] is not None else vals[0]
                truth = self.truth(pred)
                if name == 'takewhile' and not truth:
                    it.done = True
                    return END
                if name == 'dropwhile':
                    if it.index or not truth:
                        it.index = 1
                        return vals[0]
                elif truth != (name == 'filterfalse'):
                    return vals[0]
        if name == 'reversed_sequence':
            index = args[1] - it.index - 1
            if index < 0 or index >= len(args[0]):
                it.done = True
                return END
            it.index += 1
            return args[0][index]
        if name == 'repeat':
            if len(args) > 1 and it.index >= args[1]: return END
            it.index += 1
            return args[0]
        if name == 'count':
            result = (args[0] if args else 0) + it.index * (args[1] if len(args) > 1 else 1)
            it.index += 1
            return result
        if name == 'tee':
            state = args[0]
            if it.index >= len(state.cache) and not state.done:
                if state.iterator is None: state.iterator = self.iterator(state.source)
                value = self.next(state.iterator)
                if value is END: state.done = True
                else:
                    if len(state.cache) >= MAX_ITEMS: raise AnalysisLimit('tee buffer size limit exceeded')
                    state.cache.append(value)
            if it.index >= len(state.cache): return END
            value = state.cache[it.index]
            it.index += 1
            return value
        if name == 'chain.from_iterable':
            if it.sources is None: it.sources = [self.iterator(args[0]), None]
            while True:
                if it.sources[1] is not None:
                    value = self.next(it.sources[1])
                    if value is not END: return value
                item = self.next(it.sources[0])
                if item is END: return END
                it.sources[1] = self.iterator(item)
        if name == 'chain':
            if it.sources is None: it.sources = [None]
            while it.index < len(args):
                if it.sources[0] is None: it.sources[0] = self.iterator(args[it.index])
                value = self.next(it.sources[0])
                if value is not END: return value
                it.index += 1
                it.sources[0] = None
            return END
        if name == 'islice':
            if it.sources is None: it.sources = [self.iterator(args[0])]
            start, stop, step = (0, args[1], 1) if len(args) == 2 else (args[1], args[2], args[3] if len(args) > 3 else 1)
            while stop is None or it.index < stop:
                value = self.next(it.sources[0])
                index = it.index
                it.index += 1
                if value is END: return END
                if index >= start and (index - start) % step == 0: return value
            return END
        if name == 'calliter':
            value = self.call(args[0], ())
            if concrete(value) and concrete(args[1]) and value == args[1]: return END
            return value
        if name == 'cycle':
            if it.sources is None: it.sources = [self.iterator(args[0]), []]
            if not it.keywords.get('original_done'):
                value = self.next(it.sources[0])
                if value is not END:
                    if len(it.sources[1]) >= MAX_ITEMS: raise AnalysisLimit('cycle cache limit exceeded')
                    it.sources[1].append(value)
                    return value
                it.keywords['original_done'] = True
            if not it.sources[1]:
                it.done = True
                return END
            value = it.sources[1][it.index % len(it.sources[1])]
            it.index += 1
            return value
        return END

    def consume(self, obj, maximum=MAX_ITEMS):
        it = self.iterator(obj)
        result = []
        for _ in range(maximum + 1):
            value = self.next(it)
            if value is END: return result
            result.append(value)
        raise AnalysisLimit('iterator consumption limit exceeded')

    def call(self, func, args, kw=None, mode='call'):
        self.budget.tick()
        if isinstance(args, Unknown):
            self.finding('unsupported', 'cannot resolve REDUCE argument tuple')
            return UNKNOWN
        if isinstance(args, DynamicInstance) and isinstance(args.value, tuple): args = args.value
        if not isinstance(args, tuple): raise LoadError('REDUCE arguments are not a tuple')
        if kw is not None and not isinstance(kw, dict): raise LoadError('keywords are not a dict')
        self.call_depth += 1
        try:
            if self.call_depth > 256: raise AnalysisLimit('call nesting limit exceeded')
            return self._call(func, args, kw or {}, mode)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError, ZeroDivisionError) as exc:
            raise LoadError(str(exc)) from None
        finally:
            self.call_depth -= 1

    def _call(self, func, args, kw, mode):
        if isinstance(func, Unknown):
            self.finding('unsupported', 'cannot resolve callable: ' + func.reason)
            return UNKNOWN
        if any(isinstance(v, Unknown) for v in args + tuple(kw.values())):
            if isinstance(func, Ref) and func.name in ('builtins.__import__', 'importlib.import_module'): return UNKNOWN
            if not self.findings: self.finding('unsupported', 'cannot resolve call arguments')
            return UNKNOWN
        if mode == 'new' and not self.is_class(func): raise LoadError('NEWOBJ requires a class')
        if isinstance(func, Partial):
            if mode == 'new': raise LoadError('NEWOBJ requires a class')
            return self.call(func.func, func.args + args, {**func.keywords, **kw})
        if isinstance(func, Wrapped): return self.call(func.func, args, kw, mode)
        if isinstance(func, Accessor):
            if func.name == 'decorator': return Wrapped(args[0])
            if func.name == 'weakref': return func.args[0]
            if not args: raise LoadError('accessor requires an operand')
            if func.name == 'attrgetter':
                vals = []
                for path in func.args:
                    obj = args[0]
                    for part in path.split('.'): obj = self.attr(obj, part)
                    vals.append(obj)
                return vals[0] if len(vals) == 1 else tuple(vals)
            if func.name == 'itemgetter':
                vals = [self.getitem(args[0], key) for key in func.args]
                return vals[0] if len(vals) == 1 else tuple(vals)
            return self.call(self.attr(args[0], func.args[0]), func.args[1:], func.keywords)
        if isinstance(func, Bound): return self.bound(func.target, func.name, args, kw)
        if isinstance(func, DynamicClass):
            self.check_finalizer(func.attrs)
            obj = DynamicInstance(func)
            if '__new__' in func.attrs:
                new = func.attrs['__new__']
                if isinstance(new, Instance) and new.name == 'builtins.staticmethod': new = new.args[0]
                obj = self.call(new, (func,) + args, kw)
            elif func.bases:
                # Inherited constructors carry the same capabilities. In
                # particular, deriving a class does not hide Popen.__init__.
                base = func.bases[0]
                if not isinstance(base, Ref) or base.name != 'builtins.object':
                    base_mode = 'new' if mode == 'new' or '__init__' in func.attrs else 'call'
                    obj.value = self.call(base, args, kw, base_mode)
            if mode != 'new' and isinstance(obj, DynamicInstance) and '__init__' in func.attrs:
                result = self.call(self.bind_dynamic(obj, func.attrs['__init__']), args, kw)
                if result is not None and concrete(result): raise LoadError('__init__ must return None')
            return obj
        if isinstance(func, DynamicInstance):
            if '__call__' in func.cls.attrs: return self.special(func, '__call__', args) if not kw else self.call(self.bind_dynamic(func, func.cls.attrs['__call__']), args, kw)
            if func.value is not None: return self.call(func.value, args, kw)
            raise LoadError('dynamic object is not callable')
        if isinstance(func, Instance):
            if func.name in ('concurrent.futures.Future', 'concurrent.futures._base.Future'): raise LoadError('Future is not callable')
            if func.name == 'builtins.staticmethod': return self.call(func.args[0], args, kw)
            if func.name == 'types.MethodType':
                return self.call(func.args[0], (func.args[1],) + args, kw)
            if func.name == 'xmlrpc.client.ServerProxy':
                if args and args[0] == 'close': return Bound(func, '_close')
                if args and args[0] == 'transport': return Instance('xmlrpc.client.Transport')
                raise LoadError('unknown ServerProxy selector')
            known = CATALOG['classes'].get(func.name)
            if known is not None and '__call__' not in known: raise LoadError('object is not callable')
            return self.call(self.ref(func.name + '.__call__'), (func,) + args, kw)
        if not isinstance(func, Ref):
            if isinstance(func, Unknown):
                self.finding('unsupported', 'cannot resolve callable: ' + func.reason)
                return UNKNOWN
            raise LoadError('object is not callable')
        name = func.name
        if self.is_class(func): self.check_finalizer(func.attrs)
        if mode == 'new':
            if name in ('io.BytesIO', 'io.StringIO'): return Memory(b'' if name == 'io.BytesIO' else '')
            if name in ('builtins.list', 'builtins.dict', 'builtins.set', 'builtins.bytearray'):
                return {'builtins.list': list, 'builtins.dict': dict, 'builtins.set': set, 'builtins.bytearray': bytearray}[name]()
            if not self.is_class(func): raise LoadError('NEWOBJ requires a class')
            if name not in ('functools.partial', 'types.FunctionType', 'types.CodeType', 'builtins.type', 'builtins.bytes', 'builtins.bytearray', 'builtins.str', 'builtins.int', 'builtins.float', 'builtins.tuple', 'builtins.frozenset', 'builtins.range', 'builtins.slice', 'pathlib.Path', 'pathlib.PosixPath', 'pathlib.PurePath', 'pathlib.PurePosixPath'):
                return Instance(name, args)
        if args and name == 'functools.partial.__setstate__' and isinstance(args[0], Partial):
            self.build(args[0], args[1]); return None
        if args and isinstance(args[0], Instance):
            clsname, _, method = name.rpartition('.')
            if method == '__init__':
                return self.call(self.ref(clsname), args[1:], kw)
            if (method in ('start', 'run') and clsname in ('threading.Thread', 'threading.Timer', 'multiprocessing.process.BaseProcess', 'multiprocessing.Process', 'multiprocessing.context.Process') or
                method in ('submit', 'map') and clsname in ('concurrent.futures.ThreadPoolExecutor', 'concurrent.futures.ProcessPoolExecutor') or
                method == 'load' and clsname == 'pickle.Unpickler'):
                return self.bound(args[0], method, args[1:], kw)
        self.bind_signature(name, args, kw)
        if mode != 'new' and name in CATALOG['classes']:
            initializer = (CATALOG['classes'][name] or {}).get('__init__')
            if initializer and initializer[0] == 'r':
                self.bind_signature(normal_ref(initializer[1]), (UNKNOWN,) + args, kw)
        # Move explicitly supplied leading keyword arguments into positions
        # for the small semantic models below; retain optional keyword values.
        signature = CATALOG['functions'].get(name, {}).get('signature', ())
        for key, kind, _ in signature[len(args):]:
            if kind != 'POSITIONAL_OR_KEYWORD' or key not in kw: break
            args += (kw[key],)
            kw = {k: v for k, v in kw.items() if k != key}
        if name == 'builtins.object.__new__' and args and isinstance(args[0], DynamicClass):
            return DynamicInstance(args[0])
        if name == 'builtins.object.__new__' and args and isinstance(args[0], Ref):
            if not self.is_class(args[0]): raise LoadError('object.__new__ requires a class')
            return Instance(args[0].name, args[1:])
        if name.endswith('.__new__'):
            return self.call(args[0] if args and self.is_class(args[0]) else self.ref(name[:-8]), args[1:], kw, 'new')
        if name.endswith('.__call__') and name.startswith(('builtins.', 'functools.', 'operator.')) and args:
            return self.call(args[0], args[1:], kw)
        if name == 'builtins.__import__':
            module = args[0]
            fromlist = args[3] if len(args) > 3 else kw.get('fromlist', ())
            return self.module(module if fromlist else module.split('.')[0])
        if name in ('importlib.import_module', 'importlib._bootstrap._gcd_import'):
            module = args[0]
            if module.startswith('.') and len(args) > 1:
                module = args[1] + module
            return self.module(module)
        if name == 'pkgutil.resolve_name':
            path = args[0].replace(':', '.')
            parts = path.split('.')
            obj = self.module(parts[0])
            for part in parts[1:]: obj = self.attr(obj, part)
            return obj
        if name in ('builtins.object.__getattribute__', 'builtins.type.__getattribute__', 'types.ModuleType.__getattribute__'):
            return self.attr(args[0], args[1])
        if name in ('builtins.object.__setattr__', 'builtins.type.__setattr__', 'types.ModuleType.__setattr__'):
            if hasattr(args[0], 'attrs'): args[0].attrs[args[1]] = args[2]
            return None
        if name in ('copyreg.add_extension', 'copyreg.remove_extension'):
            module, attr, code = args
            if isinstance(code, DynamicInstance): code = self.call(self.ref('builtins.int'), (code,))
            code = int(code)
            if not 1 <= code <= 0x7fffffff: raise LoadError('invalid pickle extension code')
            forward = self.attr(self.module('copyreg'), '_extension_registry')
            inverse = self.attr(self.module('copyreg'), '_inverted_registry')
            key = (module, attr)
            if name.endswith('remove_extension'):
                if self.dict_get(forward, key, None) != code or self.dict_get(inverse, code, None) != key:
                    raise LoadError('pickle extension is not registered')
                self.delitem(forward, key)
                self.delitem(inverse, code)
                return None
            previous = self.dict_get(forward, key, None)
            inverted = self.dict_get(inverse, code, None)
            if previous == code and inverted == key: return None
            if previous is not None or inverted is not None: raise LoadError('pickle extension code conflict')
            self.dict_set(forward, key, code)
            self.dict_set(inverse, code, key)
            return None
        if name == 'copyreg.clear_extension_cache':
            cache = self.attr(self.module('copyreg'), '_extension_cache')
            if not isinstance(cache, dict): raise AnalysisLimit('opaque extension cache')
            cache.clear()
            return None
        if name in ('builtins.getattr', 'inspect.getattr_static'): return self.attr(args[0], args[1], args[2] if len(args) > 2 else END)
        if name == 'builtins.hasattr':
            missing = object()
            return self.attr(args[0], args[1], missing) is not missing
        if name == 'builtins.setattr':
            if isinstance(args[0], Partial):
                if args[1] in ('func', 'args', 'keywords'): raise LoadError('partial attribute is read-only')
                args[0].has_dict = True
            if hasattr(args[0], 'attrs'): args[0].attrs[args[1]] = args[2]
            return None
        if name == 'builtins.delattr':
            self.delete_attr(args[0], args[1])
            return None
        if name in ('sys.exit', 'os._exit', 'os.abort', 'builtins.exit', 'builtins.quit'):
            raise LoadError('loader terminates here')
        if name in ('builtins.vars', 'builtins.globals', 'builtins.locals'):
            if args and isinstance(args[0], Partial): args[0].has_dict = True
            return Namespace(args[0]) if args else Namespace(self.module('builtins'))
        if name == 'builtins.type':
            if len(args) == 3:
                if not isinstance(args[0], str) or not isinstance(args[1], tuple) or not isinstance(args[2], dict) or not all(self.is_class(b) for b in args[1]):
                    raise LoadError('invalid type construction')
                return DynamicClass(args[0], args[1], args[2])
            if isinstance(args[0], Module): return self.ref('types.ModuleType')
            if isinstance(args[0], Instance): return self.ref(args[0].name)
            if isinstance(args[0], DynamicInstance): return args[0].cls
            return self.ref('builtins.' + type(args[0]).__name__)
        if name == 'types.ModuleType':
            return Module(args[0], {'__name__': args[0], '__doc__': args[1] if len(args) > 1 else None,
                                    '__package__': None, '__loader__': None, '__spec__': None})
        if name in ('operator.attrgetter', 'operator.itemgetter', 'operator.methodcaller'):
            if not args: raise LoadError('accessor requires a field')
            if name.endswith('attrgetter') and (kw or not all(isinstance(a, str) for a in args)): raise LoadError('invalid attrgetter arguments')
            if name.endswith('itemgetter') and kw: raise LoadError('itemgetter does not accept keywords')
            if name.endswith('methodcaller') and not isinstance(args[0], str): raise LoadError('method name must be a string')
            return Accessor(name.split('.')[-1], args, kw)
        if name in ('functools.partial', 'functools.partialmethod'):
            if not args: raise LoadError('partial requires a function')
            if isinstance(args[0], (str, bytes, bytearray, int, float, bool, type(None), list, tuple, dict, set, frozenset, Module)):
                raise LoadError('partial requires a callable')
            if name == 'functools.partialmethod': return Instance(name, (args[0], args[1:], dict(kw)))
            if isinstance(args[0], Partial) and not args[0].has_dict:
                inner = args[0]
                return Partial(inner.func, inner.args + args[1:], {**inner.keywords, **kw})
            return Partial(args[0], args[1:], dict(kw))
        if name in ('functools.lru_cache', 'functools.cache'):
            if args and isinstance(args[0], (Ref, Partial, Wrapped, Accessor, DynamicInstance)):
                return Wrapped(args[0])
            return Accessor('decorator', ())
        if name == 'functools.wraps': return Partial(self.ref('functools.update_wrapper'), (), {'wrapped': args[0], **kw})
        if name == 'functools.update_wrapper': return args[0]
        if name == 'inspect.unwrap':
            obj = args[0]
            for _ in range(100):
                if isinstance(obj, Wrapped): obj = obj.func
                elif hasattr(obj, 'attrs') and '__wrapped__' in obj.attrs: obj = obj.attrs['__wrapped__']
                else: return obj
            raise LoadError('wrapper loop')
        if name in ('functools.singledispatch', 'functools.singledispatchmethod'): return Wrapped(args[0])
        if name in ('weakref.proxy',): return Wrapped(args[0])
        if name in ('weakref.ref', 'weakref.WeakMethod'): return Accessor('weakref', (args[0],))
        if name in ('copy.copy', 'copy.deepcopy'):
            return self.copy_value(args[0], name.endswith('deepcopy'))
        if name in ('copyreg._reconstructor', 'copyreg.__newobj__', 'copyreg.__newobj_ex__'):
            if name == 'copyreg.__newobj_ex__': return self.call(args[0], args[1], args[2], 'new')
            return self.call(args[0], args[1:] if name.endswith('__newobj__') else (), {}, 'new')
        if name in ('types.MethodType', 'builtins.staticmethod', 'builtins.classmethod'): return Instance(name, args)
        if name == 'types.CodeType': return Instance('code', args)
        if name == 'types.FunctionType':
            if args and isinstance(args[0], Instance) and args[0].name == 'code':
                self.finding('code-execution', 'materializes serialized bytecode with types.FunctionType')
                return Instance('python-function', args)
            raise LoadError('FunctionType requires a code object')
        if name in ('pickle.dumps', 'pickle.dump'):
            value = args[0]
            if not concrete(value):
                self.finding('unsupported', 'cannot serialize an opaque object statically')
                return UNKNOWN
            import pickle
            protocol = args[1 if name.endswith('dumps') else 2] if len(args) > (1 if name.endswith('dumps') else 2) else kw.get('protocol')
            raw = pickle.dumps(value, protocol=protocol)
            self.budget.add_bytes(len(raw))
            if name.endswith('dumps'): return raw
            if len(args) < 2: raise LoadError('pickle.dump requires an output stream')
            if isinstance(args[1], Memory):
                self.bound(args[1], 'write', (raw,), {})
                return None
            self.finding('file-access', 'pickle.dump writes an external stream')
            return None
        if name in ('marshal.loads', 'marshal.load'):
            value = args[0]
            if name == 'marshal.load':
                if not isinstance(value, Memory):
                    raise AnalysisLimit('cannot resolve marshal file input')
                data = self.as_bytes(value.data[value.pos:])
            else: data = self.as_bytes(value)
            if data and data[0] & 0x7f == ord('c'):
                return Instance('code', (data,))
            # The small data subset can be decoded without creating code.
            return self.marshal_data(data)
        if name in ('dill._dill._create_function', 'cloudpickle.cloudpickle._make_function', 'cloudpickle.cloudpickle._make_skeleton_function'):
            self.finding('code-execution', 'materializes serialized Python function with ' + name)
            return Instance('python-function', args)
        if name in ('dill._dill._create_code', 'cloudpickle.cloudpickle._builtin_type'):
            return Instance('code', args)
        if name in ('timeit.timeit', 'timeit.repeat', 'timeit.Timer'):
            stmt = args[0] if args else kw.get('stmt', END)
            setup = args[1] if len(args) > 1 else kw.get('setup', END)
            for source in (stmt, setup):
                if isinstance(source, str):
                    self.finding('code-execution', 'compiles source in ' + name)
                elif source is not END and not isinstance(source, (Ref, Partial, Wrapped, Accessor, DynamicInstance)):
                    raise LoadError('invalid timeit statement')
            timer = Instance('timeit.Timer', attrs={'stmt': stmt, 'setup': setup})
            if name != 'timeit.Timer': return self.bound(timer, 'timeit', (), kw)
            return timer
        if name in CODE_CALLS:
            source_index = 1 if name.startswith(('bdb.Bdb.', 'pdb.Pdb.', 'code.InteractiveInterpreter.', 'code.InteractiveConsole.')) else 0
            if name.startswith(('builtins.', 'bdb.', 'pdb.', 'code.', 'numpy.testing.')) and len(args) > source_index:
                source = args[source_index]
                if not isinstance(source, (str, bytes, bytearray, Unknown)) and not (isinstance(source, Instance) and source.name == 'code'):
                    raise LoadError('invalid Python source')
            self.finding('code-execution', 'calls ' + name)
            return UNKNOWN
        if name in PROCESS_CALLS or (name.startswith('os.') and name.split('.')[-1].startswith(('exec', 'spawn'))):
            self.finding('process', 'calls ' + name)
            return Instance(name, args) if class_name(name) else 0
        if name in FILE_CALLS or name.startswith('tempfile.') and name.rsplit('.', 1)[-1] in ('TemporaryFile', 'NamedTemporaryFile', 'TemporaryDirectory', 'mkstemp', 'mkdtemp', 'SpooledTemporaryFile', 'mktemp'):
            if name == 'logging.basicConfig' and 'filename' not in kw: return None
            if name in ('logging.FileHandler', 'logging.handlers.RotatingFileHandler', 'logging.handlers.TimedRotatingFileHandler', 'logging.handlers.WatchedFileHandler') and kw.get('delay'): return Instance(name, args)
            self.finding('file-access', 'calls ' + name)
            return Instance(name, args)
        if name.startswith('io.') and name.rsplit('.', 1)[-1] in ('read', 'read1', 'readinto', 'readinto1', 'readline', 'readlines', 'write', 'writelines', 'truncate', 'flush'):
            if args and isinstance(args[0], Memory): return self.bound(args[0], name.rsplit('.', 1)[-1], args[1:], kw)
            self.finding('file-access', 'accesses external stream via ' + name)
            return UNKNOWN
        if name in ('subprocess.Popen._execute_child', 'subprocess.Popen._posix_spawn', '_posixsubprocess.fork_exec'):
            self.finding('process', 'calls ' + name)
            return UNKNOWN
        if name in NETWORK_CALLS:
            self.finding('network', 'calls ' + name)
            return UNKNOWN
        if name in NESTED_CALLS:
            return self.nested(name, args, kw)
        if name in ('ftplib.FTP', 'ftplib.FTP_TLS', 'smtplib.SMTP', 'smtplib.SMTP_SSL', 'smtplib.LMTP'):
            host = args[0] if args else kw.get('host', '')
            if self.truth(host): self.finding('network', 'connects while constructing ' + name)
            # False host values do not replace the class's empty host string.
            attrs = {k: v for k, v in kw.items() if k != 'host'}
            return Instance(name, args, attrs)
        if name in ('socket.socket', 'socket.SocketType', 'http.client.HTTPConnection', 'http.client.HTTPSConnection', 'xmlrpc.client.ServerProxy', 'xmlrpc.client.Transport', 'xmlrpc.client.SafeTransport'):
            attrs = dict(kw)
            if name in ('http.client.HTTPConnection', 'http.client.HTTPSConnection'):
                host = args[0] if args else kw.get('host')
                port = args[1] if len(args) > 1 else kw.get('port')
                if isinstance(host, str):
                    if port is None and host.count(':') == 1:
                        hostpart, portpart = host.rsplit(':', 1)
                        if portpart.isdigit(): host, port = hostpart, int(portpart)
                    host = host.removeprefix('[').removesuffix(']')
                attrs.update(host=host, port=port if port is not None else 443 if name.endswith('HTTPSConnection') else 80, sock=None)
            return Instance('socket.socket' if name == 'socket.SocketType' else name, args, attrs)
        if name in ('io.BufferedReader', 'io.BufferedWriter', 'io.BufferedRandom', 'io.TextIOWrapper') and args and isinstance(args[0], Memory):
            if name == 'io.TextIOWrapper':
                return Memory(self.as_bytes(args[0].data).decode(kw.get('encoding', 'utf-8')))
            return args[0]
        if name in ('io.BytesIO', 'io.StringIO'):
            value = args[0] if args else (b'' if name == 'io.BytesIO' else '')
            if isinstance(value, Unknown): return Memory(value)
            if name == 'io.BytesIO' and not isinstance(value, (bytes, bytearray)): raise LoadError('BytesIO requires bytes')
            if name == 'io.StringIO' and value is not None and not isinstance(value, str): raise LoadError('StringIO requires text')
            return Memory(value or (b'' if name == 'io.BytesIO' else ''))
        if name in ('pickle.Unpickler', 'pickle.Pickler'):
            return Instance(name, args, dict(kw))
        if name in ('threading.Thread', 'threading.Timer', 'multiprocessing.Process', 'multiprocessing.context.Process'):
            attrs = dict(kw)
            if name == 'threading.Timer':
                attrs.update({'function': args[1] if len(args) > 1 else kw.get('function'),
                              'args': args[2] if len(args) > 2 else kw.get('args', ()),
                              'kwargs': args[3] if len(args) > 3 else kw.get('kwargs', {})})
            else:
                for i, key in enumerate(('group', 'target', 'name', 'args', 'kwargs')):
                    if len(args) > i: attrs[key] = args[i]
            if name in ('threading.Thread', 'multiprocessing.Process', 'multiprocessing.context.Process'):
                attrs = {'_' + k: v for k, v in attrs.items() if k in ('target', 'args', 'kwargs', 'name')}
            # Timer stores function/args/kwargs publicly and Thread.run uses
            # _target/_args/_kwargs. Do not expose scanner-only state as attrs.
            return Instance(name, args, attrs)
        if name in ('_thread.start_new_thread', '_thread.start_new', 'threading._start_new_thread'):
            return self.call(args[0], args[1], args[2] if len(args) > 2 else {})
        if name in ('concurrent.futures.ThreadPoolExecutor', 'concurrent.futures.ProcessPoolExecutor'):
            return Instance(name, args, dict(kw))
        if name in ('bdb.Bdb.runcall', 'pdb.Pdb.runcall'):
            return self.call(args[1], args[2:], kw)
        if name in ('pdb.runcall', 'contextlib.ContextDecorator.__call__', 'operator.call'):
            return self.call(args[0], args[1:], kw)
        if name == 'logging.config.dictConfig': return self.logging_config(args[0])
        if name in ('builtins.map', 'builtins.filter', 'builtins.zip', 'builtins.enumerate', 'builtins.reversed'):
            kind = name.split('.')[-1]
            if kind == 'map' and len(args) < 2 or kind == 'filter' and len(args) != 2 or kind == 'reversed' and len(args) != 1 or kind == 'enumerate' and not 1 <= len(args) <= 2:
                raise LoadError('invalid iterator argument count')
            if kind == 'zip' and any(k != 'strict' for k in kw): raise LoadError('unexpected zip keyword')
            if kind == 'enumerate' and any(k != 'start' for k in kw): raise LoadError('unexpected enumerate keyword')
            if kind in ('map', 'filter', 'reversed') and kw: raise LoadError('iterator does not accept keywords')
            if kind == 'reversed':
                if type(args[0]) not in (list, tuple, str, bytes, bytearray, range): raise LoadError('object is not reversible')
                return Lazy('reversed_sequence', (args[0], len(args[0])), keywords=kw)
            sources = args[1:] if kind in ('map', 'filter') else args[:1] if kind == 'enumerate' else args
            return Lazy(kind, args, sources=[self.iterator(v) for v in sources], keywords=kw)
        if name == 'itertools.chain.from_iterable':
            return Lazy('chain.from_iterable', args, sources=[self.iterator(args[0]), None])
        if name.startswith('itertools.'):
            kind = name.split('.')[-1]
            if kind in ('starmap', 'filterfalse', 'takewhile', 'dropwhile', 'repeat', 'count', 'chain', 'islice', 'cycle'):
                if kind == 'repeat' and 'times' in kw and len(args) == 1: args += (kw['times'],)
                if kind == 'repeat' and (not args or len(args) > 2 or len(args) == 2 and not isinstance(args[1], int)):
                    raise LoadError('invalid repeat arguments')
                if kind == 'count':
                    args = (args[0] if args else kw.get('start', 0), args[1] if len(args) > 1 else kw.get('step', 1))
                    if not all(isinstance(a, (int, float, complex, Instance)) for a in args): raise LoadError('count arguments must be numeric')
                if kind == 'islice':
                    if kw or not 2 <= len(args) <= 4: raise LoadError('invalid islice arguments')
                    indices = (0, args[1], 1) if len(args) == 2 else (args[1] or 0, args[2], args[3] if len(args) > 3 else 1)
                    if any(v is not None and (not isinstance(v, int) or v < 0) for v in indices) or indices[2] == 0:
                        raise LoadError('invalid islice indices')
                obj = Lazy(kind, args)
                if kind in ('starmap', 'filterfalse', 'takewhile', 'dropwhile'): obj.sources = [self.iterator(args[1])]
                if kind == 'islice': obj.sources = [self.iterator(args[0])]
                if kind == 'cycle': obj.sources = [self.iterator(args[0]), []]
                return obj
            if kind == 'tee':
                n = args[1] if len(args) > 1 else kw.get('n', 2)
                if not isinstance(n, int) or n < 0: raise LoadError('invalid tee count')
                if n > MAX_ITEMS: raise AnalysisLimit('tee count limit exceeded')
                state = TeeState(args[0], iterator=self.iterator(args[0]))
                return tuple(Lazy('tee', (state,)) for _ in range(n))
        if name == 'builtins.iter':
            return Lazy('calliter', args) if len(args) == 2 else self.iterator(args[0])
        if name == 'builtins.next':
            value = self.next(args[0])
            if value is END:
                if len(args) > 1: return args[1]
                raise LoadError('iterator exhausted')
            return value
        if name in ('builtins.list', 'builtins.tuple', 'builtins.set', 'builtins.frozenset', 'collections.deque'):
            if name != 'collections.deque' and kw: raise LoadError('container does not accept keyword arguments')
            maxlen = args[1] if name == 'collections.deque' and len(args) > 1 else kw.get('maxlen')
            if maxlen is not None and (not isinstance(maxlen, int) or maxlen < 0): raise LoadError('invalid deque maxlen')
            if name in ('builtins.set', 'builtins.frozenset'):
                if args and type(args[0]) in (set, frozenset):
                    if name.endswith('frozenset') and type(args[0]) is frozenset: return args[0]
                    result = frozenset(args[0]) if name.endswith('frozenset') else set(args[0])
                    self.copy_hashes(args[0], result)
                    return result
                values = set()
                if args:
                    source = self.iterator(args[0])
                    for _ in range(MAX_ITEMS + 1):
                        value = self.next(source)
                        if value is END:
                            if name.endswith('frozenset'):
                                result = frozenset(values)
                                self.copy_hashes(values, result)
                                return result
                            return values
                        self.set_add(values, value)
                    raise AnalysisLimit('set consumption limit exceeded')
                return frozenset() if name.endswith('frozenset') else values
            values = self.consume(args[0]) if args else []
            if name == 'collections.deque' and maxlen is not None: values = values[-maxlen:] if maxlen else []
            if name == 'builtins.tuple': return tuple(values)
            return values
        if name in ('builtins.dict', 'collections.OrderedDict', 'collections.defaultdict', 'collections.Counter'):
            factory = args[0] if name == 'collections.defaultdict' and args else None
            if name == 'collections.defaultdict': args = args[1:]
            values = args[0] if args else {}
            if isinstance(values, Namespace):
                if hasattr(values.target, 'attrs'): values = self.namespace_data(values.target)
                else: values = self.modules
            if isinstance(values, Instance) and values.name in ('collections.OrderedDict', 'collections.defaultdict', 'collections.Counter'):
                values = values.data or {}
            if name == 'collections.Counter' and values is None: values = {}
            if isinstance(values, dict):
                data = dict(values)
                self.copy_hashes(values, data)
                if name != 'collections.Counter':
                    for key, val in kw.items(): self.dict_set(data, key, val)
            else:
                data = {}
                source = self.iterator(values)
                for _ in range(MAX_ITEMS + 1):
                    item = self.next(source)
                    if item is END: break
                    if name == 'collections.Counter': self.dict_set(data, item, self.dict_get(data, item, 0) + 1)
                    else:
                        pair = self.consume(item)
                        if len(pair) != 2: raise LoadError('dictionary item must have two elements')
                        self.dict_set(data, pair[0], pair[1])
                else: raise AnalysisLimit('dictionary consumption limit exceeded')
                if name != 'collections.Counter':
                    for key, val in kw.items(): self.dict_set(data, key, val)
            if name == 'collections.Counter':
                for k, v in kw.items(): self.dict_set(data, k, self.dict_get(data, k, 0) + v)
            if name == 'builtins.dict': return data
            return Instance(name, attrs={'default_factory': factory} if name == 'collections.defaultdict' else {}, data=data)
        if name in ('builtins.min', 'builtins.max'):
            source = self.iterator(args[0]) if len(args) == 1 else self.iterator(list(args))
            key = kw.get('key')
            best = self.next(source)
            if best is END:
                if 'default' in kw: return kw['default']
                raise LoadError('empty sequence')
            bestkey = self.call(key, (best,)) if key is not None else best
            for _ in range(MAX_ITEMS + 1):
                value = self.next(source)
                if value is END: return best
                valuekey = self.call(key, (value,)) if key is not None else value
                if concrete(valuekey) and concrete(bestkey) and (valuekey < bestkey if name.endswith('.min') else valuekey > bestkey):
                    best, bestkey = value, valuekey
            raise AnalysisLimit('min/max consumption limit exceeded')
        if name == 'builtins.sorted':
            vals = self.consume(args[0])
            key = kw.get('key')
            keys = [self.call(key, (v,)) for v in vals] if key is not None else vals
            if all(concrete(v) for v in keys):
                return [v for _, v in sorted(zip(keys, vals), key=lambda p: p[0], reverse=kw.get('reverse', False))]
            return vals
        if name == 'functools.reduce':
            values = self.iterator(args[1])
            acc = args[2] if len(args) > 2 else self.next(values)
            if acc is END: raise LoadError('empty reduction')
            while True:
                value = self.next(values)
                if value is END: return acc
                acc = self.call(args[0], (acc, value))
        if name in ('builtins.any', 'builtins.all', 'builtins.sum'):
            it = self.iterator(args[0])
            acc = args[1] if len(args) > 1 else 0
            while True:
                value = self.next(it)
                if value is END: return acc if name.endswith('.sum') else name.endswith('.all')
                if name.endswith('.sum'):
                    acc = acc + value if concrete(acc) and concrete(value) else UNKNOWN
                elif concrete(value):
                    if name.endswith('.any') and value: return True
                    if name.endswith('.all') and not value: return False
        if name == 'operator.getitem': return self.getitem(args[0], args[1])
        if name == 'operator.setitem': return self.setitem(args[0], args[1], args[2])
        if name == 'operator.delitem': return self.delitem(args[0], args[1])
        if name == 'operator.methodcaller.__call__': return self.call(args[0], args[1:], kw)
        if name in ('builtins.bytes', 'builtins.bytearray', 'builtins.memoryview'):
            if name == 'builtins.memoryview' and (kw or len(args) != 1): raise LoadError('memoryview requires one argument')
            if len(args) > 3 or any(k not in ('source', 'encoding', 'errors') for k in kw): raise LoadError('invalid byte constructor arguments')
            value = args[0] if args else kw.get('source', b'')
            encoding = args[1] if len(args) > 1 else kw.get('encoding')
            errors = args[2] if len(args) > 2 else kw.get('errors', 'strict')
            if encoding is not None:
                if not isinstance(value, str): raise LoadError('encoding requires a text source')
                encoded = self.decode_call('codecs.encode', (value, encoding, errors), {})
                return bytearray(encoded) if name.endswith('bytearray') else encoded
            if isinstance(value, str): raise LoadError('text byte source requires an encoding')
            if isinstance(value, (Lazy, Instance, DynamicInstance)):
                source = self.iterator(value)
                result = bytearray()
                for _ in range(MAX_ITEMS + 1):
                    item = self.next(source)
                    if item is END:
                        return result if name.endswith('bytearray') else bytes(result)
                    if not isinstance(item, int) or not 0 <= item <= 255: raise LoadError('byte value must be an integer in range(256)')
                    result.append(item)
                raise AnalysisLimit('byte iterator consumption limit exceeded')
            if isinstance(value, int) and value > MAX_BYTES: raise AnalysisLimit('byte allocation limit exceeded')
            if not concrete(value): return UNKNOWN
            if name == 'builtins.memoryview': return value
            constructor = bytearray if name.endswith('.bytearray') else bytes
            return constructor(value, *args[1:])
        if name == 'builtins.bool':
            if kw or len(args) > 1: raise LoadError('invalid bool arguments')
            return self.truth(args[0]) if args else False
        if name in ('builtins.str', 'builtins.int', 'builtins.float', 'builtins.bool', 'builtins.complex', 'builtins.chr', 'builtins.ord', 'builtins.hex', 'builtins.oct', 'builtins.bin'):
            if args and isinstance(args[0], DynamicInstance):
                methods = {'builtins.str': ('__str__',), 'builtins.int': ('__int__', '__index__'),
                           'builtins.float': ('__float__', '__index__'), 'builtins.complex': ('__complex__', '__float__', '__index__')}
                for method in methods.get(name, ()):
                    result = self.special(args[0], method)
                    if result is not END: return result
            vals = tuple(self.path_value(a) for a in args)
            if all(concrete(v) for v in vals) and all(concrete(v) for v in kw.values()):
                funcs = {'str': str, 'int': int, 'float': float, 'bool': bool, 'complex': complex, 'chr': chr, 'ord': ord, 'hex': hex, 'oct': oct, 'bin': bin}
                return funcs[name.split('.')[-1]](*vals, **kw)
            return UNKNOWN
        if name == 'builtins.range':
            value = range(*args)
            if len(value) > MAX_ITEMS: raise AnalysisLimit('range size limit exceeded')
            return value
        if name == 'builtins.slice': return slice(*args)
        if name == 'builtins.len':
            if isinstance(args[0], DictView): return len(self.consume(args[0]))
            if type(args[0]) in (str, bytes, bytearray, tuple, list, dict, set, frozenset, range): return len(args[0])
            return UNKNOWN
        if name == 'builtins.dict.fromkeys':
            data = {}
            source = self.iterator(args[0])
            for _ in range(MAX_ITEMS + 1):
                key = self.next(source)
                if key is END: return data
                self.dict_set(data, key, args[1] if len(args) > 1 else None)
            raise AnalysisLimit('fromkeys consumption limit exceeded')
        if name == 'builtins.int.from_bytes': return int.from_bytes(*args, **kw)
        if name in ('os.fspath', 'os.fsdecode', 'os.fsencode'):
            value = self.path_value(args[0])
            if isinstance(value, DynamicInstance): return self.call(self.attr(value, '__fspath__'), ())
            if not isinstance(value, (str, bytes)): raise LoadError('not a path')
            if name == 'os.fsdecode': return value.decode('utf-8', 'surrogateescape') if isinstance(value, bytes) else value
            if name == 'os.fsencode': return value.encode('utf-8', 'surrogateescape') if isinstance(value, str) else value
            return value
        if name in ('os.getpid', 'os.getppid', 'os.getuid', 'os.getgid', 'os.geteuid', 'os.getegid', 'os.cpu_count'): return 1
        if name in ('os.getcwd', 'os.getcwdb'): return b'/' if name.endswith('getcwdb') else '/'
        if name == 'builtins.hash': return self.hash_value(args[0])
        if name == 'builtins.object.__hash__': return object.__hash__(args[0])
        if name in ('operator.eq', 'operator.ne'):
            result = self.equal(args[0], args[1])
            return not result if name.endswith('.ne') else result
        if name in ('builtins.abs', 'builtins.round', 'builtins.divmod', 'builtins.pow', 'builtins.hash', 'builtins.repr', 'builtins.ascii', 'builtins.format'):
            functions = {'abs': abs, 'round': round, 'divmod': divmod, 'pow': pow, 'hash': hash, 'repr': repr, 'ascii': ascii, 'format': format}
            if all(concrete(v) for v in args):
                if name == 'builtins.pow' and len(args) == 2 and isinstance(args[1], int) and (args[1] > 100_000 or isinstance(args[0], int) and abs(args[0]).bit_length() * max(0, args[1]) > MAX_BYTES * 8): raise AnalysisLimit('integer power limit exceeded')
                if name == 'builtins.format' and len(args) > 1: self.check_product('format', (args[1],))
                return functions[name.split('.')[-1]](*args, **kw)
            return UNKNOWN
        if name in ('sys.intern', 'builtins.id'): return args[0] if name == 'sys.intern' else 1
        if name.startswith(('pathlib.Path', 'pathlib.PosixPath', 'pathlib.WindowsPath', 'pathlib.PurePath', 'pathlib.PurePosixPath', 'pathlib.PureWindowsPath')):
            if name.rsplit('.', 1)[-1] in ('Path', 'PosixPath', 'WindowsPath', 'PurePath', 'PurePosixPath', 'PureWindowsPath'):
                return Instance('pathlib.Path', args)
        if name.startswith('operator.'):
            leaf = name.split('.')[-1]
            ops = {'add': operator.add, 'concat': operator.concat, 'iadd': operator.add,
                   'iconcat': operator.concat, 'mul': operator.mul, 'imul': operator.mul,
                   'sub': operator.sub, 'truediv': operator.truediv, 'floordiv': operator.floordiv,
                   'mod': operator.mod, 'neg': operator.neg, 'pos': operator.pos,
                   'not_': operator.not_, 'truth': bool, 'eq': operator.eq, 'ne': operator.ne,
                   'lt': operator.lt, 'le': operator.le, 'gt': operator.gt, 'ge': operator.ge,
                   'and_': operator.and_, 'or_': operator.or_, 'xor': operator.xor,
                   'index': operator.index, 'contains': operator.contains}
            if leaf in ops and all(concrete(v) for v in args):
                self.check_product(leaf, args)
                return ops[leaf](*args)
        decoded = self.decode_call(name, args, kw)
        if decoded is not END: return decoded
        if name.startswith(('posixpath.', 'ntpath.', 'os.path.')):
            leaf = name.split('.')[-1]
            if leaf in ('join', 'basename', 'dirname', 'split', 'splitext', 'normpath', 'normcase', 'isabs', 'commonprefix', 'commonpath', 'relpath', 'abspath') and all(concrete(a) for a in args):
                if leaf == 'abspath': return posixpath.normpath(args[0])
                return getattr(posixpath, leaf)(*args)
        # Builtin descriptors such as bytes.fromhex or dict.get, resolved by
        # GLOBAL at protocol 4, are handled as methods on inert data.
        if name.startswith('builtins.') and name.count('.') >= 2:
            cls, method = name[len('builtins.'):].split('.', 1)
            if method == 'fromhex' and cls in ('bytes', 'bytearray'):
                return self.bound(b'' if cls == 'bytes' else bytearray(), method, args, kw)
            if args and cls in ('str', 'bytes', 'bytearray', 'dict', 'list', 'tuple', 'set'):
                return self.bound(args[0], method, args[1:], kw)
        if name in ('numpy.core.multiarray._reconstruct', 'numpy._core.multiarray._reconstruct'):
            return Instance('numpy.ndarray', args)
        if name in ('numpy.core.multiarray.scalar', 'numpy._core.multiarray.scalar'):
            return Instance('numpy.scalar', args)
        if name == 'builtins.object': return Instance(name, args)
        # Scientific classes and pure numeric reducers retain their identity
        # and state. Unmodelled executable behavior receives the contract's
        # unsupported verdict, rather than silently becoming clean.
        if name in PURE_OPAQUE_NAMES or name.startswith(PURE_OPAQUE_PREFIXES):
            pending = list(args) + list(kw.values())
            seen = set()
            while pending:
                value = pending.pop()
                if isinstance(value, DynamicInstance): raise AnalysisLimit('dynamic argument to an opaque reducer')
                if isinstance(value, (tuple, list, dict)) and id(value) not in seen:
                    seen.add(id(value))
                    pending.extend(value.values() if isinstance(value, dict) else value)
            return Instance(name, args, dict(kw))
        if name.startswith('sklearn.') and not name.startswith(('sklearn.datasets.', 'sklearn.externals.')):
            return Instance(name, args, dict(kw))
        if name.startswith('builtins.') and self.is_class(func) and name.rsplit('.', 1)[-1].endswith(('Error', 'Warning', 'Exception')):
            return Instance(name, args, dict(kw))
        self.finding('unsupported', 'no static call model for ' + name)
        return UNKNOWN

    def bound(self, obj, method, args, kw):
        if method == '__getattribute__': return self.attr(obj, args[0])
        if method == '__setattr__':
            if hasattr(obj, 'attrs'): obj.attrs[args[0]] = args[1]
            return None
        if method == '__delattr__':
            self.delete_attr(obj, args[0])
            return None
        if isinstance(obj, Partial) and method == '__setstate__':
            self.build(obj, args[0]); return None
        if method == '__call__': return self.call(obj, args, kw)
        if method == '__getitem__': return self.getitem(obj, args[0])
        if method == '__setitem__': return self.setitem(obj, args[0], args[1])
        if method == '__iter__': return self.iterator(obj)
        if method == '__next__':
            value = self.next(obj)
            if value is END: raise LoadError('iterator exhausted')
            return value
        if isinstance(obj, Memory):
            if method == '__setstate__':
                state = args[0]
                if not isinstance(state, tuple) or len(state) not in (3, 4): raise LoadError('invalid in-memory stream state')
                obj.data = state[0]
                obj.pos = state[1] if len(state) == 3 else state[2]
                obj.attrs.update(state[-1] or {})
                return None
            if method == '__getstate__': return (obj.data, obj.pos, obj.attrs or None)
            if method == 'getbuffer':
                if not isinstance(obj.data, bytearray): obj.data = bytearray(obj.data)
                return obj.data
            if method == 'truncate':
                size = args[0] if args else obj.pos
                if size < 0: raise LoadError('negative truncate size')
                obj.data = obj.data[:size]
                return size
            if method in ('read', 'read1', 'readline', 'getvalue'):
                if isinstance(obj.data, Unknown): return UNKNOWN
                if method == 'getvalue': return obj.data
                n = args[0] if args else -1
                end = len(obj.data) if n is None or n < 0 else min(len(obj.data), obj.pos + n)
                if method == 'readline':
                    newline = b'\n' if isinstance(obj.data, (bytes, bytearray)) else '\n'
                    ix = obj.data.find(newline, obj.pos, end)
                    if ix >= 0: end = ix + 1
                value = obj.data[obj.pos:end]
                obj.pos = end
                return value
            if method == 'seek':
                offset, whence = args[0], args[1] if len(args) > 1 else 0
                obj.pos = offset + (obj.pos if whence == 1 else len(obj.data) if whence == 2 else 0)
                return obj.pos
            if method == 'tell': return obj.pos
            if method == 'write':
                data = args[0]
                obj.data = obj.data[:obj.pos] + data + obj.data[obj.pos + len(data):]
                obj.pos += len(data)
                self.budget.add_bytes(len(data))
                return len(data)
            if method == 'close': return None
        if isinstance(obj, Namespace):
            if method in ('get', 'pop'):
                missing = object()
                value = self.getitem(obj, args[0], missing)
                if value is missing:
                    if len(args) > 1: return args[1]
                    if method == 'get': return None
                    raise LoadError('dictionary key does not exist')
                if method == 'pop': self.delitem(obj, args[0])
                return value
            if method == 'setdefault':
                try: value = self.getitem(obj, args[0], END)
                except LoadError: value = END
                if value is END:
                    value = args[1] if len(args) > 1 else None
                    self.setitem(obj, args[0], value)
                return value
            if method == 'update':
                for key, value in dict(args[0] if args else {}, **kw).items(): self.setitem(obj, key, value)
                return None
            if method in ('keys', 'values', 'items'):
                if args or kw: raise LoadError('dictionary view method takes no arguments')
                return DictView(obj, method)
            if method == 'copy':
                data = self.namespace_data(obj.target) if hasattr(obj.target, 'attrs') else self.modules
                return data.copy() if method == 'copy' else list(getattr(data, method)())
        if isinstance(obj, Instance):
            name = obj.name
            candidate = self.attr(self.ref(name), method, None)
            if isinstance(candidate, Ref): self.bind_signature(candidate.name, (obj,) + args, kw)
            if name in ('concurrent.futures.Future', 'concurrent.futures._base.Future'):
                if method == 'result': return obj.args[0] if obj.args else UNKNOWN
                if method == 'exception': return None
                if method in ('done', 'cancelled', 'running'): return method == 'done'
            if name in ('collections.OrderedDict', 'collections.defaultdict', 'collections.Counter') and method in ('get', 'pop', 'popitem', 'setdefault', 'keys', 'values', 'items', 'copy', 'clear', 'update'):
                if obj.data is None: obj.data = {}
                return self.bound(obj.data, method, args, kw)
            if name == 'timeit.Timer' and method in ('timeit', 'repeat', 'autorange'):
                for source in (obj.attrs.get('setup', END), obj.attrs.get('stmt', END)):
                    if source is not END: self.call(source, ())
                return 0
            if name == 'code' and method == 'replace': return Instance('code', obj.args)
            if name == 'numpy.NpzFile' and method == '__getitem__':
                from modelgate.dispatch import _npy
                import zipfile
                with zipfile.ZipFile(io.BytesIO(obj.args[0])) as zf:
                    member = args[0] if args[0] in zf.namelist() else args[0] + '.npy'
                    info = zf.getinfo(member)
                    self.budget.add_bytes(info.file_size)
                    return _npy(zf.read(info), obj.args[1] + '!' + member, self, obj.attrs.get('allow_pickle', False))
            if name == 'pickle.Unpickler' and method == 'load':
                if args or kw: raise LoadError('Unpickler.load takes no arguments')
                if not obj.args or not isinstance(obj.args[0], Memory): raise AnalysisLimit('opaque Unpickler input')
                from modelgate.pickle_scan import Interpreter, MemoryReader
                if obj.data is None:
                    obj.data = Interpreter(self, MemoryReader(self, obj.args[0]), self.location + ' -> Unpickler.load')
                obj.data.persistent_callback = obj.attrs.get('persistent_load')
                return obj.data.one(reuse=True)
            if name == 'pathlib.Path' and method in ('__str__', '__fspath__', 'as_posix'):
                return self.path_value(obj)
            if name == 'pathlib.Path' and method in ('joinpath', '__truediv__', 'with_name', 'with_suffix'):
                path = self.path_value(obj)
                if method == 'with_name': return Instance(name, (posixpath.join(posixpath.dirname(path), args[0]),))
                if method == 'with_suffix': return Instance(name, (posixpath.splitext(path)[0] + args[0],))
                return Instance(name, (path,) + args)
            if name in ('threading.Thread', 'threading.Timer', 'multiprocessing.Process', 'multiprocessing.context.Process') and method in ('start', 'run'):
                if args or kw: raise LoadError('thread/process start and run take no arguments')
                if method == 'start': obj.data = {'started': True}
                if name.startswith('multiprocessing') and method == 'start': self.finding('process', 'starts ' + name)
                target = obj.attrs.get('function') if name == 'threading.Timer' else obj.attrs.get('_target')
                if target is not None:
                    return self.call(target, tuple((obj.attrs.get('args', ()) if name == 'threading.Timer' else obj.attrs.get('_args', ())) or ()), (obj.attrs.get('kwargs', {}) if name == 'threading.Timer' else obj.attrs.get('_kwargs', {})) or {})
                return None
            if name in ('threading.Thread', 'threading.Timer') and method == 'join':
                if not isinstance(obj.data, dict) or not obj.data.get('started'): raise LoadError('cannot join a thread before starting it')
                return None
            if name in ('concurrent.futures.ThreadPoolExecutor', 'concurrent.futures.ProcessPoolExecutor') and method in ('shutdown', '__enter__', '__exit__'):
                return obj if method == '__enter__' else None
            if name in ('concurrent.futures.ThreadPoolExecutor', 'concurrent.futures.ProcessPoolExecutor') and method in ('submit', 'map'):
                if not args: raise LoadError('executor requires a function')
                if 'ProcessPool' in name: self.finding('process', 'starts process executor')
                if method == 'submit': return Instance('concurrent.futures._base.Future', (self.call(args[0], args[1:], kw),))
                return Lazy('values', (self.consume(Lazy('map', args)),))
            if name in ('xmlrpc.client.ServerProxy', 'xmlrpc.client._Method') and not method.startswith('_'):
                self.finding('network', 'calls XML-RPC method ' + method)
                return UNKNOWN
            if method == '__init__':
                return self.call(self.ref(name), args, kw)
            if method == 'runcall' and name in ('bdb.Bdb', 'pdb.Pdb'):
                return self.call(args[0], args[1:], kw)
            target = self.attr(self.ref(name), method)
            return self.call(target, (obj,) + args, kw)
        if isinstance(obj, bytearray) and method in ('extend', 'append'):
            if method == 'extend': obj.extend(self.call(self.ref('builtins.bytearray'), args))
            else: obj.append(args[0])
            return None
        if type(obj) in (dict, list, set):
            if method == 'copy':
                result = obj.copy()
                self.copy_hashes(obj, result)
                return result
            if isinstance(obj, dict) and method in ('get', 'pop', 'setdefault'):
                missing = object()
                value = self.dict_get(obj, args[0], missing)
                if value is missing:
                    if method == 'setdefault':
                        value = args[1] if len(args) > 1 else None
                        self.dict_set(obj, args[0], value)
                        return value
                    if len(args) > 1: return args[1]
                    if method == 'get': return None
                    raise LoadError('dictionary key does not exist')
                if method == 'pop': self.delitem(obj, args[0])
                return value
            if isinstance(obj, dict) and method == 'update':
                source = args[0] if args else {}
                if not isinstance(source, dict): source = self.call(self.ref('builtins.dict'), (source,))
                for key, val in source.items(): self.dict_set(obj, key, val)
                for key, val in kw.items(): self.dict_set(obj, key, val)
                return None
            if isinstance(obj, set) and method == 'add':
                self.set_add(obj, args[0]); return None
            if isinstance(obj, dict) and method == 'fromkeys':
                return self.call(self.ref('builtins.dict.fromkeys'), args, kw)
            if isinstance(obj, dict) and method in ('keys', 'values', 'items'):
                if args or kw: raise LoadError('dictionary view method takes no arguments')
                return DictView(obj, method)
            if method == 'update' and args and isinstance(args[0], Lazy):
                constructor = 'builtins.dict' if isinstance(obj, dict) else 'builtins.set'
                obj.update(self.call(self.ref(constructor), args))
                return None
            if method in ('get', 'pop', 'setdefault', 'keys', 'values', 'items', 'copy', 'clear', 'update', 'append', 'extend', 'insert', 'remove', 'reverse', 'add', 'discard', 'intersection', 'union', 'difference'):
                if method in ('extend', 'update', 'intersection', 'union', 'difference') and args and isinstance(args[0], Lazy):
                    args = (self.consume(args[0]),) + args[1:]
                result = getattr(obj, method)(*args, **kw)
                return list(result) if method in ('keys', 'values', 'items') else result
            if method == 'sort' and isinstance(obj, list):
                obj[:] = self.call(self.ref('builtins.sorted'), (obj,), kw)
                return None
        if isinstance(obj, (bytes, bytearray)) and method in ('tobytes', 'tolist'):
            return bytes(obj) if method == 'tobytes' else list(obj)
        if type(obj) in (str, bytes, bytearray, tuple, int, float, bool) and method in PURE_METHODS:
            if method == 'join' and args and isinstance(args[0], (Lazy, Instance, DynamicInstance)):
                args = (self.consume(args[0]),) + args[1:]
            if all(concrete(v) for v in args) and all(concrete(v) for v in kw.values()):
                self.check_product(method, (obj,) + args)
                result = getattr(obj, method)(*args, **kw)
                if isinstance(result, (bytes, bytearray, str)):
                    self.budget.add_bytes(len(result))
                return result
            return UNKNOWN
        return UNKNOWN

    def check_product(self, operation, args):
        if operation in ('add', 'iadd', 'concat', 'iconcat', '__add__', '__radd__') and len(args) == 2:
            if isinstance(args[0], (str, bytes, bytearray, list, tuple)) and isinstance(args[1], type(args[0])):
                if len(args[0]) + len(args[1]) > MAX_BYTES: raise AnalysisLimit('concatenation size limit exceeded')
        if operation == 'join' and len(args) == 2 and isinstance(args[1], (list, tuple)):
            sizes = [len(v) for v in args[1] if isinstance(v, (str, bytes, bytearray))]
            if sum(sizes) + len(args[0]) * max(0, len(sizes) - 1) > MAX_BYTES:
                raise AnalysisLimit('joined data size limit exceeded')
        if operation == 'replace' and len(args) >= 3:
            source, old, new = args[:3]
            if isinstance(source, (str, bytes, bytearray)) and isinstance(old, type(source)) and isinstance(new, type(source)):
                if (len(source) + 1) * max(1, len(new)) > MAX_BYTES:
                    raise AnalysisLimit('replacement size limit exceeded')
        if operation in ('format', 'format_map'):
            import string
            fmt = args[0]
            if isinstance(fmt, str):
                fields = list(string.Formatter().parse(fmt))
                if len(fields) > MAX_ITEMS: raise AnalysisLimit('format field count limit exceeded')
                sizes = [len(v) for v in args[1:] if isinstance(v, (str, bytes, bytearray))]
                maxarg = max(sizes, default=1024)
                estimate = len(fmt)
                for _, field, spec, _ in fields:
                    if field is None: continue
                    import re
                    widths = [int(n) for n in re.findall(r'\d+', spec)]
                    if '{' in spec:
                        widths += [abs(a) for a in args[1:] if isinstance(a, int)]
                    estimate += max([maxarg] + widths)
                    if estimate > MAX_BYTES: raise AnalysisLimit('formatted data size limit exceeded')
            # Width/precision may otherwise request gigabytes in one native
            # formatting call, before the output-size budget can be checked.
            import re
            if any(int(n) > MAX_BYTES // 10 for n in re.findall(r'\d{7,}', args[0])):
                raise AnalysisLimit('format width limit exceeded')
        if operation in ('mul', 'imul', '__mul__', '__rmul__') and len(args) == 2:
            a, b = args
            if isinstance(a, int): a, b = b, a
            if isinstance(a, (str, bytes, bytearray, list, tuple)) and isinstance(b, int) and len(a) * b > MAX_BYTES:
                raise AnalysisLimit('data multiplication size limit exceeded')

    def as_bytes(self, value):
        if isinstance(value, Unknown): raise AnalysisLimit('cannot resolve embedded bytes')
        if isinstance(value, (bytes, bytearray)): return bytes(value)
        if isinstance(value, str): return value.encode('latin1')
        raise LoadError('expected embedded bytes')

    def decode_call(self, name, args, kw):
        if name in ('_codecs.encode', 'codecs.encode', '_codecs.decode', 'codecs.decode'):
            value = args[0]
            encoding = args[1] if len(args) > 1 else kw.get('encoding', 'utf-8')
            errors = args[2] if len(args) > 2 else kw.get('errors', 'strict')
            allowed = {'latin1', 'latin-1', 'iso8859-1', 'utf-8', 'utf8', 'ascii', 'unicode_escape', 'raw_unicode_escape', 'base64', 'base64_codec', 'hex', 'hex_codec', 'zlib', 'zlib_codec', 'bz2', 'bz2_codec', 'rot_13', 'rot13', 'utf-16', 'utf-16-le', 'utf-16-be', 'utf-32', 'utf-32-le', 'utf-32-be'}
            if encoding.lower().replace('_', '-') not in {e.replace('_', '-') for e in allowed}:
                raise AnalysisLimit('unsupported data codec')
            if encoding in ('zlib', 'zlib_codec', 'bz2', 'bz2_codec') and name.endswith('decode'):
                return self.decompress('bz2' if encoding.startswith('bz2') else 'zlib', self.as_bytes(value))
            result = (codecs.encode if name.endswith('encode') else codecs.decode)(value, encoding, errors)
            self.budget.add_bytes(len(result))
            return result
        if name.startswith('base64.'):
            leaf = name.split('.')[-1]
            funcs = {n: getattr(base64, n) for n in ('b64decode', 'standard_b64decode', 'urlsafe_b64decode', 'b85decode', 'a85decode', 'b32decode', 'b16decode', 'decodebytes', 'b64encode', 'standard_b64encode', 'urlsafe_b64encode', 'b85encode', 'a85encode', 'b32encode', 'b16encode', 'encodebytes')}
            if leaf in funcs:
                result = funcs[leaf](*args, **kw)
                self.budget.add_bytes(len(result))
                return result
        if name.startswith('binascii.'):
            leaf = name.split('.')[-1]
            if leaf in ('unhexlify', 'a2b_hex', 'a2b_base64', 'b2a_base64', 'hexlify', 'b2a_hex', 'a2b_uu', 'b2a_uu', 'a2b_qp', 'b2a_qp'):
                result = getattr(binascii, leaf)(*args, **kw)
                self.budget.add_bytes(len(result))
                return result
        if name in ('zlib.decompress', 'gzip.decompress', 'bz2.decompress', 'lzma.decompress'):
            return self.decompress(name.split('.')[0], self.as_bytes(args[0]), args[1] if len(args) > 1 else None)
        if name in ('bytes.fromhex', 'bytearray.fromhex'):
            return bytes.fromhex(args[0])
        if name in ('json.loads', 'ast.literal_eval'):
            import ast
            import json
            return (json.loads if name == 'json.loads' else ast.literal_eval)(args[0])
        return END

    def decompress(self, format, data, option=None):
        self.budget.tick()
        if format in ('gzip', 'zlib'):
            decoder = zlib.decompressobj(31 if format == 'gzip' else option if option is not None else 15)
            result = decoder.decompress(data, MAX_BYTES + 1)
            rest = decoder.unused_data
            if len(result) > MAX_BYTES or decoder.unconsumed_tail:
                raise AnalysisLimit('decompression size limit exceeded')
            if not decoder.eof: raise LoadError('truncated compressed data')
        else:
            decoder = bz2.BZ2Decompressor() if format == 'bz2' else lzma.LZMADecompressor(memlimit=MAX_BYTES)
            result = decoder.decompress(data, max_length=MAX_BYTES + 1)
            rest = decoder.unused_data
            if len(result) > MAX_BYTES or not decoder.eof:
                raise AnalysisLimit('decompression size limit exceeded')
        self.budget.add_bytes(len(result))
        # gzip, bz2 and xz loaders read concatenated compressed members.
        if format == 'gzip': rest = rest.lstrip(b'\x00')
        if rest and format != 'zlib':
            try: result += self.decompress(format, rest)
            except (OSError, EOFError, LoadError, lzma.LZMAError):
                if format == 'gzip': raise
                # bz2.decompress and lzma.decompress ignore non-stream data
                # after at least one complete compressed member.

        return result

    def nested(self, name, args, kw):
        if not args:
            value = kw.get('file', kw.get('filename', kw.get('filepath_or_buffer', END)))
            if value is END: raise LoadError('loader requires input')
        else:
            value = args[0]
        embedded_bytes = name.endswith('loads') or name == 'torch.storage._load_from_bytes'
        if embedded_bytes:
            data = self.as_bytes(value)
        elif isinstance(value, Memory):
            if isinstance(value.data, Unknown): raise AnalysisLimit('cannot resolve embedded stream data')
            data = self.as_bytes(value.data[value.pos:])
        else:
            self.finding('file-access', 'nested loader ' + name + ' reads an external path')
            return UNKNOWN
        self.depth += 1
        oldloc = self.location
        try:
            if self.depth > 64: raise AnalysisLimit('nested deserialization limit exceeded')
            if name.startswith('pickle.'):
                from modelgate.pickle_scan import Interpreter
                vm = Interpreter(self, data, oldloc + ' -> ' + name, persistent=False)
                result = vm.one()
                if isinstance(value, Memory): value.pos += vm.pos
                return result
            if self.embedded:
                return self.embedded(name, data, oldloc + ' -> ' + name, kw)
            self.finding('unsupported', 'no static model for nested loader ' + name)
            return UNKNOWN
        finally:
            self.location = oldloc
            self.depth -= 1

    def logging_config(self, config):
        if not isinstance(config, dict): raise LoadError('logging configuration must be a dictionary')
        for section in ('formatters', 'filters', 'handlers'):
            for cfg in config.get(section, {}).values():
                if not isinstance(cfg, dict): continue
                target = cfg.get('()', cfg.get('class'))
                if target is None: continue
                if isinstance(target, str): target = self.call(self.ref('pkgutil.resolve_name'), (target,))
                opts = {k: v for k, v in cfg.items() if k not in ('()', 'class', 'level', 'formatter', 'filters', '.')}
                self.call(target, (), opts)
        return None

    def marshal_data(self, data):
        # A bounded reader for ordinary marshal data; code remains a symbol.
        stream = io.BytesIO(data)
        refs = []
        def read(n):
            value = stream.read(n)
            if len(value) != n: raise LoadError('truncated marshal data')
            return value
        def integer(): return struct.unpack('<i', read(4))[0]
        def value(depth=0):
            self.budget.tick()
            if depth > 100: raise AnalysisLimit('marshal depth limit exceeded')
            tag = read(1)[0]
            code = chr(tag & 127)
            refindex = None
            if tag & 128:
                refindex = len(refs)
                refs.append(UNKNOWN)
            if code == 'c': result = Instance('code', (data,))
            elif code == 'N': result = None
            elif code == 'T': result = True
            elif code == 'F': result = False
            elif code == 'i': result = integer()
            elif code == 'g': result = struct.unpack('<d', read(8))[0]
            elif code in ('s', 't', 'u', 'a', 'A', 'z', 'Z'):
                n = read(1)[0] if code in ('z', 'Z') else integer()
                if n < 0 or n > MAX_BYTES: raise LoadError('invalid marshal string size')
                result = read(n)
                if code != 's': result = result.decode('utf-8')
            elif code in ('(', ')', '[', '<', '>'):
                n = read(1)[0] if code == ')' else integer()
                if n < 0 or n > MAX_ITEMS: raise AnalysisLimit('marshal item limit exceeded')
                result = [value(depth + 1) for _ in range(n)]
                if code in ('(', ')'): result = tuple(result)
                if code in ('<', '>'): result = set(result)
            elif code == 'r': result = refs[integer()]
            else: return UNKNOWN
            if refindex is not None: refs[refindex] = result
            return result
        return value()
