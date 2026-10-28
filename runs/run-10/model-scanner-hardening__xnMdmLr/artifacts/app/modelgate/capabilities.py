"""Call effects for inert symbolic Python objects.

This is an abstract interpreter, not a restricted unpickler. In particular it
never imports an artifact-selected module, resolves a real Python callable,
unmarshals a code object, invokes a model constructor, or opens a target path.
Only explicitly implemented operations on exact, inert builtin data run here.
"""
from __future__ import annotations

import _compat_pickle
import base64
import binascii
import codecs
import copy
import io
import operator
import posixpath

from modelgate.abstract import (Context, Value, UNKNOWN, MISSING, LoadError,
                               LimitError, Detected, literal, Mapping)
from modelgate.report import Finding
from modelgate.stdlib import NAMESPACES, SYMBOLS

ALIASES = {'posix': 'os', 'nt': 'os', '_pickle': 'pickle', '_io': 'io',
           '_operator': 'operator', '_functools': 'functools', '_socket': 'socket',
           '_threading_local': 'threading', 'numpy.core': 'numpy._core',
           'numpy.core.multiarray': 'numpy._core.multiarray',
           'numpy.core.numeric': 'numpy._core.numeric'}

# Pure module re-exports and imports. Resolving these only creates symbolic
# references. Imports themselves have no effect under the registry policy.
MODULE_NAMES = set('''os sys builtins io pickle marshal types functools operator
    importlib importlib.util pkgutil shutil subprocess pathlib socket ssl pty
    runpy timeit code bdb pdb logging logging.config numpy torch joblib pandas
    base64 zlib bz2 lzma gzip codecs _codecs collections itertools threading
    multiprocessing concurrent concurrent.futures urllib urllib.request
    urllib.parse http http.client ftplib smtplib xmlrpc xmlrpc.client
    copy copyreg posixpath ntpath tempfile glob random time math re json
    numpy.lib numpy.lib.format numpy._core numpy._core.multiarray
    torch.storage torch._utils'''.split())
REEXPORTS = {
    'os.path': 'posixpath', 'os.sys': 'sys', 'pathlib.os': 'os', 'pathlib.io': 'io',
    'pathlib.sys': 'sys', 'shutil.os': 'os', 'shutil.sys': 'sys',
    'shutil.stat': 'stat', 'shutil.collections': 'collections',
    'subprocess.os': 'os', 'subprocess.sys': 'sys', 'subprocess.io': 'io',
    'subprocess._posixsubprocess': '_posixsubprocess',
    'pickle.io': 'io', 'pickle.sys': 'sys', 'pickle.codecs': 'codecs',
    'runpy.sys': 'sys', 'runpy.importlib': 'importlib',
    'timeit.itertools': 'itertools', 'timeit.sys': 'sys',
    'code.sys': 'sys', 'bdb.os': 'os', 'pdb.os': 'os', 'pdb.sys': 'sys',
    'pkgutil.sys': 'sys', 'pkgutil.importlib': 'importlib',
    'logging.os': 'os', 'logging.sys': 'sys', 'logging.threading': 'threading',
    'logging.config.os': 'os', 'logging.config.sys': 'sys',
    'urllib.request.os': 'os', 'urllib.request.socket': 'socket',
    'urllib.request.http': 'http', 'socket.os': 'os', 'socket.sys': 'sys',
    'http.client.socket': 'socket', 'ftplib.socket': 'socket',
    'smtplib.socket': 'socket', 'ssl.socket': 'socket.socket',
    'torch.serialization.pickle': 'pickle', 'torch.serialization.os': 'os',
    'numpy.lib.format.pickle': 'pickle', 'numpy.lib._npyio_impl.pickle': 'pickle',
}
CLASS_NAMES = set('''builtins.object builtins.type builtins.str builtins.bytes
    builtins.bytearray builtins.list builtins.tuple builtins.dict builtins.set
    builtins.frozenset builtins.int builtins.float builtins.bool builtins.complex
    builtins.slice builtins.range builtins.map builtins.filter builtins.zip
    builtins.enumerate builtins.reversed builtins.staticmethod builtins.classmethod
    builtins.property functools.partial functools.partialmethod
    operator.attrgetter operator.itemgetter operator.methodcaller
    io.BytesIO io.StringIO io.FileIO io.TextIOWrapper io.BufferedReader
    io.BufferedWriter io.BufferedRandom io.BufferedRWPair
    types.FunctionType types.CodeType types.MethodType types.ModuleType
    types.SimpleNamespace subprocess.Popen socket.socket socket.SocketType
    socket.socketpair http.client.HTTPConnection http.client.HTTPSConnection
    ftplib.FTP ftplib.FTP_TLS smtplib.SMTP smtplib.SMTP_SSL smtplib.LMTP
    xmlrpc.client.ServerProxy pathlib.Path pathlib.PosixPath pathlib.WindowsPath
    pathlib.PurePath pathlib.PurePosixPath pathlib.PureWindowsPath
    threading.Thread threading.Timer multiprocessing.Process
    concurrent.futures.ThreadPoolExecutor concurrent.futures.ProcessPoolExecutor
    code.InteractiveInterpreter code.InteractiveConsole bdb.Bdb pdb.Pdb timeit.Timer
    pickle.Unpickler collections.OrderedDict collections.defaultdict collections.deque
    collections.Counter collections.UserDict collections.UserList
    numpy.ndarray numpy.dtype numpy.memmap joblib.numpy_pickle.NumpyArrayWrapper
    logging.FileHandler logging.handlers.RotatingFileHandler
    logging.handlers.TimedRotatingFileHandler logging.handlers.WatchedFileHandler
    logging.handlers.SocketHandler logging.handlers.HTTPHandler logging.StreamHandler
    tempfile.TemporaryDirectory zipfile.ZipFile tarfile.TarFile gzip.GzipFile
    bz2.BZ2File lzma.LZMAFile'''.split())

EFFECTS = {}
def effects(kind, names):
    EFFECTS.update(dict.fromkeys(names.split(), kind))

effects('code-execution', '''builtins.eval builtins.exec builtins.compile
    runpy.run_path runpy.run_module runpy._run_code runpy._run_module_code
    timeit.timeit timeit.repeat timeit.Timer.timeit timeit.Timer.repeat
    timeit.Timer.autorange code.InteractiveInterpreter.runsource
    code.InteractiveInterpreter.runcode code.InteractiveConsole.push
    bdb.Bdb.run bdb.Bdb.runeval bdb.Bdb.runctx pdb.run pdb.runeval
    pdb.runctx pdb.Pdb.run pdb.Pdb.runeval pdb.Pdb.runctx
    codeop.compile_command codeop._maybe_compile
    numpy.testing._private.utils.runstring types.FunctionType
    types.CodeType''')
effects('process', '''os.system os.popen os.execv os.execvp os.execve os.execvpe
    os.execl os.execlp os.execle os.execlpe os.spawnl os.spawnle os.spawnlp
    os.spawnlpe os.spawnv os.spawnve os.spawnvp os.spawnvpe os.posix_spawn
    os.posix_spawnp os.fork os.forkpty os.startfile subprocess.Popen
    subprocess.call subprocess.run subprocess.check_call subprocess.check_output
    subprocess.getoutput subprocess.getstatusoutput pty.spawn pty.fork
    pydoc.pipepager pydoc.tempfilepager platform.popen popen2.popen2
    multiprocessing.Pool multiprocessing.pool.Pool multiprocessing.context.BaseContext.Pool
    multiprocessing.Process.start multiprocessing.context.Process.start
    _posixsubprocess.fork_exec''')
effects('file-access', '''builtins.open io.open io.open_code io.FileIO
    os.open os.fdopen os.remove os.unlink os.rename os.renames os.replace
    os.mkdir os.makedirs os.rmdir os.removedirs os.chmod os.chown os.lchown
    os.truncate os.utime os.link os.symlink os.readlink os.stat os.lstat
    os.listdir os.scandir os.walk os.fwalk os.access os.chroot os.chdir
    os.read os.write os.ftruncate os.mknod os.mkfifo
    codecs.open shutil.make_archive shutil.unpack_archive
    h5py.File shutil.copy shutil.copyfile shutil.copy2 shutil.copytree shutil.copymode
    shutil.copystat shutil.move shutil.rmtree shutil.chown shutil.disk_usage
    pathlib.Path.open pathlib.Path.read_text pathlib.Path.read_bytes
    pathlib.Path.write_text pathlib.Path.write_bytes pathlib.Path.unlink
    pathlib.Path.rename pathlib.Path.replace pathlib.Path.mkdir pathlib.Path.rmdir
    pathlib.Path.touch pathlib.Path.chmod pathlib.Path.lchmod pathlib.Path.stat
    pathlib.Path.lstat pathlib.Path.iterdir pathlib.Path.glob pathlib.Path.rglob
    pathlib.Path.resolve pathlib.Path.exists
    pathlib.Path.is_file pathlib.Path.is_dir pathlib.Path.symlink_to
    pathlib.Path.hardlink_to pathlib.Path.link_to pathlib.Path.readlink
    tempfile.mkstemp tempfile.mkdtemp tempfile.NamedTemporaryFile
    tempfile.TemporaryFile tempfile.SpooledTemporaryFile tempfile.TemporaryDirectory
    glob.glob glob.iglob os.path.exists os.path.isfile os.path.isdir
    posixpath.exists posixpath.isfile posixpath.isdir posixpath.getsize
    posixpath.getmtime posixpath.getatime posixpath.getctime
    linecache.getline linecache.getlines fileinput.input fileinput.FileInput
    sqlite3.connect shelve.open dbm.open dbm.gnu.open dbm.ndbm.open
    dbm.dumb.open logging.FileHandler logging.handlers.RotatingFileHandler
    logging.handlers.TimedRotatingFileHandler logging.handlers.WatchedFileHandler
    logging.FileHandler.emit logging.handlers.RotatingFileHandler.emit
    logging.handlers.TimedRotatingFileHandler.emit logging.handlers.WatchedFileHandler.emit
    numpy.memmap numpy.fromfile numpy.loadtxt numpy.genfromtxt numpy.savetxt
    numpy.save numpy.savez numpy.savez_compressed torch.save pickle.dump
    pandas.read_csv pandas.read_json pandas.read_parquet''')
effects('network', '''socket.create_connection socket.socket.connect
    socket.socket.connect_ex socket.socket.sendto socket.socket.sendmsg
    socket.getaddrinfo
    socket.gethostbyname socket.gethostbyname_ex socket.gethostbyaddr
    urllib.request.urlopen urllib.request.urlretrieve
    urllib.request.OpenerDirector.open http.client.HTTPConnection.connect
    http.client.HTTPConnection.request http.client.HTTPConnection.endheaders
    http.client.HTTPConnection.send http.client.HTTPSConnection.connect
    http.client.HTTPSConnection.request http.client.HTTPSConnection.send
    ftplib.FTP.connect ftplib.FTP_TLS.connect smtplib.SMTP.connect
    smtplib.SMTP_SSL.connect smtplib.LMTP.connect
    logging.handlers.SocketHandler.emit logging.handlers.SocketHandler.makeSocket
    logging.handlers.DatagramHandler.emit logging.handlers.DatagramHandler.makeSocket
    logging.handlers.HTTPHandler.emit logging.handlers.SMTPHandler.emit
    ssl.get_server_certificate webbrowser.open webbrowser.open_new
    webbrowser.open_new_tab requests.get requests.post requests.put
    requests.delete requests.head requests.request requests.api.request
    httpx.get httpx.post httpx.request urllib3.PoolManager.request
    multiprocessing.connection.Client''')

NESTED = set('''pickle.loads pickle.load pickle._loads pickle._load cloudpickle.loads dill.loads torch.load torch.storage._load_from_bytes
    numpy.load numpy.lib.format.read_array joblib.load pandas.read_pickle
    pandas.io.pickle.read_pickle'''.split())


class Semantics:
    def __init__(self, context=None):
        self.ctx = context or Context()
        self.location = ''
        self.op = 'call'
        self.call_depth = 0

    def finding(self, kind, name):
        raise Detected(Finding(kind, self.location, f'{self.op} {name}'))

    def module(self, name):
        name = ALIASES.get(name, name)
        if name not in self.ctx.modules:
            if (name in _compat_pickle.IMPORT_MAPPING or name == 'exceptions') and name not in NAMESPACES:
                raise LoadError('Python 2 module name has not been remapped')
            self.ctx.modules[name] = Value('module', name)
        return self.ctx.modules[name]

    def ref(self, name):
        if name not in self.ctx.globals:
            self.ctx.globals[name] = Value('global', name)
        return self.ctx.globals[name]

    def global_ref(self, module, name, protocol=4, fix_imports=True):
        if not isinstance(module, str) or not isinstance(name, str):
            raise LoadError('GLOBAL requires strings')
        if protocol < 3 and fix_imports:
            module, name = _compat_pickle.NAME_MAPPING.get((module, name),
                            (_compat_pickle.IMPORT_MAPPING.get(module, module), name))
        module = ALIASES.get(module, module)
        if '.' in name and protocol < 4:
            raise LoadError('dotted GLOBAL below protocol 4')
        obj = self.module(module)
        for part in name.split('.'):
            obj = self.attr(obj, part)
        return obj

    def is_class(self, obj):
        if not isinstance(obj, Value):
            return False
        if obj.kind == 'class':
            return True
        if obj.kind != 'global':
            return False
        if obj.name in SYMBOLS:
            return SYMBOLS[obj.name]['kind'] == 'class'
        if obj.name in CLASS_NAMES:
            return True
        part = obj.name.rsplit('.', 1)[-1]
        return bool(part[:1].isupper()) and obj.name not in EFFECTS

    def attr(self, obj, name, default=MISSING):
        self.ctx.tick()
        if not isinstance(name, str):
            raise LoadError('attribute name must be a string')
        if isinstance(obj, Mapping):
            if name == '__dict__':
                return obj.attrs
            if name in obj.attrs:
                return obj.attrs[name]
        if isinstance(obj, Value):
            if name in obj.attrs:
                return obj.attrs[name]
            if obj.kind == 'remote':
                return Value('remote', obj.name + '.' + name, obj.args)
            if obj.kind == 'instance' and obj.name == 'xmlrpc.client.ServerProxy' and not name.startswith('_'):
                return Value('remote', obj.name + '.' + name, (obj,))
            if name == '__dict__':
                if obj.kind == 'module':
                    return Value('module_dict', obj.name, (obj,))
                return obj.attrs
            if obj.kind == 'module':
                full = obj.name + '.' + name
                if full == 'sys.modules':
                    return Value('modules')
                if name == '__builtins__':
                    return self.module('builtins')
                if obj.name in NAMESPACES:
                    return self.export(obj.name, name, default)
                if full in REEXPORTS:
                    target = REEXPORTS[full]
                    return self.module(target) if target in MODULE_NAMES else self.ref(target)
                if full in MODULE_NAMES:
                    return self.module(full)
                return self.ref(full)
            if name == '__class__':
                if obj.kind == 'instance':
                    return obj.attrs.get('_class', self.ref(obj.name))
                return self.ref({'module': 'types.ModuleType', 'partial': 'functools.partial',
                                 'global': 'builtins.type' if self.is_class(obj) else 'types.FunctionType',
                                 'buffer': 'io.BytesIO'}.get(obj.kind, 'builtins.object'))
            if name == '__call__':
                return Value('bound', name, (obj,))
            if obj.kind == 'global':
                if name in ('__globals__', 'func_globals', '__builtins__'):
                    if obj.name in SYMBOLS and not SYMBOLS[obj.name].get('python'):
                        if default is not MISSING:
                            return default
                        raise LoadError('builtin callable has no Python globals')
                    if name == '__builtins__':
                        return Value('module_dict', 'builtins', (self.module('builtins'),))
                    return Value('module_dict', obj.name.rsplit('.', 1)[0],
                                 (self.module(obj.name.rsplit('.', 1)[0]),))
                if name in ('__self__',):
                    return self.module(obj.name.rsplit('.', 1)[0])
                if obj.name in NAMESPACES:
                    return self.export(obj.name, name, default)
                return self.ref(obj.name + '.' + name)
            if obj.kind == 'class':
                for base in obj.args:
                    if isinstance(base, Value) and name in base.attrs:
                        return base.attrs[name]
                return Value('bound', name, (obj,))
            if obj.kind == 'partial' and name in ('func', 'args', 'keywords'):
                return obj.args[{'func': 0, 'args': 1, 'keywords': 2}[name]]
            if obj.kind == 'instance':
                cls = obj.attrs.get('_class')
                if cls is not None and name in cls.attrs:
                    value = cls.attrs[name]
                    # Python functions bind; builtin functions and partials do not.
                    if isinstance(value, Value) and value.kind == 'function':
                        return Value('method', name, (value, obj))
                    return value
            return Value('bound', name, (obj,))
        if name == '__class__':
            return self.ref('builtins.' + type(obj).__name__)
        return Value('bound', name, (obj,))

    def export(self, namespace, name, default=MISSING):
        entry = NAMESPACES.get(namespace, {}).get(name)
        if entry is None:
            if default is not MISSING:
                return default
            raise LoadError('attribute does not exist: ' + namespace + '.' + name)
        kind, value = entry
        if kind == 'module':
            return self.module(value)
        if kind in ('function', 'class'):
            return self.ref(value)
        if kind == 'data':
            return value
        if kind == 'bytes':
            return bytes.fromhex(value)
        return UNKNOWN

    def validate_call(self, name, args, kw):
        params = SYMBOLS.get(name, {}).get('params')
        if params is None:
            return
        positional = [p for p in params if p[1] in (0, 1)]
        varargs = any(p[1] == 2 for p in params)
        varkw = any(p[1] == 4 for p in params)
        if len(args) > len(positional) and not varargs:
            raise LoadError('too many positional arguments')
        used = {p[0] for p in positional[:len(args)]}
        byname = {p[0]: p for p in params}
        for key in kw:
            param = byname.get(key)
            if param is None or param[1] in (0, 2, 4):
                if not varkw:
                    raise LoadError('invalid keyword argument')
            elif key in used:
                raise LoadError('duplicate argument')
            else:
                used.add(key)
        for pname, pkind, required in params:
            if required and pkind in (0, 1, 3) and pname not in used:
                raise LoadError('missing required argument ' + pname)

    def getitem(self, obj, key):
        if isinstance(obj, Value):
            if obj.kind == 'npz':
                from modelgate.dispatch import analyze_npy
                import zipfile
                with zipfile.ZipFile(io.BytesIO(obj.attrs['data'])) as zf:
                    key = key if key in zf.namelist() else key + '.npy'
                    data = zf.read(key)
                return analyze_npy(data, obj.attrs['location'] + '!' + key, self.ctx,
                                   allow_pickle=obj.attrs['allow_pickle']) if data.startswith(b'\x93NUMPY') else data
            if obj.kind == 'modules':
                return self.module(key) if isinstance(key, str) else UNKNOWN
            if obj.kind == 'module_dict':
                return self.attr(obj.args[0], key)
            if obj.kind == 'instance':
                if obj.name in ('collections.UserDict', 'collections.UserList'):
                    return self.getitem(obj.attrs.get('data', {}), key)
                if obj.name == 'numpy.ndarray':
                    return self.getitem(obj.attrs.get('data', []), key)
                cls = obj.attrs.get('_class')
                if cls and '__getitem__' in cls.attrs:
                    return self.call(cls.attrs['__getitem__'], (obj, key))
            return UNKNOWN
        try:
            return obj[key]
        except (TypeError, KeyError, IndexError):
            raise LoadError('invalid subscription')

    def setitem(self, obj, key, value):
        if isinstance(obj, Value):
            if obj.kind == 'module_dict':
                obj.args[0].attrs[key] = value
            elif obj.kind == 'modules':
                self.ctx.modules[key] = value
            elif obj.kind == 'instance' and obj.name == 'collections.UserDict':
                obj.attrs.setdefault('data', {})[key] = value
            else:
                obj.attrs.setdefault('_items', {})[key] = value
        else:
            try:
                obj[key] = value
            except (TypeError, IndexError):
                raise LoadError('invalid item assignment')

    def build(self, obj, state):
        if isinstance(obj, Mapping):
            if isinstance(state, dict):
                obj.attrs.update(state)
                return
            if isinstance(state, tuple) and len(state) == 2:
                for part in state:
                    if isinstance(part, dict):
                        obj.attrs.update(part)
                return
            raise LoadError('invalid mapping state')
        if isinstance(obj, Value) and obj.kind == 'partial':
            if not isinstance(state, tuple) or len(state) != 4:
                raise LoadError('invalid partial state')
            func, args, kw, namespace = state
            if not isinstance(args, tuple) or (kw is not None and not isinstance(kw, dict)):
                raise LoadError('invalid partial arguments')
            obj.args = (func, args, kw or {})
            if isinstance(namespace, dict):
                obj.attrs.update(namespace)
            return
        if isinstance(obj, Value):
            cls = obj.attrs.get('_class')
            setstate = obj.attrs.get('__setstate__', MISSING)
            if setstate is MISSING and cls is not None:
                setstate = cls.attrs.get('__setstate__', MISSING)
                if setstate is not MISSING:
                    self.call(setstate, (obj, state))
                    return
            if setstate is not MISSING:
                self.call(setstate, (state,))
                return
            if obj.name == 'numpy.dtype':
                obj.attrs['_state'] = state
                return
            if obj.name == 'numpy.ndarray':
                obj.attrs['_state'] = state
                if isinstance(state, tuple) and state:
                    obj.attrs['data'] = state[-1]
                return
            if obj.kind == 'iterator':
                obj.attrs['_state'] = state
                return
        slots = None
        if isinstance(state, tuple) and len(state) == 2:
            state, slots = state
        if state is not None:
            if not isinstance(state, dict):
                raise LoadError('BUILD state must be a dictionary')
            if isinstance(obj, Value):
                obj.attrs.update(state)
            else:
                raise LoadError('object has no instance dictionary')
        if isinstance(slots, dict) and isinstance(obj, Value):
            obj.attrs.update(slots)

    def call(self, func, args=(), kw=None, mode='call'):
        self.ctx.tick()
        if not isinstance(args, tuple):
            raise LoadError('call arguments must be a tuple')
        if kw is None:
            kw = {}
        if not isinstance(kw, dict):
            raise LoadError('keyword arguments must be a dictionary')
        self.call_depth += 1
        try:
            if self.call_depth > 120:
                raise LimitError('symbolic call recursion limit')
            return self._call(func, args, kw, mode)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError, AttributeError,
                UnicodeError, ZeroDivisionError) as exc:
            raise LoadError(str(exc)) from None
        finally:
            self.call_depth -= 1

    def _call(self, func, args, kw, mode):
        if not isinstance(func, Value):
            raise LoadError('object is not callable')
        if mode == 'new' and not self.is_class(func):
            raise LoadError('NEWOBJ requires a type')
        if func.kind == 'partial':
            target, fixed, keywords = func.args
            return self.call(target, fixed + args, dict(keywords, **kw))
        if func.kind == 'method':
            return self.call(func.args[0], (func.args[1],) + args, kw)
        if func.kind == 'bound':
            return self.bound(func.args[0], func.name, args, kw)
        if func.kind == 'adapter':
            return self.adapter(func, args, kw)
        if func.kind == 'remote':
            self.finding('network', func.name)
        if func.kind == 'function':
            self.finding('code-execution', 'serialized Python function')
        if func.kind == 'class':
            obj = Value('instance', func.name, args, {'_class': func})
            if '__new__' in func.attrs:
                obj = self.call(func.attrs['__new__'], (func,) + args, kw)
            if mode != 'new' and '__init__' in func.attrs:
                self.call(func.attrs['__init__'], (obj,) + args, kw)
            return obj
        if func.kind == 'instance':
            if func.name in ('codeop.Compile', 'codeop.CommandCompiler'):
                self.finding('code-execution', func.name + '.__call__')
            cls = func.attrs.get('_class')
            if cls and '__call__' in cls.attrs:
                return self.call(cls.attrs['__call__'], (func,) + args, kw)
            return UNKNOWN
        if func.kind == 'unknown':
            raise LimitError('unresolved callable')
        if func.kind != 'global':
            raise LoadError('object is not callable')
        name = func.name
        if name.endswith('.__call__'):
            return self.call(self.ref(name[:-9]), args, kw)
        if name.endswith('.__new__'):
            if not args:
                raise LoadError('__new__ requires a type')
            if name == 'builtins.object.__new__':
                return Value('instance', args[0].name, (), {'_class': args[0]})
            return self.call(args[0], args[1:], kw, mode='new')
        if name.endswith('.__init__'):
            return self.call(self.ref(name[:-9]), args[1:], kw)
        if mode == 'new':
            if name in ('builtins.list', 'builtins.dict', 'builtins.set', 'collections.OrderedDict', 'collections.defaultdict', 'collections.deque'):
                return self.operations(name, (), {})
            if name in ('io.BytesIO', 'io.StringIO'):
                return self.operations(name, (), {})
            # Most effectful classes do their work in __init__, not __new__.
            # Value factories and immutable builtins are the exceptions.
            if name not in ('functools.partial', 'operator.attrgetter', 'operator.itemgetter',
                            'operator.methodcaller', 'types.FunctionType', 'types.CodeType',
                            'types.MethodType', 'builtins.type', 'builtins.map', 'builtins.filter',
                            'builtins.zip', 'builtins.enumerate', 'builtins.reversed',
                            'functools._lru_cache_wrapper', 'numpy.memmap') and not name.startswith(('builtins.', 'pathlib.', 'itertools.')):
                return Value('instance', name, (), {'_uninitialized': True})
        return self.global_call(name, args, kw)

    def global_call(self, name, args, kw):
        # Implemented below, grouped by behavior instead of global spelling.
        return self.operations(name, args, kw)

    def operations(self, name, args, kw):
        # Unbound builtin and supported class methods receive their receiver
        # explicitly. Route them through the same semantics as bound methods.
        owner, _, method = name.rpartition('.')
        unbound_owners = ('builtins.dict', 'builtins.list', 'builtins.set', 'builtins.str',
                          'builtins.bytes', 'builtins.bytearray', 'builtins.tuple',
                          'builtins.map', 'builtins.filter', 'builtins.zip', 'builtins.enumerate', 'builtins.reversed',
                          'threading.Thread', 'threading.Timer', 'multiprocessing.Process',
                          'concurrent.futures.ThreadPoolExecutor',
                          'concurrent.futures.ProcessPoolExecutor', 'io.BytesIO', 'io.StringIO',
                          'functools.partial')
        if (owner in unbound_owners or owner.endswith(('ThreadPoolExecutor', 'ProcessPoolExecutor', 'ThreadPool'))) and method not in ('fromhex', 'fromkeys'):
            if not args:
                raise LoadError('unbound method requires a receiver')
            return self.bound(args[0], method, args[1:], kw)
        if name in ('builtins.getattr', 'builtins.object.__getattribute__',
                    'types.ModuleType.__getattribute__', 'operator.getattr'):
            return self.attr(*args)
        if name in ('builtins.setattr', 'builtins.object.__setattr__', 'types.ModuleType.__setattr__'):
            obj, key, val = args
            if isinstance(obj, Value):
                obj.attrs[key] = val
            return None
        if name == 'builtins.delattr':
            obj, key = args
            if isinstance(obj, Value):
                obj.attrs.pop(key, None)
            return None
        if name == 'builtins.hasattr':
            return self.attr(args[0], args[1], MISSING) is not MISSING
        if name == 'builtins.vars':
            return self.attr(args[0], '__dict__') if args else {}
        if name in ('builtins.globals', 'builtins.locals'):
            return {'__builtins__': self.module('builtins')}
        if name in ('builtins.__import__', 'importlib.import_module'):
            target = args[0]
            if not isinstance(target, str):
                raise LoadError('module name is not a string')
            if name == 'builtins.__import__' and not (kw.get('fromlist') or (len(args) > 3 and args[3])):
                target = target.split('.')[0]
            return self.module(target)
        if name in ('pkgutil.resolve_name', 'pydoc.locate'):
            path = args[0]
            if not isinstance(path, str):
                return UNKNOWN
            if ':' in path:
                mod, suffix = path.split(':', 1)
                obj = self.module(mod)
                for part in suffix.split('.') if suffix else ():
                    obj = self.attr(obj, part)
                return obj
            parts = path.split('.')
            i = len(parts)
            while i > 1 and '.'.join(parts[:i]) not in MODULE_NAMES:
                i -= 1
            obj = self.module('.'.join(parts[:i]))
            for part in parts[i:]:
                obj = self.attr(obj, part)
            return obj
        if name in ('operator.attrgetter', 'operator.itemgetter', 'operator.methodcaller'):
            if not args:
                raise LoadError('operator adapter requires arguments')
            return Value('adapter', name, args, kw.copy())
        if name in ('functools.partial', 'functools.partialmethod'):
            if args and (not isinstance(args[0], Value) or args[0].kind in ('module', 'buffer', 'code')):
                raise LoadError('partial requires a callable')
            if not args:
                raise LoadError('partial requires callable')
            return Value('partial', name, (args[0], args[1:], kw.copy()))
        if name in ('functools.lru_cache', 'functools.cache'):
            if args and isinstance(args[0], Value):
                return Value('partial', name, (args[0], (), {}))
            return Value('adapter', 'decorator', ())
        if name == 'functools._lru_cache_wrapper':
            return Value('partial', name, (args[0], (), {}))
        if name in ('functools.update_wrapper', 'functools.wraps'):
            if name.endswith('wraps'):
                return Value('adapter', 'decorator', ())
            args[0].attrs['__wrapped__'] = args[1]
            return args[0]
        if name in ('copy.copy', 'copy.deepcopy'):
            obj = args[0]
            if isinstance(obj, Value):
                if obj.kind in ('global', 'module', 'class'):
                    return obj
                return Value(obj.kind, obj.name, obj.args, obj.attrs.copy())
            return copy.copy(obj)
        if name in ('weakref.proxy',):
            return Value('partial', name, (args[0], (), {}))
        if name in ('weakref.ref', 'weakref.ReferenceType', 'weakref.WeakMethod'):
            return Value('adapter', 'reference', (args[0],))
        if name in ('types.MethodType', 'builtins.instancemethod'):
            return Value('method', name, args)
        if name in ('builtins.staticmethod', 'builtins.classmethod'):
            return args[0]
        if name in ('builtins.type', 'types.new_class'):
            if name == 'builtins.type' and len(args) == 1:
                return self.attr(args[0], '__class__')
            if len(args) >= 3 and isinstance(args[2], dict):
                return Value('class', str(args[0]), tuple(args[1]), args[2].copy())
            return Value('class', str(args[0]), ())
        if name == 'os.getcwd':
            import os
            return os.getcwd()
        if name in ('types.ModuleType',):
            return Value('module', args[0] if args else '')
        if name in ('types.SimpleNamespace',):
            return Value('instance', name, (), kw.copy())
        if name in ('copyreg._reconstructor',):
            cls, base, state = args
            if isinstance(base, Value) and base.name != 'builtins.object':
                return self.call(base, (state,), mode='new')
            return self.call(cls, (), mode='new')
        if name in ('copyreg.__newobj__',):
            return self.call(args[0], args[1:], mode='new')
        if name in ('copyreg.__newobj_ex__',):
            return self.call(args[0], args[1], args[2], mode='new')
        if name in ('operator.call', 'builtins.apply'):
            return self.call(args[0], args[1:] if name == 'operator.call' else args[1],
                             kw if name == 'operator.call' else (args[2] if len(args) > 2 else {}))
        if name in ('functools.reduce',):
            func, seq = args[:2]
            values = self.iterate(seq)
            try:
                acc = args[2] if len(args) > 2 else next(values)
            except StopIteration:
                raise LoadError('reduce of empty iterable') from None
            for value in values:
                acc = self.call(func, (acc, value))
            return acc
        if name in ('builtins.map', 'builtins.filter', 'itertools.starmap',
                    'itertools.filterfalse', 'itertools.takewhile', 'itertools.dropwhile',
                    'itertools.accumulate', 'itertools.groupby'):
            return Value('iterator', name, args, kw.copy())
        if name in ('builtins.iter', 'builtins.enumerate', 'builtins.zip', 'builtins.reversed',
                    'itertools.chain', 'itertools.chain.from_iterable', 'itertools.islice',
                    'itertools.repeat', 'itertools.count', 'itertools.cycle',
                    'itertools.compress', 'itertools.zip_longest', 'itertools.pairwise'):
            return Value('iterator', name, args, kw.copy())
        if name in ('itertools.product', 'itertools.permutations', 'itertools.combinations',
                    'itertools.combinations_with_replacement'):
            if name == 'itertools.product':
                pools = tuple(tuple(self.iterate(v)) for v in args)
                repeat = kw.get('repeat', 1)
                if repeat < 0:
                    raise LoadError('negative itertools repeat')
                self.ctx.allocate(len(pools) * repeat * 8)
                return Value('iterator', name, pools, kw.copy())
            return Value('iterator', name, (tuple(self.iterate(args[0])),) + args[1:], kw.copy())
        if name == 'itertools.tee':
            count = args[1] if len(args) > 1 else 2
            return tuple(Value('iterator', 'builtins.iter', (args[0],)) for _ in range(min(count, 1000)))
        if name == 'builtins.next':
            it = self.iterate(args[0])
            try:
                return next(it)
            except StopIteration:
                if len(args) > 1:
                    return args[1]
                raise LoadError('iterator exhausted')
        if name in ('builtins.list', 'builtins.tuple', 'builtins.set', 'builtins.frozenset',
                    'collections.deque', 'collections.UserList'):
            values = list(self.iterate(args[0])) if args else []
            self.ctx.allocate(len(values) * 8)
            if name == 'collections.deque' and len(args) > 1 and args[1] is not None:
                values = values[-args[1]:] if args[1] else []
            return {'builtins.tuple': tuple, 'builtins.set': set,
                    'builtins.frozenset': frozenset}.get(name, list)(values)
        if name in ('builtins.dict', 'collections.OrderedDict', 'collections.defaultdict',
                    'collections.Counter', 'collections.UserDict'):
            seq = args[1:] if name == 'collections.defaultdict' else args
            result = {}
            if seq:
                source = seq[0]
                if isinstance(source, dict):
                    result.update(source)
                else:
                    for item in self.iterate(source):
                        if name == 'collections.Counter':
                            result[item] = result.get(item, 0) + 1
                        else:
                            k, v = item
                            result[k] = v
            result.update(kw)
            return result if name == 'builtins.dict' else Mapping(name, result)
        if name in ('builtins.sorted', 'builtins.min', 'builtins.max'):
            values = list(self.iterate(args[0])) if len(args) == 1 else list(args)
            key = kw.get('key')
            decorated = [(self.call(key, (v,)) if key is not None else v, i, v)
                         for i, v in enumerate(values)]
            if name == 'builtins.sorted':
                if all(literal(x[0]) for x in decorated):
                    decorated.sort(key=lambda x: x[0], reverse=bool(kw.get('reverse')))
                return [v for _, _, v in decorated]
            if not decorated:
                if 'default' in kw:
                    return kw['default']
                raise LoadError('empty min/max')
            if all(literal(x[0]) for x in decorated):
                return (min if name.endswith('min') else max)(decorated, key=lambda x: x[0])[2]
            return decorated[0][2]
        if name in ('builtins.any', 'builtins.all', 'builtins.sum'):
            accum = args[1] if len(args) > 1 else 0
            for value in self.iterate(args[0]):
                if name == 'builtins.any' and self.truth(value):
                    return True
                if name == 'builtins.all' and not self.truth(value):
                    return False
                if name == 'builtins.sum':
                    accum = self.operations('operator.add', (accum, value), {})
            return accum if name.endswith('sum') else name.endswith('all')
        if name in ('io.BytesIO', 'io.StringIO'):
            data = args[0] if args else (b'' if name.endswith('BytesIO') else '')
            if not isinstance(data, (str, bytes, bytearray)):
                raise LoadError('invalid memory buffer')
            return Value('buffer', name, (), {'data': data, 'pos': 0})
        if name in ('pickle.Unpickler', 'pickle._Unpickler'):
            return Value('instance', 'pickle.Unpickler', args, kw.copy())
        if name in NESTED:
            name = {'pickle._loads': 'pickle.loads', 'pickle._load': 'pickle.load',
                    'cloudpickle.loads': 'pickle.loads', 'dill.loads': 'pickle.loads'}.get(name, name)
            return self.nested(name, args, kw)
        if name in ('marshal.loads', 'marshal.load'):
            from modelgate.marshal_data import read_marshaled
            data = args[0] if name.endswith('loads') else self.bound(args[0], 'read', (), {})
            return read_marshaled(data, self.ctx)
        if name in ('types.CodeType', 'dill._dill._create_code', 'cloudpickle.cloudpickle._code_reduce'):
            if name == 'types.CodeType':
                self.validate_call(name, args, kw)
            return Value('code', name, args)
        if name == 'cloudpickle.cloudpickle._builtin_type':
            return self.ref('types.' + args[0])
        if name in ('cloudpickle.cloudpickle._make_function', 'dill._dill._create_function',
                    'dill._dill._create_function3', 'cloudpickle.cloudpickle._make_skel_func'):
            self.finding('code-execution', name)
        if name == 'types.FunctionType':
            self.validate_call(name, args, kw)
            if len(args) < 2 or not isinstance(args[1], dict):
                raise LoadError('FunctionType globals must be a dictionary')
            if not args or not isinstance(args[0], Value) or args[0].kind != 'code':
                raise LoadError('FunctionType requires a code object')
            self.finding('code-execution', name)
        if name in ('json.loads', 'json.load'):
            import json
            text = args[0] if name.endswith('loads') else self.bound(args[0], 'read', (), {})
            if not isinstance(text, (str, bytes, bytearray)):
                raise LoadError('JSON input must be text')
            # Decode with trusted, fixed hooks. Artifact callbacks are simulated
            # through call(), never passed to the real JSON implementation.
            def number(field, value, default):
                return self.call(kw[field], (value,)) if kw.get(field) else default(value)
            def pairs(items):
                if kw.get('object_pairs_hook'):
                    return self.call(kw['object_pairs_hook'], (items,))
                obj = dict(items)
                return self.call(kw['object_hook'], (obj,)) if kw.get('object_hook') else obj
            return json.loads(text, parse_int=lambda x: number('parse_int', x, int),
                              parse_float=lambda x: number('parse_float', x, float),
                              parse_constant=lambda x: number('parse_constant', x, float),
                              object_pairs_hook=pairs)
        if name == 'ast.literal_eval':
            import ast
            return ast.literal_eval(args[0])
        if name == 'ast.parse':
            return Value('instance', 'ast.Module', args)
        if name.startswith('struct.') and name.split('.')[1] in ('pack', 'unpack', 'unpack_from', 'calcsize'):
            import struct
            if literal(args) and literal(kw):
                if name.endswith('pack') and args and isinstance(args[0], str):
                    self.ctx.allocate(struct.calcsize(args[0]))
                return getattr(struct, name.split('.')[1])(*args, **kw)
            return UNKNOWN
        if name in ('codecs.getdecoder', 'codecs.getencoder'):
            return Value('adapter', 'codec_decode' if name.endswith('decoder') else 'codec_encode', args)
        if name in ('base64.b64decode', 'base64.standard_b64decode', 'base64.urlsafe_b64decode',
                    'base64.decodebytes', 'base64.b16decode', 'base64.b32decode',
                    'base64.b85decode', 'base64.a85decode', 'base64.b64encode',
                    'base64.standard_b64encode', 'base64.urlsafe_b64encode'):
            if literal(args) and literal(kw):
                return getattr(base64, name.split('.')[1])(*args, **kw)
            return UNKNOWN
        if name.startswith('binascii.') and name.split('.')[1] in ('a2b_base64', 'a2b_hex',
                'unhexlify', 'a2b_qp', 'a2b_uu', 'hexlify', 'b2a_base64'):
            if literal(args) and literal(kw):
                return getattr(binascii, name.split('.')[1])(*args, **kw)
            return UNKNOWN
        if name in ('codecs.encode', '_codecs.encode', 'codecs.decode', '_codecs.decode'):
            if literal(args) and literal(kw):
                # Only data codecs are allowed; never invoke arbitrary codec hooks.
                encoding = args[1] if len(args) > 1 else kw.get('encoding', 'utf-8')
                allowed = ('utf8', 'utf-8', 'utf-16', 'utf-32', 'ascii', 'latin1', 'latin-1',
                           'latin_1', 'iso8859-1', 'unicode_escape', 'raw_unicode_escape',
                           'base64', 'base64_codec', 'hex', 'hex_codec', 'rot_13',
                           'bz2', 'bz2_codec', 'zlib', 'zlib_codec')
                if encoding not in allowed:
                    raise LimitError('unsupported data codec')
                if encoding.startswith(('bz2', 'zlib')) and name.endswith('decode'):
                    return self.decompress(encoding.split('_')[0], args[0])
                return (codecs.encode if name.endswith('encode') else codecs.decode)(*args, **kw)
            return UNKNOWN
        if name in ('zlib.decompress', 'gzip.decompress', 'bz2.decompress', 'lzma.decompress'):
            return self.decompress(name.split('.')[0], args[0], args[1:], kw)
        if name in ('zlib.decompressobj', 'bz2.BZ2Decompressor', 'lzma.LZMADecompressor'):
            return Value('decompressor', name.split('.')[0], args, kw.copy())
        if name in ('builtins.bytes.fromhex', 'builtins.bytearray.fromhex'):
            result = bytes.fromhex(args[0])
            return bytearray(result) if 'bytearray' in name else result
        if name == 'builtins.memoryview':
            return bytes(args[0]) if isinstance(args[0], (bytes, bytearray)) else UNKNOWN
        if name in ('builtins.bytes', 'builtins.bytearray'):
            factory = bytes if name.endswith('.bytes') else bytearray
            if args and isinstance(args[0], Value):
                return factory(self.iterate(args[0]))
            if args and isinstance(args[0], int):
                self.ctx.allocate(args[0])
            return factory(*args, **kw)
        if name in ('builtins.str', 'builtins.int', 'builtins.float', 'builtins.bool',
                    'builtins.complex', 'builtins.slice', 'builtins.range', 'builtins.chr',
                    'builtins.ord', 'builtins.len', 'builtins.abs', 'builtins.round',
                    'builtins.repr', 'builtins.ascii', 'builtins.hex', 'builtins.oct',
                    'builtins.bin', 'builtins.divmod', 'builtins.pow'):
            if not literal(args) or not literal(kw):
                return UNKNOWN
            import builtins
            if name == 'builtins.pow' and len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
                if args[0].bit_length() * max(args[1], 0) > 8_000_000:
                    raise LimitError('integer exponent limit')
            return getattr(builtins, name.split('.')[1])(*args, **kw)
        if name == 'builtins.callable':
            return isinstance(args[0], Value) and args[0].kind in ('global', 'class', 'partial', 'bound', 'adapter', 'function', 'method')
        if name in ('os.fspath', 'os.fsencode', 'os.fsdecode'):
            val = args[0]
            if isinstance(val, Value) and val.name.startswith('pathlib.'):
                val = val.attrs.get('path', '')
            if name.endswith('fsencode') and isinstance(val, str):
                return val.encode()
            if name.endswith('fsdecode') and isinstance(val, bytes):
                return val.decode()
            return val
        if name in ('pathlib.Path', 'pathlib.PosixPath', 'pathlib.WindowsPath',
                    'pathlib.PurePath', 'pathlib.PurePosixPath', 'pathlib.PureWindowsPath'):
            parts = [v.attrs.get('path', '') if isinstance(v, Value) else v for v in args]
            path = posixpath.join(*parts) if parts else '.'
            return Value('instance', name, args, {'path': path})
        if name.startswith(('posixpath.', 'ntpath.', 'os.path.')):
            method = name.rsplit('.', 1)[-1]
            if method in ('join', 'basename', 'dirname', 'split', 'splitext', 'normpath',
                          'isabs', 'commonprefix', 'abspath', 'relpath', 'normcase', 'commonpath') and literal(args):
                return getattr(posixpath, method)(*args)
        if name.startswith('math.') and SYMBOLS.get(name, {}).get('kind') == 'function':
            if literal(args) and literal(kw):
                import math
                if name in ('math.factorial', 'math.comb', 'math.perm') and args and args[0] > 100000:
                    raise LimitError('integer arithmetic work limit')
                return getattr(math, name.split('.')[1])(*args, **kw)
            return UNKNOWN
        if name in ('builtins.hash', 'builtins.format', 'builtins.isinstance', 'builtins.issubclass'):
            if literal(args) and literal(kw):
                import builtins
                return getattr(builtins, name.split('.')[1])(*args, **kw)
            return UNKNOWN
        if name.startswith('operator.'):
            method = name.split('.')[1]
            if method in ('getitem', '__getitem__'):
                return self.getitem(*args)
            if method in ('setitem', '__setitem__'):
                self.setitem(*args)
                return None
            if method in ('add', 'concat', 'iadd', 'iconcat', '__add__', '__concat__'):
                if literal(args):
                    if isinstance(args[0], (str, bytes, list, tuple, bytearray)):
                        self.ctx.allocate(len(args[0]) + len(args[1]))
                    if method in ('iadd', 'iconcat') and isinstance(args[0], (list, bytearray)):
                        args[0].extend(args[1])
                        return args[0]
                    return args[0] + args[1]
                if isinstance(args[0], (list, tuple)) and isinstance(args[1], type(args[0])):
                    if method in ('iadd', 'iconcat') and isinstance(args[0], list):
                        args[0].extend(args[1])
                        return args[0]
                    return args[0] + args[1]
                return UNKNOWN
            if method in ('mul', 'imul', '__mul__') and literal(args):
                for x, n in (args, args[::-1]):
                    if isinstance(x, (str, bytes, list, tuple, bytearray)) and isinstance(n, int):
                        self.ctx.allocate(len(x) * max(n, 0))
                if method == 'imul' and isinstance(args[0], (list, bytearray)):
                    value = args[0]
                    value *= args[1]
                    return value
                return operator.mul(*args)
            if method in ('truth', 'not_', 'neg', 'pos', 'index', 'inv', 'invert',
                          'sub', 'truediv', 'floordiv', 'mod', 'xor', 'and_', 'or_',
                          'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'is_', 'is_not', 'contains'):
                return getattr(operator, method)(*args) if literal(args) else UNKNOWN
        if name in ('contextlib.ExitStack', 'contextlib._BaseExitStack'):
            return Value('instance', 'contextlib.ExitStack', (), {'callbacks': []})
        if name in ('multiprocessing.pool.ThreadPool', 'multiprocessing.dummy.Pool'):
            init = args[1] if len(args) > 1 else kw.get('initializer')
            initargs = args[2] if len(args) > 2 else kw.get('initargs', ())
            if init:
                self.call(init, tuple(initargs))
            return Value('instance', 'multiprocessing.pool.ThreadPool', args, kw.copy())
        if name in ('threading.Thread', 'threading.Timer', 'multiprocessing.Process',
                    'multiprocessing.context.Process'):
            fields = dict(kw)
            if name.endswith('Timer'):
                fields.update(dict(zip(('interval', 'function', 'args', 'kwargs'), args)))
                fields['_target'] = fields.get('function')
            else:
                fields.update(dict(zip(('group', 'target', 'name', 'args', 'kwargs'), args)))
                fields['_target'] = fields.get('target')
            fields['_args'] = fields.get('args', ())
            fields['_kwargs'] = fields.get('kwargs', {})
            return Value('instance', name, args, fields)
        if name in ('_thread.start_new_thread', 'threading._start_new_thread'):
            return self.call(args[0], args[1], args[2] if len(args) > 2 else {})
        if name.endswith('ThreadPoolExecutor') or name.endswith('ProcessPoolExecutor'):
            fields = kw.copy()
            fields.update(dict(zip(('max_workers', 'thread_name_prefix', 'initializer', 'initargs'), args)))
            return Value('instance', name, args, fields)
        if name in ('ftplib.FTP', 'ftplib.FTP_TLS', 'smtplib.SMTP', 'smtplib.SMTP_SSL', 'smtplib.LMTP'):
            host = args[0] if args else kw.get('host')
            if host:
                self.finding('network', name + '(host)')
            return Value('instance', name, args, kw.copy())
        if name in ('socket.socket', 'socket.SocketType', 'http.client.HTTPConnection',
                    'http.client.HTTPSConnection', 'xmlrpc.client.ServerProxy',
                    'urllib.request.Request', 'urllib.request.build_opener',
                    'requests.Session', 'requests.sessions.Session', 'httpx.Client',
                    'httpx.AsyncClient', 'urllib3.PoolManager'):
            return Value('instance', 'socket.socket' if name == 'socket.SocketType' else name, args, kw.copy())
        if name == 'logging.LogRecord':
            self.validate_call(name, args, kw)
            fields = dict(zip(('name', 'level', 'pathname', 'lineno', 'msg', 'args', 'exc_info', 'func', 'sinfo'), args))
            fields.update(kw)
            return Value('instance', name, args, fields)
        if name in ('logging.Logger', 'logging.LoggerAdapter', 'logging.Formatter', 'logging.Filter'):
            self.validate_call(name, args, kw)
            return Value('instance', name, args, kw.copy())
        if name in ('codeop.Compile', 'codeop.CommandCompiler', 'logging.handlers.SMTPHandler',
                    'logging.handlers.DatagramHandler', 'logging.NullHandler', 'logging.StreamHandler'):
            self.validate_call(name, args, kw)
            return Value('instance', name, args, kw.copy())
        if name in ('logging.config.DictConfigurator', 'logging.config.BaseConfigurator'):
            return Value('instance', name, args, {'config': args[0] if args else kw.get('config', {})})
        if name in ('logging.FileHandler', 'logging.handlers.RotatingFileHandler',
                    'logging.handlers.TimedRotatingFileHandler', 'logging.handlers.WatchedFileHandler'):
            params = SYMBOLS.get(name, {}).get('params', [])
            values = dict(zip((p[0] for p in params if p[1] in (0, 1)), args))
            values.update(kw)
            self.validate_call(name, args, kw)
            if values.get('delay'):
                return Value('instance', name, args, values)
            self.finding('file-access', name)
        if name in ('logging.config.dictConfig', 'logging.config.fileConfig'):
            return self.logging_config(name, args, kw)
        if name in ('timeit.Timer', 'timeit.timeit', 'timeit.repeat'):
            stmt = args[0] if args else kw.get('stmt', 'pass')
            setup = args[1] if len(args) > 1 else kw.get('setup', 'pass')
            supplied = list(args[:2]) + [kw[k] for k in ('stmt', 'setup') if k in kw]
            if any(isinstance(v, str) for v in supplied):
                self.finding('code-execution', name)
            timer = Value('instance', 'timeit.Timer', (stmt, setup))
            if name != 'timeit.Timer':
                return self.bound(timer, 'timeit', (), {})
            return timer
        if name in ('bdb.Bdb.runcall', 'pdb.runcall', 'pdb.Pdb.runcall', 'profile.runctx',
                    'cProfile.Profile.runcall', 'profile.Profile.runcall'):
            return self.call(args[0], args[1:], kw)
        if name in ('zipfile.ZipFile', 'tarfile.open', 'tarfile.TarFile', 'gzip.open',
                    'gzip.GzipFile', 'bz2.open', 'bz2.BZ2File', 'lzma.open', 'lzma.LZMAFile'):
            source = args[0] if args else kw.get('fileobj', kw.get('filename'))
            if name == 'gzip.GzipFile' and kw.get('fileobj') is not None:
                source = kw['fileobj']
            if name.startswith('tarfile.') and kw.get('fileobj') is not None:
                source = kw['fileobj']
            if isinstance(source, (str, bytes)) or (isinstance(source, Value) and source.name.startswith('pathlib.')):
                self.finding('file-access', name)
            if isinstance(source, Value) and source.kind == 'buffer':
                if name.startswith(('gzip.', 'bz2.', 'lzma.')):
                    return Value('compressed_buffer', name.split('.')[0], (source,))
                return Value('archive_buffer', 'zip' if name == 'zipfile.ZipFile' else 'tar', (source,))
            return Value('instance', name, args, kw.copy())
        if name in ('pickle.dumps',):
            # Re-serialization is uncommon and requires reduction semantics.
            return UNKNOWN
        if name == 'urllib.request.urlopen':
            url = args[0] if args else kw.get('url')
            if isinstance(url, str) and url.startswith('data:'):
                from urllib.parse import unquote_to_bytes
                header, payload = url[5:].split(',', 1)
                data = unquote_to_bytes(payload)
                if header.endswith(';base64'):
                    data = base64.b64decode(data)
                return Value('buffer', 'io.BytesIO', (), {'data': data, 'pos': 0})
        if name in EFFECTS:
            if name in ('builtins.eval', 'builtins.exec'):
                source = args[0] if args else kw.get('source')
                if not isinstance(source, (str, bytes, bytearray)) and not (isinstance(source, Value) and source.kind == 'code'):
                    raise LoadError('source is not code or text')
            if name == 'builtins.compile':
                source = args[0] if args else kw.get('source')
                if not isinstance(source, (str, bytes, bytearray)) and not (isinstance(source, Value) and source.name.startswith('ast.')):
                    raise LoadError('source is not text or AST')
            if name in ('os.system', 'os.popen'):
                command = args[0] if args else kw.get('command', kw.get('cmd'))
                if not isinstance(command, (str, bytes)):
                    raise LoadError('command must be a string')
            self.validate_call(name, args, kw)
            # Missing required arguments stop the real call before any effect.
            zero_ok = ('os.fork', 'os.forkpty', 'pty.fork', 'os.listdir', 'os.scandir', 'tempfile.mkstemp',
                       'tempfile.mkdtemp', 'tempfile.NamedTemporaryFile',
                       'tempfile.TemporaryFile', 'tempfile.TemporaryDirectory')
            if not args and not kw and name not in zero_ok:
                raise LoadError('missing capability arguments')
            self.finding(EFFECTS[name], name)
        # Model data reconstruction is inert. Its state remains symbolic so a
        # capability stored there can still be reached by a later call.
        if name in ('numpy._core.multiarray._reconstruct',):
            return Value('instance', 'numpy.ndarray', args)
        if name.startswith('numpy.') and name.rsplit('.', 1)[-1] in ('scalar', '_frombuffer'):
            return Value('instance', 'numpy.ndarray' if name.endswith('_frombuffer') else 'numpy.scalar', args)
        if name == 'builtins.object':
            return Value('instance', name)
        # Data constructors used by model picklers. Unknown *executed*
        # callables do not get a silent clean verdict: fail closed if no
        # semantics or data-construction rule covers the call.
        module, _, leaf = name.rpartition('.')
        pure_types = ('builtins', 'collections', 'datetime', 'decimal', '_decimal',
                      'fractions', 'numbers', 'array', 'types', 'enum', 're', 'random')
        model_type = (module.startswith(('numpy', 'scipy', 'sklearn', 'pandas', 'torch')) and
                      (leaf[:1].isupper() or name in ('numpy.dtype', 'numpy.ndarray', 'torch.device', 'torch.Size')))
        rebuilding = (name.startswith('torch._utils._rebuild') or
                      name.startswith('torch._tensor._rebuild') or
                      name.startswith('numpy.random._pickle.') or
                      (module.startswith('pandas.') and ('unpickle' in leaf or leaf.startswith('_new_'))))
        if name in CLASS_NAMES or model_type or rebuilding or (module in pure_types and self.is_class(self.ref(name))):
            self.validate_call(name, args, kw)
            return Value('instance', name, args, kw.copy())
        raise LimitError('no static call semantics for ' + name)

    def adapter(self, obj, args, kw):
        if obj.name in ('codec_decode', 'codec_encode'):
            result = self.operations('codecs.decode' if obj.name == 'codec_decode' else 'codecs.encode',
                                     (args[0], obj.args[0]) + args[1:], kw)
            return result, len(args[0])
        if obj.name == 'reference':
            return obj.args[0]
        if obj.name == 'decorator':
            return Value('partial', 'wrapper', (args[0], (), {}))
        if obj.name == 'operator.attrgetter':
            result = []
            for path in obj.args:
                value = args[0]
                for part in path.split('.'):
                    value = self.attr(value, part)
                result.append(value)
            return result[0] if len(result) == 1 else tuple(result)
        if obj.name == 'operator.itemgetter':
            result = [self.getitem(args[0], k) for k in obj.args]
            return result[0] if len(result) == 1 else tuple(result)
        if obj.name == 'operator.methodcaller':
            return self.call(self.attr(args[0], obj.args[0]), obj.args[1:], obj.attrs)
        return UNKNOWN

    def truth(self, obj):
        if isinstance(obj, Value):
            return True
        return bool(obj)

    def iterate(self, obj):
        if not isinstance(obj, Value):
            try:
                iterator = iter(obj)
            except TypeError:
                raise LoadError('not iterable') from None
            for val in iterator:
                self.ctx.tick()
                yield val
            return
        if obj.kind == 'buffer':
            yield from self.iterate(obj.attrs['data'][obj.attrs['pos']:].splitlines(True))
            obj.attrs['pos'] = len(obj.attrs['data'])
            return
        if obj.kind == 'instance' and obj.name == 'numpy.ndarray':
            yield from self.iterate(obj.attrs.get('data', []))
            return
        if obj.kind != 'iterator':
            cls = obj.attrs.get('_class')
            if cls and '__iter__' in cls.attrs:
                yield from self.iterate(self.call(cls.attrs['__iter__'], (obj,)))
            return
        # Save one native generator over *symbolic* operations to preserve lazy
        # iterator position and avoid executing callbacks twice after next().
        if '_generator' not in obj.attrs:
            obj.attrs['_generator'] = self.iterator_body(obj)
        generator = obj.attrs['_generator']
        while True:
            self.ctx.tick()
            try:
                yield next(generator)
            except StopIteration:
                return

    def iterator_body(self, obj):
        name, args, kw = obj.name, obj.args, obj.attrs
        if name == 'builtins.map':
            for values in zip(*(self.iterate(v) for v in args[1:])):
                yield self.call(args[0], values)
        elif name == 'itertools.starmap':
            for values in self.iterate(args[1]):
                yield self.call(args[0], tuple(self.iterate(values)))
        elif name in ('builtins.filter', 'itertools.filterfalse', 'itertools.takewhile', 'itertools.dropwhile'):
            dropping = name.endswith('dropwhile')
            for val in self.iterate(args[1]):
                keep = self.truth(self.call(args[0], (val,))) if args[0] is not None else self.truth(val)
                if name.endswith('filterfalse'):
                    keep = not keep
                if name.endswith('takewhile') and not keep:
                    return
                if dropping:
                    if keep:
                        continue
                    dropping = False
                elif name.endswith('dropwhile'):
                    pass
                elif not keep:
                    continue
                yield val
        elif name == 'builtins.iter':
            if len(args) == 2:
                while True:
                    value = self.call(args[0], ())
                    if value == args[1]:
                        return
                    yield value
            else:
                yield from self.iterate(args[0])
        elif name in ('builtins.zip', 'itertools.zip_longest'):
            if name.endswith('zip_longest'):
                import itertools
                yield from itertools.zip_longest(*(self.iterate(v) for v in args), fillvalue=kw.get('fillvalue'))
            else:
                yield from zip(*(self.iterate(v) for v in args))
        elif name == 'builtins.enumerate':
            yield from enumerate(self.iterate(args[0]), args[1] if len(args) > 1 else 0)
        elif name == 'builtins.reversed':
            if isinstance(args[0], Value):
                raise LoadError('object is not reversible')
            yield from reversed(args[0])
        elif name in ('itertools.chain', 'itertools.chain.from_iterable'):
            groups = self.iterate(args[0]) if name.endswith('from_iterable') else args
            for group in groups:
                yield from self.iterate(group)
        elif name == 'itertools.islice':
            import itertools
            yield from itertools.islice(self.iterate(args[0]), *args[1:])
        elif name == 'itertools.repeat':
            count = args[1] if len(args) > 1 else 2_000_001
            for _ in range(count):
                yield args[0]
        elif name == 'itertools.count':
            count = args[0] if args else 0
            step = args[1] if len(args) > 1 else 1
            while True:
                yield count
                count += step
        elif name == 'itertools.cycle':
            saved = []
            for value in self.iterate(args[0]):
                saved.append(value)
                yield value
            while saved:
                yield from saved
        elif name == 'itertools.compress':
            for value, selector in zip(self.iterate(args[0]), self.iterate(args[1])):
                if self.truth(selector):
                    yield value
        elif name == 'itertools.accumulate':
            it = self.iterate(args[0])
            total = kw.get('initial', MISSING)
            if total is MISSING:
                try:
                    total = next(it)
                except StopIteration:
                    return
            yield total
            fn = args[1] if len(args) > 1 else kw.get('func')
            for value in it:
                total = self.call(fn, (total, value)) if fn else self.operations('operator.add', (total, value), {})
                yield total
        elif name == 'itertools.groupby':
            fn = args[1] if len(args) > 1 else kw.get('key')
            for value in self.iterate(args[0]):
                key = self.call(fn, (value,)) if fn else value
                yield (key, Value('iterator', 'builtins.iter', ([value],)))
        elif name in ('itertools.product', 'itertools.permutations', 'itertools.combinations',
                      'itertools.combinations_with_replacement'):
            import itertools
            options = {k: v for k, v in kw.items() if not k.startswith('_')}
            yield from getattr(itertools, name.split('.')[1])(*args, **options)
        elif name == 'itertools.pairwise':
            it = self.iterate(args[0])
            try:
                previous = next(it)
            except StopIteration:
                return
            for value in it:
                yield previous, value
                previous = value

    def bound(self, obj, method, args, kw):
        if method == '__enter__':
            return obj
        if method in ('__exit__', 'close') and isinstance(obj, Value) and obj.kind == 'buffer':
            return None
        if method == '__iter__':
            return obj if isinstance(obj, Value) and obj.kind == 'iterator' else Value('iterator', 'builtins.iter', (obj,))
        if method == '__call__':
            return self.call(obj, args, kw)
        if method in ('__getattribute__', '__getattr__'):
            return self.attr(obj, *args)
        if method == '__setattr__':
            if isinstance(obj, Value):
                obj.attrs[args[0]] = args[1]
            return None
        if method == '__getitem__':
            return self.getitem(obj, *args)
        if method == '__setitem__':
            self.setitem(obj, *args)
            return None
        if method == '__setstate__':
            if isinstance(obj, Value) and obj.kind == 'buffer':
                data, pos, attrs = args[0]
                obj.attrs.update({'data': data, 'pos': pos})
                if attrs:
                    obj.attrs.update(attrs)
            else:
                self.build(obj, args[0])
            return None
        if method in ('__add__', '__radd__', '__mul__', '__rmul__'):
            operands = (args[0], obj) if method.startswith('__r') else (obj, args[0])
            return self.operations('operator.mul' if 'mul' in method else 'operator.add', operands, kw)
        if isinstance(obj, Value):
            if obj.kind == 'compressed_buffer':
                source = obj.args[0]
                if '_buffer' not in obj.attrs:
                    data = self.decompress(obj.name, source.attrs['data'][source.attrs['pos']:])
                    obj.attrs['_buffer'] = Value('buffer', 'io.BytesIO', (), {'data': data, 'pos': 0})
                return self.bound(obj.attrs['_buffer'], method, args, kw)
            if obj.kind == 'archive_buffer':
                if method in ('extract', 'extractall'):
                    self.finding('file-access', obj.name + '.' + method)
                source = obj.args[0].attrs['data']
                if obj.name == 'zip':
                    import zipfile
                    with zipfile.ZipFile(io.BytesIO(source)) as zf:
                        if method == 'namelist':
                            return zf.namelist()
                        if method in ('read', 'open'):
                            info = zf.getinfo(args[0])
                            self.ctx.allocate(info.file_size)
                            data = zf.read(info)
                            return data if method == 'read' else Value('buffer', 'io.BytesIO', (), {'data': data, 'pos': 0})
                else:
                    import tarfile
                    with tarfile.open(fileobj=io.BytesIO(source)) as tf:
                        if method in ('getnames',):
                            return tf.getnames()
                        if method == 'extractfile':
                            member = tf.getmember(args[0])
                            self.ctx.allocate(member.size)
                            return Value('buffer', 'io.BytesIO', (), {'data': tf.extractfile(member).read(), 'pos': 0})
                return UNKNOWN
            if obj.kind == 'buffer':
                data, pos = obj.attrs['data'], obj.attrs['pos']
                if method in ('getvalue', 'getbuffer'):
                    return data
                if method in ('read', 'read1', 'readline'):
                    count = args[0] if args else -1
                    end = len(data) if count < 0 else min(len(data), pos + count)
                    if method == 'readline':
                        newline = data.find(b'\n' if isinstance(data, bytes) else '\n', pos, end)
                        if newline >= 0:
                            end = newline + 1
                    obj.attrs['pos'] = end
                    return data[pos:end]
                if method == 'seek':
                    offset = args[0]
                    whence = args[1] if len(args) > 1 else 0
                    obj.attrs['pos'] = max(0, offset + (pos if whence == 1 else len(data) if whence == 2 else 0))
                    return obj.attrs['pos']
                if method == 'tell':
                    return pos
                if method == 'write':
                    val = args[0]
                    self.ctx.allocate(len(data) + len(val))
                    obj.attrs['data'] = data[:pos] + val + data[pos + len(val):]
                    obj.attrs['pos'] += len(val)
                    return len(val)
                return None
            if obj.kind in ('module_dict', 'modules'):
                if method in ('get', '__getitem__', 'pop'):
                    return self.getitem(obj, args[0])
                if method in ('update', 'setdefault'):
                    if method == 'setdefault':
                        val = self.getitem(obj, args[0])
                        return val
                    values = args[0] if args else {}
                    for key, value in values.items():
                        self.setitem(obj, key, value)
                    for key, value in kw.items():
                        self.setitem(obj, key, value)
                    return None
                if method in ('items', 'keys', 'values') and obj.kind == 'module_dict':
                    return list(getattr(obj.args[0].attrs, method)())
            if obj.kind == 'decompressor' and method in ('decompress', 'flush'):
                if not args:
                    return b''
                return self.decompress(obj.name, args[0], obj.args, obj.attrs)
            if obj.kind == 'instance':
                name = obj.name
                canonical = name
                if name in ('pathlib.PosixPath', 'pathlib.WindowsPath'):
                    canonical = 'pathlib.Path'
                if name in ('socket.SocketType',):
                    canonical = 'socket.socket'
                if method == 'absolute' and name.startswith('pathlib.'):
                    return Value('instance', name, (), {'path': posixpath.abspath(obj.attrs.get('path', '.'))})
                if method == '__fspath__' and name.startswith('pathlib.'):
                    return obj.attrs.get('path', '')
                if method in ('joinpath', 'with_name', 'with_suffix') and name.startswith('pathlib.'):
                    path = obj.attrs.get('path', '')
                    if method == 'joinpath':
                        path = posixpath.join(path, *args)
                    elif method == 'with_name':
                        path = posixpath.join(posixpath.dirname(path), args[0])
                    else:
                        path = posixpath.splitext(path)[0] + args[0]
                    return Value('instance', name, (), {'path': path})
                if name in ('logging.config.DictConfigurator', 'logging.config.BaseConfigurator'):
                    if method == 'configure':
                        return self.logging_config('logging.config.dictConfig', (obj.attrs.get('config', {}),), {})
                    if method == 'resolve':
                        return self.operations('pkgutil.resolve_name', args, kw)
                if name.startswith('webbrowser.') and method in ('open', 'open_new', 'open_new_tab'):
                    self.finding('network', name + '.' + method)
                if name == 'xmlrpc.client.ServerProxy' and method == '_ServerProxy__request':
                    self.finding('network', name + '.' + method)
                if name == 'contextlib.ExitStack':
                    callbacks = obj.attrs.setdefault('callbacks', [])
                    if method == 'callback':
                        callbacks.append((args[0], args[1:], kw))
                        return args[0]
                    if method == 'push':
                        callbacks.append((args[0], (None, None, None), {}))
                        return args[0]
                    if method in ('close', '__exit__'):
                        while callbacks:
                            fn, vals, opts = callbacks.pop()
                            self.call(fn, vals, opts)
                        return False
                if name == 'multiprocessing.pool.ThreadPool':
                    if method in ('apply', 'apply_async'):
                        vals = args[1] if len(args) > 1 else kw.get('args', ())
                        opts = args[2] if len(args) > 2 else kw.get('kwds', {})
                        return self.call(args[0], tuple(vals), opts)
                    if method in ('map', 'starmap', 'map_async', 'starmap_async'):
                        for val in self.iterate(args[1]):
                            self.call(args[0], tuple(val) if 'starmap' in method else (val,))
                        return []
                    if method in ('imap', 'imap_unordered'):
                        return Value('iterator', 'builtins.map', args[:2])
                if name == 'pickle.Unpickler' and method == 'load':
                    return self.nested('pickle.load', obj.args, obj.attrs)
                if name in ('threading.Thread', 'threading.Timer', 'multiprocessing.Process',
                            'multiprocessing.context.Process') and method in ('start', 'run'):
                    if method == 'start' and name.startswith('multiprocessing.'):
                        self.finding('process', name + '.start')
                    target = obj.attrs.get('_target', obj.attrs.get('function'))
                    if target:
                        return self.call(target, tuple(obj.attrs.get('_args', obj.attrs.get('args', ()))),
                                         obj.attrs.get('_kwargs', obj.attrs.get('kwargs', {})))
                    return None
                if name.endswith(('ThreadPoolExecutor', 'ProcessPoolExecutor')) and method in ('submit', 'map'):
                    if name.endswith('ProcessPoolExecutor'):
                        self.finding('process', name + '.' + method)
                    init = obj.attrs.get('initializer')
                    if init:
                        self.call(init, tuple(obj.attrs.get('initargs', ())))
                    if method == 'submit':
                        result = self.call(args[0], args[1:], kw)
                        return Value('instance', 'concurrent.futures.Future', (), {'result': result})
                    for vals in zip(*(self.iterate(v) for v in args[1:])):
                        self.call(args[0], vals)
                    return Value('iterator', 'builtins.iter', ([],))
                if name == 'concurrent.futures.Future' and method == 'result':
                    return obj.attrs.get('result', UNKNOWN)
                if name == 'xmlrpc.client.ServerProxy' and not method.startswith('_'):
                    self.finding('network', name + '.' + method)
                if name in ('requests.Session', 'requests.sessions.Session', 'httpx.Client', 'httpx.AsyncClient') and method in ('get', 'post', 'put', 'delete', 'head', 'patch', 'request', 'send'):
                    self.finding('network', name + '.' + method)
                if canonical == 'timeit.Timer' and method in ('timeit', 'repeat', 'autorange'):
                    for part in obj.args:
                        if isinstance(part, Value):
                            self.call(part, ())
                    return 0.0
                if method == 'runcall' and name in ('bdb.Bdb', 'pdb.Pdb', 'cProfile.Profile', 'profile.Profile'):
                    return self.call(args[0], args[1:], kw)
                target = canonical + '.' + method
                if target in EFFECTS:
                    self.finding(EFFECTS[target], target)
                if target in NESTED:
                    return self.nested(target, args, kw)
                # Inherited Python methods in the supported client types.
                if name == 'ftplib.FTP_TLS' and 'ftplib.FTP.' + method in EFFECTS:
                    self.finding(EFFECTS['ftplib.FTP.' + method], target)
                if method in ('read', 'write', 'readline', 'writelines', 'truncate', 'readinto') and name in ('io.FileIO', 'io.TextIOWrapper', 'io.BufferedReader', 'io.BufferedWriter'):
                    self.finding('file-access', target)
                cls = obj.attrs.get('_class')
                if cls and method in cls.attrs:
                    return self.call(cls.attrs[method], (obj,) + args, kw)
                return UNKNOWN
            if obj.kind == 'iterator' and method == '__next__':
                return self.operations('builtins.next', (obj,), kw)
            if obj.kind == 'partial' and method == '__reduce__':
                return (self.ref('functools.partial'), (obj.args[0],), (*obj.args, obj.attrs))
            return UNKNOWN
        # All native method calls below have exact builtin receivers and inert
        # arguments. Never dispatch a method on a deserialized Python object.
        if isinstance(obj, dict):
            if method == 'get':
                return obj.get(*args)
            if method == 'setdefault':
                return obj.setdefault(*args)
            if method in ('keys', 'values', 'items'):
                return list(getattr(obj, method)())
            if method == 'update':
                source = args[0] if args else {}
                if isinstance(source, dict):
                    obj.update(source)
                else:
                    for k, v in self.iterate(source):
                        obj[k] = v
                obj.update(kw)
                return None
            if method in ('pop', 'popitem', 'copy', 'clear'):
                return getattr(obj, method)(*args)
        if isinstance(obj, (list, set)):
            if method in ('append', 'add'):
                (obj.append if isinstance(obj, list) else obj.add)(args[0])
                return None
            if method in ('extend', 'update'):
                for value in self.iterate(args[0]):
                    (obj.append if isinstance(obj, list) else obj.add)(value)
                return None
            if method == 'sort':
                obj[:] = self.operations('builtins.sorted', (obj,), kw)
                return None
            if method in ('pop', 'remove', 'discard', 'clear', 'reverse', 'insert', 'copy'):
                return getattr(obj, method)(*args)
        if isinstance(obj, (str, bytes, bytearray)) and method in (
            'encode', 'decode', 'join', 'replace', 'split', 'rsplit', 'splitlines',
            'strip', 'lstrip', 'rstrip', 'lower', 'upper', 'capitalize', 'title',
            'swapcase', 'casefold', 'removeprefix', 'removesuffix', 'partition',
            'rpartition', 'format', 'format_map', 'hex', 'fromhex', 'startswith',
            'endswith', 'find', 'rfind', 'count', 'index', 'translate', 'maketrans'):
            if method == 'join' and args and isinstance(args[0], Value):
                args = (list(self.iterate(args[0])),) + args[1:]
            if literal(args) and literal(kw):
                value = getattr(obj, method)(*args, **kw)
                if isinstance(value, (str, bytes, bytearray)):
                    self.ctx.allocate(len(value))
                return value
        return UNKNOWN

    def decompress(self, kind, data, args=(), kw=None):
        if not isinstance(data, (bytes, bytearray)):
            raise LoadError('compressed input must be bytes')
        from modelgate.dispatch import decompress
        try:
            return decompress(bytes(data), kind, self.ctx, args, kw or {})
        except (Detected, LimitError, LoadError):
            raise
        except Exception as exc:
            raise LoadError('invalid compressed data: ' + str(exc)) from None

    def nested(self, name, args, kw):
        if not args:
            source = kw.get('file', kw.get('f', kw.get('filename', MISSING)))
            if source is MISSING:
                raise LoadError('nested loader has no input')
        else:
            source = args[0]
        raw = name in ('pickle.loads', 'torch.storage._load_from_bytes')
        buffer = isinstance(source, Value) and source.kind == 'buffer'
        if raw:
            if not isinstance(source, (bytes, bytearray)):
                raise LoadError('loads requires bytes')
            data = bytes(source)
        elif buffer:
            data = source.attrs['data'][source.attrs['pos']:]
            if not isinstance(data, bytes):
                raise LoadError('binary stream required')
        elif isinstance(source, (str, bytes)) or (isinstance(source, Value) and source.name.startswith('pathlib.')):
            self.finding('file-access', name + '(path)')
        else:
            raise LimitError('nested loader input is not statically known')
        self.ctx.nesting += 1
        try:
            if self.ctx.nesting > 30:
                raise LimitError('nested deserialization depth limit')
            from modelgate.pickle_scan import Interpreter
            where = self.location + '!' + name
            if name in ('pickle.loads', 'pickle.load'):
                machine = Interpreter(self.ctx, where, encoding=kw.get('encoding', 'ASCII'),
                                      errors=kw.get('errors', 'strict'), fix_imports=kw.get('fix_imports', True))
                result, end = machine.one(data)
                if buffer:
                    source.attrs['pos'] += end
                return result
            from modelgate.dispatch import analyze_loaded
            result = analyze_loaded(data, where, self.ctx, name, args[1:], kw)
            if buffer:
                source.attrs['pos'] = len(source.attrs['data'])
            return result
        finally:
            self.ctx.nesting -= 1

    def logging_config(self, name, args, kw):
        config = args[0] if args else kw.get('config')
        if name.endswith('fileConfig'):
            if isinstance(config, (str, bytes)):
                self.finding('file-access', name)
            raise LimitError('fileConfig from an in-memory stream')
        if not isinstance(config, dict) or config.get('version') != 1:
            raise LoadError('invalid logging configuration')
        if config.get('incremental'):
            return None
        def convert(value):
            if isinstance(value, str) and value.startswith('ext://'):
                return self.operations('pkgutil.resolve_name', (value[6:],), {})
            return value
        for group in ('formatters', 'filters', 'handlers'):
            entries = config.get(group, {})
            if not isinstance(entries, dict):
                raise LoadError('invalid logging configuration group')
            for spec in entries.values():
                if not isinstance(spec, dict):
                    raise LoadError('invalid logging factory')
                factory = spec.get('()', spec.get('class') if group == 'handlers' else None)
                if factory is None:
                    continue
                if isinstance(factory, str):
                    factory = self.operations('pkgutil.resolve_name', (factory.removeprefix('ext://'),), {})
                params = {k: convert(v) for k, v in spec.items()
                          if k not in ('()', 'class', '.', 'level', 'formatter', 'filters')}
                self.call(factory, (), params)
        return None
