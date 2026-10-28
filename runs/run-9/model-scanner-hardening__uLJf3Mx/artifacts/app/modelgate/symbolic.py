"""A small abstract Python runtime for pickle's calls.

Artifact globals are *names*, never imported objects. The only operations
performed here are explicit operations on scanner-owned primitive data. All
other calls are modelled, including lazy callbacks and reentrant loaders.
"""
from __future__ import annotations

import _compat_pickle
import base64
import binascii
import codecs
import io
import math
import operator
import re
from dataclasses import dataclass, field

from modelgate.limits import Budget, LimitError, MAX_BYTES, MAX_ITEMS, decompress
from modelgate.report import Finding


class LoadError(Exception):
    """The standard loader stops here, before any subsequent opcode runs."""


class AnalysisError(Exception):
    """An operation cannot be resolved statically."""


@dataclass(eq=False)
class Symbol:
    kind: str
    name: str = ""
    args: tuple = ()
    attrs: dict = field(default_factory=dict)


UNKNOWN = Symbol("unknown")
MISSING = object()


class MappingValue(dict):
    """A scanner-owned mapping that also models an instance dictionary."""
    def __init__(self, name, items=()):
        super().__init__(items)
        self.name = name
        self.attrs = {}


def ref(name):
    return Symbol("ref", name)


def primitive(value, depth=0):
    if depth > 40:
        return False
    if type(value) in (str, bytes, bytearray, int, float, complex, bool, type(None), range, slice):
        return True
    if type(value) in (tuple, list, set, frozenset):
        return all(primitive(v, depth + 1) for v in value)
    if type(value) is dict:
        return all(primitive(k, depth + 1) and primitive(v, depth + 1) for k, v in value.items())
    return False


ALIASES = {"posix": "os", "nt": "os", "_pickle": "pickle", "_io": "io",
           "_operator": "operator", "_functools": "functools", "_socket": "socket"}
CALL_ALIASES = {
    "socket.SocketType": "socket.socket",
    "types.LambdaType": "types.FunctionType",
    "pathlib.PosixPath": "pathlib.Path", "pathlib.WindowsPath": "pathlib.Path",
    "pickle._Unpickler": "pickle.Unpickler",
    "code.InteractiveConsole.runsource": "code.InteractiveInterpreter.runsource",
    "code.InteractiveConsole.runcode": "code.InteractiveInterpreter.runcode",
    "pdb.Pdb.run": "bdb.Bdb.run", "pdb.Pdb.runeval": "bdb.Bdb.runeval",
    "pdb.Pdb.runcall": "bdb.Bdb.runcall",
}
MODULE_ATTRS = {
    "shutil": {"os": "os", "sys": "sys", "stat": "stat"},
    "subprocess": {"os": "os", "sys": "sys", "io": "io"},
    "pickle": {"sys": "sys", "io": "io", "builtins": "builtins"},
    "pathlib": {"os": "os", "io": "io", "sys": "sys"},
    "platform": {"os": "os", "sys": "sys", "subprocess": "subprocess"},
    "tempfile": {"_os": "os", "_io": "io"},
    "random": {"_os": "os"},
    "pydoc": {"os": "os", "sys": "sys", "subprocess": "subprocess"},
    "pkgutil": {"sys": "sys", "os": "os", "importlib": "importlib"},
    "importlib": {"sys": "sys"},
    "os": {"path": "posixpath", "sys": "sys"},
    "posixpath": {"os": "os", "sys": "sys"},
    "ntpath": {"os": "os", "sys": "sys"},
    "code": {"sys": "sys", "traceback": "traceback"},
    "codecs": {"sys": "sys", "builtins": "builtins"},
    "numpy": {"lib": "numpy.lib", "core": "numpy.core", "_core": "numpy._core"},
    "numpy.lib": {"format": "numpy.lib.format"},
    "urllib": {"request": "urllib.request", "parse": "urllib.parse"},
    "http": {"client": "http.client", "server": "http.server"},
    "xmlrpc": {"client": "xmlrpc.client"},
    "multiprocessing": {"connection": "multiprocessing.connection", "pool": "multiprocessing.pool"},
    "concurrent": {"futures": "concurrent.futures"},
    "logging": {"config": "logging.config", "handlers": "logging.handlers"},
    "torch": {"storage": "torch.storage", "serialization": "torch.serialization"},
}

CODE_CALLS = set("""
 builtins.eval builtins.exec builtins.compile runpy.run_path runpy.run_module
 runpy._run_code runpy._run_module_code runpy._run_module_as_main
 timeit.timeit timeit.repeat code.InteractiveInterpreter.runsource
 code.InteractiveInterpreter.runcode code.InteractiveConsole.push
 bdb.Bdb.run bdb.Bdb.runeval pdb.run pdb.runeval
 numpy.testing._private.utils.runstring numpy.testing.utils.runstring
 types.FunctionType types.LambdaType cloudpickle.cloudpickle._make_function
 dill._dill._create_function dill._dill._create_code
""".split())
PROCESS_CALLS = set("""
 os.system os.popen os.fork os.forkpty os.posix_spawn os.posix_spawnp
 subprocess.Popen subprocess.call subprocess.run subprocess.check_call
 subprocess.check_output subprocess.getoutput subprocess.getstatusoutput
 pty.spawn pydoc.pipepager pydoc.tempfilepager asyncio.create_subprocess_exec
 asyncio.create_subprocess_shell multiprocessing.Pool multiprocessing.pool.Pool
 multiprocessing.popen_fork.Popen multiprocessing.popen_spawn_posix.Popen
 multiprocessing.popen_spawn_win32.Popen
""".split())
FILE_CALLS = set("""
 builtins.open io.open io.open_code io.FileIO io.BufferedReader io.BufferedWriter
 os.open os.fdopen os.remove os.unlink os.rename os.renames os.replace os.mkdir
 os.makedirs os.rmdir os.removedirs os.listdir os.scandir os.walk os.fwalk
 os.stat os.lstat os.access os.chmod os.chown os.lchown os.chroot os.chdir
 os.truncate os.utime os.link os.symlink os.readlink os.mknod os.mkfifo
 os.read os.write os.readv os.writev os.pread os.pwrite os.sendfile
 shutil.copy shutil.copy2 shutil.copyfile shutil.copytree shutil.copymode
 shutil.copystat shutil.move shutil.rmtree shutil.make_archive shutil.unpack_archive
 tempfile.mkstemp tempfile.mkdtemp tempfile.NamedTemporaryFile tempfile.TemporaryFile
 tempfile.TemporaryDirectory glob.glob glob.iglob
 posixpath.exists posixpath.lexists posixpath.isfile posixpath.isdir posixpath.islink
 posixpath.getsize posixpath.getmtime posixpath.getatime posixpath.getctime
 ntpath.exists ntpath.lexists ntpath.isfile ntpath.isdir ntpath.islink
 pathlib.Path.open pathlib.Path.read_text pathlib.Path.read_bytes
 pathlib.Path.write_text pathlib.Path.write_bytes pathlib.Path.unlink
 pathlib.Path.rename pathlib.Path.replace pathlib.Path.mkdir pathlib.Path.rmdir
 pathlib.Path.touch pathlib.Path.iterdir pathlib.Path.glob pathlib.Path.rglob
 pathlib.Path.stat pathlib.Path.lstat pathlib.Path.chmod pathlib.Path.lchmod
 pathlib.Path.symlink_to pathlib.Path.hardlink_to pathlib.Path.link_to
 pathlib.Path.exists pathlib.Path.is_file pathlib.Path.is_dir pathlib.Path.is_symlink
 pathlib.Path.resolve pathlib.Path.readlink pathlib.Path.owner pathlib.Path.group
 marshal.load marshal.dump json.load json.dump
 numpy.fromfile numpy.memmap numpy.loadtxt numpy.genfromtxt numpy.savetxt numpy.save numpy.savez
 numpy.savez_compressed pandas.read_csv pandas.read_json pandas.read_table pandas.read_parquet
 pandas.read_hdf pandas.read_excel pandas.read_feather pandas.read_sql
 sqlite3.connect shelve.open dbm.open dbm.dumb.open dbm.gnu.open dbm.ndbm.open
 logging.FileHandler logging.handlers.RotatingFileHandler logging.handlers.WatchedFileHandler
""".split())
NETWORK_CALLS = set("""
 socket.create_connection socket.socket.connect socket.socket.connect_ex
 socket.socket.sendto socket.socket.sendmsg socket.getaddrinfo socket.gethostbyname
 socket.gethostbyname_ex socket.gethostbyaddr socket.getnameinfo
 urllib.request.urlopen urllib.request.urlretrieve urllib.request.OpenerDirector.open
 http.client.HTTPConnection.connect http.client.HTTPConnection.request
 http.client.HTTPConnection.endheaders http.client.HTTPConnection.send
 http.client.HTTPSConnection.connect http.client.HTTPSConnection.request
 http.client.HTTPSConnection.endheaders http.client.HTTPSConnection.send
 ftplib.FTP.connect ftplib.FTP_TLS.connect ftplib.FTP.login ftplib.FTP.retrbinary
 smtplib.SMTP.connect smtplib.SMTP_SSL.connect smtplib.SMTP.sendmail
 ssl.get_server_certificate webbrowser.open webbrowser.open_new webbrowser.open_new_tab
 requests.get requests.post requests.put requests.delete requests.head requests.request
 urllib3.PoolManager.request httpx.get httpx.post httpx.request
 asyncio.open_connection multiprocessing.connection.Client
""".split())
NESTED = set("""
 pickle.loads pickle.load pickle.Unpickler.load torch.load torch.serialization.load
 torch.storage._load_from_bytes numpy.load numpy.lib.npyio.load
 numpy.lib._npyio_impl.load numpy.lib.format.read_array joblib.load
 joblib.numpy_pickle.load pandas.read_pickle pandas.io.pickle.read_pickle
""".split())

KNOWN_CLASSES = set("""
 functools.partial operator.attrgetter operator.itemgetter operator.methodcaller
 socket.socket socket.SocketType http.client.HTTPConnection http.client.HTTPSConnection
 ftplib.FTP ftplib.FTP_TLS smtplib.SMTP smtplib.SMTP_SSL
 xmlrpc.client.ServerProxy xmlrpc.client._Method subprocess.Popen
 pathlib.Path pathlib.PosixPath pathlib.WindowsPath pathlib.PurePath
 pathlib.PurePosixPath pathlib.PureWindowsPath io.BytesIO io.StringIO io.FileIO
 pickle.Unpickler pickle._Unpickler code.InteractiveInterpreter code.InteractiveConsole
 bdb.Bdb pdb.Pdb timeit.Timer threading.Thread threading.Timer
 concurrent.futures.ThreadPoolExecutor concurrent.futures.ProcessPoolExecutor
 multiprocessing.Process multiprocessing.context.Process multiprocessing.pool.ThreadPool
 types.FunctionType types.CodeType types.ModuleType types.SimpleNamespace
""".split())


class Context:
    def __init__(self):
        self.findings = []
        self.budget = Budget()
        self.modules = {}
        self.module_dict = Symbol("modules")
        self.call_depth = 0
        self.load_depth = 0
        self.loc = "pickle"
        self.op = "call"

    def finding(self, kind, detail):
        if len(self.findings) < 100:
            self.findings.append(Finding(kind, self.loc, f"{self.op}: {detail}"))

    def module(self, name):
        name = ALIASES.get(name, name)
        if name not in self.modules:
            self.modules[name] = Symbol("module", name)
        return self.modules[name]

    def global_ref(self, module, name, protocol=4):
        if not isinstance(module, str) or not isinstance(name, str):
            raise LoadError("global requires string names")
        if protocol < 3:
            module, name = _compat_pickle.NAME_MAPPING.get((module, name),
                (_compat_pickle.IMPORT_MAPPING.get(module, module), name))
        if "." in name and protocol < 4:
            raise LoadError("dotted global below protocol 4")
        value = self.module(module)
        for part in name.split("."):
            value = self.attr(value, part)
        return value

    def resolve(self, name):
        if not isinstance(name, str):
            raise LoadError("expected a qualified name")
        name = name.replace(":", ".")
        parts = name.split(".")
        value = self.module(parts[0])
        for part in parts[1:]:
            value = self.attr(value, part)
        return value

    def attr(self, obj, name, default=MISSING):
        if not isinstance(name, str):
            raise LoadError("attribute name is not a string")
        if isinstance(obj, MappingValue):
            if name == "__dict__": return obj.attrs
            if name in obj.attrs: return obj.attrs[name]
            return Symbol("method", "builtins.dict." + name, (obj,))
        if isinstance(obj, Symbol):
            if name in obj.attrs:
                return obj.attrs[name]
            if name == "__dict__":
                return Symbol("namespace", args=(obj,)) if obj.kind in ("module", "ref") else obj.attrs
            if name == "__class__":
                if obj.kind == "module":
                    return ref("types.ModuleType")
                if obj.kind == "partial":
                    return ref("functools.partial")
                return ref(obj.name) if obj.kind == "instance" else ref("builtins.type")
            if obj.kind == "module":
                if name == "__name__":
                    return obj.name
                if obj.name == "sys" and name == "modules":
                    return self.module_dict
                if name in ("__builtins__", "builtins"):
                    return self.module("builtins")
                if name in MODULE_ATTRS.get(obj.name, {}):
                    return self.module(MODULE_ATTRS[obj.name][name])
                return ref(obj.name + "." + name)
            if obj.kind == "ref":
                if name in ("__call__", "__func__"):
                    return obj
                if name == "__module__":
                    return obj.name.rsplit(".", 1)[0]
                if name in ("__name__", "__qualname__"):
                    return obj.name.rsplit(".", 1)[-1]
                if name == "__globals__":
                    return Symbol("globals", obj.name.rsplit(".", 1)[0])
                return ref(obj.name + "." + name)
            if obj.kind == "partial":
                if name == "func": return obj.args[0]
                if name == "args": return obj.args[1]
                if name == "keywords": return obj.args[2]
            if obj.kind == "wrapper" and name == "__wrapped__":
                return obj.args[0]
            if name == "__call__" and obj.kind in ("partial", "wrapper", "getter", "methodcaller", "function"):
                return obj
            if obj.kind == "instance" and obj.name in ("pathlib.PosixPath", "pathlib.WindowsPath"):
                return Symbol("method", "pathlib.Path." + name, (obj,))
            if obj.kind == "instance" and obj.name == "xmlrpc.client.ServerProxy" and not name.startswith("_"):
                return Symbol("remote", name, (obj,))
            if obj.kind == "instance":
                return Symbol("method", obj.name + "." + name, (obj,))
            if obj.kind in ("buffer", "iterator", "partial", "modules", "globals", "namespace"):
                return Symbol("method", obj.kind + "." + name, (obj,))
            if default is not MISSING:
                return default
            return UNKNOWN
        if name == "__class__":
            return ref("builtins." + type(obj).__name__)
        if type(obj) in (str, bytes, bytearray, int, float, complex, bool, list, tuple, dict, set, frozenset, range):
            if name == "__dict__":
                if default is not MISSING: return default
                raise LoadError("primitive has no __dict__")
            # Attribute lookup does not call these methods; invocation below is
            # limited to explicitly implemented, side-effect-free operations.
            return Symbol("method", "builtins." + type(obj).__name__ + "." + name, (obj,))
        if default is not MISSING:
            return default
        return UNKNOWN

    def item(self, obj, key, default=MISSING):
        if isinstance(obj, Symbol):
            if obj.kind == "namespace":
                return self.attr(obj.args[0], key, default)
            if obj.kind == "modules" and isinstance(key, str):
                return self.module(key)
            if obj.kind == "globals" and isinstance(key, str):
                if key in ("__builtins__", "builtins"):
                    return self.module("builtins")
                return self.attr(self.module(obj.name), key)
            if obj.kind == "instance" and obj.name in ("collections.UserDict", "collections.defaultdict"):
                return self.item(obj.attrs.get("data", {}), key, default)
            if obj.kind == "instance" and obj.name.startswith("numpy"):
                return self.item(obj.attrs.get("data", UNKNOWN), key, default)
            return UNKNOWN
        try:
            return obj[key]
        except (TypeError, KeyError, IndexError):
            if default is not MISSING:
                return default
            raise LoadError("invalid item lookup") from None

    def setitem(self, obj, key, value):
        if isinstance(obj, Symbol):
            if obj.kind == "namespace":
                obj.args[0].attrs[key] = value
            elif obj.kind == "modules" and isinstance(key, str):
                self.modules[ALIASES.get(key, key)] = value
            else:
                obj.attrs.setdefault("data", {})[key] = value
        else:
            try:
                obj[key] = value
            except (TypeError, IndexError):
                raise LoadError("invalid item assignment") from None

    def is_class(self, value):
        if not isinstance(value, Symbol): return False
        if value.kind == "class": return True
        if value.kind != "ref": return False
        name = value.name
        return (name in KNOWN_CLASSES or name.startswith("builtins.") and
                name.rsplit(".", 1)[-1] in {"list", "tuple", "dict", "set", "frozenset", "str", "bytes", "bytearray", "int", "float", "bool", "complex", "object", "type", "map", "filter", "zip", "range", "slice"}
                or name.rsplit(".", 1)[-1][:1].isupper())

    def newobj(self, cls, args, kwargs=None):
        if not isinstance(args, tuple) or kwargs is not None and not isinstance(kwargs, dict):
            raise LoadError("invalid NEWOBJ arguments")
        if not self.is_class(cls):
            raise LoadError("NEWOBJ target is not a class")
        if cls.kind == "class":
            new = cls.attrs.get("__new__")
            if new is not None:
                return self.call(new, (cls,) + args, kwargs or {})
            return Symbol("instance", cls.name, args, {"__class_symbol__": cls})
        if cls.name in ("functools.partial", "operator.attrgetter", "operator.itemgetter", "operator.methodcaller", "types.FunctionType", "types.CodeType", "builtins.bytes", "builtins.bytearray", "builtins.str", "builtins.tuple", "builtins.frozenset"):
            return self.call(cls, args, kwargs or {})
        return Symbol("instance", cls.name, args)

    def build(self, obj, state):
        if isinstance(obj, MappingValue):
            if isinstance(state, dict): obj.attrs.update(state)
            elif isinstance(state, tuple) and len(state) == 2:
                for part in state:
                    if isinstance(part, dict): obj.attrs.update(part)
            return
        if isinstance(obj, Symbol):
            if obj.kind == "partial" or obj.name == "functools.partial":
                if not isinstance(state, tuple) or len(state) != 4 or not isinstance(state[1], tuple):
                    raise LoadError("invalid partial state")
                obj.kind = "partial"
                obj.args = (state[0], state[1], state[2] or {})
                if isinstance(state[3], dict): obj.attrs.update(state[3])
                return
            if "__setstate__" in obj.attrs:
                self.call(obj.attrs["__setstate__"], (state,))
                return
            cls = obj.attrs.get("__class_symbol__")
            setter = cls.attrs.get("__setstate__") if isinstance(cls, Symbol) else None
            if setter:
                self.call(setter, (obj, state))
                return
            if obj.name in ("numpy.dtype",):
                obj.attrs["state"] = state
                return
            if obj.name in ("numpy.ndarray", "numpy.matrix"):
                obj.attrs["state"] = state
                if isinstance(state, tuple) and len(state) >= 5:
                    obj.attrs.update(shape=state[1], dtype=state[2], data=state[4])
                return
            if obj.kind == "iterator":
                if obj.name == "sequence" and isinstance(state, int):
                    obj.attrs["offset"] = max(0, state)
                return
            dictstate, slots = state, None
            if isinstance(state, tuple) and len(state) == 2:
                dictstate, slots = state
            if isinstance(dictstate, dict): obj.attrs.update(dictstate)
            if isinstance(slots, dict): obj.attrs.update(slots)
            return
        # Builtin primitive instances do not have an instance dictionary.
        raise LoadError("BUILD on object without state")

    def call(self, target, args=(), kwargs=None):
        self.budget.tick()
        if self.call_depth > 100:
            raise AnalysisError("call indirection exceeds analysis limit")
        if not isinstance(args, tuple):
            raise LoadError("REDUCE arguments must be a tuple")
        kwargs = kwargs or {}
        self.call_depth += 1
        try:
            return self._call(target, args, kwargs)
        except (ValueError, TypeError, IndexError, KeyError, AttributeError, OverflowError, ZeroDivisionError, UnicodeError) as exc:
            raise LoadError(str(exc)) from None
        finally:
            self.call_depth -= 1

    def _call(self, target, args, kw):
        if not isinstance(target, Symbol):
            raise LoadError("non-callable target")
        if target.kind == "unknown":
            raise AnalysisError("unresolved callable")
        if target.kind == "partial":
            fn, bound, keywords = target.args
            return self.call(fn, bound + args, dict(keywords, **kw))
        if target.kind == "wrapper":
            return self.call(target.args[0], args, kw)
        if target.kind == "bound_function":
            return self.call(target.args[0], (target.args[1],) + args, kw)
        if target.kind == "decorator":
            return Symbol("wrapper", target.name, (args[0],))
        if target.kind == "function":
            self.finding("code-execution", "serialized Python function invoked")
            return UNKNOWN
        if target.kind == "remote":
            self.finding("network", "XML-RPC method invoked")
            return UNKNOWN
        if target.kind == "getter":
            values = []
            for key in target.args:
                obj = args[0]
                if target.name == "attrgetter":
                    for part in key.split("."): obj = self.attr(obj, part)
                else: obj = self.item(obj, key)
                values.append(obj)
            return values[0] if len(values) == 1 else tuple(values)
        if target.kind == "methodcaller":
            method, bound, keywords = target.args
            return self.call(self.attr(args[0], method), bound, keywords)
        if target.kind == "class":
            obj = self.newobj(target, args, kw)
            init = target.attrs.get("__init__")
            if init: self.call(init, (obj,) + args, kw)
            return obj
        if target.kind == "instance":
            cls = target.attrs.get("__class_symbol__")
            if isinstance(cls, Symbol) and "__call__" in cls.attrs:
                return self.call(cls.attrs["__call__"], (target,) + args, kw)
            raise LoadError("instance is not callable")
        if target.kind == "method":
            return self.method(target.args[0], target.name, args, kw)
        if target.kind != "ref":
            raise LoadError("non-callable symbolic value")
        name = CALL_ALIASES.get(target.name, target.name)
        # Aliased C implementation modules have the same public objects.
        if name.startswith("pathlib.") and name.split(".")[1] in ("PosixPath", "WindowsPath") and name.count(".") > 1:
            name = "pathlib.Path." + name.split(".", 2)[2]
        if name.startswith("builtins."):
            special = self.builtin(name[9:], args, kw)
            if special is not MISSING: return special
        if name == "types.MethodType":
            return Symbol("bound_function", name, (args[0], args[1]))
        if name == "functools.partial.__setstate__":
            self.build(args[0], args[1])
            return None
        if name.startswith(("threading.Thread.", "threading.Timer.", "concurrent.futures.ThreadPoolExecutor.", "concurrent.futures.ProcessPoolExecutor.", "multiprocessing.Process.", "timeit.Timer.")) and args:
            return self.method(args[0], name, args[1:], kw)
        if name in ("importlib.import_module", "importlib.__import__"):
            module = args[0]
            if not isinstance(module, str): raise LoadError("invalid module name")
            if module.startswith(".") and len(args) > 1:
                module = args[1] + module
            return self.module(module)
        if name == "pkgutil.resolve_name": return self.resolve(args[0])
        if name in ("functools.partial", "functools.partialmethod"):
            if not args: raise LoadError("partial needs a callable")
            return Symbol("partial", name, (args[0], args[1:], kw))
        if name in ("operator.attrgetter", "operator.itemgetter"):
            return Symbol("getter", name.split(".")[1], args)
        if name == "operator.methodcaller":
            return Symbol("methodcaller", name, (args[0], args[1:], kw))
        if name in ("operator.call", "apply", "builtins.apply"):
            return self.call(args[0], args[1:], kw)
        if name in ("functools.lru_cache", "functools.cache"):
            if args and isinstance(args[0], Symbol): return Symbol("wrapper", name, (args[0],))
            return Symbol("decorator", name, args)
        if name in ("functools._lru_cache_wrapper", "functools.cached_property", "staticmethod", "classmethod", "builtins.staticmethod", "builtins.classmethod"):
            return Symbol("wrapper", name, (args[0],))
        if name in ("functools.update_wrapper", "functools.wraps"):
            if name.endswith("wraps"): return Symbol("decorator", name, args)
            if isinstance(args[0], Symbol): args[0].attrs["__wrapped__"] = args[1]
            return args[0]
        if name in ("copy.copy", "copy.deepcopy"):
            obj = args[0]
            if isinstance(obj, Symbol): return Symbol(obj.kind, obj.name, obj.args, obj.attrs.copy())
            return obj.copy() if type(obj) in (list, dict, set, bytearray) else obj
        if name == "functools.reduce":
            it = self.iterate(args[1])
            result = args[2] if len(args) > 2 else next(it)
            for item in it: result = self.call(args[0], (result, item))
            return result
        if name in ("copyreg._reconstructor",):
            cls = args[0]
            return Symbol("instance", cls.name if isinstance(cls, Symbol) else "unknown", ())
        if name in ("copyreg.__newobj__", "copyreg.__newobj_ex__"):
            return self.newobj(args[0], args[1] if name.endswith("ex__") else args[1:], args[2] if name.endswith("ex__") else None)
        if name.startswith("operator."):
            return self.operator(name.split(".", 1)[1], args)
        if name.startswith("itertools."):
            return Symbol("iterator", name.split(".", 1)[1], args, dict(kw))
        if name in ("io.BytesIO", "io.StringIO"):
            return Symbol("buffer", name, (), {"data": args[0] if args else (b"" if name.endswith("BytesIO") else ""), "pos": 0})
        if name in ("pickle.Unpickler", "pickle._Unpickler"):
            return Symbol("instance", "pickle.Unpickler", args)
        if name in NESTED:
            return self.nested(name, args, kw)
        if name in ("marshal.loads",):
            # Do not materialise executable code objects, even temporarily.
            return Symbol("bytecode", "marshal", args)
        if name in ("types.CodeType",):
            return Symbol("bytecode", name, args)
        if name in ("types.FunctionType", "types.LambdaType"):
            self.finding("code-execution", name + " materialises bytecode")
            return Symbol("function", name, args)
        if name in ("types.ModuleType",):
            return Symbol("module", args[0] if args else "")
        if name in ("logging.config.dictConfig", "logging.config.fileConfig"):
            return self.logging_config(name, args, kw)
        if name in ("timeit.Timer",):
            # Timer compiles source in its constructor. Callable statements are
            # kept until timeit/repeat is actually called.
            if any(isinstance(v, str) for v in args[:2]) or any(isinstance(kw.get(k), str) for k in ("stmt", "setup")):
                self.finding("code-execution", name)
            return Symbol("instance", name, args, dict(kw))
        if name in ("timeit.timeit", "timeit.repeat"):
            timer = self.call(ref("timeit.Timer"), args[:2], {k: v for k, v in kw.items() if k in ("stmt", "setup", "timer", "globals")})
            return self.method(timer, "timeit.Timer.timeit", (), {})
        if name in ("bdb.Bdb.runcall", "pdb.runcall"):
            return self.call(args[0], args[1:], kw)
        if name in CODE_CALLS:
            if not args and not kw: raise LoadError("missing source argument")
            self.finding("code-execution", name)
            return UNKNOWN
        if name in PROCESS_CALLS or name.startswith(("os.exec", "os.spawn")):
            if not args and name not in ("os.fork", "os.forkpty") and not kw:
                raise LoadError("missing process argument")
            self.finding("process", name)
            return Symbol("instance", name, args, dict(kw))
        if name in FILE_CALLS:
            if args and isinstance(args[0], Symbol) and args[0].kind == "buffer":
                # marshal/json on an in-memory stream have no file capability.
                if name in ("json.load", "marshal.load"):
                    return self.call(ref(name.replace(".load", ".loads")), (self.buffer_read(args[0]),), kw)
            if not args and not kw and not name.startswith("tempfile."):
                raise LoadError("missing file argument")
            self.finding("file-access", name)
            return Symbol("instance", name, args, dict(kw))
        if name in NETWORK_CALLS:
            if not args and not kw: raise LoadError("missing network argument")
            self.finding("network", name)
            return UNKNOWN
        if name in ("ftplib.FTP", "ftplib.FTP_TLS", "smtplib.SMTP", "smtplib.SMTP_SSL", "poplib.POP3", "poplib.POP3_SSL", "imaplib.IMAP4", "imaplib.IMAP4_SSL", "telnetlib.Telnet"):
            host = args[0] if args else kw.get("host", "")
            if host: self.finding("network", name + " with host")
            return Symbol("instance", name, args, dict(kw))
        if name in ("threading.Thread", "threading.Timer"):
            if name.endswith("Timer"):
                attrs = {"target": args[1] if len(args) > 1 else kw.get("function"), "args": args[2] if len(args) > 2 else kw.get("args", ()), "kwargs": args[3] if len(args) > 3 else kw.get("kwargs", {})}
            else:
                attrs = {"target": args[1] if len(args) > 1 else kw.get("target"), "args": args[3] if len(args) > 3 else kw.get("args", ()), "kwargs": args[4] if len(args) > 4 else kw.get("kwargs", {})}
            return Symbol("instance", name, args, attrs)
        if name in ("pathlib.Path", "pathlib.PosixPath", "pathlib.WindowsPath", "pathlib.PurePath", "pathlib.PurePosixPath", "pathlib.PureWindowsPath"):
            return Symbol("instance", name, args)
        if name in ("_codecs.encode", "codecs.encode", "_codecs.decode", "codecs.decode"):
            codec = args[1] if len(args) > 1 else "utf-8"
            if codec.lower().replace("_", "-") not in {"utf-8", "utf8", "utf-16", "utf-16-le", "utf-16-be", "utf-32", "ascii", "latin1", "latin-1", "iso8859-1", "unicode-escape", "raw-unicode-escape", "base64", "base64-codec", "hex", "hex-codec", "rot-13", "rot13"}:
                raise AnalysisError("unmodelled codec")
            return codecs.encode(*args) if name.endswith("encode") else codecs.decode(*args)
        if name.startswith("base64.") or name in ("binascii.a2b_base64", "binascii.unhexlify", "binascii.a2b_hex"):
            fn = name.split(".")[1]
            allowed = {"b64decode", "standard_b64decode", "urlsafe_b64decode", "b32decode", "b16decode", "b85decode", "a85decode", "b64encode", "standard_b64encode", "urlsafe_b64encode", "b32encode", "b16encode", "b85encode", "a85encode", "decodebytes", "encodebytes"}
            if fn in allowed and primitive(args): return getattr(base64, fn)(*args, **kw)
            if fn in ("a2b_base64", "unhexlify", "a2b_hex") and primitive(args): return getattr(binascii, fn)(*args, **kw)
        if name in ("zlib.decompress", "gzip.decompress", "bz2.decompress", "lzma.decompress"):
            if not isinstance(args[0], (bytes, bytearray)): raise LoadError("decompress requires bytes")
            if name == "zlib.decompress" and len(args) > 1 and args[1] not in (15,):
                import zlib
                dec = zlib.decompressobj(args[1])
                result = dec.decompress(args[0], MAX_BYTES + 1)
                if len(result) > MAX_BYTES or not dec.eof: raise LimitError("decompression limit")
                return result
            return decompress(args[0], {"zlib": "zlib", "gzip": "gzip", "bz2": "bz2", "lzma": "xz"}[name.split(".")[0]])
        if name in ("json.loads", "ast.literal_eval"):
            import ast
            import json
            result = ast.literal_eval(args[0]) if name == "ast.literal_eval" else json.loads(args[0])
            if name == "json.loads" and kw.get("object_hook"):
                result = self.json_hook(result, kw["object_hook"])
            return result
        if name == "os.fspath":
            v = args[0]
            return "/".join(v.args) if isinstance(v, Symbol) and v.name.startswith("pathlib.") else v
        if name == "numpy.dtype": return Symbol("instance", name, args)
        if name in ("numpy.core.multiarray._reconstruct", "numpy._core.multiarray._reconstruct"):
            return Symbol("instance", "numpy.ndarray", args)
        if name in ("numpy.core.numeric._frombuffer", "numpy._core.numeric._frombuffer"):
            return Symbol("instance", "numpy.ndarray", (), {"data": args[0], "dtype": args[1], "shape": args[2]})
        if name in ("collections.OrderedDict", "collections.Counter", "collections.UserDict"):
            return MappingValue(name, args[0] if args and isinstance(args[0], dict) else self.iterate(args[0]) if args else ())
        if name == "collections.defaultdict":
            return dict(args[1]) if len(args) > 1 else {}
        if name == "collections.deque": return list(self.iterate(args[0])) if args else []
        if name.startswith("math.") and name.split(".")[1] in ("sqrt", "exp", "log", "log1p", "sin", "cos", "floor", "ceil", "fabs") and primitive(args):
            return getattr(math, name.split(".")[1])(*args)
        # Standard model rebuilders and ordinary classes are represented, not
        # instantiated. Their arguments have already been evaluated by the VM.
        return Symbol("instance", name, args, dict(kw))

    def builtin(self, name, args, kw):
        if name == "getattr": return self.attr(args[0], args[1], args[2] if len(args) > 2 else MISSING)
        if name in ("setattr", "delattr"):
            obj, key = args[:2]
            if isinstance(obj, Symbol):
                if name == "setattr": obj.attrs[key] = args[2]
                else: obj.attrs.pop(key, None)
            else: raise LoadError("cannot set primitive attribute")
            return None
        if name == "vars":
            if args and isinstance(args[0], Symbol): return Symbol("namespace", args=(args[0],))
            raise LoadError("vars requires an object with a namespace")
        if name in ("globals", "locals"):
            return Symbol("namespace", args=(self.module("builtins"),))
        if name == "__import__":
            module = args[0]
            if not isinstance(module, str): raise LoadError("invalid module name")
            fromlist = args[3] if len(args) > 3 else kw.get("fromlist", ())
            return self.module(module if fromlist else module.split(".")[0])
        if name in ("object.__getattribute__", "type.__getattribute__"):
            return self.attr(args[0], args[1])
        if name in ("object.__setattr__", "type.__setattr__"):
            return self.builtin("setattr", args, kw)
        if name == "object.__new__": return self.newobj(args[0], ())
        if name == "type":
            if len(args) == 1: return self.attr(args[0], "__class__")
            if len(args) == 3 and isinstance(args[2], dict):
                return Symbol("class", args[0], args[1], args[2].copy())
            raise LoadError("invalid type arguments")
        if name == "__build_class__":
            self.call(args[0], ())
            return Symbol("class", str(args[1]))
        if name in ("map", "filter", "zip", "enumerate"):
            return Symbol("iterator", name, args, dict(kw))
        if name == "iter":
            if len(args) == 2: return Symbol("iterator", "callable", args)
            return args[0] if isinstance(args[0], Symbol) and args[0].kind == "iterator" else Symbol("iterator", "sequence", args)
        if name == "next":
            try: return next(self.iterate(args[0]))
            except StopIteration:
                if len(args) > 1: return args[1]
                raise LoadError("exhausted iterator") from None
        if name in ("list", "tuple", "set", "frozenset"):
            seq = list(self.iterate(args[0])) if args else []
            return {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[name](seq)
        if name == "dict":
            result = {}
            if args:
                if isinstance(args[0], dict): result.update(args[0])
                else: result.update(self.iterate(args[0]))
            result.update(kw)
            return result
        if name in ("bytes", "bytearray"):
            cls = bytes if name == "bytes" else bytearray
            if args and isinstance(args[0], int) and args[0] > MAX_BYTES: raise LimitError("byte allocation limit")
            if args and isinstance(args[0], Symbol): return cls(self.iterate(args[0]))
            return cls(*args, **kw)
        if name in ("bytes.fromhex", "bytearray.fromhex"):
            return (bytes if name.startswith("bytes") else bytearray).fromhex(args[0])
        if name in ("str", "int", "float", "complex", "bool"):
            if primitive(args): return {"str": str, "int": int, "float": float, "complex": complex, "bool": bool}[name](*args, **kw)
            if name == "str" and args and isinstance(args[0], Symbol) and args[0].name.startswith("pathlib."):
                return "/".join(args[0].args)
            return UNKNOWN
        if name == "range": return range(*args)
        if name == "slice": return slice(*args)
        if name == "len":
            if isinstance(args[0], Symbol): return UNKNOWN
            return len(args[0])
        if name in ("sorted", "min", "max"):
            values = list(self.iterate(args[0])) if name == "sorted" or len(args) == 1 else list(args)
            key = kw.get("key")
            keys = [self.call(key, (v,)) for v in values] if key is not None else values
            if not values:
                if name == "sorted": return []
                if "default" in kw: return kw["default"]
                raise LoadError("empty sequence")
            if any(isinstance(v, Symbol) for v in keys):
                return values if name == "sorted" else UNKNOWN
            indices = sorted(range(len(values)), key=lambda i: keys[i], reverse=kw.get("reverse", False))
            if name == "sorted": return [values[i] for i in indices]
            return values[indices[0 if name == "min" else -1]]
        if name in ("sum", "any", "all"):
            value = args[1] if len(args) > 1 else 0
            for item in self.iterate(args[0]):
                if name == "any" and item: return True
                if name == "all" and not item: return False
                if name == "sum": value = self.operator("add", (value, item))
            return value if name == "sum" else name == "all"
        if name == "reversed":
            return Symbol("iterator", "sequence", (list(reversed(list(self.iterate(args[0])))),))
        if name in ("abs", "round", "ord", "chr", "divmod", "hex", "oct", "bin", "repr", "ascii"):
            if primitive(args):
                fn = {"abs": abs, "round": round, "ord": ord, "chr": chr, "divmod": divmod, "hex": hex, "oct": oct, "bin": bin, "repr": repr, "ascii": ascii}[name]
                return fn(*args)
            return UNKNOWN
        if name == "pow":
            if len(args) < 3 and isinstance(args[1], int) and abs(args[1]) > 10000: raise LimitError("power size limit")
            return pow(*args) if primitive(args) else UNKNOWN
        if name in ("hasattr", "isinstance", "issubclass", "callable"):
            return UNKNOWN
        if name in ("staticmethod", "classmethod", "property"):
            return Symbol("wrapper", name, (args[0],))
        # Unbound primitive methods, e.g. dict.get(sys.modules, 'os').
        if "." in name:
            cls, meth = name.split(".", 1)
            if cls in ("str", "bytes", "bytearray", "dict", "list", "tuple", "set", "frozenset") and args:
                return self.method(args[0], "builtins." + name, args[1:], kw)
        return MISSING

    def operator(self, name, args):
        name = name.strip("_")
        if name == "getitem": return self.item(args[0], args[1])
        if name == "setitem":
            self.setitem(*args)
            return None
        if name == "delitem":
            if isinstance(args[0], dict): args[0].pop(args[1], None)
            return None
        if name == "call": return self.call(args[0], args[1:])
        if name in ("add", "concat", "iadd", "iconcat"):
            a, b = args
            if isinstance(a, Symbol) or isinstance(b, Symbol): return UNKNOWN
            if isinstance(a, (str, bytes, bytearray, list, tuple)) and len(a) + len(b) > MAX_BYTES: raise LimitError("concatenation limit")
            return a + b
        if name in ("mul", "imul"):
            a, b = args
            if isinstance(a, Symbol) or isinstance(b, Symbol): return UNKNOWN
            if isinstance(a, int): a, b = b, a
            if isinstance(a, (str, bytes, bytearray, list, tuple)) and isinstance(b, int) and len(a) * b > MAX_BYTES: raise LimitError("repetition limit")
            return a * b
        if name in ("length_hint",):
            return len(args[0]) if not isinstance(args[0], Symbol) else 0
        if name in ("contains", "countOf", "indexOf"):
            for i, v in enumerate(self.iterate(args[0])):
                if v == args[1] and name == "contains": return True
                if v == args[1] and name == "indexOf": return i
            return False if name == "contains" else 0
        allowed = {"sub", "truediv", "floordiv", "mod", "neg", "pos", "not", "truth", "eq", "ne", "lt", "le", "gt", "ge", "and", "or", "xor", "invert", "index", "is", "is_not"}
        if name in allowed and primitive(args):
            fn = getattr(operator, name + "_", None) or getattr(operator, name, None)
            if fn: return fn(*args)
        return UNKNOWN

    def method(self, obj, qualified, args, kw):
        name = qualified.rsplit(".", 1)[-1]
        if name in ("__getattribute__", "__getattr__"):
            return self.attr(obj, args[0])
        if name == "__setattr__":
            return self.builtin("setattr", (obj,) + args, kw)
        if name in ("__getitem__", "get"):
            return self.item(obj, args[0], args[1] if len(args) > 1 else None if name == "get" else MISSING)
        if name == "__setitem__":
            self.setitem(obj, args[0], args[1])
            return None
        if name == "__call__": return self.call(obj, args, kw)
        if name == "__setstate__":
            self.build(obj, args[0])
            return None
        if name in ("__add__", "__mul__", "__radd__", "__rmul__"):
            values = (args[0], obj) if name.startswith("__r") else (obj, args[0])
            return self.operator("add" if "add" in name else "mul", values)
        if name == "__iter__": return self.builtin("iter", (obj,), {})
        if name == "__next__": return self.builtin("next", (obj,), {})
        if isinstance(obj, Symbol):
            if obj.kind == "buffer":
                if name in ("read", "readline", "readlines"):
                    return self.buffer_read(obj, args[0] if args else -1)
                if name in ("getvalue", "getbuffer"): return obj.attrs.get("data", b"")
                if name == "seek":
                    whence = args[1] if len(args) > 1 else 0
                    base = 0 if whence == 0 else obj.attrs.get("pos", 0) if whence == 1 else len(obj.attrs["data"])
                    obj.attrs["pos"] = max(0, base + args[0])
                    return obj.attrs["pos"]
                if name == "tell": return obj.attrs.get("pos", 0)
                if name == "write":
                    data, pos = obj.attrs.get("data", b""), obj.attrs.get("pos", 0)
                    obj.attrs["data"] = data[:pos] + args[0] + data[pos + len(args[0]):]
                    obj.attrs["pos"] = pos + len(args[0])
                    return len(args[0])
                return None
            if obj.kind == "instance":
                cls = obj.name
                if cls == "pickle.Unpickler" and name == "load": return self.nested("pickle.load", obj.args[:1], kw)
                if cls in ("threading.Thread", "threading.Timer") and name in ("start", "run"):
                    fn = obj.attrs.get("target", obj.attrs.get("_target"))
                    if fn is not None:
                        return self.call(fn, tuple(obj.attrs.get("args", obj.attrs.get("_args", ())) or ()), obj.attrs.get("kwargs", obj.attrs.get("_kwargs", {})) or {})
                    return None
                if cls.startswith("multiprocessing.") and name == "start":
                    self.finding("process", qualified)
                    return UNKNOWN
                if "Executor" in cls or cls in ("multiprocessing.pool.ThreadPool",):
                    if name in ("submit", "map", "apply", "apply_async", "starmap", "map_async"):
                        if "Process" in cls: self.finding("process", qualified)
                        if name == "submit": result = self.call(args[0], args[1:], kw)
                        elif name in ("apply", "apply_async"):
                            result = self.call(args[0], tuple(args[1] if len(args) > 1 else ()), args[2] if len(args) > 2 else {})
                        else:
                            it = Symbol("iterator", "starmap" if name == "starmap" else "map", args)
                            result = list(self.iterate(it))
                        return Symbol("instance", "concurrent.futures.Future", (), {"result": result})
                if cls == "concurrent.futures.Future" and name == "result": return obj.attrs.get("result", UNKNOWN)
                if cls in ("bdb.Bdb", "pdb.Pdb") and name == "runcall": return self.call(args[0], args[1:], kw)
                if cls == "timeit.Timer" and name in ("timeit", "repeat", "autorange"):
                    for v in obj.args[:2]:
                        if isinstance(v, Symbol): self.call(v, ())
                    for k in ("stmt", "setup"):
                        v = obj.attrs.get(k)
                        if isinstance(v, Symbol): self.call(v, ())
                        elif isinstance(v, str) and v not in ("", "pass"): self.finding("code-execution", qualified)
                    return UNKNOWN
                if cls.startswith("pathlib."):
                    if name in ("joinpath", "with_name", "with_suffix"):
                        return Symbol("instance", cls, obj.args + args)
                    if name in ("__fspath__", "__str__"): return "/".join(obj.args)
                if qualified in CODE_CALLS:
                    self.finding("code-execution", qualified)
                    return UNKNOWN
                if qualified in FILE_CALLS:
                    self.finding("file-access", qualified)
                    return UNKNOWN
                if qualified in NETWORK_CALLS or cls.startswith(("http.client.", "ftplib.", "smtplib.", "poplib.", "imaplib.")) and name in ("request", "connect", "send", "login", "sendmail", "retrbinary", "retrlines", "storbinary", "open"):
                    self.finding("network", qualified)
                    return UNKNOWN
                if cls.startswith("numpy") and name in ("tolist", "item", "tobytes", "tostring"):
                    return obj.attrs.get("data", UNKNOWN)
            return self.call(ref(qualified), (obj,) + args, kw)
        if isinstance(obj, dict):
            if name == "keys": return list(obj.keys())
            if name == "values": return list(obj.values())
            if name == "items": return list(obj.items())
            if name == "update":
                if args: obj.update(args[0] if isinstance(args[0], dict) else self.iterate(args[0]))
                obj.update(kw)
                return None
            if name == "setdefault": return obj.setdefault(*args)
            if name == "pop": return obj.pop(*args)
            if name == "copy": return obj.copy()
            if name == "clear": obj.clear(); return None
        if type(obj) is list:
            if name == "append": obj.append(args[0]); return None
            if name == "extend": obj.extend(self.iterate(args[0])); return None
            if name == "pop": return obj.pop(*args)
            if name == "insert": obj.insert(*args); return None
            if name == "copy": return obj.copy()
            if name == "reverse": obj.reverse(); return None
            if name == "sort": obj[:] = self.builtin("sorted", (obj,), kw); return None
        if type(obj) in (str, bytes, bytearray):
            if name == "join":
                return obj.join(self.iterate(args[0]))
            if name in {"encode", "decode"}:
                codec = args[0] if args else "utf-8"
                if codec.lower().replace("_", "-") not in {"utf-8", "utf8", "latin-1", "latin1", "ascii", "utf-16", "utf-16-le", "utf-16-be", "unicode-escape", "raw-unicode-escape"}:
                    raise AnalysisError("unmodelled text codec")
            allowed = {"encode", "decode", "fromhex", "hex", "lower", "upper", "casefold", "strip", "lstrip", "rstrip", "replace", "split", "rsplit", "splitlines", "partition", "rpartition", "format", "format_map", "removeprefix", "removesuffix", "startswith", "endswith", "find", "index", "count", "capitalize", "swapcase", "translate", "maketrans", "__mod__"}
            if name in allowed and primitive(args) and primitive(kw):
                # format widths can request enormous allocations.
                if name in ("format", "format_map", "__mod__") and any(int(x) > MAX_BYTES for x in re.findall(r"\d{6,}", obj if isinstance(obj, str) else "")):
                    raise LimitError("format width limit")
                return getattr(obj, name)(*args, **kw)
        return UNKNOWN

    def iterate(self, obj):
        if isinstance(obj, Symbol):
            if obj.kind == "iterator":
                if "iterator" not in obj.attrs: obj.attrs["iterator"] = self._iterator(obj)
                return obj.attrs["iterator"]
            if obj.kind == "instance" and obj.name.startswith("numpy"):
                return self.iterate(obj.attrs.get("data", ()))
            raise AnalysisError("unresolved iterable")
        if not isinstance(obj, (str, bytes, bytearray, list, tuple, dict, set, frozenset, range)):
            raise LoadError("not iterable")
        def bounded():
            for v in obj:
                self.budget.tick()
                yield v
        return bounded()

    def _iterator(self, obj):
        name, args = obj.name, obj.args
        if name == "sequence":
            it = self.iterate(args[0])
            for _ in range(obj.attrs.get("offset", 0)): next(it, None)
            yield from it
        elif name in ("map", "zip", "zip_longest"):
            sources = args[1:] if name == "map" else args
            its = [self.iterate(s) for s in sources]
            if not its: return
            while True:
                self.budget.tick()
                values = [next(it, MISSING) for it in its]
                if name == "zip_longest":
                    if all(v is MISSING for v in values): return
                    values = [obj.attrs.get("fillvalue") if v is MISSING else v for v in values]
                elif any(v is MISSING for v in values): return
                yield self.call(args[0], tuple(values)) if name == "map" else tuple(values)
        elif name in ("filter", "filterfalse", "takewhile", "dropwhile"):
            dropping = name == "dropwhile"
            for v in self.iterate(args[1]):
                pred = self.call(args[0], (v,)) if args[0] is not None else v
                if name == "takewhile" and not pred: return
                if name == "dropwhile":
                    if pred and dropping: continue
                    dropping = False
                    yield v
                elif bool(pred) == (name != "filterfalse"): yield v
        elif name == "starmap":
            for v in self.iterate(args[1]): yield self.call(args[0], tuple(self.iterate(v)))
        elif name == "callable":
            while True:
                value = self.call(args[0], ())
                if value == args[1]: return
                yield value
        elif name in ("chain", "chain.from_iterable"):
            sources = args if name == "chain" else self.iterate(args[0])
            for source in sources: yield from self.iterate(source)
        elif name == "islice":
            import itertools
            yield from itertools.islice(self.iterate(args[0]), *args[1:])
        elif name == "repeat":
            n = args[1] if len(args) > 1 else MAX_ITEMS
            for _ in range(n):
                self.budget.tick()
                yield args[0]
        elif name == "count":
            v, step = (args[0] if args else 0), (args[1] if len(args) > 1 else 1)
            while True:
                self.budget.tick()
                yield v
                v += step
        elif name == "enumerate":
            yield from enumerate(self.iterate(args[0]), args[1] if len(args) > 1 else 0)
        elif name == "accumulate":
            it = self.iterate(args[0])
            initial = obj.attrs.get("initial", MISSING)
            result = next(it, MISSING) if initial is MISSING else initial
            if result is MISSING: return
            yield result
            for item in it:
                result = self.call(args[1], (result, item)) if len(args) > 1 and args[1] else self.operator("add", (result, item))
                yield result
        elif name == "compress":
            for value, flag in zip(self.iterate(args[0]), self.iterate(args[1])):
                if flag: yield value
        elif name == "cycle":
            saved = list(self.iterate(args[0]))
            if not saved: return
            while True: yield from self.iterate(saved)
        else:
            raise AnalysisError("unmodelled iterator " + name)

    def buffer_read(self, obj, n=-1):
        data, pos = obj.attrs.get("data", b""), obj.attrs.get("pos", 0)
        end = len(data) if n < 0 else min(len(data), pos + n)
        obj.attrs["pos"] = end
        return data[pos:end]

    def nested(self, name, args, kw):
        from modelgate import dispatch, pickle_scan
        if not args: raise LoadError("loader needs data")
        source = args[0]
        if name == "pickle.Unpickler.load" and isinstance(source, Symbol): source = source.args[0]
        is_loads = name in ("pickle.loads", "torch.storage._load_from_bytes")
        if isinstance(source, Symbol) and source.kind == "instance" and source.name.startswith("pathlib."):
            self.finding("file-access", name + " reads a path")
            return UNKNOWN
        if not is_loads and isinstance(source, (str, bytes, bytearray)):
            self.finding("file-access", name + " reads a path")
            return UNKNOWN
        if isinstance(source, Symbol) and source.kind == "buffer":
            data = source.attrs.get("data", b"")[source.attrs.get("pos", 0):]
        elif is_loads and isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        else:
            raise AnalysisError("nested loader input is not statically known")
        if not isinstance(data, bytes): raise LoadError("loader needs binary data")
        self.load_depth += 1
        if self.load_depth > 20: raise AnalysisError("nested loader depth limit")
        previous = self.loc, self.op
        loc = self.loc + "/" + name
        try:
            if name.startswith("pickle.") or "read_pickle" in name:
                value, end, ok = pickle_scan.analyze_pickle(data, loc, self, streams=1, persistent=False)
                if isinstance(source, Symbol) and source.kind == "buffer": source.attrs["pos"] += end
                if not ok: raise LoadError("nested pickle failed")
                return value
            mode = "torch" if name.startswith("torch.") else "numpy" if name.startswith("numpy.") else "joblib"
            allow_pickle = True
            if mode == "numpy":
                index = 1 if name.endswith("read_array") else 2
                allow_pickle = args[index] if len(args) > index else kw.get("allow_pickle", False)
            return dispatch.analyze_bytes(data, loc, self, loader=mode, allow_pickle=allow_pickle)
        finally:
            self.loc, self.op = previous
            self.load_depth -= 1

    def json_hook(self, value, hook):
        if isinstance(value, list): return [self.json_hook(v, hook) for v in value]
        if isinstance(value, dict): return self.call(hook, ({k: self.json_hook(v, hook) for k, v in value.items()},))
        return value

    def logging_config(self, name, args, kw):
        if name.endswith("fileConfig"):
            if args and isinstance(args[0], (str, bytes)):
                self.finding("file-access", name)
            else:
                self.finding("code-execution", name + " evaluates configuration")
            return None
        cfg = args[0]
        if not isinstance(cfg, dict): raise LoadError("dictConfig requires a dict")
        if cfg.get("version") != 1: raise LoadError("unsupported logging configuration version")
        def resolve_ext(v):
            return self.resolve(v[6:]) if isinstance(v, str) and v.startswith("ext://") else v
        for section in ("formatters", "filters", "handlers"):
            for item in cfg.get(section, {}).values():
                if not isinstance(item, dict): continue
                factory = item.get("()")
                if factory is None and section == "handlers": factory = item.get("class")
                if factory is not None:
                    factory = self.resolve(factory) if isinstance(factory, str) else factory
                    kwargs = {k: resolve_ext(v) for k, v in item.items() if k not in ("()", "class", ".", "level", "formatter", "filters")}
                    self.call(factory, (), kwargs)
        return None
