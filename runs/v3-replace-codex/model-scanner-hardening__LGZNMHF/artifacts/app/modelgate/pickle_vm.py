"""Symbolic standard-library semantics used by the pickle interpreter.

This module never resolves a pickle global to a real Python object. The
small set of concrete operations below act only on scanner-owned literals.
"""
from __future__ import annotations

import base64
import binascii
import copy
import io
import math
import os
import posixpath
import zlib

from modelgate.capabilities import (CAPABILITIES, CLIENTS, CONNECT_ON_HOST,
                                    MODULE_ATTRS, MODULE_NAMES, canonical)
from modelgate.symbols import (Bound, Buffer, Code, Getter, Global, Instance,
                               Lazy, MISSING, Module, Namespace, Partial,
                               Unknown, Wrapper)

PRIMITIVES = (str, bytes, bytearray, int, float, complex, bool, type(None))
DATA_MODULES = ("numpy.", "scipy.", "sklearn.", "torch.", "pandas.", "collections.",
                "datetime.", "decimal.", "fractions.", "re.", "_sre.", "uuid.",
                "zoneinfo.", "pytz.", "array.", "_array.", "builtins.")
PATH_CLASSES = {"pathlib.Path", "pathlib.PosixPath", "pathlib.WindowsPath",
                "pathlib.PurePath", "pathlib.PurePosixPath", "pathlib.PureWindowsPath"}
LOADERS = {"pickle.loads", "pickle.load", "pickle.Unpickler.load", "torch.load",
           "torch.storage._load_from_bytes", "numpy.load", "numpy.lib.format.read_array",
           "joblib.load", "pandas.read_pickle", "dill.loads", "dill.load", "cloudpickle.loads", "cloudpickle.load"}
BUILTIN_TYPES = {str: "str", bytes: "bytes", bytearray: "bytearray", int: "int", bool: "bool", float: "float",
                 complex: "complex", tuple: "tuple", list: "list", dict: "dict", set: "set", frozenset: "frozenset"}


class Calls:
    def attr(self, obj, name, default=MISSING):
        if not isinstance(name, str):
            raise self.LoadError("attribute name is not a string")
        if isinstance(obj, (Module, Global, Instance, Partial, Wrapper)) and name in obj.attrs:
            return obj.attrs[name]
        if isinstance(obj, Module):
            if name == "__dict__":
                return Namespace(obj)
            if name == "__name__":
                return obj.name
            if name == "__builtins__":
                return Namespace(self.module("builtins"))
            if obj.name == "sys" and name == "modules":
                return Instance("sys.modules")
            alias = MODULE_ATTRS.get(obj.name, {}).get(name)
            if alias:
                return self.module(alias)
            path = obj.name + "." + name
            if path in MODULE_NAMES or path in self.modules:
                return self.module(path)
            if obj.name == "builtins" and name == "None":
                return None
            if name == "__setstate__":
                return default
            return self.ref(path)
        if isinstance(obj, Global):
            if name == "__call__":
                return obj
            if name == "__dict__":
                return Namespace(obj)
            if name in ("__globals__", "func_globals"):
                return Namespace(self.module(self.owner_module(obj.name)))
            if name == "__builtins__":
                return Namespace(self.module("builtins"))
            if name in ("__name__", "__qualname__"):
                return obj.name.rsplit(".", 1)[-1]
            if name == "__module__":
                return self.owner_module(obj.name)
            if name == "__class__":
                return self.ref("builtins.type" if self.is_class(obj) else "types.FunctionType")
            if name == "__mro__":
                return (obj, self.ref("builtins.object"))
            if name in ("__setstate__",):
                return default
            return self.ref(obj.name + "." + name)
        if isinstance(obj, Partial):
            if name == "func":
                return obj.func
            if name == "args":
                return obj.args
            if name == "keywords":
                return obj.kwargs
            if name == "__dict__":
                return Namespace(obj)
            if name == "__setstate__":
                return Bound(obj, name)
            if name == "__call__":
                return obj
            if name == "__class__":
                return self.ref("functools.partial")
            return default if default is not MISSING else Unknown("partial attribute")
        if isinstance(obj, Wrapper):
            if name in ("__wrapped__", "__func__", "__call__"):
                return obj.func
            if name == "__dict__":
                return Namespace(obj)
            if name == "__setstate__":
                return default
            return self.attr(obj.func, name, default)
        if isinstance(obj, Getter) and name == "__call__":
            return obj
        if isinstance(obj, Instance):
            if name == "__dict__":
                return Namespace(obj)
            if name == "__class__":
                return self.ref(obj.cls)
            if obj.cls in PATH_CLASSES and name in ("name", "suffix", "stem", "parent"):
                path = str(obj.value if obj.value is not None else posixpath.join(*obj.args))
                if name == "name":
                    return posixpath.basename(path)
                if name == "parent":
                    return Instance(obj.cls, (posixpath.dirname(path),), value=posixpath.dirname(path))
                return posixpath.splitext(posixpath.basename(path))[1 if name == "suffix" else 0]
            if name == "__setstate__":
                return default
            return Bound(obj, name)
        if isinstance(obj, Buffer):
            if name == "__setstate__":
                return Bound(obj, name)
            return Bound(obj, name)
        if name == "__setstate__":
            return default
        if name == "__class__":
            cls = BUILTIN_TYPES.get(type(obj))
            return self.ref("builtins." + cls) if cls else Unknown("type")
        if isinstance(obj, (str, bytes, bytearray, dict, list, tuple, set, frozenset, Namespace, Lazy, Code)):
            return Bound(obj, name)
        if isinstance(obj, Unknown):
            return default if default is not MISSING else Unknown(f"{obj.reason}.{name}")
        return default if default is not MISSING else Unknown(f"missing attribute {name}")

    def owner_module(self, name):
        parts = name.split(".")
        for i, part in enumerate(parts):
            if part[:1].isupper():
                return ".".join(parts[:i])
        return name.rsplit(".", 1)[0]

    def getitem(self, obj, key, default=MISSING):
        try:
            if isinstance(obj, Namespace):
                if key in obj.owner.attrs:
                    return obj.owner.attrs[key]
                if isinstance(obj.owner, Module):
                    return self.attr(obj.owner, key)
                if default is not MISSING:
                    return default
                raise self.LoadError("missing dictionary key")
            if isinstance(obj, Instance) and obj.cls == "sys.modules":
                if not isinstance(key, str):
                    raise self.LoadError("module name is not a string")
                return self.module(key)
            if isinstance(obj, Instance) and isinstance(obj.value, (list, tuple, dict, bytes, str)):
                return self.getitem(obj.value, key, default)
            if isinstance(obj, (dict, list, tuple, str, bytes, bytearray, range)):
                return obj[key]
            if isinstance(obj, Unknown):
                return Unknown("item of unknown object")
            return self.call(self.attr(obj, "__getitem__"), (key,))
        except (KeyError, IndexError, TypeError) as exc:
            if default is not MISSING:
                return default
            raise self.LoadError(str(exc)) from None

    def setitem(self, obj, key, value):
        if isinstance(obj, Namespace):
            obj.owner.attrs[key] = value
        elif isinstance(obj, Instance) and obj.cls == "sys.modules":
            self.modules[key] = value
        elif isinstance(obj, (dict, list, bytearray)):
            obj[key] = value
        elif isinstance(obj, Instance) and isinstance(obj.value, dict):
            obj.value[key] = value
        else:
            self.call(self.attr(obj, "__setitem__"), (key, value))

    def append(self, obj, values):
        if isinstance(obj, list):
            obj.extend(values)
        elif isinstance(obj, Instance) and isinstance(obj.value, list):
            obj.value.extend(values)
        else:
            extend = self.attr(obj, "extend", default=MISSING)
            if extend is not MISSING:
                self.call(extend, (values,))
            else:
                for value in values:
                    self.call(self.attr(obj, "append"), (value,))

    def make_partial(self, args, kwargs):
        if not args:
            raise self.LoadError("partial requires a callable")
        fn = args[0]
        if isinstance(fn, Partial) and not fn.attrs:
            return Partial(fn.func, fn.args + args[1:], {**fn.kwargs, **kwargs})
        return Partial(fn, args[1:], dict(kwargs))

    def call(self, fn, args=(), kwargs=None):
        self.check_budget()
        self.call_depth += 1
        try:
            if self.call_depth > 100:
                raise self.AnalysisError("symbolic call recursion limit")
            if not isinstance(args, tuple):
                raise self.LoadError("call arguments are not a tuple")
            return self._call(fn, args, kwargs or {})
        except (ValueError, TypeError, IndexError, KeyError, AttributeError, ZeroDivisionError, OverflowError) as exc:
            # Errors from operations on literals are the errors the loader
            # would encounter before reaching any later executing opcode.
            raise self.LoadError(str(exc)) from None
        finally:
            self.call_depth -= 1

    def _call(self, fn, args, kwargs):
        if isinstance(fn, Partial):
            return self.call(fn.func, fn.args + args, {**fn.kwargs, **kwargs})
        if isinstance(fn, Wrapper):
            return self.call(fn.func, args, kwargs)
        if isinstance(fn, Getter):
            if not args:
                raise self.LoadError("getter requires an object")
            obj = args[0]
            if fn.kind == "methodcaller":
                return self.call(self.attr(obj, fn.args[0]), fn.args[1:], fn.kwargs)
            values = []
            for key in fn.args:
                value = obj
                if fn.kind == "itemgetter":
                    value = self.getitem(obj, key)
                else:
                    for part in key.split("."):
                        value = self.attr(value, part)
                values.append(value)
            return values[0] if len(values) == 1 else tuple(values)
        if isinstance(fn, Bound):
            return self.bound_call(fn.obj, fn.name, args, kwargs)
        if isinstance(fn, Instance):
            if fn.cls == "function":
                self.finding("code-execution", "invoke materialized Python bytecode")
                return Unknown()
            target = fn.attrs.get("__call__")
            if target is not None:
                return self.call(target, args, kwargs)
            if fn.cls == "xmlrpc.client.ServerProxy.method":
                self.finding("network", "invoke XML-RPC method")
                return Unknown()
            raise self.LoadError("object is not callable")
        if isinstance(fn, Unknown):
            raise self.AnalysisError(f"unresolved callable: {fn.reason}")
        if not isinstance(fn, Global):
            raise self.LoadError("value is not callable")
        name = fn.name
        # Explicit mutations of a class's call hooks are visible here too.
        if "__new__" in fn.attrs:
            value = self.call(fn.attrs["__new__"], (fn,) + args, kwargs)
            if "__init__" in fn.attrs:
                self.call(fn.attrs["__init__"], (value,) + args, kwargs)
            return value
        if name in {"builtins.getattr", "builtins.hasattr"}:
            if len(args) < 2:
                raise self.LoadError("getattr requires an object and name")
            result = self.attr(args[0], args[1], args[2] if len(args) > 2 else MISSING)
            return result is not MISSING if name.endswith("hasattr") else result
        if name in {"builtins.setattr", "builtins.delattr"}:
            obj, key = args[:2]
            if not hasattr(obj, "attrs"):
                raise self.LoadError("object has no writable attributes")
            if name.endswith("setattr"):
                obj.attrs[key] = args[2]
            else:
                obj.attrs.pop(key, None)
            return None
        if name in {"builtins.__import__", "importlib.import_module"}:
            if not args or not isinstance(args[0], str):
                raise self.LoadError("import requires a module name")
            module = args[0]
            if name == "builtins.__import__" and not (len(args) > 3 and args[3]) and not kwargs.get("fromlist"):
                module = module.split(".", 1)[0]
            # Importing modules is expressly not an effect in this policy.
            self.module(args[0])
            return self.module(module)
        if name in {"builtins.vars", "builtins.globals", "builtins.locals"}:
            return self.attr(args[0], "__dict__") if args else Namespace(self.module("builtins"))
        if name in {"pkgutil.resolve_name", "pydoc.locate"}:
            return self.resolve_name(args[0])
        if name in {"operator.attrgetter", "operator.itemgetter", "operator.methodcaller"}:
            return Getter(name.split(".")[-1], args, kwargs)
        if name == "operator.call":
            return self.call(args[0], args[1:], kwargs)
        if name == "functools.partial":
            return self.make_partial(args, kwargs)
        if name in {"copy.copy", "copy.deepcopy"}:
            if not args:
                raise self.LoadError("copy requires an object")
            # Immutable callable identities and their capabilities survive a
            # copy; partials get independent mutable state.
            if isinstance(args[0], Partial):
                p = args[0]
                return Partial(p.func, p.args, dict(p.kwargs), dict(p.attrs))
            return args[0]
        if name in {"functools.lru_cache", "functools.cache", "functools._lru_cache_wrapper"}:
            if args and self.callable(args[0]):
                return Wrapper(args[0])
            return self.ref("__scanner__.cache_decorator")
        if name == "__scanner__.cache_decorator":
            return Wrapper(args[0])
        if name == "functools.update_wrapper":
            target = args[0] if args else kwargs["wrapper"]
            wrapped = args[1] if len(args) > 1 else kwargs["wrapped"]
            if hasattr(target, "attrs"):
                target.attrs["__wrapped__"] = wrapped
            return target
        if name == "functools.wraps":
            return Partial(self.ref("functools.update_wrapper"), (), {"wrapped": args[0]})
        if name in {"types.MethodType", "builtins.method"}:
            return Partial(args[0], (args[1],))
        if name in {"builtins.staticmethod", "builtins.classmethod"}:
            return Wrapper(args[0])
        if name == "marshal.loads":
            if not args or not isinstance(args[0], (bytes, bytearray)):
                raise self.LoadError("marshal.loads requires bytes")
            # marshal is a data parser. Code objects are kept as symbolic
            # values, never materialized as executable scanner functions.
            import marshal
            import types
            value = marshal.loads(args[0])
            return Code(args[0]) if isinstance(value, types.CodeType) else value
        if name == "marshal.load":
            data = self.read_buffer(args[0])
            return self.call(self.ref("marshal.loads"), (data,))
        if name in {"types.FunctionType", "builtins.function"}:
            if args and isinstance(args[0], Code):
                self.finding("code-execution", "materialize attacker-controlled bytecode as a function")
                return Instance("function", args)
            raise self.LoadError("FunctionType requires a code object")
        if name in {"types.CodeType", "builtins.code"}:
            return Code(args)
        if name in {"types.ModuleType", "builtins.module"}:
            return Module(args[0])
        if name == "types.SimpleNamespace":
            return Instance(name, args, dict(kwargs))
        if name in {"io.BytesIO", "io.StringIO", "_pyio.BytesIO", "_pyio.StringIO"}:
            text = name.endswith("StringIO")
            value = args[0] if args else ("" if text else b"")
            if not isinstance(value, str if text else (bytes, bytearray)):
                raise self.LoadError("invalid memory buffer contents")
            return Buffer(value, text=text)
        if name in {"pickle.Unpickler", "pickle._Unpickler"}:
            return Instance("pickle.Unpickler", args)
        if name in LOADERS:
            return self.nested_load(name, args, kwargs)
        if name in CLIENTS:
            if name in CONNECT_ON_HOST:
                host = args[0] if args else kwargs.get("host", "")
                if host:
                    self.finding("network", f"{name} connects to host")
            return Instance(name, args, dict(kwargs))
        if name in PATH_CLASSES:
            pieces = tuple(self.path_value(a) for a in args) or (".",)
            return Instance(name, pieces, value=posixpath.join(*pieces))
        if name in {"threading.Thread", "threading.Timer", "multiprocessing.Process", "multiprocessing.context.Process"}:
            attrs = dict(kwargs)
            if name == "threading.Timer":
                for key, value in zip(("interval", "function", "args", "kwargs"), args):
                    attrs[key] = value
                attrs["_target"] = attrs.get("function")
            else:
                for key, value in zip(("group", "target", "name", "args", "kwargs"), args):
                    attrs[key] = value
                attrs["_target"] = attrs.get("target")
            attrs["_args"] = attrs.get("args", ())
            attrs["_kwargs"] = attrs.get("kwargs", {})
            return Instance(name, args, attrs)
        if name in {"concurrent.futures.ThreadPoolExecutor", "concurrent.futures.ProcessPoolExecutor"}:
            return Instance(name, args, dict(kwargs))
        if name in {"timeit.Timer", "code.InteractiveInterpreter", "code.InteractiveConsole", "bdb.Bdb", "pdb.Pdb", "cProfile.Profile", "profile.Profile"}:
            if name == "timeit.Timer" and ((args and isinstance(args[0], (str, bytes))) or isinstance(kwargs.get("stmt"), (str, bytes))):
                # Timer compiles attacker supplied setup/statement in __init__.
                self.finding("code-execution", "timeit.Timer compiles source")
            return Instance(name, args, dict(kwargs))
        if name == "logging.config.dictConfig":
            return self.logging_config(args[0])
        if name in CAPABILITIES:
            return self.effect(name, args, kwargs)
        if name in {"builtins.bytes.fromhex", "builtins.bytearray.fromhex"}:
            return bytes.fromhex(args[0]) if name.startswith("builtins.bytes.") else bytearray.fromhex(args[0])
        # Calling an unbound method is the same operation as calling the bound
        # method, with its receiver supplied as the first argument.
        if args and isinstance(args[0], (Instance, Buffer, Partial, Wrapper, Namespace, Module, Getter, str, bytes, bytearray, list, dict, tuple, Lazy)):
            owner, _, method = name.rpartition(".")
            if owner in PATH_CLASSES or owner in CLIENTS or owner in {
                "functools.partial", "io.BytesIO", "io.StringIO", "builtins.str", "builtins.bytes", "builtins.bytearray",
                "builtins.list", "builtins.dict", "builtins.tuple", "builtins.object", "threading.Thread", "threading.Timer",
                "multiprocessing.Process", "concurrent.futures.ThreadPoolExecutor", "concurrent.futures.ProcessPoolExecutor",
                "pickle.Unpickler", "xmlrpc.client.ServerProxy", "operator.attrgetter", "operator.methodcaller",
            }:
                return self.bound_call(args[0], method, args[1:], kwargs)
        if name in {"builtins.object.__new__", "copyreg.__newobj__", "copyreg.__newobj_ex__"}:
            return self.construct(args[0], args[1] if name.endswith("__newobj_ex__") else args[1:], args[2] if name.endswith("__newobj_ex__") else {}, new_only=True)
        if name == "copyreg._reconstructor":
            return self.construct(args[0], (), new_only=True)
        if name.startswith("operator."):
            return self.operator(name.split(".")[-1], args)
        if name.startswith("builtins."):
            return self.builtin(name.split(".")[-1], args, kwargs)
        if name.startswith("itertools.") or name == "functools.reduce":
            return self.iterator_call(name, args, kwargs)
        decoded = self.decode(name, args, kwargs)
        if decoded is not MISSING:
            return decoded
        if name in {"os.fspath", "posixpath.normpath", "posixpath.abspath", "posixpath.basename", "posixpath.dirname", "posixpath.join", "posixpath.splitext", "posixpath.split", "posixpath.expanduser", "posixpath.expandvars"}:
            if name == "os.fspath":
                return self.path_value(args[0])
            method = name.rsplit(".", 1)[-1]
            # These lexical operations never consult the filesystem.
            return getattr(posixpath, method)(*(self.path_value(a) for a in args))
        if name == "operator.length_hint":
            return len(args[0]) if hasattr(args[0], "__len__") else 0
        if name.startswith("math.") and name.rsplit(".", 1)[-1] in {"sqrt", "ceil", "floor", "trunc", "fabs", "sin", "cos", "log", "exp", "isfinite", "isinf", "isnan", "factorial", "gcd", "prod"}:
            return getattr(math, name.rsplit(".", 1)[-1])(*args)
        if name == "numpy.dtype":
            return Instance(name, args)
        if name in {"numpy._core.multiarray._reconstruct", "numpy._core.numeric._frombuffer", "numpy.frombuffer", "numpy.ndarray", "numpy.array", "numpy.asarray"}:
            return Instance("numpy.ndarray", args)
        if name in {"numpy._core.multiarray.scalar"}:
            return Instance("numpy.scalar", args)
        if name in {"collections.OrderedDict", "collections.defaultdict", "collections.Counter"}:
            values = args[1:] if name.endswith("defaultdict") else args
            data = self.builtin("dict", values, kwargs)
            return Instance(name, args, value=data)
        if name == "collections.deque":
            return self.materialize(args[0]) if args else []
        if name.endswith(".__new__") and args and isinstance(args[0], Global):
            return self.construct(args[0], args[1:], kwargs, new_only=True)
        if name.startswith(DATA_MODULES) or name in {"_codecs.encode", "_codecs.decode", "codecs.encode", "codecs.decode"}:
            # Numeric/estimator reconstruction and immutable standard data
            # types carry state, but do not invoke a saved Python callback.
            return Instance(name, args, dict(kwargs))
        if self.is_class(fn):
            return Instance(name, args, dict(kwargs))
        raise self.AnalysisError(f"unmodelled callable {name}")

    def callable(self, obj):
        return isinstance(obj, (Global, Partial, Wrapper, Getter, Bound)) or isinstance(obj, Instance) and (obj.cls == "function" or "__call__" in obj.attrs)

    def path_value(self, value):
        if isinstance(value, Instance) and value.cls in PATH_CLASSES:
            return value.value if value.value is not None else posixpath.join(*value.args)
        if isinstance(value, (str, bytes)):
            return value
        if isinstance(value, Unknown):
            raise self.AnalysisError("unresolved filesystem path")
        raise self.LoadError("value is not a filesystem path")

    def external_path(self, path):
        if isinstance(path, (int, Unknown)):
            return True
        try:
            value = os.fsdecode(self.path_value(path))
        except self.LoadError:
            return True
        if self.artifact_path:
            absolute = os.path.abspath(value)
            if absolute == self.artifact_path:
                return False
            if self.artifact_dir and absolute.startswith(self.artifact_path + os.sep):
                return False
        return True

    def effect(self, name, args, kwargs):
        kind = CAPABILITIES[name]
        if kind == "code-execution":
            if not args and not kwargs:
                raise self.LoadError("missing source or callable")
        if kind == "file-access":
            path = args[0] if args else kwargs.get("file", kwargs.get("filename", kwargs.get("path", Unknown())))
            if isinstance(path, Buffer):
                return Instance(name, args, dict(kwargs), value=path)
            if isinstance(path, Instance) and path.cls in PATH_CLASSES:
                pass
            elif name.startswith("pathlib.") and args and isinstance(args[0], (str, bytes)):
                # Path's unbound methods require a Path receiver.
                raise self.LoadError("Path method requires a Path object")
            if not self.external_path(path):
                if name in {"os.rename", "os.renames", "os.replace", "os.link", "os.symlink", "shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.move"} and len(args) > 1:
                    if not self.external_path(args[1]):
                        return Unknown("artifact-local file access")
                else:
                    return Unknown("artifact-local file access")
        if name in {"os.system", "os.popen", "builtins.eval", "builtins.exec", "builtins.compile"} and not args:
            raise self.LoadError("missing required argument")
        self.finding(kind, f"call {name}")
        return Instance(name, args)

    def resolve_name(self, name):
        if not isinstance(name, str):
            raise self.LoadError("name is not a string")
        if ":" in name:
            module, attributes = name.split(":", 1)
            result = self.module(module)
            parts = attributes.split(".") if attributes else []
        else:
            parts = name.split(".")
            # Longest known module prefix. For an ordinary name, the last
            # component is an attribute, not an arbitrarily imported module.
            size = 1
            for i in range(1, len(parts)):
                module = ".".join(parts[:i])
                if module in MODULE_NAMES or module in self.modules or module in MODULE_ATTRS:
                    size = i
            result = self.module(".".join(parts[:size]))
            parts = parts[size:]
        for part in parts:
            result = self.attr(result, part)
        return result

    def read_buffer(self, obj, size=-1):
        if not isinstance(obj, Buffer):
            raise self.LoadError("loader requires a readable file object")
        if size is None or size < 0:
            data = obj.data[obj.pos:]
        else:
            data = obj.data[obj.pos:obj.pos + size]
        obj.pos += len(data)
        return data

    def nested_load(self, name, args, kwargs):
        if not args:
            raise self.LoadError("loader requires input")
        source = args[0]
        if name in {"pickle.loads", "dill.loads", "cloudpickle.loads"}:
            if not isinstance(source, (bytes, bytearray)):
                raise self.LoadError("loads requires bytes")
            return self.load(bytes(source), self.location + "!pickle", all_streams=False, abort_on_error=True)
        if name in {"pickle.load", "dill.load", "cloudpickle.load", "pickle.Unpickler.load"}:
            if isinstance(source, Instance) and source.cls == "pickle.Unpickler":
                source = source.args[0]
            if not isinstance(source, Buffer):
                raise self.LoadError("pickle.load requires a file object")
            stream = io.BytesIO(source.data[source.pos:])
            previous = self.location, self.protocol
            try:
                value = self.run_stream(stream)
                source.pos += stream.tell()
                return value
            finally:
                self.location, self.protocol = previous
        if name == "torch.storage._load_from_bytes":
            if not isinstance(source, (bytes, bytearray)):
                raise self.LoadError("_load_from_bytes requires bytes")
            source = Buffer(bytes(source))
        if not isinstance(source, Buffer):
            if isinstance(source, (str, bytes)) or isinstance(source, Instance) and source.cls in PATH_CLASSES:
                if self.external_path(source):
                    self.finding("file-access", f"{name} loads an external path")
                return Unknown("path-based nested model")
            raise self.LoadError("loader requires a path or file object")
        data = source.data[source.pos:]
        if not isinstance(data, (bytes, bytearray)):
            raise self.LoadError("binary loader requires bytes")
        from modelgate import dispatch
        location = self.location + "!" + name
        if name in {"numpy.load", "numpy.lib.format.read_array"}:
            import zipfile
            if data.startswith(b"\x93NUMPY"):
                objects, offset = dispatch.npy_payload(data)
                allow = kwargs.get("allow_pickle", args[1] if name.endswith("read_array") and len(args) > 1 else args[2] if len(args) > 2 else False)
                if objects:
                    if not allow:
                        raise self.LoadError("object arrays require allow_pickle=True")
                    return self.load(data[offset:], location, all_streams=False, abort_on_error=True)
                return Instance("numpy.ndarray")
            if zipfile.is_zipfile(io.BytesIO(data)):
                # NPZ members are lazy; registry loads each member as an array.
                return dispatch.scan_bytes(data, location, vm=self, depth=self.depth)
            if name.endswith("read_array"):
                raise self.LoadError("missing NumPy magic")
            allow = kwargs.get("allow_pickle", args[2] if len(args) > 2 else False)
            if not allow:
                raise self.LoadError("numpy.load will not unpickle this data")
            return self.load(data, location, all_streams=False, abort_on_error=True)
        if name in {"torch.load", "torch.storage._load_from_bytes"}:
            return dispatch.scan_bytes(data, location, vm=self, depth=self.depth)
        if name == "joblib.load":
            expanded = dispatch.decompress(data)
            while expanded is not None:
                data = expanded
                expanded = dispatch.decompress(data)
            return self.load(data, location, all_streams=False, abort_on_error=True)
        if name == "pandas.read_pickle":
            compression = kwargs.get("compression", "infer")
            if compression not in (None, "infer"):
                data = dispatch.decompress(data) or data
            return self.load(data, location, all_streams=False, abort_on_error=True)
        raise self.AnalysisError(f"unmodelled nested loader {name}")

    def operator(self, name, args):
        if name in {"getitem", "__getitem__"}:
            return self.getitem(*args)
        if name in {"setitem", "__setitem__"}:
            self.setitem(*args)
            return None
        if name in {"delitem", "__delitem__"}:
            obj, key = args
            if isinstance(obj, Namespace):
                del obj.owner.attrs[key]
            else:
                del obj[key]
            return None
        if name in {"add", "concat", "iadd", "iconcat"}:
            if isinstance(args[0], (bytes, bytearray, str, list, tuple)) and len(args[0]) + len(args[1]) > 256 * 1024 * 1024:
                raise self.AnalysisError("oversized concatenation")
            return args[0] + args[1]
        if name in {"mul", "imul"}:
            if isinstance(args[0], (bytes, bytearray, str, list, tuple)) and len(args[0]) * args[1] > 256 * 1024 * 1024:
                raise self.AnalysisError("oversized repetition")
            return args[0] * args[1]
        if name in {"sub", "isub"}:
            return args[0] - args[1]
        if name in {"truediv", "itruediv"}:
            return args[0] / args[1]
        if name in {"floordiv", "ifloordiv"}:
            return args[0] // args[1]
        if name in {"mod", "imod"}:
            return args[0] % args[1]
        if name in {"or_", "ior"}:
            return args[0] | args[1]
        if name in {"and_", "iand"}:
            return args[0] & args[1]
        if name in {"xor", "ixor"}:
            return args[0] ^ args[1]
        if name in {"not_", "truth"}:
            return not bool(args[0]) if name == "not_" else bool(args[0])
        if name in {"eq", "ne", "lt", "le", "gt", "ge", "is_", "is_not"}:
            if name in {"eq", "is_"}:
                return args[0] == args[1]
            if name in {"ne", "is_not"}:
                return args[0] != args[1]
            if name == "lt":
                return args[0] < args[1]
            if name == "le":
                return args[0] <= args[1]
            if name == "gt":
                return args[0] > args[1]
            return args[0] >= args[1]
        if name == "neg":
            return -args[0]
        if name == "pos":
            return +args[0]
        if name == "index":
            return int(args[0])
        if name == "length_hint":
            return len(args[0]) if hasattr(args[0], "__len__") else 0
        if name == "contains":
            return args[1] in args[0]
        if name == "countOf":
            return self.materialize(args[0]).count(args[1])
        if name == "indexOf":
            return self.materialize(args[0]).index(args[1])
        raise self.AnalysisError(f"unmodelled operator {name}")

    def decode(self, name, args, kwargs):
        if name in {"_codecs.encode", "codecs.encode", "_codecs.decode", "codecs.decode"}:
            encoding = str(args[1] if len(args) > 1 else "utf-8").lower().replace("_", "-")
            decode = name.endswith("decode")
            source = args[0]
            if encoding in {"base64", "base64-codec"}:
                return base64.decodebytes(source) if decode else base64.encodebytes(source)
            if encoding in {"hex", "hex-codec"}:
                return binascii.unhexlify(source) if decode else binascii.hexlify(source)
            if encoding in {"zlib", "zlib-codec", "zip"}:
                return self.decompress_literal(source) if decode else zlib.compress(source)
            if encoding in {"bz2", "bz2-codec"}:
                from modelgate.dispatch import decompress
                return decompress(source) if decode else MISSING
            if encoding not in {"utf-8", "utf8", "ascii", "latin1", "latin-1", "iso-8859-1", "utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le", "utf-32-be", "unicode-escape", "raw-unicode-escape"}:
                raise self.AnalysisError(f"unsupported literal encoding {encoding}")
            errors = args[2] if len(args) > 2 else "strict"
            return source.decode(encoding, errors) if decode else source.encode(encoding, errors)
        if name in {"base64.b64decode", "base64.standard_b64decode", "base64.decodebytes", "base64.decodestring", "base64.urlsafe_b64decode"}:
            if name.endswith("urlsafe_b64decode"):
                return base64.urlsafe_b64decode(args[0])
            return base64.b64decode(*args, **kwargs)
        if name in {"base64.b64encode", "base64.standard_b64encode", "base64.encodebytes", "base64.urlsafe_b64encode"}:
            return base64.urlsafe_b64encode(args[0]) if name.endswith("urlsafe_b64encode") else base64.b64encode(*args, **kwargs)
        if name in {"base64.b16decode", "base64.b32decode", "base64.b85decode", "base64.a85decode", "base64.b16encode", "base64.b32encode", "base64.b85encode", "base64.a85encode"}:
            return getattr(base64, name.split(".")[-1])(*args, **kwargs)
        if name in {"binascii.a2b_base64", "binascii.unhexlify", "binascii.a2b_hex", "binascii.b2a_hex", "binascii.hexlify", "binascii.b2a_base64"}:
            return getattr(binascii, name.split(".")[-1])(*args, **kwargs)
        if name == "zlib.decompress":
            return self.decompress_literal(*args, **kwargs)
        if name in {"gzip.decompress", "bz2.decompress", "lzma.decompress"}:
            from modelgate.dispatch import decompress
            value = decompress(args[0])
            if value is None:
                raise self.LoadError("invalid compressed data")
            return value
        if name == "zlib.decompressobj":
            return Instance(name, args)
        if name == "zlib.compress":
            return zlib.compress(*args, **kwargs)
        if name in {"json.loads", "ast.literal_eval"}:
            if name.startswith("json"):
                import json
                # Hooks are callbacks, not literal JSON parsing.
                value = json.loads(args[0])
                for hook in ("object_pairs_hook", "object_hook"):
                    if hook in kwargs:
                        def walk(obj):
                            if isinstance(obj, dict):
                                obj = {k: walk(v) for k, v in obj.items()}
                                return self.call(kwargs[hook], (list(obj.items()) if hook == "object_pairs_hook" else obj,))
                            if isinstance(obj, list):
                                return [walk(v) for v in obj]
                            return obj
                        value = walk(value)
                return value
            import ast
            return ast.literal_eval(args[0])
        return MISSING

    def decompress_literal(self, data, wbits=zlib.MAX_WBITS, bufsize=16384):
        obj = zlib.decompressobj(wbits)
        value = obj.decompress(data, 256 * 1024 * 1024 + 1)
        if len(value) > 256 * 1024 * 1024 or not obj.eof:
            raise self.AnalysisError("oversized or truncated embedded compression")
        return value

    def builtin(self, name, args, kwargs):
        if name in {"tuple", "list", "set", "frozenset", "dict"}:
            values = self.materialize(args[0]) if args and not isinstance(args[0], dict) else args[0] if args else []
            if name == "dict":
                if isinstance(values, Namespace):
                    values = values.owner.attrs
                result = dict(values)
                result.update(kwargs)
                return result
            return {"tuple": tuple, "list": list, "set": set, "frozenset": frozenset}[name](values)
        if name in {"bytes", "bytearray"}:
            cls = bytes if name == "bytes" else bytearray
            if args and isinstance(args[0], int) and args[0] > 256 * 1024 * 1024:
                raise self.AnalysisError("oversized byte literal")
            if args and isinstance(args[0], Lazy):
                return cls(self.materialize(args[0]))
            if args and isinstance(args[0], Buffer):
                return cls(args[0].data)
            return cls(*args, **kwargs)
        if name == "str":
            if args and isinstance(args[0], Instance) and args[0].cls in PATH_CLASSES:
                return self.path_value(args[0])
            if args and isinstance(args[0], (Global, Module, Unknown, Instance)):
                return Unknown("string of symbolic value")
            return str(*args, **kwargs)
        if name in {"int", "float", "bool", "complex"}:
            return {"int": int, "float": float, "bool": bool, "complex": complex}[name](*args, **kwargs)
        if name == "range":
            return range(*args)
        if name == "slice":
            return slice(*args)
        if name == "memoryview":
            return args[0] if isinstance(args[0], (bytes, bytearray)) else Unknown("buffer view")
        if name == "object":
            return Instance("builtins.object")
        if name == "type":
            if len(args) == 1:
                obj = args[0]
                if isinstance(obj, Instance):
                    return self.ref(obj.cls)
                if isinstance(obj, (Global, Partial, Wrapper, Module)):
                    return self.attr(obj, "__class__")
                return self.ref("builtins." + BUILTIN_TYPES.get(type(obj), "object"))
            if len(args) == 3:
                cls = self.ref("__dynamic__." + str(args[0]) + str(self.steps))
                cls.attrs.update(args[2])
                return cls
            raise self.LoadError("invalid type arguments")
        if name in {"map", "filter", "zip", "enumerate", "reversed", "iter"}:
            if name == "map":
                if len(args) < 2:
                    raise self.LoadError("map requires an iterable")
                return Lazy("map", (args[0], tuple(self.iterate(a) for a in args[1:])))
            if name == "filter":
                return Lazy("filter", (args[0], self.iterate(args[1])))
            if name == "zip":
                return Lazy("zip", tuple(self.iterate(a) for a in args))
            if name == "enumerate":
                return Lazy("enumerate", (self.iterate(args[0]), args[1] if len(args) > 1 else 0))
            if name == "reversed":
                return self.iterate(self.materialize(args[0])[::-1])
            if len(args) > 1:
                return Lazy("callable", args)
            return self.iterate(args[0])
        if name == "next":
            try:
                return self.next_value(args[0])
            except StopIteration:
                if len(args) > 1:
                    return args[1]
                raise self.LoadError("iterator exhausted") from None
        if name in {"any", "all"}:
            iterator = self.iterate(args[0])
            while True:
                try:
                    item = self.next_value(iterator)
                except StopIteration:
                    return name == "all"
                if name == "any" and item:
                    return True
                if name == "all" and not item:
                    return False
        if name in {"sorted", "min", "max"}:
            values = self.materialize(args[0]) if name == "sorted" or len(args) == 1 else list(args)
            key = kwargs.get("key")
            keys = [self.call(key, (v,)) for v in values] if key is not None else values
            if not values:
                if name == "sorted":
                    return []
                if "default" in kwargs:
                    return kwargs["default"]
                raise self.LoadError("empty sequence")
            if name == "sorted":
                try:
                    indices = sorted(range(len(values)), key=lambda i: keys[i], reverse=kwargs.get("reverse", False))
                    return [values[i] for i in indices]
                except TypeError:
                    return values
            try:
                index = (min if name == "min" else max)(range(len(values)), key=lambda i: keys[i])
                return values[index]
            except TypeError:
                return Unknown("extremum of symbolic values")
        if name == "sum":
            value = args[1] if len(args) > 1 else 0
            for item in self.materialize(args[0]):
                value = self.operator("add", (value, item))
            return value
        if name == "len":
            if isinstance(args[0], Buffer):
                return len(args[0].data)
            if isinstance(args[0], Instance) and args[0].value is not None:
                return len(args[0].value)
            return len(args[0])
        if name in {"repr", "ascii"}:
            if isinstance(args[0], PRIMITIVES):
                return repr(args[0]) if name == "repr" else ascii(args[0])
            return Unknown("representation of symbolic object")
        if name in {"abs", "round", "pow", "divmod", "chr", "ord", "hex", "oct", "bin", "hash", "id"}:
            functions = {"abs": abs, "round": round, "pow": pow, "divmod": divmod, "chr": chr, "ord": ord, "hex": hex, "oct": oct, "bin": bin, "hash": hash, "id": id}
            if name == "pow" and len(args) > 1 and abs(args[1]) > 100000:
                raise self.AnalysisError("oversized integer operation")
            return functions[name](*args)
        if name == "callable":
            return self.callable(args[0])
        if name in {"isinstance", "issubclass"}:
            obj, cls = args[:2]
            target = self.attr(obj, "__class__") if name == "isinstance" else obj
            classes = cls if isinstance(cls, tuple) else (cls,)
            return any(isinstance(c, Global) and isinstance(target, Global) and c.name == target.name for c in classes)
        if name == "print":
            if "file" in kwargs and not isinstance(kwargs["file"], Buffer):
                self.finding("file-access", "print to an external file")
            return None
        if name == "input":
            self.finding("file-access", "input reads an external stream")
            return Unknown()
        if name == "__build_class__":
            self.call(args[0], ())
            return self.ref("__dynamic__." + str(args[1]))
        if name.endswith("Error") or name in {"Exception", "BaseException", "Warning"}:
            return Instance("builtins." + name, args)
        if name in {"__new__", "__init__"}:
            return Instance("builtins.object", args)
        raise self.AnalysisError(f"unmodelled builtin {name}")

    def iterate(self, obj):
        if isinstance(obj, Lazy):
            return obj
        if isinstance(obj, (tuple, list, str, bytes, bytearray, range)):
            return Lazy("sequence", (obj,))
        if isinstance(obj, (dict, set, frozenset)):
            return Lazy("sequence", (list(obj),))
        if isinstance(obj, Namespace):
            return self.iterate(list(obj.owner.attrs))
        if isinstance(obj, Instance) and obj.value is not None:
            return self.iterate(obj.value)
        raise self.AnalysisError("cannot determine iterable contents")

    def next_value(self, it):
        self.check_budget()
        if not isinstance(it, Lazy):
            raise self.LoadError("object is not an iterator")
        kind, args = it.kind, it.args
        if kind == "sequence":
            if it.index >= len(args[0]):
                raise StopIteration
            value = args[0][it.index]
            it.index += 1
            return value
        if kind == "map":
            values = []
            for source in args[1]:
                values.append(self.next_value(source))
            return self.call(args[0], tuple(values))
        if kind in {"filter", "filterfalse"}:
            while True:
                value = self.next_value(args[1])
                predicate = self.call(args[0], (value,)) if args[0] is not None else value
                if bool(predicate) == (kind == "filter"):
                    return value
        if kind in {"zip", "starmap"}:
            if kind == "zip":
                if not args:
                    raise StopIteration
                values = []
                for source in args:
                    values.append(self.next_value(source))
                return tuple(values)
            return self.call(args[0], tuple(self.next_value(args[1])))
        if kind == "enumerate":
            value = self.next_value(args[0])
            index = it.index + args[1]
            it.index += 1
            return (index, value)
        if kind == "repeat":
            if len(args) > 1 and it.index >= args[1]:
                raise StopIteration
            it.index += 1
            return args[0]
        if kind == "count":
            value = (args[0] if args else 0) + it.index * (args[1] if len(args) > 1 else 1)
            it.index += 1
            return value
        if kind == "callable":
            value = self.call(args[0], ())
            if value == args[1]:
                raise StopIteration
            return value
        if kind == "chain":
            while it.index < len(args):
                try:
                    return self.next_value(args[it.index])
                except StopIteration:
                    it.index += 1
            raise StopIteration
        if kind == "chain_from":
            while True:
                if it.state is None:
                    it.state = self.iterate(self.next_value(args[0]))
                try:
                    return self.next_value(it.state)
                except StopIteration:
                    it.state = None
        if kind == "islice":
            source, start, stop, step = args
            if stop is not None and it.index >= stop:
                raise StopIteration
            while True:
                value = self.next_value(source)
                index = it.index
                it.index += 1
                if index >= start and (index - start) % step == 0:
                    return value
                if stop is not None and it.index >= stop:
                    raise StopIteration
        if kind in {"takewhile", "dropwhile"}:
            if it.state == "done":
                raise StopIteration
            while True:
                value = self.next_value(args[1])
                if kind == "dropwhile" and it.state == "yield":
                    return value
                predicate = self.call(args[0], (value,))
                if kind == "takewhile":
                    if not predicate:
                        it.state = "done"
                        raise StopIteration
                    return value
                if not predicate:
                    it.state = "yield"
                    return value
        if kind == "accumulate":
            source, fn = args
            value = self.next_value(source)
            if it.index == 0 and it.state is None:
                it.state = value
            else:
                it.state = self.call(fn, (it.state, value)) if fn is not None else self.operator("add", (it.state, value))
            it.index += 1
            return it.state
        if kind == "thread_result":
            if it.index >= len(args[1]):
                raise StopIteration
            item = args[1][it.index]
            it.index += 1
            return item
        raise self.AnalysisError(f"unmodelled iterator {kind}")

    def materialize(self, obj):
        if isinstance(obj, (list, tuple, str, bytes, bytearray, range, dict, set, frozenset)):
            if len(obj) > 1000000:
                raise self.AnalysisError("oversized iterable")
            return list(obj)
        it = self.iterate(obj)
        values = []
        while True:
            if len(values) > 100000:
                raise self.AnalysisError("iterator expansion limit")
            try:
                values.append(self.next_value(it))
            except StopIteration:
                return values
            if self.findings:
                return values

    def iterator_call(self, name, args, kwargs):
        name = name.split(".")[-1]
        if name == "reduce":
            values = self.iterate(args[1])
            if len(args) > 2:
                result = args[2]
            else:
                try:
                    result = self.next_value(values)
                except StopIteration:
                    raise self.LoadError("reduce of empty iterator") from None
            while True:
                try:
                    item = self.next_value(values)
                except StopIteration:
                    return result
                result = self.call(args[0], (result, item))
                if self.findings:
                    return result
        if name == "starmap":
            return Lazy("starmap", (args[0], self.iterate(args[1])))
        if name in {"filterfalse", "takewhile", "dropwhile"}:
            return Lazy(name, (args[0], self.iterate(args[1])))
        if name in {"repeat", "count"}:
            return Lazy(name, args)
        if name == "chain":
            return Lazy("chain", tuple(self.iterate(a) for a in args))
        if name == "from_iterable":
            return Lazy("chain_from", (self.iterate(args[0]),))
        if name == "islice":
            start = args[1] if len(args) > 2 else 0
            stop = args[2] if len(args) > 2 else args[1]
            step = args[3] if len(args) > 3 else 1
            return Lazy(name, (self.iterate(args[0]), start or 0, stop, step))
        if name == "accumulate":
            it = Lazy(name, (self.iterate(args[0]), args[1] if len(args) > 1 else kwargs.get("func")))
            if kwargs.get("initial") is not None:
                it.state = kwargs["initial"]
            return it
        if name == "tee":
            # No callbacks are evaluated by tee itself. Each clone buffers
            # values from the same source when subsequently consumed.
            return tuple(Lazy("tee", (args[0],)) for _ in range(args[1] if len(args) > 1 else 2))
        if name in {"product", "permutations", "combinations", "combinations_with_replacement", "zip_longest", "compress", "pairwise"}:
            import itertools
            # These iterator factories consume their inputs (but have no
            # callable argument). Use only scanner-owned materialized values.
            if name in {"permutations", "combinations", "combinations_with_replacement"}:
                inputs = (self.materialize(args[0]),) + args[1:]
            else:
                inputs = tuple(self.materialize(a) for a in args)
            result = getattr(itertools, name)(*inputs, **kwargs)
            import itertools as trusted_itertools
            values = list(trusted_itertools.islice(result, 100001))
            if len(values) > 100000:
                raise self.AnalysisError("iterator expansion limit")
            return self.iterate(values)
        raise self.AnalysisError(f"unmodelled iterator factory {name}")

    def bound_call(self, obj, method, args, kwargs):
        if method in {"__getattribute__", "__getattr__"}:
            return self.attr(obj, args[0])
        if method in {"__setattr__", "__delattr__"}:
            return self.call(self.ref("builtins.setattr" if method == "__setattr__" else "builtins.delattr"), (obj,) + args)
        if method == "__call__":
            return self.call(obj, args, kwargs)
        if method == "__getitem__":
            return self.getitem(obj, args[0])
        if method == "__setitem__":
            self.setitem(obj, *args)
            return None
        if isinstance(obj, Partial):
            if method == "__setstate__":
                self.build(obj, args[0])
                return None
            if method == "__reduce__":
                return (self.ref("functools.partial"), (obj.func,), (obj.func, obj.args, obj.kwargs, obj.attrs))
        if isinstance(obj, Buffer):
            if method in {"read", "read1", "readall"}:
                return self.read_buffer(obj, args[0] if args else -1)
            if method == "readline":
                end = obj.data.find("\n" if obj.text else b"\n", obj.pos)
                size = len(obj.data) - obj.pos if end < 0 else end + 1 - obj.pos
                if args and args[0] >= 0:
                    size = min(size, args[0])
                return self.read_buffer(obj, size)
            if method in {"getvalue", "getbuffer"}:
                return obj.data
            if method == "tell":
                return obj.pos
            if method == "seek":
                base = 0 if len(args) < 2 or args[1] == 0 else obj.pos if args[1] == 1 else len(obj.data)
                obj.pos = max(0, base + args[0])
                return obj.pos
            if method == "write":
                value = args[0]
                padding = ("\0" if obj.text else b"\0") * max(0, obj.pos - len(obj.data))
                obj.data = obj.data[:obj.pos] + padding + value + obj.data[obj.pos + len(value):]
                obj.pos += len(value)
                return len(value)
            if method == "truncate":
                obj.data = obj.data[:args[0] if args else obj.pos]
                return len(obj.data)
            if method in {"flush", "close"}:
                return None
            if method == "__setstate__":
                obj.data, obj.pos = args[0][:2]
                return None
        if isinstance(obj, Namespace):
            if method in {"get", "pop", "setdefault"}:
                key = args[0]
                value = self.getitem(obj, key, args[1] if len(args) > 1 else None)
                if method == "pop":
                    obj.owner.attrs.pop(key, None)
                elif method == "setdefault":
                    obj.owner.attrs.setdefault(key, value)
                return value
            if method == "update":
                data = args[0] if args else kwargs
                if isinstance(data, Namespace):
                    data = data.owner.attrs
                obj.owner.attrs.update(data)
                return None
            if method in {"keys", "values", "items", "copy"}:
                data = obj.owner.attrs
                return dict(data) if method == "copy" else list(getattr(data, method)())
        if isinstance(obj, Instance):
            name = obj.cls + "." + method
            if obj.cls in {"threading.Thread", "threading.Timer", "multiprocessing.Process", "multiprocessing.context.Process"} and method in {"start", "run"}:
                if obj.cls.startswith("multiprocessing") and method == "start":
                    self.finding("process", f"call {name}")
                else:
                    target = obj.attrs.get("_target", obj.attrs.get("target"))
                    if target is not None:
                        self.call(target, tuple(obj.attrs.get("_args", obj.attrs.get("args", ()))), obj.attrs.get("_kwargs", obj.attrs.get("kwargs", {})))
                return None
            if obj.cls in {"concurrent.futures.ThreadPoolExecutor", "concurrent.futures.ProcessPoolExecutor"}:
                if method in {"submit", "map"}:
                    if obj.cls.endswith("ProcessPoolExecutor"):
                        self.finding("process", f"call {name}")
                        return Unknown()
                    if method == "submit":
                        value = self.call(args[0], args[1:], kwargs)
                        return Instance("concurrent.futures.Future", value=value)
                    # Executor.map submits immediately, even if its result is
                    # never consumed. itertools.starmap/map do not.
                    it = self.builtin("map", args, {})
                    values = self.materialize(it)
                    return self.iterate(values)
                if method in {"shutdown", "__exit__"}:
                    return None
                if method == "__enter__":
                    return obj
            if obj.cls == "concurrent.futures.Future" and method == "result":
                return obj.value
            if obj.cls == "pickle.Unpickler" and method == "load":
                return self.nested_load("pickle.load", (obj.args[0],), kwargs)
            if obj.cls == "sys.modules":
                if method in {"get", "pop", "setdefault"}:
                    value = self.module(args[0])
                    return value
            if obj.cls == "xmlrpc.client.ServerProxy":
                if method == "__call__" and args and args[0] == "close":
                    return self.ref("__scanner__.noop")
                return Instance("xmlrpc.client.ServerProxy.method", (obj, method)) if not args and not kwargs else self.xmlrpc_call(obj, method, args)
            if obj.cls == "zlib.decompressobj":
                if method == "decompress":
                    return self.decompress_literal(args[0], obj.args[0] if obj.args else zlib.MAX_WBITS)
                if method == "flush":
                    return b""
            if obj.cls in PATH_CLASSES:
                path = self.path_value(obj)
                if method in {"__str__", "__fspath__", "as_posix"}:
                    return path
                if method in {"joinpath", "__truediv__"}:
                    value = posixpath.join(path, *(self.path_value(a) for a in args))
                    return Instance(obj.cls, (value,), value=value)
                if method == "with_suffix":
                    value = posixpath.splitext(path)[0] + args[0]
                    return Instance(obj.cls, (value,), value=value)
                if method == "with_name":
                    value = posixpath.join(posixpath.dirname(path), args[0])
                    return Instance(obj.cls, (value,), value=value)
                name = "pathlib.Path." + method
            if name in CAPABILITIES:
                # The receiver carries the pathname for Path methods and the
                # client identity for network methods.
                return self.effect(name, (obj,) + args, kwargs)
            if obj.cls in {"timeit.Timer"} and method in {"timeit", "repeat", "autorange"}:
                self.finding("code-execution", f"call {name}")
                return Unknown()
            if obj.cls in {"collections.OrderedDict", "collections.defaultdict", "collections.Counter"} and isinstance(obj.value, dict):
                return self.bound_call(obj.value, method, args, kwargs)
            if obj.cls in {"io.BufferedReader", "io.BufferedWriter", "gzip.GzipFile", "bz2.BZ2File", "lzma.LZMAFile"} and isinstance(obj.value, Buffer):
                return self.bound_call(obj.value, method, args, kwargs)
            if method in {"__setstate__", "__init__"} and obj.cls.startswith(DATA_MODULES):
                if method == "__setstate__":
                    self.build(obj, args[0])
                return None
            if method in {"__iter__"} and obj.value is not None:
                return self.iterate(obj.value)
            if method in {"__enter__"}:
                return obj
            if method in {"__exit__", "close"} and obj.cls in CLIENTS:
                return None
            if method in {"__reduce__", "__reduce_ex__"}:
                return (self.ref(obj.cls), obj.args, obj.attrs)
            raise self.AnalysisError(f"unmodelled method {name}")
        if isinstance(obj, Lazy):
            if method in {"__next__", "next"}:
                try:
                    return self.next_value(obj)
                except StopIteration:
                    raise self.LoadError("iterator exhausted") from None
            if method == "__iter__":
                return obj
        if isinstance(obj, (str, bytes, bytearray)):
            if method in {"encode", "decode"}:
                return self.decode("codecs." + method, (obj,) + args, kwargs)
            if method == "fromhex" and isinstance(obj, (bytes, bytearray)):
                return type(obj).fromhex(args[0])
            if method == "join":
                return obj.join(self.materialize(args[0]))
            if method in {"lower", "upper", "strip", "lstrip", "rstrip", "replace", "split", "rsplit", "splitlines", "partition", "rpartition", "removeprefix", "removesuffix", "hex", "startswith", "endswith", "find", "rfind", "index", "rindex", "count", "capitalize", "casefold", "title", "swapcase", "center", "ljust", "rjust", "zfill", "expandtabs", "translate", "format", "format_map", "__add__", "__mul__", "__mod__"}:
                return getattr(obj, method)(*args, **kwargs)
        if isinstance(obj, (list, dict, set, tuple, bytearray)):
            allowed = {
                list: {"append", "extend", "insert", "pop", "remove", "clear", "copy", "reverse", "count", "index"},
                dict: {"get", "pop", "setdefault", "popitem", "update", "clear", "copy", "keys", "values", "items"},
                set: {"add", "update", "discard", "remove", "pop", "clear", "copy", "union", "intersection", "difference"},
                tuple: {"count", "index"}, bytearray: {"append", "extend", "insert", "pop", "remove", "clear", "copy", "reverse"},
            }[type(obj)]
            if method == "sort" and isinstance(obj, list):
                obj[:] = self.builtin("sorted", (obj,), kwargs)
                return None
            if method == "extend" and isinstance(obj, (list, bytearray)):
                obj.extend(self.materialize(args[0]))
                return None
            if method in allowed:
                value = getattr(obj, method)(*args, **kwargs)
                return list(value) if method in {"keys", "values", "items"} else value
        if method == "__iter__":
            return self.iterate(obj)
        if method == "__len__":
            return self.builtin("len", (obj,), {})
        if isinstance(obj, Global) and method == "__new__":
            return self.construct(obj, args[1:] if args and args[0] is obj else args, kwargs, new_only=True)
        raise self.AnalysisError(f"unmodelled bound method {method}")

    def xmlrpc_call(self, obj, method, args):
        self.finding("network", f"invoke XML-RPC method {method}")
        return Unknown()

    def logging_config(self, config):
        if not isinstance(config, dict) or config.get("version") != 1:
            raise self.LoadError("invalid logging configuration")
        for group in ("formatters", "filters", "handlers"):
            for entry in config.get(group, {}).values():
                if not isinstance(entry, dict):
                    continue
                factory = entry.get("()")
                if factory is None and group == "handlers":
                    factory = entry.get("class")
                if factory is not None:
                    if isinstance(factory, str):
                        factory = self.resolve_name(factory)
                    fields = {k: v for k, v in entry.items() if k not in {"()", "class", "level", "formatter", "filters", "."}}
                    self.call(factory, (), fields)
        return None
