"""Static knowledge about Python modules, classes and callables.

Backed by ``kb_data.json`` (built by ``devtools/build_kb.py``).  Nothing here
imports a module named by an artifact.
"""
from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Python 2 module names the unpickler maps back with fix_imports (protocol < 3)
# and which do not exist under Python 3.  At protocol >= 3 importing them
# fails, which aborts the load.
try:
    import _compat_pickle as _compat  # stdlib data tables, no side effects
    IMPORT_MAPPING = dict(_compat.IMPORT_MAPPING)
    NAME_MAPPING = dict(_compat.NAME_MAPPING)
except Exception:  # noqa: BLE001 - keep working on an odd interpreter
    IMPORT_MAPPING, NAME_MAPPING = {}, {}

# Modules that are the same module under another name.
MODULE_ALIASES = {
    "nt": "os",
    "_pickle": "pickle",
    "cPickle": "pickle",
    "_io": "io",
    "__builtin__": "builtins",
    "exceptions": "builtins",
}

# Third-party packages that are not installed here but whose submodules a
# pickle may name; an attribute path through them is resolved by name.
THIRD_PARTY_MODULES = {
    "torch", "torch.serialization", "torch.storage", "torch._utils", "torch._tensor",
    "torch.hub", "torch.jit", "torch.utils", "torch.utils.data", "torch.package",
    "torch.multiprocessing", "torch.nn", "torch.nn.parameter", "torch.nn.modules",
    "torch.utils.cpp_extension", "torch.utils.model_zoo", "torch.ops", "torch._ops",
    "joblib", "joblib.numpy_pickle", "sklearn", "sklearn.externals",
    "sklearn.externals.joblib", "pandas", "pandas.io", "pandas.io.pickle",
    "pandas.compat", "pandas.compat.pickle_compat", "dill", "dill._dill", "cloudpickle",
    "cloudpickle.cloudpickle", "yaml", "requests", "requests.api", "requests.sessions",
    "urllib3", "httpx", "aiohttp", "paramiko", "pexpect", "sh", "IPython",
    "IPython.utils", "IPython.utils.process", "tensorflow", "tensorflow.io",
    "tensorflow.io.gfile", "tensorflow.keras", "tensorflow.keras.models",
    "tensorflow.keras.utils", "keras", "keras.models", "keras.utils", "keras.saving",
    "pip", "pip._internal", "setuptools", "distutils", "distutils.spawn",
    "distutils.util", "distutils.file_util", "distutils.dir_util", "jsonpickle",
    "idlelib", "lib2to3", "test", "test.support",
}


class KB:
    def __init__(self):
        with open(os.path.join(_HERE, "kb_data.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        self.modules = data["modules"]
        self.sysmodules = data["sysmodules"]
        self.classes = data["classes"]
        self.canon = data["canon"]
        self.danger_table = {k: tuple(v) for k, v in data["danger"].items()}
        self.nested_table = data["nested"]
        self.modeled = set(data["modeled"])
        self.signatures = data.get("signatures", {})
        self._kinds_raw = data.get("kinds", {})
        self._kinds = {}
        self.known_modules = set(data["known_modules"])
        py2_only = set()
        for old in IMPORT_MAPPING:
            if old not in self.known_modules and old not in self.sysmodules:
                py2_only.add(old)
        for (old, _name) in NAME_MAPPING:
            if old not in self.known_modules and old not in self.sysmodules:
                py2_only.add(old)
        py2_only.update({"__builtin__", "copy_reg", "commands", "urllib2", "httplib", "cPickle"})
        self.py2_only = py2_only

    # -- names ------------------------------------------------------------
    def canonical(self, path: str) -> str:
        return self.canon.get(path, path)

    def module_name(self, name: str) -> str:
        """Canonical module for a ``sys.modules`` key (``os.path`` -> ``posixpath``)."""
        name = MODULE_ALIASES.get(name, name)
        return self.sysmodules.get(name, name)

    def is_module(self, name: str) -> bool:
        return (name in self.known_modules or name in self.sysmodules
                or name in THIRD_PARTY_MODULES or name in MODULE_ALIASES)

    def is_known_module(self, name: str) -> bool:
        """A module whose attributes the knowledge base describes."""
        return name in self.modules

    def module_attr(self, module: str, attr: str):
        entry = self.modules.get(module)
        if entry is None:
            return None
        return entry.get(attr)

    def attr_kind(self, module: str, attr: str):
        """'func', 'class', 'enum' for an attribute of a described module, else None."""
        k = self._kinds.get(module)
        if k is None:
            raw = self._kinds_raw.get(module)
            if raw is None:
                return None
            k = self._kinds[module] = (set(raw[0].split()), set(raw[1].split()), set(raw[2].split()))
        if attr in k[0]:
            return "func"
        if attr in k[1]:
            return "class"
        if attr in k[2]:
            return "enum"
        return None

    def class_info(self, name: str):
        return self.classes.get(name)

    def class_attr(self, cls: str, attr: str):
        info = self.classes.get(cls)
        if info is None:
            return None
        return info["attrs"].get(attr)

    # -- capabilities -----------------------------------------------------
    def danger(self, name: str):
        return self.danger_table.get(name)

    def nested(self, name: str):
        return self.nested_table.get(name)

    def accepts(self, name: str, nargs: int, kwnames) -> bool:
        """False when the callable's signature rejects this call (TypeError before it runs)."""
        sig = self.signatures.get(name)
        if sig is None:
            return True
        pos, req_pos, var_pos, kwonly, kwonly_req, var_kw = sig
        if nargs > len(pos) and not var_pos:
            return False
        bound = set()
        for i, pname in enumerate(pos):
            if i < nargs:
                if pname is not None and pname in kwnames:
                    return False  # multiple values for an argument
                bound.add(i)
            elif pname is not None and pname in kwnames:
                bound.add(i)
        if any(i not in bound for i in range(req_pos)):
            return False
        named = {p for p in pos if p is not None} | set(kwonly)
        if not var_kw and any(k not in named for k in kwnames):
            return False
        return all(k in kwnames for k in kwonly_req)


_KB = None


def get() -> KB:
    global _KB
    if _KB is None:
        _KB = KB()
    return _KB
