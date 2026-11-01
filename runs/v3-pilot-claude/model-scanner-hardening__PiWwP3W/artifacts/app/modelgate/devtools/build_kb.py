#!/usr/bin/env python3
"""Build modelgate/kb_data.json, the scanner's static knowledge base.

Run once at development time (never by the scanner):

    python3 -I /app/modelgate/devtools/build_kb.py

It imports the standard library modules that are safe to import (plus numpy)
and records, for every module:

* attributes that are modules (``shutil.os`` is ``os``, ``os.path`` is
  ``posixpath``),
* attributes that are, by identity, one of the callables or objects named in
  ``capabilities.py`` (``posix.system`` is ``os.system``),
* the ``sys.modules`` aliases (``os.path`` -> ``posixpath``),

and, for every class, the attributes reachable through its MRO that are such
callables (``pdb.Pdb.run`` is ``bdb.Bdb.run``).  The scanner resolves a
pickle's globals against this data instead of importing anything.
"""
import importlib
import json
import os
import pkgutil
import sys
import types
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PKG))

from modelgate import capabilities  # noqa: E402

# Modules whose import has visible side effects or drags in GUIs/tests.
BLOCK = {
    "antigravity", "this", "__hello__", "__phello__", "__hello_alias__", "__hello_only__",
    "__phello_alias__", "idlelib", "tkinter", "turtle", "turtledemo", "test", "readline",
    "_tkinter", "pydoc_data", "xxlimited", "xxlimited_35", "xxsubtype", "_testcapi",
    "_testinternalcapi", "_testbuffer", "_testimportmultiple", "_testmultiphase",
    "_testsinglephase", "_xxtestfuzz", "_ctypes_test", "_xxsubinterpreters",
    "_xxinterpchannels", "_testclinic",
}


def blocked(name):
    parts = name.split(".")
    if parts[0] in BLOCK:
        return True
    return any(p in ("__main__", "tests", "test", "conftest", "_pyinstaller", "mypy_plugin") for p in parts[1:])


def import_all():
    names = set()
    for top in sorted(sys.stdlib_module_names):
        if top in BLOCK:
            continue
        names.add(top)
    names.add("numpy")
    mods = {}
    warnings.simplefilter("ignore")
    queue = sorted(names)
    seen = set()
    while queue:
        name = queue.pop(0)
        if name in seen or blocked(name):
            continue
        seen.add(name)
        try:
            mod = importlib.import_module(name)
        except BaseException:  # noqa: BLE001
            continue
        mods[name] = mod
        path = getattr(mod, "__path__", None)
        if path:
            try:
                for info in pkgutil.iter_modules(path, name + "."):
                    if not blocked(info.name):
                        queue.append(info.name)
            except BaseException:  # noqa: BLE001
                pass
    # numpy.testing._private.utils is wanted although numpy.testing is skipped above
    for extra in ("numpy.testing", "numpy.testing._private.utils", "numpy.lib.format",
                  "numpy.lib.npyio", "numpy.ma", "numpy.ma.core", "numpy.f2py",
                  "numpy.f2py.crackfortran", "numpy.f2py.capi_maps", "numpy.ctypeslib",
                  "numpy.lib._datasource", "numpy.lib._npyio_impl", "numpy.lib._format_impl"):
        try:
            mods[extra] = importlib.import_module(extra)
        except BaseException:  # noqa: BLE001
            pass
    for name, mod in list(sys.modules.items()):
        if isinstance(mod, types.ModuleType) and name not in mods and not blocked(name):
            mods[name] = mod
    return mods


def norm(raw):
    if isinstance(raw, (staticmethod, classmethod)):
        return raw.__func__
    return raw


def class_lookup(cls, name):
    try:
        mro = type.mro(cls) if isinstance(cls, type) else cls.__mro__
    except Exception:  # noqa: BLE001
        mro = getattr(cls, "__mro__", (cls,))
    for klass in mro:
        d = getattr(klass, "__dict__", {})
        if name in d:
            return True, norm(d[name])
    return False, None


def resolve_path(path, mods):
    """Resolve a dotted path to an object, or raise LookupError."""
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        modname = ".".join(parts[:i])
        mod = mods.get(modname)
        if mod is None:
            continue
        obj = mod
        for attr in parts[i:]:
            if isinstance(obj, types.ModuleType):
                d = vars(obj)
                if attr in d:
                    obj = d[attr]
                elif "__getattr__" in d:  # PEP 562 lazy attribute (stdlib only)
                    try:
                        obj = getattr(obj, attr)
                    except Exception as exc:  # noqa: BLE001
                        raise LookupError(path) from exc
                else:
                    raise LookupError(path)
            elif isinstance(obj, type):
                found, val = class_lookup(obj, attr)
                if not found:
                    raise LookupError(path)
                obj = val
            else:
                try:
                    obj = getattr(obj, attr)
                except Exception as exc:  # noqa: BLE001
                    raise LookupError(path) from exc
        return obj
    raise LookupError(path)


def resolve_bound(path, mods):
    """Like resolve_path, but through getattr, so class/static methods come out bound."""
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        mod = mods.get(".".join(parts[:i]))
        if mod is None:
            continue
        obj = mod
        for attr in parts[i:]:
            if isinstance(obj, types.ModuleType):
                d = vars(obj)
                if attr not in d and "__getattr__" not in d:
                    raise LookupError(path)
            obj = getattr(obj, attr)
        return obj
    raise LookupError(path)


def is_simple(obj):
    return obj is None or isinstance(obj, (bool, int, float, str, bytes)) and len(repr(obj)) < 200


def main():
    mods = import_all()
    print(f"imported {len(mods)} modules", file=sys.stderr)

    tables = []
    for path, val in capabilities.DANGEROUS.items():
        tables.append((path, "danger", val))
    for path, val in capabilities.NESTED_LOADERS.items():
        tables.append((path, "nested", val))
    for path in capabilities.MODELED:
        tables.append((path, "modeled", None))
    for path in capabilities.SPECIAL_OBJECTS:
        tables.append((path, "special", None))

    canon_of_id = {}     # id(obj) -> canonical name
    keep = {}            # id -> obj (keeps objects alive)
    canon = {}           # written path -> canonical name
    danger = {}          # canonical -> [kind, cond]
    nested = {}          # canonical -> loader
    modeled = set()
    unresolved = []
    for path, what, val in tables:
        try:
            obj = resolve_path(path, mods)
        except LookupError:
            obj = None
        if obj is None or is_simple(obj):
            cname = path
            unresolved.append(path)
        else:
            cname = canon_of_id.setdefault(id(obj), path)
            keep[id(obj)] = obj
        canon[path] = cname
        if what == "danger":
            kind, cond = (val, None) if isinstance(val, str) else val
            danger.setdefault(cname, [kind, cond])
            # a dangerous class is dangerous through its own __init__ too
            if isinstance(obj, type):
                found, init = class_lookup(obj, "__init__")
                if found and init is not object.__init__ and isinstance(init, types.FunctionType):
                    iname = canon_of_id.setdefault(id(init), path + ".__init__")
                    keep[id(init)] = init
                    danger.setdefault(iname, [kind, ["shift", cond]] if cond else [kind, None])
        elif what == "nested":
            nested.setdefault(cname, val)
        elif what == "modeled":
            modeled.add(cname)
    interesting = dict(canon_of_id)

    # call signatures of capability-bearing callables, so a call the real
    # callable would reject (wrong arity) is not reported as executing it
    import inspect
    signatures = {}
    for path, what, val in tables:
        if what not in ("danger", "nested", "modeled"):
            continue
        cname = canon.get(path)
        if cname in signatures:
            continue
        try:
            obj = resolve_bound(path, mods)  # as a call would see it: classmethods bound
            sig = inspect.signature(obj)
        except Exception:  # noqa: BLE001 - no signature available
            continue
        pos, req_pos, kwonly_req, names = [], 0, [], []
        var_pos = var_kw = False
        for prm in sig.parameters.values():
            if prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD):
                pos.append(prm.name if prm.kind == prm.POSITIONAL_OR_KEYWORD else None)
                if prm.default is prm.empty:
                    req_pos = len(pos)
            elif prm.kind == prm.VAR_POSITIONAL:
                var_pos = True
            elif prm.kind == prm.KEYWORD_ONLY:
                names.append(prm.name)
                if prm.default is prm.empty:
                    kwonly_req.append(prm.name)
            elif prm.kind == prm.VAR_KEYWORD:
                var_kw = True
        signatures[cname] = [pos, req_pos, var_pos, names, kwonly_req, var_kw]

    # sys.modules aliases
    sysmods = {}
    for key, mod in sys.modules.items():
        if isinstance(mod, types.ModuleType) and getattr(mod, "__name__", key) != key:
            sysmods[key] = mod.__name__

    modules = {}
    classes = {}
    class_seen = {}

    def describe_value(value):
        if isinstance(value, types.ModuleType):
            return "m:" + value.__name__
        v = norm(value)
        name = interesting.get(id(v))
        if name is not None:
            if isinstance(v, type):
                return "c:" + name
            return ("f:" if callable(v) else "o:") + name
        if isinstance(v, type):
            cname = class_name(v)
            if cname is not None:
                return "c:" + cname
        return None

    def class_name(cls):
        if id(cls) in class_seen:
            return class_seen[id(cls)]
        class_seen[id(cls)] = None
        try:
            qual = f"{cls.__module__}.{cls.__qualname__}"
        except Exception:  # noqa: BLE001
            return None
        attrs = {}
        try:
            mro = type.mro(cls) if isinstance(cls, type) and type(cls) is type else cls.__mro__
        except Exception:  # noqa: BLE001
            mro = getattr(cls, "__mro__", (cls,))
        for klass in mro:
            if klass is object or klass is type:
                continue  # handled generically by the scanner
            d = getattr(klass, "__dict__", {})
            for k, raw in list(d.items()):
                if k in attrs:
                    continue
                v = norm(raw)
                name = interesting.get(id(v))
                if name is not None:
                    if isinstance(raw, (staticmethod, classmethod)) or type(raw).__name__ == "classmethod_descriptor":
                        tag = "s:"  # never binds the instance
                    else:
                        tag = "c:" if isinstance(v, type) else "f:" if callable(v) else "o:"
                    attrs[k] = tag + name
                elif isinstance(v, types.ModuleType):
                    attrs[k] = "m:" + v.__name__
        is_interesting = id(cls) in interesting
        if not attrs and not is_interesting:
            return None
        if is_interesting:
            qual = interesting[id(cls)]
        heap = bool(cls.__flags__ & (1 << 9))
        classes[qual] = {"attrs": attrs, "heap": heap}
        class_seen[id(cls)] = qual
        keep[id(cls)] = cls
        return qual

    for modname, mod in sorted(mods.items()):
        entry = {}
        try:
            items = list(vars(mod).items())
        except Exception:  # noqa: BLE001
            continue
        for attr, value in items:
            if not isinstance(attr, str):
                continue
            desc = describe_value(value)
            if desc is None and not isinstance(value, (types.ModuleType, type)):
                v = norm(value)
                name = interesting.get(id(v))
                if name is not None:
                    desc = "o:" + name
            if desc is not None:
                entry[attr] = desc
        modules[modname] = entry
    # classes reached only through interesting paths, and lazily exported
    # module attributes the vars() walk above cannot see
    for path, cname in canon.items():
        try:
            obj = resolve_path(path, mods)
        except LookupError:
            continue
        if isinstance(obj, type):
            class_name(obj)
        modname, _, attr = path.rpartition(".")
        if modname in modules and attr not in modules[modname]:
            desc = describe_value(obj)
            if desc is not None:
                modules[modname][attr] = desc

    # kind of every module attribute: f(unction), c(lass), e(num class)
    import enum
    kinds = {}
    for modname, mod in sorted(mods.items()):
        f, c, e = [], [], []
        try:
            items = list(vars(mod).items())
        except Exception:  # noqa: BLE001
            continue
        for attr, value in items:
            if not isinstance(attr, str) or isinstance(value, types.ModuleType):
                continue
            if isinstance(value, type):
                try:
                    (e if issubclass(value, enum.Enum) else c).append(attr)
                except TypeError:
                    c.append(attr)
            elif callable(value):
                f.append(attr)
        kinds[modname] = [" ".join(sorted(f)), " ".join(sorted(c)), " ".join(sorted(e))]

    out = {
        "kinds": kinds,
        "modules": modules,
        "sysmodules": sysmods,
        "classes": classes,
        "canon": canon,
        "danger": danger,
        "nested": nested,
        "signatures": signatures,
        "modeled": sorted(modeled),
        "known_modules": sorted(mods),
    }
    dest = os.path.join(PKG, "kb_data.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"), sort_keys=True)
    print(f"wrote {dest}: {len(modules)} modules, {len(classes)} classes, "
          f"{len(danger)} dangerous, {len(unresolved)} unresolved paths", file=sys.stderr)
    for p in unresolved:
        print("  unresolved:", p, file=sys.stderr)


if __name__ == "__main__":
    main()
