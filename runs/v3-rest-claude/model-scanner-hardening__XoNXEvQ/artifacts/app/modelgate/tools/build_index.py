"""Regenerate modelgate/data/pyindex.json from this Python installation.

Run OFFLINE, never as part of a scan (it imports the whole standard library):

    cd "$(mktemp -d)" && python3 -I /app/modelgate/tools/build_index.py /app/modelgate/data/pyindex.json
"""

import builtins
import importlib
import inspect
import json
import pkgutil
import sys
import types
import warnings

warnings.simplefilter("ignore")

SKIP_TOP = {
    "antigravity", "this", "__hello__", "__phello__", "idlelib", "test", "tkinter", "turtle",
    "turtledemo", "lib2to3", "pydoc_data", "ensurepip", "venv", "msilib", "winreg", "winsound",
    "_winapi", "msvcrt", "nt", "_msi", "_overlapped", "nturl2path", "_wmi", "_tkinter",
    "__main__", "_testcapi", "_testinternalcapi", "_testbuffer", "_testimportmultiple",
    "_testmultiphase", "_testsinglephase", "_xxtestfuzz", "_testclinic", "xxlimited",
    "xxlimited_35", "xxsubtype", "_ctypes_test", "_scproxy", "_aix_support", "_osx_support_x",
    "sitecustomize", "usercustomize", "_pyrepl", "readline", "__mp_main__",
}
EXTRA_TOP = ["numpy"]

PRIORITY = [
    "builtins", "os", "os.path", "sys", "io", "functools", "operator", "itertools",
    "collections", "types", "copyreg", "codecs", "subprocess", "socket", "shutil", "pickle",
    "marshal", "importlib", "pathlib", "tempfile", "copy", "threading", "multiprocessing",
    "asyncio", "ctypes", "urllib.request", "http.client", "ftplib", "smtplib", "numpy",
]


def is_skipped(name):
    parts = name.split(".")
    if parts[0] in SKIP_TOP:
        return True
    if "__main__" in parts or "tests" in parts or "test" in parts or "_tests" in parts:
        return True
    if name.startswith("numpy.") and any(p in ("f2py", "distutils", "_pyinstaller", "testing._private.extbuild") for p in parts):
        return name.startswith("numpy.distutils") or name.startswith("numpy._pyinstaller")
    if name.startswith("encodings.") and name not in ("encodings.aliases",):
        return True
    return False


def collect_module_names():
    names = set()
    tops = sorted(set(sys.stdlib_module_names) | set(EXTRA_TOP))
    for top in tops:
        if is_skipped(top):
            continue
        try:
            mod = importlib.import_module(top)
        except Exception:
            continue
        names.add(top)
        path = getattr(mod, "__path__", None)
        if path:
            for info in pkgutil.walk_packages(path, top + ".", onerror=lambda n: None):
                if is_skipped(info.name):
                    continue
                try:
                    importlib.import_module(info.name)
                    names.add(info.name)
                except BaseException:
                    pass
    # include anything that got imported as a side effect
    for name in list(sys.modules):
        if name in ("__main__", "__mp_main__"):
            continue
        if not is_skipped(name) and isinstance(sys.modules.get(name), types.ModuleType):
            names.add(name)
    return names


def order_key(name):
    if name in PRIORITY:
        return (0, PRIORITY.index(name), name)
    private = any(p.startswith("_") for p in name.split("."))
    return (2 if private else 1, name.count("."), name)


def kind_of(obj):
    if isinstance(obj, types.ModuleType):
        return "module"
    if isinstance(obj, type):
        return "c"
    if isinstance(obj, types.FunctionType):
        return "f"
    if isinstance(obj, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        return "b"
    if isinstance(obj, (types.MethodDescriptorType, types.WrapperDescriptorType, types.ClassMethodDescriptorType)):
        return "md"
    if isinstance(obj, types.MethodWrapperType):
        return "b"
    if isinstance(obj, types.MethodType):
        return "bm"  # bound python method
    if isinstance(obj, (property, types.GetSetDescriptorType, types.MemberDescriptorType)):
        return "p"
    if isinstance(obj, classmethod):
        return "cm"
    if isinstance(obj, staticmethod):
        return "sm"
    if callable(obj):
        return "x"
    return None


def main(out):
    mod_names = sorted(collect_module_names(), key=order_key)
    modules = {}
    objects = {}
    ids = {}  # id(obj) -> objid
    keep = []  # keep objects alive so ids stay unique

    def obj_id(obj, path):
        key = id(obj)
        if key in ids:
            return ids[key]
        ids[key] = path
        keep.append(obj)
        return path

    mod_by_id = {}
    for name in mod_names:
        m = sys.modules.get(name)
        if isinstance(m, types.ModuleType):
            mod_by_id.setdefault(id(m), name)

    special = {"sys.modules": sys.modules, "os.environ": __import__("os").environ,
               "sys.stdin": sys.stdin, "sys.stdout": sys.stdout, "sys.stderr": sys.stderr}
    for path, val in special.items():
        oid = obj_id(val, path)
        objects.setdefault(oid, ["i"])

    pending_classes = []
    for name in mod_names:
        m = sys.modules.get(name)
        if not isinstance(m, types.ModuleType):
            continue
        table = {}
        try:
            items = list(vars(m).items())
        except Exception:
            continue
        for attr, val in items:
            if not isinstance(attr, str):
                continue
            if isinstance(val, types.ModuleType):
                mname = mod_by_id.get(id(val))
                table[attr] = "m:" + (mname or getattr(val, "__name__", attr))
                continue
            if id(val) in ids and objects.get(ids[id(val)]) == ["i"]:
                table[attr] = "o:" + ids[id(val)]
                continue
            k = kind_of(val)
            if k is None:
                table[attr] = "v"
                continue
            oid = obj_id(val, f"{name}.{attr}")
            table[attr] = "o:" + oid
            if oid not in objects:
                if k == "c":
                    objects[oid] = ["c"]
                    pending_classes.append((oid, val))
                elif k == "sm" or k == "cm":
                    objects[oid] = [k]
                else:
                    objects[oid] = [k]
        modules[name] = table


    # classes: own attributes and MRO
    seen = set()
    while pending_classes:
        oid, cls = pending_classes.pop(0)
        if oid in seen:
            continue
        seen.add(oid)
        own = {}
        try:
            cdict = dict(vars(cls))
        except Exception:
            cdict = {}
        for attr, val in cdict.items():
            if not isinstance(attr, str):
                continue
            if isinstance(val, types.ModuleType):
                continue
            k = kind_of(val)
            if k is None:
                own[attr] = "v"
                continue
            if isinstance(val, types.ClassMethodDescriptorType):
                sub = obj_id(val, f"{oid}.{attr}")
                objects.setdefault(sub, ["md"])
                own[attr] = f"cm:{sub}"
                continue
            if k in ("sm", "cm"):
                inner = val.__func__
                ik = kind_of(inner) or "x"
                iid = obj_id(inner, f"{oid}.{attr}")
                objects.setdefault(iid, [ik])
                own[attr] = f"{k}:{iid}"
                continue
            sub = obj_id(val, f"{oid}.{attr}")
            own[attr] = "o:" + sub
            if sub not in objects:
                objects[sub] = [k]
                if k == "c":
                    pending_classes.append((sub, val))
        mro = []
        for base in cls.__mro__[1:]:
            if base is object:
                continue
            bid = ids.get(id(base))
            if bid is None:
                mod = getattr(base, "__module__", "?")
                bid = obj_id(base, f"{mod}.{getattr(base, '__qualname__', base.__name__)}")
                objects.setdefault(bid, ["c"])
                pending_classes.append((bid, base))
            mro.append(bid)
        heap = bool(cls.__flags__ & (1 << 9))
        has_dict = "__dict__" in dir(cls) or any("__dict__" in vars(b) for b in cls.__mro__ if b is not object)
        objects[oid] = ["c", int(heap), int(has_dict), mro, own]

    by_oid = {}
    for k, v in ids.items():
        pass
    for obj in keep:
        oid = ids.get(id(obj))
        if oid is None or oid not in objects:
            continue
        entry = objects[oid]
        flags = ""
        if inspect.isgeneratorfunction(obj):
            flags += "g"
        if inspect.iscoroutinefunction(obj):
            flags += "a"
        if inspect.isasyncgenfunction(obj):
            flags += "A"
        sig = None
        if callable(obj) and entry[0] in ("f", "b", "c", "md", "x", "bm"):
            try:
                s = inspect.signature(obj)
                parts = []
                for p in s.parameters.values():
                    parts.append(f"{p.name}:{int(p.kind)}{'=' if p.default is not p.empty else ''}")
                sig = ",".join(parts)
            except Exception:
                sig = None
        objects[oid] = entry + [{"fl": flags, "sig": sig}] if entry[0] != "c" else entry + [{"fl": flags, "sig": sig}]

    aliases = {}
    for name in mod_names:
        m = sys.modules.get(name)
        if isinstance(m, types.ModuleType):
            canon = mod_by_id.get(id(m))
            if canon and canon != name:
                aliases[name] = canon
    with open(out, "w") as fh:
        json.dump(compact({"modules": modules, "objects": objects, "aliases": aliases}), fh,
                  separators=(",", ":"), sort_keys=True)
    print("modules", len(modules), "objects", len(objects))



def compact(d):
    M, O = d["modules"], d["objects"]
    DROP = {"__doc__", "__module__", "__weakref__", "__annotations__", "__slots__", "__qualname__",
            "__abstractmethods__", "_abc_impl", "__parameters__", "__orig_bases__", "__match_args__",
            "__dataclass_fields__", "__dataclass_params__", "__firstlineno__", "__static_attributes__",
            "__type_params__", "__hash__"}
    mods = {}
    for m, t in M.items():
        mods[m] = {a: ("=" if r == "o:" + m + "." + a else r) for a, r in t.items()}
    objs = {}
    for oid, e in O.items():
        k = e[0]
        if k == "c":
            heap, has_dict, mro, own, meta = e[1], e[2], e[3], e[4], e[5] if len(e) > 5 else {}
            own2 = {a: ("=" if r == "o:" + oid + "." + a else r) for a, r in own.items() if a not in DROP}
            objs[oid] = ["c", heap, has_dict, mro, own2, meta.get("sig")]
        else:
            meta = e[1] if len(e) > 1 else {}
            fl = meta.get("fl") or ""
            sig = meta.get("sig")
            objs[oid] = [k, fl, sig] if (fl or sig is not None) else [k]
    return {"modules": mods, "objects": objs, "aliases": d.get("aliases", {})}


if __name__ == "__main__":
    main(sys.argv[1])
