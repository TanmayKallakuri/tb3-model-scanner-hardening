"""Read-only index of the standard library (and numpy) built offline.

``data/pyindex.json`` maps every module of this Python installation to its
attributes, and every function / class reachable from them to a canonical
object id, so that aliases (``posix.system`` is ``os.system``, ``shutil.os``
is ``os``, ``_io.open`` is ``builtins.open``) resolve to one identity without
the scanner ever importing a module named by an artifact.
"""
from __future__ import annotations

import json
import os
import sys

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pyindex.json")

_modules: dict | None = None
_objects: dict | None = None
_aliases: dict | None = None


def _load():
    global _modules, _objects, _aliases
    if _modules is None:
        with open(_DATA, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        _modules = doc["modules"]
        _objects = doc["objects"]
        _aliases = doc.get("aliases", {})


def module_canonical(name: str) -> str:
    """The canonical name of a module that is importable under several names
    (``posixpath`` is ``os.path``)."""
    _load()
    return _aliases.get(name, name)


def modules() -> dict:
    _load()
    return _modules


def objects() -> dict:
    _load()
    return _objects


STDLIB_TOP = frozenset(sys.stdlib_module_names)


def module_table(name: str):
    """The attribute table of an indexed module, or None."""
    _load()
    return _modules.get(name)


def object_entry(oid: str):
    _load()
    return _objects.get(oid)


def module_attr_ref(mod: str, attr: str):
    """Reference string for ``mod.attr`` ("m:..", "o:..", "v") or None."""
    t = module_table(mod)
    if t is None:
        return None
    r = t.get(attr)
    if r == "=":
        return f"o:{mod}.{attr}"
    return r


def class_lookup(cls_oid: str, attr: str):
    """Find ``attr`` on an indexed class through its MRO.

    Returns (ref, owner_oid) where ref is "o:..", "cm:..", "sm:.." or "v",
    or None when no class in the MRO defines it (``object`` excluded).
    """
    e = object_entry(cls_oid)
    if not e or e[0] != "c":
        return None
    chain = [cls_oid] + list(e[3])
    for c in chain:
        ce = object_entry(c)
        if not ce or ce[0] != "c":
            continue
        own = ce[4]
        if attr in own:
            r = own[attr]
            if r == "=":
                r = f"o:{c}.{attr}"
            return r, c
    return None


def class_info(cls_oid: str):
    """(heap_type, has_instance_dict, mro list) for an indexed class."""
    e = object_entry(cls_oid)
    if not e or e[0] != "c":
        return None
    return bool(e[1]), bool(e[2]), list(e[3])


def owner_module(oid: str):
    """The longest indexed module name that prefixes ``oid``."""
    parts = oid.split(".")
    for i in range(len(parts) - 1, 0, -1):
        cand = ".".join(parts[:i])
        if module_table(cand) is not None:
            return cand
    return None


def resolve_path(path: str):
    """Resolve a dotted path such as ``http.client.HTTPConnection.request``
    to its canonical object id (used to normalise the capability tables).
    Returns None for paths outside the index."""
    parts = path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        mod = ".".join(parts[:i])
        if module_table(mod) is None:
            continue
        cur_ref = f"m:{mod}"
        ok = True
        for p in parts[i:]:
            if cur_ref.startswith("m:"):
                r = module_attr_ref(cur_ref[2:], p)
            elif cur_ref.startswith("o:"):
                hit = class_lookup(cur_ref[2:], p)
                r = hit[0] if hit else None
                if r and (r.startswith("cm:") or r.startswith("sm:")):
                    r = "o:" + r[3:]
            else:
                r = None
            if not r or r == "v":
                ok = False
                break
            cur_ref = r
        if ok and cur_ref.startswith("o:"):
            return cur_ref[2:]
        if ok and cur_ref.startswith("m:"):
            return cur_ref
        return None
    return None
