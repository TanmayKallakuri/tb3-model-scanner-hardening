"""Re-entrant deserialization: a pickle calling another loader.

A nested loader is judged by what it would load: embedded bytes are run
through the same emulator (sharing the interpreter state, so rebinding done
by the outer pickle is visible); a loader given a path reads that path,
which is file access.
"""
from __future__ import annotations

from modelgate import kb
from modelgate.symbolic import FileObj, Inst, LoadError, Opaque, Real, Sym, describe


def _arg(args, kwargs, idx, *names, default=None):
    if len(args) > idx:
        return args[idx]
    for n in names:
        if n in kwargs:
            return kwargs[n]
    return default


def _is_path(v):
    if isinstance(v, (str, bytes)):
        return True
    if isinstance(v, Inst) and isinstance(v.cls, Real):
        from modelgate.interp import canonical

        name = canonical(v.cls.obj) or ""
        return name.startswith("pathlib.")
    return False


def _maybe_bytes(v):
    """Could this model value be a bytes-like object at run time?"""
    from modelgate.interp import not_value_like
    from modelgate.symbolic import Iter

    if not_value_like(v) or isinstance(v, (Iter, FileObj)):
        return False
    if isinstance(v, Inst) and isinstance(v.cls, Real) and isinstance(v.cls.obj, type):
        return issubclass(v.cls.obj, (bytes, bytearray, memoryview))
    return True


def run(I, kind, name, args, kwargs):
    if I.depth >= 40:
        I.finding("unsupported", f"{name}: nested loaders too deep to analyse")
        return Opaque("nested load")
    I.depth += 1
    saved = (I._loc, I._pos)
    outer_site = I.site
    try:
        return _run(I, kind, name, args, kwargs, outer_site)
    except LoadError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OSError, EOFError, LookupError,
            UnicodeError, OverflowError, __import__("zipfile").BadZipFile, __import__("struct").error) as exc:
        # the nested loader raises on malformed input: the outer load fails
        raise LoadError(f"{name} failed: {type(exc).__name__}") from None
    finally:
        I.depth -= 1
        I._loc, I._pos = saved


def _run(I, kind, name, args, kwargs, site):
    from modelgate import formats
    from modelgate.pickle_vm import load_one

    loc = f"{site}>{name}"
    if kind == "pickle-bytes":
        data = _arg(args, kwargs, 0, "data", "bytes", "str")
        b = I.to_bytes(data)
        if b is None:
            if isinstance(data, Sym) and _maybe_bytes(data):
                I.finding(kb.CODE, f"{name} on data the scanner cannot reconstruct ({describe(data)})", site)
                return Opaque(f"result of {name}")
            raise LoadError(f"{name}: a bytes-like object is required")
        enc = kwargs.get("encoding", "ASCII")
        fix = kwargs.get("fix_imports", True)
        v, _ = load_one(I, b, 0, location=loc, encoding=enc if isinstance(enc, str) else "ASCII",
                        fix_imports=bool(fix))
        return v
    if kind == "pickle-file":
        fo = _arg(args, kwargs, 0, "file")
        if isinstance(fo, FileObj):
            if fo.path is not None or fo.data is None:
                if fo.path is None:
                    I.finding(kb.CODE, f"{name} on a stream the scanner cannot reconstruct", site)
                return Opaque(f"result of {name}")
            data = fo.data if isinstance(fo.data, bytes) else None
            if data is None:
                raise LoadError("file must be opened in binary mode")
            v, end = load_one(I, data, fo.pos, location=loc)
            fo.pos = end
            return v
        if isinstance(fo, Sym):
            I.finding(kb.CODE, f"{name} on an object the scanner cannot reconstruct ({describe(fo)})", site)
            return Opaque(f"result of {name}")
        raise LoadError(f"{name}: file must have read and readline attributes")
    if kind == "torch" and kwargs.get("weights_only") is True:
        # the restricted weights-only unpickler refuses every global outside
        # its allowlist: nothing the data names can be called
        return Opaque(f"result of {name}(weights_only=True)")
    if kind in ("torch", "numpy", "joblib", "pandas", "numpy-read-array"):
        target = _arg(args, kwargs, 0, "f", "file", "filename", "filepath_or_buffer", "fp")
        if _is_path(target):
            I.finding(kb.FILE, f"{name} reads the file {describe(target)}", site)
            return Opaque(f"result of {name}")
        if isinstance(target, FileObj):
            if target.path is not None:
                return Opaque(f"result of {name}")
            if not isinstance(target.data, bytes):
                I.finding(kb.CODE, f"{name} on a stream the scanner cannot reconstruct", site)
                return Opaque(f"result of {name}")
            data = target.data[target.pos:]
            if kind == "torch":
                return formats.torch_load_bytes(I, data, loc)
            if kind == "numpy":
                allow = _arg(args, kwargs, 2, "allow_pickle", default=False)
                enc = _arg(args, kwargs, 4, "encoding", default="ASCII")
                return formats.numpy_load_bytes(I, data, loc, bool(allow) if not isinstance(allow, Sym) else True,
                                                enc if isinstance(enc, str) else "ASCII")
            if kind == "numpy-read-array":
                allow = _arg(args, kwargs, 1, "allow_pickle", default=False)
                if data[:6] != b"\x93NUMPY":
                    raise LoadError("the magic string is not correct")
                return formats.numpy_load_bytes(I, data, loc, bool(allow) if not isinstance(allow, Sym) else True)
            if kind == "joblib":
                return formats.joblib_load_bytes(I, data, loc)
            v, _ = load_one(I, data, 0, location=loc)
            return v
        if isinstance(target, Sym):
            I.finding(kb.CODE, f"{name} on an object the scanner cannot reconstruct ({describe(target)})", site)
            return Opaque(f"result of {name}")
        raise LoadError(f"{name}: unsupported argument")
    if kind == "torch-bytes":
        data = I.to_bytes(_arg(args, kwargs, 0, "b"))
        if data is None:
            I.finding(kb.CODE, f"{name} on data the scanner cannot reconstruct", site)
            return Opaque(f"result of {name}")
        return formats.torch_load_bytes(I, data, loc)
    I.finding(kb.CODE, f"call of nested loader {name}", site)
    return Opaque(f"result of {name}")
