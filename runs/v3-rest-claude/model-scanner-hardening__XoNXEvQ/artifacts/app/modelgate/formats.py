"""Format detection and the load semantics of each standard loader.

Content, not extension, decides the format at every level.  Each top-level
interpretation of a file (and each archive member) is simulated in its own
Context, as a separate load; nested loaders invoked by a pickle share the
Context of the pickle that invoked them.
"""
from __future__ import annotations

import ast
import os
import struct

from modelgate import containers, keras_scan, savedmodel_scan
from modelgate.containers import ContainerError, TooLarge
from modelgate.report import Finding
from modelgate.semantics import Budget, Context, Sem
from modelgate.unpickler import Unpickler
from modelgate.values import Opaque, SimError

TORCH_MAGIC = 0x1950A86A20F9469CFC6C
TORCH_PROTOCOL = 1001
TORCH_LEGACY_PREFIX = b"\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19."
MAX_DEPTH = 8
MAX_NESTED = 12
SAVEDMODEL_SKIP = ("variables/", "assets/", "fingerprint.pb", "keras_metadata.pb", "saved_model.pb",
                   "saved_model.pbtxt")
LEGACY_NAMES = ("magic number", "protocol version", "system info", "object", "storage keys")


def _ascii(v):
    if isinstance(v, bytes):
        try:
            return v.decode("ascii")
        except UnicodeDecodeError:
            return None
    return v


def torch_zip_persistent(pid):
    if not isinstance(pid, tuple) or not pid:
        raise SimError("torch persistent id must be a tuple")
    if _ascii(pid[0]) != "storage":
        raise SimError("Unknown typename for persistent_load, expected 'storage'")
    if len(pid) != 5:
        raise SimError("bad storage persistent id")
    return Opaque("torch storage")


def torch_legacy_persistent(pid):
    if not isinstance(pid, tuple) or not pid:
        raise SimError("torch persistent id must be a tuple")
    typename = _ascii(pid[0])
    if typename == "module":
        if len(pid) < 2:
            raise SimError("bad module persistent id")
        return pid[1]
    if typename == "storage":
        if len(pid) != 6:
            raise SimError("bad storage persistent id")
        return Opaque("torch storage")
    raise SimError("unknown persistent id type")


def npy_header(data: bytes):
    """(header_dict, data_offset) of a .npy file, or raise SimError like
    numpy.lib.format.read_array would."""
    if data[:6] != containers.NPY_MAGIC or len(data) < 10:
        raise SimError("not a .npy file")
    major, minor = data[6], data[7]
    if (major, minor) == (1, 0):
        hlen = struct.unpack("<H", data[8:10])[0]
        off = 10
        enc = "latin1"
    elif (major, minor) in ((2, 0), (3, 0)):
        if len(data) < 12:
            raise SimError("truncated .npy header")
        hlen = struct.unpack("<I", data[8:12])[0]
        off = 12
        enc = "latin1" if major == 2 else "utf8"
    else:
        raise SimError(f"unsupported .npy version {major}.{minor}")
    if off + hlen > len(data):
        raise SimError("truncated .npy header")
    try:
        header = data[off:off + hlen].decode(enc)
    except UnicodeDecodeError:
        raise SimError("bad .npy header encoding") from None
    try:
        d = ast.literal_eval(header)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        if (major, minor) <= (2, 0):
            try:
                d = ast.literal_eval(_filter_header(header))
            except Exception:
                raise SimError("cannot parse .npy header") from None
        else:
            raise SimError("cannot parse .npy header") from None
    if not isinstance(d, dict) or set(d.keys()) != {"descr", "fortran_order", "shape"}:
        raise SimError("bad .npy header keys")
    if not isinstance(d["shape"], tuple) or not all(isinstance(x, int) for x in d["shape"]):
        raise SimError("bad .npy shape")
    if not isinstance(d["fortran_order"], bool):
        raise SimError("bad .npy fortran_order")
    return d, off + hlen


def _filter_header(s):
    import io
    import tokenize
    tokens = []
    last_token_was_number = False
    for token in tokenize.generate_tokens(io.StringIO(s).readline):
        token_type = token[0]
        token_string = token[1]
        if last_token_was_number and token_type == tokenize.NAME and token_string == "L":
            continue
        tokens.append(token)
        last_token_was_number = token_type == tokenize.NUMBER
    return tokenize.untokenize(tokens)


def descr_hasobject(descr) -> bool:
    try:
        from numpy.lib.format import descr_to_dtype
        return bool(descr_to_dtype(descr).hasobject)
    except ImportError:
        pass
    except Exception:
        raise SimError("bad .npy descr") from None
    if isinstance(descr, str):
        return descr.lstrip("<>|=").startswith("O")
    if isinstance(descr, list):
        for f in descr:
            if isinstance(f, tuple) and len(f) >= 2 and descr_hasobject(f[1]):
                return True
    return False


class Scanner:
    def __init__(self, deadline: float):
        self.deadline = deadline
        self.findings: list[Finding] = []

    # ------------------------------------------------------------ contexts
    def new_ctx(self) -> Context:
        ctx = Context(self.deadline)
        ctx.nested = lambda loader, data, opts, _ctx=ctx: self.nested(_ctx, loader, data, opts)
        return ctx

    def run_ctx(self, ctx: Context):
        self.findings.extend(ctx.findings)

    def unsupported(self, location, detail):
        self.findings.append(Finding("unsupported", location, detail))

    # ------------------------------------------------------------ entry points
    def scan_dir(self, path: str, location: str):
        res = savedmodel_scan.scan_saved_model_dir(path, location)
        if res is not None:
            self.findings.extend(res)
            return
        # tf.saved_model.load reads only <path>/saved_model.pb(txt): without one nothing is loaded
        return

    def scan_bytes(self, data: bytes, location: str, name: str, depth: int):
        if depth > MAX_DEPTH:
            self.unsupported(location, "containers nested too deeply")
            return
        if not data:
            return
        try:
            self._scan_bytes(data, location, name, depth)
        except Budget:
            raise
        except TooLarge as exc:
            self.unsupported(location, f"container too large to analyse: {exc}")
        except ContainerError as exc:
            self.unsupported(location, str(exc))

    def _scan_bytes(self, data, location, name, depth):
        head = data[:8]
        lname = name.lower()
        if lname in ("saved_model.pb", "saved_model.pbtxt"):
            pb = data if lname.endswith(".pb") else None
            self.findings.extend(savedmodel_scan.scan_saved_model(pb, None if pb is not None else data,
                                                                  location.rsplit("/", 1)[0]))
            return
        if head[:2] == containers.GZIP_MAGIC:
            inner = containers.gunzip(data)
            if inner is None:
                raise ContainerError("corrupt gzip data")
            self.scan_bytes(inner, f"{location}!gunzip", _strip_ext(name, (".gz", ".tgz")), depth + 1)
            return
        if head[:3] == containers.BZ2_MAGIC:
            inner = containers.bunzip2(data)
            if inner is not None:
                self.scan_bytes(inner, f"{location}!bunzip2", _strip_ext(name, (".bz2", ".tbz2", ".tbz")),
                                depth + 1)
                return
        if head[:6] == containers.XZ_MAGIC:
            inner = containers.unxz(data)
            if inner is None:
                raise ContainerError("corrupt xz data")
            self.scan_bytes(inner, f"{location}!unxz", _strip_ext(name, (".xz", ".txz")), depth + 1)
            return
        if head[:4] in (containers.ZIP_LOCAL, containers.ZIP_EOCD):
            self.scan_zip(data, location, name, depth)
            return
        if head[:6] == containers.NPY_MAGIC:
            ctx = self.new_ctx()
            try:
                self.load_npy(ctx, data, location, allow_pickle=True)
            except SimError:
                pass
            self.run_ctx(ctx)
            return
        if self._is_hdf5(data):
            self.scan_h5(data, location)
            return
        if containers.is_tar(data):
            members = containers.tar_members(data)
            self.scan_members(members, location, depth)
            return
        if containers.is_zip(data):
            # a zip with bytes before its first local header (zipfile accepts it)
            self.scan_zip(data, location, name, depth)
            return
        if containers.looks_like_zlib(data):
            # joblib.dump(compress=...) default: a raw zlib stream holding the pickle
            inner = containers.unzlib(data)
            if inner is not None:
                ctx = self.new_ctx()
                self.load_joblib_stream(ctx, inner, f"{location}!zlib")
                self.run_ctx(ctx)
                return
        if head[:2] == b"ZF":
            # joblib < 0.10 compressed pickle: "ZF" + hex length + zlib data
            try:
                int(data[2:21].strip(), 16)
                start = 22 if data[21:22] == b" " else 21
                inner = containers.unzlib(data[start:])
            except ValueError:
                inner = None
            if inner is not None:
                ctx = self.new_ctx()
                self.load_joblib_stream(ctx, inner, f"{location}!zfile")
                self.run_ctx(ctx)
                return
        self.scan_pickle_bytes(data, location, name)

    @staticmethod
    def _is_hdf5(data):
        for off in (0, 512, 1024, 2048, 4096, 8192):
            if data[off:off + 8] == containers.HDF5_MAGIC:
                return True
        return False

    # ------------------------------------------------------------ pickles
    def scan_pickle_bytes(self, data, location, name):
        lname = name.lower()
        joblib_content = b"NumpyArrayWrapper" in data or b"NDArrayWrapper" in data
        if joblib_content and (b"joblib" in data):
            ctx = self.new_ctx()
            self.load_joblib_stream(ctx, data, location)
            self.run_ctx(ctx)
            return
        ctx = self.new_ctx()
        try:
            if data.startswith(TORCH_LEGACY_PREFIX):
                self.load_torch_legacy(ctx, data, location)
            else:
                self.load_pickle_sequence(ctx, data, location)
        except SimError:
            pass
        self.run_ctx(ctx)

    def load_pickle_sequence(self, ctx, data, location, encoding="ASCII"):
        """pickle.load called repeatedly until end of file."""
        sem = Sem(ctx)
        pos = 0
        i = 0
        while pos < len(data):
            label = location if i == 0 else f"{location}[stream {i}]"
            u = Unpickler(sem, data, pos, flavor="c", encoding=encoding, label=label)
            try:
                value, end = u.load()
            except SimError:
                return
            if i == 0 and type(value) is int and value == TORCH_MAGIC:
                try:
                    self.load_torch_legacy(ctx, data, location, start=end, first_done=True)
                except SimError:
                    pass
                return
            pos = end
            i += 1

    def load_torch_legacy(self, ctx, data, location, start=0, first_done=False):
        """torch.load on a legacy checkpoint: five pickles; the fourth (the
        object) with torch's persistent_load and storage names."""
        sem = Sem(ctx)
        pos = start
        result = Opaque("legacy checkpoint")
        for i in range(1 if first_done else 0, 5):
            u = Unpickler(sem, data, pos, flavor="c", encoding="utf-8",
                          persistent=torch_legacy_persistent if i == 3 else None, torch=(i == 3),
                          label=f"{location}[{LEGACY_NAMES[i]} pickle]")
            value, pos = u.load()
            if i == 0 and not (type(value) is int and value == TORCH_MAGIC):
                raise SimError("Invalid magic number; corrupt file?")
            if i == 1 and not (type(value) is int and value == TORCH_PROTOCOL):
                raise SimError(f"Invalid protocol version: {value!r}")
            if i == 3:
                result = value
        return result, pos

    def load_joblib_stream(self, ctx, data, location):
        """joblib.load: one pickle read by joblib's (pure-Python) NumpyUnpickler,
        arrays read from the stream right after their wrapper."""
        sem = Sem(ctx)
        u = Unpickler(sem, data, 0, flavor="py", encoding="ASCII", joblib=True, label=location)
        try:
            return u.load()
        except SimError:
            return Opaque("joblib object"), len(data)

    def load_npy(self, ctx, data, location, allow_pickle=True, encoding="ASCII"):
        d, off = npy_header(data)
        if descr_hasobject(d["descr"]):
            if not allow_pickle:
                raise SimError("Object arrays cannot be loaded when allow_pickle=False")
            sem = Sem(ctx)
            u = Unpickler(sem, data, off, flavor="c", encoding=encoding, label=f"{location}[object array]")
            return u.load()
        return Opaque("numpy array"), len(data)

    def load_torch_zip(self, ctx, members: dict, prefix: str, location: str):
        name = f"{prefix}/data.pkl"
        sem = Sem(ctx)
        u = Unpickler(sem, members[name], 0, flavor="c", encoding="utf-8", persistent=torch_zip_persistent,
                      torch=True, label=f"{location}!{name}")
        return u.load()

    # ------------------------------------------------------------ zip / tar
    def scan_zip(self, data, location, name, depth):
        lz = containers.LazyZip(data)
        first = lz.first
        at0 = data[:4] == containers.ZIP_LOCAL
        if "config.json" in lz and self._is_keras_zip(lz):
            self.findings.extend(keras_scan.scan_keras_config_bytes(lz.read("config.json"), f"{location}!config.json"))
            return
        if at0 and first and "/" in first:
            prefix = first.split("/", 1)[0]
            if f"{prefix}/data.pkl" in lz:
                ctx = self.new_ctx()
                try:
                    self.load_torch_zip(ctx, {f"{prefix}/data.pkl": lz.read(f"{prefix}/data.pkl")}, prefix, location)
                except SimError:
                    pass
                self.run_ctx(ctx)
                return
        members = lz.members()
        if not at0 and data[:4] != containers.ZIP_EOCD:
            # bytes before the first local header: pickle.load / torch.load
            # would parse them as a pickle stream
            ctx = self.new_ctx()
            self.load_pickle_sequence(ctx, data, f"{location}[bytes before the zip]")
            self.run_ctx(ctx)
        self.scan_members(members, location, depth)

    @staticmethod
    def _is_keras_zip(lz):
        if "model.weights.h5" in lz or "model.weights.npz" in lz:
            return True
        if "metadata.json" in lz:
            try:
                return b"keras_version" in lz.read("metadata.json")[:65536]
            except (ContainerError, TooLarge):
                return False
        return False

    def scan_members(self, members, location, depth):
        sm_dirs = {}
        for n, b in members:
            base = n.rsplit("/", 1)[-1]
            if base in ("saved_model.pb", "saved_model.pbtxt"):
                sm_dirs.setdefault(n[: len(n) - len(base)], {})[base] = b
        for d, files in sm_dirs.items():
            pb = files.get("saved_model.pb")
            self.findings.extend(savedmodel_scan.scan_saved_model(
                pb, files.get("saved_model.pbtxt") if pb is None else None, f"{location}!{d.rstrip('/') or '.'}"))
        for n, b in members:
            skip = False
            for d in sm_dirs:
                if n.startswith(d) and n[len(d):].startswith(SAVEDMODEL_SKIP):
                    skip = True
                    break
            if skip:
                continue
            self.scan_bytes(b, f"{location}!{n}", n.rsplit("/", 1)[-1], depth + 1)

    # ------------------------------------------------------------ keras h5
    def scan_h5(self, data, location):
        try:
            self.findings.extend(keras_scan.scan_h5_bytes(data, location))
        except Budget:
            raise
        except Exception as exc:  # noqa: BLE001 - unreadable HDF5
            self.unsupported(location, f"cannot read HDF5 file: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ nested loaders
    def nested(self, ctx: Context, loader: str, data: bytes, opts: dict):
        if ctx.depth >= MAX_NESTED:
            ctx.unsupported("nested loaders nested too deeply")
            return Opaque("nested result"), 0
        saved_loc = ctx.loc
        ctx.depth += 1
        label = f"{saved_loc} -> {loader}"
        try:
            sem = Sem(ctx)
            if loader in ("pickle", "pickle-py"):
                u = Unpickler(sem, data, 0, flavor="py" if loader == "pickle-py" else "c",
                              encoding=opts.get("encoding", "ASCII"), fix_imports=opts.get("fix_imports", True),
                              label=label)
                return u.load()
            if loader == "pandas":
                try:
                    return Unpickler(sem, data, 0, flavor="c", label=label).load()
                except SimError:
                    return Unpickler(sem, data, 0, flavor="py", label=label + " (compat)").load()
            if loader in ("numpy", "npy"):
                allow = opts.get("allow_pickle", False)
                if data[:6] == containers.NPY_MAGIC:
                    return self.load_npy(ctx, data, label, allow_pickle=allow)
                if loader == "npy":
                    raise SimError("the magic string is not correct")
                if data[:4] in (containers.ZIP_LOCAL, containers.ZIP_EOCD):
                    members, _ = containers.zip_members(data)
                    for n, b in members:
                        if b[:6] == containers.NPY_MAGIC:
                            try:
                                self.load_npy(ctx, b, f"{label}!{n}", allow_pickle=allow)
                            except SimError:
                                pass
                    return Opaque("NpzFile"), len(data)
                if not allow:
                    raise SimError("Cannot load file containing pickled data when allow_pickle=False")
                return Unpickler(sem, data, 0, flavor="c", label=label).load()
            if loader == "torch":
                if data[:4] == containers.ZIP_LOCAL:
                    lz = containers.LazyZip(data)
                    first = lz.first
                    if not first or "/" not in first:
                        raise SimError("file in archive is not in a subdirectory")
                    prefix = first.split("/", 1)[0]
                    if f"{prefix}/data.pkl" not in lz:
                        raise SimError("archive has no data.pkl")
                    value, _ = self.load_torch_zip(ctx, {f"{prefix}/data.pkl": lz.read(f"{prefix}/data.pkl")},
                                                   prefix, label)
                    return value, len(data)
                return self.load_torch_legacy(ctx, data, label)
            if loader == "joblib":
                inner = data
                if data[:2] == containers.GZIP_MAGIC:
                    inner = containers.gunzip(data)
                elif data[:3] == containers.BZ2_MAGIC:
                    inner = containers.bunzip2(data)
                elif data[:6] == containers.XZ_MAGIC:
                    inner = containers.unxz(data)
                elif containers.looks_like_zlib(data):
                    inner = containers.unzlib(data)
                if inner is None:
                    raise SimError("joblib: cannot decompress")
                u = Unpickler(sem, inner, 0, flavor="py", joblib=True, label=label)
                value, _ = u.load()
                return value, len(data)
            ctx.unsupported(f"unknown nested loader {loader}")
            return Opaque("nested result"), 0
        except (ContainerError, TooLarge) as exc:
            raise SimError(f"nested {loader}: {exc}") from None
        finally:
            ctx.depth -= 1
            ctx.loc = saved_loc


def _strip_ext(name, exts):
    low = name.lower()
    for e in exts:
        if low.endswith(e):
            base = name[: len(name) - len(e)]
            return base + ".tar" if e in (".tgz", ".tbz2", ".tbz", ".txz") else base
    return name
