"""Format detection and the loaders the registry's policy names.

``scan_blob`` analyses one artifact (or archive member) from its bytes: the
content, never the name, decides the format.  The nested-load helpers model
what ``torch.load``, ``numpy.load`` and ``joblib.load`` read when a pickle
hands them embedded data.
"""
from __future__ import annotations

import ast
import bz2
import io
import lzma
import struct
import tarfile
import zipfile
import zlib

from modelgate.pickle_vm import (
    PickleStop, ScanAbort, Unpickler, is_torch_legacy, run_streams, run_torch_legacy,
)
from modelgate.values import NpzObj, Opaque

MAX_MEMBER = 256 * 1024 * 1024        # largest archive member or decompressed stream
MAX_TOTAL = 1024 * 1024 * 1024        # total bytes unpacked for one artifact
MAX_DEPTH = 8                          # container nesting

NPY_MAGIC = b"\x93NUMPY"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


class Budget:
    def __init__(self):
        self.used = 0

    def take(self, n):
        self.used += n
        if self.used > MAX_TOTAL:
            raise ScanAbort("archive expands beyond the size budget")


def _budget(ctx):
    b = getattr(ctx, "budget", None)
    if b is None:
        b = ctx.budget = Budget()
    return b


# ------------------------------------------------------------ decompression
def _drain(decomp_factory, data, ctx, what):
    """Decompress all concatenated members; tolerate trailing junk."""
    out = bytearray()
    rest = data
    members = 0
    while rest:
        d = decomp_factory()
        try:
            chunk = d.decompress(rest, MAX_MEMBER + 1 - len(out))
        except (OSError, EOFError, ValueError, zlib.error, lzma.LZMAError) as exc:
            if members == 0:
                raise ScanAbort(f"{what} data cannot be decompressed: {exc}") from None
            break
        out += chunk
        if len(out) > MAX_MEMBER:
            raise ScanAbort(f"{what} member expands beyond the size budget")
        members += 1
        if not d.eof:
            break  # truncated member: keep what decompressed
        rest = d.unused_data
        if what == "gzip":
            rest = rest.lstrip(b"\x00")
            if rest[:2] != b"\x1f\x8b":
                break
        elif what == "bz2":
            if rest[:3] != b"BZh":
                break
        elif what == "xz":
            if rest[:6] != b"\xfd7zXZ\x00":
                rest = rest.lstrip(b"\x00")
                if rest[:6] != b"\xfd7zXZ\x00":
                    break
        else:
            break
    _budget(ctx).take(len(out))
    return bytes(out)


def gunzip(data, ctx):
    return _drain(lambda: zlib.decompressobj(16 + zlib.MAX_WBITS), data, ctx, "gzip")


def bunzip2(data, ctx):
    return _drain(bz2.BZ2Decompressor, data, ctx, "bz2")


def unxz(data, ctx):
    return _drain(lambda: lzma.LZMADecompressor(lzma.FORMAT_XZ), data, ctx, "xz")


def unlzma(data, ctx):
    return _drain(lambda: lzma.LZMADecompressor(lzma.FORMAT_ALONE), data, ctx, "lzma")


def unzlib(data, ctx):
    return _drain(zlib.decompressobj, data, ctx, "zlib")


# ------------------------------------------------------------- detection
def zip_end_ok(data):
    """True when the file ends with a zip end-of-central-directory record."""
    if len(data) < 22:
        return False
    tail = data[-(22 + 65535):]
    i = tail.rfind(b"PK\x05\x06")
    while i >= 0:
        comment_len = struct.unpack("<H", tail[i + 20:i + 22])[0] if i + 22 <= len(tail) else -1
        if i + 22 + comment_len == len(tail):
            return True
        i = tail.rfind(b"PK\x05\x06", 0, i)
    return False


def _tar_header_ok(block):
    if len(block) < 512:
        return False
    try:
        chk = int(block[148:156].split(b"\x00", 1)[0].strip() or b"-1", 8)
    except ValueError:
        return False
    unsigned = sum(block[:148]) + 256 + sum(block[156:512])
    signed = sum(struct.unpack("148b", block[:148])) + 256 + sum(struct.unpack("356b", block[156:512]))
    return chk in (unsigned, signed) and block[:1] != b"\x00"


def sniff(data):
    head = data[:8]
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    if head[:3] == b"BZh" and head[3:4] in b"123456789" and head[3:4] != b"":
        return "bz2"
    if head[:6] == b"\xfd7zXZ\x00":
        return "xz"
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06") or (head[:2] == b"PK" and zip_end_ok(data)):
        return "zip"
    if head[:6] == NPY_MAGIC:
        return "npy"
    if head[:8] == HDF5_MAGIC:
        return "hdf5"
    if len(data) >= 512 and (data[257:262] == b"ustar" or _tar_header_ok(data[:512])):
        return "tar"
    if zip_end_ok(data):
        return "zip"
    if head[:1] == b"\x78" and len(head) > 1 and (head[0] * 256 + head[1]) % 31 == 0:
        return "zlib"
    if head[:3] == b"]\x00\x00":
        return "lzma"
    for off in (512, 1024, 2048, 4096, 8192):
        if data[off:off + 8] == HDF5_MAGIC:
            return "hdf5"
    return "other"


# ---------------------------------------------------------------- dispatch
def scan_blob(ctx, data, location, name="", depth=0, top=False):
    """Analyse an artifact or archive member given its content."""
    ctx.check_time()
    if depth > MAX_DEPTH:
        raise ScanAbort("containers nested too deeply")
    kind = sniff(data)
    if kind == "gzip":
        return scan_blob(ctx, gunzip(data, ctx), f"{location}|gunzip", name[:-3] if name.endswith(".gz") else name, depth + 1)
    if kind == "bz2":
        return scan_blob(ctx, bunzip2(data, ctx), f"{location}|bunzip2", name, depth + 1)
    if kind == "xz":
        return scan_blob(ctx, unxz(data, ctx), f"{location}|unxz", name, depth + 1)
    if kind == "zip":
        return scan_zip(ctx, data, location, depth)
    if kind == "tar":
        try:
            return scan_tar(ctx, data, location, depth)
        except tarfile.TarError:
            pass
    if kind == "npy":
        try:
            scan_npy(ctx, data, location, allow_pickle=True)
        except PickleStop:
            pass  # numpy.load raises here; what ran before the failure is already recorded
        return None
    if kind == "hdf5":
        from modelgate import keras_scan
        return keras_scan.scan_h5(ctx, data, location)
    if kind in ("zlib", "lzma"):
        try:
            inner = unzlib(data, ctx) if kind == "zlib" else unlzma(data, ctx)
        except ScanAbort:
            inner = None
        if inner:
            return scan_blob(ctx, inner, f"{location}|{'inflate' if kind == 'zlib' else 'unlzma'}", name, depth + 1)
    return scan_pickle_file(ctx, data, location, name, top)


def scan_pickle_file(ctx, data, location, name, top):
    """Raw pickle streams, a legacy torch checkpoint, or a joblib file."""
    if is_torch_legacy(data):
        run_torch_legacy(ctx, data, location)
        return
    count, stop, _results, _ok = run_streams(ctx, data, location, joblib=True)
    if count == 0 and top and not ctx.findings and _not_a_pickle(ctx, location):
        ctx.report("unsupported", location, "content is not a recognised model format")


def _not_a_pickle(ctx, location):
    return any(n.startswith(f"{location}: not a pickle") for n in ctx.notes)


# -------------------------------------------------------------------- zip
def _zip_open(data):
    try:
        return zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, ValueError, OSError, EOFError, struct.error, NotImplementedError) as exc:
        raise ScanAbort(f"zip archive cannot be read: {exc}") from None


def _zip_read(ctx, zf, info, location):
    if info.file_size > MAX_MEMBER:
        raise ScanAbort(f"{location}!{info.filename}: member too large")
    if info.flag_bits & 0x1:
        ctx.report("unsupported", f"{location}!{info.filename}", "encrypted archive member cannot be analysed")
        return None
    try:
        out = zf.read(info)
    except (zipfile.BadZipFile, ValueError, OSError, EOFError, NotImplementedError, zlib.error,
            lzma.LZMAError, RuntimeError) as exc:
        ctx.report("unsupported", f"{location}!{info.filename}", f"archive member cannot be read: {exc}")
        return None
    _budget(ctx).take(len(out))
    return out


def _latest(infos):
    """A later entry with the same name wins."""
    latest = {}
    for info in infos:
        latest[info.filename] = info
    return latest


def torch_zip_layout(data, infos):
    """(prefix, names) when the zip is laid out as torch.save writes it."""
    if data[:4] != b"PK\x03\x04" or not infos:
        return None
    first = infos[0].filename
    if "/" not in first:
        return None
    prefix = first.split("/", 1)[0]
    names = {i.filename for i in infos}
    if f"{prefix}/data.pkl" not in names:
        return None
    if f"{prefix}/version" not in names and f"{prefix}/.data/version" not in names:
        return None
    return prefix


def scan_zip(ctx, data, location, depth):
    zf = _zip_open(data)
    infos = zf.infolist()
    prefix = torch_zip_layout(data, infos)
    if prefix is not None:
        _scan_torch_zip(ctx, zf, infos, prefix, location)
        return
    latest = _latest(infos)
    if "config.json" in latest and ("metadata.json" in latest or "model.weights.h5" in latest
                                    or "model.weights.npz" in latest):
        from modelgate import keras_scan
        raw = _zip_read(ctx, zf, latest["config.json"], location)
        if raw is not None and keras_scan.looks_like_keras_config(raw):
            keras_scan.scan_keras_config_bytes(ctx, raw, f"{location}!config.json")
            return
    _scan_members(ctx, {n: (lambda i=i: _zip_read(ctx, zf, i, location)) for n, i in latest.items()
                        if not n.endswith("/")}, location, depth)


def _scan_torch_zip(ctx, zf, infos, prefix, location):
    latest = _latest(infos)
    info = latest[f"{prefix}/data.pkl"]
    raw = _zip_read(ctx, zf, info, location)
    if raw is None:
        return None
    storages = {n[len(prefix) + 6:] for n in latest if n.startswith(f"{prefix}/data/")}
    up = Unpickler(ctx, raw, 0, f"{location}!{info.filename}", encoding="utf-8",
                   persistent="torch-zip", storages=storages, torch=True)
    ok, value, _end = up.run()
    if not ok:
        ctx.note(f"{location}!{info.filename}: loading stops: {up.stop_reason}")
    return value


def _scan_members(ctx, members, location, depth):
    """Load each member of an unpacked archive by its own format.

    ``members`` maps a member path to a zero-argument reader.  Directories
    holding a SavedModel are loaded as one SavedModel.
    """
    from modelgate import savedmodel_scan
    sm_dirs = set()
    for n in members:
        base = n.rsplit("/", 1)[-1]
        if base in ("saved_model.pb", "saved_model.pbtxt"):
            sm_dirs.add(n[: -len(base)])
    for d in sorted(sm_dirs):
        def read(rel, d=d):
            r = members.get(d + rel)
            return r() if r is not None else None
        savedmodel_scan.scan_savedmodel(ctx, read, f"{location}!{d or './'}")
    for n, reader in members.items():
        if any(n.startswith(d) for d in sm_dirs):
            continue
        ctx.check_time()
        content = reader()
        if content is None or not content:
            continue
        scan_blob(ctx, content, f"{location}!{n}", n, depth + 1)


# -------------------------------------------------------------------- tar
def scan_tar(ctx, data, location, depth):
    tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:")
    latest = {}
    try:
        for m in tf:
            ctx.check_time()
            name = m.name
            while name.startswith("./"):
                name = name[2:]
            if m.isreg() or m.islnk():
                latest[name] = m
            elif name in latest:
                del latest[name]
    except tarfile.TarError as exc:
        if not latest:
            raise
        ctx.note(f"{location}: tar archive ends early: {exc}")

    def reader(m):
        def read():
            if m.size > MAX_MEMBER:
                raise ScanAbort(f"{location}!{m.name}: member too large")
            try:
                f = tf.extractfile(m)
                out = f.read() if f is not None else None
            except (tarfile.TarError, OSError, KeyError) as exc:
                ctx.report("unsupported", f"{location}!{m.name}", f"archive member cannot be read: {exc}")
                return None
            if out is not None:
                _budget(ctx).take(len(out))
            return out
        return read

    _scan_members(ctx, {n: reader(m) for n, m in latest.items()}, location, depth)


# -------------------------------------------------------------------- npy
def _descr_has_object(descr, depth=0):
    if depth > 32:
        return True
    if isinstance(descr, str):
        return descr.lstrip("<>|=!").startswith("O")
    if isinstance(descr, (list, tuple)):
        for field in descr:
            if isinstance(field, (list, tuple)) and len(field) >= 2:
                if _descr_has_object(field[1], depth + 1):
                    return True
            elif isinstance(field, str):
                continue
        return False
    return False


def npy_header(data):
    """(offset of the payload, header dict) or None when numpy would refuse it."""
    if data[:6] != NPY_MAGIC or len(data) < 10:
        return None
    major = data[6]
    if major == 1:
        hlen = struct.unpack("<H", data[8:10])[0]
        start = 10
    elif major in (2, 3):
        if len(data) < 12:
            return None
        hlen = struct.unpack("<I", data[8:12])[0]
        start = 12
    else:
        return None
    raw = data[start:start + hlen]
    if len(raw) < hlen:
        return None
    try:
        text = raw.decode("utf-8" if major == 3 else "latin1")
        d = ast.literal_eval(text)
    except (ValueError, SyntaxError, UnicodeDecodeError, MemoryError, RecursionError, TypeError):
        return None
    if not isinstance(d, dict) or set(d) != {"descr", "fortran_order", "shape"}:
        return None
    if not isinstance(d["shape"], tuple) or not all(isinstance(x, int) for x in d["shape"]):
        return None
    if not isinstance(d["fortran_order"], bool):
        return None
    return start + hlen, d


def scan_npy(ctx, data, location, allow_pickle=True):
    """numpy.load / read_array: an object array is one pickle after the header."""
    hdr = npy_header(data)
    if hdr is None:
        ctx.note(f"{location}: numpy would refuse this header")
        return Opaque("array")
    start, d = hdr
    hasobject = _descr_has_object(d["descr"])
    try:
        from numpy.lib import format as npformat  # numpy's own descr parser
        hasobject = bool(npformat.descr_to_dtype(d["descr"]).hasobject)
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - numpy refuses the descr, so it loads nothing
        ctx.note(f"{location}: numpy rejects dtype {d['descr']!r}")
        return Opaque("array")
    if not hasobject:
        return Opaque("numpy array")
    if not allow_pickle:
        raise PickleStop("Object arrays cannot be loaded when allow_pickle=False")
    up = Unpickler(ctx, data, start, f"{location}@pickle", encoding="ASCII")
    ok, value, _end = up.run()
    if not ok:
        ctx.note(f"{location}: object array pickle stops: {up.stop_reason}")
        raise PickleStop(f"object array failed to load: {up.stop_reason}")
    return value


# ---------------------------------------------------------- nested loaders
def torch_load_bytes(ctx, data, site):
    if data[:4] == b"PK\x03\x04":
        zf = _zip_open(data)
        infos = zf.infolist()
        prefix = torch_zip_layout(data, infos)
        if prefix is None:
            raise PickleStop("torch.load: not a valid checkpoint archive")
        return _scan_torch_zip(ctx, zf, infos, prefix, site) or Opaque("torch.load result")
    if not data:
        raise PickleStop("torch.load: empty input")
    return run_torch_legacy(ctx, data, site)


def numpy_load_bytes(ctx, data, allow_pickle, npz_ok, site):
    if data[:6] == NPY_MAGIC:
        return scan_npy(ctx, data, site, allow_pickle=allow_pickle)
    if npz_ok and data[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        zf = _zip_open(data)
        members = {}
        for n, info in _latest(zf.infolist()).items():
            raw = _zip_read(ctx, zf, info, site)
            if raw is not None:
                members[n] = raw
        return NpzObj(members, allow_pickle)
    if not allow_pickle:
        raise PickleStop("Cannot load file containing pickled data when allow_pickle=False")
    up = Unpickler(ctx, data, 0, site, encoding="ASCII")
    ok, value, _end = up.run()
    if not ok:
        raise PickleStop(f"numpy.load pickle failed: {up.stop_reason}")
    return value


def npz_member(ctx, npz, key, site):
    if not isinstance(key, str):
        raise PickleStop("npz key must be a string")
    name = key if key in npz.members else key + ".npy" if key + ".npy" in npz.members else None
    if name is None:
        raise PickleStop(f"{key} is not a file in the archive")
    raw = npz.members[name]
    if raw[:6] == NPY_MAGIC:
        return scan_npy(ctx, raw, f"{site}!{name}", allow_pickle=npz.allow_pickle)
    return raw


def joblib_load_bytes(ctx, data, site):
    kind = sniff(data)
    if kind == "gzip":
        data = gunzip(data, ctx)
    elif kind == "bz2":
        data = bunzip2(data, ctx)
    elif kind == "xz":
        data = unxz(data, ctx)
    elif kind == "zlib":
        data = unzlib(data, ctx)
    elif kind == "lzma":
        data = unlzma(data, ctx)
    elif data[:4] == b"\x04\x22\x4d\x18":
        raise ScanAbort("lz4-compressed joblib data cannot be analysed")
    up = Unpickler(ctx, data, 0, site, python=True, encoding="ASCII", joblib=True)
    ok, value, _end = up.run()
    if not ok:
        raise PickleStop(f"joblib.load failed: {up.stop_reason}")
    return value
