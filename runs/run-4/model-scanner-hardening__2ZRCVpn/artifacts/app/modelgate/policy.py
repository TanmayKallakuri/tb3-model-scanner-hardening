"""Capability tables and name resolution for the ModelGate scanner.

The policy (see /app/docs/threat-model.md) is capability-based: a reference to
a callable is a *finding* only when that callable is actually invoked in a way
that would execute attacker code, start a process, touch a file outside the
artifact, or open a network connection.  This module holds the tables that say
which fully-resolved dotted names carry which capability, which are nested
loaders, and which are the harmless "ways a capability is reached" (getattr,
partial, attrgetter, ...).  The pickle interpreter in ``pickle_scan`` resolves
names to dotted paths and consults ``classify`` here.
"""
from __future__ import annotations

import _compat_pickle

# ---------------------------------------------------------------------------
# Module aliasing.  The CPython unpickler maps Python-2 names back to their
# Python-3 spelling (the _compat_pickle tables) and several modules are simply
# different names for the same thing.
# ---------------------------------------------------------------------------

# Extra single-segment module aliases beyond the _compat_pickle import table.
_MODULE_ALIASES = {
    "posix": "os",
    "nt": "os",
    "_pickle": "pickle",
    "cPickle": "pickle",
    "_io": "io",
    "_codecs": "codecs",
    "_socket": "socket",
    "_ssl": "ssl",
    "_functools": "functools",
    "_operator": "operator",
    "_collections": "collections",
    "__builtin__": "builtins",
    "copy_reg": "copyreg",
}

# Attributes that, read off *any* object, yield a module of the same name
# (``shutil.os`` is ``os``; a module reached through another module's
# attribute).  Reaching the module is harmless; only a later capability call
# on it is a finding.
_SUBMODULES = {
    "os", "posix", "nt", "subprocess", "sys", "socket", "shutil",
    "builtins", "importlib", "base64", "codecs", "zlib", "bz2", "lzma",
    "marshal", "pickle", "functools", "operator", "io", "urllib", "http",
    "runpy", "pty", "webbrowser", "ssl", "ftplib", "smtplib", "requests",
}


def _apply_compat(module: str, name: str) -> tuple[str, str]:
    """Replicate Unpickler.find_class's Python-2 compatibility remapping."""
    if (module, name) in _compat_pickle.NAME_MAPPING:
        module, name = _compat_pickle.NAME_MAPPING[(module, name)]
    elif module in _compat_pickle.IMPORT_MAPPING:
        module = _compat_pickle.IMPORT_MAPPING[module]
    return module, name


def _alias_segment(seg: str) -> str:
    if seg in _MODULE_ALIASES:
        return _MODULE_ALIASES[seg]
    if seg in _compat_pickle.IMPORT_MAPPING:
        return _compat_pickle.IMPORT_MAPPING[seg]
    return seg


def normalize_path(path: str) -> str:
    """Normalise a dotted path's leading module segment(s)."""
    segs = path.split(".")
    if not segs:
        return path
    # A two-segment Python-2 name rename (e.g. __builtin__.reduce).
    if len(segs) >= 2 and (segs[0], segs[1]) in _compat_pickle.NAME_MAPPING:
        new_mod, new_name = _compat_pickle.NAME_MAPPING[(segs[0], segs[1])]
        segs = new_mod.split(".") + new_name.split(".") + segs[2:]
    segs[0] = _alias_segment(segs[0])
    # urllib2 etc. can alias to a dotted module (urllib.request); re-split.
    return ".".join(segs)


def apply_compat(module: str, name: str) -> tuple[str, str]:
    return _apply_compat(module, name)


def resolve_global(module: str, name: str) -> str:
    """Return the normalised dotted path for a GLOBAL/STACK_GLOBAL/INST."""
    module, name = _apply_compat(module, name)
    return normalize_path(f"{module}.{name}" if name else module)


def is_submodule_attr(name: str) -> bool:
    return name in _SUBMODULES


# ---------------------------------------------------------------------------
# Capability tables.  Keys are fully-resolved dotted paths.  Value is
# (finding-kind, arg-policy); arg-policy "always" fires on any invocation,
# "if_args" only when the call is given at least one positional argument
# (client constructors that connect only when handed a host).
# ---------------------------------------------------------------------------

_A = "code-execution"
_B = "process"
_C = "file-access"
_D = "network"

_ALWAYS = "always"
_IF_ARGS = "if_args"


def _entries(kind, argpolicy, names):
    return {n: (kind, argpolicy) for n in names}


PICKLE_CAPS: dict[str, tuple[str, str]] = {}

PICKLE_CAPS.update(_entries(_A, _ALWAYS, [
    "builtins.eval", "builtins.exec", "builtins.execfile", "builtins.compile",
    "marshal.loads",
    "runpy.run_path", "runpy.run_module", "runpy._run_code",
    "runpy._run_module_code",
    "timeit.timeit", "timeit.repeat",
    "code.InteractiveInterpreter.runsource",
    "code.InteractiveInterpreter.runcode", "code.interact",
    "bdb.Bdb.run", "bdb.Bdb.runeval", "bdb.Bdb.runcall",
    "pdb.run", "pdb.runeval", "pdb.runcall", "pdb.Pdb.run",
    "cProfile.run", "profile.run",
    "numpy.testing._private.utils.runstring",
    "logging.config.dictConfig", "logging.config.fileConfig",
]))

PICKLE_CAPS.update(_entries(_B, _ALWAYS, [
    "os.system", "os.popen", "os.startfile", "os.fork",
    "os.execl", "os.execle", "os.execlp", "os.execlpe",
    "os.execv", "os.execve", "os.execvp", "os.execvpe",
    "os.spawnl", "os.spawnle", "os.spawnlp", "os.spawnlpe",
    "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe",
    "os.posix_spawn", "os.posix_spawnp",
    "subprocess.call", "subprocess.run",
    "subprocess.check_call", "subprocess.check_output",
    "subprocess.getoutput", "subprocess.getstatusoutput",
    "pty.spawn", "pty.fork",
    "pydoc.pipepager", "pydoc.tempfilepager",
]))

# Constructor capabilities: the process/connection/file access happens in
# __init__, and materialising a function needs its code argument, so a
# zero-argument construction (or a NEWOBJ / argument-less INST that runs only
# __new__) does nothing and is not a finding.
PICKLE_CAPS.update(_entries(_B, _IF_ARGS, ["subprocess.Popen"]))
PICKLE_CAPS.update(_entries(_A, _IF_ARGS, ["types.FunctionType"]))
PICKLE_CAPS.update(_entries(_C, _IF_ARGS, ["io.FileIO", "fileinput.FileInput"]))

PICKLE_CAPS.update(_entries(_C, _ALWAYS, [
    "builtins.open", "io.open", "os.open", "os.fdopen",
    "os.remove", "os.unlink", "os.rename", "os.renames", "os.replace",
    "os.mkdir", "os.makedirs", "os.rmdir", "os.removedirs",
    "os.link", "os.symlink", "os.truncate", "os.ftruncate",
    "shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copyfileobj",
    "shutil.copytree", "shutil.move", "shutil.rmtree",
    "shutil.make_archive", "shutil.unpack_archive",
    "pathlib.Path.open", "pathlib.Path.write_text", "pathlib.Path.write_bytes",
    "pathlib.Path.read_text", "pathlib.Path.read_bytes", "pathlib.Path.unlink",
    "pathlib.Path.mkdir", "pathlib.Path.rename", "pathlib.Path.replace",
    "pathlib.Path.rmdir", "pathlib.Path.touch",
    "codecs.open", "fileinput.input", "fileinput.FileInput",
    "tempfile.mkstemp", "tempfile.mkdtemp", "tempfile.NamedTemporaryFile",
    "tempfile.TemporaryFile",
    "gzip.open", "bz2.open", "lzma.open",
    "numpy.save", "numpy.savez", "numpy.savez_compressed", "numpy.savetxt",
    "numpy.fromfile", "numpy.loadtxt", "numpy.genfromtxt", "numpy.memmap",
]))

PICKLE_CAPS.update(_entries(_D, _ALWAYS, [
    "socket.create_connection", "socket.socket.connect",
    "socket.socket.connect_ex",
    "urllib.request.urlopen", "urllib.request.urlretrieve",
    "urllib.request.URLopener.open", "urllib.request.URLopener.retrieve",
    "urllib.request.OpenerDirector.open",
    "http.client.HTTPConnection.connect", "http.client.HTTPConnection.request",
    "http.client.HTTPSConnection.connect",
    "http.client.HTTPSConnection.request",
    "ftplib.FTP.connect", "ftplib.FTP.login",
    "smtplib.SMTP.connect", "imaplib.IMAP4.open", "poplib.POP3.connect",
    "telnetlib.Telnet.open",
    "ssl.get_server_certificate",
    "webbrowser.open", "webbrowser.open_new", "webbrowser.open_new_tab",
    "requests.get", "requests.post", "requests.put", "requests.delete",
    "requests.head", "requests.patch", "requests.request",
    "requests.sessions.Session.get", "requests.sessions.Session.post",
    "requests.sessions.Session.request",
]))

PICKLE_CAPS.update(_entries(_D, _IF_ARGS, [
    "ftplib.FTP", "ftplib.FTP_TLS",
    "smtplib.SMTP", "smtplib.SMTP_SSL", "smtplib.LMTP",
    "poplib.POP3", "poplib.POP3_SSL",
    "imaplib.IMAP4", "imaplib.IMAP4_SSL",
    "nntplib.NNTP", "nntplib.NNTP_SSL",
    "telnetlib.Telnet",
]))

# ---------------------------------------------------------------------------
# Nested loaders: calling one on embedded data re-enters a loader, which is a
# finding only if what it would load is itself a finding.  On a path argument
# it reads that path (file-access); on opaque data we cannot prove it benign
# so we fail closed.
# ---------------------------------------------------------------------------
NESTED_LOADERS: dict[str, str] = {
    "pickle.loads": "pickle-bytes",
    "pickle.load": "pickle-file",
    "joblib.load": "joblib",
    "joblib.numpy_pickle.load": "joblib",
    "pandas.read_pickle": "pandas",
    "numpy.load": "numpy",
    "numpy.lib.format.read_array": "numpy-array",
    "torch.load": "torch",
    "torch.storage._load_from_bytes": "torch-bytes",
}

# Harmless on their own; they are how a capability is *reached*.  The call that
# finally reaches the capability is the finding.
TRANSFORMS: dict[str, str] = {
    "builtins.getattr": "getattr",
    "builtins.__import__": "import",
    "importlib.import_module": "import",
    "operator.attrgetter": "attrgetter",
    "operator.methodcaller": "methodcaller",
    "functools.partial": "partial",
    "functools.partialmethod": "partial",
    "pkgutil.resolve_name": "resolve_name",
    "pydoc.locate": "resolve_name",
    "copy.copy": "identity",
    "copy.deepcopy": "identity",
    "functools.lru_cache": "wrap",
    "functools.cache": "wrap",
    "functools.reduce": "reduce_fn",
    "builtins.map": "map",
    "builtins.filter": "map",
    "itertools.starmap": "map",
    "operator.call": "call",
}

# Callables that consume an iterator (so a map/filter of a capability over data
# actually runs the capability).
CONSUMERS = {
    "builtins.list", "builtins.tuple", "builtins.set", "builtins.frozenset",
    "builtins.dict", "builtins.sorted", "builtins.sum", "builtins.max",
    "builtins.min", "builtins.any", "builtins.all", "collections.deque",
}

# Side-effect-free data transforms whose concrete result we can compute so that
# embedded-data arguments to nested loaders become visible.
PURE_DATA = {
    "base64.b64decode", "base64.standard_b64decode", "base64.urlsafe_b64decode",
    "base64.b16decode", "base64.b32decode", "base64.b85decode",
    "base64.a85decode", "base64.decodebytes",
    "binascii.a2b_base64", "binascii.unhexlify", "binascii.a2b_hex",
    "zlib.decompress", "gzip.decompress", "bz2.decompress", "lzma.decompress",
    "codecs.encode", "codecs.decode",
    "operator.add", "operator.concat", "operator.mod", "operator.getitem",
    "builtins.bytes", "builtins.bytearray", "builtins.str",
    "builtins.bytes.fromhex", "builtins.bytearray.fromhex",
}

# Constructors returning an in-memory file wrapping bytes (so a nested loader
# reading from them sees embedded data, not a path).
BYTESIO = {"io.BytesIO", "io.StringIO"}

# Sinks that run a callable argument *when called* (executor submit, pool
# apply/map, start_new_thread).  A capability passed here is executed on the
# call; a benign callable is not a finding.
SINKS = {
    "_thread.start_new_thread",
    "concurrent.futures.ThreadPoolExecutor.submit",
    "concurrent.futures.ProcessPoolExecutor.submit",
    "multiprocessing.pool.Pool.apply", "multiprocessing.pool.Pool.apply_async",
    "multiprocessing.pool.Pool.map", "multiprocessing.pool.Pool.map_async",
    "multiprocessing.pool.Pool.imap", "multiprocessing.pool.Pool.starmap",
}

# Thread/process constructors store a target that runs only when the object is
# started (constructing one executes nothing - like an unconsumed iterator).
THREAD_CTORS = {
    "threading.Thread", "threading.Timer",
    "multiprocessing.Process", "multiprocessing.context.Process",
}


def classify(path: str):
    """Return (category, info) for a fully-resolved dotted path."""
    path = normalize_path(path)
    if path in PICKLE_CAPS:
        return "cap", PICKLE_CAPS[path]
    if path in NESTED_LOADERS:
        return "loader", NESTED_LOADERS[path]
    if path in TRANSFORMS:
        return "transform", TRANSFORMS[path]
    if path in CONSUMERS:
        return "consumer", None
    if path in BYTESIO:
        return "bytesio", None
    if path in PURE_DATA:
        return "puredata", None
    return "plain", None


# ---------------------------------------------------------------------------
# TensorFlow op table (closed; see threat-model section 4).  An op not listed
# in any set carries no capability wherever it appears.
# ---------------------------------------------------------------------------
TF_OPS: dict[str, str] = {}
TF_OPS.update({op: _A for op in ["PyFunc", "PyFuncStateless", "EagerPyFunc"]})
TF_OPS.update({op: _C for op in [
    "ReadFile", "WriteFile", "MatchingFiles", "MatchingFilesDataset",
    "ImmutableConst", "WholeFileReader", "WholeFileReaderV2",
    "TextLineReader", "TextLineReaderV2", "FixedLengthRecordReader",
    "FixedLengthRecordReaderV2", "TFRecordReader", "TFRecordReaderV2",
    "LMDBReader", "LMDBDataset", "ExperimentalLMDBDataset", "TextLineDataset",
    "TFRecordDataset", "FixedLengthRecordDataset", "DatasetToTFRecord",
    "ExperimentalDatasetToTFRecord", "SaveDataset", "SaveDatasetV2",
    "LoadDataset", "SnapshotDataset", "SnapshotDatasetV2",
    "LoadAndRemapMatrix", "GenerateVocabRemapping", "Save", "SaveSlices",
    "Restore", "RestoreSlice",
]})
TF_OPS.update({op: _D for op in [
    "DataServiceDataset", "DataServiceDatasetV2", "DataServiceDatasetV3",
    "DataServiceDatasetV4", "RegisterDataset", "RegisterDatasetV2",
    "DistributedSave",
]})
