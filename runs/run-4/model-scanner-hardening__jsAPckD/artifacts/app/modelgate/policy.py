"""What the scanner looks for, as data tables.

The policy (see /app/docs/threat-model.md) is defined by *capabilities*, not
names: an artifact is malicious when loading it, or calling a function it
exports, would (A) execute attacker code, (B) start a process, (C) touch a
file outside the artifact, or (D) open a network connection.

For pickles this module provides the canonical tables the abstract
interpreter in ``pklvm`` consults; for TensorFlow and Keras the op/marker
tables the respective scanners use.
"""
from __future__ import annotations

import _compat_pickle

# Finding kinds (must match report-schema.md).
CODE = "code-execution"
PROC = "process"
FILE = "file-access"
NET = "network"

# --------------------------------------------------------------------------
# Module aliasing.  The unpickler maps Python-2 names back with fix_imports
# (the _compat_pickle tables) and several C modules are the same object as
# their public name.  ``posix``/``nt`` *are* ``os``; ``_io`` *is* ``io``.
# --------------------------------------------------------------------------
MODULE_ALIASES = dict(_compat_pickle.IMPORT_MAPPING)
MODULE_ALIASES.update({
    "posix": "os",
    "nt": "os",
    "ce": "os",
    "_io": "io",
    "_pickle": "pickle",
    "cPickle": "pickle",
    "StringIO": "io",
    "cStringIO": "io",
    "_socket": "socket",
    "_ssl": "ssl",
    "_functools": "functools",
    "_operator": "operator",
    "_collections": "collections",
    "_thread": "_thread",
    "thread": "_thread",
    "dumbdbm": "dbm.dumb",
})

# (module, name) -> (module, name) for specific Python-2 renames.
NAME_ALIASES = dict(_compat_pickle.NAME_MAPPING)

# A handful of "<module>.<attr> is itself a module" facts, used when a dotted
# GLOBAL/getattr chain is walked (``shutil.os`` is ``os``).
SUBMODULE_ALIASES = {
    ("shutil", "os"): "os",
    ("shutil", "sys"): "sys",
    ("os", "path"): "os.path",
    ("numpy", "testing"): "numpy.testing",
}


def alias_module(module: str) -> str:
    seen = set()
    while module in MODULE_ALIASES and module not in seen:
        seen.add(module)
        module = MODULE_ALIASES[module]
    return module


# --------------------------------------------------------------------------
# Capability callables.  A *call* of any of these is a finding of the given
# kind, no matter how the loader reached it.  Keys are canonical dotted paths
# ("os.system", "pathlib.Path.write_text", "socket.socket.connect").  Unbound
# methods appear as the callable when an attacker writes
# ``(socket.socket.connect, (sock, addr))`` in a __reduce__.
# --------------------------------------------------------------------------
_CODE = [
    "builtins.eval", "builtins.exec", "builtins.compile",
    "builtins.execfile", "builtins.breakpoint",
    "runpy.run_path", "runpy.run_module", "runpy._run_code", "runpy._run_module_code",
    "timeit.timeit", "timeit.repeat",
    "code.InteractiveInterpreter.runsource", "code.InteractiveInterpreter.runcode",
    "code.interact",
    "bdb.Bdb.run", "bdb.Bdb.runeval", "bdb.Bdb.runcall",
    "pdb.run", "pdb.runeval", "pdb.runcall", "pdb.Pdb.run",
    "numpy.testing._private.utils.runstring",
    "logging.config.dictConfig", "logging.config.fileConfig",
    "yaml.load", "yaml.unsafe_load", "yaml.full_load",
    "asyncio.base_events.BaseEventLoop.run_in_executor",
]
_PROC = [
    "os.system", "os.popen", "os.popen2", "os.popen3", "os.popen4", "os.startfile",
    "os.execl", "os.execle", "os.execlp", "os.execlpe",
    "os.execv", "os.execve", "os.execvp", "os.execvpe",
    "os.spawnl", "os.spawnle", "os.spawnlp", "os.spawnlpe",
    "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe",
    "os.posix_spawn", "os.posix_spawnp",
    "subprocess.Popen", "subprocess.call", "subprocess.run",
    "subprocess.check_call", "subprocess.check_output",
    "subprocess.getoutput", "subprocess.getstatusoutput",
    "pty.spawn", "pty.fork",
    "pydoc.pipepager", "pydoc.tempfilepager",
    "posix.system",  # survives even without fix_imports on Linux
]
_FILE = [
    "builtins.open", "io.open", "io.FileIO", "os.open", "os.fdopen",
    "os.remove", "os.unlink", "os.rename", "os.renames", "os.replace",
    "os.mkdir", "os.makedirs", "os.rmdir", "os.removedirs", "os.truncate",
    "os.chmod", "os.chown", "os.link", "os.symlink", "os.mknod",
    "shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copyfileobj",
    "shutil.copytree", "shutil.move", "shutil.rmtree",
    "shutil.make_archive", "shutil.unpack_archive",
    "tempfile.NamedTemporaryFile", "tempfile.TemporaryFile",
    "tempfile.mkstemp", "tempfile.mkdtemp", "tempfile.mktemp",
    "fileinput.input", "fileinput.FileInput",
    "codecs.open", "dbm.open", "gzip.open", "bz2.open", "lzma.open",
    "aiofiles.open",
]
_NET = [
    "socket.create_connection", "socket.create_server",
    "socket.socket.connect", "socket.socket.connect_ex", "socket.socket.sendto",
    "urllib.request.urlopen", "urllib.request.urlretrieve",
    "urllib.request.URLopener.open", "urllib.request.URLopener.retrieve",
    "urllib.request.OpenerDirector.open", "urllib.request.FancyURLopener.open",
    "http.client.HTTPConnection.connect", "http.client.HTTPConnection.request",
    "http.client.HTTPSConnection.connect", "http.client.HTTPSConnection.request",
    "ftplib.FTP.connect", "ftplib.FTP.login", "ftplib.FTP_TLS.connect",
    "smtplib.SMTP.connect", "poplib.POP3.user", "imaplib.IMAP4.open",
    "ssl.get_server_certificate", "ssl.SSLContext.wrap_socket",
    "webbrowser.open", "webbrowser.open_new", "webbrowser.open_new_tab",
    "requests.get", "requests.post", "requests.put", "requests.head",
    "requests.request", "requests.api.request",
    "requests.Session.get", "requests.Session.post", "requests.Session.request",
]

CAP: dict[str, str] = {}
for _p in _CODE:
    CAP[_p] = CODE
for _p in _PROC:
    CAP[_p] = PROC
for _p in _FILE:
    CAP[_p] = FILE
for _p in _NET:
    CAP[_p] = NET

# pathlib bound-method capabilities, for every concrete Path flavour.
for _cls in ("Path", "PosixPath", "WindowsPath", "PurePath", "PurePosixPath", "PureWindowsPath"):
    for _m, _k in (
        ("write_text", FILE), ("write_bytes", FILE), ("read_text", FILE),
        ("read_bytes", FILE), ("open", FILE), ("unlink", FILE), ("rename", FILE),
        ("replace", FILE), ("rmdir", FILE), ("mkdir", FILE), ("touch", FILE),
        ("chmod", FILE), ("symlink_to", FILE), ("hardlink_to", FILE),
    ):
        CAP[f"pathlib.{_cls}.{_m}"] = _k

# --------------------------------------------------------------------------
# Network *constructors* that connect only when given a host argument.
# ``ftplib.FTP()`` opens nothing; ``ftplib.FTP("host")`` connects.
# --------------------------------------------------------------------------
NETCTOR = {
    "ftplib.FTP", "ftplib.FTP_TLS",
    "smtplib.SMTP", "smtplib.SMTP_SSL", "smtplib.LMTP",
    "poplib.POP3", "poplib.POP3_SSL",
    "imaplib.IMAP4", "imaplib.IMAP4_SSL",
    "telnetlib.Telnet",
    "nntplib.NNTP", "nntplib.NNTP_SSL",
}

# --------------------------------------------------------------------------
# Nested loaders: calling one is a finding only when what it would load is a
# finding.  Judged by the argument -- embedded bytes are analysed, a path is
# a file read, an unknown argument fails closed.
# --------------------------------------------------------------------------
LOADER_PICKLE = {
    "pickle.loads", "pickle.load", "_pickle.loads", "_pickle.load",
    "pickle.Unpickler.load", "dill.loads", "dill.load",
    "cloudpickle.loads", "cloudpickle.load",
    "joblib.load", "joblib.numpy_pickle.load",
    "pandas.read_pickle", "pandas.io.pickle.read_pickle",
}
LOADER_NUMPY = {
    "numpy.load", "numpy.lib.format.read_array",
    "numpy.lib.npyio.load", "numpy.lib._npyio_impl.read_array",
}
LOADER_TORCH = {
    "torch.load", "torch.storage._load_from_bytes", "torch.jit.load",
    "torch.serialization.load",
}
LOADER_MARSHAL = {"marshal.loads", "marshal.load"}

ALL_LOADERS = LOADER_PICKLE | LOADER_NUMPY | LOADER_TORCH | LOADER_MARSHAL

# --------------------------------------------------------------------------
# Enablers: carry no capability of their own, but are the route to one.
# --------------------------------------------------------------------------
ENABLERS = {
    "builtins.getattr": "getattr",
    "builtins.setattr": "setattr",
    "operator.attrgetter": "attrgetter",
    "operator.methodcaller": "methodcaller",
    "functools.partial": "partial",
    "functools.partialmethod": "partial",
    "builtins.__import__": "import",
    "importlib.import_module": "import",
    "importlib.__import__": "import",
    "importlib._bootstrap._gcd_import": "import",
    "pkgutil.resolve_name": "resolve_name",
    "operator.call": "call",
    "builtins.apply": "call",
    "operator.getitem": "getitem",
    "operator.__getitem__": "getitem",
    "builtins.vars": "vars",
    "builtins.getitem": "getitem",
    "functools.reduce": "reduce",
    "builtins.map": "map",
    "builtins.filter": "filter",
    "itertools.starmap": "map",
    "functools.lru_cache": "identity",
    "functools.cache": "identity",
    "copy.copy": "identity",
    "copy.deepcopy": "identity",
}

# Callables that consume a lazy map/filter (and so run its function).
CONSUMERS = {
    "builtins.list", "builtins.tuple", "builtins.set", "builtins.frozenset",
    "builtins.dict", "builtins.sorted", "builtins.min", "builtins.max",
    "builtins.sum", "builtins.any", "builtins.all", "builtins.next",
    "builtins.str.join", "collections.deque", "functools.reduce",
    "itertools.chain", "builtins.list.extend",
}

# Constant-folding transforms, so embedded payloads assembled before a nested
# load are judged after assembly (base64/zlib/hex/concatenation/bytearray).
TRANSFORMS = {
    "base64.b64decode", "base64.standard_b64decode", "base64.urlsafe_b64decode",
    "base64.b16decode", "base64.b32decode", "base64.b85decode", "base64.a85decode",
    "base64.decodebytes",
    "binascii.unhexlify", "binascii.a2b_base64", "binascii.a2b_hex",
    "zlib.decompress", "gzip.decompress", "bz2.decompress", "lzma.decompress",
    "codecs.decode", "codecs.encode", "_codecs.decode", "_codecs.encode",
    "builtins.bytes", "builtins.bytearray", "builtins.bytes.fromhex",
    "operator.add", "operator.concat", "operator.iadd",
    "builtins.str", "builtins.bytes.decode", "builtins.str.encode",
}

# --------------------------------------------------------------------------
# TensorFlow op table (closed; see threat-model section 4).
# --------------------------------------------------------------------------
TF_OPS = {}
for _op in ("PyFunc", "PyFuncStateless", "EagerPyFunc"):
    TF_OPS[_op] = CODE
for _op in (
    "ReadFile", "WriteFile", "MatchingFiles", "MatchingFilesDataset",
    "ImmutableConst", "WholeFileReader", "WholeFileReaderV2",
    "TextLineReader", "TextLineReaderV2", "FixedLengthRecordReader",
    "FixedLengthRecordReaderV2", "TFRecordReader", "TFRecordReaderV2",
    "LMDBReader", "LMDBDataset", "ExperimentalLMDBDataset",
    "TextLineDataset", "TFRecordDataset", "FixedLengthRecordDataset",
    "DatasetToTFRecord", "ExperimentalDatasetToTFRecord",
    "SaveDataset", "SaveDatasetV2", "LoadDataset",
    "SnapshotDataset", "SnapshotDatasetV2",
    "LoadAndRemapMatrix", "GenerateVocabRemapping",
    "Save", "SaveSlices", "Restore", "RestoreSlice",
):
    TF_OPS[_op] = FILE
for _op in (
    "DataServiceDataset", "DataServiceDatasetV2", "DataServiceDatasetV3",
    "DataServiceDatasetV4", "RegisterDataset", "RegisterDatasetV2",
    "DistributedSave",
):
    TF_OPS[_op] = NET

# Node attributes that reference a function in the graph's function library.
TF_FUNC_ATTR_OPS = {
    "If", "StatelessIf", "While", "StatelessWhile", "Case",
    "PartitionedCall", "StatefulPartitionedCall", "MapDefun", "For",
}
