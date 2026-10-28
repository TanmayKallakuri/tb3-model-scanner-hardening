"""What the scanner looks for, expressed as capabilities rather than names.

The registry policy (``/app/docs/threat-model.md``) defines an artifact as
malicious when loading it, or invoking a function it exports, would execute
attacker-controlled code (A), start a process (B), touch a file outside the
artifact (C) or open a network connection (D).  The tables below name the
callables and graph ops that carry those capabilities.  A name is only a
finding when the capability is *executed*; the pickle interpreter in
``pickle_scan`` decides that, this module only says what each callable is.
"""
from __future__ import annotations

import _compat_pickle

# --------------------------------------------------------------------------
# Finding kinds (must match /app/docs/report-schema.md).
CODE = "code-execution"
PROC = "process"
FILE = "file-access"
NET = "network"

# --------------------------------------------------------------------------
# Module-alias resolution.  The unpickler treats several module names as the
# same module; a capability reached through an alias is the same capability.
# `_compat_pickle` carries the Python 2 -> 3 tables the unpickler applies with
# fix_imports; the extras below are genuine current aliases (posix/nt re-export
# os, _pickle is the C pickle, and so on) that hold at every protocol.
_EXTRA_MODULE_ALIASES = {
    "posix": "os",
    "nt": "os",
    "_pickle": "pickle",
    "cPickle": "pickle",
    "_io": "io",
    "cStringIO": "io",
    "StringIO": "io",
    "_socket": "socket",
    "_functools": "functools",
    "_operator": "operator",
    "_collections": "collections",
    "_json": "json",
}


def resolve(module: str, name: str) -> tuple[str, str]:
    """Map (module, name) through the unpickler's alias tables."""
    key = (module, name)
    if key in _compat_pickle.NAME_MAPPING:
        module, name = _compat_pickle.NAME_MAPPING[key]
    elif module in _compat_pickle.IMPORT_MAPPING:
        module = _compat_pickle.IMPORT_MAPPING[module]
    module = _EXTRA_MODULE_ALIASES.get(module, module)
    # A dotted first component can itself be an alias (e.g. ``posix.path``).
    head, _, rest = module.partition(".")
    if rest and head in _EXTRA_MODULE_ALIASES:
        module = _EXTRA_MODULE_ALIASES[head] + "." + rest
    return module, name


# --------------------------------------------------------------------------
# Capability table.  Keys are fully-qualified dotted names (module + qualname)
# after alias resolution.  Value is (kind, trigger):
#   "call"      -> a finding whenever the callable is invoked.
#   "call_args" -> a finding only when invoked with at least one argument
#                  (constructors that connect only when given a host, etc.).
CAPABILITIES: dict[str, tuple[str, str]] = {}


def _add(kind, trigger, names):
    for n in names:
        CAPABILITIES[n] = (kind, trigger)


# (A) code execution ---------------------------------------------------------
_add(CODE, "call", [
    "builtins.eval", "builtins.exec", "builtins.compile", "builtins.execfile",
    "builtins.breakpoint",
    "types.FunctionType", "types.CodeType",
    "marshal.loads",
    "runpy.run_path", "runpy._run_code", "runpy._run_module_code",
    "runpy.run_module", "runpy.run_module_as_main",
    "timeit.timeit", "timeit.repeat", "timeit.Timer",
    "code.InteractiveInterpreter.runsource",
    "code.InteractiveInterpreter.runcode",
    "code.InteractiveConsole.interact", "code.interact",
    "bdb.Bdb.run", "bdb.Bdb.runeval", "bdb.Bdb.runcall",
    "pdb.run", "pdb.runeval", "pdb.runcall", "pdb.Pdb.run",
    "numpy.testing._private.utils.runstring",
    "logging.config.dictConfig", "logging.config.fileConfig",
    "logging.config.listen",
    "cProfile.run", "cProfile.runctx", "profile.run", "profile.runctx",
])

# (B) process ----------------------------------------------------------------
_add(PROC, "call", [
    "os.system", "os.popen", "os.popen2", "os.popen3", "os.popen4",
    "os.posix_spawn", "os.posix_spawnp", "os.startfile",
    "os.execl", "os.execle", "os.execlp", "os.execlpe",
    "os.execv", "os.execve", "os.execvp", "os.execvpe",
    "os.spawnl", "os.spawnle", "os.spawnlp", "os.spawnlpe",
    "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe",
    "subprocess.Popen", "subprocess.call", "subprocess.run",
    "subprocess.check_call", "subprocess.check_output",
    "subprocess.getoutput", "subprocess.getstatusoutput",
    "pty.spawn", "pty.fork",
    "platform.popen",
    "pydoc.pipepager", "pydoc.tempfilepager",
])

# (C) file access ------------------------------------------------------------
_add(FILE, "call", [
    "builtins.open", "io.open", "io.FileIO", "io.open_code",
    "os.open", "os.fdopen",
    "os.remove", "os.unlink", "os.rename", "os.renames", "os.replace",
    "os.mkdir", "os.makedirs", "os.rmdir", "os.removedirs",
    "os.link", "os.symlink", "os.truncate", "os.chmod", "os.chown",
    "os.lchown", "os.chflags", "os.mkfifo", "os.mknod",
    "os.scandir", "os.listdir", "os.walk", "os.readlink",
    "shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copyfileobj",
    "shutil.copytree", "shutil.move", "shutil.rmtree", "shutil.make_archive",
    "shutil.unpack_archive", "shutil.copymode", "shutil.copystat",
    "shutil.chown", "shutil.disk_usage",
    "codecs.open",
    "gzip.open", "gzip.GzipFile", "bz2.open", "bz2.BZ2File",
    "lzma.open", "lzma.LZMAFile",
    "tarfile.open", "tarfile.TarFile", "zipfile.ZipFile", "zipfile.PyZipFile",
    "tempfile.mkstemp", "tempfile.mkdtemp", "tempfile.NamedTemporaryFile",
    "tempfile.TemporaryFile", "tempfile.SpooledTemporaryFile",
    "tempfile.TemporaryDirectory",
    "fileinput.input", "fileinput.FileInput",
    "sqlite3.connect",
    "dbm.open", "shelve.open",
    "numpy.save", "numpy.savez", "numpy.savez_compressed", "numpy.savetxt",
    "numpy.loadtxt", "numpy.genfromtxt", "numpy.fromfile", "numpy.memmap",
    "numpy.fromregex", "numpy.DataSource",
    "pandas.read_csv", "pandas.read_table", "pandas.read_json",
    "pandas.read_parquet", "pandas.read_excel", "pandas.read_hdf",
    "pathlib.Path.open", "pathlib.Path.write_text", "pathlib.Path.write_bytes",
    "pathlib.Path.read_text", "pathlib.Path.read_bytes", "pathlib.Path.unlink",
    "pathlib.Path.mkdir", "pathlib.Path.rmdir", "pathlib.Path.rename",
    "pathlib.Path.replace", "pathlib.Path.touch", "pathlib.Path.symlink_to",
    "pathlib.Path.hardlink_to", "pathlib.Path.chmod", "pathlib.Path.rglob",
    "pathlib.PosixPath.open", "pathlib.PosixPath.write_text",
    "pathlib.PosixPath.write_bytes", "pathlib.PosixPath.read_text",
    "pathlib.PosixPath.read_bytes", "pathlib.PosixPath.unlink",
    "pathlib.PosixPath.mkdir", "pathlib.PosixPath.rename",
    "pathlib.PosixPath.replace", "pathlib.PosixPath.touch",
    "pathlib.WindowsPath.open", "pathlib.WindowsPath.write_text",
    "pathlib.WindowsPath.write_bytes", "pathlib.WindowsPath.unlink",
])

# (D) network ----------------------------------------------------------------
# Callables that connect whenever invoked.
_add(NET, "call", [
    "socket.create_connection",
    "socket.socket.connect", "socket.socket.connect_ex",
    "urllib.request.urlopen", "urllib.request.urlretrieve",
    "urllib.request.URLopener.open", "urllib.request.FancyURLopener.open",
    "http.client.HTTPConnection.connect", "http.client.HTTPConnection.request",
    "http.client.HTTPSConnection.connect", "http.client.HTTPSConnection.request",
    "ftplib.FTP.connect", "smtplib.SMTP.connect",
    "ssl.get_server_certificate",
    "webbrowser.open", "webbrowser.open_new", "webbrowser.open_new_tab",
    "requests.get", "requests.post", "requests.put", "requests.delete",
    "requests.head", "requests.patch", "requests.request", "requests.api.get",
    "requests.sessions.Session.request",
])
# Client constructors that connect only when handed a host argument.
_add(NET, "call_args", [
    "ftplib.FTP", "ftplib.FTP_TLS",
    "smtplib.SMTP", "smtplib.SMTP_SSL", "smtplib.LMTP",
    "poplib.POP3", "poplib.POP3_SSL",
    "imaplib.IMAP4", "imaplib.IMAP4_SSL", "imaplib.IMAP4_stream",
    "nntplib.NNTP", "nntplib.NNTP_SSL",
    "telnetlib.Telnet",
])

# --------------------------------------------------------------------------
# Callables that carry no capability of their own but are the ways a
# capability-bearing callable is reached.  The interpreter models their result
# symbolically; the call that finally reaches the capability is the finding.
INDIRECTION = {
    "builtins.getattr": "getattr",
    "builtins.__import__": "import",
    "importlib.import_module": "import",
    "importlib.__import__": "import",
    "operator.attrgetter": "attrgetter",
    "operator.methodcaller": "methodcaller",
    "functools.partial": "partial",
    "functools.partialmethod": "partial",
}

# --------------------------------------------------------------------------
# Nested loaders: handing embedded data to another loader that would then do
# any of (A)-(D) is itself (A).  A loader given bytes is judged by what those
# bytes would load; a loader given a path reads that path -> (C).  The value is
# the format the embedded bytes are in.
NESTED_LOADERS = {
    "pickle.loads": "pickle",
    "pickle.load": "pickle",
    "pickle.Unpickler.load": "pickle",
    "pandas.read_pickle": "pickle",
    "pandas.io.pickle.read_pickle": "pickle",
    "joblib.load": "pickle",
    "joblib.numpy_pickle.load": "pickle",
    "numpy.load": "numpy",
    "numpy.lib.format.read_array": "numpy",
    "numpy.lib.npyio.load": "numpy",
    "torch.load": "torch",
    "torch.serialization.load": "torch",
    "torch.storage._load_from_bytes": "torch",
}

# --------------------------------------------------------------------------
# TensorFlow graph ops (section 4 of the threat model is a closed table).
TF_OPS: dict[str, str] = {}
for _op in ("PyFunc", "PyFuncStateless", "EagerPyFunc"):
    TF_OPS[_op] = CODE
for _op in (
    "ReadFile", "WriteFile", "MatchingFiles", "MatchingFilesDataset",
    "ImmutableConst", "WholeFileReader", "WholeFileReaderV2",
    "TextLineReader", "TextLineReaderV2", "FixedLengthRecordReader",
    "FixedLengthRecordReaderV2", "TFRecordReader", "TFRecordReaderV2",
    "LMDBReader", "LMDBDataset", "ExperimentalLMDBDataset", "TextLineDataset",
    "TFRecordDataset", "FixedLengthRecordDataset", "DatasetToTFRecord",
    "ExperimentalDatasetToTFRecord", "SaveDataset", "SaveDatasetV2",
    "LoadDataset", "SnapshotDataset", "SnapshotDatasetV2", "LoadAndRemapMatrix",
    "GenerateVocabRemapping", "Save", "SaveSlices", "Restore", "RestoreSlice",
):
    TF_OPS[_op] = FILE
for _op in (
    "DataServiceDataset", "DataServiceDatasetV2", "DataServiceDatasetV3",
    "DataServiceDatasetV4", "RegisterDataset", "RegisterDatasetV2",
    "DistributedSave",
):
    TF_OPS[_op] = NET
