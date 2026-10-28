"""Capability policy tables for the ModelGate scanner.

The verdict is driven by *capability*, not by names appearing as data.  A
pickle global, a TensorFlow op or a Keras layer is a finding only when loading
the artifact with its standard loader would actually execute the capability it
carries (see /app/docs/threat-model.md).  These tables name the capabilities;
the scanners decide whether they are reached and executed.

Module names here are already in their canonical (Python 3) form; the pickle
scanner normalises aliases (``posix``/``nt`` -> ``os``, ``__builtin__`` ->
``builtins`` and the rest of the Python 2 compatibility tables) before lookup.
"""
from __future__ import annotations

import _compat_pickle

# ---------------------------------------------------------------------------
# Module alias resolution (pickle)
# ---------------------------------------------------------------------------
# Python 2 module names the unpickler rewrites with ``fix_imports`` plus the
# aliases that denote the very same module object in CPython.
MODULE_ALIASES = dict(_compat_pickle.IMPORT_MAPPING)
MODULE_ALIASES.update({
    "posix": "os",
    "nt": "os",
    "_pickle": "pickle",
    "cPickle": "pickle",
    "_io": "io",
    "_socket": "socket",
    "_functools": "functools",
    "_operator": "operator",
    "_collections": "collections",
    "_json": "json",
    "_thread": "threading",
})
# (module, name) pairs the unpickler rewrites wholesale (Python 2 -> 3).
NAME_ALIASES = dict(_compat_pickle.NAME_MAPPING)

# Finding kinds
CODE = "code-execution"
PROC = "process"
FILE = "file-access"
NET = "network"

# ---------------------------------------------------------------------------
# Direct capability-bearing callables: full dotted name -> finding kind.
# A call that reaches one of these (REDUCE / INST / OBJ with args / a partial
# fired / a getattr chain resolved and then called) is the finding.
# ---------------------------------------------------------------------------
CAPABILITIES: dict[str, str] = {}

def _add(kind, names):
    for n in names:
        CAPABILITIES[n] = kind

_add(CODE, [
    "builtins.eval", "builtins.exec", "builtins.compile",
    "builtins.execfile",
    "marshal.loads", "marshal.load",
    "types.FunctionType", "types.CodeType",
    "runpy.run_path", "runpy.run_module", "runpy._run_code",
    "runpy._run_module_code",
    "timeit.timeit", "timeit.repeat", "timeit.Timer",
    "code.InteractiveInterpreter.runsource",
    "code.InteractiveInterpreter.runcode",
    "code.InteractiveConsole.interact", "code.interact",
    "bdb.Bdb.run", "bdb.Bdb.runeval", "bdb.Bdb.runcall", "bdb.Bdb.runctx",
    "pdb.run", "pdb.runeval", "pdb.runcall", "pdb.runctx", "pdb.Pdb.run",
    "numpy.testing._private.utils.runstring",
    "logging.config.dictConfig", "logging.config.fileConfig",
    "ctypes.CDLL", "ctypes.cdll.LoadLibrary", "ctypes.PyDLL",
    "ctypes.WinDLL", "ctypes.OleDLL",
    "cffi.FFI.dlopen",
])

_add(PROC, [
    "os.system", "os.popen", "os.startfile",
    "os.posix_spawn", "os.posix_spawnp",
    "os.execl", "os.execle", "os.execlp", "os.execlpe",
    "os.execv", "os.execve", "os.execvp", "os.execvpe",
    "os.spawnl", "os.spawnle", "os.spawnlp", "os.spawnlpe",
    "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe",
    "os.popen2", "os.popen3", "os.popen4",
    "os.fork", "os.forkpty",
    "subprocess.Popen", "subprocess.call", "subprocess.run",
    "subprocess.check_call", "subprocess.check_output",
    "subprocess.getoutput", "subprocess.getstatusoutput",
    "pty.spawn", "pty.fork",
    "platform.popen",
    "pydoc.pipepager", "pydoc.tempfilepager",
    "multiprocessing.Process.start",
])

_add(FILE, [
    "builtins.open", "io.open", "io.FileIO", "codecs.open",
    "os.remove", "os.unlink", "os.rename", "os.renames", "os.replace",
    "os.mkdir", "os.makedirs", "os.rmdir", "os.removedirs",
    "os.link", "os.symlink", "os.truncate", "os.ftruncate",
    "os.chmod", "os.chown", "os.lchown", "os.open",
    "shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copyfileobj",
    "shutil.copytree", "shutil.move", "shutil.rmtree",
    "shutil.make_archive", "shutil.unpack_archive",
    "pathlib.Path.write_text", "pathlib.Path.write_bytes",
    "pathlib.Path.read_text", "pathlib.Path.read_bytes",
    "pathlib.Path.open", "pathlib.Path.unlink", "pathlib.Path.rename",
    "pathlib.Path.replace", "pathlib.Path.mkdir", "pathlib.Path.rmdir",
    "pathlib.Path.touch",
    "pathlib.PosixPath.write_text", "pathlib.PosixPath.write_bytes",
    "pathlib.PosixPath.open", "pathlib.PosixPath.unlink",
    "pathlib.WindowsPath.write_text", "pathlib.WindowsPath.write_bytes",
    "fileinput.input", "fileinput.FileInput",
    "gzip.open", "gzip.GzipFile", "bz2.open", "bz2.BZ2File",
    "lzma.open", "lzma.LZMAFile",
    "tarfile.open", "zipfile.ZipFile", "dbm.open",
])

_add(NET, [
    "socket.create_connection", "socket.create_server",
    "socket.socket.connect", "socket.socket.connect_ex",
    "urllib.request.urlopen", "urllib.request.urlretrieve",
    "urllib.request.URLopener", "urllib.request.FancyURLopener",
    "urllib.request.URLopener.open",
    "http.client.HTTPConnection.connect", "http.client.HTTPConnection.request",
    "http.client.HTTPSConnection.connect", "http.client.HTTPSConnection.request",
    "ftplib.FTP.connect", "ftplib.FTP.login", "ftplib.FTP_TLS.connect",
    "smtplib.SMTP.connect",
    "ssl.get_server_certificate",
    "webbrowser.open", "webbrowser.open_new", "webbrowser.open_new_tab",
    "webbrowser.get",
    "telnetlib.Telnet.open",
    "requests.get", "requests.post", "requests.put", "requests.head",
    "requests.delete", "requests.patch", "requests.request",
    "requests.api.get", "requests.api.post", "requests.sessions.Session.get",
])

# Nested loaders: re-entrant deserialization.  Judged by what they would load:
# embedded bytes -> recurse; a path argument -> a file read.
NESTED_LOADERS = {
    "pickle.loads", "pickle.load", "pickle.Unpickler",
    "_pickle.loads", "_pickle.load", "_pickle.Unpickler",
    "pandas.read_pickle", "pandas.io.pickle.read_pickle",
    "joblib.load", "joblib.numpy_pickle.load",
    "numpy.load", "numpy.lib.format.read_array",
    "numpy.lib.npyio.load",
    "torch.load", "torch.storage._load_from_bytes",
    "torch.jit.load",
}

# Host-sensitive client constructors: a connection opens only when a host is
# supplied as the first positional argument (``ftplib.FTP(host)`` connects,
# ``ftplib.FTP()`` does not).
HOST_CONSTRUCTORS = {
    "ftplib.FTP": NET, "ftplib.FTP_TLS": NET,
    "smtplib.SMTP": NET, "smtplib.SMTP_SSL": NET, "smtplib.LMTP": NET,
    "poplib.POP3": NET, "poplib.POP3_SSL": NET,
    "imaplib.IMAP4": NET, "imaplib.IMAP4_SSL": NET,
    "telnetlib.Telnet": NET,
    "nntplib.NNTP": NET, "nntplib.NNTP_SSL": NET,
}

# Reach mechanisms: carry no capability of their own; they are how a
# capability-bearing callable is reached.  Mapped to a handler tag.
REACH = {
    "builtins.getattr": "getattr",
    "builtins.__import__": "import",
    "importlib.import_module": "import",
    "importlib.__import__": "import",
    "functools.partial": "partial",
    "functools.reduce": "reduce",
    "builtins.apply": "apply",
    "operator.call": "apply",
    "operator.attrgetter": "attrgetter",
    "operator.methodcaller": "methodcaller",
}

# Classes worth tracking as typed instances, so a dangerous *method* invoked on
# the instance later (via a bound-method global, getattr or methodcaller) is
# caught.  full class name -> instance tag.
TYPED_INSTANCES = {
    "socket.socket": "socket",
    "http.client.HTTPConnection": "httpconn",
    "http.client.HTTPSConnection": "httpconn",
    "ftplib.FTP": "ftp", "ftplib.FTP_TLS": "ftp",
    "smtplib.SMTP": "smtp", "smtplib.SMTP_SSL": "smtp",
}

# tag -> {method name -> kind}
INSTANCE_METHODS = {
    "socket": {"connect": NET, "connect_ex": NET},
    "httpconn": {"connect": NET, "request": NET},
    "ftp": {"connect": NET, "login": NET, "retrbinary": NET, "storbinary": NET},
    "smtp": {"connect": NET, "sendmail": NET, "login": NET},
}

# Method names specific enough to flag when invoked through methodcaller/getattr
# on a base that resolves to a known module.
DANGER_METHODS = {
    "system": PROC, "popen": PROC, "Popen": PROC,
    "connect": NET, "request": NET, "urlopen": NET,
}

# ---------------------------------------------------------------------------
# TensorFlow op table (closed; see threat-model section 4).
# ---------------------------------------------------------------------------
TF_OPS = {}
for _op in ["PyFunc", "PyFuncStateless", "EagerPyFunc"]:
    TF_OPS[_op] = CODE
for _op in ["ReadFile", "WriteFile", "MatchingFiles", "MatchingFilesDataset",
            "ImmutableConst", "WholeFileReader", "WholeFileReaderV2",
            "TextLineReader", "TextLineReaderV2", "FixedLengthRecordReader",
            "FixedLengthRecordReaderV2", "TFRecordReader", "TFRecordReaderV2",
            "LMDBReader", "LMDBDataset", "ExperimentalLMDBDataset",
            "TextLineDataset", "TFRecordDataset", "FixedLengthRecordDataset",
            "DatasetToTFRecord", "ExperimentalDatasetToTFRecord",
            "SaveDataset", "SaveDatasetV2", "LoadDataset", "SnapshotDataset",
            "SnapshotDatasetV2", "LoadAndRemapMatrix", "GenerateVocabRemapping",
            "Save", "SaveSlices", "Restore", "RestoreSlice"]:
    TF_OPS[_op] = FILE
for _op in ["DataServiceDataset", "DataServiceDatasetV2", "DataServiceDatasetV3",
            "DataServiceDatasetV4", "RegisterDataset", "RegisterDatasetV2",
            "DistributedSave"]:
    TF_OPS[_op] = NET


def resolve_alias(module: str, name: str) -> tuple[str, str]:
    """Normalise a (module, name) global to canonical Python 3 names."""
    if (module, name) in NAME_ALIASES:
        module, name = NAME_ALIASES[(module, name)]
    module = MODULE_ALIASES.get(module, module)
    return module, name
