"""Capability tables and name resolution for the ModelGate scanner.

The policy (see /app/docs/threat-model.md) is defined by *capabilities*, not
names: an artifact is malicious when loading it, or calling a function it
exports, would (A) execute attacker code, (B) start a process, (C) touch a
file outside the artifact, or (D) open a network connection.  These tables
say which callables / graph ops carry which capability; the format scanners
decide whether the capability is actually *reached* (called), which is what
makes it a finding.
"""
from __future__ import annotations

import _compat_pickle

# --- pickle: module/name aliasing -----------------------------------------
# The unpickler treats several module names as aliases.  posix/nt are the real
# modules os is built on, _pickle/_io back pickle/io; the Python-2 names are
# mapped back by fix_imports (the _compat_pickle tables) for pickles written
# below protocol 3.  We apply them unconditionally: they only ever rewrite an
# old/alias name to its canonical one.
_EXTRA_IMPORT_ALIASES = {
    "posix": "os",
    "nt": "os",
    "_pickle": "pickle",
    "cPickle": "pickle",
    "_io": "io",
}


def resolve(module: str, name: str) -> tuple[str, str]:
    """Map a (module, name) pair to its canonical (module, name)."""
    if (module, name) in _compat_pickle.NAME_MAPPING:
        module, name = _compat_pickle.NAME_MAPPING[(module, name)]
    elif module in _compat_pickle.IMPORT_MAPPING:
        module = _compat_pickle.IMPORT_MAPPING[module]
    # apply alias again after NAME_MAPPING may have changed the module
    module = _EXTRA_IMPORT_ALIASES.get(module, module)
    module = _compat_pickle.IMPORT_MAPPING.get(module, module)
    module = _EXTRA_IMPORT_ALIASES.get(module, module)
    return module, name


# --- pickle: capability-bearing callables ----------------------------------
# name -> (finding-kind, mode).  mode "always": any call is a finding.
# mode "needarg": a finding only when called with at least one positional
# argument (the connect-on-construction clients: FTP(host), SMTP(host)).
CODE = "code-execution"
PROC = "process"
FILE = "file-access"
NET = "network"

CAPABILITIES: dict[str, tuple[str, str]] = {}


def _add(kind, mode, names):
    for n in names:
        CAPABILITIES[n] = (kind, mode)


# (A) code execution -- evaluate / compile / materialise code
_add(CODE, "always", [
    "builtins.eval", "builtins.exec", "builtins.compile", "builtins.execfile",
    "builtins.input",  # py2 input() == eval(raw_input())
    "marshal.loads", "marshal.load",
    "types.FunctionType", "types.CodeType", "types.LambdaType",
    "runpy.run_path", "runpy.run_module", "runpy._run_code", "runpy._run_module_code",
    "timeit.timeit", "timeit.repeat", "timeit.Timer",
    "code.InteractiveInterpreter.runsource", "code.InteractiveInterpreter.runcode",
    "code.interact",
    "bdb.Bdb.run", "bdb.Bdb.runeval", "bdb.Bdb.runcall",
    "pdb.run", "pdb.runeval", "pdb.runcall", "pdb.runctx", "pdb.main", "pdb.set_trace",
    "numpy.testing._private.utils.runstring",
    "logging.config.dictConfig", "logging.config.fileConfig", "logging.config.listen",
    "cProfile.run", "cProfile.runctx", "profile.run", "profile.runctx",
])

# (B) process
_add(PROC, "always", [
    "os.system", "os.popen", "os.popen2", "os.popen3", "os.popen4",
    "os.execl", "os.execle", "os.execlp", "os.execlpe",
    "os.execv", "os.execve", "os.execvp", "os.execvpe",
    "os.spawnl", "os.spawnle", "os.spawnlp", "os.spawnlpe",
    "os.spawnv", "os.spawnve", "os.spawnvp", "os.spawnvpe",
    "os.posix_spawn", "os.posix_spawnp", "os.startfile",
    "subprocess.Popen", "subprocess.call", "subprocess.run",
    "subprocess.check_call", "subprocess.check_output",
    "subprocess.getoutput", "subprocess.getstatusoutput",
    "pty.spawn", "platform.popen", "pydoc.pipepager", "pydoc.tempfilepager",
    "pydoc.ttypager",
])

# (C) file access  (read / write / create / delete / move a file)
_add(FILE, "always", [
    "builtins.open", "io.open", "io.FileIO", "os.fdopen",
    "os.remove", "os.unlink", "os.rename", "os.renames", "os.replace",
    "os.mkdir", "os.makedirs", "os.rmdir", "os.removedirs", "os.truncate",
    "os.link", "os.symlink", "os.mknod",
    "shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copytree",
    "shutil.move", "shutil.rmtree", "shutil.copyfileobj", "shutil.copymode",
    "shutil.make_archive", "shutil.unpack_archive",
    "pathlib.Path.write_text", "pathlib.Path.write_bytes", "pathlib.Path.read_text",
    "pathlib.Path.read_bytes", "pathlib.Path.open", "pathlib.Path.unlink",
    "pathlib.Path.mkdir", "pathlib.Path.rmdir", "pathlib.Path.rename",
    "pathlib.Path.replace", "pathlib.Path.touch",
    "fileinput.input", "fileinput.FileInput",
    "tempfile.NamedTemporaryFile",
    "dbm.open", "sqlite3.connect",
])

# (D) network
_add(NET, "always", [
    "socket.create_connection", "socket.create_server",
    "socket.socket.connect", "socket.socket.connect_ex",
    "urllib.request.urlopen", "urllib.request.urlretrieve",
    "urllib.request.URLopener.open", "urllib.request.URLopener.retrieve",
    "http.client.HTTPConnection.connect", "http.client.HTTPConnection.request",
    "http.client.HTTPSConnection.connect", "http.client.HTTPSConnection.request",
    "ftplib.FTP.connect", "ftplib.FTP.login",
    "smtplib.SMTP.connect",
    "ssl.get_server_certificate", "webbrowser.open", "webbrowser.open_new",
    "antigravity",
])
_add(NET, "needarg", [
    "ftplib.FTP", "ftplib.FTP_TLS", "smtplib.SMTP", "smtplib.SMTP_SSL",
    "poplib.POP3", "poplib.POP3_SSL", "imaplib.IMAP4", "imaplib.IMAP4_SSL",
    "telnetlib.Telnet", "xmlrpc.client.Transport.request",
])

# Callables that carry no capability of their own but are the *path* by which a
# capability-bearing callable is reached.  The call that reaches the capability
# is the finding, so these only transform the abstract value.
INDIRECTION = {
    "builtins.getattr": "getattr",
    "builtins.__import__": "import",
    "importlib.import_module": "import",
    "operator.attrgetter": "attrgetter",
    "operator.methodcaller": "methodcaller",
    "functools.partial": "partial",
}

# Nested loaders: deserialize embedded data (recurse) or read a path (file).
# kind: "bytes" takes a bytes/buffer and loads it; "pathorbytes" accepts either
# a filesystem path (-> reads that file) or embedded data.
NESTED_LOADERS = {
    "pickle.loads": "bytes",
    "pickle.load": "pathorbytes",
    "numpy.lib.format.read_array": "bytes",
    "torch.storage._load_from_bytes": "bytes",
    "torch.load": "pathorbytes",
    "numpy.load": "pathorbytes",
    "numpy.lib.npyio.load": "pathorbytes",
    "joblib.load": "pathorbytes",
    "joblib.numpy_pickle.load": "pathorbytes",
    "pandas.read_pickle": "pathorbytes",
    "pandas.io.pickle.read_pickle": "pathorbytes",
}

# Pure byte transforms the scanner may evaluate on literal arguments to see
# through obfuscation (base64 / compression) feeding a nested loader.  They
# have no side effects; evaluating them never touches the artifact's behaviour.
BYTE_TRANSFORMS = {
    "zlib.decompress", "bz2.decompress", "lzma.decompress", "gzip.decompress",
    "base64.b64decode", "base64.standard_b64decode", "base64.urlsafe_b64decode",
    "base64.b32decode", "base64.b16decode", "base64.a85decode", "base64.b85decode",
    "base64.decodebytes",
    "binascii.a2b_base64", "binascii.unhexlify", "binascii.a2b_hex",
    "codecs.decode",
    "bytes.fromhex",
}

BYTESIO = {"io.BytesIO", "_io.BytesIO", "io.StringIO", "_io.StringIO"}

# --- TensorFlow op table (closed; section 4 of the threat model) ------------
TF_OPS: dict[str, str] = {}
for _op in ("PyFunc", "PyFuncStateless", "EagerPyFunc"):
    TF_OPS[_op] = CODE
for _op in (
    "ReadFile", "WriteFile", "MatchingFiles", "MatchingFilesDataset",
    "ImmutableConst", "WholeFileReader", "WholeFileReaderV2", "TextLineReader",
    "TextLineReaderV2", "FixedLengthRecordReader", "FixedLengthRecordReaderV2",
    "TFRecordReader", "TFRecordReaderV2", "LMDBReader", "LMDBDataset",
    "ExperimentalLMDBDataset", "TextLineDataset", "TFRecordDataset",
    "FixedLengthRecordDataset", "DatasetToTFRecord", "ExperimentalDatasetToTFRecord",
    "SaveDataset", "SaveDatasetV2", "LoadDataset", "SnapshotDataset",
    "SnapshotDatasetV2", "LoadAndRemapMatrix", "GenerateVocabRemapping",
    "Save", "SaveSlices", "Restore", "RestoreSlice",
):
    TF_OPS[_op] = FILE
for _op in (
    "DataServiceDataset", "DataServiceDatasetV2", "DataServiceDatasetV3",
    "DataServiceDatasetV4", "RegisterDataset", "RegisterDatasetV2", "DistributedSave",
):
    TF_OPS[_op] = NET
