"""Capability tables and name normalisation for ModelGate.

The registry policy (/app/docs/threat-model.md) is defined in terms of
*capabilities*, not names: an artifact is malicious only when loading it would
actually execute attacker code, start a process, touch an outside file, or open
a network connection.  So the pickle scanner never matches on strings that are
mere data; it tracks what the unpickler would actually call, and consults the
tables here to decide whether that call is a capability.

Everything in this module is pure data plus a couple of pure helpers; no
artifact-controlled name is ever imported or executed.
"""
from __future__ import annotations

import _compat_pickle

# ---------------------------------------------------------------------------
# Capability callables.  Keyed by canonical "module.qualname".  A finding is
# raised only when one of these is *called* by the unpickler, never when it is
# merely referenced, stored, or imported.
#
# These lists are deliberately a superset of real capabilities: a legitimate
# model never calls open()/system()/urlopen() while being unpickled, so listing
# extra sinks cannot cause a false positive.  False positives come from the
# opposite mistake -- flagging a benign callable -- so benign constructors
# (numpy, sklearn, torch rebuilders, collections, operator.itemgetter, os.fspath
# ...) are deliberately absent.
# ---------------------------------------------------------------------------

CODE_EXEC = {
    "builtins.eval", "builtins.exec", "builtins.compile", "builtins.execfile",
    "builtins.__build_class__",
    # Materialising serialized bytecode into a function is (A).  The function
    # type resolves to builtins.function; keep types.FunctionType for the
    # no-introspection fallback path too.
    "types.FunctionType", "builtins.function",
    "runpy.run_path", "runpy.run_module", "runpy._run_code", "runpy._run_module_code",
    "timeit.timeit", "timeit.repeat",
    "code.InteractiveInterpreter.runsource", "code.InteractiveInterpreter.runcode",
    "code.InteractiveConsole.interact", "code.interact",
    "bdb.Bdb.run", "bdb.Bdb.runeval", "bdb.Bdb.runcall",
    "pdb.run", "pdb.runeval", "pdb.runcall", "pdb.Pdb.run",
    "numpy.testing._private.utils.runstring",
    "logging.config.dictConfig", "logging.config.fileConfig", "logging.config.listen",
    "cProfile.run", "profile.run", "cProfile.runctx", "profile.runctx",
    "ast.literal_eval",  # harmless itself but keep out of benign; not flagged (see note below)
}
# ast.literal_eval is NOT a capability (it evaluates only literals); drop it.
CODE_EXEC.discard("ast.literal_eval")

PROCESS = {
    "os.system", "os.popen", "os.popen2", "os.popen3", "os.popen4",
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
}

FILE_ACCESS = {
    "builtins.open", "io.open", "io.FileIO", "io.open_code", "codecs.open",
    "os.remove", "os.unlink", "os.rename", "os.renames", "os.replace",
    "os.mkdir", "os.makedirs", "os.rmdir", "os.removedirs",
    "os.link", "os.symlink", "os.truncate", "os.open",
    "shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copytree",
    "shutil.copyfileobj", "shutil.move", "shutil.rmtree", "shutil.make_archive",
    "shutil.unpack_archive",
    "pathlib.Path.open", "pathlib.Path.write_text", "pathlib.Path.write_bytes",
    "pathlib.Path.read_text", "pathlib.Path.read_bytes", "pathlib.Path.unlink",
    "pathlib.Path.rename", "pathlib.Path.replace", "pathlib.Path.mkdir",
    "pathlib.Path.rmdir", "pathlib.Path.touch",
    "fileinput.input", "fileinput.FileInput",
    "gzip.open", "gzip.GzipFile", "bz2.open", "bz2.BZ2File",
    "lzma.open", "lzma.LZMAFile", "tarfile.open", "zipfile.ZipFile",
    "dbm.open", "shelve.open",
}

NETWORK = {
    "socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex",
    "urllib.request.urlopen", "urllib.request.urlretrieve",
    "urllib.request.URLopener.open", "urllib.request.OpenerDirector.open",
    "http.client.HTTPConnection.connect", "http.client.HTTPConnection.request",
    "http.client.HTTPSConnection.connect", "http.client.HTTPSConnection.request",
    "ftplib.FTP.connect",
    "ssl.get_server_certificate", "ssl.SSLContext.wrap_socket",
    "webbrowser.open", "webbrowser.open_new", "webbrowser.open_new_tab",
    "antigravity",  # placeholder removed below
}
NETWORK.discard("antigravity")

# Network client constructors that connect only when given a host argument.
# Constructed with no host they open nothing (threat model section 1 (D)).
HOST_SENSITIVE = {
    "ftplib.FTP", "ftplib.FTP_TLS",
    "smtplib.SMTP", "smtplib.SMTP_SSL", "smtplib.LMTP",
    "poplib.POP3", "poplib.POP3_SSL",
    "imaplib.IMAP4", "imaplib.IMAP4_SSL",
    "nntplib.NNTP", "nntplib.NNTP_SSL",
    "telnetlib.Telnet",
}

# Re-entrant deserialisation: a call hands embedded data (or a path) to another
# loader that itself performs (A)-(D).  The nested payload is scanned; a nested
# loader given a path reads that path (file access).
NESTED_LOADERS = {
    "pickle.loads", "pickle.load", "_pickle.loads", "_pickle.load",
    "pickle.Unpickler.load", "_pickle.Unpickler.load",
    "torch.load", "torch.storage._load_from_bytes",
    "numpy.load", "numpy.lib.format.read_array", "numpy.lib.npyio.load",
    "joblib.load", "joblib.numpy_pickle.load",
    "pandas.read_pickle", "pandas.io.pickle.read_pickle",
    "dill.loads", "dill.load",
}

# Routing callables: carry no capability of their own, but are the ways a
# capability is reached.  Tracked by role so the scanner can see through them.
GETATTR = {"builtins.getattr"}
ATTRGETTER = {"operator.attrgetter"}
METHODCALLER = {"operator.methodcaller"}
PARTIAL = {"functools.partial", "functools.partialmethod"}
IMPORTERS = {"builtins.__import__", "importlib.import_module", "importlib.__import__"}
RESOLVE_NAME = {"pkgutil.resolve_name"}
CALLERS = {"operator.call", "builtins.apply"}          # call(f, *a) -> f(*a)
REDUCER = {"functools.reduce", "_functools.reduce"}    # reduce(f, seq) calls f
MAPPERS = {"builtins.map", "builtins.filter", "itertools.starmap"}
KEYED = {"builtins.sorted", "builtins.min", "builtins.max"}  # key=f calls f
# Consumers force a lazy map/filter to actually call its function.
CONSUMERS = {
    "builtins.list", "builtins.tuple", "builtins.set", "builtins.frozenset",
    "builtins.dict", "builtins.sorted", "builtins.min", "builtins.max",
    "builtins.sum", "builtins.any", "builtins.all", "builtins.next",
    "collections.deque", "collections.OrderedDict",
    "itertools.chain", "functools.reduce",
}
WRAPPERS = {  # hand a callable back unchanged (roughly)
    "functools.lru_cache", "functools.cache", "functools.wraps",
    "functools.partial", "copy.copy", "copy.deepcopy",
}
THREAD_STARTERS = {  # construct-with-target then run; treat target as invoked
    "threading.Thread", "multiprocessing.Process",
    "_thread.start_new_thread",
}

KIND_OF = {}
for _n in CODE_EXEC:
    KIND_OF[_n] = "code-execution"
for _n in PROCESS:
    KIND_OF[_n] = "process"
for _n in FILE_ACCESS:
    KIND_OF[_n] = "file-access"
for _n in NETWORK:
    KIND_OF[_n] = "network"


# ---------------------------------------------------------------------------
# Module alias normalisation (threat model section 3, last bullet).
# ---------------------------------------------------------------------------

MODULE_ALIASES = {
    "posix": "os", "nt": "os",
    "_io": "io",
    "_pickle": "pickle", "cPickle": "pickle",
    "__builtin__": "builtins", "__builtins__": "builtins",
    "copy_reg": "copyreg",
    "_socket": "socket",
    "_functools": "functools",
    "_operator": "operator",
    "_codecs": "codecs",
    "_collections": "collections",
    "_thread": "threading", "thread": "threading",
}


def normalize(module: str, name: str) -> str:
    """Return a canonical ``module.qualname`` string for a pickle global.

    Applies the Python 2 -> 3 fix_imports tables (exactly as the C unpickler's
    ``find_class`` does for protocols below 3) and the module aliases above.
    ``name`` may be dotted (``Bdb.run``); it is preserved.
    """
    module = module or ""
    name = name or ""
    # _compat_pickle NAME_MAPPING keys on (module, top-level name).
    top = name.split(".", 1)[0]
    rest = name[len(top):]
    if (module, top) in _compat_pickle.NAME_MAPPING:
        module, top = _compat_pickle.NAME_MAPPING[(module, top)]
        name = top + rest
    elif module in _compat_pickle.IMPORT_MAPPING:
        module = _compat_pickle.IMPORT_MAPPING[module]
    module = MODULE_ALIASES.get(module, module)
    # a second pass catches posix->os after fix_imports, etc.
    module = MODULE_ALIASES.get(module, module)
    return f"{module}.{name}" if name else module


# ---------------------------------------------------------------------------
# TensorFlow op table (threat model section 4, closed).
# ---------------------------------------------------------------------------

TF_OPS = {}
for _op in ("PyFunc", "PyFuncStateless", "EagerPyFunc"):
    TF_OPS[_op] = "code-execution"
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
    TF_OPS[_op] = "file-access"
for _op in (
    "DataServiceDataset", "DataServiceDatasetV2", "DataServiceDatasetV3",
    "DataServiceDatasetV4", "RegisterDataset", "RegisterDatasetV2",
    "DistributedSave",
):
    TF_OPS[_op] = "network"
# Everything else -- including SaveV2, RestoreV2, MergeV2Checkpoints,
# ShardedFilename, ShardedFilespec -- carries no capability.

# Node attributes whose value references a graph function by name.  Used to
# confirm reachability; the scanner scans every library function regardless.
TF_FUNC_ATTR_OPS = {
    "If", "StatelessIf", "While", "StatelessWhile", "Case",
    "PartitionedCall", "StatefulPartitionedCall", "MapDefun", "For",
}
