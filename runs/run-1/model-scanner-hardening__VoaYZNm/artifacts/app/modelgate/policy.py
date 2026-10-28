"""Capability summaries for the static interpreters.

A reference is inert. These summaries are consulted only for an actual call;
argument-dependent operations (loaders, client constructors, etc.) are handled
by the interpreter. TensorFlow's table is deliberately exact and closed.
"""

TF_OPS = {}
for _kind, _names in {
    "code-execution": "PyFunc PyFuncStateless EagerPyFunc",
    "file-access": """ReadFile WriteFile MatchingFiles MatchingFilesDataset ImmutableConst
        WholeFileReader WholeFileReaderV2 TextLineReader TextLineReaderV2
        FixedLengthRecordReader FixedLengthRecordReaderV2 TFRecordReader TFRecordReaderV2
        LMDBReader LMDBDataset ExperimentalLMDBDataset TextLineDataset TFRecordDataset
        FixedLengthRecordDataset DatasetToTFRecord ExperimentalDatasetToTFRecord
        SaveDataset SaveDatasetV2 LoadDataset SnapshotDataset SnapshotDatasetV2
        LoadAndRemapMatrix GenerateVocabRemapping Save SaveSlices Restore RestoreSlice""",
    "network": """DataServiceDataset DataServiceDatasetV2 DataServiceDatasetV3
        DataServiceDatasetV4 RegisterDataset RegisterDatasetV2 DistributedSave""",
}.items():
    TF_OPS.update(dict.fromkeys(_names.split(), _kind))

# Exact qualified names. Do not use substring matching on names or data.
CALL_EFFECTS = {}

def _add(kind, module, names):
    CALL_EFFECTS.update({module + "." + name: kind for name in names.split()})

_add("code-execution", "builtins", "eval exec compile")
_add("code-execution", "runpy", "run_path run_module _run_code _run_module_code _run_module_as_main")
_add("code-execution", "timeit", "timeit repeat Timer Timer.timeit Timer.repeat Timer.autorange")
_add("code-execution", "code", "InteractiveInterpreter.runsource InteractiveInterpreter.runcode InteractiveConsole.runsource InteractiveConsole.runcode InteractiveConsole.push")
_add("code-execution", "codeop", "compile_command _maybe_compile Compile.__call__ CommandCompiler.__call__")
_add("code-execution", "bdb", "Bdb.run Bdb.runeval Bdb.runctx")
_add("code-execution", "pdb", "run runeval runctx Pdb.run Pdb.runeval Pdb.runctx")
_add("code-execution", "numpy.testing._private.utils", "runstring")
_add("process", "os", """system popen execv execve execvp execvpe execl execle execlp execlpe
    spawnl spawnle spawnlp spawnlpe spawnv spawnve spawnvp spawnvpe posix_spawn posix_spawnp
    fork forkpty startfile""")
_add("process", "subprocess", "Popen call run check_call check_output getoutput getstatusoutput")
_add("process", "pty", "spawn fork")
_add("process", "platform", "popen _syscmd_ver _syscmd_file")
_add("process", "pydoc", "pipepager tempfilepager")
_add("process", "asyncio", "create_subprocess_exec create_subprocess_shell")
_add("process", "multiprocessing", "Pool Manager Process.start")
_add("process", "multiprocessing.context", "Process.start SpawnProcess.start ForkProcess.start ForkServerProcess.start")
_add("process", "multiprocessing.process", "BaseProcess.start")
for _mod in ("multiprocessing.popen_fork", "multiprocessing.popen_spawn_posix", "multiprocessing.popen_spawn_win32", "multiprocessing.popen_forkserver"):
    _add("process", _mod, "Popen")
_add("file-access", "builtins", "open")
_add("file-access", "io", "open open_code FileIO")
_add("file-access", "os", """open remove unlink rename renames replace mkdir makedirs rmdir removedirs
    listdir scandir stat lstat access chmod chown lchmod lchown truncate utime link symlink readlink
    chdir chroot mkfifo mknod statvfs walk fwalk read write fdopen""")
_add("file-access", "shutil", "copy copyfile copyfileobj copymode copystat copy2 copytree move rmtree make_archive unpack_archive chown")
for _cls in ("Path", "PosixPath", "WindowsPath"):
    _add("file-access", "pathlib", " ".join(_cls + "." + m for m in (
        "open read_text read_bytes write_text write_bytes unlink rename replace mkdir rmdir touch "
        "chmod lchmod owner group stat lstat exists is_file is_dir is_symlink is_block_device "
        "is_char_device is_fifo is_socket iterdir glob rglob walk readlink symlink_to hardlink_to link_to resolve").split()))
_add("file-access", "os.path", "exists lexists isfile isdir islink getsize getmtime getatime getctime samefile ismount")
_add("file-access", "tempfile", "TemporaryFile NamedTemporaryFile SpooledTemporaryFile TemporaryDirectory mkstemp mkdtemp mktemp")
_add("file-access", "codecs", "open")
_add("file-access", "fileinput", "input FileInput")
_add("file-access", "sqlite3", "connect")
_add("file-access", "sqlite3.dbapi2", "connect")
_add("file-access", "shelve", "open")
for _mod in ("dbm", "dbm.gnu", "dbm.ndbm", "dbm.dumb"):
    _add("file-access", _mod, "open")
_add("file-access", "logging", "FileHandler")
_add("file-access", "logging.handlers", "WatchedFileHandler RotatingFileHandler TimedRotatingFileHandler")
_add("file-access", "numpy", "fromfile memmap loadtxt genfromtxt savetxt save savez savez_compressed")
_add("network", "socket", "create_connection create_server getaddrinfo gethostbyname gethostbyname_ex gethostbyaddr getnameinfo socket.connect socket.connect_ex socket.sendto socket.sendmsg")
_add("network", "urllib.request", "urlopen urlretrieve OpenerDirector.open URLopener.open FancyURLopener.open Request.get_method")
# Request construction and parsing are inert; get_method is inert as well.
CALL_EFFECTS.pop("urllib.request.Request.get_method")
_add("network", "http.client", "HTTPConnection.connect HTTPConnection.request HTTPConnection.send HTTPConnection.endheaders HTTPSConnection.connect HTTPSConnection.request HTTPSConnection.send HTTPSConnection.endheaders")
_add("network", "ftplib", "FTP.connect FTP.login FTP.retrbinary FTP.retrlines FTP.storbinary FTP.storlines FTP_TLS.connect FTP_TLS.login")
_add("network", "smtplib", "SMTP.connect SMTP.sendmail SMTP.send_message SMTP_SSL.connect SMTP_SSL.sendmail SMTP_SSL.send_message LMTP.connect")
_add("network", "ssl", "get_server_certificate SSLSocket.connect SSLSocket.connect_ex")
_add("network", "webbrowser", "open open_new open_new_tab")
_add("network", "requests", "get post put delete head options patch request Session.request Session.get Session.post")
_add("network", "requests.api", "get post put delete head options patch request")
_add("network", "requests.sessions", "Session.request Session.get Session.post")
_add("network", "urllib3", "request PoolManager.request HTTPConnectionPool.urlopen HTTPConnectionPool.request HTTPSConnectionPool.urlopen HTTPSConnectionPool.request")
_add("network", "multiprocessing.connection", "Client SocketClient PipeClient")

NESTED_LOADERS = {
    "pickle.loads": "pickle-bytes", "pickle.load": "pickle-file",
    "pickle.Unpickler.load": "pickle-unpickler", "pickle._Unpickler.load": "pickle-unpickler",
    "torch.load": "torch", "torch.serialization.load": "torch",
    "torch.storage._load_from_bytes": "torch-bytes",
    "numpy.load": "numpy", "numpy.lib.npyio.load": "numpy", "numpy.lib._npyio_impl.load": "numpy",
    "numpy.lib.format.read_array": "numpy-array",
    "joblib.load": "artifact", "joblib.numpy_pickle.load": "artifact",
    "pandas.read_pickle": "artifact", "pandas.io.pickle.read_pickle": "artifact",
    "dill.loads": "pickle-bytes", "dill.load": "pickle-file",
    "cloudpickle.loads": "pickle-bytes", "cloudpickle.load": "pickle-file",
}

# Other entry points to the same OS and interpreter facilities.
_add("process", "_posixsubprocess", "fork_exec")
_add("process", "multiprocessing.pool", "Pool")
_add("process", "concurrent.futures.process", "ProcessPoolExecutor.submit ProcessPoolExecutor.map")
_add("process", "concurrent.futures", "ProcessPoolExecutor.submit ProcessPoolExecutor.map")
_add("code-execution", "pydoc", "importfile")
_add("file-access", "pydoc", "writedoc writedocs")
_add("file-access", "linecache", "getline getlines updatecache checkcache")
_add("file-access", "tokenize", "open")
_add("file-access", "py_compile", "compile")
_add("file-access", "compileall", "compile_file compile_dir compile_path")
_add("file-access", "glob", "glob")
_add("file-access", "zipfile", "PyZipFile.writepy ZipFile.extract ZipFile.extractall ZipFile.write")
_add("file-access", "tarfile", "TarFile.extract TarFile.extractall TarFile.add")
_add("network", "http.client", "HTTPConnection._send_request HTTPSConnection._send_request")
_add("network", "urllib.request", "AbstractHTTPHandler.do_open HTTPHandler.http_open HTTPSHandler.https_open")
_add("network", "socket", "socket.sendall socket.send")
NESTED_LOADERS["numpy.lib._format_impl.read_array"] = "numpy-array"

# Trusted data reconstruction operations. Unknown *called* globals fail closed;
# an unknown global merely stored as data still needs no summary.
DATA_CALLS = set()
def _data(module, names):
    DATA_CALLS.update(module + "." + name for name in names.split())

_data("builtins", "object complex range memoryview property staticmethod classmethod super enumerate zip reversed round abs pow divmod format repr ascii hash id isinstance issubclass callable hasattr all any sum print")
_data("types", "SimpleNamespace MappingProxyType GenericAlias UnionType")
_data("collections", "deque ChainMap UserDict UserList UserString namedtuple")
_data("collections.abc", "_CallableGenericAlias")
_data("numpy", """dtype ndarray array asarray asanyarray ascontiguousarray asfortranarray
    arange linspace logspace geomspace zeros ones empty full zeros_like ones_like empty_like full_like
    eye identity frombuffer fromiter diagonal diag concatenate stack hstack vstack dstack column_stack
    row_stack reshape transpose squeeze expand_dims broadcast_to broadcast_arrays tile repeat copy
    int8 int16 int32 int64 uint8 uint16 uint32 uint64 float16 float32 float64 float128 longdouble
    complex64 complex128 clongdouble bool_ int_ uint intc intp uintp double single byte ubyte short
    ushort longlong ulonglong object_ str_ bytes_ void datetime64 timedelta64
    abs absolute add subtract multiply divide true_divide floor_divide power square sqrt cbrt reciprocal
    exp exp2 expm1 log log2 log10 log1p sin cos tan arcsin arccos arctan arctan2 hypot sinh cosh tanh
    arcsinh arccosh arctanh deg2rad rad2deg radians degrees negative positive sign signbit copysign
    ceil floor trunc rint round around clip maximum minimum fmax fmin sum prod mean std var min max
    nan_to_num isnan isinf isfinite isneginf isposinf logical_and logical_or logical_not logical_xor
    equal not_equal greater less greater_equal less_equal bitwise_and bitwise_or bitwise_xor invert
    left_shift right_shift mod remainder fmod divmod dot matmul inner outer einsum tensordot where
    take sort argsort argmin argmax cumsum cumprod nansum nanprod nanmean nanstd nanvar nanmin nanmax
    unique bincount histogram histogram2d histogramdd searchsorted nonzero count_nonzero finfo iinfo
    result_type promote_types can_cast issubdtype vectorize frompyfunc""")
_data("numpy.core.multiarray", "_reconstruct scalar _scalar ndarray _get_ndarray_c_version")
_data("numpy.core.numeric", "_frombuffer")
_data("numpy.core._multiarray_umath", "_reconstruct scalar _frombuffer")
_data("numpy.random._pickle", "__randomstate_ctor __generator_ctor __bit_generator_ctor")
_data("numpy.ma.core", "_mareconstruct MaskedArray MaskedConstant")
_data("numpy.polynomial.polynomial", "Polynomial")
_data("torch._utils", """_rebuild_tensor _rebuild_tensor_v2 _rebuild_tensor_v3 _rebuild_parameter
    _rebuild_parameter_with_state _rebuild_sparse_tensor _rebuild_qtensor _rebuild_meta_tensor_no_storage
    _rebuild_device_tensor_from_numpy _rebuild_device_tensor_from_cpu_tensor _rebuild_wrapper_subclass
    _rebuild_nested_tensor _rebuild_xla_tensor _rebuild_xla_tensor_v2 _rebuild_cuda_tensor""")
_data("torch._tensor", "_rebuild_from_type _rebuild_from_type_v2")
_data("torch", "Tensor Size device dtype layout memory_format tensor")
_data("torch.nn.parameter", "Parameter UninitializedParameter UninitializedBuffer")
_data("torch.storage", "TypedStorage UntypedStorage")
_data("pandas._libs.internals", "_unpickle_block")
_data("pandas.core.indexes.base", "_new_Index")
_data("pandas.core.arrays.categorical", "_unpickle_compat")
_data("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked")
_data("pandas._libs.tslibs.timestamps", "_unpickle_timestamp")
_data("pandas._libs.tslibs.timedeltas", "_timedelta_unpickle")
_data("pandas._libs.tslibs.nattype", "__nat_unpickle")
_data("pandas._libs.tslibs.offsets", "__pyx_unpickle_BaseOffset")
_data("pandas._libs.missing", "C_NAType __pyx_unpickle_C_NAType")
_data("logging", "StreamHandler NullHandler Formatter Filter LogRecord Logger getLogger")
_data("code", "InteractiveInterpreter InteractiveConsole")
_data("codeop", "Compile CommandCompiler")
_data("bdb", "Bdb")
_data("pdb", "Pdb")
_data("urllib.request", "Request URLopener FancyURLopener build_opener OpenerDirector")
_data("ssl", "SSLContext")
_data("ctypes", "CFUNCTYPE PYFUNCTYPE WINFUNCTYPE")
_data("multiprocessing", "Process")
_data("multiprocessing.context", "Process ForkProcess SpawnProcess ForkServerProcess")
_data("concurrent.futures", "ThreadPoolExecutor ProcessPoolExecutor")
_data("concurrent.futures.thread", "ThreadPoolExecutor")
_data("concurrent.futures.process", "ProcessPoolExecutor")

PURE_MODULES = {"math", "cmath", "datetime", "decimal", "_decimal", "fractions", "statistics",
                "time", "re", "re._compiler", "_sre", "string", "struct", "_struct", "bisect", "_bisect",
                "operator", "_operator", "itertools", "uuid", "ipaddress", "urllib.parse"}
# Exclusions from otherwise data-oriented modules must precede their summaries.
_add("file-access", "ssl", "create_default_context SSLContext.load_verify_locations SSLContext.load_default_certs SSLContext.load_cert_chain SSLContext.set_default_verify_paths")
_add("file-access", "uuid", "getnode _getnode")
_add("process", "uuid", "_get_command_stdout _ip_getnode _ifconfig_getnode _arp_getnode _netstat_getnode _lanscan_getnode")
