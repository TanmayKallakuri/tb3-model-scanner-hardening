"""Capability descriptions. These apply to *calls*, never to stored names.

TensorFlow's table is closed by the registry policy. Python entries describe
stdlib/library operations; argument-dependent operations live in pickle_scan.
"""

TF_OPS = {}
for _kind, _names in {
    "code-execution": "PyFunc PyFuncStateless EagerPyFunc",
    "file-access": """ReadFile WriteFile MatchingFiles MatchingFilesDataset
        ImmutableConst WholeFileReader WholeFileReaderV2 TextLineReader
        TextLineReaderV2 FixedLengthRecordReader FixedLengthRecordReaderV2
        TFRecordReader TFRecordReaderV2 LMDBReader LMDBDataset
        ExperimentalLMDBDataset TextLineDataset TFRecordDataset
        FixedLengthRecordDataset DatasetToTFRecord ExperimentalDatasetToTFRecord
        SaveDataset SaveDatasetV2 LoadDataset SnapshotDataset SnapshotDatasetV2
        LoadAndRemapMatrix GenerateVocabRemapping Save SaveSlices Restore RestoreSlice""",
    "network": """DataServiceDataset DataServiceDatasetV2 DataServiceDatasetV3
        DataServiceDatasetV4 RegisterDataset RegisterDatasetV2 DistributedSave""",
}.items():
    TF_OPS.update(dict.fromkeys(_names.split(), _kind))

# Methods appear qualified by their receiver's class as well as ordinary
# module globals. Aliases and bound receivers are resolved by the interpreter.
CALL_CAPABILITIES = {}

def _add(kind, module, names):
    CALL_CAPABILITIES.update({module + "." + name: kind for name in names.split()})

_add("code-execution", "builtins", "eval exec compile")
_add("code-execution", "runpy", "run_module run_path _run_code _run_module_code _run_module_as_main")
_add("code-execution", "code", "InteractiveInterpreter.runsource InteractiveInterpreter.runcode InteractiveConsole.push")
_add("code-execution", "codeop", "compile_command Compile.__call__ CommandCompiler.__call__")
_add("code-execution", "bdb", "Bdb.run Bdb.runeval Bdb.runctx")
_add("code-execution", "pdb", "run runeval runctx Pdb.run Pdb.runeval Pdb.runctx")
_add("code-execution", "numpy.testing._private.utils", "runstring")
_add("code-execution", "numpy.f2py", "compile")
_add("code-execution", "profile", "run runctx Profile.run Profile.runctx")
_add("code-execution", "cProfile", "run runctx Profile.run Profile.runctx")
_add("code-execution", "trace.Trace", "run runctx")
_add("process", "os", """system popen execv execve execvp execvpe execl execle
    execlp execlpe spawnl spawnle spawnlp spawnlpe spawnv spawnve spawnvp spawnvpe
    posix_spawn posix_spawnp fork forkpty startfile""")
_add("process", "subprocess", "Popen call run check_call check_output getoutput getstatusoutput")
_add("process", "pty", "spawn fork")
_add("process", "platform", "popen _syscmd_ver _syscmd_file")
_add("process", "pydoc", "pipepager tempfilepager")
_add("process", "_posixsubprocess", "fork_exec")
_add("process", "multiprocessing.util", "spawnv_passfds")
_add("process", "distutils.spawn", "spawn")
_add("process", "setuptools._distutils.spawn", "spawn")
_add("file-access", "builtins", "open")
_add("file-access", "io", "open open_code FileIO")
_add("file-access", "os", """open fdopen remove unlink rename renames replace mkdir
    makedirs rmdir removedirs listdir scandir stat lstat access chmod chown lchown
    lchmod truncate utime link symlink readlink chdir chroot read write pread pwrite
    sendfile copy_file_range mknod mkfifo statvfs fstat ftruncate fsync fdatasync""")
_add("file-access", "os.path", "exists lexists isfile isdir islink ismount getsize getmtime getatime getctime samefile")
_add("file-access", "shutil", """copy copy2 copyfile copyfileobj copymode copystat
    copytree move rmtree chown disk_usage make_archive unpack_archive""")
_add("file-access", "glob", "glob")
_add("file-access", "tempfile", "TemporaryFile NamedTemporaryFile TemporaryDirectory mkstemp mkdtemp SpooledTemporaryFile.rollover")
_add("file-access", "pathlib.Path", """open read_text read_bytes write_text write_bytes
    unlink rename replace mkdir rmdir touch chmod lchmod owner group stat lstat
    exists is_file is_dir is_symlink is_mount is_block_device is_char_device
    is_fifo is_socket readlink symlink_to hardlink_to link_to resolve samefile""")
_add("network", "socket", "create_connection create_server getaddrinfo gethostbyname gethostbyname_ex gethostbyaddr")
_add("network", "socket.socket", "connect connect_ex sendto sendmsg bind listen accept send sendall")
_add("network", "urllib.request", "urlopen urlretrieve OpenerDirector.open URLopener.open FancyURLopener.open")
_add("network", "http.client.HTTPConnection", "connect request send endheaders")
_add("network", "http.client.HTTPSConnection", "connect request send endheaders")
_add("network", "ftplib.FTP", "connect login retrbinary retrlines storbinary storlines transfercmd ntransfercmd")
_add("network", "ftplib.FTP_TLS", "connect login retrbinary retrlines storbinary storlines transfercmd ntransfercmd")
_add("network", "smtplib.SMTP", "connect sendmail send_message login ehlo helo")
_add("network", "smtplib.SMTP_SSL", "connect sendmail send_message login ehlo helo")
_add("network", "ssl", "get_server_certificate SSLSocket.connect SSLSocket.connect_ex")
_add("network", "multiprocessing.connection", "Client SocketClient")
_add("network", "webbrowser", "open open_new open_new_tab")
_add("network", "requests", "request get post put patch delete head options Session.request Session.get Session.post")
_add("network", "httpx", "request get post put patch delete head options Client.request Client.get Client.post")
_add("file-access", "sqlite3", "connect")
_add("file-access", "dbm", "open")
_add("file-access", "shelve", "open")
_add("file-access", "logging", "FileHandler")
_add("file-access", "logging.handlers", "RotatingFileHandler TimedRotatingFileHandler WatchedFileHandler")

NESTED_LOADERS = {
    "pickle.loads": "pickle-bytes", "pickle.load": "pickle-file",
    "pickle.Unpickler.load": "pickle-file", "pickle._Unpickler.load": "pickle-file",
    "torch.load": "torch", "torch.serialization.load": "torch",
    "torch.storage._load_from_bytes": "torch-bytes",
    "numpy.load": "numpy", "numpy.lib.npyio.load": "numpy",
    "numpy.lib._npyio_impl.load": "numpy", "numpy.lib.format.read_array": "npy",
    "joblib.load": "joblib", "joblib.numpy_pickle.load": "joblib",
    "pandas.read_pickle": "pickle-file", "pandas.io.pickle.read_pickle": "pickle-file",
    "dill.loads": "pickle-bytes", "dill.load": "pickle-file",
    "cloudpickle.loads": "pickle-bytes", "cloudpickle.load": "pickle-file",
}
