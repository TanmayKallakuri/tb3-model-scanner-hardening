"""Capabilities of calls, never of global references or strings.

Argument-dependent operations are interpreted separately. The TensorFlow table
is closed by policy, whereas Python offers many APIs for the same capability.
"""

TF_OPS = {}
for kind, names in {
    'code-execution': 'PyFunc PyFuncStateless EagerPyFunc',
    'file-access': '''ReadFile WriteFile MatchingFiles MatchingFilesDataset
        ImmutableConst WholeFileReader WholeFileReaderV2 TextLineReader
        TextLineReaderV2 FixedLengthRecordReader FixedLengthRecordReaderV2
        TFRecordReader TFRecordReaderV2 LMDBReader LMDBDataset
        ExperimentalLMDBDataset TextLineDataset TFRecordDataset
        FixedLengthRecordDataset DatasetToTFRecord ExperimentalDatasetToTFRecord
        SaveDataset SaveDatasetV2 LoadDataset SnapshotDataset SnapshotDatasetV2
        LoadAndRemapMatrix GenerateVocabRemapping Save SaveSlices Restore RestoreSlice''',
    'network': '''DataServiceDataset DataServiceDatasetV2 DataServiceDatasetV3
        DataServiceDatasetV4 RegisterDataset RegisterDatasetV2 DistributedSave''',
}.items():
    TF_OPS.update(dict.fromkeys(names.split(), kind))

CALLS = {}


def _add(kind, module, names):
    CALLS.update({f'{module}.{name}': kind for name in names.split()})


_add('code-execution', 'builtins', 'eval exec compile execfile')
_add('code-execution', 'types', 'FunctionType LambdaType')
_add('code-execution', 'runpy', 'run_path run_module _run_code _run_module_code _run_module_as_main')
_add('code-execution', 'timeit', 'timeit repeat')
_add('code-execution', 'code', 'InteractiveInterpreter.runsource InteractiveInterpreter.runcode InteractiveConsole.push')
_add('code-execution', 'code.InteractiveConsole', 'runsource runcode')
_add('code-execution', 'codeop', 'compile_command _maybe_compile')
_add('code-execution', 'codeop.Compile', '__call__')
_add('code-execution', 'codeop.CommandCompiler', '__call__')
_add('code-execution', 'bdb', 'Bdb.run Bdb.runeval Bdb.runctx')
_add('code-execution', 'pdb', 'run runeval runctx Pdb.run Pdb.runeval Pdb.runctx')
_add('code-execution', 'profile', 'run runctx Profile.run Profile.runctx')
_add('code-execution', 'cProfile', 'run runctx Profile.run Profile.runctx')
_add('code-execution', 'numpy.testing._private.utils', 'runstring')
_add('code-execution', 'numpy.testing.utils', 'runstring')
_add('code-execution', 'numpy.f2py', 'compile')
_add('code-execution', 'numpy.ctypeslib', 'load_library')
_add('code-execution', 'pandas', 'eval')
_add('code-execution', 'pandas.core.computation.eval', 'eval')
_add('code-execution', 'torch.hub', 'load')
_add('code-execution', 'torch.utils.cpp_extension', 'load load_inline')
_add('code-execution', 'importlib.machinery', 'SourceFileLoader.load_module SourceFileLoader.exec_module SourcelessFileLoader.load_module SourcelessFileLoader.exec_module ExtensionFileLoader.load_module')
_add('code-execution', 'imp', 'load_source load_compiled load_dynamic')
_add('code-execution', 'ctypes', 'CDLL PyDLL WinDLL OleDLL cdll.LoadLibrary pydll.LoadLibrary')

_add('process', 'os', '''system popen execl execle execlp execlpe execv execve
    execvp execvpe spawnl spawnle spawnlp spawnlpe spawnv spawnve spawnvp
    spawnvpe posix_spawn posix_spawnp fork forkpty startfile''')
_add('process', 'subprocess', 'Popen call run check_call check_output getoutput getstatusoutput')
_add('process', 'pty', 'spawn fork')
_add('process', 'platform', 'popen _syscmd_ver _syscmd_uname _syscmd_file')
_add('process', 'pydoc', 'pipepager tempfilepager')
_add('process', 'multiprocessing', 'Process.start Pool')
_add('process', 'multiprocessing.process', 'BaseProcess.start')
_add('process', 'multiprocessing.pool', 'Pool')

_add('file-access', 'builtins', 'open')
_add('file-access', 'io', 'open open_code FileIO')
_add('file-access', 'os', '''open fdopen remove unlink rename renames replace mkdir
    makedirs rmdir removedirs listdir scandir stat lstat readlink link symlink
    chmod chown lchmod lchown utime truncate read write chdir chroot access''')
_add('file-access', 'os.path', 'exists lexists isfile isdir islink getsize getmtime getatime getctime samefile')
_add('file-access', 'shutil', '''copy copy2 copyfile copyfileobj copymode copystat
    copytree move rmtree make_archive unpack_archive disk_usage chown which''')
_add('file-access', 'pathlib.Path', '''open read_text read_bytes write_text write_bytes
    unlink rename replace mkdir rmdir touch chmod lchmod stat lstat exists is_dir
    is_file is_symlink iterdir glob rglob resolve owner group readlink
    symlink_to hardlink_to link_to''')
_add('file-access', 'tempfile', 'TemporaryFile NamedTemporaryFile mkstemp mkdtemp TemporaryDirectory')
_add('file-access', 'glob', 'glob')
_add('file-access', 'shelve', 'open')
_add('file-access', 'dbm', 'open')
_add('file-access', 'dbm.dumb', 'open')
_add('file-access', 'sqlite3', 'connect')
_add('file-access', 'numpy', 'fromfile memmap loadtxt genfromtxt savetxt save savez savez_compressed')
_add('file-access', 'numpy.core.memmap', 'memmap')
_add('file-access', 'numpy._core.memmap', 'memmap')
_add('file-access', 'logging', 'FileHandler')
_add('file-access', 'logging.handlers', 'RotatingFileHandler TimedRotatingFileHandler WatchedFileHandler')
_add('file-access', 'h5py', 'File')
_add('file-access', 'gzip', 'open GzipFile')
_add('file-access', 'bz2', 'open BZ2File')
_add('file-access', 'lzma', 'open LZMAFile')
_add('file-access', 'tarfile', 'open TarFile.open')
_add('file-access', 'zipfile', 'ZipFile')
_add('file-access', 'zipfile.ZipFile', 'extract extractall write')
_add('file-access', 'tarfile.TarFile', 'extract extractall add')
_add('file-access', 'numpy.lib._datasource', 'open DataSource.open Repository.open')
_add('file-access', 'numpy', 'DataSource.open')
_add('file-access', 'pandas', 'HDFStore')

_add('network', 'socket', 'create_connection getaddrinfo gethostbyname gethostbyname_ex gethostbyaddr')
_add('network', 'socket.socket', 'connect connect_ex sendto sendmsg')
_add('network', 'urllib.request', 'urlopen urlretrieve OpenerDirector.open')
_add('network', 'urllib', 'urlopen urlretrieve')
_add('network', 'http.client.HTTPConnection', 'connect request endheaders send')
_add('network', 'http.client.HTTPSConnection', 'connect request endheaders send')
_add('network', 'ftplib.FTP', 'connect login transfercmd retrbinary storbinary')
_add('network', 'ftplib.FTP_TLS', 'connect login transfercmd retrbinary storbinary')
_add('network', 'smtplib.SMTP', 'connect sendmail send_message login')
_add('network', 'smtplib.SMTP_SSL', 'connect sendmail send_message login')
_add('network', 'ssl', 'get_server_certificate')
_add('network', 'webbrowser', 'open open_new open_new_tab')
_add('network', 'requests', 'request get post put patch delete head options')
_add('network', 'requests.api', 'request get post put patch delete head options')
_add('network', 'requests.sessions.Session', 'request get post put patch delete head options send')
_add('network', 'torch.hub', 'download_url_to_file load_state_dict_from_url')
_add('network', 'torch.utils.model_zoo', 'load_url')

HOST_CLIENTS = {'ftplib.FTP', 'ftplib.FTP_TLS', 'smtplib.SMTP', 'smtplib.SMTP_SSL',
                'telnetlib.Telnet', 'poplib.POP3', 'poplib.POP3_SSL',
                'imaplib.IMAP4', 'imaplib.IMAP4_SSL'}

NESTED_LOADERS = {
    'pickle.loads', 'pickle.load', 'pickle._loads', 'pickle._load',
    'pickle.Unpickler.load', 'pickle._Unpickler.load',
    'torch.load', 'torch.serialization.load', 'torch.storage._load_from_bytes',
    'numpy.load', 'numpy.lib.npyio.load', 'numpy.lib._npyio_impl.load',
    'numpy.lib.format.read_array', 'joblib.load', 'joblib.numpy_pickle.load',
    'pandas.read_pickle', 'pandas.io.pickle.read_pickle',
    'dill.loads', 'dill.load', 'cloudpickle.loads', 'cloudpickle.load',
}

# These APIs construct ordinary model state or perform computations. Their
# results can still carry capabilities, so the interpreter preserves identity
# and class names rather than discarding them. No imported artifact global is
# ever used as a Python callable by the scanner.
INERT_GLOBALS = set('''
    builtins.object builtins.repr builtins.ascii builtins.hex builtins.oct
    builtins.abs builtins.round builtins.divmod builtins.pow builtins.hash
    builtins.isinstance builtins.issubclass builtins.callable builtins.hasattr
    types.SimpleNamespace types.CodeType types.ModuleType types.MappingProxyType
    code.InteractiveInterpreter code.InteractiveConsole codeop.Compile
    codeop.CommandCompiler bdb.Bdb pdb.Pdb profile.Profile cProfile.Profile
    trace.Trace timeit.Timer logging.config.DictConfigurator
    socket.socket socket.SocketType http.client.HTTPConnection
    http.client.HTTPSConnection urllib.request.Request
    urllib.request.build_opener urllib.request.ProxyHandler
    xmlrpc.client.ServerProxy xmlrpc.client.Transport xmlrpc.client.SafeTransport
    requests.Session requests.sessions.Session
    ssl.SSLContext ssl.MemoryBIO
    threading.Thread multiprocessing.Process multiprocessing.process.BaseProcess
    multiprocessing.context.Process
    logging.Formatter logging.Filter logging.StreamHandler logging.NullHandler
    logging.handlers.MemoryHandler logging.handlers.QueueHandler
    logging.getLogger logging.Logger logging.LoggerAdapter
    pickle.PickleBuffer
'''.split())

INERT_MODULES = {
    'math', 'cmath', 'decimal', '_decimal', 'fractions', 'datetime', '_datetime',
    'array', 'collections', 'collections.abc', 're', 're._compiler', 're._parser',
    'sre_compile', 'sre_parse', 'copy', 'copyreg', 'numbers', 'typing', 'enum',
    'ipaddress', 'statistics', 'string', 'struct', '_struct', 'itertools',
}
