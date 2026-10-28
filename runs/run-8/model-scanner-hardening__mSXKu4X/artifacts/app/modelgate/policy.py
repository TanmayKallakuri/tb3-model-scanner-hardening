"""Capability summaries. Names are consulted at *calls*, never in model data."""

TF_OPS = {}
for _kind, _ops in {
    'code-execution': 'PyFunc PyFuncStateless EagerPyFunc',
    'file-access': '''ReadFile WriteFile MatchingFiles MatchingFilesDataset ImmutableConst
        WholeFileReader WholeFileReaderV2 TextLineReader TextLineReaderV2
        FixedLengthRecordReader FixedLengthRecordReaderV2 TFRecordReader TFRecordReaderV2
        LMDBReader LMDBDataset ExperimentalLMDBDataset TextLineDataset TFRecordDataset
        FixedLengthRecordDataset DatasetToTFRecord ExperimentalDatasetToTFRecord
        SaveDataset SaveDatasetV2 LoadDataset SnapshotDataset SnapshotDatasetV2
        LoadAndRemapMatrix GenerateVocabRemapping Save SaveSlices Restore RestoreSlice''',
    'network': '''DataServiceDataset DataServiceDatasetV2 DataServiceDatasetV3
        DataServiceDatasetV4 RegisterDataset RegisterDatasetV2 DistributedSave''',
}.items():
    TF_OPS.update(dict.fromkeys(_ops.split(), _kind))

CALL_CAPABILITIES = {}
def _add(kind, module, names):
    CALL_CAPABILITIES.update({module + '.' + n: kind for n in names.split()})

for _module, _names in {
    'builtins': 'eval exec compile',
    'runpy': 'run_path run_module _run_code _run_module_code _run_module_as_main',
    'timeit': 'timeit repeat Timer.timeit Timer.repeat Timer.autorange',
    'code': 'interact InteractiveInterpreter.runsource InteractiveInterpreter.runcode InteractiveConsole.push InteractiveConsole.interact',
    'bdb': 'Bdb.run Bdb.runeval Bdb.runctx',
    'pdb': 'run runeval runcall Pdb.run Pdb.runeval Pdb.runctx',
    'numpy.testing._private.utils': 'runstring',
    'logging.config': 'dictConfig fileConfig listen',
    'types': 'FunctionType LambdaType',
    'ctypes': 'CFUNCTYPE PYFUNCTYPE WINFUNCTYPE',
}.items():
    _add('code-execution', _module, _names)
for _module, _names in {
    'os': '''system popen execv execve execvp execvpe execl execle execlp execlpe
        spawnv spawnve spawnvp spawnvpe spawnl spawnle spawnlp spawnlpe posix_spawn
        posix_spawnp fork forkpty startfile''',
    'subprocess': '''Popen run call check_call check_output getoutput getstatusoutput
        _fork_exec''',
    'pty': 'spawn fork',
    'pydoc': 'pipepager tempfilepager',
    '_posixsubprocess': 'fork_exec',
    '_winapi': 'CreateProcess',
    'multiprocessing': 'Process.start Pool',
    'multiprocessing.context': 'BaseContext.Pool BaseContext.Process',
    'multiprocessing.process': 'BaseProcess.start',
    'multiprocessing.pool': 'Pool ThreadPool',
    'asyncio': 'create_subprocess_exec create_subprocess_shell',
}.items():
    _add('process', _module, _names)
for _module, _names in {
    'builtins': 'open',
    'io': 'open open_code FileIO.open',
    'os': '''open creat fdopen remove unlink rename renames replace mkdir makedirs
        rmdir removedirs listdir scandir walk fwalk stat lstat access chmod fchmod
        chown lchown fchown link symlink readlink truncate ftruncate utime chdir
        chroot mknod mkfifo sendfile read write pread pwrite readv writev''',
    'shutil': '''copy copy2 copyfile copyfileobj copymode copystat copytree move rmtree
        make_archive unpack_archive chown disk_usage''',
    'glob': 'glob iglob',
    'tempfile': 'mkstemp mkdtemp NamedTemporaryFile TemporaryFile TemporaryDirectory SpooledTemporaryFile',
    'fileinput': 'input FileInput',
    'linecache': 'getline getlines updatecache',
    'tokenize': 'open',
    'sqlite3': 'connect',
    'dbm': 'open',
    'dbm.dumb': 'open',
    'dbm.gnu': 'open',
    'dbm.ndbm': 'open',
    'shelve': 'open',
    'pathlib': '''Path.open Path.read_text Path.read_bytes Path.write_text Path.write_bytes
        Path.touch Path.unlink Path.rename Path.replace Path.mkdir Path.rmdir Path.iterdir
        Path.glob Path.rglob Path.stat Path.lstat Path.chmod Path.lchmod Path.symlink_to
        Path.hardlink_to Path.link_to Path.readlink Path.resolve Path.exists Path.is_file
        Path.is_dir Path.owner Path.group Path.samefile Path.walk''',
    'numpy': 'save savez savez_compressed savetxt loadtxt genfromtxt fromfile memmap',
    'torch': 'save',
    'joblib': 'dump',
    'pandas': 'read_csv read_table read_json read_excel read_parquet read_feather read_hdf',
    'h5py': 'File',
}.items():
    _add('file-access', _module, _names)
for _module, _names in {
    'socket': '''create_connection create_server getaddrinfo gethostbyname gethostbyname_ex
        gethostbyaddr socket.connect socket.connect_ex socket.sendto socket.sendmsg
        socket.bind socket.listen''',
    'urllib.request': 'urlopen urlretrieve Request.get_method OpenerDirector.open',
    'http.client': '''HTTPConnection.connect HTTPConnection.request HTTPConnection.endheaders
        HTTPConnection.send HTTPSConnection.connect HTTPSConnection.request
        HTTPSConnection.endheaders HTTPSConnection.send''',
    'ftplib': 'FTP.connect FTP.login FTP_TLS.connect FTP_TLS.login',
    'smtplib': 'SMTP.connect SMTP_SSL.connect LMTP.connect',
    'poplib': 'POP3 POP3_SSL',
    'imaplib': 'IMAP4 IMAP4_SSL IMAP4_stream',
    'ssl': 'get_server_certificate',
    'webbrowser': 'open open_new open_new_tab get',
    'requests': 'get post put delete patch head options request Session.request Session.get Session.post',
    'urllib3': 'request PoolManager.request',
    'httpx': 'get post put delete patch head options request Client.get Client.request',
    'asyncio': 'open_connection start_server',
}.items():
    _add('network', _module, _names)
# Constructing Request, HTTPConnection, Socket, or ServerProxy is inert.
CALL_CAPABILITIES.pop('urllib.request.Request.get_method', None)
CALL_CAPABILITIES.pop('webbrowser.get', None)

NESTED_LOADERS = {
    'pickle.loads', 'pickle.load', 'pickle.Unpickler.load',
    'torch.load', 'torch.storage._load_from_bytes',
    'numpy.load', 'numpy.lib.format.read_array', 'numpy.ndarray.loads',
    'joblib.load', 'joblib.numpy_pickle.load', 'pandas.read_pickle',
    'pandas.io.pickle.read_pickle', 'dill.loads', 'dill.load',
    'cloudpickle.loads', 'cloudpickle.load',
}
PYTHON2_MODULE_ALIASES = {
    'posix': 'os', 'nt': 'os', '_pickle': 'pickle', '_io': 'io',
    '_operator': 'operator', '_functools': 'functools', '_socket': 'socket',
    '__builtin__': 'builtins', 'copy_reg': 'copyreg', 'commands': 'subprocess',
    'urllib2': 'urllib.request', 'httplib': 'http.client', 'cPickle': 'pickle',
    'cStringIO': 'io', 'StringIO': 'io',
}
