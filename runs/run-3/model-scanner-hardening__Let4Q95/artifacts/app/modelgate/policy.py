"""Capability tables, applied to executing calls, never to names in data."""

TF_OPS = {}
for _kind, _ops in {
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
    TF_OPS.update(dict.fromkeys(_ops.split(), _kind))

CALLS = {}
for _kind, _names in {
    'code-execution': '''
        builtins.eval builtins.exec builtins.compile builtins.execfile
        runpy.run_path runpy.run_module runpy._run_code runpy._run_module_code
        timeit.timeit timeit.repeat timeit.Timer
        code.InteractiveInterpreter.runsource code.InteractiveInterpreter.runcode
        code.InteractiveConsole.push code.compile_command codeop.compile_command
        bdb.Bdb.run bdb.Bdb.runeval bdb.Bdb.runctx bdb.Bdb.runcall
        pdb.run pdb.runeval pdb.runctx pdb.runcall
        pdb.Pdb.run pdb.Pdb.runeval pdb.Pdb.runctx pdb.Pdb.runcall
        numpy.testing._private.utils.runstring numpy.testing.utils.runstring
        numpy.f2py.compile types.FunctionType types.LambdaType
        cloudpickle.cloudpickle._make_function dill._dill._create_function
    ''',
    'process': '''
        os.system os.popen os.execl os.execle os.execlp os.execlpe
        os.execv os.execve os.execvp os.execvpe os.spawnl os.spawnle
        os.spawnlp os.spawnlpe os.spawnv os.spawnve os.spawnvp os.spawnvpe
        os.posix_spawn os.posix_spawnp os.fork os.forkpty os.startfile
        subprocess.Popen subprocess.call subprocess.run subprocess.check_call
        subprocess.check_output subprocess.getoutput subprocess.getstatusoutput
        subprocess._fork_exec _posixsubprocess.fork_exec
        pty.spawn platform.popen pydoc.pipepager pydoc.tempfilepager
        multiprocessing.Process.start multiprocessing.process.BaseProcess.start
        multiprocessing.pool.Pool multiprocessing.Pool
        concurrent.futures.ProcessPoolExecutor.submit
        asyncio.create_subprocess_exec asyncio.create_subprocess_shell
    ''',
    'file-access': '''
        builtins.open io.open io.open_code io.FileIO
        os.open os.read os.write os.remove os.unlink os.rename os.renames
        os.replace os.mkdir os.makedirs os.rmdir os.removedirs os.listdir
        os.scandir os.stat os.lstat os.readlink os.link os.symlink os.truncate
        os.chmod os.chown os.lchmod os.lchown os.utime os.access os.chdir
        os.chroot os.walk os.fwalk os.mkfifo os.mknod
        shutil.copy shutil.copy2 shutil.copyfile shutil.copytree shutil.move
        shutil.rmtree shutil.copymode shutil.copystat shutil.make_archive
        shutil.unpack_archive shutil.disk_usage
        tempfile.mkstemp tempfile.mkdtemp tempfile.TemporaryFile
        tempfile.NamedTemporaryFile tempfile.TemporaryDirectory
        pathlib.Path.open pathlib.Path.read_text pathlib.Path.read_bytes
        pathlib.Path.write_text pathlib.Path.write_bytes pathlib.Path.unlink
        pathlib.Path.rename pathlib.Path.replace pathlib.Path.mkdir
        pathlib.Path.rmdir pathlib.Path.touch pathlib.Path.iterdir
        pathlib.Path.glob pathlib.Path.rglob pathlib.Path.stat pathlib.Path.lstat
        pathlib.Path.exists pathlib.Path.is_file pathlib.Path.is_dir
        pathlib.Path.chmod pathlib.Path.lchmod pathlib.Path.symlink_to
        pathlib.Path.hardlink_to pathlib.Path.readlink pathlib.Path.resolve
        glob.glob glob.iglob linecache.getline linecache.getlines
        fileinput.input fileinput.FileInput sqlite3.connect
        shelve.open dbm.open dbm.dumb.open dbm.gnu.open dbm.ndbm.open
        logging.FileHandler logging.handlers.RotatingFileHandler
        logging.handlers.TimedRotatingFileHandler logging.handlers.WatchedFileHandler
        numpy.fromfile numpy.memmap numpy.savetxt numpy.save numpy.savez
        numpy.savez_compressed numpy.ndarray.tofile numpy.ndarray.dump
        torch.save joblib.dump pandas.to_pickle
    ''',
    'network': '''
        socket.create_connection socket.getaddrinfo socket.gethostbyname
        socket.gethostbyname_ex socket.gethostbyaddr
        socket.socket.connect socket.socket.connect_ex socket.socket.sendto
        socket.socket.sendmsg
        urllib.request.urlopen urllib.request.urlretrieve urllib.request.OpenerDirector.open
        http.client.HTTPConnection.connect http.client.HTTPConnection.request
        http.client.HTTPConnection.endheaders http.client.HTTPConnection.send
        http.client.HTTPSConnection.connect http.client.HTTPSConnection.request
        http.client.HTTPSConnection.endheaders http.client.HTTPSConnection.send
        ftplib.FTP.connect ftplib.FTP.login ftplib.FTP_TLS.connect
        smtplib.SMTP.connect smtplib.SMTP_SSL.connect
        poplib.POP3 poplib.POP3_SSL imaplib.IMAP4 imaplib.IMAP4_SSL
        ssl.get_server_certificate webbrowser.open webbrowser.open_new
        webbrowser.open_new_tab requests.get requests.post requests.put
        requests.delete requests.head requests.patch requests.request
        requests.api.request requests.sessions.Session.request
        requests.sessions.Session.get requests.sessions.Session.post
        urllib3.PoolManager.request urllib3.HTTPConnectionPool.urlopen
    ''',
}.items():
    CALLS.update(dict.fromkeys(_names.split(), _kind))

NESTED_LOADERS = {
    'pickle.loads': 'pickle-bytes', 'pickle.load': 'pickle-file',
    'pickle.Unpickler.load': 'pickle-unpickler',
    'torch.load': 'artifact', 'torch.storage._load_from_bytes': 'torch-bytes',
    'numpy.load': 'numpy', 'numpy.lib.npyio.load': 'numpy',
    'numpy.lib._npyio_impl.load': 'numpy',
    'numpy.lib.format.read_array': 'numpy-array',
    'joblib.load': 'artifact', 'joblib.numpy_pickle.load': 'artifact',
    'pandas.read_pickle': 'artifact', 'pandas.io.pickle.read_pickle': 'artifact',
    'cloudpickle.loads': 'pickle-bytes', 'cloudpickle.load': 'pickle-file',
    'dill.loads': 'pickle-bytes', 'dill.load': 'pickle-file',
}
