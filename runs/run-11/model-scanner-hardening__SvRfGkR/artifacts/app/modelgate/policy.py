"""Capability summaries. A reference to a callable is not an execution."""

TF_OPS = {}
for _kind, _names in {
    "code-execution": "PyFunc PyFuncStateless EagerPyFunc",
    "file-access": """ReadFile WriteFile MatchingFiles MatchingFilesDataset
        ImmutableConst WholeFileReader WholeFileReaderV2 TextLineReader
        TextLineReaderV2 FixedLengthRecordReader FixedLengthRecordReaderV2
        TFRecordReader TFRecordReaderV2 LMDBReader LMDBDataset ExperimentalLMDBDataset
        TextLineDataset TFRecordDataset FixedLengthRecordDataset DatasetToTFRecord
        ExperimentalDatasetToTFRecord SaveDataset SaveDatasetV2 LoadDataset
        SnapshotDataset SnapshotDatasetV2 LoadAndRemapMatrix GenerateVocabRemapping
        Save SaveSlices Restore RestoreSlice""",
    "network": """DataServiceDataset DataServiceDatasetV2 DataServiceDatasetV3
        DataServiceDatasetV4 RegisterDataset RegisterDatasetV2 DistributedSave""",
}.items():
    TF_OPS.update(dict.fromkeys(_names.split(), _kind))

# Argument dependent calls, nested deserializers and callbacks are interpreted
# separately. These are effects of calls, never matches against artifact text.
CALL_EFFECTS = {}
for _kind, _names in {
    "code-execution": """
        builtins.eval builtins.exec builtins.compile
        runpy.run_path runpy.run_module runpy._run_code runpy._run_module_code
        timeit.timeit timeit.repeat timeit.Timer
        code.InteractiveInterpreter.runsource code.InteractiveInterpreter.runcode
        code.InteractiveConsole.push bdb.Bdb.run bdb.Bdb.runeval bdb.Bdb.runctx
        pdb.run pdb.runeval pdb.runctx
        numpy.testing._private.utils.runstring
        types.FunctionType types.LambdaType types.CodeType
        cloudpickle.cloudpickle._make_function cloudpickle.cloudpickle._make_skeleton_class
        dill._dill._create_function dill._dill._create_code
        importlib.machinery.SourceFileLoader.load_module
        importlib.machinery.SourceFileLoader.exec_module
        importlib.machinery.SourcelessFileLoader.load_module
        importlib.machinery.SourcelessFileLoader.exec_module
    """,
    "process": """
        os.system os.popen os.execv os.execvp os.execve os.execvpe
        os.execl os.execlp os.execle os.execlpe
        os.spawnl os.spawnlp os.spawnle os.spawnlpe os.spawnv os.spawnvp
        os.spawnve os.spawnvpe os.posix_spawn os.posix_spawnp os.fork os.forkpty
        os.startfile subprocess.Popen subprocess.call subprocess.run
        subprocess.check_call subprocess.check_output subprocess.getoutput
        subprocess.getstatusoutput pty.spawn pty.fork pydoc.pipepager
        multiprocessing.Pool multiprocessing.pool.Pool
        multiprocessing.Process.start multiprocessing.process.BaseProcess.start
        _posixsubprocess.fork_exec _winapi.CreateProcess
    """,
    "file-access": """
        builtins.open io.open io.open_code io.FileIO
        os.open os.fdopen os.read os.write os.readv os.writev os.pread os.pwrite
        os.remove os.unlink os.rename os.renames os.replace os.mkdir os.makedirs
        os.rmdir os.removedirs os.listdir os.scandir os.stat os.lstat os.statvfs
        os.chmod os.chown os.lchown os.utime os.link os.symlink os.truncate
        os.readlink os.walk os.fwalk os.access os.chroot os.chdir
        shutil.copy shutil.copy2 shutil.copyfile shutil.copyfileobj shutil.copytree
        shutil.move shutil.rmtree shutil.chown shutil.copymode shutil.copystat
        shutil.make_archive shutil.unpack_archive
        pathlib.Path.open pathlib.Path.read_text pathlib.Path.read_bytes
        pathlib.Path.write_text pathlib.Path.write_bytes pathlib.Path.unlink
        pathlib.Path.rename pathlib.Path.replace pathlib.Path.mkdir pathlib.Path.rmdir
        pathlib.Path.touch pathlib.Path.iterdir pathlib.Path.glob pathlib.Path.rglob
        pathlib.Path.stat pathlib.Path.lstat pathlib.Path.exists pathlib.Path.is_file
        pathlib.Path.is_dir pathlib.Path.chmod pathlib.Path.lchmod pathlib.Path.owner
        pathlib.Path.group pathlib.Path.readlink pathlib.Path.symlink_to
        pathlib.Path.hardlink_to pathlib.Path.link_to
        tempfile.mkstemp tempfile.mkdtemp tempfile.NamedTemporaryFile
        tempfile.TemporaryFile tempfile.TemporaryDirectory
        glob.glob glob.iglob glob._glob1 glob._glob2
        logging.FileHandler logging.handlers.RotatingFileHandler
        logging.handlers.TimedRotatingFileHandler logging.handlers.WatchedFileHandler
        numpy.fromfile numpy.memmap numpy.save numpy.savez numpy.savez_compressed
        numpy.savetxt numpy.loadtxt numpy.genfromtxt numpy.recfromcsv numpy.recfromtxt
        torch.save joblib.dump pandas.to_pickle pandas.read_csv pandas.read_table
        pandas.read_json pandas.read_parquet pandas.read_hdf pandas.read_excel
        h5py.File sqlite3.connect shelve.open dbm.open
    """,
    "network": """
        socket.create_connection socket.socket.connect socket.socket.connect_ex
        socket.socket.sendto socket.socket.sendmsg socket.getaddrinfo
        socket.gethostbyname socket.gethostbyname_ex socket.gethostbyaddr
        urllib.request.urlopen urllib.request.urlretrieve urllib.request.OpenerDirector.open
        http.client.HTTPConnection.connect http.client.HTTPConnection.request
        http.client.HTTPConnection.send http.client.HTTPConnection.endheaders
        http.client.HTTPSConnection.connect http.client.HTTPSConnection.request
        ftplib.FTP.connect ftplib.FTP_TLS.connect smtplib.SMTP.connect smtplib.SMTP_SSL.connect
        imaplib.IMAP4 imaplib.IMAP4_SSL poplib.POP3 poplib.POP3_SSL nntplib.NNTP
        telnetlib.Telnet.open ssl.get_server_certificate
        webbrowser.open webbrowser.open_new webbrowser.open_new_tab
        requests.get requests.post requests.put requests.delete requests.request
        requests.head requests.patch requests.options requests.api.request
        requests.Session.request requests.Session.get requests.Session.post
        httpx.get httpx.post httpx.request
    """,
}.items():
    CALL_EFFECTS.update(dict.fromkeys(_names.split(), _kind))
