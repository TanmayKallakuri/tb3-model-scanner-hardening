"""Capability catalogues used by the static interpreters.

A name in this catalogue is a capability only when the interpreter calls it.
The TensorFlow table, in contrast, is the closed op table in the registry policy.
"""
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

# Constructors with deferred effects are handled separately by the interpreter.
CALL_CAPABILITIES = {}
for _kind, _names in {
    'code-execution': '''builtins.eval builtins.exec builtins.compile
        runpy.run_path runpy.run_module runpy._run_code runpy._run_module_code
        timeit.timeit timeit.repeat timeit.Timer.timeit timeit.Timer.repeat timeit.Timer.autorange
        code.InteractiveInterpreter.runsource code.InteractiveInterpreter.runcode
        code.InteractiveConsole.push bdb.Bdb.run bdb.Bdb.runeval bdb.Bdb.runcall
        pdb.run pdb.runeval pdb.runcall pdb.Pdb.run pdb.Pdb.runeval pdb.Pdb.runcall
        numpy.testing._private.utils.runstring numpy.testing.utils.runstring
        numpy.f2py.compile types.FunctionType types.LambdaType types.CodeType
        importlib._bootstrap_external._LoaderBasics.exec_module
        importlib.machinery.SourceFileLoader.exec_module
        importlib.machinery.SourceFileLoader.load_module
        importlib.machinery.SourcelessFileLoader.exec_module
        importlib.machinery.SourcelessFileLoader.load_module
        imp.load_source imp.load_compiled imp.load_module
        doctest.run_docstring_examples doctest.testmod doctest.testfile
        site.execsitecustomize site.execusercustomize''',
    'process': '''os.system os.popen os.execv os.execve os.execvp os.execvpe os.execl
        os.execle os.execlp os.execlpe os.spawnl os.spawnle os.spawnlp os.spawnlpe
        os.spawnv os.spawnve os.spawnvp os.spawnvpe os.posix_spawn os.posix_spawnp
        os.fork os.forkpty os.startfile subprocess.Popen subprocess.call subprocess.run
        subprocess.check_call subprocess.check_output subprocess.getoutput
        subprocess.getstatusoutput subprocess._fork_exec _posixsubprocess.fork_exec
        pty.spawn pty.fork pydoc.pipepager pydoc.tempfilepager
        multiprocessing.Pool multiprocessing.pool.Pool multiprocessing.pool.ThreadPool
        multiprocessing.context.BaseContext.Pool''',
    'file-access': '''builtins.open builtins.input io.open io.open_code io.FileIO io.BufferedReader
        io.BufferedWriter io.BufferedRandom io.TextIOWrapper
        os.open os.fdopen os.read os.write os.readv os.writev os.pread os.pwrite
        os.remove os.unlink os.rename os.replace os.mkdir os.makedirs os.rmdir os.removedirs
        os.renames os.link os.symlink os.truncate os.chmod os.chown os.lchmod os.lchown
        os.utime os.mknod os.mkfifo os.chdir os.chroot os.listdir os.scandir os.walk os.fwalk
        os.stat os.lstat os.access os.readlink
        shutil.copy shutil.copy2 shutil.copyfile shutil.copyfileobj shutil.copymode
        shutil.copystat shutil.copytree shutil.move shutil.rmtree shutil.chown
        pathlib.Path.open pathlib.Path.read_text pathlib.Path.read_bytes
        pathlib.Path.write_text pathlib.Path.write_bytes pathlib.Path.touch pathlib.Path.unlink
        pathlib.Path.mkdir pathlib.Path.rmdir pathlib.Path.rename pathlib.Path.replace
        pathlib.Path.symlink_to pathlib.Path.hardlink_to pathlib.Path.link_to
        pathlib.Path.iterdir pathlib.Path.glob pathlib.Path.rglob pathlib.Path.stat
        pathlib.Path.lstat pathlib.Path.chmod pathlib.Path.lchmod pathlib.Path.owner
        pathlib.Path.group pathlib.Path.readlink pathlib.Path.exists pathlib.Path.is_file
        pathlib.Path.is_dir pathlib.Path.is_symlink
        tempfile.mkstemp tempfile.mkdtemp tempfile.NamedTemporaryFile tempfile.TemporaryFile
        tempfile.TemporaryDirectory tempfile.SpooledTemporaryFile
        glob.glob glob.iglob linecache.getline linecache.getlines fileinput.input
        logging.FileHandler logging.handlers.RotatingFileHandler
        logging.handlers.TimedRotatingFileHandler logging.handlers.WatchedFileHandler
        zipfile.ZipFile.extract zipfile.ZipFile.extractall tarfile.TarFile.extract
        tarfile.TarFile.extractall''',
    'network': '''socket.create_connection socket.socket.connect socket.socket.connect_ex
        socket.socket.sendto socket.socket.sendmsg socket.getaddrinfo socket.gethostbyname
        socket.gethostbyname_ex socket.gethostbyaddr urllib.request.urlopen
        urllib.request.urlretrieve urllib.request.OpenerDirector.open
        http.client.HTTPConnection.connect http.client.HTTPConnection.request
        http.client.HTTPConnection.endheaders http.client.HTTPConnection.send
        http.client.HTTPSConnection.connect http.client.HTTPSConnection.request
        http.client.HTTPSConnection.endheaders http.client.HTTPSConnection.send
        ftplib.FTP.connect ftplib.FTP.login ftplib.FTP_TLS.connect
        smtplib.SMTP.connect smtplib.SMTP.sendmail smtplib.SMTP.send_message
        smtplib.SMTP_SSL.connect ssl.get_server_certificate ssl.SSLSocket.connect
        ssl.SSLSocket.connect_ex webbrowser.open webbrowser.open_new webbrowser.open_new_tab
        xmlrpc.client._Method.__call__ xmlrpc.client.ServerProxy.__call__
        logging.handlers.SocketHandler.emit logging.handlers.DatagramHandler.emit
        logging.handlers.HTTPHandler.emit logging.handlers.SMTPHandler.emit''',
}.items():
    CALL_CAPABILITIES.update(dict.fromkeys(_names.split(), _kind))

# Required arguments for the common effectful callables. Bad calls stop the
# unpickler before the capability is reached. Entries are parameter names so
# functools.partial's keyword arguments obey the same rule as positional ones.
REQUIRED_ARGUMENTS = {
    'os.system': ('command',), 'os.popen': ('cmd',),
    'os.execv': ('path','argv'), 'os.execvp': ('file','args'),
    'os.execve': ('path','argv','env'), 'os.execvpe': ('file','args','env'),
    'os.spawnl': ('mode','file'), 'os.spawnv': ('mode','file','args'),
    'os.posix_spawn': ('path','argv','env'), 'os.posix_spawnp': ('path','argv','env'),
    'subprocess.Popen': ('args',),
    'subprocess.call': ('args',), 'subprocess.run': ('args',),
    'subprocess.check_call': ('args',), 'subprocess.check_output': ('args',),
    'subprocess.getoutput': ('cmd',), 'subprocess.getstatusoutput': ('cmd',),
    'builtins.eval': ('source',), 'builtins.exec': ('source',),
    'builtins.compile': ('source','filename','mode'),
    'builtins.open': ('file',), 'io.open': ('file',), 'io.FileIO': ('file',),
    'io.open_code': ('path',), 'os.open': ('path','flags'),
    'os.remove': ('path',), 'os.unlink': ('path',), 'os.mkdir': ('path',),
    'os.makedirs': ('name',), 'os.rmdir': ('path',), 'os.rename': ('src','dst'),
    'os.replace': ('src','dst'), 'shutil.copy': ('src','dst'),
    'shutil.copyfile': ('src','dst'), 'shutil.move': ('src','dst'),
    'shutil.rmtree': ('path',), 'runpy.run_path': ('path_name',),
    'runpy.run_module': ('mod_name',),
    'socket.create_connection': ('address',), 'socket.socket.connect': ('self','address'),
    'socket.socket.connect_ex': ('self','address'), 'urllib.request.urlopen': ('url',),
    'webbrowser.open': ('url',), 'webbrowser.open_new': ('url',),
    'webbrowser.open_new_tab': ('url',), 'ssl.get_server_certificate': ('addr',),
    'pty.spawn': ('argv',), 'pydoc.pipepager': ('text','cmd'),
    'http.client.HTTPConnection.request': ('self','method','url'),
    'http.client.HTTPSConnection.request': ('self','method','url'),
    'types.FunctionType': ('code','globals'), 'types.LambdaType': ('code','globals'),
    'types.CodeType': ('argcount','posonlyargcount','kwonlyargcount','nlocals','stacksize',
                       'flags','codestring','constants','names','varnames','filename','name',
                       'qualname','firstlineno','linetable','exceptiontable'),
}

for _names, _kind in [
    ('''numpy.save numpy.savez numpy.savez_compressed numpy.loadtxt numpy.genfromtxt
        numpy.savetxt numpy.fromfile numpy.memmap numpy.lib.format.open_memmap
        numpy.ndarray.tofile''', 'file-access'),
    ('''gzip.GzipFile gzip.open bz2.BZ2File bz2.open lzma.LZMAFile lzma.open
        zipfile.ZipFile tarfile.open tarfile.TarFile.open h5py.File dbm.open
        sqlite3.connect shelve.open shelve.DbfilenameShelf
        posixpath.exists posixpath.lexists posixpath.isfile posixpath.isdir
        posixpath.islink posixpath.ismount posixpath.samefile posixpath.getsize
        posixpath.getmtime posixpath.getatime posixpath.getctime pathlib.Path.resolve
        pathlib.Path.expanduser''', 'file-access'),
    ('''ctypes.util.find_library platform._syscmd_file platform._syscmd_ver
        platform._syscmd_uname distutils.spawn.spawn''','process'),
    ('''builtins.breakpoint ctypes.pythonapi.PyRun_SimpleString
        ctypes.pythonapi.PyRun_SimpleStringFlags ctypes.pythonapi.PyRun_String
        ctypes.pythonapi.PyRun_StringFlags numpy.ctypeslib.load_library''','code-execution'),
]:
    CALL_CAPABILITIES.update(dict.fromkeys(_names.split(),_kind))
# These send on a pre-existing connection; they do not connect a fresh client.
for _name in ('ftplib.FTP.login','smtplib.SMTP.sendmail','smtplib.SMTP.send_message'):
    CALL_CAPABILITIES.pop(_name,None)

CALL_CAPABILITIES['numpy.ndarray.dump']='file-access'
MAX_POSITIONAL_ARGUMENTS = {
    'os.system':1, 'os.popen':3, 'builtins.eval':3, 'builtins.exec':3,
    'builtins.compile':6, 'builtins.open':8, 'io.open':8, 'io.FileIO':4,
    'os.remove':1, 'os.unlink':1, 'os.rename':2, 'os.replace':2,
    'socket.create_connection':3, 'types.FunctionType':5, 'types.LambdaType':5,
}

CALL_CAPABILITIES.pop('multiprocessing.pool.ThreadPool',None)

CALL_CAPABILITIES['zoneinfo.ZoneInfo']='file-access'
REQUIRED_ARGUMENTS['zoneinfo.ZoneInfo']=('key',)
REQUIRED_ARGUMENTS['numpy.memmap']=('filename',)

for _name in ("numpy.lib._datasource.open", "numpy.lib._datasource.DataSource.open",
              "numpy.lib.npyio.DataSource.open", "numpy.lib._datasource.DataSource.exists",
              "numpy.lib.npyio.DataSource.exists", "numpy.lib._npyio_impl._savez",
              "numpy.lib.npyio._savez", "numpy._core.records.fromfile",
              "numpy.core.records.fromfile"):
    CALL_CAPABILITIES[_name]='file-access'
