"""Known standard-library operation identities and their effects when called.

The interpreter resolves aliases and indirection before consulting this table.
"""
import _compat_pickle

ALIASES = {
    "posix": "os", "nt": "os", "_pickle": "pickle", "_io": "io",
    "_operator": "operator", "_functools": "functools", "_socket": "socket",
    "__builtin__": "builtins", "copy_reg": "copyreg",
    "numpy.core": "numpy._core", "numpy.core.multiarray": "numpy._core.multiarray",
    "numpy.core.numeric": "numpy._core.numeric",
}


def canonical(module, name=None, protocol=4):
    if protocol < 3:
        if name is not None and (module, name) in _compat_pickle.NAME_MAPPING:
            module, name = _compat_pickle.NAME_MAPPING[module, name]
        else:
            module = _compat_pickle.IMPORT_MAPPING.get(module, module)
    module = ALIASES.get(module, module)
    return module if name is None else (module, name)


CAPABILITIES = {}
for _kind, _names in {
    "code-execution": '''builtins.eval builtins.exec builtins.compile
        runpy.run_path runpy.run_module runpy._run_code runpy._run_module_code
        timeit.timeit timeit.repeat timeit.Timer.timeit timeit.Timer.repeat timeit.Timer.autorange
        code.InteractiveInterpreter.runsource code.InteractiveInterpreter.runcode
        code.InteractiveConsole.push code.InteractiveConsole.runsource code.InteractiveConsole.runcode
        bdb.Bdb.run bdb.Bdb.runeval bdb.Bdb.runctx bdb.Bdb.runcall
        pdb.run pdb.runeval pdb.runctx pdb.runcall pdb.Pdb.run pdb.Pdb.runeval pdb.Pdb.runcall
        numpy.testing._private.utils.runstring numpy.testing.runstring
        cProfile.run cProfile.runctx profile.run profile.runctx
        cProfile.Profile.run cProfile.Profile.runctx profile.Profile.run profile.Profile.runctx
        doctest.run_docstring_examples doctest.testmod doctest.testfile
        ctypes.CDLL ctypes.PyDLL ctypes.WinDLL ctypes.OleDLL ctypes.cdll.LoadLibrary
        ctypes.pydll.LoadLibrary ctypes.windll.LoadLibrary importlib.machinery.SourceFileLoader.load_module
        importlib.machinery.SourcelessFileLoader.load_module''',
    "process": '''os.system os.popen os.execv os.execvp os.execve os.execvpe
        os.execl os.execlp os.execle os.execlpe os.spawnl os.spawnle os.spawnlp
        os.spawnlpe os.spawnv os.spawnve os.spawnvp os.spawnvpe os.posix_spawn
        os.posix_spawnp os.fork os.forkpty os.startfile
        subprocess.Popen subprocess.call subprocess.run subprocess.check_call subprocess.check_output
        subprocess.getoutput subprocess.getstatusoutput subprocess._fork_exec
        pty.spawn pydoc.pipepager pydoc.tempfilepager asyncio.create_subprocess_exec
        asyncio.create_subprocess_shell multiprocessing.Process.start
        multiprocessing.context.Process.start multiprocessing.pool.Pool
        multiprocessing.Pool multiprocessing.Manager multiprocessing.managers.BaseManager.start
        concurrent.futures.ProcessPoolExecutor.submit concurrent.futures.ProcessPoolExecutor.map''',
    "file-access": '''builtins.open io.open io.open_code io.FileIO io.BufferedReader io.BufferedWriter
        os.open os.fdopen os.remove os.unlink os.rename os.renames os.replace os.mkdir os.makedirs
        os.rmdir os.removedirs os.link os.symlink os.truncate os.chmod os.chown os.lchmod os.lchown
        os.utime os.stat os.lstat os.listdir os.scandir os.readlink os.access os.walk os.fwalk
        os.chroot os.chdir os.mkfifo os.mknod os.read os.write os.ftruncate
        os.sendfile os.copy_file_range os.getxattr os.setxattr os.removexattr os.listxattr
        shutil.copy shutil.copyfile shutil.copy2 shutil.copytree shutil.copymode shutil.copystat
        shutil.move shutil.rmtree shutil.chown shutil.make_archive shutil.unpack_archive
        pathlib.Path.open pathlib.Path.read_text pathlib.Path.read_bytes pathlib.Path.write_text
        pathlib.Path.write_bytes pathlib.Path.unlink pathlib.Path.rename pathlib.Path.replace
        pathlib.Path.mkdir pathlib.Path.rmdir pathlib.Path.touch pathlib.Path.symlink_to pathlib.Path.hardlink_to
        pathlib.Path.link_to pathlib.Path.stat pathlib.Path.lstat pathlib.Path.exists pathlib.Path.is_file
        pathlib.Path.is_dir pathlib.Path.is_symlink pathlib.Path.iterdir pathlib.Path.glob pathlib.Path.rglob
        pathlib.Path.chmod pathlib.Path.lchmod pathlib.Path.readlink pathlib.Path.resolve
        tempfile.TemporaryFile tempfile.NamedTemporaryFile tempfile.SpooledTemporaryFile
        tempfile.TemporaryDirectory tempfile.mkstemp tempfile.mkdtemp tempfile.mktemp
        glob.glob glob.iglob linecache.getline linecache.getlines
        tokenize.open fileinput.input fileinput.FileInput configparser.ConfigParser.read
        logging.FileHandler logging.handlers.RotatingFileHandler logging.handlers.TimedRotatingFileHandler
        logging.config.fileConfig sqlite3.connect dbm.open shelve.open
        gzip.open bz2.open lzma.open zipfile.ZipFile tarfile.open
        numpy.fromfile numpy.memmap numpy.save numpy.savez numpy.savez_compressed numpy.savetxt
        numpy.loadtxt numpy.genfromtxt pandas.read_csv pandas.read_json pandas.read_excel
        pandas.read_hdf pandas.read_feather pandas.read_parquet h5py.File''',
    "network": '''socket.create_connection socket.create_server socket.socket.connect socket.socket.connect_ex
        socket.socket.sendto socket.socket.sendmsg socket.socket.sendmsg_afalg
        socket.getaddrinfo socket.gethostbyname socket.gethostbyname_ex socket.gethostbyaddr
        urllib.request.urlopen urllib.request.urlretrieve urllib.request.OpenerDirector.open
        urllib.request.URLopener.open http.client.HTTPConnection.connect http.client.HTTPConnection.request
        http.client.HTTPConnection.send http.client.HTTPConnection.endheaders
        http.client.HTTPSConnection.connect http.client.HTTPSConnection.request
        http.client.HTTPSConnection.send http.client.HTTPSConnection.endheaders
        ftplib.FTP.connect ftplib.FTP.login ftplib.FTP.transfercmd ftplib.FTP_TLS.connect
        ftplib.FTP_TLS.login smtplib.SMTP.connect smtplib.SMTP_SSL.connect smtplib.LMTP.connect
        ssl.get_server_certificate ssl.wrap_socket ssl.SSLContext.wrap_socket
        webbrowser.open webbrowser.open_new webbrowser.open_new_tab
        requests.get requests.post requests.put requests.delete requests.patch requests.head requests.request
        requests.Session.request requests.sessions.Session.request
        xmlrpc.client.ServerProxy.__call__ logging.handlers.SocketHandler.createSocket
        logging.handlers.SocketHandler.emit logging.handlers.DatagramHandler.emit
        asyncio.open_connection asyncio.open_unix_connection''',
}.items():
    CAPABILITIES.update(dict.fromkeys(_names.split(), _kind))

# Class constructors with no effect. Their methods may have effects later.
CLIENTS = {
    "socket.socket", "http.client.HTTPConnection", "http.client.HTTPSConnection",
    "ftplib.FTP", "ftplib.FTP_TLS", "smtplib.SMTP", "smtplib.SMTP_SSL", "smtplib.LMTP",
    "xmlrpc.client.ServerProxy", "urllib.request.Request", "urllib.request.OpenerDirector",
    "urllib.request.URLopener", "ssl.SSLContext", "requests.Session", "requests.sessions.Session",
    "logging.handlers.SocketHandler", "logging.handlers.DatagramHandler",
}
CONNECT_ON_HOST = {"ftplib.FTP", "ftplib.FTP_TLS", "smtplib.SMTP", "smtplib.SMTP_SSL", "smtplib.LMTP"}
CLASS_NAMES = CLIENTS | {
    "builtins.object", "builtins.type", "builtins.list", "builtins.dict", "builtins.tuple",
    "builtins.set", "builtins.frozenset", "builtins.str", "builtins.bytes", "builtins.bytearray",
    "builtins.int", "builtins.float", "builtins.bool", "builtins.complex", "builtins.range",
    "builtins.slice", "builtins.map", "builtins.filter", "builtins.zip", "builtins.enumerate",
    "builtins.staticmethod", "builtins.classmethod", "functools.partial", "types.FunctionType",
    "types.CodeType", "types.MethodType", "types.ModuleType", "types.SimpleNamespace",
    "code.InteractiveInterpreter", "code.InteractiveConsole", "bdb.Bdb", "pdb.Pdb", "timeit.Timer",
    "threading.Thread", "threading.Timer", "multiprocessing.Process", "multiprocessing.context.Process",
    "concurrent.futures.ThreadPoolExecutor", "concurrent.futures.ProcessPoolExecutor",
    "pathlib.Path", "pathlib.PosixPath", "pathlib.WindowsPath", "pathlib.PurePath",
    "pathlib.PurePosixPath", "pathlib.PureWindowsPath", "io.BytesIO", "io.StringIO", "io.FileIO",
    "subprocess.Popen", "numpy.ndarray", "numpy.dtype", "collections.OrderedDict", "collections.defaultdict",
    "collections.deque", "collections.Counter", "logging.FileHandler", "zipfile.ZipFile",
}

# These module attributes are modules, rather than unrelated globals with the
# same spelling. Additional identities come from explicit imports and BUILD.
MODULE_ATTRS = {
    "shutil": {"os": "os", "sys": "sys", "stat": "stat", "collections": "collections"},
    "pathlib": {"os": "os", "sys": "sys", "io": "io", "functools": "functools"},
    "subprocess": {"os": "os", "sys": "sys", "io": "io", "builtins": "builtins"},
    "random": {"_os": "os"}, "tempfile": {"_os": "os", "_io": "io"},
    "posixpath": {"os": "os", "sys": "sys"}, "ntpath": {"os": "os", "sys": "sys"},
    "os": {"path": "posixpath", "sys": "sys"},
    "socket": {"os": "os", "sys": "sys", "io": "io"},
    "pkgutil": {"os": "os", "sys": "sys", "importlib": "importlib"},
    "platform": {"os": "os", "sys": "sys", "subprocess": "subprocess"},
    "pickle": {"sys": "sys", "io": "io", "re": "re", "_compat_pickle": "_compat_pickle"},
    "copy": {"types": "types"}, "inspect": {"os": "os", "sys": "sys", "types": "types"},
    "pydoc": {"os": "os", "sys": "sys", "inspect": "inspect", "subprocess": "subprocess"},
    "webbrowser": {"os": "os", "sys": "sys", "subprocess": "subprocess"},
    "http.client": {"socket": "socket", "ssl": "ssl", "io": "io"},
    "urllib.request": {"os": "os", "sys": "sys", "http": "http", "socket": "socket"},
    "numpy": {"lib": "numpy.lib", "core": "numpy._core", "_core": "numpy._core", "testing": "numpy.testing"},
}

MODULE_NAMES = {
    "builtins", "os", "sys", "io", "pickle", "torch", "torch.storage", "numpy", "numpy.lib",
    "numpy.lib.format", "joblib", "pandas", "importlib", "importlib.util", "importlib.machinery",
    "http", "http.client", "urllib", "urllib.request", "xmlrpc", "xmlrpc.client", "logging", "logging.config",
    "logging.handlers", "multiprocessing", "multiprocessing.pool", "multiprocessing.context",
    "concurrent", "concurrent.futures", "numpy.testing", "numpy._core", "numpy._core.multiarray",
}
