"""Knowledge base: which callables carry a capability of the policy.

Keys are canonical names, ``<module>.<qualname>`` of the object the name
resolves to, with module aliases normalised (``posix``/``nt`` -> ``os``,
``_io`` -> ``io``, ``_socket`` -> ``socket`` ...; see ``MODULE_ALIASES``).
Methods are keyed by ``<module>.<Class>.<method>`` and receive ``self`` as
their first argument.  A value is ``(kind, condition)``; ``condition`` is
``None`` (always) or a function ``(args, kwargs) -> bool`` deciding whether
the call reaches the capability.

The tables cover the policy's seed list and the other standard-library and
common third-party callables with the same capabilities.
"""
from __future__ import annotations

CODE = "code-execution"
PROC = "process"
FILE = "file-access"
NET = "network"

# module name normalisation applied to canonical names and to imports
MODULE_ALIASES = {
    "posix": "os",
    "nt": "os",
    "posixpath": "os.path",
    "ntpath": "os.path",
    "_io": "io",
    "_pyio": "io",
    "_pickle": "pickle",
    "_socket": "socket",
    "_operator": "operator",
    "_functools": "functools",
    "_collections": "collections",
    "_collections_abc": "collections.abc",
    "_codecs": "codecs",
    "_ssl": "ssl",
    "_sqlite3": "sqlite3",
    "_ctypes": "ctypes",
    "_json": "json",
    "_heapq": "heapq",
    "_bisect": "bisect",
    "_struct": "struct",
    "_random": "random",
    "_datetime": "datetime",
    "_weakref": "weakref",
    "_signal": "signal",
    "_locale": "locale",
    "_string": "string",
    "_bz2": "bz2",
    "_lzma": "lzma",
    "_thread": "_thread",
    "_asyncio": "asyncio",
    "_compression": "_compression",
    "_frozen_importlib": "importlib._bootstrap",
    "_frozen_importlib_external": "importlib._bootstrap_external",
}

# Modules whose names, when they appear as an attribute of a module the
# scanner does not import (``torch.os``), are taken to be that module.
COMMON_MODULE_ATTRS = {
    "os", "sys", "subprocess", "shutil", "socket", "builtins", "pickle", "io",
    "importlib", "pathlib", "tempfile", "marshal", "types", "functools",
    "operator", "runpy", "code", "pty", "platform", "ctypes", "urllib",
    "http", "ftplib", "smtplib", "webbrowser", "codecs", "base64", "zlib",
    "json", "logging", "threading", "multiprocessing", "collections",
    "itertools", "inspect", "warnings", "glob", "copy", "copyreg", "pydoc",
    "timeit", "pdb", "bdb", "posix", "nt", "signal", "asyncio", "ssl",
}


def _arg(args, kwargs, idx, name, default=None):
    if len(args) > idx:
        return args[idx]
    return kwargs.get(name, default)


def _truthy_arg(idx, name, default=None):
    def cond(args, kwargs):
        v = _arg(args, kwargs, idx, name, default)
        if isinstance(v, (str, bytes)) or v is None:
            return bool(v)
        return True  # symbolic: assume a host was given
    return cond


def _not_none_arg(idx, name):
    def cond(args, kwargs):
        return _arg(args, kwargs, idx, name, None) is not None
    return cond


def _path_arg(idx, name):
    """A path-like argument (not a file object / in-memory buffer)."""
    from modelgate.symbolic import FileObj

    def cond(args, kwargs):
        v = _arg(args, kwargs, idx, name, None)
        if v is None:
            return False
        if isinstance(v, FileObj) and v.path is None:
            return False
        return True
    return cond


def _sqlite_path(args, kwargs):
    v = _arg(args, kwargs, 0, "database", None)
    if isinstance(v, str) and (v == ":memory:" or v.startswith("file::memory:") or v == ""):
        return False
    if isinstance(v, bytes) and v in (b":memory:", b""):
        return False
    return True


def _not_delay(args, kwargs):
    # logging.FileHandler(filename, mode='a', encoding=None, delay=False, errors=None)
    v = _arg(args, kwargs, 3, "delay", False)
    return not (v is True or v == 1)


def _not_delay_rot(idx):
    def cond(args, kwargs):
        v = _arg(args, kwargs, idx, "delay", False)
        return not (v is True or v == 1)
    return cond


# --------------------------------------------------------------------------
# functions and methods: calling them performs the capability
CAPS: dict[str, tuple] = {}


def _add(kind, names, cond=None):
    for n in names.split():
        CAPS[n] = (kind, cond)


# (B) process
_add(PROC, """
    os.system os.popen os.execl os.execle os.execlp os.execlpe os.execv os.execve
    os.execvp os.execvpe os._execvpe os.spawnl os.spawnle os.spawnlp os.spawnlpe
    os.spawnv os.spawnve os.spawnvp os.spawnvpe os._spawnvef os.posix_spawn
    os.posix_spawnp os.fork os.forkpty os.startfile
    subprocess.call subprocess.check_call subprocess.check_output subprocess.run
    subprocess.getoutput subprocess.getstatusoutput _posixsubprocess.fork_exec
    pty.spawn pty.fork pydoc.pipepager pydoc.tempfilepager
    ctypes.util.find_library ctypes.util._findLib_gcc ctypes.util._findSoname_ldconfig
    ctypes.util._findLib_ld ctypes.util._get_soname
    platform._syscmd_file platform._syscmd_ver
    multiprocessing.process.BaseProcess.start
    multiprocessing.context.BaseContext.Pool multiprocessing.context.BaseContext.Manager
    multiprocessing.managers.BaseManager.start multiprocessing.managers.BaseManager.get_server
    concurrent.futures.process.ProcessPoolExecutor.submit
    concurrent.futures.process.ProcessPoolExecutor.map
    asyncio.subprocess.create_subprocess_shell asyncio.subprocess.create_subprocess_exec
    uuid._popen uuid._ifconfig_getnode uuid._ip_getnode uuid._arp_getnode
    uuid._lanscan_getnode uuid._netstat_getnode
    numpy.f2py.run_main numpy.f2py.f2py2e.run_main numpy.f2py.compile
    numpy.distutils.exec_command.exec_command
    os.popen2 os.popen3 os.popen4 commands.getoutput commands.getstatusoutput
""")

# (A) code execution: evaluate / compile source, materialise bytecode
_add(CODE, """
    builtins.eval builtins.exec builtins.compile builtins.execfile
    builtins.function
    runpy.run_path runpy.run_module runpy._run_code runpy._run_module_code
    runpy._run_module_as_main
    timeit.timeit timeit.repeat timeit.main timeit.Timer.timeit timeit.Timer.repeat
    timeit.Timer.autorange
    code.InteractiveInterpreter.runsource code.InteractiveInterpreter.runcode
    code.InteractiveConsole.push code.InteractiveConsole.interact code.interact
    code.compile_command codeop.compile_command codeop.Compile.__call__
    codeop.CommandCompiler.__call__ codeop._maybe_compile
    bdb.Bdb.run bdb.Bdb.runeval bdb.Bdb.runctx
    pdb.run pdb.runeval pdb.runctx pdb.main pdb.Pdb._runscript pdb.Pdb._run
    pdb.Pdb.default pdb.Pdb.onecmd
    cProfile.run cProfile.runctx cProfile.Profile.run cProfile.Profile.runctx
    cProfile.main
    profile.run profile.runctx profile.Profile.run profile.Profile.runctx profile.main
    trace.Trace.run trace.Trace.runctx trace.main
    doctest.run_docstring_examples doctest.debug_src doctest.debug_script
    doctest.debug doctest.testmod doctest.testfile doctest.DocTestRunner.run
    doctest.DebugRunner.run doctest.DocTestCase.runTest doctest.DocTestCase.debug
    doctest._test
    numpy.testing._private.utils.runstring numpy.testing.utils.runstring
    numpy.testing.runstring
    logging.config.dictConfig logging.config.fileConfig logging.config.listen
    logging.config.DictConfigurator.configure logging.config.valid_ident
    importlib._bootstrap_external.SourceLoader.exec_module
    importlib._bootstrap_external._LoaderBasics.exec_module
    importlib._bootstrap_external.FileLoader.load_module
    importlib._bootstrap_external.SourceFileLoader.load_module
    importlib._bootstrap_external.SourcelessFileLoader.load_module
    importlib._bootstrap_external.SourceLoader.source_to_code
    importlib.abc.InspectLoader.source_to_code importlib.abc.InspectLoader.exec_module
    importlib._bootstrap._load importlib._bootstrap._exec
    importlib._bootstrap_external._compile_bytecode
    zipimport.zipimporter.load_module zipimport.zipimporter.exec_module
    site.addsitedir site.addpackage site.main site.execsitecustomize
    site.execusercustomize
    ctypes.CDLL ctypes.PyDLL ctypes.LibraryLoader.LoadLibrary ctypes.LibraryLoader.__getattr__
    ctypes.LibraryLoader.__getitem__ ctypes.dlopen ctypes._dlopen ctypes.CDLL.__init__
    numpy.ctypeslib.load_library numpy.ctypeslib._ctypeslib.load_library
    unittest.main unittest.main.TestProgram
    ensurepip.bootstrap ensurepip._bootstrap ensurepip._run_pip
    keras.src.utils.python_utils.func_load keras.utils.python_utils.func_load
    tensorflow.python.keras.utils.generic_utils.func_load
    keras.utils.generic_utils.func_load
    dill._dill._create_function dill._dill._create_code dill._dill._load_type
    torch.jit.load torch.jit._serialization.load torch.package.PackageImporter
    torch.hub.load torch.hub._load_local
    yaml.unsafe_load yaml.full_load yaml.load_all yaml.unsafe_load_all
    jsonpickle.decode jsonpickle.unpickler.decode
""")

# (C) file access
_add(FILE, """
    io.open io.open_code io.FileIO codecs.open builtins.open builtins.file
    os.open os.remove os.unlink os.rename os.renames os.replace os.mkdir os.makedirs
    os.rmdir os.removedirs os.link os.symlink os.truncate os.chmod os.lchmod os.chown
    os.lchown os.utime os.mkfifo os.mknod os.setxattr os.removexattr os.chroot
    shutil.copy shutil.copy2 shutil.copyfile shutil.copymode shutil.copystat
    shutil.copytree shutil.move shutil.rmtree shutil.chown shutil.make_archive
    shutil.unpack_archive shutil._rmtree_unsafe shutil._rmtree_safe_fd
    shutil._make_tarball shutil._make_zipfile shutil._unpack_zipfile
    shutil._unpack_tarfile shutil._copytree shutil._copyxattr
    pathlib.Path.write_text pathlib.Path.write_bytes pathlib.Path.read_text
    pathlib.Path.read_bytes pathlib.Path.open pathlib.Path.touch pathlib.Path.mkdir
    pathlib.Path.rmdir pathlib.Path.unlink pathlib.Path.rename pathlib.Path.replace
    pathlib.Path.symlink_to pathlib.Path.hardlink_to pathlib.Path.chmod
    pathlib.Path.lchmod pathlib.Path.link_to
    tempfile.mkstemp tempfile.mkdtemp tempfile.NamedTemporaryFile tempfile.TemporaryFile
    tempfile._mkstemp_inner tempfile.TemporaryDirectory.__init__
    linecache.getline linecache.getlines linecache.updatecache
    fileinput.FileInput.__next__ fileinput.FileInput._readline fileinput.FileInput.readline
    shelve.open dbm.open dbm.dumb.open dbm.gnu.open dbm.ndbm.open
    mailbox.mbox mailbox.Maildir mailbox.MH mailbox.Babyl mailbox.MMDF
    py_compile.compile compileall.compile_dir compileall.compile_file compileall.compile_path
    zipapp.create_archive filecmp.cmp filecmp.cmpfiles
    netrc.netrc
    configparser.RawConfigParser.read
    pkgutil.get_data
    zipfile.ZipFile.extractall zipfile.ZipFile.extract zipfile.ZipFile.write
    zipfile.ZipFile._extract_member zipfile.Path.read_text zipfile.Path.read_bytes
    tarfile.TarFile.extractall tarfile.TarFile.extract tarfile.TarFile.add
    tarfile.TarFile._extract_member tarfile.TarFile.makefile
    importlib._bootstrap_external.FileLoader.get_data
    importlib.resources.read_text importlib.resources.read_binary
    venv.create venv.EnvBuilder.create
    http.cookiejar.FileCookieJar.load http.cookiejar.FileCookieJar.save
    http.cookiejar.FileCookieJar.revert
    xml.etree.ElementTree.ElementTree.write
    numpy.save numpy.savez numpy.savez_compressed numpy.savetxt numpy.loadtxt
    numpy.genfromtxt numpy.fromfile numpy.ndarray.tofile numpy.ndarray.dump
    numpy.lib.format.open_memmap numpy.lib.npyio.DataSource.open numpy.DataSource.open
    numpy.lib._datasource.DataSource.open numpy.lib._datasource.open
    numpy.core.records.fromfile numpy.rec.fromfile numpy._core.records.fromfile
    torch.save torch.serialization.save torch.from_file torch.UntypedStorage.from_file
    torch.storage.UntypedStorage.from_file torch.TypedStorage.from_file
    torch.storage.TypedStorage.from_file
    joblib.dump joblib.numpy_pickle.dump
    pandas.read_csv pandas.read_table pandas.read_json pandas.read_excel
    pandas.read_parquet pandas.read_feather pandas.read_hdf pandas.read_sql
    pandas.io.parsers.readers.read_csv pandas.DataFrame.to_pickle pandas.to_pickle
    pandas.io.pickle.to_pickle pandas.DataFrame.to_csv
""")

# conditional file access: a path argument (not an in-memory buffer)
for _n, _i, _p in (
    ("zipfile.ZipFile", 0, "file"),
    ("tarfile.open", 0, "name"),
    ("tarfile.TarFile.open", 1, "name"),
    ("tarfile.TarFile", 0, "name"),
    ("gzip.open", 0, "filename"),
    ("gzip.GzipFile", 0, "filename"),
    ("bz2.open", 0, "filename"),
    ("bz2.BZ2File", 0, "filename"),
    ("lzma.open", 0, "filename"),
    ("lzma.LZMAFile", 0, "filename"),
    ("wave.open", 0, "f"),
    ("aifc.open", 0, "f"),
    ("sunau.open", 0, "f"),
    ("xml.etree.ElementTree.parse", 0, "source"),
    ("xml.etree.ElementTree.iterparse", 0, "source"),
    ("xml.etree.ElementTree.ElementTree.parse", 1, "source"),
    ("xml.dom.minidom.parse", 0, "file"),
    ("xml.dom.pulldom.parse", 0, "stream_or_string"),
    ("xml.sax.parse", 0, "source"),
    ("plistlib.load", 0, "fp"),
    ("mimetypes.read_mime_types", 0, "file"),
    ("numpy.memmap", 0, "filename"),
):
    CAPS[_n] = (FILE, _path_arg(_i, _p))
CAPS["sqlite3.connect"] = (FILE, _sqlite_path)
CAPS["logging.FileHandler"] = (FILE, _not_delay)
CAPS["logging.handlers.RotatingFileHandler"] = (FILE, _not_delay_rot(5))
CAPS["logging.handlers.TimedRotatingFileHandler"] = (FILE, _not_delay_rot(5))
CAPS["logging.handlers.WatchedFileHandler"] = (FILE, _not_delay_rot(3))
CAPS["logging.handlers.BaseRotatingHandler"] = (FILE, _not_delay_rot(3))
CAPS["logging.basicConfig"] = (FILE, lambda a, k: "filename" in k)

# (D) network
_add(NET, """
    socket.create_connection socket.create_server socket.socket.connect
    socket.socket.connect_ex socket.socket.sendto socket.socket.sendmsg
    socket.socket.bind socket.getaddrinfo socket.gethostbyname socket.gethostbyname_ex
    socket.gethostbyaddr socket.getnameinfo socket.getfqdn
    ssl.get_server_certificate ssl.SSLSocket.connect ssl.SSLSocket.connect_ex
    ssl.SSLSocket._real_connect
    urllib.request.urlopen urllib.request.urlretrieve urllib.request.OpenerDirector.open
    urllib.request.URLopener.open urllib.request.URLopener.retrieve
    urllib.request.URLopener.open_http urllib.request.URLopener.open_https
    urllib.request.URLopener.open_ftp urllib.request.URLopener.open_file
    urllib.request.URLopener.open_local_file
    urllib.request.AbstractHTTPHandler.do_open urllib.request.HTTPHandler.http_open
    urllib.request.HTTPSHandler.https_open urllib.request.FTPHandler.ftp_open
    urllib.request.FileHandler.file_open urllib.request.FileHandler.open_local_file
    urllib.robotparser.RobotFileParser.read
    http.client.HTTPConnection.connect http.client.HTTPConnection.request
    http.client.HTTPConnection.send http.client.HTTPConnection.endheaders
    http.client.HTTPConnection._send_output http.client.HTTPConnection._send_request
    http.client.HTTPSConnection.connect
    ftplib.FTP.connect ftplib.FTP_TLS.connect smtplib.SMTP.connect
    poplib.POP3._create_socket imaplib.IMAP4.open imaplib.IMAP4._create_socket
    imaplib.IMAP4_SSL.open imaplib.IMAP4_SSL._create_socket
    nntplib.NNTP._create_socket telnetlib.Telnet.open
    webbrowser.open webbrowser.open_new webbrowser.open_new_tab
    webbrowser.BaseBrowser.open webbrowser.BaseBrowser.open_new
    webbrowser.BaseBrowser.open_new_tab webbrowser.GenericBrowser.open
    webbrowser.BackgroundBrowser.open webbrowser.UnixBrowser.open
    webbrowser.UnixBrowser._invoke webbrowser.Konqueror.open webbrowser.Edge.open
    webbrowser.Chrome.open webbrowser.Chromium.open webbrowser.Opera.open
    webbrowser.Elinks.open webbrowser.Mozilla.open webbrowser.Epiphany.open
    socketserver.TCPServer.server_bind socketserver.BaseServer.serve_forever
    socketserver.TCPServer.server_activate http.server.test http.server.HTTPServer.server_bind
    wsgiref.simple_server.make_server
    multiprocessing.connection.Client multiprocessing.connection.Listener
    multiprocessing.connection.SocketClient multiprocessing.connection.SocketListener
    xmlrpc.client.ServerProxy._ServerProxy__request xmlrpc.client.Transport.request
    xmlrpc.client.Transport.single_request xmlrpc.client.Transport.make_connection
    xmlrpc.client._Method.__call__
    asyncio.streams.open_connection asyncio.streams.start_server
    asyncio.base_events.BaseEventLoop.create_connection
    torch.hub.download_url_to_file torch.hub.load_state_dict_from_url
    torch.utils.model_zoo.load_url keras.utils.get_file tensorflow.keras.utils.get_file
    requests.get requests.post requests.put requests.delete requests.head requests.request
    requests.api.get requests.api.post requests.api.request requests.Session.request
    urllib3.PoolManager.request urllib3.request
""")

# constructors that connect when given a host
for _n in ("ftplib.FTP", "ftplib.FTP_TLS", "smtplib.SMTP", "smtplib.SMTP_SSL", "smtplib.LMTP"):
    CAPS[_n] = (NET, _truthy_arg(0, "host", ""))
for _n in ("poplib.POP3", "poplib.POP3_SSL", "nntplib.NNTP", "nntplib.NNTP_SSL",
           "imaplib.IMAP4", "imaplib.IMAP4_SSL",
           "socketserver.TCPServer", "socketserver.UDPServer", "socketserver.UnixStreamServer",
           "socketserver.UnixDatagramServer", "socketserver.ThreadingTCPServer",
           "socketserver.ForkingTCPServer", "http.server.HTTPServer",
           "http.server.ThreadingHTTPServer", "xmlrpc.server.SimpleXMLRPCServer",
           "xmlrpc.server.DocXMLRPCServer", "xmlrpc.server.MultiPathXMLRPCServer"):
    CAPS[_n] = (NET, None)
CAPS["telnetlib.Telnet"] = (NET, _not_none_arg(0, "host"))
CAPS["imaplib.IMAP4_stream"] = (PROC, None)
CAPS["subprocess.Popen"] = (PROC, None)
CAPS["asyncio.subprocess.Process"] = (PROC, None)
CAPS["multiprocessing.pool.Pool"] = (PROC, None)
CAPS["multiprocessing.popen_fork.Popen"] = (PROC, None)
CAPS["multiprocessing.popen_spawn_posix.Popen"] = (PROC, None)
CAPS["multiprocessing.popen_forkserver.Popen"] = (PROC, None)
CAPS["multiprocessing.managers.SyncManager.start"] = (PROC, None)
CAPS["timeit.Timer"] = (CODE, lambda a, k: isinstance(_arg(a, k, 0, "stmt", "pass"), str)
                        or isinstance(_arg(a, k, 1, "setup", "pass"), str))

# module-level fallbacks: any other public callable of these modules is
# treated as carrying the capability unless it is listed as harmless below
MODULE_DEFAULTS = {
    "_posixsubprocess": PROC,
    "os": None,  # decided by OS_* tables below
}

# os functions that carry no capability of the policy
OS_HARMLESS = set("""
    WCOREDUMP WEXITSTATUS WIFCONTINUED WIFEXITED WIFSIGNALED WIFSTOPPED WSTOPSIG WTERMSIG
    _check_methods _exists _fspath _get_exports_list abort access close closerange
    confstr cpu_count ctermid device_encoding dup dup2 eventfd eventfd_read eventfd_write
    fchdir fpathconf fsdecode fsencode fspath fstat fstatvfs fsync fdatasync get_blocking
    get_exec_path get_inheritable get_terminal_size getcwd getcwdb getegid getenv getenvb
    geteuid getgid getgrouplist getgroups getloadavg getlogin getpgid getpgrp getpid
    getppid getpriority getrandom getresgid getresuid getsid getuid getxattr isatty
    kill killpg listdir listxattr lseek lstat major makedev minor nice pathconf
    pidfd_open pipe pipe2 putenv read readlink readv register_at_fork scandir
    sched_get_priority_max sched_get_priority_min sched_getaffinity sched_getparam
    sched_getscheduler sched_rr_get_interval sched_yield set_blocking set_inheritable
    stat statvfs strerror sysconf tcgetpgrp times ttyname umask uname unsetenv urandom
    wait wait3 wait4 waitid waitpid waitstatus_to_exitcode walk fwalk _fwalk
    _walk memfd_create openpty login_tty sync getenv fdopen write writev pread pwrite
    preadv pwritev sendfile splice copy_file_range lockf posix_fadvise posix_fallocate
    fchmod fchown ftruncate chdir setegid seteuid setgid setgroups setpgid setpgrp
    setpriority setregid setresgid setresuid setreuid setsid setuid setns unshare
    sched_setaffinity sched_setparam sched_setscheduler tcsetpgrp initgroups
    _exit dup _wrap_close PathLike DirEntry stat_result statvfs_result terminal_size
    times_result uname_result sched_param error Mapping MutableMapping abc sys st
    GenericAlias fsencode
""".split())

# os.path is pure apart from these
OSPATH_HARMLESS_ALL = True

# --------------------------------------------------------------------------
# third-party names (modules the scanner does not import) normalised for
# lookup: prefixes that are re-exports of the same objects
UNRES_PREFIX_ALIASES = (
    ("numpy.core.", "numpy._core."),
    ("torch.serialization.load", "torch.load"),
    ("joblib.numpy_pickle.load", "joblib.load"),
    ("sklearn.externals.joblib.", "joblib."),
    ("pandas.io.pickle.read_pickle", "pandas.read_pickle"),
    ("pandas.core.frame.DataFrame", "pandas.DataFrame"),
    ("numpy.lib.npyio.load", "numpy.load"),
    ("numpy.lib._npyio_impl.load", "numpy.load"),
    ("numpy.lib.format.read_array", "numpy.lib.format.read_array"),
    ("numpy.lib._format_impl.read_array", "numpy.lib.format.read_array"),
    ("numpy.lib.format.open_memmap", "numpy.lib.format.open_memmap"),
)

# third-party nested loaders: handled by the interpreter (re-entrant
# deserialization), keyed by normalised name -> loader kind
NESTED_LOADERS = {
    "pickle.loads": "pickle-bytes",
    "pickle._loads": "pickle-bytes",
    "pickle.load": "pickle-file",
    "pickle._load": "pickle-file",
    "cPickle.loads": "pickle-bytes",
    "dill.loads": "pickle-bytes",
    "dill._dill.loads": "pickle-bytes",
    "dill.load": "pickle-file",
    "dill._dill.load": "pickle-file",
    "cloudpickle.loads": "pickle-bytes",
    "cloudpickle.load": "pickle-file",
    "cloudpickle.cloudpickle.loads": "pickle-bytes",
    "torch.load": "torch",
    "torch.storage._load_from_bytes": "torch-bytes",
    "numpy.load": "numpy",
    "numpy.lib.format.read_array": "numpy-read-array",
    "joblib.load": "joblib",
    "pandas.read_pickle": "pandas",
    "shelve.Shelf.__getitem__": "none",
}


# argument types without which the call raises TypeError before acting
_SEQ = (str, bytes, bytearray, list, tuple, set, frozenset, dict, range)
ARG_CHECKS = {
    "os.system": (0, "command", (str, bytes)),
    "os.popen": (0, "cmd", (str,)),
    "os.execv": (0, "path", (str, bytes)),
    "os.execve": (0, "path", (str, bytes, int)),
    "os.execl": (0, "file", (str, bytes)),
    "os.execlp": (0, "file", (str, bytes)),
    "os.execvp": (0, "file", (str, bytes)),
    "os.spawnl": (0, "mode", (int,)),
    "os.spawnv": (0, "mode", (int,)),
    "urllib.request.urlopen": (0, "url", (str,)),
    "webbrowser.open": (0, "url", (str,)),
    "socket.create_connection": (0, "address", (tuple, list)),
}


# functions returning instances of known classes, so that methods called on
# the result are modelled: canonical function name -> canonical class name
RETURNS = {
    "ssl.create_default_context": "ssl.SSLContext",
    "ssl._create_unverified_context": "ssl.SSLContext",
    "ssl._create_default_https_context": "ssl.SSLContext",
    "ssl.SSLContext.wrap_socket": "ssl.SSLSocket",
    "urllib.request.build_opener": "urllib.request.OpenerDirector",
    "asyncio.events.new_event_loop": "asyncio.base_events.BaseEventLoop",
    "asyncio.events.get_event_loop": "asyncio.base_events.BaseEventLoop",
    "socket.fromfd": "socket.socket",
    "socket.socket.dup": "socket.socket",
    "logging.getLogger": "logging.Logger",
    "multiprocessing.get_context": "multiprocessing.context.BaseContext",
    "multiprocessing.context.BaseContext.get_context": "multiprocessing.context.BaseContext",
    "pickle.Unpickler": "pickle.Unpickler",
    "telnetlib.Telnet": "telnetlib.Telnet",
    "zipfile.ZipFile.open": "io.BytesIO",
}

# distinctive names of capability callables: a third-party module attribute
# with this name is a re-export of the standard one (``torch.hub.urlopen``)
REEXPORTED = {
    "urlopen": "urllib.request.urlopen", "urlretrieve": "urllib.request.urlretrieve",
    "check_output": "subprocess.check_output", "check_call": "subprocess.check_call",
    "getoutput": "subprocess.getoutput", "getstatusoutput": "subprocess.getstatusoutput",
    "Popen": "subprocess.Popen", "execv": "os.execv", "execve": "os.execve", "execvp": "os.execvp",
    "execl": "os.execl", "execlp": "os.execlp", "spawnv": "os.spawnv", "spawnl": "os.spawnl",
    "posix_spawn": "os.posix_spawn", "posix_spawnp": "os.posix_spawnp",
    "create_connection": "socket.create_connection", "rmtree": "shutil.rmtree",
}

# third-party classes whose __new__ (so also NEWOBJ) performs the capability
NEW_CAPS = {
    "numpy.memmap": (FILE, _path_arg(0, "filename")),
    "numpy._core.memmap.memmap": (FILE, _path_arg(0, "filename")),
    "numpy.core.memmap.memmap": (FILE, _path_arg(0, "filename")),
}



def _popen_args_ok(args, kwargs):
    """subprocess.Popen(args, bufsize, executable, stdin, stdout, stderr, ...)
    raises before starting anything for these argument errors."""
    from modelgate.symbolic import Sym
    if not args and "args" not in kwargs:
        return False
    from modelgate.symbolic import Real
    first = args[0] if args else kwargs.get("args")
    if isinstance(first, Real):
        return False  # a function, class or module is not an argument list
    plain = lambda v: not isinstance(v, Sym)  # noqa: E731
    if len(args) > 1 and plain(args[1]) and not (isinstance(args[1], int) and not isinstance(args[1], bool)
                                                 or args[1] is True or args[1] is False):
        return False
    if len(args) > 2 and plain(args[2]) and not isinstance(args[2], (str, bytes, type(None))):
        return False
    for v in args[3:6]:
        if plain(v) and not isinstance(v, (int, type(None))):
            return False
    return True


# callables that raise (TypeError/ValueError) before acting on these inputs
ARG_VALIDATORS = {
    "subprocess.Popen": _popen_args_ok,
    "subprocess.call": _popen_args_ok,
    "subprocess.check_call": _popen_args_ok,
    "subprocess.check_output": _popen_args_ok,
    "subprocess.run": _popen_args_ok,
}



def _exec_argv_ok(idx):
    """os.execv*/execl*: an empty argument list raises ValueError first."""
    def check(args, kwargs):
        from modelgate.symbolic import Sym
        if idx is None:  # execl(file, *args)
            return len(args) >= 2
        v = args[idx] if len(args) > idx else kwargs.get("argv", kwargs.get("args"))
        if isinstance(v, (list, tuple)) and not v:
            return False
        return v is None or isinstance(v, (Sym, list, tuple))
    return check


for _n in ("os.execv", "os.execve", "os.execvp", "os.execvpe"):
    ARG_VALIDATORS[_n] = _exec_argv_ok(1)
for _n in ("os.execl", "os.execle", "os.execlp", "os.execlpe"):
    ARG_VALIDATORS[_n] = _exec_argv_ok(None)


# further callables with the same capabilities
_add(CODE, """
    pydoc.importfile typing.get_type_hints typing._eval_type typing.ForwardRef._evaluate
    inspect.get_annotations annotationlib.get_annotations
    keras.models.load_model keras.saving.load_model keras.src.saving.saving_api.load_model
    tensorflow.keras.models.load_model tensorflow.saved_model.load
    torch.utils.cpp_extension.load torch.utils.cpp_extension.load_inline
    torch.utils.cpp_extension.CppExtension
""")
_add(FILE, """
    tensorflow.io.read_file tensorflow.io.write_file tensorflow.io.gfile.GFile
    tensorflow.io.gfile.remove tensorflow.io.gfile.copy scipy.io.loadmat scipy.io.savemat
    pandas.read_csv pandas.DataFrame.to_parquet
""")
RETURNS["webbrowser.get"] = "webbrowser.BaseBrowser"

_add(NET, "xmlrpc.client.ServerProxy.__request")
_add(FILE, "zipfile._path.Path.read_text zipfile._path.Path.read_bytes zipfile._path.Path.open "
           "importlib.resources._legacy.read_text importlib.resources._legacy.read_binary "
           "importlib.resources._legacy.open_text importlib.resources._legacy.open_binary")
_add(FILE, "_gdbm.open _dbm.open")



# ---------------------------------------------------------------------------
# third-party packages assumed installed where models are loaded; a module
# that is neither standard library nor one of these fails to import
KNOWN_THIRD_PARTY = set("""
    numpy scipy sklearn torch torchvision torchaudio torchtext transformers tensorflow keras tf_keras
    tensorboard jax jaxlib flax optax haiku pandas joblib xgboost lightgbm catboost dill cloudpickle
    h5py google protobuf onnx onnxruntime PIL cv2 matplotlib seaborn statsmodels sentencepiece
    tokenizers safetensors accelerate datasets huggingface_hub lightning pytorch_lightning timm
    yaml requests urllib3 numba sympy networkx tqdm gensim nltk spacy fastai mxnet paddle
    sklearn_crfsuite imblearn category_encoders shap lime optuna hyperopt mlflow wandb ray dask
    pyarrow polars fsspec pydantic attr attrs typing_extensions six packaging setuptools pkg_resources
    pip wheel certifi charset_normalizer idna regex scikit_learn skimage imageio plotly bokeh
    einops opt_einsum ml_dtypes absl tree dm_tree wrapt gast astunparse termcolor flatbuffers
    tensorflow_hub tensorflow_text tensorflow_probability tensorflow_datasets keras_nlp keras_cv
    keras_hub peft trl bitsandbytes deepspeed apex fairseq allennlp sentence_transformers
    diffusers open_clip clip whisper torchmetrics triton sympy mpmath filelock jinja2 markupsafe
    psutil cython Cython numexpr bottleneck tables openpyxl xlrd lxml bs4 sqlalchemy
    __main__ copy_reg
""".split())



def _not_callable(v):
    from modelgate.handlers import is_callable_value
    return not is_callable_value(v)


def _thread_ok(args, kwargs):
    from modelgate.symbolic import Opaque
    group = args[0] if args else kwargs.get("group")
    return group is None or isinstance(group, Opaque)


def _address_ok(args, kwargs):
    from modelgate.symbolic import Sym
    a = args[0] if args else kwargs.get("address")
    if isinstance(a, (tuple, list)):
        return len(a) == 2
    return a is None or isinstance(a, Sym)


_MODE_CHARS = set("rwxabt+U")


def _open_ok(args, kwargs):
    mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
    if isinstance(mode, str):
        m = set(mode)
        if len(mode) != len(m) or not m <= _MODE_CHARS:
            return False
        if len(m & set("rwxa")) != 1 and "U" not in m:
            return False
        if "b" in m and "t" in m:
            return False
    return True


def _first_callable(args, kwargs):
    return not args or not _not_callable(args[0])


def _lru_cache_ok(args, kwargs):
    from modelgate.symbolic import Sym
    v = args[0] if args else kwargs.get("maxsize", 128)
    return v is None or isinstance(v, (int, Sym)) and not (isinstance(v, Sym) and _not_callable(v))


def _iter_ok(args, kwargs):
    return len(args) != 2 or not _not_callable(args[0])


def _popen_first_ok(args, kwargs):
    from modelgate.interp import not_value_like
    from modelgate.symbolic import Sym
    first = args[0] if args else kwargs.get("args")
    if first is not None and not_value_like(first):
        return False
    if first is not None and not isinstance(first, Sym) and not isinstance(first, _SEQ):
        return False  # an int / float / None is not an argument list
    return _popen_args_ok(args, kwargs)


ARG_VALIDATORS.update({
    "threading.Thread": _thread_ok,
    "socket.create_connection": _address_ok,
    "io.open": _open_ok,
    "builtins.open": _open_ok,
    "codecs.open": _open_ok,
    "functools.cache": _first_callable,
    "functools.singledispatch": _first_callable,
    "atexit.register": _first_callable,
    "functools.lru_cache": _lru_cache_ok,
    "builtins.iter": _iter_ok,
})
for _n in ("subprocess.Popen", "subprocess.call", "subprocess.check_call", "subprocess.check_output",
           "subprocess.run"):
    ARG_VALIDATORS[_n] = _popen_first_ok
for _n in ("subprocess.getoutput", "subprocess.getstatusoutput", "os.system", "os.popen"):
    ARG_VALIDATORS.setdefault(_n, lambda a, k: not (a and __import__("modelgate.interp").interp.not_value_like(a[0])))

ARG_CHECKS["os.fspath"] = (0, "path", (str, bytes))
ARG_CHECKS["os.fsencode"] = (0, "filename", (str, bytes))
ARG_CHECKS["os.fsdecode"] = (0, "filename", (str, bytes))
KNOWN_THIRD_PARTY |= set("""
    omegaconf hydra lightning_fabric fastcore detectron2 mmcv mmdet mmengine ultralytics tiktoken
    faiss umap hdbscan prophet pmdarima sktime mlxtend feature_engine river surprise implicit lightfm
    gym gymnasium stable_baselines3 tensorflow_addons keras_tuner autogluon h2o pycaret flaml
    distributed xarray shapely geopandas pyspark torch_geometric dgl sklearn2pmml skl2onnx onnxmltools
    treelite evaluate sacremoses jieba gluonts neuralprophet darts pytorch_forecasting einx kornia
    albumentations torchdata torchrec fairscale xformers flash_attn vllm sentence_transformers
    llama_cpp gguf mlx coremltools tensorrt openvino paddlenlp paddleocr scikit_image numpy_financial
    seqeval rapidfuzz Levenshtein editdistance pyod pyts tslearn statsforecast neuralforecast mlforecast
""".split())

_add(CODE, """
    torch.fx.graph_module.reduce_graph_module torch.fx.graph_module.reduce_package_graph_module
    torch.fx.graph_module.reduce_deploy_graph_module torch.fx.graph_module._forward_from_src
    torch.fx.graph_module._exec_with_source torch.fx.graph_module._deserialize_graph_module
    cloudpickle.cloudpickle._make_function cloudpickle.cloudpickle_fast._make_function
    cloudpickle.cloudpickle._make_code cloudpickle.cloudpickle._fill_function
    sympy.sympify sympy.core.sympify.sympify sympy.parsing.sympy_parser.parse_expr
    sympy.parsing.sympy_parser.eval_expr
""")
CAPS["yaml.load"] = (CODE, lambda a, k: len(a) > 1 or "Loader" in k)

# found by scanning the standard library for callers of process/exec/network primitives
_add(PROC, """
    platform.processor platform.platform platform.architecture platform._Processor.get
    platform._Processor.from_subprocess mailcap.findmatch mailcap.test pipes.Template.copy
    pipes.Template.open pipes.Template.open_r pipes.Template.open_w multiprocessing.util.spawnv_passfds
    multiprocessing.forkserver.ForkServer.ensure_running multiprocessing.forkserver.main
    multiprocessing.resource_tracker.ResourceTracker.ensure_running imaplib.IMAP4_stream.open
""")
_add(CODE, """
    optparse.Values.read_file optparse.Values.read_module multiprocessing.managers.MakeProxyType
    multiprocessing.sharedctypes.make_property
""")
_add(NET, """
    logging.handlers.SocketHandler.makeSocket logging.handlers.SocketHandler.createSocket
    smtplib.LMTP.connect urllib.request.ftpwrapper.init urllib.request.ftpwrapper
    xml.dom.xmlbuilder.DOMBuilder.parseURI
""")
