"""Test-only launcher: reject scanner writes (except its report), subprocesses,
and networking through CPython's audit interface. Artifacts are never loaded.
"""
import os
import runpy
import sys

sys.dont_write_bytecode = True
cli, report, *paths = sys.argv[1:]
report = os.path.abspath(report)


def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.fork', 'os.forkpty', 'os.posix_spawn', 'os.exec', 'os.spawn'):
        raise RuntimeError('scanner attempted a process: ' + event)
    if event.startswith('socket.') and event not in ('socket.__new__',):
        raise RuntimeError('scanner attempted networking: ' + event)
    if event == 'socket.__new__':
        raise RuntimeError('scanner attempted a socket')
    if event == 'open':
        path, mode, flags = args
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+'))
        writing |= isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing and (not isinstance(path, (str, bytes)) or os.path.abspath(os.fsdecode(path)) != report):
            raise RuntimeError('scanner attempted an extra write: ' + repr(path))
    if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.link', 'os.symlink', 'os.truncate'):
        raise RuntimeError('scanner attempted a filesystem change: ' + event)


sys.addaudithook(audit)
sys.argv = [cli, 'scan', '--output', report, *paths]
runpy.run_path(cli, run_name='__main__')
