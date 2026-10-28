"""Run the actual CLI under a guard against scanner side effects.

Usage: python3 -I audit_cli.py REPORT ARTIFACT [ARTIFACT ...]
The guard is installed before importing scanner code and optional parsers.
"""
import sys
sys.dont_write_bytecode = True
import os
import runpy

report = os.path.abspath(sys.argv[1])
paths = sys.argv[2:]
violations = []


def guard(event, args):
    denied = False
    if event == 'open':
        path, mode, flags = args
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if isinstance(flags, int) and flags & write_flags:
            denied = not isinstance(path, (str, bytes)) or os.path.abspath(path) != report
    elif event in ('os.system', 'subprocess.Popen', 'os.posix_spawn', 'os.fork', 'os.forkpty',
                   'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'socket.gethostbyname',
                   'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.link', 'os.symlink', 'os.truncate'):
        denied = True
    if denied:
        violations.append((event, repr(args)))
        raise RuntimeError('forbidden scanner side effect: ' + event)


sys.addaudithook(guard)
cli = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cli.py')
sys.argv = [cli, 'scan', '--output', report] + paths
try:
    runpy.run_path(cli, run_name='__main__')
except SystemExit as error:
    if error.code: raise
if violations:
    raise SystemExit('side effects blocked: ' + repr(violations))
