"""Test-only audit wrapper; run from a separate Python process."""
import sys
sys.dont_write_bytecode = True
import os
import runpy

cli, report, *artifacts = sys.argv[1:]
violations = []


def audit(event, args):
    blocked = event.startswith(('subprocess.', 'os.exec', 'os.spawn', 'os.posix_spawn'))
    blocked |= event in ('os.system', 'os.fork', 'os.forkpty', 'socket.connect', 'socket.bind',
                         'socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyaddr',
                         'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.link', 'os.symlink',
                         'os.truncate', 'os.chmod', 'os.chown', 'os.utime')
    if event == 'open':
        path, mode, flags = args
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        blocked |= writing and path != report
    if blocked:
        violations.append((event, repr(args)))
        raise RuntimeError('scanner side effect: ' + event)


sys.addaudithook(audit)
sys.argv = [cli, 'scan', '--output', report] + artifacts
try:
    runpy.run_path(cli, run_name='__main__')
except SystemExit as exc:
    if exc.code:
        raise
if violations:
    raise RuntimeError(str(violations))
