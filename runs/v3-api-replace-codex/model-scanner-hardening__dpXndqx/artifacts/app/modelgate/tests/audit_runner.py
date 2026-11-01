"""Run the real CLI with an audit hook that records forbidden side effects."""
import sys
sys.dont_write_bytecode = True
import os
import runpy
import traceback

report, artifact = sys.argv[1:]
violations = []
threads_before = len(os.listdir('/proc/self/task'))


def audit(event, args):
    blocked = event in {'os.system', 'os.fork', 'os.forkpty', 'os.posix_spawn',
                        'subprocess.Popen', 'socket.connect', 'socket.getaddrinfo',
                        'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.link',
                        'os.symlink', 'os.truncate', 'os.chmod'} or event.startswith(('os.exec', 'os.spawn'))
    if event == 'open':
        path, mode, flags = args
        writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        blocked = bool(writing) and path != report
    if blocked:
        violations.append((event + ': ' + str(args) + '\n' + ''.join(traceback.format_stack(limit=18))) if event == 'open' else event)
        raise RuntimeError('scanner side effect: ' + event)


sys.addaudithook(audit)
sys.argv = ['modelgate', 'scan', '--output', report, artifact]
try:
    runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cli.py'), run_name='__main__')
except SystemExit as exc:
    if exc.code:
        raise
if len(os.listdir('/proc/self/task')) > threads_before:
    violations.append('native worker pool created')
if violations:
    print('forbidden scanner events: ' + ', '.join(violations), file=sys.stderr)
    sys.exit(99)
