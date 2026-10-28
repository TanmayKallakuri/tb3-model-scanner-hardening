"""Prevent incidental side effects in parser dependencies as well as our code."""
from __future__ import annotations
import os
import sys

BLOCKED_EVENTS = {
    'subprocess.Popen', 'os.system', 'os.fork', 'os.forkpty', 'os.posix_spawn',
    'os.exec', 'os.spawn', 'os.startfile', 'os.startfile/2',
    'socket.__new__', 'socket.connect', 'socket.bind', 'socket.getaddrinfo',
    'socket.gethostbyname', 'socket.gethostbyaddr', 'socket.getnameinfo',
    'os.remove', 'os.rmdir', 'os.mkdir', 'os.rename', 'os.link', 'os.symlink',
    'os.chmod', 'os.chown', 'os.truncate', 'os.utime', 'os.mknod', 'os.mkfifo',
}


def install(report_path):
    """The CLI's sole write authority is the exact report pathname."""
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND

    def check(event, args):
        if event in BLOCKED_EVENTS:
            raise PermissionError('scanner blocked side effect: ' + event)
        if event == 'open':
            path, mode, flags = args
            if flags & write_flags and path != report_path:
                raise PermissionError('scanner blocked a write outside its report')

    sys.addaudithook(check)
