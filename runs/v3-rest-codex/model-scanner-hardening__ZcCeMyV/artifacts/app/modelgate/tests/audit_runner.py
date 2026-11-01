"""Run the actual CLI while auditing forbidden scanner side effects.

Use with evaluate.py --scanner .../tests/audit_runner.py. This is a test
harness only: the deployed CLI does not depend on audit hooks for safety.
"""
import sys
sys.dont_write_bytecode=True
import os
import runpy

output=os.path.abspath(sys.argv[sys.argv.index('--output')+1])
violations=[]


def audit(event,args):
    forbidden=False
    if event=='open':
        path,mode,flags=args
        write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        forbidden=bool(write and (not isinstance(path,(str,bytes)) or os.path.abspath(os.fsdecode(path))!=output))
    elif event in ('os.system','os.fork','os.forkpty','os.posix_spawn','os.exec','subprocess.Popen',
                   'socket.connect','socket.connect_ex','socket.bind','socket.getaddrinfo',
                   'os.remove','os.rename','os.mkdir','os.rmdir','os.link','os.symlink','os.truncate'):
        forbidden=True
    elif event=='pickle.find_class':
        forbidden=True
    if forbidden:
        violations.append((event,str(args)[:200]))
        raise RuntimeError('scanner attempted forbidden behavior: '+event)


sys.addaudithook(audit)
cli=os.path.join(os.path.dirname(os.path.dirname(__file__)),'cli.py')
try:
    runpy.run_path(cli,run_name='__main__')
finally:
    if violations:
        print('AUDIT FAILURE:',violations,file=sys.stderr)
        raise SystemExit(91)
