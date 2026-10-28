"""Build inert standard-library name/signature metadata, never used at scan time."""
import importlib
import inspect
import json
import types
from pathlib import Path
import sys
sys.dont_write_bytecode = True
modules = '''builtins os posix sys io _io pickle _pickle copy copyreg functools _functools
operator _operator importlib importlib.util pkgutil shutil subprocess pathlib socket
_socket ssl pty runpy timeit code codeop bdb pdb logging logging.config logging.handlers
base64 binascii zlib bz2 lzma gzip codecs _codecs collections collections.abc itertools
threading _thread multiprocessing multiprocessing.context multiprocessing.pool
multiprocessing.connection concurrent concurrent.futures concurrent.futures.thread
concurrent.futures.process urllib urllib.request urllib.parse http http.client
ftplib smtplib xmlrpc xmlrpc.client posixpath ntpath tempfile glob random time math re
json struct marshal types pydoc platform linecache fileinput sqlite3 shelve dbm
webbrowser zipfile tarfile weakref contextlib csv socketserver selectors stat
encodings enum signal inspect ast dataclasses _collections _contextvars contextvars
queue _queue fractions decimal numbers array datetime calendar textwrap heapq bisect
abc unittest.mock'''.split()
alias = {'posix': 'os', 'nt': 'os', '_io': 'io', '_pickle': 'pickle', '_operator':'operator',
         '_functools': 'functools', '_socket': 'socket', '_collections': 'collections',
         '_threading_local': 'threading'}
spaces, symbols = {}, {}

def canonical(obj, fallback):
    mod = getattr(obj, '__module__', None)
    qual = getattr(obj, '__qualname__', getattr(obj, '__name__', None))
    if isinstance(mod, str) and isinstance(qual, str) and qual and '<' not in qual:
        mod = alias.get(mod, mod)
        if mod == 'builtins' and qual == 'module':
            return 'types.ModuleType'
        if mod == 'builtins' and qual == 'function':
            return 'types.FunctionType'
        if mod == 'builtins' and qual == 'code':
            return 'types.CodeType'
        if mod == 'builtins' and qual == 'method':
            return 'types.MethodType'
        return mod + '.' + qual
    return fallback

pending = []
for m in modules:
    if m in alias and alias[m] in modules:
        continue
    try:
        obj = importlib.import_module(m)
        pending.append((alias.get(m, m), obj, True))
    except ImportError:
        pass
while pending:
    namespace, obj, descend = pending.pop()
    if namespace in spaces:
        continue
    entries = {}
    spaces[namespace] = entries
    for attr in dir(obj):
        if attr in ('__builtins__', '__doc__', '__text_signature__'):
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        fullname = namespace + '.' + attr
        if isinstance(value, types.ModuleType):
            entries[attr] = ['module', alias.get(value.__name__, value.__name__)]
        elif isinstance(value, type) or callable(value):
            # Instance-bound methods expose per-instance state; retain their
            # exported name instead of collapsing to their declaring class.
            target = canonical(value, fullname)
            if inspect.ismethod(value) and not isinstance(value.__self__, type):
                target = fullname
            kind = 'class' if isinstance(value, type) else 'function'
            entries[attr] = [kind, target]
            meta = symbols.setdefault(target, {'kind': kind})
            if kind == 'function':
                meta['python'] = inspect.isfunction(value) or inspect.ismethod(value)
            if kind in ('function', 'class'):
                try:
                    sig = inspect.signature(value)
                    meta['params'] = [[p.name, int(p.kind), p.default is inspect.Parameter.empty]
                                      for p in sig.parameters.values()]
                except (ValueError, TypeError):
                    pass
            if kind == 'class' and descend:
                pending.append((target, value, False))
        elif type(value) in (str, bytes, int, float, bool, type(None)):
            if isinstance(value, bytes):
                entries[attr] = ['bytes', value.hex()]
            elif isinstance(value, str) and len(value) > 10000:
                entries[attr] = ['unknown', None]
            else:
                entries[attr] = ['data', value]
        else:
            entries[attr] = ['unknown', None]
with open(Path(__file__).resolve().parents[1] / 'stdlib_metadata.json', 'w') as f:
    json.dump({'namespaces': spaces, 'symbols': symbols}, f, separators=(',',':'))
print(len(spaces), 'namespaces;',len(symbols),'symbols')
