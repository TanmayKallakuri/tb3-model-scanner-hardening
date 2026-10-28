# Development-time index of trusted standard library objects. The scanner
# reads the resulting inert data; it never imports modules named by artifacts.
import sys
sys.dont_write_bytecode = True
import importlib, inspect, json, types
from pathlib import Path
names = '''builtins os sys posixpath ntpath shutil subprocess pickle io operator functools
importlib importlib.util importlib.machinery pkgutil socket urllib urllib.request urllib.parse
http http.client ftplib smtplib ssl xmlrpc xmlrpc.client threading _thread concurrent
concurrent.futures concurrent.futures.thread concurrent.futures.process multiprocessing
multiprocessing.pool multiprocessing.process multiprocessing.context pathlib copy copyreg types
marshal base64 binascii codecs _codecs zlib gzip bz2 lzma itertools collections collections.abc
logging logging.config logging.handlers code bdb pdb timeit runpy pty pydoc tempfile glob time
math re struct random ast json csv decimal fractions datetime array weakref contextlib traceback
signal stat warnings enum inspect sqlite3 shelve dbm imaplib poplib telnetlib webbrowser
ctypes ctypes.util queue reprlib socketserver configparser html.entities html.parser http.cookies
http.cookiejar http.server urllib.robotparser xmlrpc.server xml.etree.ElementTree'''.split()
modules = {}
for name in names:
 try:
  modules[name] = importlib.import_module(name)
 except ImportError:
  pass
result = {}
for name, mod in modules.items():
 attrs = {}
 for key, value in list(vars(mod).items()):
  if key in ('__builtins__', '__loader__', '__spec__', '__file__', '__cached__', '__doc__'):
   continue
  if isinstance(value, types.ModuleType):
   attrs[key] = ['module', value.__name__]
  elif inspect.isclass(value) or inspect.isroutine(value):
   module = getattr(value, '__module__', None)
   qualname = getattr(value, '__qualname__', getattr(value, '__name__', None))
   if module and qualname and '<' not in qualname:
    attrs[key] = ['ref', module + '.' + qualname]
  elif type(value) in (str, bool, int, float, type(None)):
   if not isinstance(value, str) or len(value) < 512:
    attrs[key] = ['literal', value]
 result[name] = attrs
with (Path(__file__).resolve().parents[1] / 'stdlib_symbols.json').open('w') as fh:
 json.dump(result, fh, separators=(',', ':'), sort_keys=True)
print(len(result), 'modules;', sum(len(v) for v in result.values()), 'attributes')

# Binding checks distinguish capabilities from calls that necessarily raise
# before the callable's body can run. Descriptors keep their self parameter.
signatures = {}

def save_signature(name, value):
 try:
  signature = inspect.signature(value)
 except (TypeError, ValueError):
  return
 signatures[name] = [[int(p.kind), p.name, p.default is inspect.Parameter.empty]
                     for p in signature.parameters.values()]

for name, mod in modules.items():
 for key, value in list(vars(mod).items()):
  if inspect.isclass(value) or inspect.isroutine(value):
   save_signature(name + '.' + key, value)
   if inspect.isclass(value) and getattr(value, '__module__', '').split('.')[0] in names:
    for method, descriptor in list(vars(value).items()):
     if inspect.isroutine(descriptor):
      save_signature(name + '.' + key + '.' + method, getattr(value, method))
with (Path(__file__).resolve().parents[1] / 'stdlib_signatures.json').open('w') as fh:
 json.dump(signatures, fh, separators=(',', ':'), sort_keys=True)
print(len(signatures), 'call signatures')
