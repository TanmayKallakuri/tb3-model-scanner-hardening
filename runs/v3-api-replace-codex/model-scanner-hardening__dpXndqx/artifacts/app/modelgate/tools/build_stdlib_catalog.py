"""Rebuild stdlib.json from trusted, installed standard-library modules.

Development utility only. The scanner reads the frozen catalog and never
imports a target module based on an artifact's contents.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib
import builtins
import inspect
import json
import types

names = '''builtins sys os posix _pickle _io _operator _functools _socket posixpath ntpath io pickle copy copyreg functools operator
collections itertools types base64 binascii codecs _codecs zlib gzip bz2 lzma
marshal pathlib shutil subprocess socket ssl urllib urllib.request urllib.parse
http http.client ftplib smtplib xmlrpc.client webbrowser pty pydoc timeit runpy
code bdb pdb logging logging.config logging.handlers threading _thread
concurrent concurrent.futures concurrent.futures.thread concurrent.futures.process
multiprocessing multiprocessing.context multiprocessing.process multiprocessing.pool
multiprocessing.dummy multiprocessing.popen_fork multiprocessing.util
importlib importlib.util importlib.machinery importlib._bootstrap
pkgutil contextlib tempfile sqlite3 dbm shelve fileinput linecache tokenize ast json
struct math decimal fractions datetime re csv array weakref queue sched
asyncio asyncio.subprocess inspect statistics random unittest.mock
'''.split()
aliases = {'posix':'os', 'nt':'os','_io':'io','_pickle':'pickle','_functools':'functools',
           '_operator':'operator','_socket':'socket','_collections':'collections','_threading_local':'threading'}
preferred = {id(types.FunctionType):'types.FunctionType', id(types.CodeType):'types.CodeType',
             id(types.MethodType):'types.MethodType', id(types.ModuleType):'types.ModuleType'}
classes = {}
functions = {}
modules = {}

def canon(name):
 p,sep,rest=name.partition('.')
 return aliases.get(p,p)+(sep+rest if sep else '')

def refname(obj, fallback):
 if id(obj) in preferred: return preferred[id(obj)]
 mod = getattr(obj,'__module__',None)
 qual = getattr(obj,'__qualname__',getattr(obj,'__name__',None))
 if isinstance(mod,str) and isinstance(qual,str) and '<' not in qual:
  return canon(mod+'.'+qual)
 return fallback

def encode(obj, fallback):
 if obj is builtins.__dict__: return ['n','builtins']
 if isinstance(obj,types.ModuleType): return ['m',obj.__name__]
 if isinstance(obj,(type,types.FunctionType,types.BuiltinFunctionType,types.MethodDescriptorType,types.WrapperDescriptorType,types.ClassMethodDescriptorType)):
  name=refname(obj,fallback)
  if isinstance(obj,type):
   classes.setdefault(name,None)
  else:
   meta={}
   if isinstance(obj,types.FunctionType): meta['globals']=obj.__globals__.get('__name__')
   elif isinstance(getattr(obj,'__self__',None),types.ModuleType): meta['self']=obj.__self__.__name__
   try:
    sig=inspect.signature(obj)
    meta['signature']=[[p.name,p.kind.name,p.default is not inspect.Parameter.empty] for p in sig.parameters.values()]
   except (TypeError,ValueError): pass
   functions[name]=meta
  return ['r',name]
 if obj is None or type(obj) in (bool,int,float,str):
  if isinstance(obj,str) and len(obj)>1024: return ['u','data']
  return ['d',obj]
 if type(obj) in (tuple,list) and len(obj)<=32 and all(v is None or type(v) in (bool,int,float,str) for v in obj):
  return ['t' if type(obj) is tuple else 'd',list(obj)]
 return ['u',refname(type(obj),'object')]

loaded = {}
for name in names:
 try: loaded[name]=importlib.import_module(name)
 except ImportError: continue
# Lazy exports and submodule bindings must be resolved after all imports.
for attr in ('ThreadPoolExecutor', 'ProcessPoolExecutor'):
 getattr(loaded['concurrent.futures'], attr)
for name, mod in loaded.items():
 modules[name]={k:encode(v,name+'.'+k) for k,v in list(vars(mod).items()) if k not in ('environ','environb','argv','path','orig_argv','_xoptions','_stdlib_dir','__file__','__cached__')}
 if name=='os': modules[name]['path']=['m',canon(mod.path.__name__)]

# Process only exported stdlib classes, without evaluating descriptors.
for name in list(classes):
 if not name.startswith(tuple(n+'.' for n in names)): continue
 if not (name.startswith(('io.', 'socket.', 'pathlib.', 'functools.', 'operator.', 'bdb.', 'pdb.', 'code.', 'types.', 'importlib.', 'logging.')) or name in (
  'builtins.object','builtins.type','builtins.str','builtins.bytes','builtins.bytearray',
  'builtins.dict','builtins.list','builtins.tuple','builtins.set','builtins.frozenset',
  'builtins.int','builtins.float','builtins.bool','builtins.map','builtins.filter',
  'builtins.zip','builtins.range','builtins.enumerate','builtins.reversed','builtins.slice',
  'builtins.memoryview','builtins.staticmethod','builtins.classmethod','builtins.property',
  'builtins.super','timeit.Timer','http.client.HTTPConnection','http.client.HTTPSConnection',
  'ftplib.FTP','ftplib.FTP_TLS','smtplib.SMTP','smtplib.SMTP_SSL','smtplib.LMTP',
  'xmlrpc.client.ServerProxy','xmlrpc.client._Method','threading.Thread','threading.Timer','subprocess.Popen',
  'concurrent.futures.thread.ThreadPoolExecutor','concurrent.futures.process.ProcessPoolExecutor',
  'multiprocessing.process.BaseProcess','multiprocessing.context.Process')): continue
 parts=name.split('.')
 obj=None
 for cut in range(len(parts)-1,0,-1):
  mn='.'.join(parts[:cut])
  if mn in modules:
   try:
    obj=importlib.import_module(mn)
    for part in parts[cut:]: obj=vars(obj)[part]
   except (KeyError,TypeError): obj=None
   break
 if not isinstance(obj,type): continue
 attrs={}
 for base in reversed(obj.__mro__):
  for k,v in vars(base).items():
   bound_classmethod = isinstance(v,(classmethod,types.ClassMethodDescriptorType))
   if isinstance(v,(staticmethod,classmethod)): v=v.__func__
   attrs[k]=encode(v,name+'.'+k)
   if bound_classmethod and attrs[k][0]=='r':
    try:
     sig=inspect.signature(getattr(obj,k))
     functions[attrs[k][1]]['signature']=[[p.name,p.kind.name,p.default is not inspect.Parameter.empty] for p in sig.parameters.values()]
    except (TypeError,ValueError,AttributeError): pass
 classes[name]=attrs

with open(Path(__file__).resolve().parents[1] / 'stdlib.json', 'w') as f:
 json.dump({'modules':modules,'classes':classes,'functions':functions},f,separators=(',',':'),sort_keys=True)
print(len(modules),len(classes),len(functions))
