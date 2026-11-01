"""Content-based, in-memory container traversal and model format dispatch."""
from __future__ import annotations

import ast
import bz2
import gzip
import io
import json
import lzma
import os
import struct
import tarfile
import zipfile
import zlib

from modelgate import keras_scan, pickle_scan, savedmodel_scan
from modelgate.report import Finding, Result
from modelgate.symbolic import Runtime, UNKNOWN, Halt, Unsupported, Detected

MAX_EXPANDED = 128 * 1024 * 1024
PICKLE_EXTS = ('.pkl','.pickle','.joblib','.pt','.pth','.ckpt','.bin')
# The set of bytes which can begin a pickle, including the text protocols.
PICKLE_START = set(b'\x80(}])NIFJ KLMPSUVXcdiolptgqhrb0123456789'.replace(b' ',b'')) | {
    0x81,0x82,0x83,0x84,0x88,0x89,0x8a,0x8b,0x8c,0x8d,0x8e,0x8f,0x90,0x91,0x93,0x94,0x95,0x96}


def decompress(data, kind, wbits=None):
    if not isinstance(data,(bytes,bytearray)):
        raise Halt('compressed data must be bytes')
    if kind=='gzip':
        remaining=bytes(data); chunks=[]; size=0
        while remaining:
            # gzip permits padding between/after members.
            remaining=remaining.lstrip(b'\0')
            if not remaining: break
            obj=zlib.decompressobj(31)
            chunk=obj.decompress(remaining,MAX_EXPANDED-size+1)
            size+=len(chunk)
            if size>MAX_EXPANDED or obj.unconsumed_tail:
                raise Unsupported('expanded container size limit')
            if not obj.eof: raise Halt('truncated gzip stream')
            chunks.append(chunk); remaining=obj.unused_data
        return b''.join(chunks)
    if kind=='zlib':
        obj=zlib.decompressobj(wbits if isinstance(wbits,int) else 15)
        out=obj.decompress(data,MAX_EXPANDED+1)
        if len(out)>MAX_EXPANDED or obj.unconsumed_tail: raise Unsupported('expanded data size limit')
        if not obj.eof: raise Halt('truncated zlib stream')
        return out
    cls={'bz2':bz2.BZ2Decompressor,'lzma':lzma.LZMADecompressor}[kind]
    remaining=bytes(data); chunks=[]; size=0
    while remaining:
        obj=cls(); out=obj.decompress(remaining,max_length=MAX_EXPANDED-size+1)
        size+=len(out)
        if size>MAX_EXPANDED or not obj.eof:
            raise Unsupported('expanded data size limit or truncated compressed stream')
        chunks.append(out); remaining=obj.unused_data
    return b''.join(chunks)


def _compression(data):
    if data.startswith(b'\x1f\x8b'): return 'gzip'
    if data.startswith(b'BZh'): return 'bz2'
    if data.startswith(b'\xfd7zXZ\x00'): return 'lzma'
    if len(data)>=2 and data[0]&15==8 and data[0]>>4<=7 and int.from_bytes(data[:2],'big')%31==0:
        return 'zlib'
    # joblib also writes the legacy lzma "alone" header.
    if data.startswith(b']\x00\x00'): return 'lzma'
    return None


def _zip(data):
    # A nested ZIP byte string inside a pickle also makes is_zipfile() true:
    # zipfile tolerates bytes after EOCD. Our artifact ZIP shape has an EOCD
    # (including its comment) at EOF, with arbitrary bytes *before* headers.
    # This keeps inert embedded archives under their enclosing loader.
    start=data.rfind(b'PK\x05\x06',max(0,len(data)-65557))
    if start<0 or start+22>len(data): return False
    comment_length=int.from_bytes(data[start+20:start+22],'little')
    return start+22+comment_length==len(data) and zipfile.is_zipfile(io.BytesIO(data))


def _hdf(data):
    magic=b'\x89HDF\r\n\x1a\n'
    if data.startswith(magic): return True
    offset=512
    while offset+8<=len(data):
        if data[offset:offset+8]==magic: return True
        offset*=2
    return False


def _object_dtype(descr):
    if isinstance(descr,str):
        d=descr.lstrip('<>=|')
        return d=='object' or (d.startswith('O') and (not d[1:] or d[1:].isdigit()))
    if isinstance(descr,list):
        return any(isinstance(field,(tuple,list)) and len(field)>1 and _object_dtype(field[1]) for field in descr)
    if isinstance(descr,tuple) and descr:
        return _object_dtype(descr[0])
    if isinstance(descr,dict): return any(_object_dtype(x) for x in descr.get('formats',[]))
    return False


def numpy_array(data,location,rt,allow_pickle=True,mmap_mode=None):
    if len(data)<10: raise Halt('truncated npy header')
    major,minor=data[6:8]
    if major==1:
        hlen=struct.unpack('<H',data[8:10])[0]; start=10
    elif major in (2,3):
        if len(data)<12: raise Halt('truncated npy header')
        hlen=struct.unpack('<I',data[8:12])[0]; start=12
    else: raise Unsupported('unknown npy version')
    if hlen>1024*1024 or start+hlen>len(data): raise Halt('invalid npy header length')
    try: header=ast.literal_eval(data[start:start+hlen].decode('utf-8' if major==3 else 'latin1').strip())
    except (ValueError,SyntaxError,UnicodeError): raise Halt('invalid npy header') from None
    if not isinstance(header,dict): raise Halt('invalid npy header')
    if _object_dtype(header.get('descr')):
        if not allow_pickle or mmap_mode is not None: raise Halt('object array cannot be loaded with these numpy options')
        value,used=pickle_scan.evaluate(data[start+hlen:],location+'!object-data',rt,first=True)
        return value,start+hlen+used
    return UNKNOWN,len(data)


def _tar(data):
    if len(data)<512: return False
    if data[257:263] in (b'ustar\0',b'ustar '): return True
    try:
        stored=int(data[148:156].strip(b'\0 ') or b'0',8)
        return stored>0 and stored==sum(data[:148])+8*32+sum(data[156:512])
    except ValueError: return False


def _emit(findings):
    if findings: raise Detected(findings[0])


def _torch_pickle(infos,magic):
    if not magic.startswith(b'PK\x03\x04'): return None
    for name in infos:
        if name!='data.pkl' and not name.endswith('/data.pkl'): continue
        prefix=name[:-len('data.pkl')]
        if prefix+'version' not in infos: continue
        if prefix and not all(n.startswith(prefix) for n in infos): continue
        return name
    return None


def _scan(data,location,rt,depth=0,member=False,loader=None,options=None):
    if depth>12: raise Unsupported('container nesting limit exceeded')
    if not isinstance(data,bytes): data=bytes(data)
    if len(data)>MAX_EXPANDED: raise Unsupported('artifact size limit exceeded')
    rt.location=location
    options=options or {}
    # A nested pickle reader consumes exactly one stream, regardless of content.
    if loader in ('pickle.loads','pickle.load','pickle.Unpickler.load','pickle._Unpickler.load'):
        return pickle_scan.evaluate(data,location,rt,first=True,fix_imports=options.get('fix_imports',True))
    if loader=='numpy.lib.format.read_array':
        if not data.startswith(b'\x93NUMPY'): raise Halt('not an npy array')
        return numpy_array(data,location,rt,options.get('allow_pickle',False))
    if loader in ('torch.load','torch.storage._load_from_bytes') and not data.startswith(b'PK\x03\x04'):
        return pickle_scan.evaluate(data,location,rt,legacy=True,persistent=True)
    if loader == 'numpy.load' and not data.startswith((b'PK\x03\x04',b'\x93NUMPY')):
        if not options.get('allow_pickle',False): raise Halt('numpy.load disallows pickled data')
        return pickle_scan.evaluate(data,location,rt,first=True)
    if loader == 'pandas.read_pickle':
        compression=options.get('compression')
        if isinstance(compression,dict): compression=compression.get('method')
        kind={'gz':'gzip','gzip':'gzip','bz2':'bz2','xz':'lzma','lzma':'lzma','zlib':'zlib'}.get(compression)
        if kind: data=decompress(data,kind)
        if compression=='zip':
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                entries={i.filename:i for i in zf.infolist() if not i.is_dir()}
                if len(entries)!=1: raise Halt('pandas pickle zip needs one member')
                info=next(iter(entries.values()))
                if info.file_size>MAX_EXPANDED: raise Unsupported('expanded zip size limit')
                data=zf.read(info)
        return pickle_scan.evaluate(data,location,rt,first=True)
    if _tar(data):
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as tf:
            entries={info.name:info for info in tf.getmembers() if info.isfile()}
            if sum(info.size for info in entries.values())>MAX_EXPANDED:
                raise Unsupported('expanded tar size limit exceeded')
            for name,info in entries.items():
                fh=tf.extractfile(info)
                if fh:
                    try: _scan(fh.read(),location+'!'+name,rt,depth+1,True)
                    except Halt: pass
        return UNKNOWN,len(data)
    # zipfile uses the central directory, including a prefix before the first
    # local header and the final version of duplicate member names.
    if _zip(data):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos={info.filename:info for info in zf.infolist() if not info.is_dir()}
            def read(name):
                info=infos[name]
                if info.flag_bits&1: raise Unsupported('encrypted zip member')
                if info.file_size>MAX_EXPANDED: raise Unsupported('model metadata member size limit')
                return zf.read(info)
            if loader == 'numpy.load':
                from modelgate.symbolic import Instance
                arrays={n[:-4] if n.endswith('.npy') else n:read(n) for n in infos}
                return Instance(rt.ref('numpy.lib.npyio.NpzFile'),(),{'_arrays':arrays,'_allow_pickle':options.get('allow_pickle',False),'files':list(arrays)}),len(data)
            if 'config.json' in infos and 'metadata.json' in infos:
                _emit(keras_scan.scan_config(read('config.json'),location+'!config.json'))
                return UNKNOWN,len(data)
            # Select the checkpoint's common archive prefix, not the first
            # entry whose basename happens to be data.pkl. Extra nested pickle
            # records are not loaded by torch.load.
            name=_torch_pickle(infos,data[:4])
            if loader in ('torch.load','torch.storage._load_from_bytes') and not data.startswith(b'PK\x03\x04'):
                raise Halt('torch.load requires local zip header at offset zero')
            if name is not None:
                value,_=pickle_scan.evaluate(read(name),location+'!'+name,rt,first=True,persistent=True)
                return value,len(data)
            for name,info in infos.items():
                try:
                    if info.file_size>MAX_EXPANDED:
                        with zf.open(info) as member_fh:
                            _scan_fileobj(member_fh,info.file_size,location+'!'+name,rt,depth+1,True)
                    else:
                        _scan(read(name),location+'!'+name,rt,depth+1,True)
                except Halt: pass  # each member has its own standard loader
            return UNKNOWN,len(data)
    kind=_compression(data)
    if kind:
        if kind=='gzip' and loader is None:
            remaining=data
            number=0
            while remaining.lstrip(b'\0'):
                remaining=remaining.lstrip(b'\0')
                compressed=remaining
                engine=zlib.decompressobj(31)
                chunks=[]
                expanded=0
                pending=compressed
                while True:
                    chunk=engine.decompress(pending,1024*1024)
                    expanded+=len(chunk)
                    if expanded>512*1024*1024: raise Unsupported('expanded gzip member work limit')
                    if expanded<=MAX_EXPANDED: chunks.append(chunk)
                    else: chunks.clear()
                    rt.tick()
                    if engine.eof: break
                    pending=engine.unconsumed_tail
                    if not pending: raise Halt('truncated gzip member')
                try:
                    if expanded>MAX_EXPANDED:
                        consumed=len(compressed)-len(engine.unused_data)
                        with gzip.GzipFile(fileobj=io.BytesIO(compressed[:consumed]),mode='rb') as member_fh:
                            _scan_fileobj(member_fh,expanded,location+f'!gzip[{number}]',rt,depth+1,True)
                    else:
                        _scan(b''.join(chunks),location+f'!gzip[{number}]',rt,depth+1,True)
                except Halt: pass
                remaining=engine.unused_data
                number+=1
            return UNKNOWN,len(data)
        if kind in ('bz2','lzma') and loader is None:
            cls=bz2.BZ2File if kind=='bz2' else lzma.LZMAFile
            with cls(io.BytesIO(data),'rb') as expanded_fh:
                initial=expanded_fh.read(MAX_EXPANDED+1)
                if len(initial)<=MAX_EXPANDED:
                    return _scan(initial,location+'!'+kind,rt,depth+1,member)[0],len(data)
                size=len(initial)
                del initial
                while True:
                    chunk=expanded_fh.read(1024*1024)
                    if not chunk: break
                    size+=len(chunk)
                    if size>512*1024*1024: raise Unsupported('expanded container work limit')
                    rt.tick()
                expanded_fh.seek(0)
                _scan_fileobj(expanded_fh,size,location+'!'+kind,rt,depth+1,member)
                return UNKNOWN,len(data)
        return _scan(decompress(data,kind),location+'!'+kind,rt,depth+1,member,loader,options)[0],len(data)
    if data.startswith(b'\x93NUMPY'):
        allow=options.get('allow_pickle',False) if loader=='numpy.load' else True
        return numpy_array(data,location,rt,allow,options.get('mmap_mode'))
    if _hdf(data):
        _emit(keras_scan.scan_h5(data,location))
        return UNKNOWN,len(data)
    filename=location.rsplit('!',1)[-1].replace('\\','/').rsplit('/',1)[-1]
    if filename in ('saved_model.pb','saved_model.pbtxt'):
        _emit(savedmodel_scan.scan_saved_model(data,location,filename.endswith('pbtxt')))
        return UNKNOWN,len(data)
    ext=os.path.splitext(filename)[1].lower()
    if loader in ('torch.load','torch.storage._load_from_bytes'):
        return pickle_scan.evaluate(data,location,rt,legacy=True,persistent=True)
    if (data and data[0] in PICKLE_START) or ext in PICKLE_EXTS or loader in ('joblib.load','pandas.read_pickle'):
        return pickle_scan.evaluate(data,location,rt,joblib=True,first=loader in ('joblib.load','pandas.read_pickle','numpy.load'))
    if member:
        # Ancillary metadata, numerical storage and weights have no loader
        # behavior. They are never searched for suspicious text or opcodes.
        return UNKNOWN,len(data)
    raise Unsupported('unrecognized artifact format')


def _scan_fileobj(fh,size,location,rt,depth=0,member=False):
    """Inspect large weight containers using seekable, read-only file objects.

    Numerical storage has no loader behavior. Read only its header; HDF5 and
    ZIP parsers can seek directly to configuration/pickle metadata instead.
    """
    if depth>12: raise Unsupported('container nesting limit exceeded')
    if size<=MAX_EXPANDED:
        fh.seek(0)
        return _scan(fh.read(),location,rt,depth,member)
    rt.location=location
    fh.seek(0);head=fh.read(12)
    if head.startswith(b'\x93NUMPY'):
        major=head[6]
        start=10 if major==1 else 12
        length=int.from_bytes(head[8:start],'little')
        if length>1024*1024: raise Unsupported('npy header size limit')
        fh.seek(0);header=fh.read(start+length)
        try: config=ast.literal_eval(header[start:].decode('utf-8' if major==3 else 'latin1').strip())
        except (ValueError,SyntaxError,UnicodeError): raise Halt('invalid npy header') from None
        if not isinstance(config,dict): raise Halt('invalid npy header')
        if not _object_dtype(config.get('descr')):
            numpy_array(header,location,rt)
            return UNKNOWN,size
        raise Unsupported('object pickle array exceeds analysis size limit')
    # HDF5 permits a power-of-two user block before its signature.
    hdf=head.startswith(b'\x89HDF\r\n\x1a\n')
    offset=512
    while not hdf and offset+8<=size:
        fh.seek(offset);hdf=fh.read(8)==b'\x89HDF\r\n\x1a\n'
        offset*=2
    if hdf:
        fh.seek(0)
        _emit(keras_scan.scan_h5(fh,location))
        return UNKNOWN,size
    fh.seek(max(0,size-65557));tail=fh.read(65557)
    if _zip(tail):
        fh.seek(0)
        with zipfile.ZipFile(fh) as zf:
            infos={i.filename:i for i in zf.infolist() if not i.is_dir()}
            def read(name):
                info=infos[name]
                if info.file_size>MAX_EXPANDED: raise Unsupported('model metadata member size limit')
                return zf.read(info)
            if 'config.json' in infos and 'metadata.json' in infos:
                _emit(keras_scan.scan_config(read('config.json'),location+'!config.json'))
                return UNKNOWN,size
            name=_torch_pickle(infos,head)
            if name is not None:
                value,_=pickle_scan.evaluate(read(name),location+'!'+name,rt,first=True,persistent=True)
                return value,size
            for name,info in infos.items():
                try:
                    with zf.open(info) as member_fh:
                        _scan_fileobj(member_fh,info.file_size,location+'!'+name,rt,depth+1,True)
                except Halt: pass
        return UNKNOWN,size
    raise Unsupported('large artifact needs an unsupported in-memory analysis')


def load_content(data,location,rt,loader=None,options=None):
    return _scan(data,location,rt,loader=loader,options=options)


def scan_path(path: str) -> Result:
    rt=Runtime(path,directory=os.path.isdir(path))
    rt.load_callback=load_content
    try:
        if rt.directory:
            return Result(path,savedmodel_scan.scan_saved_model_dir(path))
        with open(path,'rb') as fh:
            size=os.fstat(fh.fileno()).st_size
            if size>MAX_EXPANDED:
                _scan_fileobj(fh,size,os.path.basename(path),rt)
                return Result(path,[])
            data=fh.read(MAX_EXPANDED+1)
        rt.artifact_data=data
        _scan(data,os.path.basename(path),rt)
        return Result(path,[])
    except Detected as exc:
        return Result(path,[exc.finding])
    except Halt:
        return Result(path,[])
    except Unsupported as exc:
        return Result(path,[Finding('unsupported',rt.location or path,str(exc))])
