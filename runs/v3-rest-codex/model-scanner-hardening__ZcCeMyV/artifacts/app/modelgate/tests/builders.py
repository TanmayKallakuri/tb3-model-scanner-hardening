"""Construct pickle byte streams as data. No test ever unpickles a payload."""
import struct
import base64
import marshal

# Serialized trusted test code; fixtures never execute it.
CODE_BYTES = marshal.dumps((lambda x: x).__code__)
CODE_BASE64 = base64.b64encode(CODE_BYTES).decode("ascii")
from dataclasses import dataclass

@dataclass
class Global:
    module: str
    name: str

@dataclass
class Call:
    fn: object
    args: object = ()

@dataclass
class New:
    cls: object
    args: tuple = ()
    kw: object = None

@dataclass
class Build:
    obj: object
    state: object

@dataclass
class Inst:
    cls: Global
    args: tuple = ()
    obj: bool = False

@dataclass
class Memo:
    obj: object
    index: int = 0

@dataclass
class Get:
    index: int = 0

@dataclass
class Raw:
    data: bytes


def emit(v,proto=4):
    e=lambda x:emit(x,proto)
    if isinstance(v,Raw): return v.data
    if isinstance(v,Global): return b'c'+v.module.encode()+b'\n'+v.name.encode()+b'\n'
    if isinstance(v,Call): return e(v.fn)+e(v.args)+b'R'
    if isinstance(v,New): return e(v.cls)+e(v.args)+(b'\x81' if v.kw is None else e(v.kw)+b'\x92')
    if isinstance(v,Build): return e(v.obj)+e(v.state)+b'b'
    if isinstance(v,Inst):
        if v.obj: return b'('+e(v.cls)+b''.join(map(e,v.args))+b'o'
        return b'('+b''.join(map(e,v.args))+b'i'+v.cls.module.encode()+b'\n'+v.cls.name.encode()+b'\n'
    if isinstance(v,Memo): return e(v.obj)+b'p'+str(v.index).encode()+b'\n'
    if isinstance(v,Get): return b'g'+str(v.index).encode()+b'\n'
    if v is None: return b'N'
    if v is True: return b'I01\n'
    if v is False: return b'I00\n'
    if isinstance(v,int): return b'I'+str(v).encode()+b'\n'
    if isinstance(v,float): return b'F'+repr(v).encode()+b'\n'
    if isinstance(v,str):
        if proto==0: return b'V'+v.encode('raw_unicode_escape').replace(b'\n',b'\\u000a')+b'\n'
        raw=v.encode(); return b'X'+struct.pack('<I',len(raw))+raw
    if isinstance(v,bytearray):
        if proto>=5: return b'\x96'+struct.pack('<Q',len(v))+bytes(v)
        return e(Call(Global('builtins','bytearray'),(bytes(v),)))
    if isinstance(v,bytes):
        if proto>=3: return b'B'+struct.pack('<I',len(v))+v
        return e(Call(Global('_codecs','encode'),(v.decode('latin1'),'latin1')))
    if isinstance(v,tuple): return b'('+b''.join(map(e,v))+b't'
    if isinstance(v,list): return b'('+b''.join(map(e,v))+b'l'
    if isinstance(v,dict): return b'('+b''.join(e(k)+e(val) for k,val in v.items())+b'd'
    raise TypeError(type(v))


def dump(v,proto=4):
    return (b'\x80'+bytes([proto]) if proto else b'')+emit(v,proto)+b'.'


def G(path):
    mod,name=path.rsplit('.',1)
    return Global(mod,name)


def C(path,*args):
    return Call(G(path),args)


def attr(obj,name):
    return C('builtins.getattr',obj,name)
