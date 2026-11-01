"""Static pickle stack machine, including concatenated and joblib streams.

This module deliberately does not use an Unpickler. GLOBAL produces symbolic
references; every call and BUILD is interpreted by modelgate.symbolic.
"""
from __future__ import annotations

import io
import math
import pickletools
import struct
from modelgate.symbolic import (Runtime, Ref, Instance, Partial, Unknown, UNKNOWN,
                                Halt, Unsupported, Detected, classname)
from modelgate.report import Finding

MARK = object()
TORCH_MAGIC = 0x1950A86A20F9469CFC6C


def evaluate(data: bytes, location: str, runtime: Runtime, *, first=False,
             persistent=False, legacy=False, joblib=False, fix_imports=True):
    """Return (value, bytes consumed), without executing artifact code.

    Known loader exceptions terminate evaluation. The caller chooses whether
    those are the end of this artifact or an exception in a surrounding load.
    """
    fh = io.BytesIO(data)
    value = UNKNOWN
    stream = 0
    legacy_candidate = False
    limit = 5 if legacy else 1 if first else None
    while fh.tell() < len(data) and (limit is None or stream < limit):
        start = fh.tell()
        value = _stream(fh, location, runtime, stream, persistent, joblib, fix_imports)
        stream += 1
        if legacy and ((stream == 1 and value != TORCH_MAGIC) or (stream == 2 and value != 1001)):
            raise Halt('invalid torch legacy header')
        if fh.tell() <= start:
            raise Unsupported('pickle parser made no progress')
        # Legacy torch files end in raw storage, not another pickle.
        if not first and not legacy:
            if stream == 1:
                legacy_candidate = isinstance(value,int) and value == TORCH_MAGIC
            elif stream == 2 and legacy_candidate and value == 1001:
                limit = 5
                persistent = True
    return value, fh.tell()


def _stream(fh, location, rt, stream, persistent, joblib, fix_imports=True):
    stack, memo = [], {}
    proto = 0
    frame_end = None
    context=(stack,memo)
    rt.vm_roots.append(context)

    def pop():
        if not stack: raise Halt('pickle stack underflow')
        value=stack.pop()
        if value is MARK: raise Halt('unexpected pickle MARK')
        return value

    def marked():
        for i in range(len(stack)-1,-1,-1):
            if stack[i] is MARK:
                result=stack[i+1:]; del stack[i:]; return result
        raise Halt('pickle MARK missing')

    def pairs(values):
        if len(values)%2: raise Halt('odd dict items')
        return zip(values[::2],values[1::2])

    try:
        for op,arg,pos in pickletools.genops(fh):
            rt.tick()
            name=op.name
            if frame_end is not None and pos<frame_end and fh.tell()>frame_end:
                raise Halt('pickle opcode crosses frame boundary')
            rt.location=f'{location}:stream[{stream}]@{pos} {name}'
            if name=='PROTO':
                if arg>5: raise Unsupported('unsupported pickle protocol')
                proto=arg
            elif name=='FRAME':
                if frame_end is not None and pos<frame_end: raise Halt('FRAME before previous frame ended')
                if arg>len(fh.getbuffer())-fh.tell(): raise Halt('truncated pickle frame')
                frame_end=fh.tell()+arg
            elif name=='MARK': stack.append(MARK)
            elif name in ('NONE','NEWTRUE','NEWFALSE'):
                stack.append({'NONE':None,'NEWTRUE':True,'NEWFALSE':False}[name])
            elif name in ('INT','BININT','BININT1','BININT2','LONG','LONG1','LONG4','FLOAT','BINFLOAT',
                          'STRING','BINSTRING','SHORT_BINSTRING','UNICODE','BINUNICODE',
                          'SHORT_BINUNICODE','BINUNICODE8','BINBYTES','SHORT_BINBYTES','BINBYTES8'):
                stack.append(arg)
            elif name=='BYTEARRAY8': stack.append(bytearray(arg))
            elif name=='EMPTY_LIST': stack.append([])
            elif name=='EMPTY_TUPLE': stack.append(())
            elif name=='EMPTY_DICT': stack.append({})
            elif name=='EMPTY_SET': stack.append(set())
            elif name=='LIST': stack.append(marked())
            elif name=='TUPLE': stack.append(tuple(marked()))
            elif name in ('TUPLE1','TUPLE2','TUPLE3'):
                n=int(name[-1]); values=[pop() for _ in range(n)][::-1]; stack.append(tuple(values))
            elif name=='DICT':
                obj={}
                for key,value in pairs(marked()): rt.setitem(obj,key,value)
                stack.append(obj)
            elif name=='FROZENSET':
                values=marked()
                for value in values: rt.hash_key(value)
                stack.append(frozenset(values))
            elif name in ('APPEND','APPENDS'):
                values=[pop()] if name=='APPEND' else marked()
                if not stack: raise Halt('missing append target')
                obj=stack[-1]
                if isinstance(obj,list): obj.extend(values)
                else:
                    # CPython calls extend for APPENDS, append for APPEND.
                    rt.call(rt.getattr(obj,'append' if name=='APPEND' else 'extend'),
                            (values[0],) if name=='APPEND' else (values,))
            elif name in ('SETITEM','SETITEMS'):
                if name=='SETITEM':
                    value=pop(); key=pop(); items=[(key,value)]
                else: items=list(pairs(marked()))
                if not stack: raise Halt('missing setitem target')
                for key,value in items: rt.setitem(stack[-1],key,value)
            elif name=='ADDITEMS':
                items=marked()
                if not stack: raise Halt('missing set target')
                if isinstance(stack[-1],set):
                    for value in items: rt.hash_key(value)
                    stack[-1].update(items)
                else:
                    for value in items: rt.call(rt.getattr(stack[-1],'add'),(value,))
            elif name in ('PUT','BINPUT','LONG_BINPUT','MEMOIZE'):
                if not stack or stack[-1] is MARK: raise Halt('cannot memoize missing value')
                index=len(memo) if name=='MEMOIZE' else int(arg)
                if index<0: raise Halt('negative memo index')
                memo[index]=stack[-1]
            elif name in ('GET','BINGET','LONG_BINGET'):
                if int(arg) not in memo: raise Halt('missing pickle memo entry')
                stack.append(memo[int(arg)])
            elif name=='DUP':
                if not stack: raise Halt('empty stack DUP')
                stack.append(stack[-1])
            elif name=='POP':
                if not stack: raise Halt('empty stack POP')
                stack.pop()
            elif name=='POP_MARK': marked()
            elif name=='GLOBAL':
                module,attribute=arg.split(' ',1)
                stack.append(rt.global_ref(module,attribute,proto,fix_imports))
            elif name=='STACK_GLOBAL':
                attribute=pop(); module=pop()
                if isinstance(module,Unknown) or isinstance(attribute,Unknown): raise Unsupported('unresolved STACK_GLOBAL name')
                if type(module) is not str or type(attribute) is not str: raise Halt('STACK_GLOBAL needs exact strings')
                stack.append(rt.global_ref(module,attribute,proto,fix_imports))
            elif name=='REDUCE':
                args=pop(); fn=pop()
                stack.append(rt.call(fn,args))
            elif name in ('NEWOBJ','NEWOBJ_EX'):
                kw=pop() if name=='NEWOBJ_EX' else {}
                args=pop(); cls=pop()
                if not isinstance(kw,dict): raise Halt('NEWOBJ_EX needs dict kwargs')
                stack.append(rt.call(cls,args,kw,new_only=True))
            elif name in ('INST','OBJ'):
                values=marked()
                if name=='INST':
                    module,attribute=arg.split(' ',1); cls=rt.global_ref(module,attribute,proto,fix_imports)
                else:
                    if not values: raise Halt('OBJ needs class/callable')
                    cls,values=values[0],values[1:]
                stack.append(rt.call(cls,tuple(values),new_only=not values and rt.isclass(cls)))
            elif name=='BUILD':
                state=pop()
                if not stack: raise Halt('BUILD needs instance')
                obj=stack[-1]; rt.build(obj,state)
                if joblib and classname(obj) in ('joblib.numpy_pickle.NumpyArrayWrapper',
                                                 'joblib.numpy_pickle_compat.NDArrayWrapper',
                                                 'joblib.numpy_pickle_compat.ZNDArrayWrapper'):
                    stack[-1]=_joblib_array(fh,obj,rt,location,stream)
            elif name in ('PERSID','BINPERSID'):
                pid=pop() if name=='BINPERSID' else arg
                if not persistent: raise Halt('no persistent-id handler')
                stack.append(Instance(rt.ref('torch.Storage'),(pid,)))
            elif name in ('EXT1','EXT2','EXT4'):
                # No extensions can be registered by an artifact without an
                # earlier call. Account for those explicitly registered here.
                registry=getattr(rt,'extensions',{})
                if arg not in registry: raise Halt('unregistered pickle extension')
                module,attribute=registry[arg]
                stack.append(rt.global_ref(module,attribute,proto,fix_imports))
            elif name=='NEXT_BUFFER':
                raise Halt('no out-of-band buffers supplied by standard loader')
            elif name=='READONLY_BUFFER':
                value=pop()
                if not isinstance(value,(bytes,bytearray)): raise Halt('value does not expose a buffer')
                stack.append(bytes(value))
            elif name=='STOP':
                # pickle.load reads the entire frame into its private buffer.
                # A fresh pickle.load on the underlying file starts after it.
                if frame_end is not None and fh.tell()<frame_end: fh.seek(frame_end)
                value=pop()
                if len(rt.vm_roots)==1: rt.returned_roots.append(value)
                rt.collect_finalizers((value,),ignore=context)
                return value
            else:
                raise Unsupported('unmodelled pickle opcode '+name)
            rt.collect_finalizers()
    except Halt:
        rt.collect_finalizers(ignore=context)
        raise
    except (Detected,Unsupported):
        raise
    except (ValueError,IndexError,KeyError,TypeError,OverflowError,UnicodeError) as exc:
        # This is a parse/stack/type error, not an unresolved callable.
        raise Halt('invalid pickle: '+str(exc)) from None
    finally:
        rt.vm_roots.pop()
    raise Halt('pickle has no STOP')


def _dtype_info(dtype):
    if isinstance(dtype,Instance) and dtype.args:
        state=dtype.attrs.get('_pickle_state')
        if isinstance(state,tuple) and len(state)>=8 and isinstance(state[-1],int) and state[-1]&1:
            return True,8
        spec=dtype.args[0]
        if isinstance(spec,str):
            import re
            text=spec.lstrip('<>=|')
            if text.startswith('O'): return True,8
            m=re.fullmatch(r'([?bBiIuUfFcCmMSUVa])([0-9]*)',text)
            if m:
                n=int(m[2] or 1)
                return False,n*(4 if m[1]=='U' else 1)
        state=dtype.attrs.get('_pickle_state')
        if isinstance(state,tuple) and len(state)>5 and isinstance(state[5],int) and state[5]>0:
            return False,state[5]
    raise Unsupported('cannot determine joblib array dtype')


def _joblib_array(fh,obj,rt,location,stream):
    state=obj.attrs
    if classname(obj).endswith(('.NDArrayWrapper','.ZNDArrayWrapper')):
        rt.fail('file-access','legacy joblib array wrapper loads an external file')
    shape=state.get('shape')
    if not isinstance(shape,tuple) or not all(isinstance(n,int) and n>=0 for n in shape):
        raise Halt('invalid joblib array shape')
    is_object,itemsize=_dtype_info(state.get('dtype'))
    if is_object:
        return _stream(fh,location+'!object-array',rt,stream,False,False)
    if state.get('numpy_array_alignment_bytes') is not None:
        raw=fh.read(1)
        if not raw: raise Halt('missing joblib array alignment')
        fh.seek(raw[0],1)
    size=math.prod(shape)*itemsize
    if size>len(fh.getbuffer())-fh.tell(): raise Halt('truncated joblib array data')
    fh.seek(size,1)
    return Instance(rt.ref('numpy.ndarray'),(shape,state.get('dtype')))


def scan_pickle(data: bytes, location: str, **options) -> list[Finding]:
    rt=options.pop('runtime',None) or Runtime()
    try:
        evaluate(data,location,rt,**options)
        return []
    except Detected as exc:
        return [exc.finding]
    except Halt:
        return []
    except Unsupported as exc:
        return [Finding('unsupported',rt.location or location,str(exc))]
