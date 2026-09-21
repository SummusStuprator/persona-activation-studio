"""Read-only tiled unembedding from the original GGUF, never a weight-file copy."""
from pathlib import Path
import ctypes as C
import struct
import numpy as np
from threadpoolctl import threadpool_limits


def head_tensor(path,dim,vocab):
    size=Path(path).stat().st_size
    with open(path,'rb',buffering=1024*1024) as f:
        def number(fmt):
            n=struct.calcsize('<'+fmt);data=f.read(n)
            if len(data)!=n:raise ValueError('Truncated GGUF header.')
            return struct.unpack('<'+fmt,data)[0]
        def string(keep=True):
            n=number('Q')
            if n>size-f.tell():raise ValueError('Invalid GGUF string size.')
            if keep:return f.read(n).decode('utf-8','replace')
            f.seek(n,1)
        formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
        def value(kind):
            if kind in formats:return number(formats[kind])
            if kind==8:string(False);return None
            if kind!=9:raise ValueError('Unknown metadata type.')
            sub,count=number('I'),number('Q')
            if count>10000000:raise ValueError('Invalid metadata count.')
            if sub in formats:f.seek(struct.calcsize('<'+formats[sub])*count,1)
            else:
                for _ in range(count):value(sub)
        if f.read(4)!=b'GGUF' or number('I') not in (2,3):raise ValueError('Expected little-endian GGUF.')
        nt,nk=number('Q'),number('Q');alignment=32
        for _ in range(nk):
            key=string();item=value(number('I'))
            if key=='general.alignment':alignment=int(item)
        heads={}
        for _ in range(nt):
            name=string();rank=number('I');shape=[number('Q') for _ in range(rank)];kind=number('I');offset=number('Q')
            if name in ('output.weight','token_embd.weight'):heads[name]=(shape,kind,offset)
        name='output.weight' if 'output.weight' in heads else 'token_embd.weight'
        if name not in heads:raise ValueError('GGUF output/tied embedding tensor not found.')
        shape,kind,offset=heads[name]
        if shape!=[dim,vocab]:raise ValueError(f'Unembedding shape mismatch: {shape}, expected {[dim,vocab]}.')
        start=(f.tell()+alignment-1)//alignment*alignment
        return name,kind,start+offset,size


class Traits(C.Structure):
    # Exact public struct at pinned llama.cpp c0bc8591e.
    _fields_=[('type_name',C.c_char_p),('blck_size',C.c_int64),('interleave',C.c_int64),
              ('type_size',C.c_size_t),('quantized',C.c_bool),('to_float',C.c_void_p),('from_float',C.c_void_p)]


def project(engine,directions,progress=None):
    matrix=np.asarray(directions,np.float32)
    if matrix.ndim==1:matrix=matrix[:,None]
    if matrix.shape[0]!=engine.dim or not np.isfinite(matrix).all():raise ValueError('Invalid head projection matrix.')
    root=Path(__file__).resolve().parent.parent
    dll=C.CDLL(str(root/'native'/'runtime'/'ggml-base.dll'))
    dll.ggml_get_type_traits.argtypes=[C.c_int];dll.ggml_get_type_traits.restype=C.POINTER(Traits)
    dll.ggml_row_size.argtypes=[C.c_int,C.c_int64];dll.ggml_row_size.restype=C.c_size_t
    name,kind,offset,size=head_tensor(engine.model['path'],engine.dim,engine.vocab_size)
    traits=dll.ggml_get_type_traits(kind).contents
    row_bytes=dll.ggml_row_size(kind,engine.dim)
    if offset+row_bytes*engine.vocab_size>size:raise ValueError('Unembedding extends beyond GGUF.')
    if kind not in (0,1) and not traits.to_float:raise ValueError('This quantization has no native dequantizer.')
    convert=C.CFUNCTYPE(None,C.c_void_p,C.c_void_p,C.c_int64)(traits.to_float) if traits.to_float else None
    out=np.empty((engine.vocab_size,matrix.shape[1]),np.float32)
    with open(engine.model['path'],'rb') as f,threadpool_limits(limits=4):
        f.seek(offset)
        for start in range(0,engine.vocab_size,128):
            count=min(128,engine.vocab_size-start);raw=f.read(row_bytes*count)
            if len(raw)!=row_bytes*count:raise ValueError('Truncated unembedding tile.')
            if kind in (0,1):tile=np.frombuffer(raw,dtype='<f4' if kind==0 else '<f2').astype(np.float32).reshape(count,engine.dim)
            else:
                tile=np.empty((count,engine.dim),np.float32);buffer=C.create_string_buffer(raw)
                convert(buffer,tile.ctypes.data,count*engine.dim)
            out[start:start+count]=tile@matrix
            if progress and (start%4096==0 or start+count==engine.vocab_size):progress(start+count,engine.vocab_size,'Tiled read-only GGUF unembedding; no weight-file copies')
    return out,{'tensor':name,'quantization':traits.type_name.decode(),'source':engine.model['path'],
                'method':'Native dequantization of <=128-row temporary tiles followed by float32 matrix multiplication.',
                'weight_files_written':False,'maximum_tile_float_bytes':128*engine.dim*4}
