"""Streaming GGUF v2/v3 header inspection; tensor payloads are never read."""
import struct
from pathlib import Path
FORMATS = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}

def read_header(path):
    size = Path(path).stat().st_size
    with open(path, 'rb', buffering=1024*1024) as f:
        def number(fmt):
            n = struct.calcsize('<'+fmt); data = f.read(n)
            if len(data) != n: raise ValueError('Truncated GGUF header.')
            return struct.unpack('<'+fmt, data)[0]
        def string(keep=True):
            n = number('Q')
            if n > size-f.tell(): raise ValueError('Invalid GGUF string length.')
            if keep:
                if n > 4*1024*1024: raise ValueError('Oversized GGUF metadata string.')
                return f.read(n).decode('utf-8', 'replace')
            f.seek(n, 1)
        def value(kind, keep):
            if kind in FORMATS:
                result = number(FORMATS[kind]); return result if keep else None
            if kind == 8: return string(keep)
            if kind != 9: raise ValueError('Unknown GGUF metadata type.')
            element, count = number('I'), number('Q')
            if count > 10000000: raise ValueError('Oversized GGUF array.')
            if element in FORMATS:
                count_bytes = count*struct.calcsize('<'+FORMATS[element])
                if count_bytes > size-f.tell(): raise ValueError('Truncated GGUF array.')
                f.seek(count_bytes, 1)
            else:
                for _ in range(count): value(element, False)
            return None
        if f.read(4) != b'GGUF': raise ValueError('Not a GGUF weight file.')
        version = number('I')
        if version not in (2,3): raise ValueError(f'Unsupported GGUF version {version}.')
        tensors, count = number('Q'), number('Q')
        if tensors > 1000000 or count > 1000000: raise ValueError('Invalid GGUF counts.')
        fields = {}
        for _ in range(count):
            key, kind = string(), number('I')
            keep = kind != 9 and (kind != 8 or key in ('general.architecture','tokenizer.chat_template'))
            item = value(kind, keep)
            if keep: fields[key] = item
        entries = []
        for _ in range(tensors):
            name, dimensions = string(), number('I')
            if not 1 <= dimensions <= 8: raise ValueError('Invalid tensor rank.')
            shape = [number('Q') for _ in range(dimensions)]
            kind, offset = number('I'), number('Q')
            entries.append((offset, name))
        alignment = int(fields.get('general.alignment',32))
        if alignment <= 0 or alignment > 1048576: raise ValueError('Invalid GGUF alignment.')
        start = (f.tell()+alignment-1)//alignment*alignment
        entries.sort()
        weights = []
        for i, (offset,name) in enumerate(entries):
            end = entries[i+1][0] if i+1 < len(entries) else size-start
            if offset > end or start+end > size: raise ValueError('Invalid GGUF tensor offset.')
            weights.append((name, end-offset))
    return fields, weights
