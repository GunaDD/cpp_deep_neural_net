import numpy as np, struct, gzip, pathlib, sys

def read_idx(path):
    with gzip.open(path, 'rb') if path.name.endswith('.gz') else open(path, 'rb') as f:
        code, = struct.unpack('>I', f.read(4))
        if code == 0x00000803:  # images
            n, rows, cols = struct.unpack('>III', f.read(12))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows*cols)
        elif code == 0x00000801:  # labels
            n, = struct.unpack('>I', f.read(4))
            return np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError('bad magic')

root = pathlib.Path('./data/mnist')
x = read_idx(root/'t10k-images-idx3-ubyte')
y = read_idx(root/'t10k-labels-idx1-ubyte')
x.astype('float32').tofile(root/'t10k-images.f32')   
y.tofile(root/'t10k-labels.u8')        

p = read_idx(root/'train-images-idx3-ubyte')
q = read_idx(root/'train-labels-idx1-ubyte')
p.astype('float32').tofile(root/'train-images.f32')
q.tofile(root/'train-labels.u8')
