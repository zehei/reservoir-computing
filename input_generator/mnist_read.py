import numpy as np
import struct
import gzip

train_image='input_generator/mnist_data/train-images-idx3-ubyte.gz'
train_label='input_generator/mnist_data/train-labels-idx1-ubyte.gz'
test_image='input_generator/mnist_data/t10k-images-idx3-ubyte.gz'
test_label='input_generator/mnist_data/t10k-labels-idx1-ubyte.gz'

def read_image(fz):
    with gzip.open(fz, 'rb') as f:
        header=f.read(16)
        mn,num,nrow,ncol=struct.unpack('>4i',header)
        assert mn == 2051
        im=np.empty((num,nrow,ncol))
        npixel=nrow*ncol
        for i in range(num):
            buf=struct.unpack('>%dB' % npixel, f.read(npixel))
            im[i,:,:]=np.asarray(buf).reshape((nrow,ncol))
    return im

def read_label(fz):
    with gzip.open(fz,'rb') as f:
        header=f.read(8)
        mn,num=struct.unpack('>2i', header)
        assert mn == 2049
        label = np.array(struct.unpack('>%dB' % num, f.read()),dtype=int)
    return label