import os # contains functions for intercaing with operating system
import gzip # gzip is a simple interface to compress and decompress files
import numpy as np # fondamental package for scentific computing with python


def load_zalando_train(path):
    # Load zalando data train from path
    # os.path.join() allows to join path
    labels_path = os.path.join(path, 'train-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, 'train-images-idx3-ubyte.gz')
    # gzip.open() allows to open a file compressed
    with gzip.open(labels_path, 'rb') as lbpath: #read labels
        # create a buffer by reading a line of the file through the read() function
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
        # after reading all the lines the buffer will be the complete file
    with gzip.open(images_path, 'rb') as imgpath: #read images
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels


def load_zalando_t10k(path):
    # Load zalando data test from path
    labels_path = os.path.join(path, 't10k-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, 't10k-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels
