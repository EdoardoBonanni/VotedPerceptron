import os
import gzip
import numpy as np


def load_MNist_train(path):
    #Load zalando data from `path
    labels_path = os.path.join(path, 'train-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, 'train-images-idx3-ubyte.gz')
    with gzip.open(labels_path, 'rb') as lbpath:  # read labels
        lbpath.read(8)
        # create a buffer by reading a line of the file through the read() function
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
        # after reading all the lines the buffer will be the complete file
    with gzip.open(images_path, 'rb') as imgpath:  # read images
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer,
                               dtype=np.uint8).reshape(len(labels),
                                                       784).astype(np.float32)
    return images, labels


def load_MNist_t10k(path):
    #Load zalando data from `path
    labels_path = os.path.join(path, 't10k-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, 't10k-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath: #read labels
        lbpath.read(8)
        # create a buffer by reading a line of the file through the read() function
        buffer = lbpath.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
        # after reading all the lines the buffer will be the complete file
    with gzip.open(images_path, 'rb') as imgpath: #read images
        imgpath.read(16)
        buffer = imgpath.read()
        images = np.frombuffer(buffer,
                               dtype=np.uint8).reshape(len(labels),
                                                       784).astype(np.float32)
    return images, labels
