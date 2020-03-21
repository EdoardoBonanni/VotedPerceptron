import os
import gzip
import numpy as np


def load_MNist_train(path):
    #Load zalando data from `path
    labels_path = os.path.join(path, 'train-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, 'train-images-idx3-ubyte.gz')
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels


def load_MNist_t10k(path):
    #Load zalando data from `path
    labels_path = os.path.join(path, 't10k-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, 't10k-images-idx3-ubyte.gz')
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels