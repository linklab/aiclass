# -*- coding:utf-8 -*-

import numpy as np
import datasource.data as data
import os
import gzip

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
MNIST_DIR = ROOT_DIR + "mnist"
FASHION_MNIST_DIR = ROOT_DIR + "fashion_mnist"


def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array([1, 0, 4])
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    t = np.zeros((vector.size, num_classes), dtype=np.float64)
    for idx, row in enumerate(t):
        row[int(vector[idx])] = 1.0
    return t


class MNIST_Data(data.Base_Data):
    # http://yann.lecun.com/exdb/mnist/
    def __init__(self, validation_size=5000, is_onehot_target=True):
        images, labels = load_mnist(path=MNIST_DIR, kind='train')

        self.validation_input = images[:validation_size]
        self.validation_target = labels[:validation_size]

        self.train_input = images[validation_size:]
        self.train_target = labels[validation_size:]

        self.test_input, self.test_target = load_mnist(path=MNIST_DIR, kind='t10k')

        self.labels = ['Zero', 'One', 'Two', 'Three', 'Four',
                       'Five', 'Six', 'Seven', 'Eight', 'Nine']

        if is_onehot_target:
            self.train_target = convertToOneHot(self.train_target, num_classes=10)
            self.validation_target = convertToOneHot(self.validation_target, num_classes=10)
            self.test_target = convertToOneHot(self.test_target, num_classes=10)

        super().__init__()


class Fashion_MNIST_Data(data.Base_Data):
    # https://github.com/zalandoresearch/fashion-mnist
    def __init__(self, validation_size=5000, is_onehot_target=True):
        images, labels = load_mnist(path=FASHION_MNIST_DIR, kind='train')

        self.validation_input = images[:validation_size]
        self.validation_target = labels[:validation_size]

        self.train_input = images[validation_size:]
        self.train_target = labels[validation_size:]

        self.test_input, self.test_target = load_mnist(path=FASHION_MNIST_DIR, kind='t10k')

        self.labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat',
                       'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

        if is_onehot_target:
            self.train_target = convertToOneHot(self.train_target, num_classes=10)
            self.validation_target = convertToOneHot(self.validation_target, num_classes=10)
            self.test_target = convertToOneHot(self.test_target, num_classes=10)

        super().__init__()


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    labels = labels.astype(np.float64, copy=False)
    images = images.astype(np.float64, copy=False)

    return images, labels
