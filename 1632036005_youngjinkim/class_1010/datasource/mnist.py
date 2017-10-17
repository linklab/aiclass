# -*- coding:utf-8 -*-

import numpy as np
import datasource.data as data
import os
import gzip

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'
MNIST_DIR = ROOT_DIR + "mnist"
FASHION_MNIST_DIR = ROOT_DIR + "fashion_mnist"

class MNIST_Data(data.Base_Data):
    # http://yann.lecun.com/exdb/mnist/
    def __init__(self, validation_size=5000):
        images, labels = load_mnist(path=MNIST_DIR, kind='train')

        self.validation_input = images[:validation_size]
        self.validation_target = labels[:validation_size]

        self.train_input = images[validation_size:]
        self.train_target = labels[validation_size:]

        self.test_input, self.test_target = load_mnist(path=MNIST_DIR, kind='t10k')

        self.labels = ['Zero', 'One', 'Two', 'Three', 'Four',
                       'Five', 'Six', 'Seven', 'Eight', 'Nine']

        super().__init__()


class Fashion_MNIST_Data(data.Base_Data):
    # https://github.com/zalandoresearch/fashion-mnist
    def __init__(self, validation_size=5000):
        images, labels = load_mnist(path=FASHION_MNIST_DIR, kind='train')

        self.validation_input = images[:validation_size]
        self.validation_target = labels[:validation_size]

        self.train_input = images[validation_size:]
        self.train_target = labels[validation_size:]

        self.test_input, self.test_target = load_mnist(path=FASHION_MNIST_DIR, kind='t10k')

        self.labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat',
                       'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

        super().__init__()


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels