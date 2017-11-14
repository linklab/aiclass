# -*- coding:utf-8 -*-
# https://medium.com/towards-data-science/train-test-split-and-cross-validation-in-python-80b61beca4b6

import numpy as np
import datasource.data as data
import os
import gzip
from sklearn.model_selection import KFold

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
    def __init__(self, validation_size=5000, n_splits=1, is_onehot_target=True, cnn=False):
        super().__init__(n_splits)

        self.validation_size = validation_size
        self.n_splits = n_splits

        self.images, self.targets = load_mnist(path=MNIST_DIR, kind='train', cnn=cnn)
        self.test_input, self.test_target = load_mnist(path=MNIST_DIR, kind='t10k', cnn=cnn)
        self.num_test_data = len(self.test_input)

        self.labels = ['Zero', 'One', 'Two', 'Three', 'Four',
                       'Five', 'Six', 'Seven', 'Eight', 'Nine']

        if is_onehot_target:
            self.targets = convertToOneHot(self.targets, num_classes=10)

        self.reset_kfold()

        self.num_train_data = len(self.images) - validation_size
        self.num_validation_data = validation_size

    def reset_kfold(self):
        if self.n_splits == 1:
            self.validation_input = self.images[:self.validation_size]
            self.validation_target = self.targets[:self.validation_size]

            self.train_input = self.images[self.validation_size:]
            self.train_target = self.targets[self.validation_size:]
        else:
            kf = KFold(n_splits=self.n_splits)
            self.splitted_indices = kf.split(self.images)

class Fashion_MNIST_Data(data.Base_Data):
    # https://github.com/zalandoresearch/fashion-mnist
    def __init__(self, validation_size=5000, n_splits=1, is_onehot_target=True, cnn=False):
        super().__init__(n_splits)

        self.validation_size = validation_size
        self.n_splits = n_splits

        self.images, self.targets = load_mnist(path=FASHION_MNIST_DIR, kind='train', cnn=cnn)
        self.test_input, self.test_target = load_mnist(path=MNIST_DIR, kind='t10k', cnn=cnn)
        self.num_test_data = len(self.test_input)

        self.labels = ['t_shirt_top', 'trouser', 'pullover', 'dress', 'coat',
                       'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

        if is_onehot_target:
            self.targets = convertToOneHot(self.targets, num_classes=10)

        self.reset_kfold()

        self.num_train_data = len(self.images) - validation_size
        self.num_validation_data = validation_size

    def reset_kfold(self):
        if self.n_splits == 1:
            self.validation_input = self.images[:self.validation_size]
            self.validation_target = self.targets[:self.validation_size]

            self.train_input = self.images[self.validation_size:]
            self.train_target = self.targets[self.validation_size:]
        else:
            kf = KFold(n_splits=self.n_splits)
            self.splitted_indices = kf.split(self.images)


def load_mnist(path, kind='train', cnn=False):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        if cnn:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 1, 28, 28)
        else:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    labels = labels.astype(np.float64, copy=False)
    images = images.astype(np.float64, copy=False)

    return images, labels


if __name__ == '__main__':
    data = MNIST_Data(validation_size=5000, n_splits=12, is_onehot_target=True, cnn=True)
    #data = Fashion_MNIST_Data(validation_size=5000, n_splits=12, is_onehot_target=True)

    print("N_Splits:", data.n_splits)
    for i in range(12):
        data.set_next_train_and_validation_data()
        print(i)
        print(data.train_input.shape, np.sum(data.train_input))
        print(data.train_target.shape, np.sum(data.train_target))
        print()
        print(data.validation_input.shape, np.sum(data.validation_input))
        print(data.validation_target.shape, np.sum(data.validation_target))
        print()
        print(data.test_input.shape, np.sum(data.test_input))
        print(data.test_target.shape, np.sum(data.test_target))
        print()
