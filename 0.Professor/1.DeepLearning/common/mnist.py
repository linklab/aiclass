# coding: utf-8
import urllib.request
import os.path
import gzip
import pickle
import os
import numpy as np

class mnist_data:
    def __init__(self, dataset_dir="/Users/yhhan/git/aiclass/0.Professor/3.VanillaNN/MNIST_data/."):
        self.url_base = 'http://yann.lecun.com/exdb/mnist/'
        self.key_file = {
            'train_img':'train-images-idx3-ubyte.gz',
            'train_label':'train-labels-idx1-ubyte.gz',
            'test_img':'t10k-images-idx3-ubyte.gz',
            'test_label':'t10k-labels-idx1-ubyte.gz'
        }

        self.dataset_dir = os.path.dirname(dataset_dir)
        self.save_file = dataset_dir + "/mnist.pkl"

        self.train_num = 60000
        self.test_num = 10000
        self.img_dim = (1, 28, 28)
        self.img_size = 784

    def _download(self, file_name):
        file_path = self.dataset_dir + "/" + file_name
        print(file_path)
        if os.path.exists(file_path):
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(self.url_base + file_name, file_path)
        print("Done")

    def download_mnist(self):
        for v in self.key_file.values():
            self._download(v)

    def _load_label(self, file_name):
        file_path = self.dataset_dir + "/" + file_name

        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        print("Done")

        return labels

    def _load_img(self, file_name):
        file_path = self.dataset_dir + "/" + file_name

        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, self.img_size)
        print("Done")

        return data

    def _convert_numpy(self):
        dataset = {}
        dataset['train_img'] = self._load_img(self.key_file['train_img'])
        dataset['train_label'] = self._load_label(self.key_file['train_label'])
        dataset['test_img'] = self._load_img(self.key_file['test_img'])
        dataset['test_label'] = self._load_label(self.key_file['test_label'])

        dataset['validation_img'] = dataset['train_img'][55000:]
        dataset['validation_label'] = dataset['train_label'][55000:]
        dataset['train_img'] =  dataset['train_img'][:55000]
        dataset['train_label'] = dataset['train_label'][:55000]
        return dataset

    def init_mnist(self):
        self.download_mnist()
        dataset = self._convert_numpy()
        print("Creating pickle file ...")
        with open(self.save_file, 'wb') as f:
            pickle.dump(dataset, f, -1)
        print("Done!")

    def _change_one_hot_label(self, X):
        T = np.zeros((X.size, 10))
        for idx, row in enumerate(T):
            row[X[idx]] = 1

        return T

    def load_mnist(self, normalize=True, flatten=True, one_hot_label=False):
        if not os.path.exists(self.save_file):
            self.init_mnist()

        with open(self.save_file, 'rb') as f:
            dataset = pickle.load(f)

        if normalize:
            for key in ('train_img', 'validation_img', 'test_img'):
                dataset[key] = dataset[key].astype(np.float32)
                dataset[key] /= 255.0

        if one_hot_label:
            dataset['train_label'] = self._change_one_hot_label(dataset['train_label'])
            dataset['validation_label'] = self._change_one_hot_label(dataset['validation_label'])
            dataset['test_label'] = self._change_one_hot_label(dataset['test_label'])

        if not flatten:
            for key in ('train_img', 'validation_img', 'test_img'):
                dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

        return (dataset['train_img'], dataset['train_label']), (dataset['validation_img'], dataset['validation_label']), (dataset['test_img'], dataset['test_label'])