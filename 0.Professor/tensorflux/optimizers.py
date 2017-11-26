# -*- coding:utf-8 -*-
import sys
import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for key in params.keys():
            if not key.startswith("running"):
                params[key].value = params[key].value - self.learning_rate * grads[key]


class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, param in params.items():
                if not key.startswith("running"):
                    self.v[key] = np.zeros_like(param.value)

        for key in params.keys():
            if not key.startswith("running"):
                self.v[key] = self.momentum * self.v[key] - self.learning_rate * grads[key]
                params[key].value = params[key].value + self.v[key]


class NAG:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None

    def update(self, params, cloned_network, is_numba):
        if self.v is None:
            self.v = {}
            for key, param in params.items():
                if not key.startswith("running"):
                    self.v[key] = np.zeros_like(param.value)

        for key, param in cloned_network.params.items():
            if not key.startswith("running"):
                param.value = param.value - self.momentum * self.v[key]
        grads = cloned_network.backward_propagation(is_numba)
        del cloned_network

        for key in params.keys():
            if not key.startswith("running"):
                self.v[key] = self.momentum * self.v[key] - self.learning_rate * grads[key]
                params[key].value = params[key].value + self.v[key]


class AdaGrad:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.g = None
        self.e = 1.0e-7

    def update(self, params, grads):
        if self.g is None:
            self.g = {}
            for key, param in params.items():
                if not key.startswith("running"):
                    self.g[key] = np.zeros_like(param.value)

        for key in params.keys():
            if not key.startswith("running"):
                self.g[key] = self.g[key] + grads[key] ** 2
                if np.isnan(grads[key]).any():
                    sys.exit(-1)
                params[key].value = params[key].value - (self.learning_rate / np.sqrt(self.g[key] + self.e)) * grads[key]


class Adam:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        self.e = 1.0e-7

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, param in params.items():
                if not key.startswith("running"):
                    self.m[key] = np.zeros_like(param.value)
                    self.v[key] = np.zeros_like(param.value)

        self.iter += 1

        learning_rate_t = self.learning_rate * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            if not key.startswith("running"):
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2
                params[key].value = params[key].value - (learning_rate_t / np.sqrt(self.v[key] + self.e)) * self.m[key]