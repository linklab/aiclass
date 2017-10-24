# -*- coding:utf-8 -*-
import sys
import math
import numpy as np
from numba import jit

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.params = None

    #
    # @jit
    def update(self, grads):
        for key in self.params.keys():
            self.params[key].value = self.params[key].value - self.learning_rate * grads[key]

            # print(type(self.params)) # dict
            # print(type(self.params[key])) # tensorflux.graph.Variable
            # print(type(self.params[key].value)) # numpy.ndarray


class Adam:
    def __init__(self, learning_rate=0.001):
        # beta1 = 0.9, beta2 = 0.999
        self.learning_rate = learning_rate
        self.alpha_t = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.t = 0
        self.params = None
        self.params_m = None
        self.params_v = None
        self.e = 0.00000001


    def update(self, grads):


        for key in self.params.keys():
            self.params[key].value = self.params[key].value - self.learning_rate * grads[key]

            self.params_m[key] = \
                ((self.beta1 * self.params_m[key]) + ((1 - self.beta1) * grads[key])) / (
                1 - pow(self.beta1, self.t))
            self.params_v[key] = \
                ((self.beta2 * self.params_v[key]) + ((1 - self.beta2) * pow(grads[key], 2))) \
                                                          / (1 - pow(self.beta2, self.t))

            self.params[key].value = \
                self.params[key].value - (self.learning_rate * self.params_m[key]) \
                                         / (np.sqrt(self.params_v[key]) + self.e)

    def update_(self, grads):

        for key in self.params.keys():
            # self.params_m[key].value = (self.beta1*self.params_m[key].value) + ((1-self.beta1) * grads[key]
            # self.params_v[key].value = (self.beta2*self.params_v[key].value) + ((1-self.beta2) * pow(grads[key], 2)
            self.params_m[key].value = \
                ((self.beta1 * self.params_m[key].value) + ((1 - self.beta1) * grads[key])) / (1 - pow(self.beta1, self.t))
            self.params_v[key].value = \
                ((self.beta2 * self.params_v[key].value) + ((1 - self.beta2) * pow(grads[key],2))) \
                                                          / (1 - pow(self.beta2, self.t))

            self.params[key].value = \
                self.params[key].value - ((self.learning_rate * self.params_m[key].value) \
                                         / (np.sqrt(self.params_v[key].value) + self.e))

        self.t += 1

    def update_2(self, grads):
        for key in self.params.keys():
            # advanced
            self.params_m[key].value = (self.beta1 * self.params_m[key].value) + ((1 - self.beta1) * grads[key])
            self.params_v[key].value = (self.beta2 * self.params_v[key].value) + ((1 - self.beta2) * pow(grads[key], 2))

            self.alpha_t = self.learning_rate * np.sqrt(1 - pow(self.beta2, self.t)) / (1 - pow(self.beta1, self.t))

            self.params[key].value = \
                self.params[key].value - (self.alpha_t * self.params_m[key].value) \
                                         / (np.sqrt(self.params_v[key].value) + self.e)

            self.t += 1

class op:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.params = None
        self.beta1 = 0.9
        self.t = 1.0
        # self. m

    #
    # @jit
    def update_(self, grads):
        for key in self.params.keys():
            self.params[key].value = self.params[key].value - self.learning_rate * grads[key]

    def update(self, grads):
        # if self.m is None:
            # self.m = {}
            # for key, param in self.params.items():
            #     self.m[key] = np.zeros_like(param.value)

        for key in self.params.keys():
            # self.params[key].value = self.params[key].value - (self.beta1 * self.params[key].value)*grads[key]
                                     #
                                     # + ((1.0 - self.beta1) * grads[key])
            self.params[key].value = np.dot(self.beta1, self.params[key].value) + np.dot((1.0 - self.beta1),grads[key])
            #     ((self.beta1 * self.params[key].value) + ((1.0 - self.beta1) * grads[key])) / (
            #         1.0 - pow(self.beta1, self.t))
            # self.t += 1.0