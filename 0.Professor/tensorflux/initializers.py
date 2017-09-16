# -*- coding:utf-8 -*-
import numpy as np
import tensorflux.graph as tfg
import tensorflux.functions as tff


class Initializer:
    def __init__(self, params, params_size_list):
        self.params = params
        self.params_size_list = params_size_list
        self.initialize_params()

    def initialize_params(self):
        pass

    def get_params(self):
        return self.params


class Value_Assignment_Initializer(Initializer):
    def initialize_params(self):
        for idx in range(0, len(self.params_size_list) - 1):
            self.params['W' + str(idx)] = tfg.Variable(5.0, name='W' + str(idx))
            self.params['b' + str(idx)] = tfg.Variable(-1.0, name='b' + str(idx))


class Zero_Initializer(Initializer):
    def initialize_params(self):
        for idx in range(0, len(self.params_size_list) - 1):
            self.params['W' + str(idx)] = tfg.Variable(
                                                np.zeros(shape=(self.params_size_list[idx], self.params_size_list[idx + 1])),
                                                name='W' + str(idx))
            self.params['b' + str(idx)] = tfg.Variable(
                                                np.zeros(shape=(self.params_size_list[idx + 1],)),
                                                name='b' + str(idx))


class Truncated_Normal_Initializer(Initializer):
    def initialize_params(self):
        for idx in range(0, len(self.params_size_list) - 1):
            self.params['W' + str(idx)] = tfg.Variable(
                tff.get_truncated_normal(shape=(self.params_size_list[idx], self.params_size_list[idx + 1]),
                                         mean=0.0, sd=1.0, low=-3.0, upp=3.0),
                name='W' + str(idx))
            self.params['b' + str(idx)] = tfg.Variable(
                tff.get_truncated_normal(shape=(self.params_size_list[idx + 1],),
                                         mean=0.0, sd=1.0, low=-3.0, upp=3.0),
                name='b' + str(idx))

