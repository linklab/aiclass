# -*- coding:utf-8 -*-
import numpy as np
import tensorflux.graph as tfg
import tensorflux.functions as tff


class Initializer:
    def __init__(self, shape, name):
        self.shape = shape
        self.name = name

        self.param = None
        self.initialize_param()

    def initialize_param(self):
        pass

    def get_variable(self):
        return self.param


class Value_Assignment_Initializer(Initializer):
    def __init__(self, value, name):
        self.value = value
        super().__init__([1], name)

    def initialize_param(self):
        self.param = tfg.Variable(self.value, name=self.name)


class Zero_Initializer(Initializer):
    def initialize_param(self):
        self.param = tfg.Variable(np.zeros(shape=self.shape), name=self.name)


class One_Initializer(Initializer):
    def initialize_param(self):
        self.param = tfg.Variable(np.ones(shape=self.shape), name=self.name)

class Randn_Initializer(Initializer):
    def initialize_param(self):
        self.param = tfg.Variable(np.random.randn(self.shape[0], self.shape[1]), name=self.name)

class Point_One_Initializer(Initializer):
    def initialize_param(self):
        self.param = tfg.Variable(np.ones(shape=self.shape) * 0.1, name=self.name)


class Random_Normal_Initializer(Initializer):
    """
    Parameters :
    loc : float -- Mean (“centre”) of the distribution.
    scale : float -- Standard deviation (spread or “width”) of the distribution.
    size : tuple of ints -- Output shape.
    """
    def initialize_param(self):
        self.param = tfg.Variable(np.random.normal(loc=0.0, scale=0.1, size=self.shape), name=self.name)


class Random_Uniform_Initializer(Initializer):
    """
    Parameters :
    size : tuple of ints -- Output shape.
    """
    def initialize_param(self):
        self.param = tfg.Variable(np.random.random(size=self.shape), name=self.name)


class Truncated_Normal_Initializer(Initializer):
    def __init__(self, shape, name, mean=0.0, sd=1.0, low=-1.0, upp=1.0):
        self.mean = mean
        self.sd = sd
        self.low = low
        self.upp = upp
        super().__init__(shape, name)

    def initialize_param(self):
        self.param = tfg.Variable(tff.get_truncated_normal(shape=self.shape,
                                                           mean=self.mean,
                                                           sd=self.sd,
                                                           low=self.low,
                                                           upp=self.upp), name=self.name)
