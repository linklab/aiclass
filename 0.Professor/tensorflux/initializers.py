# -*- coding:utf-8 -*-
# https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
# http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
# https://github.com/google/prettytensor/blob/a69f13998258165d6682a47a931108d974bab05e/prettytensor/layers.py
# https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/keras/_impl/keras/initializers.py

import numpy as np
import tensorflux.graph as tfg
import tensorflux.functions as tff
import math


class Initializer:
    def __init__(self, shape, name, mean=0.0, sd=1.0):
        self.shape = shape
        self.name = name
        self.mean = mean
        self.sd = sd

        self.param = None
        self.initialize_param()

    def initialize_param(self):
        pass


class Value_Assignment_Initializer(Initializer):
    def __init__(self, value, name):
        self.value = value
        super().__init__([1], 0.0, 1.0, name)

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
    def initialize_param(self):
        self.param = tfg.Variable(np.random.normal(loc=self.mean, scale=self.sd, size=self.shape), name=self.name)


class Random_Uniform_Initializer(Initializer):
    def initialize_param(self):
        self.param = tfg.Variable(np.random.random(size=self.shape), name=self.name)


class Truncated_Normal_Initializer(Initializer):
    def initialize_param(self):
        self.param = tfg.Variable(tff.get_truncated_normal(shape=self.shape,
                                                           mean=self.mean,
                                                           sd=self.sd,
                                                           low=-self.sd,
                                                           upp=self.sd), name=self.name)

class Lecun_Normal(Initializer):
    def initialize_param(self):
        sd = math.sqrt(1.0 / self.shape[0])
        self.param = tfg.Variable(tff.get_truncated_normal(shape=self.shape,
                                                           mean=self.mean,
                                                           sd=sd,
                                                           low=-sd,
                                                           upp=sd), name=self.name)


class Lecun_Uniform(Initializer):
    def initialize_param(self):
        sd = math.sqrt(1.0 / self.shape[0])
        self.param = tfg.Variable(np.random.uniform(low=-sd,
                                                    high=sd,
                                                    size=self.shape), name=self.name)


class Xavier_Normal(Initializer): # Glorot_Normal
    def initialize_param(self):
        if len(self.shape) == 2:
            sd = math.sqrt(1.0 / (self.shape[0] + self.shape[1]))
        else:
            sd = math.sqrt(1.0 / self.shape[0])
        self.param = tfg.Variable(tff.get_truncated_normal(shape=self.shape,
                                                           mean=self.mean,
                                                           sd=sd,
                                                           low=-sd,
                                                           upp=sd), name=self.name)


class Xavier_Uniform(Initializer):
    def initialize_param(self):
        if len(self.shape) == 2:
            sd = math.sqrt(1.0 / (self.shape[0] + self.shape[1]))
        else:
            sd = math.sqrt(1.0 / self.shape[0])
        self.param = tfg.Variable(np.random.uniform(low=-sd,
                                                    high=sd,
                                                    size=self.shape), name=self.name)


class He_Normal(Initializer):
    def initialize_param(self):
        if len(self.shape) == 2:
            sd = math.sqrt(2.0 / (self.shape[0] + self.shape[1]))
        else:
            sd = math.sqrt(2.0 / self.shape[0])
        self.param = tfg.Variable(tff.get_truncated_normal(shape=self.shape,
                                                           mean=self.mean,
                                                           sd=sd,
                                                           low=-sd,
                                                           upp=sd), name=self.name)


class He_Uniform(Initializer):
    def initialize_param(self):
        if len(self.shape) == 2:
            sd = math.sqrt(2.0 / (self.shape[0] + self.shape[1]))
        else:
            sd = math.sqrt(2.0 / self.shape[0])
        self.param = tfg.Variable(np.random.uniform(low=-sd,
                                                    high=sd,
                                                    size=self.shape), name=self.name)


class Conv_Xavier_Normal(Initializer): # Glorot_Normal
    def initialize_param(self):
        sd = math.sqrt(1.0 / (self.shape[1] * self.shape[2] * self.shape[3]))
        self.param = tfg.Variable(tff.get_truncated_normal(shape=self.shape,
                                                           mean=self.mean,
                                                           sd=sd,
                                                           low=-sd,
                                                           upp=sd), name=self.name)


class Conv_Xavier_Uniform(Initializer):
    def initialize_param(self):
        sd = math.sqrt(1.0 / (self.shape[1] * self.shape[2] * self.shape[3]))
        self.param = tfg.Variable(np.random.uniform(low=-sd,
                                                    high=sd,
                                                    size=self.shape), name=self.name)


class Conv_He_Normal(Initializer):
    def initialize_param(self):
        sd = math.sqrt(2.0 / (self.shape[1] * self.shape[2] * self.shape[3]))
        self.param = tfg.Variable(tff.get_truncated_normal(shape=self.shape,
                                                           mean=self.mean,
                                                           sd=sd,
                                                           low=-sd,
                                                           upp=sd), name=self.name)


class Conv_He_Uniform(Initializer):
    def initialize_param(self):
        sd = math.sqrt(2.0 / (self.shape[1] * self.shape[2] * self.shape[3]))
        self.param = tfg.Variable(np.random.uniform(low=-sd,
                                                    high=sd,
                                                    size=self.shape), name=self.name)