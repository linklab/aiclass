# -*- coding:utf-8 -*-
import numpy as np
import tensorflux.graph as tfg
import tensorflux.functions as tff

# 가중치 (weight)와 바이어스 (bias)의 값을 초기화?
# Initializer((3,3)->Matrix, "aaa")
class Initializer:
    def __init__(self, shape, name):
        self.shape = shape
        self.name = name

        self.param = None
        self.initialize_param()

    def initialize_param(self):
        pass
    # 의미 없음 (단순 get 함수 -> 파이선에서는 필요 없음?)
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

# [[1,1,1], [1,1,1], [1,1,1]] 가 생성됨
class One_Initializer(Initializer):
    def initialize_param(self):
        self.param = tfg.Variable(np.ones(shape=self.shape), name=self.name)

# [[1,1,1], [1,1,1], [1,1,1]]에 0.1을 곱하여 [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]을 생성
class Point_One_Initializer(Initializer):
    def initialize_param(self):
        self.param = tfg.Variable(np.ones(shape=self.shape) * 0.1, name=self.name)

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
