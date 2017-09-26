# -*- coding:utf-8 -*-
import math
import numpy as np

#Squared Error를 줄이는 것이 목적
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.params = None

    def update(self, grads):
        for key in self.params.keys():
            self.params[key].value = self.params[key].value - self.learning_rate * grads[key]
