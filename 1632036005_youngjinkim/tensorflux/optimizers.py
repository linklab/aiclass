# -*- coding:utf-8 -*-
import sys
import numpy as np

import numba
from numba import jit, float32, int32, void, cuda


class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.params = None

    @jit
    def update(self, grads):
        for key in self.params.keys():
            self.params[key].value = self.params[key].value - self.learning_rate * grads[key]
