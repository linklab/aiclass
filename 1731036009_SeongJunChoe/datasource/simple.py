# -*- coding:utf-8 -*-

import numpy as np
import datasource.data as data


class Simple_Function_Data(data.Base_Data):
    # f(x)=10𝑥+4
    def __init__(self):
        self.train_input = np.array([-1.0, 1.0, 2.0, 3.0])
        self.train_target = np.array([-6.0, 14.0, 24.0, 34.0])

        self.validation_input = np.array([1.5, 2.5])
        self.validation_target = np.array([19.0, 29.0])

        self.test_input = np.array([0, 4.0])
        self.test_target = np.array([4.0, 44.0])

        super().__init__()


class Or_Gate_Data(data.Base_Data):
    def __init__(self):
        self.train_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.train_target = np.array([0.0, 1.0, 1.0, 1.0])

        self.validation_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.validation_target = np.array([0.0, 1.0, 1.0, 1.0])

        self.test_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.test_target = np.array([0.0, 1.0, 1.0, 1.0])
        super().__init__()


class And_Gate_Data(data.Base_Data):
    def __init__(self):
        self.train_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.train_target = np.array([0.0, 0.0, 0.0, 1.0])

        self.validation_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.validation_target = np.array([0.0, 0.0, 0.0, 1.0])

        self.test_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.test_target = np.array([0.0, 0.0, 0.0, 1.0])
        super().__init__()


class Xor_Gate_Data(data.Base_Data):
    def __init__(self):
        self.train_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.train_target = np.array([0.0, 1.0, 1.0, 0.0])

        self.validation_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.validation_target = np.array([0.0, 1.0, 1.0, 0.0])

        self.test_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.test_target = np.array([0.0, 1.0, 1.0, 0.0])
        super().__init__()