# -*- coding:utf-8 -*-

import numpy as np


class Data:
    def __init__(self):
        self.num_train_data = len(self.training_input)
        self.num_validation_data = len(self.validation_input)
        self.num_test_data = len(self.test_input)


class Simple_Function_Data(Data):
    # f(x)=10𝑥+4
    def __init__(self):
        self.training_input = np.array([-1.0, 1.0, 2.0, 3.0])
        self.training_target = np.array([-6.0, 14.0, 24.0, 34.0])

        self.validation_input = np.array([1.5, 2.5])
        self.validation_target = np.array([19.0, 29.0])

        self.test_input = np.array([0, 4.0])
        self.test_target = np.array([4.0, 44.0])

        super().__init__()


class Or_Gate_Data(Data):
    def __init__(self):
        self.training_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.training_target = np.array([0.0, 1.0, 1.0, 1.0])

        self.validation_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.validation_target = np.array([0.0, 1.0, 1.0, 1.0])

        self.test_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.test_target = np.array([0.0, 1.0, 1.0, 1.0])
        super().__init__()


class And_Gate_Data(Data):
    def __init__(self):
        self.training_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.training_target = np.array([0.0, 0.0, 0.0, 1.0])

        self.validation_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.validation_target = np.array([0.0, 0.0, 0.0, 1.0])

        self.test_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.test_target = np.array([0.0, 0.0, 0.0, 1.0])
        super().__init__()


class Xor_Gate_Data(Data):
    def __init__(self):
        self.training_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.training_target = np.array([0.0, 1.0, 1.0, 0.0])

        self.validation_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.validation_target = np.array([0.0, 1.0, 1.0, 0.0])

        self.test_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.test_target = np.array([0.0, 1.0, 1.0, 0.0])
        super().__init__()