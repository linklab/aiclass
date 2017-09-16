# -*- coding:utf-8 -*-

import numpy as np


class Simple_Function_Data:
    # f(x)=10𝑥+4
    def __init__(self):
        self.training_input = np.array([1.0, 2.0, 3.0])
        self.training_target = np.array([14.0, 24.0, 34.0])

        self.validation_input = np.array([1.5, 2.5])
        self.validation_target = np.array([19.0, 29.0])

        self.test_input = np.array([0, 4.0])
        self.test_target = np.array([4.0, 44.0])

        self.numTrainData = len(self.training_input)
        self.numValidationData = len(self.validation_input)
        self.numTestData = len(self.test_input)


class Or_Gate_Data:
    def __init__(self):
        self.training_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.training_target = np.array([0.0, 1.0, 1.0, 1.0])

        self.validation_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.validation_target = np.array([0.0, 1.0, 1.0, 1.0])

        self.test_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.test_target = np.array([0.0, 1.0, 1.0, 1.0])

        self.numTrainData = len(self.training_input)
        self.numValidationData = len(self.validation_input)
        self.numTestData = len(self.test_input)


class And_Gate_Data:
    def __init__(self):
        self.training_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.training_target = np.array([0.0, 0.0, 0.0, 1.0])

        self.validation_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.validation_target = np.array([0.0, 0.0, 0.0, 1.0])

        self.test_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.test_target = np.array([0.0, 0.0, 0.0, 1.0])

        self.numTrainData = len(self.training_input)
        self.numValidationData = len(self.validation_input)
        self.numTestData = len(self.test_input)


class Xor_Gate_Data:
    def __init__(self):
        self.training_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.training_target = np.array([0.0, 1.0, 1.0, 0.0])

        self.validation_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.validation_target = np.array([0.0, 1.0, 1.0, 0.0])

        self.test_input = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.test_target = np.array([0.0, 1.0, 1.0, 0.0])

        self.numTrainData = len(self.training_input)
        self.numValidationData = len(self.validation_input)
        self.numTestData = len(self.test_input)