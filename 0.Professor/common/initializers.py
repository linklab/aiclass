# coding: utf-8
import numpy as np


class Initializer:
    def __init__(self, params, params_size_list):
        self.params = params
        self.params_size_list = params_size_list

    def initialize_params(self):
        pass

    def get_params(self):
        return self.params


class Zero_Initializer(Initializer):
    def initialize_params(self, use_batch_normalization):
        for idx in range(1, len(self.params_size_list)):
            self.params['W' + str(idx)] = np.zeros(self.params_size_list[idx - 1], self.params_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(self.params_size_list[idx])


class N1_Initializer(Initializer):
    def initialize_params(self):
        for idx in range(1, len(self.params_size_list)):
            self.params['W' + str(idx)] = np.random.randn(self.params_size_list[idx - 1], self.params_size_list[idx])
            self.params['b' + str(idx)] = np.random.randn(self.params_size_list[idx])


class N2_Initializer(Initializer):
    def initialize_params(self):
        for idx in range(1, len(self.params_size_list)):
            self.params['W' + str(idx)] = np.random.randn(self.params_size_list[idx - 1], self.params_size_list[idx]) * 0.01
            self.params['b' + str(idx)] = np.random.randn(self.params_size_list[idx]) * 0.01


class Xavier_Initializer(Initializer):
    def initialize_params(self):
        for idx in range(1, len(self.params_size_list)):
            self.params['W' + str(idx)] = np.random.randn(self.params_size_list[idx - 1], self.params_size_list[idx]) / np.sqrt(self.params_size_list[idx - 1])
            self.params['b' + str(idx)] = np.random.randn(self.params_size_list[idx]) / np.sqrt(self.params_size_list[idx - 1])


class He_Initializer(Initializer):
    def initialize_params(self):
        for idx in range(1, len(self.params_size_list)):
            self.params['W' + str(idx)] = np.random.randn(self.params_size_list[idx - 1], self.params_size_list[idx]) * np.sqrt(2) / np.sqrt(self.params_size_list[idx - 1])
            self.params['b' + str(idx)] = np.random.randn(self.params_size_list[idx]) * np.sqrt(2) / np.sqrt(self.params_size_list[idx - 1])