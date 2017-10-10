# -*- coding:utf-8 -*-


class Base_Data:
    def __init__(self):
        self.num_train_data = len(self.train_input)
        self.num_validation_data = len(self.validation_input)
        self.num_test_data = len(self.test_input)