# -*- coding:utf-8 -*-

class Base_Data:
    def __init__(self, n_splits):
        self.num_train_data = None
        self.num_validation_data = None
        self.num_test_data = None

        self.n_splits = n_splits

    def set_next_train_and_validation_data(self):
        if self.n_splits > 1:
            train_indices, validation_indices = next(self.splitted_indices)

            self.validation_input = self.images[validation_indices]
            self.validation_target = self.targets[validation_indices]

            self.train_input = self.images[train_indices]
            self.train_target = self.targets[train_indices]