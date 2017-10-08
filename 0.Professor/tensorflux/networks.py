from collections import OrderedDict
import tensorflux.graph as tfg
import tensorflux.enums as tfe
import tensorflux.layers as tfl
import tensorflux.session as tfs
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class Neural_Network(tfg.Graph):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.input_node = None
        self.target_node = None

        self.activator = None
        self.initializer = None
        self.optimizer = None

        self.params = {}

        self.output = None
        self.error = None

        self.session = tfs.Session()
        super().__init__()

    def set_data(self, input_node, target_node):
        self.input_node = input_node
        self.target_node = target_node

    def initialize_param(self, initializer=tfe.Initializer.Zero.value):
        pass

    def layering(self, activator=tfe.Activator.ReLU.value):
        pass

    def set_optimizer(self, optimizer=tfe.Optimizer.SGD.value, learning_rate=0.01):
        self.optimizer = optimizer(learning_rate=learning_rate)
        self.optimizer.params = self.params

    def numerical_derivative(self, session, feed_data):
        delta = 1e-4  # 0.0001

        grads = {}

        for param_key, param in self.params.items():
            temp_val = param.value

            # f(x + delta) 계산
            param.value = param.value + delta
            fxh1 = session.run(self.error, feed_dict=feed_data, verbose=False)

            param.value = temp_val

            # f(x - delta) 계산
            param.value = param.value - delta
            fxh2 = session.run(self.error, feed_dict=feed_data, verbose=False)

            # f(x + delta) - f(x - delta) / 2 * delta 계산
            grads[param_key] = (fxh1 - fxh2) / (2 * delta)
            param.value = temp_val
        return grads

    def learning(self, max_epoch, data, verbose=False):
        for epoch in range(max_epoch):
            sum_train_error = 0.0
            for idx in range(data.num_train_data):
                train_input_data = data.training_input[idx]
                train_target_data = data.training_target[idx]

                sum_train_error += self.session.run(self.error,
                                                    {
                                                        self.input_node: train_input_data,
                                                        self.target_node: train_target_data
                                                    }, verbose)

                grads = self.numerical_derivative(self.session,
                                                  {
                                                      self.input_node: train_input_data,
                                                      self.target_node: train_target_data
                                                  })

                self.optimizer.update(grads=grads)

            sum_validation_error = 0.0
            for idx in range(data.num_validation_data):
                validation_input_data = data.validation_input[idx]
                validation_target_data = data.validation_target[idx]
                sum_validation_error += self.session.run(self.error,
                                                         {
                                                             self.input_node: validation_input_data,
                                                             self.target_node: validation_target_data
                                                         }, verbose)

            print("Epoch {:3d} Completed - Average Train Error: {:7.6f} - Average Validation Error: {:7.6f}".format(
                epoch,
                sum_train_error / data.num_train_data,
                sum_validation_error / data.num_validation_data
            ))

            if verbose:
                print()

    def get_params_str(self):
        params_str = ""
        for param_key, param in self.params.items():
            params_str = params_str + param_key + ": " + str(param.value) + ", "
        params_str = params_str[0:-2]
        return params_str

    def get_param_describe(self):
        """
        :return: starts.description
        skewness - https://ko.wikipedia.org/wiki/%EB%B9%84%EB%8C%80%EC%B9%AD%EB%8F%84
        kurtosis - https://ko.wikipedia.org/wiki/%EC%B2%A8%EB%8F%84
        """
        param_flatten_list = []
        for param in self.params.values():
            param_flatten_list.extend([item for item in param.value.flatten()])
        return stats.describe(np.array(param_flatten_list))

    def print_feed_forward(self, num_data, input_data, target_data, verbose=False):
        for idx in range(num_data):
            train_input_data = input_data[idx]
            train_target_data = target_data[idx]

            output = self.session.run(self.output, {self.input_node: train_input_data}, verbose)
            print("Input Data: {:>5}, Feed Forward Output: {:>6}, Target: {:>6}".format(
                str(train_input_data), np.array2string(output), str(train_target_data)))

    def draw_and_show(self):
        nx.draw_networkx(self, with_labels=True)
        plt.show(block=True)


class Single_Neuron_Network(Neural_Network):
    def __init__(self, input_size, output_size):
        self.affine = None
        super().__init__(input_size, output_size)

    def initialize_scalar_param(self, value1, value2, initializer=tfe.Initializer.Value_Assignment.value):
        self.params['W0'] = initializer(value1, name='W0').get_variable()
        self.params['b0'] = initializer(value2, name='b0').get_variable()

    def initialize_param(self, initializer=tfe.Initializer.Zero.value):
        self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
        self.params['b0'] = initializer(shape=(self.output_size,), name='b0').get_variable()

    def layering(self, activator=tfe.Activator.ReLU.value):
        self.activator = activator
        self.affine = tfl.Affine(self.params['W0'], self.input_node, self.params['b0'], name="A", graph=self)
        self.output = activator(self.affine, name="O", graph=self)
        self.error = tfl.SquaredError(self.output, self.target_node, name="SE", graph=self)

    def learning_bp(self, max_epoch, data, verbose=False):
        grads = {}
        for epoch in range(max_epoch):
            sum_train_error = 0.0
            for idx in range(data.num_train_data):
                if verbose:
                    print("Epoch: {:d}, Train Data Index: {:d}".format(epoch, idx))
                train_input_data = data.training_input[idx]
                train_target_data = data.training_target[idx]

                #forward
                sum_train_error += self.session.run(self.error,
                                                    {
                                                        self.input_node: train_input_data,
                                                        self.target_node: train_target_data
                                                    }, verbose)
                #backward
                d_error = self.error.backward(1.0)
                d_output = self.output.backward(d_error)
                _ = self.affine.backward(d_output)

                grads['W0'] = self.affine.dw
                grads['b0'] = self.affine.db

                self.optimizer.update(grads=grads)

                grads.clear()

            sum_validation_error = 0.0
            for idx in range(data.num_validation_data):
                validation_input_data = data.validation_input[idx]
                validation_target_data = data.validation_target[idx]
                sum_validation_error += self.session.run(self.error,
                                                         {
                                                             self.input_node: validation_input_data,
                                                             self.target_node: validation_target_data
                                                         }, verbose)

            print("Epoch {:3d} Completed - Average Train Error: {:7.6f} - Average Validation Error: {:7.6f}".format(
                epoch,
                sum_train_error / data.num_train_data,
                sum_validation_error / data.num_validation_data
            ))

            if verbose:
                print()


class Two_Neurons_Network(Neural_Network):
    def __init__(self, input_size, output_size):
        self.affine0 = None
        self.activation0 = None
        self.affine1 = None
        super().__init__(input_size, output_size)

    def initialize_param(self, initializer=tfe.Initializer.Zero.value):
        self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
        self.params['b0'] = initializer(shape=(self.output_size,), name='b0').get_variable()
        self.params['W1'] = initializer(shape=(self.output_size, self.output_size), name='W1').get_variable()
        self.params['b1'] = initializer(shape=(self.output_size,), name='b1').get_variable()

    def layering(self, activator=tfe.Activator.ReLU.value):
        self.activator = activator
        self.affine0 = tfl.Affine(self.params['W0'], self.input_node, self.params['b0'], name="A0", graph=self)
        self.activation0 = activator(self.affine0, name="O0", graph=self)
        self.affine1 = tfl.Affine(self.params['W1'], self.activation0, self.params['b1'], name="A1", graph=self)
        self.output = activator(self.affine1, name="O1", graph=self)
        self.error = tfl.SquaredError(self.output, self.target_node, name="SE", graph=self)

    def learning_bp(self, max_epoch, data, verbose=False):
        grads = {}
        for epoch in range(max_epoch):
            sum_train_error = 0.0
            for idx in range(data.num_train_data):
                if verbose:
                    print("Epoch: {:d}, Train Data Index: {:d}".format(epoch, idx))
                train_input_data = data.training_input[idx]
                train_target_data = data.training_target[idx]

                #forward
                sum_train_error += self.session.run(self.error,
                                                    {
                                                        self.input_node: train_input_data,
                                                        self.target_node: train_target_data
                                                    }, verbose)
                #backward
                d_error = self.error.backward(1.0)
                d_output = self.output.backward(d_error)
                d_affine1 = self.affine1.backward(d_output)
                d_activation0 = self.activation0.backward(d_affine1)
                _ = self.affine0.backward(d_activation0)

                grads['W0'] = self.affine0.dw
                grads['b0'] = self.affine0.db
                grads['W1'] = self.affine1.dw
                grads['b1'] = self.affine1.db

                self.optimizer.update(grads=grads)

                grads.clear()

            sum_validation_error = 0.0
            for idx in range(data.num_validation_data):
                validation_input_data = data.validation_input[idx]
                validation_target_data = data.validation_target[idx]
                sum_validation_error += self.session.run(self.error,
                                                         {
                                                             self.input_node: validation_input_data,
                                                             self.target_node: validation_target_data
                                                         }, verbose)

            print("Epoch {:3d} Completed - Average Train Error: {:7.6f} - Average Validation Error: {:7.6f}".format(
                epoch,
                sum_train_error / data.num_train_data,
                sum_validation_error / data.num_validation_data
            ))

            if verbose:
                print()


class Three_Neurons_Network(Neural_Network):
    def __init__(self, input_size, output_size):
        self.affine0 = None
        self.activation0 = None
        self.affine1 = None
        self.activation1 = None
        self.affine2 = None
        super().__init__(input_size, output_size)

    def initialize_param(self, initializer=tfe.Initializer.Zero.value):
        self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
        self.params['b0'] = initializer(shape=(self.output_size,), name='b0').get_variable()

        self.params['W1'] = initializer(shape=(self.input_size, self.output_size), name='W1').get_variable()
        self.params['b1'] = initializer(shape=(self.output_size,), name='b1').get_variable()

        self.params['W2'] = initializer(shape=(self.input_size, self.output_size), name='W2').get_variable()
        self.params['b2'] = initializer(shape=(self.output_size,), name='b2').get_variable()

    def layering(self, activator=tfe.Activator.ReLU.value):
        self.activator = activator

        self.affine0 = tfl.Affine(self.params['W0'], self.input_node, self.params['b0'], name="A0", graph=self)
        self.activation0 = activator(self.affine0, name="O0", graph=self)

        self.affine1 = tfl.Affine(self.params['W1'], self.input_node, self.params['b1'], name="A1", graph=self)
        self.activation1 = activator(self.affine1, name="O1", graph=self)

        self.affine2 = tfl.Affine2(self.params['W2'], self.activation0, self.activation1, self.params['b2'], name="A2", graph=self)
        self.output = activator(self.affine2, name="O2", graph=self)

        self.error = tfl.SquaredError(self.output, self.target_node, name="SE", graph=self)

    def learning_bp(self, max_epoch, data, verbose=False):
        grads = {}
        for epoch in range(max_epoch):
            sum_train_error = 0.0
            for idx in range(data.num_train_data):
                if verbose:
                    print("Epoch: {:d}, Train Data Index: {:d}".format(epoch, idx))
                train_input_data = data.training_input[idx]
                train_target_data = data.training_target[idx]

                #forward
                sum_train_error += self.session.run(self.error,
                                                    {
                                                        self.input_node: train_input_data,
                                                        self.target_node: train_target_data
                                                    }, verbose)
                #backward
                d_error = self.error.backward(1.0)
                d_output = self.output.backward(d_error)
                d_affine2 = self.affine2.backward(d_output)

                d_activation0 = self.activation0.backward(d_affine2[0][0])
                _ = self.affine0.backward(d_activation0)

                d_activation1 = self.activation1.backward(d_affine2[0][1])
                _ = self.affine1.backward(d_activation1)

                grads['W0'] = self.affine0.dw
                grads['b0'] = self.affine0.db
                grads['W1'] = self.affine1.dw
                grads['b1'] = self.affine1.db
                grads['W2'] = self.affine2.dw
                grads['b2'] = self.affine2.db

                self.optimizer.update(grads=grads)

                grads.clear()

            sum_validation_error = 0.0
            for idx in range(data.num_validation_data):
                validation_input_data = data.validation_input[idx]
                validation_target_data = data.validation_target[idx]
                sum_validation_error += self.session.run(self.error,
                                                         {
                                                             self.input_node: validation_input_data,
                                                             self.target_node: validation_target_data
                                                         }, verbose)

            print("Epoch {:3d} Completed - Average Train Error: {:7.6f} - Average Validation Error: {:7.6f}".format(
                epoch,
                sum_train_error / data.num_train_data,
                sum_validation_error / data.num_validation_data
            ))

            if verbose:
                print()