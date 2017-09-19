from collections import OrderedDict
import tensorflux2.graph as tfg
import tensorflux2.enums as tfe
import tensorflux2.layers as tfl
import tensorflux2.session as tfs
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class Neural_Network(tfg.Graph):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.input_node = None
        self.target_node = None

        self.activator = None
        self.initializer = None
        self.optimizer = None

        self.params = OrderedDict()

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

        grads = OrderedDict()

        for param_key, param in self.params.items():
            temp_val = param.value

            # f(x + delta) 계산
            param.value = param.value + delta
            fxh1 = session.run(self.error, feed_dict=feed_data, vervose=False)

            param.value = temp_val

            # f(x - delta) 계산
            param.value = param.value - delta
            fxh2 = session.run(self.error, feed_dict=feed_data, vervose=False)

            # f(x + delta) - f(x - delta) / 2 * delta 계산
            grads[param_key] = (fxh1 - fxh2) / (2 * delta)
            param.value = temp_val
        return grads

    def learning(self, max_epoch, data, x, target):
        for epoch in range(max_epoch):
            sum_train_error = 0.0
            for idx in range(data.num_train_data):
                train_input_data = data.training_input[idx]
                train_target_data = data.training_target[idx]

                grads = self.numerical_derivative(self.session, {x: train_input_data, target: train_target_data})
                self.optimizer.update(grads=grads)
                sum_train_error += self.session.run(self.error, {x: train_input_data, target: train_target_data}, vervose=False)

            sum_validation_error = 0.0
            for idx in range(data.num_validation_data):
                validation_input_data = data.validation_input[idx]
                validation_target_data = data.validation_target[idx]
                sum_validation_error += self.session.run(self.error,
                                                         {x: validation_input_data, target: validation_target_data},
                                                        vervose=False)

            print("Epoch {:3d} Completed - Average Train Error: {:7.6f} - Average Validation Error: {:7.6f}".format(
                epoch, sum_train_error / data.num_train_data, sum_validation_error / data.num_validation_data))

    def print_feed_forward(self, num_data, input_data, target_data, x):
        for idx in range(num_data):
            train_input_data = input_data[idx]
            train_target_data = target_data[idx]

            output = self.session.run(self.output,
                                      {x: train_input_data},
                                      vervose=False)
            print("Input Data: {:>5}, Feed Forward Output: {:>6}, Target: {:>6}".format(
                str(train_input_data), np.array2string(output), str(train_target_data)))

    def draw_and_show(self):
        nx.draw_networkx(self, with_labels=True)
        plt.show(block=True)


class Single_Neuron_Network(Neural_Network):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def initialize_scalar_param(self, value1, value2, initializer=tfe.Initializer.Value_Assignment.value):
        self.params['W0'] = initializer(value1, name='W0').get_variable()
        self.params['b0'] = initializer(value2, name='b0').get_variable()

    def initialize_param(self, initializer=tfe.Initializer.Zero.value):
        self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
        self.params['b0'] = initializer(shape=(self.output_size,), name='b0').get_variable()

    def layering(self, activator=tfe.Activator.ReLU.value):
        self.activator = activator
        u = tfl.Affine(self.params['W0'],
                       self.input_node,
                       self.params['b0'],
                       name="A")
        self.output = activator(u, name="O")
        self.error = tfl.SquaredError(self.output, self.target_node, name="SE")
        if isinstance(self, nx.Graph):
            self.add_edge(self.params['W0'], u)
            self.add_edge(self.input_node, u)
            self.add_edge(self.params['b0'], u)
            self.add_edge(u, self.output)
            self.add_edge(self.output, self.error)
            self.add_edge(self.error, self.target_node)


class Two_Neurons_Network(Neural_Network):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def initialize_param(self, initializer=tfe.Initializer.Zero.value):
        self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
        self.params['b0'] = initializer(shape=(self.output_size,), name='b0').get_variable()
        self.params['W1'] = initializer(shape=(self.output_size, self.output_size), name='W1').get_variable()
        self.params['b1'] = initializer(shape=(self.output_size,), name='b1').get_variable()

    def layering(self, activator=tfe.Activator.ReLU.value):
        self.activator = activator
        u0 = tfl.Affine(self.params['W0'],
                        self.input_node,
                        self.params['b0'], name="A0")
        o0 = activator(u0, name="O0")
        u1 = tfl.Affine(self.params['W1'],
                        o0,
                        self.params['b1'], name="A1")
        self.output = activator(u1, name="O1")
        self.error = tfl.SquaredError(self.output, self.target_node, name="SE")
        if isinstance(self, nx.Graph):
            self.add_edge(self.params['W0'], u0)
            self.add_edge(self.input_node, u0)
            self.add_edge(self.params['b0'], u0)
            self.add_edge(u0, o0)
            self.add_edge(self.params['W1'], u1)
            self.add_edge(o0, u1)
            self.add_edge(self.params['b1'], u1)
            self.add_edge(u1, self.output)
            self.add_edge(self.output, self.error)
            self.add_edge(self.error, self.target_node)


class Three_Neurons_Network(Neural_Network):
    pass