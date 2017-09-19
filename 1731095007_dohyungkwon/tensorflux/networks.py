from collections import OrderedDict
import tensorflux.graph as tfg
import tensorflux.enums as tfe
import tensorflux.layers as tfl
import tensorflux.session as tfs
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class Neural_Network(tfg.Graph): # base class
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.input_node = None # 붙일 노드
        self.target_node = None # 붙일 노드

        self.activator = None # 네트워크 내에서 활용할 객체
        self.initializer = None
        self.optimizer = None

        self.params = OrderedDict() # 순서를 관리하는 dictionary
        # a = {}대신에, a = OrderdDict()
        # a[1] = 'aaa'
        # a[2] = 'bbb'
        # dictionary는 순서 보장이 안됨

        self.output = None # Optimizer붙이기 전, feed_forward 통과하여 나오는 값
        self.error = None

        self.session = tfs.Session()
        super().__init__() # graph..

    def set_data(self, input_node, target_node):
        self.input_node = input_node
        self.target_node = target_node

    def initialize_param(self, initializer=tfe.Initializer.Zero.value): # enum이 쓰이는 방식
        # Zero.value ;enum 문법
        # Zero_Initializer 클래스가 value가 됨
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

    # https://www.dropbox.com/s/ni4r8gyfr1lw2t3/02.Artificial_Single_Neuron.pdf?dl=0
    # page 18
    def learning(self, max_epoch, data, x, target):
        for epoch in range(max_epoch):
            sum_train_error = 0.0
            for idx in range(data.num_train_data):
                train_input_data = data.training_input[idx]
                train_target_data = data.training_target[idx]

                # this file, line 51
                # https://www.dropbox.com/s/ni4r8gyfr1lw2t3/02.Artificial_Single_Neuron.pdf?dl=0
                # page 23, 24
                grads = self.numerical_derivative(self.session, {x: train_input_data, target: train_target_data})
                self.optimizer.update(grads=grads)
                sum_train_error += self.session.run(self.error, {x: train_input_data, target: train_target_data}, vervose=False)

            sum_validation_error = 0.0
            for idx in range(data.num_validation_data): # 검증용 데이터에서는 update를 하지 않음
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

            # output; by ReLU
            # vervose True해보기
            output = self.session.run(self.output, {x: train_input_data}, vervose=False)
            print("Input Data: {:>5}, Feed Forward Output: {:>6}, Target: {:>6}".format(
                str(train_input_data), np.array2string(output), str(train_target_data)))

    def draw_and_show(self):
        nx.draw_networkx(self, with_labels=True)
        plt.show(block=True)


class Single_Neuron_Network(Neural_Network):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def initialize_scalar_param(self, value1, value2,
                                initializer=tfe.Initializer.Value_Assignment.value):
        self.params['W0'] = initializer(value1, name='W0').get_variable()
        #initializer클래스의 생성자 호출. 즉 initializer(value1, name='W0')까지가 객체 생성
        #initializers.py line22-, line 18-19
        self.params['b0'] = initializer(value2, name='b0').get_variable()
        # Variable객체 두 개의 레퍼런스가 param에 할당되는 격

    def initialize_param(self, initializer=tfe.Initializer.Zero.value):
        self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
        self.params['b0'] = initializer(shape=(self.output_size,), name='b0').get_variable()

    def layering(self, activator=tfe.Activator.ReLU.value): # layers.py
        self.activator = activator
        u = tfl.Affine(self.params['W0'], self.input_node, self.params['b0'], name="A")
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
        u0 = tfl.Affine(self.params['W0'], self.input_node, self.params['b0'], name="A0")
        o0 = activator(u0, name="O0") # ReLU 클래스 생성자 격임
        u1 = tfl.Affine(self.params['W1'], o0, self.params['b1'], name="A1")
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
