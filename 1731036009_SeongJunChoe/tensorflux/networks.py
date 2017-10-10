from collections import OrderedDict
import tensorflux.graph as tfg
import tensorflux.enums as tfe
import tensorflux.layers as tfl
import tensorflux.session as tfs
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class Neural_Network(tfg.Graph):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # (input) -> (동작) -> (target)
        self.input_node = None
        self.target_node = None

        self.activator = None   # 사용할 함수
        self.initializer = None
        self.optimizer = None

        self.params = OrderedDict() # 가중치, bias 등을 모아둔 dictionary
        # 파이선은 배열의 인덱스를 해쉬를 통한 키값으로 접근하기 때문에
        # a[1] = 'aaa'
        # b[1] = 'bbb'
        # c[1] = 'ccc'
        # 로 넣어도 'aaa', 'bbb', 'ccc'의 순으로 출력된다는 보장을 못함
        # 그러므로 OrderedDict을 통해 순서대로 나올 수 있도록 함

        self.output = None
        self.error = None   # output과 target의 결과 차

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

            # f(x + delta) - f(x - delta) / 2 * delta 계산 (02.Artificial_Single_Nueral_Network.pdf 24 page)
            grads[param_key] = (fxh1 - fxh2) / (2 * delta)
            param.value = temp_val
        return grads

    def learning(self, max_epoch, data, x, target, verbose=False):
        for epoch in range(max_epoch):
            sum_train_error = 0.0
            for idx in range(data.num_train_data):
                train_input_data = data.training_input[idx]
                train_target_data = data.training_target[idx]

                # print(train_input_data)
                # print(train_target_data)
                grads = self.numerical_derivative(self.session, {x: train_input_data, target: train_target_data})
                self.optimizer.update(grads=grads)
                sum_train_error += self.session.run(self.error, {x: train_input_data, target: train_target_data}, verbose)

            sum_validation_error = 0.0
            for idx in range(data.num_validation_data):
                validation_input_data = data.validation_input[idx]
                validation_target_data = data.validation_target[idx]
                sum_validation_error += self.session.run(self.error,
                                                         {x: validation_input_data, target: validation_target_data},
                                                        vervose=False)

            if epoch % 1000 == 0:
                print("Epoch {:3d} Completed - Average Train Error: {:7.6f} - Average Validation Error: {:7.6f}".format(
                    epoch, sum_train_error / data.num_train_data, sum_validation_error / data.num_validation_data))

    def print_feed_forward(self, num_data, input_data, target_data, x):
        for idx in range(num_data):
            train_input_data = input_data[idx]
            train_target_data = target_data[idx]

            output = self.session.run(self.output, {x: train_input_data}, vervose=False)
            print("Input Data: {:>5}, Feed Forward Output: {:>6}, Target: {:>6}".format(
                str(train_input_data), np.array2string(output), str(train_target_data)))

    def draw_and_show(self):
        nx.draw_networkx(self, with_labels=True)
        plt.show(block=True)


class Single_Neuron_Network(Neural_Network):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
    # initializer는 함수 사용시 설정되지 않으므로 tfe.Initializer.Value_Assignment.value로 설정
    def initialize_scalar_param(self, value1, value2, initializer=tfe.Initializer.Value_Assignment.value):
        self.params['W0'] = initializer(value1, name='W0').get_variable()   # 파이선의 생성자
        self.params['b0'] = initializer(value2, name='b0').get_variable()   # 파이선의 생성자

    def initialize_param(self, initializer=tfe.Initializer.Zero.value):
        self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
        self.params['b0'] = initializer(shape=(self.output_size,), name='b0').get_variable()

    def layering(self, activator=tfe.Activator.ReLU.value): # activator에 ReLU 클래스를 넣음
        self.activator = activator
        u = tfl.Affine(self.params['W0'],
                       self.input_node,
                       self.params['b0'],
                       name="A")
        self.output = activator(u, name="O")
        self.error = tfl.SquaredError(self.output, self.target_node, name="SE")
        # 싱글 뉴런 네트워크를 시각화
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
        u0 = tfl.AffineFL(self.params['W0'],
                        self.input_node,
                        self.params['b0'],
                        name="A0")
        o0 = activator(u0, name="O0")
        u1 = tfl.AffineFL(self.params['W1'],
                        o0,
                        self.params['b1'],
                        name="A1")
        self.output = activator(u1, name="O1")  # 최종적인 feed_forward output
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
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def initialize_param(self, initializer=tfe.Initializer.Zero.value):
        self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
        self.params['b0'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b0').get_variable()
        self.params['W1'] = initializer(shape=(self.input_size, self.output_size), name='W1').get_variable()
        self.params['b1'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b1').get_variable()
        self.params['W2'] = initializer(shape=(self.input_size, self.output_size), name='W2').get_variable()
        self.params['b2'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b2').get_variable()

        # self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
        # self.params['b0'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b0').get_variable()
        # self.params['W1'] = initializer(shape=(self.input_size, self.output_size), name='W1').get_variable()
        # self.params['b1'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b1').get_variable()
        # self.params['W2'] = initializer(shape=(self.input_size, self.output_size), name='W2').get_variable()
        # self.params['b2'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b2').get_variable()

    def layering(self, activator=tfe.Activator.ReLU.value):
        self.activator = activator
        u0 = tfl.AffineFL(self.params['W0'],
                          self.input_node,
                          self.params['b0'],
                          name="A0")
        o0 = activator(u0, name="O0")
        u1 = tfl.AffineFL(self.params['W1'],
                          self.input_node,
                          self.params['b1'],
                          name="A1")
        o1 = activator(u1, name="O1")
        u2 = tfl.AffineSL(self.params['W2'],
                          o0,
                          o1,
                          self.params['b2'],
                          name="A2")
        self.output = activator(u2, name="O2")
        self.error = tfl.SquaredError(self.output, self.target_node, name="SE")

        if isinstance(self, nx.Graph):
            self.add_edge(self.input_node, u0)
            self.add_edge(self.input_node, u1)
            self.add_edge(self.params['W0'], u0)
            self.add_edge(self.params['b0'], u0)
            self.add_edge(self.params['W1'], u1)
            self.add_edge(self.params['b1'], u1)
            self.add_edge(u0, o0)
            self.add_edge(u1, o1)
            self.add_edge(self.params['W2'], u2)
            self.add_edge(o0, u2)
            self.add_edge(o1, u2)
            self.add_edge(self.params['b2'], u2)
            self.add_edge(u2, self.output)
            self.add_edge(self.output, self.error)
            self.add_edge(self.error, self.target_node)

# class Three_Neurons_Network(Neural_Network):
#     def __init__(self, input_size, output_size):
#         super().__init__(input_size, output_size)
#
#     def initialize_param(self, initializer=tfe.Initializer.Zero.value):
#         self.params['W11'] = initializer(shape=(self.input_size, self.output_size), name='W11').get_variable()
#         self.params['W12'] = initializer(shape=(self.input_size, self.output_size), name='W12').get_variable()
#         self.params['b1'] = initializer(shape=(self.output_size,), name='b1').get_variable()
#         self.params['W21'] = initializer(shape=(self.input_size, self.output_size), name='W21').get_variable()
#         self.params['W22'] = initializer(shape=(self.input_size, self.output_size), name='W22').get_variable()
#         self.params['b2'] = initializer(shape=(self.output_size,), name='b2').get_variable()
#         self.params['W31'] = initializer(shape=(self.output_size, self.output_size), name='W31').get_variable()
#         self.params['W32'] = initializer(shape=(self.output_size, self.output_size), name='W32').get_variable()
#         self.params['b3'] = initializer(shape=(self.output_size, ), name='b3').get_variable()
#
#     def layering(self, activator=tfe.Activator.ReLU.value):
#         self.activator = activator
#         u0 =
# #



# class Three_Neurons_Network(Neural_Network):
#     def __init__(self, input_size, output_size):
#         super().__init__(input_size, output_size)
#
#     def initialize_param(self, initializer=tfe.Initializer.Zero.value):
#         self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
#         self.params['b0'] = initializer(shape=(self.output_size,), name='b0').get_variable()
#         self.params['W1'] = initializer(shape=(self.output_size, self.output_size), name='W1').get_variable()
#         self.params['b1'] = initializer(shape=(self.output_size,), name='b1').get_variable()
#
#     def layering(self, activator=tfe.Activator.ReLU.value):
#         self.activator = activator
#         u0 = tfl.Affine(self.params['W0'],
#                         self.input_node,
#                         self.params['b0'],
#                         name="A0")
#         o0 = activator(u0, name="O0")
#         u1 = tfl.Affine(self.params['W1'],
#                         o0,
#                         self.params['b1'],
#                         name="A1")
#         self.output = activator(u1, name="O1")  # 최종적인 feed_forward output
#         self.error = tfl.SquaredError(self.output, self.target_node, name="SE")
#         if isinstance(self, nx.Graph):
#             self.add_edge(self.params['W0'], u0)
#             self.add_edge(self.input_node, u0)
#             self.add_edge(self.params['b0'], u0)
#             self.add_edge(u0, o0)
#             self.add_edge(self.params['W1'], u1)
#             self.add_edge(o0, u1)
#             self.add_edge(self.params['b1'], u1)
#             self.add_edge(u1, self.output)
#             self.add_edge(self.error, self.target_node)
#             self.add_edge(self.output, self.error)
