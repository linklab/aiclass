from collections import OrderedDict
import tensorflux.graph as tfg
import tensorflux.enums as tfe
import tensorflux.layers as tfl
import tensorflux.session as tfs
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# class Neural_Network(tfg.Graph):
#     def __init__(self, input_size, output_size):
#         self.input_size = input_size
#         self.output_size = output_size
#
#         # (input) -> (동작) -> (target)
#         self.input_node = None
#         self.target_node = None
#
#         self.activator = None   # 사용할 함수
#         self.initializer = None
#         self.optimizer = None
#
#         self.params = {} #OrderedDict() # 가중치, bias 등을 모아둔 dictionary
#         # 파이선은 배열의 인덱스를 해쉬를 통한 키값으로 접근하기 때문에
#         # a[1] = 'aaa'
#         # b[1] = 'bbb'
#         # c[1] = 'ccc'
#         # 로 넣어도 'aaa', 'bbb', 'ccc'의 순으로 출력된다는 보장을 못함
#         # 그러므로 OrderedDict을 통해 순서대로 나올 수 있도록 함
#
#         self.output = None
#         self.error = None   # output과 target의 결과 차
#
#         self.session = tfs.Session()
#         super().__init__()
#
#     def set_data(self, input_node, target_node):
#         self.input_node = input_node
#         self.target_node = target_node
#
#     def initialize_param(self, initializer=tfe.Initializer.Zero.value):
#         pass
#
#     def layering(self, activator=tfe.Activator.ReLU.value):
#         pass
#
#     def set_optimizer(self, optimizer=tfe.Optimizer.SGD.value, learning_rate=0.01):
#         self.optimizer = optimizer(learning_rate=learning_rate)
#         self.optimizer.params = self.params
#
#     def numerical_derivative(self, session, feed_data):
#         delta = 1e-4  # 0.0001
#
#         grads = OrderedDict()
#
#         for param_key, param in self.params.items():
#             temp_val = param.value
#
#             # f(x + delta) 계산
#             param.value = param.value + delta
#             fxh1 = session.run(self.error, feed_dict=feed_data, vervose=False)
#
#             param.value = temp_val
#
#             # f(x - delta) 계산
#             param.value = param.value - delta
#             fxh2 = session.run(self.error, feed_dict=feed_data, vervose=False)
#
#             # f(x + delta) - f(x - delta) / 2 * delta 계산 (02.Artificial_Single_Nueral_Network.pdf 24 page)
#             grads[param_key] = (fxh1 - fxh2) / (2 * delta)
#             param.value = temp_val
#         return grads
#
#     def learning(self, max_epoch, data, x, target, verbose=False):
#         for epoch in range(max_epoch):
#             sum_train_error = 0.0
#             for idx in range(data.num_train_data):
#                 train_input_data = data.training_input[idx]
#                 train_target_data = data.training_target[idx]
#
#                 # print(train_input_data)
#                 # print(train_target_data)
#                 grads = self.numerical_derivative(self.session, {x: train_input_data, target: train_target_data})
#                 self.optimizer.update(grads=grads)
#                 sum_train_error += self.session.run(self.error, {x: train_input_data, target: train_target_data}, verbose)
#
#             sum_validation_error = 0.0
#             for idx in range(data.num_validation_data):
#                 validation_input_data = data.validation_input[idx]
#                 validation_target_data = data.validation_target[idx]
#                 sum_validation_error += self.session.run(self.error,
#                                                          {x: validation_input_data, target: validation_target_data},
#                                                         vervose=False)
#
#             if epoch % 1000 == 0:
#                 print("Epoch {:3d} Completed - Average Train Error: {:7.6f} - Average Validation Error: {:7.6f}".format(
#                     epoch, sum_train_error / data.num_train_data, sum_validation_error / data.num_validation_data))
#
#     def print_feed_forward(self, num_data, input_data, target_data, x):
#         for idx in range(num_data):
#             train_input_data = input_data[idx]
#             train_target_data = target_data[idx]
#
#             output = self.session.run(self.output, {x: train_input_data}, vervose=False)
#             print("Input Data: {:>5}, Feed Forward Output: {:>6}, Target: {:>6}".format(
#                 str(train_input_data), np.array2string(output), str(train_target_data)))
#
#     def draw_and_show(self):
#         nx.draw_networkx(self, with_labels=True)
#         plt.show(block=True)
#
#
# class Single_Neuron_Network(Neural_Network):
#     def __init__(self, input_size, output_size):
#         super().__init__(input_size, output_size)
#     # initializer는 함수 사용시 설정되지 않으므로 tfe.Initializer.Value_Assignment.value로 설정
#     def initialize_scalar_param(self, value1, value2, initializer=tfe.Initializer.Value_Assignment.value):
#         self.params['W0'] = initializer(value1, name='W0').get_variable()   # 파이선의 생성자
#         self.params['b0'] = initializer(value2, name='b0').get_variable()   # 파이선의 생성자
#
#     def initialize_param(self, initializer=tfe.Initializer.Zero.value):
#         self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
#         self.params['b0'] = initializer(shape=(self.output_size,), name='b0').get_variable()
#
#     def layering(self, activator=tfe.Activator.ReLU.value): # activator에 ReLU 클래스를 넣음
#         self.activator = activator
#         u = tfl.Affine(self.params['W0'],
#                        self.input_node,
#                        self.params['b0'],
#                        name="A")
#         self.output = activator(u, name="O")
#         self.error = tfl.SquaredError(self.output, self.target_node, name="SE")
#         # 싱글 뉴런 네트워크를 시각화
#         if isinstance(self, nx.Graph):
#             self.add_edge(self.params['W0'], u)
#             self.add_edge(self.input_node, u)
#             self.add_edge(self.params['b0'], u)
#             self.add_edge(u, self.output)
#             self.add_edge(self.output, self.error)
#             self.add_edge(self.error, self.target_node)
#
#
# class Two_Neurons_Network(Neural_Network):
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
#         u0 = tfl.AffineFL(self.params['W0'],
#                         self.input_node,
#                         self.params['b0'],
#                         name="A0")
#         o0 = activator(u0, name="O0")
#         u1 = tfl.AffineFL(self.params['W1'],
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
#             self.add_edge(self.output, self.error)
#             self.add_edge(self.error, self.target_node)
#
#
# class Three_Neurons_Network(Neural_Network):
#     def __init__(self, input_size, output_size):
#         super().__init__(input_size, output_size)
#
#     def initialize_param(self, initializer=tfe.Initializer.Zero.value):
#         self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
#         self.params['b0'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b0').get_variable()
#         self.params['W1'] = initializer(shape=(self.input_size, self.output_size), name='W1').get_variable()
#         self.params['b1'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b1').get_variable()
#         self.params['W2'] = initializer(shape=(self.input_size, self.output_size), name='W2').get_variable()
#         self.params['b2'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b2').get_variable()
#
#         # self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
#         # self.params['b0'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b0').get_variable()
#         # self.params['W1'] = initializer(shape=(self.input_size, self.output_size), name='W1').get_variable()
#         # self.params['b1'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b1').get_variable()
#         # self.params['W2'] = initializer(shape=(self.input_size, self.output_size), name='W2').get_variable()
#         # self.params['b2'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b2').get_variable()
#
#     def layering(self, activator=tfe.Activator.ReLU.value):
#         self.activator = activator
#         u0 = tfl.AffineFL(self.params['W0'],
#                           self.input_node,
#                           self.params['b0'],
#                           name="A0")
#         o0 = activator(u0, name="O0")
#         u1 = tfl.AffineFL(self.params['W1'],
#                           self.input_node,
#                           self.params['b1'],
#                           name="A1")
#         o1 = activator(u1, name="O1")
#         u2 = tfl.AffineSL(self.params['W2'],
#                           o0,
#                           o1,
#                           self.params['b2'],
#                           name="A2")
#         self.output = activator(u2, name="O2")
#         self.error = tfl.SquaredError(self.output, self.target_node, name="SE")
#
#         if isinstance(self, nx.Graph):
#             self.add_edge(self.input_node, u0)
#             self.add_edge(self.input_node, u1)
#             self.add_edge(self.params['W0'], u0)
#             self.add_edge(self.params['b0'], u0)
#             self.add_edge(self.params['W1'], u1)
#             self.add_edge(self.params['b1'], u1)
#             self.add_edge(u0, o0)
#             self.add_edge(u1, o1)
#             self.add_edge(self.params['W2'], u2)
#             self.add_edge(o0, u2)
#             self.add_edge(o1, u2)
#             self.add_edge(self.params['b2'], u2)
#             self.add_edge(u2, self.output)
#             self.add_edge(self.output, self.error)
#             self.add_edge(self.error, self.target_node)
#
# # class Three_Neurons_Network(Neural_Network):
# #     def __init__(self, input_size, output_size):
# #         super().__init__(input_size, output_size)
# #
# #     def initialize_param(self, initializer=tfe.Initializer.Zero.value):
# #         self.params['W11'] = initializer(shape=(self.input_size, self.output_size), name='W11').get_variable()
# #         self.params['W12'] = initializer(shape=(self.input_size, self.output_size), name='W12').get_variable()
# #         self.params['b1'] = initializer(shape=(self.output_size,), name='b1').get_variable()
# #         self.params['W21'] = initializer(shape=(self.input_size, self.output_size), name='W21').get_variable()
# #         self.params['W22'] = initializer(shape=(self.input_size, self.output_size), name='W22').get_variable()
# #         self.params['b2'] = initializer(shape=(self.output_size,), name='b2').get_variable()
# #         self.params['W31'] = initializer(shape=(self.output_size, self.output_size), name='W31').get_variable()
# #         self.params['W32'] = initializer(shape=(self.output_size, self.output_size), name='W32').get_variable()
# #         self.params['b3'] = initializer(shape=(self.output_size, ), name='b3').get_variable()
# #
# #     def layering(self, activator=tfe.Activator.ReLU.value):
# #         self.activator = activator
# #         u0 =
# # #
#
#
#
# # class Three_Neurons_Network(Neural_Network):
# #     def __init__(self, input_size, output_size):
# #         super().__init__(input_size, output_size)
# #
# #     def initialize_param(self, initializer=tfe.Initializer.Zero.value):
# #         self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
# #         self.params['b0'] = initializer(shape=(self.output_size,), name='b0').get_variable()
# #         self.params['W1'] = initializer(shape=(self.output_size, self.output_size), name='W1').get_variable()
# #         self.params['b1'] = initializer(shape=(self.output_size,), name='b1').get_variable()
# #
# #     def layering(self, activator=tfe.Activator.ReLU.value):
# #         self.activator = activator
# #         u0 = tfl.Affine(self.params['W0'],
# #                         self.input_node,
# #                         self.params['b0'],
# #                         name="A0")
# #         o0 = activator(u0, name="O0")
# #         u1 = tfl.Affine(self.params['W1'],
# #                         o0,
# #                         self.params['b1'],
# #                         name="A1")
# #         self.output = activator(u1, name="O1")  # 최종적인 feed_forward output
# #         self.error = tfl.SquaredError(self.output, self.target_node, name="SE")
# #         if isinstance(self, nx.Graph):
# #             self.add_edge(self.params['W0'], u0)
# #             self.add_edge(self.input_node, u0)
# #             self.add_edge(self.params['b0'], u0)
# #             self.add_edge(u0, o0)
# #             self.add_edge(self.params['W1'], u1)
# #             self.add_edge(o0, u1)
# #             self.add_edge(self.params['b1'], u1)
# #             self.add_edge(u1, self.output)
# #             self.add_edge(self.error, self.target_node)
# #             self.add_edge(self.output, self.error)
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

    def learning(self, max_epoch, data, bp=True, print_period=10, verbose=False):
        for epoch in range(max_epoch):
            if verbose and epoch % print_period == 0:
                print()
                print("--------------")
                print("[Epoch {:d}]".format(epoch))
                print()
                verbose2 = True
            else:
                verbose2 = False

            sum_train_error = 0.0
            for idx in range(data.num_train_data):
                train_input_data = data.training_input[idx]
                train_target_data = data.training_target[idx]

                ## 각 데이터별로 에러를 모으는 부분 (forward만 뽑아낸 뒤, 에러만 추출)
                sum_train_error += self.session.run(self.error,
                                                    {
                                                        self.input_node: train_input_data,
                                                        self.target_node: train_target_data
                                                    }, verbose2)

                ## bp의 값의 true 여부를 가지고 grads에 넣을 값을 정함
                if bp:
                    grads = self.backward_propagation()
                else:
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
                                                         }, False)

            if epoch % print_period == 0:
                print("Epoch {:3d} Completed - Average Train Error: {:7.6f} - Average Validation Error: {:7.6f}".format(
                    epoch,
                    sum_train_error / data.num_train_data,
                    sum_validation_error / data.num_validation_data
                ))

    def backward_propagation(self):
        pass

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

    def backward_propagation(self):
        grads = {}

        d_error = self.error.backward(1.0)
        d_output = self.output.backward(d_error)
        _ = self.affine.backward(d_output)

        grads['W0'] = self.affine.dw
        grads['b0'] = self.affine.db

        return grads


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

    def backward_propagation(self):
        grads = {}

        d_error = self.error.backward(1.0)
        d_output = self.output.backward(d_error)
        d_affine1 = self.affine1.backward(d_output)
        d_activation0 = self.activation0.backward(d_affine1)
        _ = self.affine0.backward(d_activation0)

        grads['W0'] = self.affine0.dw
        grads['b0'] = self.affine0.db
        grads['W1'] = self.affine1.dw
        grads['b1'] = self.affine1.db

        return grads


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
        self.params['b0'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b0').get_variable()

        self.params['W1'] = initializer(shape=(self.input_size, self.output_size), name='W1').get_variable()
        self.params['b1'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b1').get_variable()

        self.params['W2'] = initializer(shape=(self.input_size, self.output_size), name='W2').get_variable()
        self.params['b2'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b2').get_variable()
        print(self.get_params_str())


    def layering(self, activator=tfe.Activator.ReLU.value):
        self.activator = activator

        self.affine0 = tfl.Affine(self.params['W0'], self.input_node, self.params['b0'], name="A0", graph=self)
        self.activation0 = activator(self.affine0, name="O0", graph=self)

        self.affine1 = tfl.Affine(self.params['W1'], self.input_node, self.params['b1'], name="A1", graph=self)
        self.activation1 = activator(self.affine1, name="O1", graph=self)

        self.affine2 = tfl.Affine2(self.params['W2'], self.activation0, self.activation1, self.params['b2'], name="A2", graph=self)
        self.output = activator(self.affine2, name="O2", graph=self)

        self.error = tfl.SquaredError(self.output, self.target_node, name="SE", graph=self)

    def backward_propagation(self):
        grads = {}

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

        return grads


class Multi_Layer_Network(Neural_Network):
    def __init__(self, input_size, hidden_size_list, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)

        self.params_size_list = None
        self.layers = OrderedDict()

        self.affine0 = None
        self.activation0 = None
        self.affine1 = None
        self.activation1 = None
        self.affine2 = None

        super().__init__(input_size, output_size)

    def initialize_param(self, initializer=tfe.Initializer.Zero.value):
        self.params_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        for idx in range(len(self.params_size_list) - 1):
            self.params['W' + str(idx)] = initializer(shape=(self.params_size_list[idx], self.params_size_list[idx + 1]))
            self.params['b' + str(idx)] = np.zeros(self.params_size_list[idx + 1])

    def layering(self, activator=tfe.Activator.ReLU.value):
        self.activator = activator

        for idx in range(self.hidden_layer_num):
            self.layers['affine' + str(idx)] = tfl.Affine(
                self.params['W' + str(idx)], self.input_node, self.params['b' + str(idx)], name='affine' + str(idx), graph=self
            )
            self.layers['activation' + str(idx)] = activator(self.layers['affine' + str(idx)], name='activation' + str(idx), graph=self)

        idx = self.hidden_layer_num
        self.layers['affine' + str(idx)] = tfl.Affine(
            self.params['W' + str(idx)], self.input_node, self.params['b' + str(idx)], name='affine' + str(idx), graph=self
        )
        self.output = activator(self.layers['affine' + str(idx)], name='output', graph=self)

        #self.last_layer = SoftmaxWithCrossEntropyLoss()


        self.error = tfl.SquaredError(self.output, self.target_node, name="SE", graph=self)