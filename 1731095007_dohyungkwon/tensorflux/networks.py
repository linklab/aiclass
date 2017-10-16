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
            if verbose and epoch % print_period == 0: # print_period번마다 출력
                print()
                print("--------------")
                print("[Epoch {:d}]".format(epoch))
                print()
                verbose2 = True # 뒤에 쓰이는 변수
            else:
                verbose2 = False

            sum_train_error = 0.0
            for idx in range(data.num_train_data):
                train_input_data = data.train_input[idx]
                train_target_data = data.train_target[idx]

                sum_train_error += self.session.run(self.error,
                                                    {
                                                        self.input_node: train_input_data,
                                                        self.target_node: train_target_data
                                                    }, verbose2)

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



# ~171009
# from collections import OrderedDict
# import tensorflux.graph as tfg
# import tensorflux.enums as tfe
# import tensorflux.layers as tfl
# import tensorflux.session as tfs
# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# class Neural_Network(tfg.Graph): # base class
#     def __init__(self, input_size, output_size):
#         self.input_size = input_size
#         self.output_size = output_size
#
#         self.input_node = None # 붙일 노드
#         self.target_node = None # 붙일 노드
#
#         self.activator = None # 네트워크 내에서 활용할 객체
#         self.initializer = None
#         self.optimizer = None
#
#         self.params = OrderedDict() # 순서를 관리하는 dictionary
#         # a = {}대신에, a = OrderdDict()
#         # a[1] = 'aaa'
#         # a[2] = 'bbb'
#         # dictionary는 순서 보장이 안됨
#
#         self.output = None # Optimizer붙이기 전, feed_forward 통과하여 나오는 값
#         self.error = None
#
#         self.session = tfs.Session()
#         super().__init__() # graph..
#
#     def set_data(self, input_node, target_node):
#         self.input_node = input_node
#         self.target_node = target_node
#
#     def initialize_param(self, initializer=tfe.Initializer.Zero.value): # enum이 쓰이는 방식
#         # Zero.value ;enum 문법
#         # Zero_Initializer 클래스가 value가 됨
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
#             # f(x + delta) - f(x - delta) / 2 * delta 계산
#             grads[param_key] = (fxh1 - fxh2) / (2 * delta)
#             param.value = temp_val
#         return grads
#
#     # https://www.dropbox.com/s/ni4r8gyfr1lw2t3/02.Artificial_Single_Neuron.pdf?dl=0
#     # page 18
#     def learning(self, max_epoch, data, x, target, verbose=False):
#         for epoch in range(max_epoch):
#             sum_train_error = 0.0
#             for idx in range(data.num_train_data):
#                 train_input_data = data.training_input[idx]
#                 train_target_data = data.training_target[idx]
#
#                 # this file, line 51
#                 # https://www.dropbox.com/s/ni4r8gyfr1lw2t3/02.Artificial_Single_Neuron.pdf?dl=0
#                 # page 23, 24
#                 grads = self.numerical_derivative(self.session, {x: train_input_data, target: train_target_data})
#                 self.optimizer.update(grads=grads)
#                 sum_train_error += self.session.run(self.error, {x: train_input_data, target: train_target_data}, vervose=False)
#
#             sum_validation_error = 0.0
#             for idx in range(data.num_validation_data): # 검증용 데이터에서는 update를 하지 않음
#                 validation_input_data = data.validation_input[idx]
#                 validation_target_data = data.validation_target[idx]
#                 sum_validation_error += self.session.run(self.error,
#                                                          {x: validation_input_data, target: validation_target_data},
#                                                         vervose=False)
#             if (epoch % 100 == 0):
#                 print("Epoch {:3d} Completed - Average Train Error: {:7.6f} - Average Validation Error: {:7.6f}".format(
#                     epoch, sum_train_error / data.num_train_data, sum_validation_error / data.num_validation_data))
#
#     def get_params_str(self):
#         params_str = ""
#         for param_key, param in self.params.items():
#             params_str = params_str + param_key + ": " + str(param.value) + ", "
#         params_str = params_str[0:-2]
#         return params_str
#
#     def print_feed_forward(self, num_data, input_data, target_data, x):
#         for idx in range(num_data):
#             train_input_data = input_data[idx]
#             train_target_data = target_data[idx]
#
#             # output; by ReLU
#             # vervose True해보기
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
#
#     def initialize_scalar_param(self, value1, value2,
#                                 initializer=tfe.Initializer.Value_Assignment.value):
#         self.params['W0'] = initializer(value1, name='W0').get_variable()
#         #initializer클래스의 생성자 호출. 즉 initializer(value1, name='W0')까지가 객체 생성
#         #initializers.py line22-, line 18-19
#         self.params['b0'] = initializer(value2, name='b0').get_variable()
#         # Variable객체 두 개의 레퍼런스가 param에 할당되는 격
#
#     def initialize_param(self, initializer=tfe.Initializer.Zero.value):
#         self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
#         self.params['b0'] = initializer(shape=(self.output_size,), name='b0').get_variable()
#
#     def layering(self, activator=tfe.Activator.ReLU.value): # layers.py
#         self.activator = activator
#         u = tfl.Affine(self.params['W0'], self.input_node, self.params['b0'], name="A")
#         self.output = activator(u, name="O")
#         self.error = tfl.SquaredError(self.output, self.target_node, name="SE")
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
#         u0 = tfl.Affine(self.params['W0'], self.input_node, self.params['b0'], name="A0")
#         o0 = activator(u0, name="O0") # ReLU 클래스 생성자 격임
#         u1 = tfl.Affine(self.params['W1'], o0, self.params['b1'], name="A1")
#         self.output = activator(u1, name="O1")
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
# # class Three_Neurons_Network(Neural_Network):
# #     def __init__(self, input_size, output_size):
# #         super().__init__(input_size, output_size)
# #
# #     def initialize_param(self, initializer=tfe.Initializer.Zero.value):
# #         self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
# #         self.params['b0'] = initializer(shape=(self.output_size,), name='b0').get_variable()
# #         self.params['W1'] = initializer(shape=(self.input_size, self.output_size), name='W1').get_variable()
# #         self.params['b1'] = initializer(shape=(self.output_size,), name='b1').get_variable()
# #         # self.params['z'] = initializer(shape=(self.input_size, self.output_size), name='z').get_variable()
# #         self.params['W2'] = initializer(shape=(self.input_size, self.output_size), name='W2').get_variable()
# #         self.params['b2'] = initializer(shape=(self.output_size,), name='b2').get_variable()
# #
# #     def layering(self, activator=tfe.Activator.ReLU.value):
# #         self.activator = activator
# #         u0 = tfl.Affine(self.params['W0'], self.input_node, self.params['b0'], name="A0")
# #         o0 = activator(u0, name="O0") # ReLU 클래스 생성자 격임
# #         # self.params['z'].value = o0
# #         u1 = tfl.Affine(self.params['W1'], self.input_node, self.params['b1'], name="A1")
# #         o1 = activator(u1, name="O1")  # ReLU 클래스 생성자 격임
# #         # self.params['z'].value = o1
# #
# #         u2 = tfl.Affine(self.params['W2'], [o0,o1], self.params['b2'], name="A2")
# #         o2 = activator(u2, name="o2")
# #
# #         self.output = activator(o2, name="O2")
# #         self.error = tfl.SquaredError(self.output, self.target_node, name="SE")
# #
# #         if isinstance(self, nx.Graph):
# #             self.add_edge(self.params['W0'], u0)
# #             self.add_edge(self.input_node, u0)
# #             self.add_edge(self.params['b0'], u0)
# #             self.add_edge(u0, o0)
# #
# #             self.add_edge(self.params['W1'], u1)
# #             self.add_edge(self.input_node, u1)
# #             self.add_edge(self.params['b1'], u1)
# #             self.add_edge(u1, o1)
# #
# #             self.add_edge(self.params['W2'], u2)
# #             self.add_edge([o0,o1], u2)
# #             self.add_edge(self.params['b2'], u2)
# #             self.add_edge(u2, self.output)
# #             self.add_edge(self.output, self.error)
# #             self.add_edge(self.error, self.target_node)
#
#
#
# # class Three_Neurons_Network_2(Neural_Network):
# #     def __init__(self, input_size, output_size):
# #         super().__init__(input_size, output_size)
# #
# #     def initialize_param(self, initializer=tfe.Initializer.Zero.value):
# #         self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
# #         self.params['b0'] = initializer(shape=(self.output_size,), name='b0').get_variable()
# #         self.params['W1'] = initializer(shape=(self.input_size, self.output_size), name='W1').get_variable()
# #         self.params['b1'] = initializer(shape=(self.output_size,), name='b1').get_variable()
# #         self.params['W2'] = initializer(shape=(self.output_size, self.output_size), name='W2').get_variable()
# #         self.params['b2'] = initializer(shape=(self.output_size,), name='b2').get_variable()
# #
# #     def layering(self, activator=tfe.Activator.ReLU.value):
# #         self.activator = activator
# #         u0 = tfl.Affine(self.params['W0'], self.input_node, self.params['b0'], name="A0")
# #         o0 = activator(u0, name="O0") # ReLU 클래스 생성자 격임
# #         u1 = tfl.Affine(self.params['W1'], self.input_node, self.params['b1'], name="A1")
# #         o1 = activator(u1, name="O1")  # ReLU 클래스 생성자 격임
# #         u2 = tfl.Affine(self.params['W2'], o1, self.params['b2'], name="A2")
# #         self.output = activator(u2, name="O2")
# #         self.error = tfl.SquaredError(self.output, self.target_node, name="SE")
# #         if isinstance(self, nx.Graph):
# #             self.add_edge(self.params['W0'], u0)
# #             self.add_edge(self.input_node, u0)
# #             self.add_edge(self.params['b0'], u0)
# #             self.add_edge(u0, o0)
# #
# #             self.add_edge(self.params['W1'], u1)
# #             self.add_edge(self.input_node, u1)
# #             self.add_edge(self.params['b1'], u1)
# #             self.add_edge(u1, o1)
# #
# #             self.add_edge(self.params['W2'], u2)
# #             self.add_edge(o0, u2)
# #             self.add_edge(o1, u2)
# #             self.add_edge(self.params['b2'], u2)
# #             self.add_edge(u2, self.output)
# #             self.add_edge(self.output, self.error)
# #             self.add_edge(self.error, self.target_node)
#
#
# #
# # class Three_Neurons_Network(Neural_Network):
# #     def __init__(self, input_size, output_size):
# #         super().__init__(input_size, output_size)
# #
# #     def initialize_param(self, initializer=tfe.Initializer.Zero.value):
# #         self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
# #         self.params['b0'] = initializer(shape=(self.output_size,), name='b0').get_variable()
# #         self.params['W1'] = initializer(shape=(self.output_size, self.output_size), name='W1').get_variable()
# #         self.params['b1'] = initializer(shape=(self.output_size,), name='b1').get_variable()
# #         self.params['W2'] = initializer(shape=(self.output_size, self.output_size), name='W2').get_variable()
# #         self.params['b2'] = initializer(shape=(self.output_size,), name='b2').get_variable()
# #         print(self.get_params_str)
# #
# #     def layering(self, activator=tfe.Activator.ReLU.value):
# #         self.activator = activator
# #         u0 = tfl.Affine(self.params['W0'], self.input_node, self.params['b0'], name="A0")
# #         o0 = activator(u0, name="O0") # ReLU 클래스 생성자 격임
# #         u1 = tfl.Affine(self.params['W1'], o0, self.params['b1'], name="A1")
# #         o1 = activator(u1, name="O1")  # ReLU 클래스 생성자 격임
# #         u2 = tfl.Affine(self.params['W2'], o1, self.params['b2'], name="A2")
# #         self.output = activator(u2, name="O2")
# #         self.error = tfl.SquaredError(self.output, self.target_node, name="SE")
# #         if isinstance(self, nx.Graph):
# #             self.add_edge(self.params['W0'], u0)
# #             self.add_edge(self.input_node, u0)
# #             self.add_edge(self.params['b0'], u0)
# #             self.add_edge(u0, o0)
# #             self.add_edge(self.params['W1'], u1)
# #             self.add_edge(o0, u1)
# #             self.add_edge(self.params['b1'], u1)
# #             self.add_edge(u1, o1)
# #             self.add_edge(self.params['W2'], u2)
# #             self.add_edge(o1, u2)
# #             self.add_edge(self.params['b2'], u2)
# #             self.add_edge(u2, self.output)
# #             self.add_edge(self.output, self.error)
# #             self.add_edge(self.error, self.target_node)
# #
# #
# #
# # # 교수님 coding
# # class Three_Neurons_Network(Neural_Network):
# #     def __init__(self, input_size, output_size):
# #         super().__init__(input_size, output_size)
# #
# #     def initialize_param(self, initializer=tfe.Initializer.Zero.value):
# #         self.params['W0'] = initializer(shape=(self.input_size, self.output_size), name='W0').get_variable()
# #         self.params['b0'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b0').get_variable()
# #
# #         self.params['W1'] = initializer(shape=(self.input_size, self.output_size), name='W1').get_variable()
# #         self.params['b1'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b0').get_variable()
# #
# #         self.params['W2'] = initializer(shape=(self.input_size, self.output_size), name='W2').get_variable()
# #         self.params['b2'] = tfe.Initializer.Point_One.value(shape=(self.output_size,), name='b0').get_variable()
# #
# #     def layering(self, activator=tfe.Activator.ReLU.value):
# #         self.activator = activator
# #
# #         u0 = tfl.Affine(self.params['W0'], self.input_node, self.params['b0'], name="A0")
# #         o0 = activator(u0, name="O0", graph=self)
# #
# #         u1 = tfl.Affine(self.params['W1'], self.input_node, self.params['b1'], name="A1")
# #         o1 = activator(u1, name="O1", graph=self)
# #
# #         # [o0, o1] 노드화가 필요
# #         # Variable과 placeholder 등은 말단에 있어야지 아래와 같이 레이어링 중간에 있어서는 안됨
# #         # k = tfg.Variable([o0, o1])
# #         # session.py.. Variable이 왜 중간에 들어가면 안된다고 한건지.. 이해 x
# #         u2 = tfl.Affine2(self.params['W2'], o0, o1, self.params['b2'], name="A2")
# #         self.output = activator(u2, name="O2", graph=self)
# #
# #         self.error = tfl.SquaredError(self.output, self.target_node, name="SE")
#
#
# # HJM
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
#         print(self.get_params_str)
#
#     def layering(self, activator=tfe.Activator.ReLU.value):
#         self.activator = activator
#         u0 = tfl.Affine(self.params['W0'], self.input_node, self.params['b0'], name="A0")
#         o0 = activator(u0, name="O0")
#         u1 = tfl.Affine(self.params['W1'], self.input_node, self.params['b1'], name="A1")
#         o1 = activator(u1, name="O1")
#         u2 = tfl.Affine2(self.params['W2'], o0, o1, self.params['b2'], name="A2")
#
#         self.output = activator(u2, name="O2")
#         # self.output = activator(z1, name="O2")
#         self.error = tfl.SquaredError(self.output, self.target_node, name="SE")
#
#         if isinstance(self, nx.Graph):
#             self.add_edge(self.params['W0'], u0)
#             self.add_edge(self.input_node, u0)
#             self.add_edge(self.params['b0'], u0)
#             self.add_edge(u0, o0)
#
#             self.add_edge(self.params['W1'], u1)
#             self.add_edge(self.input_node, u1)
#             self.add_edge(self.params['b1'], u1)
#             self.add_edge(u1, o1)
#
#             self.add_edge(self.params['W2'], u2)
#             self.add_edge(o0, u2)
#             self.add_edge(o1, u2)
#             self.add_edge(self.params['b2'], u2)
#             self.add_edge(u2, self.output)
#             self.add_edge(self.output, self.error)
#             self.add_edge(self.error, self.target_node)
#
