from collections import OrderedDict
import tensorflux.graph as tfg
import tensorflux.enums as tfe
import tensorflux.layers as tfl
import tensorflux.session as tfs
import tensorflux.functions as tff
import tensorflux.initializers as tfi
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
from networkx.drawing.nx_agraph import graphviz_layout


class Deep_Neural_Network(tfg.Graph):
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
        self.max_epoch = None

        self.session = tfs.Session()
        super().__init__()

    def set_data_node(self, input_node, target_node):
        self.input_node = input_node
        self.target_node = target_node

    def initialize_param(self, initializer=tfe.Initializer.Zero.value):
        pass

    def layering(self, activator=tfe.Activator.ReLU.value):
        pass

    def set_optimizer(self, optimizer=tfe.Optimizer.SGD.value, learning_rate=0.01):
        self.optimizer = optimizer(learning_rate=learning_rate)
        self.optimizer.params = self.params

    def backward_propagation(self):
        pass

    def get_params_str(self):
        params_str = ""
        for param_key, param in self.params.items():
            params_str = params_str + param_key + ": " + str(param.value) + ", "
        params_str = params_str[0:-2]
        return params_str

    def get_all_param_describe(self):
        """
        :return: starts.description
        skewness - https://ko.wikipedia.org/wiki/%EB%B9%84%EB%8C%80%EC%B9%AD%EB%8F%84
        kurtosis - https://ko.wikipedia.org/wiki/%EC%B2%A8%EB%8F%84
        """
        all_param_flatten_list = []
        for param in self.params.values():
            all_param_flatten_list.extend([item for item in param.value.flatten()])
        return stats.describe(np.array(all_param_flatten_list))

    def print_feed_forward(self, num_data, input_data, target_data, verbose=False):
        for idx in range(num_data):
            train_input_data = input_data[idx]
            train_target_data = target_data[idx]

            output = self.session.run(self.output, {self.input_node: train_input_data}, verbose)
            print("Input Data: {:>5}, Feed Forward Output: {:>6}, Target: {:>6}".format(
                str(train_input_data), np.array2string(output), str(train_target_data)))

    def draw_and_show(self, figsize=(8, 8)):
        pos = graphviz_layout(self)
        plt.figure(figsize=figsize)
        nx.draw_networkx(self, pos=pos, with_labels=True)
        plt.show(block=True)

    def draw_error_values_and_accuracy(self, figsize=(20, 5)):
        # Draw Error Values and Accuracy
        plt.figure(figsize=figsize)

        epoch_list = np.arange(self.max_epoch)

        plt.subplot(121)
        plt.plot(epoch_list, self.train_error_list, 'r', label='Train')
        plt.plot(epoch_list, self.validation_error_list, 'g', label='Validation')
        plt.ylabel('Error')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='upper right')

        plt.subplot(122)
        plt.plot(epoch_list, self.test_accuracy_list, 'b', label='Test')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.show()


class Multi_Layer_Network(Deep_Neural_Network):
    def __init__(self,
                 input_size,
                 hidden_size_list,
                 output_size,
                 input_node=None,
                 target_node=None,
                 init_mean=0.0,
                 init_sd=0.01,
                 activator=tfe.Activator.ReLU.value,
                 optimizer=tfe.Optimizer.SGD.value,
                 learning_rate=0.01):

        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)

        self.params_size_list = None
        self.layers = OrderedDict()

        self.train_error_list = []
        self.validation_error_list = []
        self.test_accuracy_list = []

        super().__init__(input_size, output_size)

        self.set_data_node(input_node, target_node)
        self.initialize_normal_random_param(mean=init_mean, sd=init_sd)
        self.layering(activator)
        self.set_optimizer(optimizer, learning_rate)

    def initialize_param(self, initializer=tfe.Initializer.Zero.value):
        self.params_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(self.hidden_layer_num + 1):
            self.params['W' + str(idx)] = initializer(
                shape=(self.params_size_list[idx], self.params_size_list[idx + 1]),
                name="W" + str(idx)
            ).get_variable()

            self.params['b' + str(idx)] = initializer(
                shape=(self.params_size_list[idx + 1],),
                name="b" + str(idx)
            ).get_variable()

    def initialize_normal_random_param(self, mean=0.0, sd=0.1):
        self.params_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(self.hidden_layer_num + 1):
            self.params['W' + str(idx)] = tfi.Random_Normal_Initializer(
                shape=(self.params_size_list[idx], self.params_size_list[idx + 1]),
                name="W" + str(idx),
                mean=mean,
                sd=sd
            ).get_variable()

            self.params['b' + str(idx)] = tfi.Random_Normal_Initializer(
                shape=(self.params_size_list[idx + 1],),
                name="b" + str(idx),
                mean=mean,
                sd=sd
            ).get_variable()

    def layering(self, activator=tfe.Activator.ReLU.value):
        self.activator = activator

        input_node = self.input_node
        for idx in range(self.hidden_layer_num):
            self.layers['affine' + str(idx)] = tfl.Affine(
                self.params['W' + str(idx)],
                input_node,
                self.params['b' + str(idx)],
                name='affine' + str(idx),
                graph=self
            )
            self.layers['activation' + str(idx)] = activator(
                self.layers['affine' + str(idx)],
                name='activation' + str(idx),
                graph=self
            )
            input_node = self.layers['activation' + str(idx)]

        idx = self.hidden_layer_num
        self.layers['affine' + str(idx)] = tfl.Affine(
            self.params['W' + str(idx)],
            self.layers['activation' + str(idx - 1)],
            self.params['b' + str(idx)],
            name='affine' + str(idx),
            graph=self
        )
        self.output = self.layers['affine' + str(idx)]

        self.error = tfl.SoftmaxWithCrossEntropyLoss(self.output, self.target_node, name="SCEL", graph=self)

    def feed_forward(self, input_data):
        return self.session.run(self.output, {self.input_node: input_data}, verbose=False)

    def backward_propagation(self):
        grads = {}

        d_error = self.error.backward(1.0)
        din = d_error

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            din = layer.backward(din)

        for idx in range(self.hidden_layer_num + 1):
            grads['W' + str(idx)] = self.layers['affine' + str(idx)].dw
            grads['b' + str(idx)] = self.layers['affine' + str(idx)].db

        return grads

    def learning(self, max_epoch, data, batch_size=1000, print_period=10, verbose=False):
        self.max_epoch = max_epoch

        if verbose:
            self.draw_params_histogram()

        num_batch = math.ceil(data.num_train_data / batch_size)

        for epoch in range(max_epoch):
            for i in range(num_batch):
                i_batch = data.train_input[i * batch_size: i * batch_size + batch_size]
                t_batch = data.train_target[i * batch_size: i * batch_size + batch_size]

                #forward
                self.session.run(self.error,
                                {
                                    self.input_node: i_batch,
                                    self.target_node: t_batch
                                }, False)

                #backward
                grads = self.backward_propagation()

                self.optimizer.update(grads=grads)

            batch_mask = np.random.choice(data.num_train_data, batch_size)
            i_batch = data.train_input[batch_mask]
            t_batch = data.train_target[batch_mask]

            train_error = self.session.run(self.error,
                                         {
                                             self.input_node: i_batch,
                                             self.target_node: t_batch
                                         }, False)
            self.train_error_list.append(train_error)

            validation_error = self.session.run(self.error,
                                         {
                                             self.input_node: data.validation_input,
                                             self.target_node: data.validation_target
                                         }, False)
            self.validation_error_list.append(validation_error)

            forward_final_output = self.feed_forward(input_data=data.test_input)

            test_accuracy = tff.accuracy(forward_final_output, data.test_target)
            self.test_accuracy_list.append(test_accuracy)

            if epoch % print_period == 0:
                print("Epoch {:3d} Completed - Train Error: {:7.6f} - Validation Error: {:7.6f} - Test Accuracy: {:7.6f}".format(
                    epoch,
                    float(train_error),
                    float(validation_error),
                    float(test_accuracy)
                ))

                if verbose:
                    self.draw_params_histogram()
                    print()

    def draw_params_histogram(self):
        f, axarr = plt.subplots(1, (self.hidden_layer_num + 1) * 2, figsize=(10 * (self.hidden_layer_num + 1), 5))

        for idx in range(self.hidden_layer_num + 1):
            w_values = self.layers['affine' + str(idx)].w_value.flatten()
            b_values = self.layers['affine' + str(idx)].b_value.flatten()

            axarr[idx].hist(w_values, 20)
            axarr[idx].set_title("W{:d}, mean: {:5.4f}, std: {:5.4f}".format(idx, np.mean(w_values), np.std(w_values)))

            axarr[idx + 3].hist(b_values, 20)
            axarr[idx + 3].set_title("b{:d}, mean: {:5.4f}, std: {:5.4f}".format(idx, np.mean(b_values), np.std(b_values)))

        f.subplots_adjust(wspace=0.5)
        plt.show()

    def draw_false_prediction(self, test_input, test_target, num=5, figsize=(20, 5)):
        forward_final_output = self.feed_forward(input_data=test_input)
        y = np.argmax(forward_final_output, axis=1)
        target = np.argmax(test_target, axis=1)

        diff_index_list = []
        for i in range(test_input):
            if y[i] != target[i]:
                diff_index_list.append(i)
        plt.figure(figsize=figsize)

        for i in range(num):
            j = diff_index_list[i]
            print("False Prediction Index: %s, Prediction: %s, Ground Truth: %s" % (j, y[j], target[j]))
            img = np.array(test_input[j])
            img.shape = (28, 28)
            plt.subplot(150 + (i + 1))
            plt.imshow(img, cmap='gray')

    def get_param_describe(self, layer_num=0, kind="W"):
        assert layer_num <= self.hidden_layer_num

        if kind == "W":
            param_flatten_list = self.params['affine' + str(layer_num)].w_value.flatten()
        else:
            param_flatten_list = self.params['affine' + str(layer_num)].b_value.flatten()
        return stats.describe(np.array(param_flatten_list))
