from collections import OrderedDict
import tensorflux.enums as tfe
import tensorflux.layers as tfl
import tensorflux.session as tfs
import tensorflux.functions as tff
import matplotlib.pyplot as plt
import networkx as nx
import tensorflux.Deep_Neural_Network as dnn
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
from scipy import stats
import math
import pickle
import sys
import random
import string

"""
    conv - relu - conv - relu - pool -
    conv - relu - conv - relu - pool -
    conv - relu - conv - relu - pool -
    affine - relu - dropout - affine - dropout - softmax
"""

class CNN(dnn.Deep_Neural_Network):
    def __init__(self,
                 input_dim,
                 conv_param_list,
                 fc_size,
                 output_size,
                 input_node=None,
                 target_node=None,
                 initializer=tfe.Initializer.Normal.value,
                 init_sd=0.01,
                 activator=tfe.Activator.ReLU.value,
                 optimizer=tfe.Optimizer.SGD.value,
                 learning_rate=0.01):

        super().__init__()

        self.input_dim = input_dim
        self.conv_param_list = conv_param_list
        self.fc_size = fc_size
        self.output_size = output_size

        self.input_node = input_node
        self.target_node = target_node

        self.activator = activator
        self.initializer = initializer
        self.optimizer = optimizer(learning_rate=learning_rate)

        self.params = {}
        self.optimal_epoch_and_params = None

        self.output = None
        self.error = None
        self.max_epoch = None

        self.session = tfs.Session()

        self.mode_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

        print("Multi Layer Network Model - ID:", self.mode_id)

        self.params_size_list = None
        self.layers = OrderedDict()

        self.train_error_list = []
        self.validation_error_list = []
        self.test_accuracy_list = []

        self.min_validation_error_epoch = sys.maxsize
        self.min_train_error = sys.float_info.max
        self.min_validation_error = sys.float_info.max
        self.min_fold_idx = sys.maxsize
        self.test_accuracy_at_min_validation_error_epoch = 0.0

        self.min_validation_error_per_fold = []

        self.param_mean_list = {}
        self.param_variance_list = {}
        self.param_skewness_list = {}
        self.param_kurtosis_list = {}

        self.output_mean_list = {}
        self.output_variance_list = {}
        self.output_skewness_list = {}
        self.output_kurtosis_list = {}

        self.initialize_param(sd=init_sd)
        self.layering()

    def get_conv_layer_output_size(self, filter_num, pre_channel_num, input_size, filter_size, padding_size, stride_size):
        image_output_size = (input_size - filter_size + 2 * padding_size) / stride_size + 1
        return (filter_num, pre_channel_num, image_output_size, image_output_size)

    def get_pooling_layer_output_size(self, filter_num, pre_channel_num, input_size, filter_size, stride_size):
        image_output_size = (input_size - filter_size) / stride_size + 1
        return (filter_num, pre_channel_num, image_output_size, image_output_size)

    def initialize_param(self, mean=0.0, sd=1.0):
        #self.pre_param_size_list = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, fc_size])
        self.params_size_list = []
        channel_num = self.input_dim[0]
        for conv_param in self.conv_param_list:
            self.params_size_list.append(channel_num * conv_param['filter_size'] * conv_param['filter_size'])
            channel_num = conv_param['filter_num']

        self.params_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        self.param_mean_list['W'] = {}
        self.param_variance_list['W'] = {}
        self.param_skewness_list['W'] = {}
        self.param_kurtosis_list['W'] = {}

        self.param_mean_list['b'] = {}
        self.param_variance_list['b'] = {}
        self.param_skewness_list['b'] = {}
        self.param_kurtosis_list['b'] = {}

        for idx in range(self.hidden_layer_num + 1):
            if self.initializer is tfe.Initializer.Normal.value:
                self.params['W' + str(idx)] = self.initializer(
                    shape=(self.params_size_list[idx], self.params_size_list[idx + 1]),
                    name="W" + str(idx),
                    mean=mean,
                    sd=sd
                ).param
            elif self.initializer is tfe.Initializer.Truncated_Normal.value:
                self.params['W' + str(idx)] = self.initializer(
                    shape=(self.params_size_list[idx], self.params_size_list[idx + 1]),
                    name="W" + str(idx),
                    mean=mean,
                    sd=sd,
                    low=-sd,
                    upp=sd
                ).param
            else:
                self.params['W' + str(idx)] = self.initializer(
                    shape=(self.params_size_list[idx], self.params_size_list[idx + 1]),
                    name="W" + str(idx)
                ).param

            self.params['b' + str(idx)] = tfe.Initializer.Zero.value(
                shape=(self.params_size_list[idx + 1],),
                name="b" + str(idx)
            ).param

            self.param_mean_list['W'][idx] = []
            self.param_variance_list['W'][idx] = []
            self.param_skewness_list['W'][idx] = []
            self.param_kurtosis_list['W'][idx] = []

            self.param_mean_list['b'][idx] = []
            self.param_variance_list['b'][idx] = []
            self.param_skewness_list['b'][idx] = []
            self.param_kurtosis_list['b'][idx] = []

    def layering(self, refitting=False):
        input_node = self.input_node

        if not refitting:
            self.output_mean_list['affine'] = {}
            self.output_variance_list['affine'] = {}
            self.output_skewness_list['affine'] = {}
            self.output_kurtosis_list['affine'] = {}

            self.output_mean_list['activation'] = {}
            self.output_variance_list['activation'] = {}
            self.output_skewness_list['activation'] = {}
            self.output_kurtosis_list['activation'] = {}

        for idx in range(self.hidden_layer_num):
            self.layers['affine' + str(idx)] = tfl.Affine(
                self.params['W' + str(idx)],
                input_node,
                self.params['b' + str(idx)],
                name='affine' + str(idx),
                graph=self
            )
            self.layers['activation' + str(idx)] = self.activator(
                self.layers['affine' + str(idx)],
                name='activation' + str(idx),
                graph=self
            )
            input_node = self.layers['activation' + str(idx)]

            if not refitting:
                self.output_mean_list['affine'][idx] = []
                self.output_variance_list['affine'][idx] = []
                self.output_skewness_list['affine'][idx] = []
                self.output_kurtosis_list['affine'][idx] = []

                self.output_mean_list['activation'][idx] = []
                self.output_variance_list['activation'][idx] = []
                self.output_skewness_list['activation'][idx] = []
                self.output_kurtosis_list['activation'][idx] = []

        idx = self.hidden_layer_num
        self.layers['affine' + str(idx)] = tfl.Affine(
            self.params['W' + str(idx)],
            self.layers['activation' + str(idx - 1)],
            self.params['b' + str(idx)],
            name='affine' + str(idx),
            graph=self
        )
        self.output = self.layers['affine' + str(idx)]

        if not refitting:
            self.output_mean_list['affine'][idx] = []
            self.output_variance_list['affine'][idx] = []
            self.output_skewness_list['affine'][idx] = []
            self.output_kurtosis_list['affine'][idx] = []

        self.error = tfl.SoftmaxWithCrossEntropyLoss(self.output, self.target_node, name="SCEL", graph=self)

    def feed_forward(self, input_data, is_numba=False):
        return self.session.run(self.output, {self.input_node: input_data}, is_numba, verbose=False)

    def backward_propagation(self, is_numba):
        grads = {}

        d_error = self.error.backward(1.0, is_numba)
        din = d_error

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            din = layer.backward(din, is_numba)

        for idx in range(self.hidden_layer_num + 1):
            grads['W' + str(idx)] = self.layers['affine' + str(idx)].dw
            grads['b' + str(idx)] = self.layers['affine' + str(idx)].db

        return grads

    def learning(self, max_epoch, data, batch_size=1000, print_period=10, is_numba=False, verbose=False):
        print("-- Learning Started --")
        self.max_epoch = max_epoch

        for fold_idx in range(data.n_splits):
            data.set_next_train_and_validation_data()
            num_batch = math.ceil(data.num_train_data / batch_size)

            if fold_idx == 0:
                self.set_learning_process_specification(data, batch_size, 0, print_period, is_numba, fold_idx, max_epoch, verbose)

            print("Fold: ", fold_idx)
            for epoch in range(1, max_epoch + 1):
                for i in range(num_batch):
                    i_batch = data.train_input[i * batch_size: i * batch_size + batch_size]
                    t_batch = data.train_target[i * batch_size: i * batch_size + batch_size]

                    #forward
                    self.session.run(
                        self.error,
                        {
                            self.input_node: i_batch,
                            self.target_node: t_batch
                        },
                        is_numba=is_numba,
                        verbose=False)

                    #backward
                    if isinstance(self.optimizer, tfe.Optimizer.NAG.value):
                        #cloned_network = copy.deepcopy(self)
                        cloned_network = pickle.loads(pickle.dumps(self, -1))
                        self.optimizer.update(params=self.params, cloned_network=cloned_network, is_numba=is_numba)
                    else:
                        grads = self.backward_propagation(is_numba)
                        self.optimizer.update(params=self.params, grads=grads)

                self.set_learning_process_specification(data, batch_size, epoch, print_period, is_numba, fold_idx, max_epoch, verbose)

            print()

            self.min_train_error = float(self.train_error_list[self.min_validation_error_epoch])
            self.min_validation_error = float(self.validation_error_list[self.min_validation_error_epoch])
            self.test_accuracy_at_min_validation_error_epoch = float(self.test_accuracy_list[self.min_validation_error_epoch])

            self.min_validation_error_per_fold.append(self.min_validation_error)

            print("[Best Epoch (based on Validation Error) and Its Performance]")
            print("Global Epoch:{:3d} (Fold:{:3d} & Epoch:{:3d}) - Train Error:{:6.5f} - Validation Error:{:6.5f} - Test Accuracy:{:6.5f}".format(
                self.min_validation_error_epoch,
                self.min_fold_idx,
                self.min_validation_error_epoch - self.max_epoch * self.min_fold_idx,
                self.min_train_error,
                self.min_validation_error,
                self.test_accuracy_at_min_validation_error_epoch
            ))
            print()

        self.load_params(data.n_splits)
        self.layering(refitting=True)

        self.mean_min_validation_error_for_all_folds = np.mean(self.min_validation_error_per_fold)

        print("Params are set to the best model!!!")
        print("-- Learning Finished --")
        print()

    def set_learning_process_specification(self, data, batch_size, epoch, print_period, is_numba, fold_idx, max_epoch, verbose):
        batch_mask = np.random.choice(data.num_train_data, batch_size)
        i_batch = data.train_input[batch_mask]
        t_batch = data.train_target[batch_mask]

        train_error = self.session.run(
            self.error,
            {
                self.input_node: i_batch,
                self.target_node: t_batch
            },
            is_numba=is_numba,
            verbose=False)
        self.train_error_list.append(train_error)

        validation_error = self.session.run(
            self.error,
            {
                self.input_node: data.validation_input,
                self.target_node: data.validation_target
            },
            is_numba=is_numba,
            verbose=False)
        self.validation_error_list.append(validation_error)

        min_flag = False
        if validation_error < self.min_validation_error:
            self.min_validation_error = validation_error
            self.min_validation_error_epoch = epoch + fold_idx * max_epoch
            self.min_fold_idx = fold_idx
            self.save_params()
            min_flag = True

        forward_final_output = self.feed_forward(input_data=data.test_input, is_numba=is_numba)

        test_accuracy = tff.accuracy(forward_final_output, data.test_target)
        self.test_accuracy_list.append(test_accuracy)

        for idx in range(self.hidden_layer_num + 1):
            d = self.get_param_describe(layer_num=idx, kind="W")
            self.param_mean_list['W'][idx].append(d.mean)
            self.param_variance_list['W'][idx].append(d.variance)
            self.param_skewness_list['W'][idx].append(d.skewness)
            self.param_kurtosis_list['W'][idx].append(d.kurtosis)

            d = self.get_param_describe(layer_num=idx, kind="b")
            self.param_mean_list['b'][idx].append(d.mean)
            self.param_variance_list['b'][idx].append(d.variance)
            self.param_skewness_list['b'][idx].append(d.skewness)
            self.param_kurtosis_list['b'][idx].append(d.kurtosis)

            d = self.get_activation_describe(layer_num=idx, kind="affine")
            self.output_mean_list['affine'][idx].append(d.mean)
            self.output_variance_list['affine'][idx].append(d.variance)
            self.output_skewness_list['affine'][idx].append(d.skewness)
            self.output_kurtosis_list['affine'][idx].append(d.kurtosis)

            if idx != self.hidden_layer_num:
                d = self.get_activation_describe(layer_num=idx, kind="activation")
                self.output_mean_list['activation'][idx].append(d.mean)
                self.output_variance_list['activation'][idx].append(d.variance)
                self.output_skewness_list['activation'][idx].append(d.skewness)
                self.output_kurtosis_list['activation'][idx].append(d.kurtosis)

        if epoch % print_period == 0:
            print(
                "Epoch {:3d} Completed - Train Error:{:6.5f} - Validation Error:{:6.5f} - Test Accuracy:{:6.5f}".format(
                    epoch,
                    float(train_error),
                    float(validation_error),
                    float(test_accuracy)
                ),
                end=""
            )
            if min_flag:
                print(" <== Minimal Val. Error")
            else:
                print()

            if verbose:
                self.draw_params_histogram()
                for idx in range(self.hidden_layer_num + 1):
                    desc_obj = self.get_param_describe(layer_num=idx, kind="W")
                    num = "{:10d}".format(desc_obj.nobs)
                    min = "{:5.4f}".format(desc_obj.minmax[0])
                    max = "{:5.4f}".format(desc_obj.minmax[1])
                    mean = "{:5.4f}".format(desc_obj.mean)
                    variance = "{:5.4f}".format(desc_obj.variance)
                    skewness = "{:5.4f}".format(desc_obj.skewness)
                    kurtosis = "{:5.4f}".format(desc_obj.kurtosis)

                    print('W' + str(idx) + '-',
                          "num:{:10s}, min:{:5s}, max:{:5s}, mean:{:5s}, variance:{:5s}, skewness:{:5s}, kurtosis:{:5s}".format(
                              num, min, max, mean, variance, skewness, kurtosis
                          )
                    )

                for idx in range(self.hidden_layer_num + 1):
                    desc_obj = self.get_param_describe(layer_num=idx, kind="b")
                    num = "{:10d}".format(desc_obj.nobs)
                    min = "{:5.4f}".format(desc_obj.minmax[0])
                    max = "{:5.4f}".format(desc_obj.minmax[1])
                    mean = "{:5.4f}".format(desc_obj.mean)
                    variance = "{:5.4f}".format(desc_obj.variance)
                    skewness = "{:5.4f}".format(desc_obj.skewness)
                    kurtosis = "{:5.4f}".format(desc_obj.kurtosis)

                    print('b' + str(idx) + '-',
                          "num:{:10s}, min:{:5s}, max:{:5s}, mean:{:5s}, variance:{:5s}, skewness:{:5s}, kurtosis:{:5s}".format(
                              num, min, max, mean, variance, skewness, kurtosis
                          )
                    )

                print()

    def save_params(self):
        #optimal_params = copy.deepcopy(self.params)
        optimal_params = pickle.loads(pickle.dumps(self.params, -1))
        self.optimal_epoch_and_params = [self.min_validation_error_epoch, optimal_params]

    def load_params(self, n_splits):
        acc_epoch = self.optimal_epoch_and_params[0]
        o_epoch = acc_epoch - self.max_epoch * self.min_fold_idx

        print("Load Params from Fold {:3d} & Epoch {:3d}".format(self.min_fold_idx, o_epoch))
        self.params = self.optimal_epoch_and_params[1]

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

    def print_feed_forward(self, num_data, input_data, target_data, is_numba, verbose=False):
        for idx in range(num_data):
            train_input_data = input_data[idx]
            train_target_data = target_data[idx]

            output = self.session.run(self.output, {self.input_node: train_input_data}, is_numba, verbose)
            print("Input Data: {:>5}, Feed Forward Output: {:>6}, Target: {:>6}".format(
                str(train_input_data), np.array2string(output), str(train_target_data)))

    def draw_and_show(self, figsize=(8, 8)):
        pos = graphviz_layout(self)
        plt.figure(figsize=figsize)
        nx.draw_networkx(self, pos=pos, with_labels=True)
        plt.show(block=True)

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

    def draw_error_values_and_accuracy(self, figsize=(20, 5)):
        # Draw Error Values and Accuracy
        plt.figure(figsize=figsize)

        epoch_list = np.arange(len(self.train_error_list))

        plt.subplot(121)
        plt.plot(epoch_list, self.train_error_list, 'r', label='Train')
        plt.plot(epoch_list, self.validation_error_list, 'g', label='Validation')
        plt.ylabel('Error')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(122)
        plt.plot(epoch_list, self.test_accuracy_list, 'b', label='Test')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')
        plt.show()

    def draw_param_description(self, figsize=(20, 5)):
        # Draw Error Values and Accuracy
        plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=.5)

        epoch_list = np.arange(len(self.param_mean_list['W'][0]))

        color_dic = {
            0: 'r',
            1: 'b',
            2: 'g',
        }

        plt.subplot(241)
        for idx in range(self.hidden_layer_num + 1):
            plt.plot(epoch_list, self.param_mean_list['W'][idx], color_dic[idx], label='W' + str(idx))
        plt.ylabel('Mean')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(242)
        for idx in range(self.hidden_layer_num + 1):
            plt.plot(epoch_list, self.param_variance_list['W'][idx], color_dic[idx], label='W' + str(idx))
        plt.ylabel('Variance')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(243)
        for idx in range(self.hidden_layer_num + 1):
            plt.plot(epoch_list, self.param_skewness_list['W'][idx], color_dic[idx], label='W' + str(idx))
        plt.ylabel('Skewness')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(244)
        for idx in range(self.hidden_layer_num + 1):
            plt.plot(epoch_list, self.param_kurtosis_list['W'][idx], color_dic[idx], label='W' + str(idx))
        plt.ylabel('Kurtosis')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(245)
        for idx in range(self.hidden_layer_num + 1):
            plt.plot(epoch_list, self.param_mean_list['b'][idx], color_dic[idx], label='b' + str(idx))
        plt.ylabel('Mean')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(246)
        for idx in range(self.hidden_layer_num + 1):
            plt.plot(epoch_list, self.param_variance_list['b'][idx], color_dic[idx], label='b' + str(idx))
        plt.ylabel('Variance')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(247)
        for idx in range(self.hidden_layer_num + 1):
            plt.plot(epoch_list, self.param_skewness_list['b'][idx], color_dic[idx], label='b' + str(idx))
        plt.ylabel('Skewness')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(248)
        for idx in range(self.hidden_layer_num + 1):
            plt.plot(epoch_list, self.param_kurtosis_list['b'][idx], color_dic[idx], label='b' + str(idx))
        plt.ylabel('Kurtosis')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.show()

    def draw_output_description(self, figsize=(20, 5)):
        plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=.5)

        epoch_list = np.arange(len(self.output_mean_list['affine'][0]))

        color_dic = {
            0: 'r',
            1: 'b',
            2: 'g',
        }

        plt.subplot(241)
        for idx in range(self.hidden_layer_num + 1):
            plt.plot(epoch_list, self.output_mean_list['affine'][idx], color_dic[idx], label='affine' + str(idx))
        plt.ylabel('Mean')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(242)
        for idx in range(self.hidden_layer_num + 1):
            plt.plot(epoch_list, self.output_variance_list['affine'][idx], color_dic[idx], label='affine' + str(idx))
        plt.ylabel('Variance')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(243)
        for idx in range(self.hidden_layer_num + 1):
            plt.plot(epoch_list, self.output_skewness_list['affine'][idx], color_dic[idx], label='affine' + str(idx))
        plt.ylabel('Skewness')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(244)
        for idx in range(self.hidden_layer_num + 1):
            plt.plot(epoch_list, self.output_kurtosis_list['affine'][idx], color_dic[idx], label='affine' + str(idx))
        plt.ylabel('Kurtosis')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(245)
        for idx in range(self.hidden_layer_num):
            plt.plot(epoch_list, self.output_mean_list['activation'][idx], color_dic[idx], label='activation' + str(idx))
        plt.ylabel('Mean')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(246)
        for idx in range(self.hidden_layer_num):
            plt.plot(epoch_list, self.output_variance_list['activation'][idx], color_dic[idx], label='activation' + str(idx))
        plt.ylabel('Variance')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(247)
        for idx in range(self.hidden_layer_num):
            plt.plot(epoch_list, self.output_skewness_list['activation'][idx], color_dic[idx], label='activation' + str(idx))
        plt.ylabel('Skewness')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.subplot(248)
        for idx in range(self.hidden_layer_num):
            plt.plot(epoch_list, self.output_kurtosis_list['activation'][idx], color_dic[idx], label='activation' + str(idx))
        plt.ylabel('Kurtosis')
        plt.xlabel('Epochs')
        plt.grid(True)
        plt.legend(loc='lower left')

        plt.show()

    def draw_false_prediction(self, test_input, test_target, labels, num=5, figsize=(20, 5)):
        forward_final_output = self.feed_forward(input_data=test_input, is_numba=False)
        y = np.argmax(forward_final_output, axis=1)
        if test_target.ndim != 1:
            target = np.argmax(test_target, axis=1)
        else:
            target = test_target

        diff_index_list = []
        for i in range(len(test_input)):
            if y[i] != target[i]:
                diff_index_list.append(i)
        plt.figure(figsize=figsize)

        for i in range(num):
            j = diff_index_list[i]
            print("False Prediction Index: {:d}, Prediction: {:s}, Ground Truth: {:s}".format(j, labels[y[j]], labels[int(target[j])]))
            img = np.array(test_input[j])
            img.shape = (28, 28)
            plt.subplot(150 + (i + 1))
            plt.imshow(img, cmap='gray')

        plt.show()

    def get_param_describe(self, layer_num=0, kind="W"):
        assert layer_num <= self.hidden_layer_num

        if kind == "W":
            param_flatten_list = self.params['W' + str(layer_num)].value.flatten()
        else:
            param_flatten_list = self.params['b' + str(layer_num)].value.flatten()

        return stats.describe(np.array(param_flatten_list))

    def get_activation_describe(self, layer_num=0, kind="affine"):
        assert layer_num <= self.hidden_layer_num

        if kind == "affine":
            output_flatten_list = self.layers['affine' + str(layer_num)].output.flatten()
        else:
            output_flatten_list = self.layers['activation' + str(layer_num)].output.flatten()

        return stats.describe(np.array(output_flatten_list))
