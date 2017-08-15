from collections import OrderedDict
from common.layers import *
from common.optimizers import *
from common.initializers import *


activation_layers = {
    'Sigmoid': Sigmoid,
    'ReLU': ReLU
}

optimizers = {
    "SGD": SGD,
    "Momentum": Momentum,
    "Nesterov": Nesterov,
    "AdaGrad": AdaGrad,
    "RMSprop": RMSprop,
    "Adam": Adam
}

initializers = {
    'Zero': Zero_Initializer,
    'N1': N1_Initializer,
    'N2': N2_Initializer,
    'Xavier': Xavier_Initializer,
    'He': He_Initializer
}


class MultiLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size, activation='ReLU', initializer='He',
                 optimizer='AdaGrad', learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)

        # Weight Initialization
        self.params = {}
        self.weight_initialization(initializer)

        # Layering
        self.layers = OrderedDict()
        self.last_layer = None
        self.layering(activation)

        # Optimizer Initialization
        self.optimizer = optimizers[optimizer](lr=learning_rate)

    def weight_initialization(self, initializer):
        params_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        initializer_obj = initializers[initializer](self.params, params_size_list)
        initializer_obj.initialize_params();

    def layering(self, activation):
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.layers['Activation' + str(idx)] = activation_layers[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithCrossEntropyLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def backpropagation_gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        din = 1
        din = self.last_layer.backward(din)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            din = layer.backward(din)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads

    def learning(self, x_batch, t_batch):
        grads = self.backpropagation_gradient(x_batch, t_batch)
        self.optimizer.update(self.params, grads)


class MultiLayerNetExtended:
    def __init__(self, input_size, hidden_size_list, output_size, activation='ReLU', initializer='N2',
                 optimizer='AdaGrad', learning_rate=0.01,
                 use_batch_normalization=False,
                 use_weight_decay=False, weight_decay_lambda=0.0,
                 use_dropout=False, dropout_ratio_list=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)

        self.use_batch_normalization = use_batch_normalization

        self.use_weight_decay = use_weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        self.use_dropout = use_dropout
        self.dropout_ratio_list = dropout_ratio_list

        # Weight Initialization
        self.params = {}
        self.weight_initialization(initializer)

        # Layering
        self.layers = OrderedDict()
        self.last_layer = None
        self.layering(activation)

        # Optimization Method
        self.optimizer = optimizers[optimizer](lr=learning_rate)

    def weight_initialization(self, initializer):
        params_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        initializer_obj = initializers[initializer](self.params,
                                                    params_size_list,
                                                    self.use_batch_normalization)
        initializer_obj.initialize_params();

    def layering(self, activation):
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            if self.use_batch_normalization:
                self.layers['Batch_Normalization' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)],
                                                                                   self.params['beta' + str(idx)])
            self.layers['Activation' + str(idx)] = activation_layers[activation]()

            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(self.dropout_ratio_list[idx - 1])

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithCrossEntropyLoss()

    def predict(self, x, is_train=False):
        for key, layer in self.layers.items():
            if "BatchNorm" in key or "Dropout" in key:
                x = layer.forward(x, is_train)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, is_train=False):
        y = self.predict(x, is_train)

        if self.use_weight_decay:
            weight_decay = 0.0
            for idx in range(1, self.hidden_layer_num + 2):
                W = self.params['W' + str(idx)]
                weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
            return self.last_layer.forward(y, t) + weight_decay
        else:
            return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x, is_train=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def backpropagation_gradient(self, x, t):
        # forward
        self.loss(x, t, is_train=True)

        # backward
        din = 1
        din = self.last_layer.backward(din)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            din = layer.backward(din)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            if self.use_weight_decay:
                grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)]
            else:
                grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batch_normalization and idx <= self.hidden_layer_num:
                grads['gamma' + str(idx)] = self.layers['Batch_Normalization' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['Batch_Normalization' + str(idx)].dbeta

        return grads

    def learning(self, x_batch, t_batch):
        grads = self.backpropagation_gradient(x_batch, t_batch)
        self.optimizer.update(self.params, grads)