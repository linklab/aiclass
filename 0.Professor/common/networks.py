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

