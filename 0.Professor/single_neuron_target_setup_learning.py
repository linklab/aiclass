from __future__ import print_function
import math

class Data:
    def __init__(self):
        self.training_input_value = 1.0
        self.training_y_target = 10.0


class Neuron:
    def __init__(self, init_w = 0.0, init_b = 0.0):
        self.w = init_w   # weight of one input
        self.b = init_b   # bias

    def relu_activation(self, v):
        return max(0.0, v)

    def feedforward(self, input):
        v = self.w * input + self.b
        return self.relu_activation(v)

    def squared_error(self, input, y_target):
        return 1.0 / 2.0 * math.pow(self.feedforward(input) - y_target, 2)

    def learning(self, alpha, maxNumIter):
        data = Data()
        for i in xrange(maxNumIter):
            self.squared_error(data.training_input_value, data.training_y_target)

if __name__ == '__main__':
    my_neuron = Neuron(5.0, -1.0)
    print(my_neuron.feedforward(1.0))
    print(my_neuron.feedforward(2.0))
    print(my_neuron.feedforward(3.0))