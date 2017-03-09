from __future__ import print_function
import math

class Neuron:
    def __init__(self, init_w = 0.0, init_b = 0.0):
        self.w = init_w   # weight of one input
        self.b = init_b   # bias

    def v(self, w, b, input):
        return w * input + b

    def activation(self, v):
        return max(0.0, v)

    def feedforward(self, input):
        v = self.v(self.w, self.b, input)
        return self.activation(v)

    def squared_error(self, input, y_target):
        return 1.0 / 2.0 * math.pow(self.feedforward(input) - y_target, 2)

class Data:
    def __init__(self):
        self.training_input_value = [1.0, 2.0, 3.0]
        self.training_y_target = [6.0, 7.0, 8.0]
        self.numTrainData = len(self.training_input_value)

if __name__ == '__main__':
    n = Neuron(5.0, -1.0)
    d = Data()
    for idx in xrange(d.numTrainData):
        input = d.training_input_value[idx]
        y = n.feedforward(input)
        y_target = d.training_y_target[idx]
        print("x: {0}, y: {1}, y_target: {2}, error: {3}".format(
            input,
            n.feedforward(input),
            y_target,
            n.squared_error(input, y_target)))
