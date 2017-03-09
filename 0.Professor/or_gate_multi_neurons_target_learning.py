from __future__ import print_function
import numpy as np
import random
import math

class GateNeuron:
    def __init__(self):
        self.w = np.array([random.random(), random.random()])   # weight of one input
        self.b = np.array([random.random()])   # bias
        print("Initial w: {0}, b: {1}".format(self.w, self.b))

    def v(self, w, b, input):
        return np.dot(w, input) + b

    def activation(self, v):
        return max(np.array([0.0]), v)

    def y(self, input):
        v = self.v(self.w, self.b, input)
        return self.activation(v)

    def activation_derivative(self, v):
        if v >= 0:
            return 1
        else:
            return 0

    def numerical_activation_derivative(self, v):
        delta = 0.00000001
        return (self.activation(v + delta) - self.activation(v)) / delta

class MultiNeurons:
    def __init__(self):
        self.n1 = GateNeuron()
        self.n2 = GateNeuron()

    def v(self, w, b, input):
        return np.dot(w, input) + b

    def activation(self, v):
        return max(np.array([0.0]), v)

    def y(self, input):
        input1 = self.n1.y(input)
        input2 = self.n2.y(input)
        input = np.array([input1, input2])
        v = self.v(self.w, self.b, input)
        return self.activation(v)

    def squared_error(self, input, y_target):
        error = 1.0 / 2.0 * math.pow(self.y(input) - y_target, 2)
        return error

    def learning(self, alpha, maxEpoch, data):
        for i in xrange(maxEpoch):
            for idx in xrange(data.numTrainData):
                input = data.training_input_value[idx]
                y = self.y(input)
                y_target = data.training_y_target[idx]

                v = self.n1.v(self.n1.w, self.n1.b, input)
                error = self.n1.y(input) - y_target[0]

                self.n1.w = self.n1.w - alpha * error * self.n1.activation_derivative(v) * input
                self.n1.b = self.n1.b - alpha * error * self.n1.activation_derivative(v)

                v = self.n2.v(self.n2.w, self.n2.b, input)
                error = self.n2.y(input) - y_target[1]

                self.n2.w = self.n2.w - alpha * error * self.n2.activation_derivative(v) * input
                self.n2.b = self.n2.b - alpha * error * self.n2.activation_derivative(v)

            sum = 0.0
            for idx in xrange(data.numTrainData):
                sum = sum + self.squared_error(data.training_input_value[idx], data.training_y_target[idx])
            print("Epoch {0}: Error: {1}, w: {2}, b: {3}".format(i, sum / data.numTrainData, self.w, self.b))

class Data:
    def __init__(self):
        self.training_input_value = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.training_y_target = np.array([0.0, 1.0, 1.0, 1.0])
        self.numTrainData = len(self.training_input_value)

if __name__ == '__main__':
    mn = MultiNeurons()
    d = Data()
    for idx in xrange(d.numTrainData):
        input = d.training_input_value[idx]
        y = mn.y(input)
        y_target = d.training_y_target[idx]
        print("x: {0}, y: {1}, y_target: {2}, error: {3}".format(
            input,
            mn.y(input),
            y_target,
            mn.squared_error(input, y_target)))

    mn.learning(0.1, 1000, d)

    for idx in xrange(d.numTrainData):
        input = d.training_input_value[idx]
        y = mn.y(input)
        y_target = d.training_y_target[idx]
        print("x: {0}, y: {1}, y_target: {2}, error: {3}".format(
            input,
            mn.y(input),
            y_target,
            mn.squared_error(input, y_target)))