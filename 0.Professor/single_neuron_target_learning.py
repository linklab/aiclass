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

    def activation_derivative(self, v):
        if v >= 0:
            return 1
        else:
            return 0

    def numerical_activation_derivative(self, v):
        delta = 0.00000001
        return (self.activation(v + delta) - self.activation(v)) / delta

    def learning(self, alpha, maxEpoch, data):
        print("Before learning, w: {0}, b: {1}".format(self.w, self.b))
        for i in xrange(maxEpoch):
            for idx in xrange(data.numTrainData):
                input = data.training_input_value[idx]
                y = n.feedforward(input)
                y_target = data.training_y_target[idx]

                v = self.v(self.w, self.b, input)
                error = self.feedforward(input) - y_target

                self.w = self.w - alpha * error * self.activation_derivative(v) * input
                self.b = self.b - alpha * error * self.activation_derivative(v)

                #self.w = self.w - alpha * error * self.numerical_activation_derivative(v) * input
                #self.b = self.b - alpha * error * self.numerical_activation_derivative(v)

            sum = 0.0
            for idx in xrange(data.numTrainData):
                sum = sum + self.squared_error(data.training_input_value[idx], data.training_y_target[idx])
            print("Epoch {0}: Error: {1}, w: {2}, b: {3}".format(i, sum / data.numTrainData, self.w, self.b))

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

    n.learning(0.1, 100, d)

    for idx in xrange(d.numTrainData):
        input = d.training_input_value[idx]
        y = n.feedforward(input)
        y_target = d.training_y_target[idx]
        print("x: {0}, y: {1}, y_target: {2}, error: {3}".format(
            input,
            n.feedforward(input),
            y_target,
            n.squared_error(input, y_target)))
