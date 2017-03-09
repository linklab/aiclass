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

    def squared_error(self, input, y_target):
        return 1.0 / 2.0 * math.pow(self.y(input) - y_target, 2)

class Data:
    def __init__(self):
        self.training_input_value = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.training_y_target = np.array([0.0, 0.0, 0.0, 1.0])
        self.numTrainData = len(self.training_input_value)

if __name__ == '__main__':
    n = GateNeuron()
    d = Data()
    for idx in xrange(d.numTrainData):
        input = d.training_input_value[idx]
        y = n.y(input)
        y_target = d.training_y_target[idx]
        print("x: {0}, y: {1}, y_target: {2}, error: {3}".format(
            input,
            n.y(input),
            y_target,
            n.squared_error(input, y_target)))