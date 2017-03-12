from __future__ import print_function
import math

class Neuron:
    def __init__(self, init_w = 0.0, init_b = 0.0):
        self.w = init_w   # weight of one input
        self.b = init_b   # bias
        print("Initial w: {0}, b: {1}".format(self.w, self.b))

    def u(self, input):
        return self.w * input + self.b

    def f(self, u):
        return max(0.0, u)

    def z(self, input):
        u = self.u(input)
        return self.f(u)

    def squared_error(self, input, z_target):
        return 1.0 / 2.0 * math.pow(self.z(input) - z_target, 2)

class Data:
    def __init__(self):
        self.training_input_value = [1.0, 2.0, 3.0]
        self.training_z_target = [6.0, 7.0, 8.0]
        self.numTrainData = len(self.training_input_value)

if __name__ == '__main__':
    n = Neuron(5.0, -1.0)
    d = Data()
    for idx in xrange(d.numTrainData):
        input = d.training_input_value[idx]
        z = n.z(input)
        z_target = d.training_z_target[idx]
        error = n.squared_error(input, z_target)
        print("x: {0}, z: {1}, z_target: {2}, error: {3}".format(input, z, z_target, error))
