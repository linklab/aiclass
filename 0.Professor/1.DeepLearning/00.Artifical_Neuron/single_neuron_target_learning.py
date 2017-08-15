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

    def f_derivative(self, u):
        if u >= 0:
            return 1
        else:
            return 0

    # def numerical_f_derivative(self, u):
    #     delta = 0.00000001
    #     return (self.f(u + delta) - self.f(u)) / delta

    def d_E_over_d_w(self, input, z_target):
        u = self.u(input)
        z = self.f(u)
        error = z - z_target
        return error * self.f_derivative(u) * input

    def d_E_over_d_b(self, input, z_target):
        u = self.u(input)
        z = self.f(u)
        error = z - z_target
        return error * self.f_derivative(u)

    def learning(self, alpha, maxEpoch, data):
        for i in xrange(maxEpoch):
            for idx in xrange(data.numTrainData):
                input = data.training_input_value[idx]
                z_target = data.training_z_target[idx]

                self.w = self.w - alpha * self.d_E_over_d_w(input, z_target)
                self.b = self.b - alpha * self.d_E_over_d_b(input, z_target)

            sum = 0.0
            for idx in xrange(data.numTrainData):
                sum = sum + self.squared_error(data.training_input_value[idx], data.training_z_target[idx])
            print("Epoch {0}: Error: {1}, w: {2}, b: {3}".format(i, sum / data.numTrainData, self.w, self.b))

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

    n.learning(0.1, 100, d)

    for idx in xrange(d.numTrainData):
        input = d.training_input_value[idx]
        z = n.z(input)
        z_target = d.training_z_target[idx]
        error = n.squared_error(input, z_target)
        print("x: {0}, z: {1}, z_target: {2}, error: {3}".format(input, z, z_target, error))
