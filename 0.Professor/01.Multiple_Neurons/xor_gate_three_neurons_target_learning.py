from __future__ import print_function
import numpy as np
import random
import math

class Neuron1:
    def __init__(self):
        self.w1 = np.array([random.random(), random.random()])   # weight of one input
        self.b1 = random.random()   # bias
        print("Neuron1 - Initial w1: {0}, b1: {1}".format(self.w1, self.b1))

    def u1(self, input):
        return np.dot(self.w1, input) + self.b1

    def f(self, u1):
        return max(0.0, u1)

    def z1(self, input):
        u1 = self.u1(input)
        return self.f(u1)

class Neuron2:
    def __init__(self):
        self.w2 = np.array([random.random(), random.random()])   # weight of one input
        self.b2 = random.random()   # bias
        print("Neuron2 - Initial w2: {0}, b2: {1}".format(self.w2, self.b2))

    def u2(self, input):
        return np.dot(self.w2, input) + self.b2

    def f(self, u2):
        return max(0.0, u2)

    def z2(self, input):
        u2 = self.u2(input)
        return self.f(u2)

class Neuron3:
    def __init__(self, n1, n2):
        self.w3 = np.array([random.random(), random.random()])   # weight of one input
        self.b3 = random.random()   # bias
        self.n1 = n1
        self.n2 = n2
        print("Neuron2 - Initial w3: {0}, b3: {1}".format(self.w3, self.b3))

    def u3(self, input):
        z1 = self.n1.z1(input)
        z2 = self.n2.z2(input)
        z = np.array([z1, z2])
        return np.dot(self.w3, z) + self.b3

    def f(self, u3):
        return max(0.0, u3)

    def z3(self, input):
        u3 = self.u3(input)
        return self.f(u3)

    def squared_error(self, input, z_target):
        return 1.0 / 2.0 * math.pow(self.z3(input) - z_target, 2)

    def d_E_over_d_w3(self, input, z_target):
        u3 = self.u3(input)
        if u3 >= 0.0:
            z3 = self.z3(input)
            z1 = self.n1.z1(input)
            z2 = self.n2.z2(input)
            z = np.array([z1, z2])
            return (z3 - z_target) * z
        else:
            return 0.0

    def d_E_over_d_b3(self, input, z_target):
        u3 = self.u3(input)
        if u3 >= 0.0:
            z3 = self.z3(input)
            return z3 - z_target
        else:
            return 0.0

    def d_E_over_d_w2(self, input, z_target):
        u3 = self.u3(input)
        u2 = self.n2.u2(input)
        if u3 >= 0.0 and u2 >= 0.0:
            return (self.f(u3) - z_target) * self.w3[1] * input
        else:
            return 0.0

    def d_E_over_d_b2(self, input, z_target):
        u3 = self.u3(input)
        u2 = self.n2.u2(input)
        if u3 >= 0.0 and u2 >= 0.0:
            return (self.f(u3) - z_target) * self.w3[1]
        else:
            return 0.0

    def d_E_over_d_w1(self, input, z_target):
        u3 = self.u3(input)
        u1 = self.n1.u1(input)
        if u3 >= 0.0 and u1 >= 0.0:
            return (self.f(u3) - z_target) * self.w3[0] * input
        else:
            return 0.0

    def d_E_over_d_b1(self, input, z_target):
        u3 = self.u3(input)
        u1 = self.n1.u1(input)
        if u3 >= 0.0 and u1 >= 0.0:
            return (self.f(u3) - z_target) * self.w3[0]
        else:
            return 0.0

    def learning(self, alpha, maxEpoch, data):
        for i in xrange(maxEpoch):
            for idx in xrange(data.numTrainData):
                input = data.training_input_value[idx]
                z_target = data.training_z_target[idx]

                self.n1.w1 = self.n1.w1 - alpha * self.d_E_over_d_w1(input, z_target)
                self.n1.b1 = self.n1.b1 - alpha * self.d_E_over_d_b1(input, z_target)
                self.n2.w2 = self.n2.w2 - alpha * self.d_E_over_d_w2(input, z_target)
                self.n2.b2 = self.n2.b2 - alpha * self.d_E_over_d_b2(input, z_target)
                self.w3 = self.w3 - alpha * self.d_E_over_d_w3(input, z_target)
                self.b3 = self.b3 - alpha * self.d_E_over_d_b3(input, z_target)

            sum = 0.0
            for idx in xrange(data.numTrainData):
                sum = sum + self.squared_error(data.training_input_value[idx], data.training_z_target[idx])
            print("Epoch {0}: Error: {1}, w1: {2}, b1: {3}, w2: {4}, b2: {5}, w3: {6}, b3: {7}".format(i, sum / data.numTrainData, self.n1.w1, self.n1.b1, self.n2.w2, self.n2.b2, self.w3, self.b3))

class Data:
    def __init__(self):
        self.training_input_value = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.training_z_target = np.array([0.0, 1.0, 1.0, 0.0])
        self.numTrainData = len(self.training_input_value)

if __name__ == '__main__':
    n1 = Neuron1()
    n2 = Neuron2()
    n3 = Neuron3(n1, n2)
    d = Data()
    for idx in xrange(d.numTrainData):
        input = d.training_input_value[idx]
        z3 = n3.z3(input)
        z_target = d.training_z_target[idx]
        error = n3.squared_error(input, z_target)
        print("x: {0}, z3: {1}, z_target: {2}, error: {3}".format(input, z3, z_target, error))

    n3.learning(0.05, 2000, d)

    for idx in xrange(d.numTrainData):
        input = d.training_input_value[idx]
        z3 = n3.z3(input)
        z_target = d.training_z_target[idx]
        error = n3.squared_error(input, z_target)
        print("x: {0}, z3: {1}, z_target: {2}, error: {3}".format(input, z3, z_target, error))