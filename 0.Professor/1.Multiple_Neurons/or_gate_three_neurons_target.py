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

class Data:
    def __init__(self):
        self.training_input_value = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
        self.training_z_target = np.array([0.0, 1.0, 1.0, 1.0])
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