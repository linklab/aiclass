from __future__ import print_function

class Neuron:
    def __init__(self, w = 0.0, b = 0.0):
        self.w = w   # weight of one input
        self.b = b   # bias

    def activation(self, v):
        return max(0.0, v)

    def feedforward(self, input):
        v = self.w * input + self.b
        return self.activation(v)


if __name__ == '__main__':
    my_neuron = Neuron(2.0, 1.0)
    print(my_neuron.feedforward(0.6))
    print(my_neuron.feedforward(0.8))
    print(my_neuron.feedforward(1.0))