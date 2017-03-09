from __future__ import print_function

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

if __name__ == '__main__':
    n = Neuron(5.0, -1.0)
    print(n.feedforward(1.0))
    print(n.feedforward(2.0))
    print(n.feedforward(3.0))