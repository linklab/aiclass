from __future__ import print_function

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

if __name__ == '__main__':
    n = Neuron(5.0, -1.0)
    print(n.z(1.0))
    print(n.z(2.0))
    print(n.z(3.0))