import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - x ** 2


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=1):
        # Add column of ones to X
        # This is to add the bias unit to the input layer

        self.inputSize = 2
        self.hiddenSize = 2
        self.outputSize = 1

        self.inputBias = True
        self.hiddenBias = True
        self.w1=np.random.randn(self.inputSize+int(self.inputBias), self.hiddenSize)
        self.w2=np.random.randn(self.hiddenSize + int(self.hiddenBias), self.outputSize)
        print("w1: ", self.w1)
        print("으하: ", self.w2)

        print("daf",self.w1.ravel())
        print("dmdm", self.w2.ravel())
        X1=np.array([[0,0],[0,1],[1,0],[1,1]])
        X1 = np.hstack((X,np.ones((X.shape[0],1))))
        print("X1: ",X1)
        print("w1: ", self.w1)
        z2=np.dot(X1, self.w1)
        print("z2: ",z2)

        ones = np.atleast_2d(np.ones(X.shape[0]))
        print("before",(np.ones(X.shape[0])))
        print("ones", ones)
        print("X", X)
        X = np.concatenate((ones.T, X), axis=1)

        print("after",X)
        print("X.shape[0]",X.shape[0])
        print("X.shape[1]", X.shape[1])

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            print(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
            q = (np.random.randint(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],[10,11,12]]).shape[0]))
            print("q", q)
            print("X0", X[0])
            print("X1", X[1])
            print("X2", X[2])
            print("X3", X[3])
            a = [X[i]]
            print(a)
            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l])
                print("l",l)
                print("a[l]",a[l])
                print("weight[l]", self.weights[l])
                print("dotvalue",dot_value)
                activation = self.activation(dot_value)
                print("act",activation)
                a.append(activation)
                print("a",a)
            # output layer
            print("y[i]",y[i])
            print("a[-1]",a[-1])
            error = y[i] - a[-1]
            print("error",error)
            deltas = [error * self.activation_prime(a[-1])]
            print("act_prime",self.activation_prime(a[-1]))
            print("deltas",deltas)
            # we need to begin at the second to last layer
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0:
                print('epochs:', k)
                shape = (2,1)
                print(shape[0])

    def predict(self, x):
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


if __name__ == '__main__':

    nn = NeuralNetwork([2, 2, 1])
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 1, 0])
    nn.fit(X, y)
    for e in X:
        print(e, nn.predict(e))