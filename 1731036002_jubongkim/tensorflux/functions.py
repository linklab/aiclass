import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import numba
from numba import jit, float32, int32, void, cuda

@jit
def get_truncated_normal(shape, mean=0, sd=1, low=0, upp=10):
    x = truncnorm(a=(low - mean) / sd, b=(upp - mean) / sd, loc=mean, scale=sd)
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    x = x.rvs(num_elements)
    x = x.reshape(shape)
    return x

@jit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@jit
def squared_error(output_value, target_value):
    return 0.5 * math.pow(output_value - target_value, 2)

@jit
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

@jit
def cross_entropy_error(y, t):
    y[y == 0] = 1e-15
    if y.ndim == 1 and t.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size

@jit
def accuracy(forward_final_output, target):
    y = np.argmax(forward_final_output, axis=1)
    if target.ndim != 1:
        target = np.argmax(target, axis=1)

    accuracy = np.sum(y == target) / float(forward_final_output.shape[0])
    return accuracy

if __name__ == "__main__":
    # x1 = get_truncated_normal(shape=(1, 10000), mean=2, sd=1, low=1, upp=10)
    # x2 = get_truncated_normal(shape=(1, 10000), mean=5.5, sd=1, low=1, upp=10)
    # x3 = get_truncated_normal(shape=(1, 10000), mean=8, sd=1, low=1, upp=10)
    #
    # fig, ax = plt.subplots(3, sharex=True)
    # ax[0].hist(x1.flatten())
    # ax[1].hist(x2.flatten())
    # ax[2].hist(x3.flatten())
    # plt.show()

    q = np.array([[3.3, 1.2, 9.4], [7.1, 2.2, 3.3], [1.9, 9.2, 2.3]])
    t = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    print(accuracy(q, t))

    q = np.array([[3.3, 1.2, 9.4], [7.1, 2.2, 3.3], [1.9, 9.2, 2.3]])
    t = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    print(accuracy(q, t))