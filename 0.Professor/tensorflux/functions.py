import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from numba import jit, float64, uint8, void, cuda


def get_truncated_normal(shape, mean=0, sd=1, low=0, upp=10):
    x = truncnorm(a=(low - mean) / sd, b=(upp - mean) / sd, loc=mean, scale=sd)
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    x = x.rvs(num_elements)
    x = x.reshape(shape)
    return x


def sigmoid(x, is_numba):
    if is_numba:
        return _sigmoid(x)
    else:
        return 1 / (1 + np.exp(-x))


@jit(nopython=True)
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def squared_error(output_value, target_value, is_numba):
    if is_numba:
        return _squared_error(output_value, target_value)
    else:
        return 0.5 * np.power(output_value - target_value, 2.0)


@jit(nopython=True)
def _squared_error(output_value, target_value):
    return 0.5 * np.power(output_value - target_value, 2.0)


def softmax(x, is_numba):
    if is_numba:
        return _softmax(x)
    else:
        if x.ndim == 2:
            x = x.T
            x = x - x.max()
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - x.max()
        return np.exp(x) / np.sum(np.exp(x))


@jit(nopython=True)
def _softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - x.max()
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - x.max()
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t, is_numba):
    if y.ndim == 1 and t.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    y[y == 0] = 1e-15
    batch_size = y.shape[0]

    if is_numba:
        return _cross_entropy_error(y, t, batch_size)
    else:
        return -np.sum(t * np.log(y)) / batch_size


@jit(nopython=True)
def _cross_entropy_error(y, t, batch_size):
    return -np.sum(t * np.log(y)) / batch_size


def accuracy(forward_final_output, target):
    y = np.argmax(forward_final_output, axis=1)
    if target.ndim != 1:
        target = np.argmax(target, axis=1)

    accuracy = np.sum(y == target) / float(forward_final_output.shape[0])
    return accuracy

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : 이미지 데이터
    filter_h : 필터 높이
    filter_w : 필터 폭
    stride : 스트라이드
    pad : 패드

    Returns
    -------
    col : 2차원행렬
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 이미지 데이터 Shape（例：(10, 1, 28, 28)）
    filter_h
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


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

    # a = np.array([1.0, 2.0, 3.0])
    # print(sigmoid(a, is_numba=True))
    # print()
    #
    # b = np.array([1.0, 0.0, 1.0])
    # c = np.array([0.0, 1.0, 0.0])
    # print(squared_error(b, c, is_numba=True))
    # print()
    #
    # print(cross_entropy_error(b, c, is_numba=True))
    # print()
    #
    # d = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    # print(softmax(d, is_numba=True))
    # print()
    #
    #
    #
    # q = np.array([[3.3, 1.2, 9.4], [7.1, 2.2, 3.3], [1.9, 9.2, 2.3]])
    # t = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # print(accuracy(q, t, is_numba=True))
    #
    # q = np.array([[3.3, 1.2, 9.4], [7.1, 2.2, 3.3], [1.9, 9.2, 2.3]])
    # t = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # print(accuracy(q, t, is_numba=True))

    x = get_truncated_normal(shape=(100,), mean=0.0, sd=1.0, low=-1.0, upp=1.0)
    print(x)