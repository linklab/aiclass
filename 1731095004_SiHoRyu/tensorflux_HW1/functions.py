import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


def get_truncated_normal(shape, mean=0, sd=1, low=0, upp=10):
    x = truncnorm(a=(low - mean) / sd, b=(upp - mean) / sd, loc=mean, scale=sd)
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    x = x.rvs(num_elements)
    x = x.reshape(shape)
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def squared_error(output_value, target_value):
    return 0.5 * math.pow(output_value - target_value, 2)


if __name__ == "__main__":
    #10000개의 데이터, 평균 2, 표준편차 1, low 값 이하는 아니고 upp 값 이상이 아닌 값을 지니게
    x1 = get_truncated_normal(shape=(1, 10000), mean=2, sd=1, low=1, upp=10)
    x2 = get_truncated_normal(shape=(1, 10000), mean=5.5, sd=1, low=1, upp=10)
    x3 = get_truncated_normal(shape=(1, 10000), mean=8, sd=1, low=1, upp=10)

    fig, ax = plt.subplots(3, sharex=True)
    ax[0].hist(x1.flatten())
    ax[1].hist(x2.flatten())
    ax[2].hist(x3.flatten())
    plt.show()