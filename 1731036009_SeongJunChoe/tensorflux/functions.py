import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# 자세한 내용은 각자 확인
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
    # ovA = np.ndarray(shape=(3,), dtype=float, order='F')
    # cnt = 0
    # for n in output_value:
    #     ovA[cnt] = 0.5 * math.pow(n-target_value, 2)
    #     # print(ovA[cnt])
    #     cnt = cnt + 1
    #
    # print(np.argmax(ovA))
    # return np.argmax(ovA)
    return 0.5 * math.pow(output_value - target_value, 2)

# functions 파일을 테스트함
if __name__ == "__main__":
    # sd : standard nevigation => 값을 크게 하면 분포가 커짐
    x1 = get_truncated_normal(shape=(1, 10000), mean=2, sd=5, low=1, upp=10)
    x2 = get_truncated_normal(shape=(1, 10000), mean=5.5, sd=5, low=1, upp=10)
    x3 = get_truncated_normal(shape=(1, 10000), mean=8, sd=5, low=1, upp=10)

    fig, ax = plt.subplots(3, sharex=True)
    # 히스토그램을 그림
    ax[0].hist(x1.flatten())    # flatten : 데이터를 풀어주는 것!!! 히스토그램을 그리지 위해서는 중요
    ax[1].hist(x2.flatten())
    ax[2].hist(x3.flatten())
    plt.show()