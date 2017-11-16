# [Note] http://numba.pydata.org/
# [Note] http://numba.pydata.org/numba-doc/0.35.0/index.html
# [Note] conda update numba
# [Note] conda install cudatoolkit


import numba
from numba import jit, float32, int32, void, cuda
from numpy import arange
from timeit import default_timer as timer

print("Numba Version", numba.__version__)

# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.
@jit
def jit_sum_1(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result


@jit(float32(int32[:]))
def jit_sum_2(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result


def sum(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result


@cuda.jit
def cuda_jit_sum(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result


a = arange(9000000).reshape(3000, 3000)
print(a)

print()

s = timer()
result = jit_sum_1(a)
e = timer()
print("JIT_SUM_1: {:7.6f} ms".format((e - s) * 1000))
print(result)

print()

s = timer()
result = jit_sum_2(a)
e = timer()
print("JIT_SUM_2: {:7.6f} ms".format((e - s) * 1000))
print(result)

print()

s = timer()
result = sum(a)
e = timer()
print("NORMAL_SUM: {:7.6f} ms".format((e - s) * 1000))
print(result)

print()

s = timer()
result = cuda_jit_sum(a)
e = timer()
print("{:7.6f} ms".format((e - s) * 1000))
print(result)