# [Note] http://numba.pydata.org/
# [Note] http://numba.pydata.org/numba-doc/0.35.0/index.html
# [Note] conda update numba
# [Note] conda install cudatoolkit
# [Note] conda install cudatoolkit
# nvprof python vector_add_test.py


import numba
import numpy as np
from numba import jit, float32, int32, void
from numba import guvectorize, vectorize, cuda
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


@jit(float32(float32[:,:]))
def jit_sum_2(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

#@vectorize([float32(float32[:,:])], target='cuda')
@cuda.jit(void(float32[:,:]))
def cuda_sum(arr):
    #M, N = arr.shape
    M = 3000
    N = 3000
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]

def sum(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result


a = np.arange(900000000, dtype=np.float32).reshape(30000, 30000)
print(a.shape)

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
result = cuda_sum(a)
e = timer()
print("CUDA_SUM: {:7.6f} ms".format((e - s) * 1000))
print(result)

print()

s = timer()
result = sum(a)
e = timer()
print("NORMAL_SUM: {:7.6f} ms".format((e - s) * 1000))
print(result)



