import numba
import numpy as np
from numba import jit, float32, int32, void
from numba import guvectorize, vectorize, cuda
from timeit import default_timer as timer
print("Numba Version", numba.__version__)

def normal_vectoradd(a, b):
    return a + b

@vectorize(["float32(float32, float32)"], target='cuda')
def cuda_vectoradd(a, b):
    return a + b

N = 300000000

A = np.ones(N, dtype=np.float32)
B = np.ones(N, dtype=np.float32)

start = timer()
C = normal_vectoradd(A, B)
elapsed_time = timer() - start

print(elapsed_time, "seconds")
print(C[:5])
print(C[-5:])

print()

start = timer()
C = cuda_vectoradd(A, B)
elapsed_time = timer() - start

print(elapsed_time, "seconds")
print(C[:5])
print(C[-5:])

