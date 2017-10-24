import numpy as np

y = np.array([[0.0, 0.3, 0.7]])
y[y == 0] = 1e-15
t = np.array([[0, 1, 0]])

print(y.shape, t.shape)
w = np.log(y)
print(w)

e = t * w
print(e)
print(-np.sum(e))
# Axis or axes along which a sum is performed.  The default, axis=None, will sum all of the elements of the input array.
print(-np.sum(e, axis=0))
print(-np.sum(e, axis=1))
print(-np.sum(e, axis=(0, 1)))
print()

y = np.array([[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]])
print(y.shape)
print(np.sum(y, axis=0))
print(np.sum(y, axis=1))
print(np.sum(y, axis=(0, 1)))


