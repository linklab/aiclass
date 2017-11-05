import copy
import pickle

class A(object): pass

d = {}
for i in range(1000):
    d[i] = A()

def copy1():
    return copy.deepcopy(d)

def copy2():
    return pickle.loads(pickle.dumps(d, -1))

a = copy1()

b = copy2()

print(type(a), len(a), a )

print(type(b), len(b), b )

print(a is b)
print(a is d)