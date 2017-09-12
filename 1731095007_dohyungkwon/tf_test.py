from tensorflux import graph as tfg # graph 모듈을 객체화
from tensorflux import session as tfs
import networkx as nx
import matplotlib.pyplot as plt

g = tfg.Graph()
g.initialize()

#--------------
# Create variables
a = tfg.Variable(5.0, name="a")
b = tfg.Variable(1.0, name="b")

# Create placeholder
x = tfg.Placeholder(name="x")

# Create hidden node y
y = tfg.Mul(a, x, name="y")

# Create output node z
z = tfg.Add(y, b, name="z")

session = tfs.Session()
output = session.run(z, {x: 1.0}) # z:operation
print(output)
print()
print("-------------------")
#--------------
# Create variables
# Not scalar, list
A = tfg.Variable([[1, 0], [0, -1]], name="A")
b = tfg.Variable([1, 1], name="b")

# Create placeholder
x = tfg.Placeholder(name="x")

# It means A*x+b

# Create hidden node y
y = tfg.Matmul(A, x, name="y")

# Create output node z
z = tfg.Add(y, b, name="z")

session = tfs.Session()
output = session.run(z, {x: [1, 2]})
print(output)
print()
print("-------------------")
#--------------
#nx.draw_networkx(g, with_labels=True)
#plt.show(block=True)


W = tfg.Variable(5.0, name="W")
b = tfg.Variable(-1.0, name="b")

# Create placeholder
x = tfg.Placeholder(name="x")

# It means A*x+b

# Create hidden node y
y = tfg.Matmul(W, x, name="y")

# Create output node z
z = tfg.Add(y, b, name="z")

session = tfs.Session()
output = session.run(z, {x: 1.0})
print(output)
output = session.run(z, {x: 2.0})
print(output)
output = session.run(z, {x: 3.0})
print(output)
print()