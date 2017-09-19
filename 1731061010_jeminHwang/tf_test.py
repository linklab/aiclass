from tensorflux import graph as tfg
from tensorflux import session as tfs
import networkx as nx
import matplotlib.pyplot as plt

# g= graph.Graph()
g= tfg.Graph()
g.initialize()

# Create variables
a = tfg.Variable(5.0, name='a')
b = tfg.Variable(1.0, name='b')

# Create placeholder
x = tfg.Placeholder(name='x')

# Create hidden node y
y = tfg.Mul(a, x, name="y")

# Create output node z
z = tfg.Add(y, b, name="z")

#nx.draw_networkx(g, with_labels=True)
#plt.show(block=True)

session = tfs.Session()
output = session.run(z, {x: 1.0})
print(output)

print(z.input_nodes[0],z.input_nodes[1])
print(z.output)
print(z.consumers)
print(y.consumers[0])
print(x.consumers[0])
print(a.consumers[0])
print(a.consumers)
# print(a.consumers[1]) 여러번 사용될경우 생김

print('***********************************')

# Create a new graph
g = tfg.Graph()
g.initialize()

# Create variables
A = tfg.Variable([[1, 0], [0, -1]], name="A")       ##A=[1  0] b=[1] x=[1] y.output=[ 1]
b = tfg.Variable([1, 1], name="b")                  ##  [0 -1]   [1]   [2]          [-2]

# Create placeholder
x = tfg.Placeholder(name="x")

# Create hidden node y
y = tfg.Matmul(A, x, name="y")

# Create output node z
z = tfg.Add(y, b, name="z")

#nx.draw_networkx(g, with_labels=True)
#plt.show(block=True)

session = tfs.Session()
output = session.run(z, {x: [1, 2]})
print(output)
print(y.output)

print('***********************************')

# g= graph.Graph()
g= tfg.Graph()
g.initialize()

# Create variables
w = tfg.Variable(5.0, name='w')
b = tfg.Variable(1.0, name='b')

# Create placeholder
x = tfg.Placeholder(name='x')

# Create hidden node y
y = tfg.Mul(w, x, name="w")

# Create output node z
z = tfg.Add(y, b, name="z")


session = tfs.Session()
output = session.run(z, {x: 1.0})
print(output)

session = tfs.Session()
output = session.run(z, {x: 2.0})
print(output)

session = tfs.Session()
output = session.run(z, {x: 3.0})
print(output)