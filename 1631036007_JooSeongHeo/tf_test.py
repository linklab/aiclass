from tensorflux import graph as tfg
from tensorflux import session as tfs
import matplotlib.pyplot as plt
import networkx as nx
g = tfg.Graph()
g.initialize()

#
# a = tfg.Variable(5.0, name="a")
# b = tfg.Variable(1.0, name="b")
#
# # Create placeholder
# x = tfg.Placeholder(name="x")
#
#
# # Create hidden node y
# y = tfg.Mul(a, x, name="y")
#
# # Create output node z
# z = tfg.Add(y, b, name="z")
#
# session = tfs.Session()
# output = session.run(z, {x: 2.0})
# print(output)


# Create a new graph
g = tfg.Graph()
g.initialize()

# Create variables
A = tfg.Variable([[1, 0], [0, -1]], name="A")
b = tfg.Variable([1, 1], name="b")

# Create placeholder
x = tfg.Placeholder(name="x")

# # Create hidden node y
# y = tfg.Matmul(A, x, name="y")
#
# # Create output node z
# z = tfg.Add(y, b, name="z")

#nx.draw_networkx(g, with_labels=True)
#plt.show(block=True)

u = Affine(w,x,b)

session = tfs.Session()
output = session.run(z, {x: [1, 2]})
print(output)

nx.draw_networkx(g, with_labels=True)
plt.show(block=True)