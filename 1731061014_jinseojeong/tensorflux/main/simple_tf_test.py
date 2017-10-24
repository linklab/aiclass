# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import networkx as nx
import matplotlib.pyplot as plt
import tensorflux.graph as tfg
import tensorflux.session as tfs
import tensorflux.functions as tff


# Create a new graph
g = tfg.Graph()
g.initialize()

# Create variables
a = tfg.Variable(5.0, name="A")
b = tfg.Variable(1.0, name="b")

# Create placeholder
x = tfg.Placeholder(name="x")

# Create hidden node y
y = tfg.Mul(a, x, name="y")

# Create output node z
z = tfg.Add(y, b, name="z")

# nx.draw_networkx(g, with_labels=True)
# plt.show(block=True)

session = tfs.Session()
output = session.run(z, feed_dict={x: 1.0})
print(output)


# 20170912
# initialize a new graph
g.initialize()

# Create variables
A = tfg.Variable([[1, 0], [0, -1]], name="A")
b = tfg.Variable([1, 1], name="b")

# Create placeholder
x = tfg.Placeholder(name="x")

# Create hidden node y
y = tfg.Matmul(A, x, name="y")

# Create output node z
z = tfg.Add(y, b, name="z")

session = tfs.Session()
output = session.run(z, feed_dict={x: [1, 2]})
print(output)


# 20170912
# initialize a new graph
g.initialize()

# Create variables
w = tfg.Variable(5.0, name="w")
b = tfg.Variable(-1.0, name="b")

# Create placeholder
x = tfg.Placeholder(name="x")

# Create hidden node y
y = tfg.Affine(w, x, b, name="y")

session = tfs.Session()

output = session.run(y, feed_dict={x: 1.0})
squared_error = tff.squared_error(output, 14.0)
print("Output: {:4.1f}, Squared_Error: {:5.1f}".format(output, squared_error))

output = session.run(y, feed_dict={x: 2.0})
squared_error = tff.squared_error(output, 24.0)
print("Output: {:4.1f}, Squared_Error: {:5.1f}".format(output, squared_error))

output = session.run(y, feed_dict={x: 3.0})
squared_error = tff.squared_error(output, 34.0)
print("Output: {:4.1f}, Squared_Error: {:5.1f}".format(output, squared_error))