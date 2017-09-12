from tensorflux import graph as tfg
from tensorflux import session as tfs
import networkx as nx
import matplotlib.pyplot as plt

g = tfg.Graph()
g.initialize()

# Create variables
w = tfg.Variable(5.0, name="a")
b = tfg.Variable(-1.0, name="b")

# Create placeholder
x = tfg.Placeholder(name="x")

## Create hidden node y
#y = tfg.Matmul(A, x, name="y")
#
## Create output node z
#z = tfg.Add(y, b, name="z")

#nx.draw_networkx(g, with_labels=True)
#plt.show(block=True)

session = tfs.Session()
output = session.run(z, {x: 1.0})
output = session.run(z, {x: 2.0})
print(output)