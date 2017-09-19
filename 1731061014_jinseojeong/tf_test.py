
import matplotlib.pyplot as plt
from networkx import networkx as nx
from tensorflux import graph as tfg
from tensorflux import session as tfs
from tensorflux2 import layer as lay

g = tfg.Graph()
g.initialize()

w = tfg.Variable(5.0, name="w")
b = tfg.Variable(-1.0, name="b")
x = tfg.Placeholder(name="x")

lay.Affine()

session = tfs.Session()


# nx.draw_networkx(g)
# plt.show()
#
# session = tfs.Session()
# output = session.run(z, {x: [1, 2]})
# print(output)
