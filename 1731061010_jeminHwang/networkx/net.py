import networkx as nx
import matplotlib.pyplot as plt

g=nx.Graph()
g.add_edge('a','b')

nx.draw_networkx(g)
plt.show()

#nx.draw_networkx(g, with_labels=True)
#plt.show(block=True)