from Tensorflux import graph as tfg
from Tensorflux import session as tfs
import networkx_test as nx
import matplotlib.pyplot as plt

g = tfg.Graph() #그래프 객체 생성
g.initialize() #생성한 그래프 객체 초기화

# Create variables
A = tfg.Variable([[1, 0], [0, -1]], name="A")
b = tfg.Variable([1, 1], name="b")

# Create placeholder
x = tfg.Placeholder(name="x")

# Create hidden node y
y = tfg.Matmul(A, x, name="y")

# Create output node z
z = tfg.Add(y, b, name="z")

nx.draw_networkx(g, with_labels=True)
plt.show(block=True)

session = tfs.Session()
output = session.run(z, {x: [1, 2]})
print(output)