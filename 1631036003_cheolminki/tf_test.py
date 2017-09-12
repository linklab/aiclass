# -*- coding: utf-8 -*-
from tensorflux import graph as tfg #tensorflux 패키지의 graph 모듈을 tfg라는 이름으로 객체선
from tensorflux import session as tfs
import networkx as nx
import matplotlib.pyplot as plt

g = tfg.Graph()
g.initialize()

"""
# variable 생성
a = tfg.Variable(5.0, name="a")
b = tfg.Variable(1.0, name="b")

# placeholder 생성
x = tfg.Placeholder(name = "X")

# Create hidden node y
y = tfg.Mul(a, x, name="y")

# Create output node z
z = tfg.Add(y, b, name="z") #z : operation

session = tfs.Session()
output = session.run(z, {x: 1.0})
print(output)
"""

# Create variables
A = tfg.Variable([[1, 0], [0, -1]], name="a")
b = tfg.Variable([1, 1], name="b")

# Create placeholder
x = tfg.Placeholder(name="x")

# Create hidden node y
y = tfg.Matmul(A, x, name="y")      #a, x, b가 행렬이기 때문에 mul이 아니라 matmul

# Create output node z
z = tfg.Add(y, b, name="z")

nx.draw_networkx(g, with_labels=True)
plt.show(block=True)

session = tfs.Session()
output = session.run(z, {x: [1, 2]})
print(output)