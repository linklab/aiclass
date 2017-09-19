from tensorflux import graph as tfg
from tensorflux import session as tfs
#import networkx as nx
#import matplotlib.pyplot as plt

g = tfg.Graph() # 컴퓨테이션 그래프 클래스를 생성
g.initialize() # 컴퓨테이션 그래프 클래스를 초기화
#"""
# Create variables
a = tfg.Variable(5.0, name="a") #변수 클래스 a를 생성하고 초기화
b = tfg.Variable(-1.0, name="b") #변수 클래스 b를 생성하고 초기화

# Create placeholder
x = tfg.Placeholder(name="x") #플레이스홀더는 이후 입력할 값으로 현재는 값까지는 초기화하지 않음

# Create hidden node y
y = tfg.Mul(a, x, name="y") #곱 연산 클래스를 생성

# Create output node z
z = tfg.Add(y, b, name="z") #합 연산 클래스를 생성

#nx.draw_networkx(g, with_labels=True)
#plt.show(block=True)

# 수행~~~~~~~~~~~~~~~~~~~~~~
session = tfs.Session()
output = session.run(z, {x: 1.0})
print (output)
output = session.run(z, {x: 2.0})
print (output)
output = session.run(z, {x: 3.0})
print(output)
#"""
"""
# Create variables
A = tfg.Variable([[1, 0], [0, -1]], name="A")
b = tfg.Variable([1, 1], name="b")

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
"""