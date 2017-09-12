from tensorflux import graph as tfg # graph 모듈을 객체화
from tensorflux import session as tfs

g = tfg.Graph()
g.initialize()

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