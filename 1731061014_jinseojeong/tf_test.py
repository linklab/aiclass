from tensorflux2 import graph as tfg
from tensorflux2 import session as tfs

g = tfg.Graph()
g.initialize()

# Create variables
w = tfg.Variable(5.0, name="a")
b = tfg.Variable(-1.0, name="b")

# Create placeholder
x = tfg.Placeholder(name="x")

# Create hidden node y
y = tfg.Mul(w, x, name="y")

# Create output node z
z = tfg.Add(y, b, name="z")


session = tfs.Session()
output = session.run(z, {x: 1.0})
print(output)
output = session.run(z, {x: 2.0})
print(output)
output = session.run(z, {x: 3.0})
print(output)


