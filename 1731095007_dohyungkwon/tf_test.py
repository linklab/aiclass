from tensorflux import graph as tfg # graph 모듈을 객체화

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

