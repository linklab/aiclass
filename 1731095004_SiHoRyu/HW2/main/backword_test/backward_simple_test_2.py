import HW2.graph as tfg
import HW2.session as tfs

apple_price = 100
apple_num = 2

orange_price = 150
orange_num = 3

tax = 1.1

g = tfg.Graph()

# Create variables
a = tfg.Variable(apple_price, name="a")
b = tfg.Variable(apple_num, name="b")

c = tfg.Variable(orange_price, name="c")
d = tfg.Variable(orange_num, name="d")

e = tfg.Variable(tax, name="e")

# Create Mul operation node
f = tfg.Mul(a, b, name="f")
g = tfg.Mul(c, d, name="g")

# Create Add operation node
h = tfg.Add(f, g, name="h")

# Create Mul operation node
i = tfg.Mul(h, e, name="i")

session = tfs.Session()
# forward
final_price = session.run(i, verbose=False)
print("final_price: {:f}".format(float(final_price)))

print()

#backward
d_in = 1
d_total_price, d_tax = i.backward(d_in)
print("d_total_price: {:f}".format(float(d_total_price)))
print("d_tax: {:f}".format(float(d_tax)))

d_total_apple_price, d_total_orange_price = h.backward(d_total_price)
print("d_total_apple_price: {:f}".format(float(d_total_apple_price)))
print("d_total_orange_price: {:f}".format(float(d_total_orange_price)))

d_apple_price, d_apple_num = f.backward(d_total_apple_price)
print("d_apple_price: {:f}".format(float(d_apple_price)))
print("d_apple_num: {:f}".format(float(d_apple_num)))

d_orange_price, d_orange_num = g.backward(d_total_orange_price)
print("d_orange_price: {:f}".format(float(d_orange_price)))
print("d_orange_num: {:f}".format(float(d_orange_num)))