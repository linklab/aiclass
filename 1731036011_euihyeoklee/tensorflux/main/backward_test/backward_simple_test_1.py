import tensorflux.graph as tfg
import tensorflux.session as tfs

apple_price = 100
apple_num = 2
tax = 1.1

g = tfg.Graph()

# Create variables
a = tfg.Variable(apple_price, name="a")
b = tfg.Variable(apple_num, name="b")
c = tfg.Variable(tax, name="c")

# Create Mul operation node
d = tfg.Mul(a, b, name="d")

# Create Mul operation node
e = tfg.Mul(d, c, name="e")

session = tfs.Session()

# forward
total_apple_price = session.run(d, verbose=False)
print("total_apple_price: {:f}".format(float(total_apple_price)))
final_price = session.run(e, verbose=False)
print("final_price: {:f}".format(float(final_price)))

print()

#backward
d_in = 1
d_total_apple_price, d_tax = e.backward(d_in)
print("d_total_apple_price: {:f}".format(float(d_total_apple_price)))
print("d_tax: {:f}".format(float(d_tax)))

d_apple_price, d_num = d.backward(d_total_apple_price)
print("d_apple_price: {:f}".format(float(d_apple_price)))
print("d_num: {:f}".format(float(d_num)))