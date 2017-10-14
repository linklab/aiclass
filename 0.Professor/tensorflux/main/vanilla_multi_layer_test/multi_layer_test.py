import tensorflux.graph as tfg
import tensorflux.networks as tfn
import tensorflux.enums as tfe
import datasource.mnist as mnist
import tensorflux.functions as tff
import math

input_size = 784
hidden_layer1_size = 128
hidden_layer2_size = 128
output_size = 10
batch_size = 1000

n = tfn.Multi_Layer_Network(
    input_size=input_size,
    hidden_size_list=[hidden_layer1_size, hidden_layer2_size],
    output_size=10
)

x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")

n.set_data(x, target)
n.initialize_param(initializer=tfe.Initializer.Uniform.value)
n.layering(activator=tfe.Activator.ReLU.value)
n.set_optimizer(optimizer=tfe.Optimizer.SGD.value, learning_rate=0.1)

data = mnist.MNIST_Data()

forward_final_output = n.feed_forward(
    input_data=data.test_input
)

print(forward_final_output.shape)
print(tff.accuracy(forward_final_output, data.test_target))

n.learning(max_epoch=100, data=data, batch_size=batch_size, print_period=10, verbose=False)

