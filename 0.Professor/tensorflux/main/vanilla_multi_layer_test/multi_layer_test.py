import tensorflux.graph as tfg
import tensorflux.deep_learning_networks as tfn
import tensorflux.enums as tfe
import datasource.mnist as mnist
import tensorflux.functions as tff

input_size = 784
hidden_layer1_size = 128
hidden_layer2_size = 128
output_size = 10
batch_size = 1000

x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")

n = tfn.Multi_Layer_Network(
    input_size=input_size,
    hidden_size_list=[hidden_layer1_size, hidden_layer2_size],
    output_size=10,
    input_node=x,
    target_node=target,
    init_mean=0.0,
    init_sd=0.01,
    activator=tfe.Activator.ReLU.value,
    optimizer=tfe.Optimizer.SGD.value,
    learning_rate=0.01
)

#n.draw_and_show()

data = mnist.MNIST_Data()

forward_final_output = n.feed_forward(input_data=data.test_input)
print(forward_final_output.shape)
print(tff.accuracy(forward_final_output, data.test_target))

n.learning(max_epoch=40, data=data, batch_size=batch_size, print_period=1, verbose=False)

forward_final_output = n.feed_forward(input_data=data.test_input)
print(tff.accuracy(forward_final_output, data.test_target))