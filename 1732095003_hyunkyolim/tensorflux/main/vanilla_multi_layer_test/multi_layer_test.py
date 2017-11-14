import tensorflux.graph as tfg
import tensorflux.Multi_Layer_Network as tfn
import tensorflux.enums as tfe
import datasource.mnist as mnist
import tensorflux.functions as tff

input_size = 784
hidden_layer1_size = 128
hidden_layer2_size = 128
output_size = 10

x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")

n = tfn.Multi_Layer_Network(
    input_size=input_size,
    hidden_size_list=[hidden_layer1_size, hidden_layer2_size],
    output_size=output_size,
    input_node=x,
    target_node=target,
    initializer=tfe.Initializer.Normal.value,
    init_sd=0.01,
    # initializer=tfe.Initializer.Xavier.value,
    activator=tfe.Activator.ReLU.value,
    optimizer=tfe.Optimizer.Adam.value,
    learning_rate=0.001
)

#n.draw_and_show()

data = mnist.MNIST_Data(validation_size=5000, n_splits=12, is_onehot_target=True)

forward_final_output = n.feed_forward(input_data=data.test_input, is_numba=False)
print(forward_final_output.shape)
print(tff.accuracy(forward_final_output, data.test_target))

batch_size = 1000
n.learning(max_epoch=5, data=data, batch_size=batch_size, print_period=1, is_numba=True, verbose=False)

forward_final_output = n.feed_forward(input_data=data.test_input, is_numba=False)
print(tff.accuracy(forward_final_output, data.test_target))