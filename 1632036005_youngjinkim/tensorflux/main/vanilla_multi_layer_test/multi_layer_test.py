import tensorflux.graph as tfg
import tensorflux.deep_learning_networks as tfn
import tensorflux.enums as tfe
import datasource.mnist as mnist
import tensorflux.functions as tff
import math

input_size = 784
hidden_layer1_size = 128
hidden_layer2_size = 128
output_size = 10
model_params_dir = "../../../tmp"

x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")

n = tfn.Multi_Layer_Network(
    input_size=input_size,
    hidden_size_list=[hidden_layer1_size, hidden_layer2_size],
    output_size=output_size,
    input_node=x,
    target_node=target,
    # initializer=tfe.Initializer.Normal.value,
    # init_sd=0.01,
    # initializer=tfe.Initializer.Xavier.value,
    initializer=tfe.Initializer.Standard_Uniform.value,
    # activator=tfe.Activator.ReLU.value,
    activator=tfe.Activator.Tanh.value,
    # activator=tfe.Activator.SoftSign.value,
    optimizer=tfe.Optimizer.Adam.value,
    learning_rate=0.001,
    model_params_dir=model_params_dir
)

#n.draw_and_show()

data = mnist.MNIST_Data()

forward_final_output = n.feed_forward(input_data=data.test_input, is_numba=False)
print(forward_final_output.shape)
print(tff.accuracy(forward_final_output, data.test_target))

batch_size = 1000
n.learning(max_epoch=40, data=data, batch_size=batch_size, print_period=1, is_numba=True, verbose=False)

forward_final_output = n.feed_forward(input_data=data.test_input, is_numba=False)
print(tff.accuracy(forward_final_output, data.test_target))