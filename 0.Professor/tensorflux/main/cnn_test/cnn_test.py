import tensorflux.graph as tfg
import tensorflux.CNN as tfn
import tensorflux.enums as tfe
import datasource.mnist as mnist
import tensorflux.functions as tff

"""
    conv - relu - conv- relu - pool -
    conv - relu - conv- relu - pool -
    conv - relu - conv- relu - pool -
    affine - relu - dropout - affine - dropout - softmax
"""

input_dim = (1, 28, 28)
conv_param_0 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
conv_param_2 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
conv_param_3 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
conv_param_4 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
conv_param_5 = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
fc_size = 64
output_size = 10

x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")

n = tfn.CNN(
    input_dim=input_dim,
    conv_param_list=[conv_param_0, conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5],
    fc_size=fc_size,
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