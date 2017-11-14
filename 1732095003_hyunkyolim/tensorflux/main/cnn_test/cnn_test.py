import tensorflux.graph as tfg
import tensorflux.CNN as tfn
import tensorflux.enums as tfe
import datasource.mnist as mnist
import tensorflux.functions as tff

"""
    conv0 (relu0) - conv1 (relu1) - pool2 - 
    conv3 (relu3) - conv4 (relu4) - pool5 - 
    affine6 (relu6) - affine7 - softmax (output)
"""

input_dim = (1, 28, 28)
cnn_param_list = [
    {'type': 'conv', 'filter_num': 2, 'filter_size': 3, 'pad': 1, 'stride': 1},
    # {'type': 'conv', 'filter_num': 4, 'filter_size': 3, 'pad': 1, 'stride': 1},
    {'type': 'pool', 'filter_size': 2, 'stride': 2},
    {'type': 'conv', 'filter_num': 4, 'filter_size': 3, 'pad': 1, 'stride': 1},
    # {'type': 'conv', 'filter_num': 8, 'filter_size': 3, 'pad': 1, 'stride': 1},
    {'type': 'pool', 'filter_size': 2, 'stride': 2},
]
fc_hidden_size = 64
output_size = 10

x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")

n = tfn.CNN(
    input_dim=input_dim,
    cnn_param_list=cnn_param_list,
    fc_hidden_size=fc_hidden_size,
    output_size=output_size,
    input_node=x,
    target_node=target,
    conv_initializer=tfe.Initializer.Conv_Xavier_Normal.value,
    initializer=tfe.Initializer.Normal.value,
    init_sd=0.01,
    # initializer=tfe.Initializer.Xavier.value,
    activator=tfe.Activator.ReLU.value,
    optimizer=tfe.Optimizer.Adam.value,
    learning_rate=0.001
)

#n.draw_and_show()

data = mnist.MNIST_Data(validation_size=5000, n_splits=12, is_onehot_target=True, cnn=True)

forward_final_output = n.feed_forward(input_data=data.test_input, is_numba=False)
print(forward_final_output.shape)
print(tff.accuracy(forward_final_output, data.test_target))

batch_size = 1000
n.learning(max_epoch=5, data=data, batch_size=batch_size, print_period=1, is_numba=False, verbose=False)

forward_final_output = n.feed_forward(input_data=data.test_input, is_numba=False)
print(tff.accuracy(forward_final_output, data.test_target))