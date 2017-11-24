import tensorflux.graph as tfg
import tensorflux.CNN as tfn
import tensorflux.enums as tfe
import datasource.mnist as mnist
import tensorflux.functions as tff

"""
    conv0 (batch_normal0, relu0) - pool1 - 
    conv2 (batch_normal2, relu2) - pool3 - 
    affine4 (relu4) - affine5 - softmax (output)
"""

input_dim = (1, 28, 28)
cnn_param_list = [
    {'type': 'conv', 'filter_num': 3, 'filter_h': 3, 'filter_w': 3, 'pad': 1, 'stride': 1},
    {'type': 'pooling', 'filter_h': 2, 'filter_w': 2, 'stride': 2},
    {'type': 'conv', 'filter_num': 3, 'filter_h': 3, 'filter_w': 3, 'pad': 1, 'stride': 1},
    {'type': 'pooling', 'filter_h': 2, 'filter_w': 2, 'stride': 2},
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
    use_batch_normalization=True,
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

forward_final_output = n.feed_forward(input_data=data.test_input, is_train=False, is_numba=False)
print(forward_final_output.shape)
print(tff.accuracy(forward_final_output, data.test_target))

batch_size = 1000
n.learning(max_epoch=5, data=data, batch_size=batch_size, print_period=1, is_numba=False, verbose=False)

forward_final_output = n.feed_forward(input_data=data.test_input, is_train=False, is_numba=False)
print(tff.accuracy(forward_final_output, data.test_target))