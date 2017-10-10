# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import tensorflux.graph as tfg
import tensorflux.networks as tfn
import tensorflux.enums as tfe
import datasource.simple as simple_data

n = tfn.Three_Neurons_Network(input_size=2, output_size=1)

x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")

n.set_data(x, target)
<<<<<<< HEAD:0.Professor/tensorflux/main/three_neuron_test.py
n.initialize_param(initializer=tfe.Initializer.Randn.value)
n.layering(activator=tfe.Activator.ReLU.value)
n.set_optimizer(optimizer=tfe.Optimizer.SGD.value, learning_rate=0.01)
=======
n.initialize_param(initializer=tfe.Initializer.Truncated_Normal.value)
#n.layering(activator=tfe.Activator.Sigmoid.value)
n.set_optimizer(optimizer=tfe.Optimizer.SGD.value, learning_rate=0.05)
>>>>>>> 42bb2ea7de8923c739823420515b7783170fd34b:1631036003_cheolminki/tensorflux2/main/three_neuron_test.py

#n.draw_and_show()

data = simple_data.Xor_Gate_Data()

n.print_feed_forward(
    num_data=data.num_train_data,
    input_data=data.training_input,
    target_data=data.training_target,
    x=x,
    verbose=False
)

n.learning(max_epoch=5000, data=data, x=x, target=target)

n.print_feed_forward(
    num_data=data.num_test_data,
    input_data=data.test_input,
    target_data=data.test_target,
    x=x,
    verbose=False
)