# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import Tensorflux.graph_new as tfg
import Tensorflux.networks as tfn
import Tensorflux.enums as tfe
import datasource.simple as simple_data

n = tfn.Single_Neuron_Network(input_size=2, output_size=1)

x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")
n.set_data(x, target)
#
n.initialize_param(initializer=tfe.Initializer.Zero.value)
n.layering(activator=tfe.Activator.ReLU.value)
n.set_optimizer(optimizer=tfe.Optimizer.SGD.value, learning_rate=0.01)
#
data = simple_data.Or_Gate_Data()
#data = simple_data.Xor_Gate_Data()
n.print_feed_forward(num_data=data.num_train_data, input_data=data.training_input, target_data=data.training_target, x=x)
n.learning(max_epoch=1000, data=data, x=x, target=target)
n.print_feed_forward(num_data=data.num_test_data, input_data=data.test_input, target_data=data.test_target, x=x)

n.draw_and_show()