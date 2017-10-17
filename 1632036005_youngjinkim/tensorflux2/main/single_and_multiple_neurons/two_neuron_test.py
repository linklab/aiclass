# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import tensorflux2.graph as tfg
import tensorflux2.networks as tfn
import tensorflux2.enums as tfe
import datasource.simple as simple_data

n = tfn.Two_Neurons_Network(input_size=2, output_size=1)

x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")
n.set_data(x, target)
n.initialize_param(initializer=tfe.Initializer.Point_One.value)
n.layering(activator=tfe.Activator.ReLU.value)
n.set_optimizer(optimizer=tfe.Optimizer.SGD.value, learning_rate=0.01)

#n.draw_and_show()

data = simple_data.And_Gate_Data()

n.print_feed_forward(
    num_data=data.num_train_data,
    input_data=data.train_input,
    target_data=data.train_target,
    verbose=False
)

#n.learning(max_epoch=200, data=data, bp=False, print_period=10, verbose=False)
n.learning(max_epoch=200, data=data, bp=True, print_period=10, verbose=False)


n.print_feed_forward(
    num_data=data.num_test_data,
    input_data=data.test_input,
    target_data=data.test_target,
    verbose=False
)

#n.draw_and_show()
