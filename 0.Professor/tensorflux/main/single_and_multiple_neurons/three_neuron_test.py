# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import tensorflux.graph as tfg
import tensorflux.networks as tfn
import tensorflux.enums as tfe
import datasource.simple as simple_data

n = tfn.Three_Neurons_Network(input_size=2, output_size=1)

x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")

n.set_data(x, target)
n.initialize_param(initializer=tfe.Initializer.Uniform.value)
start_param_description = n.get_param_describe()

while not (start_param_description.variance > 0.17 and 0.0 < start_param_description.mean - 0.5 < 0.05):
    print("Start Param Variance: {:f}, Start Param Mean: {:f} --> Re-initialize".format(
        start_param_description.variance,
        start_param_description.mean
    ))
    n.initialize_param(initializer=tfe.Initializer.Uniform.value)
    start_param_description = n.get_param_describe()

n.layering(activator=tfe.Activator.ReLU.value)
n.set_optimizer(optimizer=tfe.Optimizer.SGD.value, learning_rate=0.05)

#n.draw_and_show()

data = simple_data.Xor_Gate_Data()

n.print_feed_forward(
    num_data=data.num_train_data,
    input_data=data.training_input,
    target_data=data.training_target,
    verbose=False
)

#n.learning(max_epoch=1000, data=data, verbose=False)
n.learning_bp(max_epoch=1500, data=data, verbose=False)

n.print_feed_forward(
    num_data=data.num_test_data,
    input_data=data.test_input,
    target_data=data.test_target,
    verbose=False
)

end_param_description = n.get_param_describe()

print(start_param_description)
print(end_param_description)
