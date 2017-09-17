# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import tensorflux.graph as tfg
import tensorflux.networks as tfn
import tensorflux.enums as tfe
import tensorflux.session as tfs
import datasource.simple as simple_data
import numpy as np

### Single_Neuron_Network Test - I
n = tfn.Single_Neuron_Network(input_size=1, output_size=1)

x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")
n.set_data(x, target)

n.initialize_scalar_param(5.0, -1.0)

n.layering(activator=tfe.Activator.ReLU.value)

n.set_optimizer(optimizer=tfe.Optimizer.SGD.value, learning_rate=0.01)

session = tfs.Session()

data = simple_data.Simple_Function_Data()

for idx in range(data.num_train_data):
    train_input_data = data.training_input[idx]
    train_target_data = data.training_target[idx]

    output = session.run(n.output, {x: train_input_data}, vervose=False)
    print("Train Data: {:>5}, Feed Forward Output: {:>6}, Target: {:>6}".format(
        str(train_input_data), np.array2string(output), str(train_target_data)))
    print()

max_epoch = 100
for epoch in range(max_epoch):
    sum_train_error = 0.0
    for idx in range(data.num_train_data):
        train_input_data = data.training_input[idx]
        train_target_data = data.training_target[idx]

        grads = n.numerical_derivative(session, {x: train_input_data, target: train_target_data})
        n.optimizer.update(grads=grads)
        sum_train_error += session.run(n.error, {x: train_input_data, target: train_target_data}, vervose=False)

    sum_validation_error = 0.0
    for idx in range(data.num_validation_data):
        validation_input_data = data.validation_input[idx]
        validation_target_data = data.validation_target[idx]
        sum_validation_error += session.run(n.error, {x: validation_input_data, target: validation_target_data}, vervose=False)

    print("Epoch {:3d} Completed - w0: {:7.6f}, b0: {:7.6f} - Average Train Error: {:7.6f} - Average Validation Error: {:7.6f}".format(
        epoch, n.params['W0'].value, n.params['b0'].value, sum_train_error / data.num_train_data, sum_validation_error / data.num_validation_data))

print()

for idx in range(data.num_test_data):
    test_input_data = data.test_input[idx]
    test_target_data = data.test_target[idx]

    output = session.run(n.output, {x: test_input_data}, vervose=False)
    print("Test Data: {:>5}, Feed Forward Output: {:>12}, Target: {:>6}".format(
        str(test_input_data), np.array2string(output), str(test_target_data)))
    print()

#n.draw_and_show()
