# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import tensorflux.graph as tfg
import tensorflux.networks as tfn
import tensorflux.enums as tfe
import datasource.simple as simple_data

n = tfn.Three_Neurons_Network(input_size=2, output_size=1)

x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")
n.set_data(x, target)
n.initialize_param(initializer=tfe.Initializer.Random.value) # 사용되는 가중치가 0, bias가 0이므로 ReLU 함수가 동작하지 않음
n.layering(activator=tfe.Activator.ReLU.value)
n.set_optimizer(optimizer=tfe.Optimizer.SGD.value, learning_rate=0.01)

data = simple_data.Xor_Gate_Data()

n.print_feed_forward(num_data=data.num_train_data,
                     input_data=data.training_input,
                     target_data=data.training_target,
                     x=x)
n.learning(max_epoch=10000, data=data, x=x, target=target)

n.print_feed_forward(num_data=data.num_test_data,
                     input_data=data.test_input,
                     target_data=data.test_target,
                     x=x)

# n.draw_and_show()
