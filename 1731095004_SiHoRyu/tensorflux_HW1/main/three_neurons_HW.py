# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import tensorflux_HW1.graph as tfg
import tensorflux_HW1.networks as tfn
import tensorflux_HW1.enums as tfe
import datasource.simple as simple_data

#세개의 뉴런을 사용해서 XOR 게이트가 학습되어야 함

n = tfn.Two_Neurons_Network(input_size=2, output_size=1)

x = tfg.Placeholder(name="x")

target = tfg.Placeholder(name="target")

n.set_data(x, target)

n.initialize_param(initializer=tfe.Initializer.Point_One.value)

n.layering(activator=tfe.Activator.ReLU.value)
n.set_optimizer(optimizer=tfe.Optimizer.SGD.value, learning_rate=0.01)

#XOR gate data
data = simple_data.Xor_Gate_Data()

n.print_feed_forward(num_data=data.num_train_data, input_data=data.training_input, target_data=data.training_target, x=x)

#print(data.num_train_data)
#print(data.training_input)

n.learning(max_epoch=2, data=data, x=x, target=target)

n.print_feed_forward(num_data=data.num_test_data, input_data=data.test_input, target_data=data.test_target, x=x)

n.draw_and_show()


#Three neron network 숙제
#networks.py 에 작성
