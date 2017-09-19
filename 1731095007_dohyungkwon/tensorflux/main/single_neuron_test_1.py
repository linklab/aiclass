# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import tensorflux.graph as tfg
import tensorflux.networks as tfn
import tensorflux.enums as tfe
import datasource.simple as simple_data

n = tfn.Single_Neuron_Network(input_size=1, output_size=1)

# Ctrl+B, Ctrl+E
x = tfg.Placeholder(name="x")# datasource/simple.py lin16
target = tfg.Placeholder(name="target")# datasource/simple.py lin17
n.set_data(x, target) # networks.py line35-37
n.initialize_scalar_param(5.0, -1.0) #https://www.dropbox.com/s/ni4r8gyfr1lw2t3/02.Artificial_Single_Neuron.pdf?dl=0
# #10page
# networks.py line114

n.layering(activator=tfe.Activator.ReLU.value) # ReLU.value : 클래스
n.set_optimizer(optimizer=tfe.Optimizer.SGD.value, learning_rate=0.01) # networks.py
# network(computational graph) 구성 종료.

data = simple_data.Simple_Function_Data()

#networks.py line98
n.print_feed_forward(num_data=data.num_train_data,
                     input_data=data.training_input,
                     target_data=data.training_target, x=x)
# ReLU단에서 값이 나온다.
#https://www.dropbox.com/s/ni4r8gyfr1lw2t3/02.Artificial_Single_Neuron.pdf?dl=0
# #13page

# x; placeholder
n.learning(max_epoch=100, data=data, x=x, target=target) # networks.py

n.print_feed_forward(num_data=data.num_test_data,
                     input_data=data.test_input,
                     target_data=data.test_target, x=x)

#n.draw_and_show()
