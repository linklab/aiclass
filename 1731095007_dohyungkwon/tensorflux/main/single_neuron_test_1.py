# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import tensorflux.graph as tfg
import tensorflux.networks as tfn
import tensorflux.enums as tfe
import datasource.simple as simple_data

n = tfn.Single_Neuron_Network(input_size=1, output_size=1)

x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")
n.set_data(x, target)
n.initialize_scalar_param(5.0, -1.0)
n.layering(activator=tfe.Activator.ReLU.value)
n.set_optimizer(optimizer=tfe.Optimizer.SGD.value, learning_rate=0.01)

#n.draw_and_show()

data = simple_data.Simple_Function_Data()

n.print_feed_forward(
    num_data=data.num_train_data,
    input_data=data.train_input,
    target_data=data.train_target,
    verbose=False
)

#n.learning(max_epoch=500, data=data, bp=False, print_period=10, verbose=False)
#backpropagation = False --> Numerical Derivative
#verbose = True --> learning 과정 출력
n.learning(max_epoch=500, data=data, bp=True, print_period=10, verbose=False)

n.print_feed_forward(
    num_data=data.num_test_data,
    input_data=data.test_input,
    target_data=data.test_target,
    verbose=False
)

#n.draw_and_show()




# ~171009
# # Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
# import tensorflux.graph as tfg
# import tensorflux.networks as tfn
# import tensorflux.enums as tfe
# import datasource.simple as simple_data
#
# n = tfn.Single_Neuron_Network(input_size=1, output_size=1)
#
# # Ctrl+B, Ctrl+E
# x = tfg.Placeholder(name="x")# datasource/simple.py lin16
# target = tfg.Placeholder(name="target")# datasource/simple.py lin17
# n.set_data(x, target) # networks.py line35-37
# n.initialize_scalar_param(5.0, -1.0) #https://www.dropbox.com/s/ni4r8gyfr1lw2t3/02.Artificial_Single_Neuron.pdf?dl=0
# # #10page
# # networks.py line114
#
# n.layering(activator=tfe.Activator.ReLU.value) # ReLU.value : 클래스
# n.set_optimizer(optimizer=tfe.Optimizer.SGD.value, learning_rate=0.058) # networks.py
# # network(computational graph) 구성 종료.
#
# data = simple_data.Simple_Function_Data()
#
# #networks.py line98
# n.print_feed_forward(num_data=data.num_train_data,
#                      input_data=data.training_input,
#                      target_data=data.training_target, x=x)
# # ReLU단에서 값이 나온다.
# #https://www.dropbox.com/s/ni4r8gyfr1lw2t3/02.Artificial_Single_Neuron.pdf?dl=0
# # #13page
#
# # x; placeholder
# n.learning(max_epoch=100, data=data, x=x, target=target) # networks.py
# # datasource/simple.py에서 데이터의 개수를 늘려주거나
# # max_epoch를 늘려줌으로써
# # learning_rate을 바꾸어주어도 된다.
# # 함수 estimation의 결과는 더 좋아지게 된다.
#
# n.print_feed_forward(num_data=data.num_test_data,
#                      input_data=data.test_input,
#                      target_data=data.test_target, x=x)
#
# # 이것이 바로 function estimation(generalization) 원리
#
# #n.draw_and_show()
