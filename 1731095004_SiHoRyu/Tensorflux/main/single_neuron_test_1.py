# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import Tensorflux.graph_new as tfg
import Tensorflux.networks as tfn
import Tensorflux.enums as tfe
import datasource.simple as simple_data

n = tfn.Single_Neuron_Network(input_size=1, output_size=1)

#placeholder는 데이터가 비어있는 하나의 노드
x = tfg.Placeholder(name="x")
target = tfg.Placeholder(name="target")
#set_data는 플레이스홀더에 값을 setting하는 역할
n.set_data(x, target)

n.initialize_scalar_param(5.0, -1.0)
#뉴런이 하나일 때도 layering 해야함, affine과 activator층을 쌓는 작업
#하나의 뉴런은 affine과 함수로 구성됨
n.layering(activator=tfe.Activator.ReLU.value)
#weight와 bias를 최적화하는 것 optimizer in here we use SGD
n.set_optimizer(optimizer=tfe.Optimizer.SGD.value, learning_rate=0.01)

data = simple_data.Simple_Function_Data()
#x에 값을 집어 넣어서 ReLu에서 값을 뽑음

n.print_feed_forward(num_data=data.num_train_data, input_data=data.training_input, target_data=data.training_target, x=x)
#Feed forward output 값을 target값과 더 가깝게 하려면
#epoch 횟수를 늘리거나
#simple 데이터에서 데이터 갯수를 늘려준다(training data)
n.learning(max_epoch=10000, data=data, x=x, target=target)
n.print_feed_forward(num_data=data.num_test_data, input_data=data.test_input, target_data=data.test_target, x=x)

#n.draw_and_show()
