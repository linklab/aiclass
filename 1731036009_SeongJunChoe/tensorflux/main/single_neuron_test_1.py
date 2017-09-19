# Reference: http://www.deepideas.net/deep-learning-from-scratch-i-computational-graphs/
import tensorflux.graph as tfg
import tensorflux.networks as tfn
import tensorflux.enums as tfe
import datasource.simple as simple_data

n = tfn.Single_Neuron_Network(input_size=1, output_size=1)

# ctrl + B : 해당 함수, 클래스의 원형으로 찾아가는 것
# ctrl + / : 지정한 범위 주석처리
x = tfg.Placeholder(name="x")  # 나중에 값이 지정되는 위치
target = tfg.Placeholder(name="target")
n.set_data(x, target)   # 입력 노드와 출력 노드를 설정

n.initialize_scalar_param(5.0, -1.0)    #(가중치, 바이어스)
n.layering(activator=tfe.Activator.ReLU.value)  # 층을 쌓는 것. -> u 연산 : Affine, f(u) 연산 : Relu
#     (x) -> (affine) -> (Relu) -> (SGD) -> (target)
# (weight)-^-(bias)                 ^-(optimizer)
n.set_optimizer(optimizer=tfe.Optimizer.SGD.value, learning_rate=0.01)  # Optimizer는 30번째 라인인 n.learning에서 수행됨
# 최적화기를 설정 : SGD 최적화기 사용

data = simple_data.Simple_Function_Data()

# 훈련 데이터로 테스트
n.print_feed_forward(num_data=data.num_train_data,
                     input_data=data.training_input,
                     target_data=data.training_target,
                     x=x)

n.learning(max_epoch=100,   # epoch : 횟수?
           data=data,       # data : 사용할 데이터?
           x=x,             # x : 지정한 값
           target=target)   # target : 목표치?

# 테스트 데이터로 테스트 -> 훈련 데이터로도 가능
n.print_feed_forward(num_data=data.num_test_data,
                     input_data=data.test_input,
                     target_data=data.test_target,
                     x=x)
# 아직 Relu를 통과하지 않음????
# n.draw_and_show()함

# caffe ? tensorflow ? pytorch ?
