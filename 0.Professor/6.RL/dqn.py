# -*- coding: utf-8 -*-
import tensorflow as tf
import gym
import numpy as np
import random as ran

env = gym.make('CartPole-v0')

# 꺼내서 사용할 리플레이 갯수
REPLAY = 10
# 리플레이를 저장할 리스트
REPLAY_MEMORY = []
# 미니배치
MINIBATCH = 50

INPUT = env.observation_space.shape[0]
OUTPUT = env.action_space.n

# 하이퍼파라미터
LEARNING_LATE = 0.1
NUM_EPISODE = 4000
e = 0.1
DISCOUNT = 0.9
rList = []

# 네트워크 클래스 구성
class DQN:
    def __init__(self, session, input_size, output_size, name="main"):

        # 네트워크 정보 입력
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        # 네트워크 생성
        self.build_network()

    def build_network(self, width = 10, L_rate = 1e-1):

        # 네트워크 구조
        self.x=tf.placeholder(dtype=tf.float32, shape=[None, self.input_size])

        W1 = tf.get_variable('W1',shape=[self.input_size, width],initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable('W2',shape=[width, self.output_size],initializer=tf.contrib.layers.xavier_initializer())

        L1 = tf.nn.tanh(tf.matmul(self.x,W1))

        self.Q_pre = tf.matmul(L1, W2)

        self.y = tf.placeholder(dtype=tf.float32, shape=(1, env.action_space.n))

        # 손실 함수
        self.loss = tf.reduce_sum(tf.square(self.y - self.Q_pre))
        self.train = tf.train.AdamOptimizer(learning_rate=L_rate).minimize(self.loss)

    # 예측한 Q값 구하기
    def predict(self, state):
        s_t = np.reshape(state, [1,self.input_size])
        return self.session.run(self.Q_pre, feed_dict={self.x : s_t})
    # 네트워크 학습
    def update(self, x, y):
        self.session.run(self.train, feed_dict={self.x : x, self.y : y})

# 미니배치를 이용한 학습
def replay_train(DQN, replay_memory, replay):
    for sample in ran.sample(replay_memory, replay):
        s_r, a_r, r_r, s1_r, d_r = sample
        Q = DQN.predict(s_r)
        # DQN 알고리즘으로 학습
        if d_r:
            Q[0, a_r] = -100
        else:
            Q[0, a_r] = r_r + DISCOUNT * np.max(DQN.predict(s1_r))

        DQN.update(np.reshape(s_r, [1, DQN.input_size]), Q)
# 메인
def main():
    with tf.Session() as sess:
        # mainDQN 이라는 DQN 클래스 생성
        mainDQN = DQN(sess, INPUT, OUTPUT)

        # 변수 초기화
        sess.run(tf.global_variables_initializer())
        for step in range(NUM_EPISODE):

            s = env.reset()
            e = 1. / ((step/10)+1)
            rall = 0
            d = False
            count=0

            while not d and count < 5000:
                env.render()
                count+=1
                # e-greedy 를 사용하여 action값 구함
                if e > np.random.rand(1):
                    a = env.action_space.sample()
                else:
                    a = np.argmax(mainDQN.predict(s))

                # action을 취함
                s1, r, d, _ = env.step(a)

                # state, action, reward, next_state, done 을 메모리에 저장
                REPLAY_MEMORY.append([s,a,r,s1,d])

                # 메모리에 50000개 이상의 값이 들어가면 가장 먼저 들어간 것부터 삭제
                if len(REPLAY_MEMORY) > 50000:
                    del REPLAY_MEMORY[0]

                rall += r
                s = s1

            # 10 번의 스탭마다 미니배치로 학습
            if step % 10 == 1 :
                for _ in range(MINIBATCH):
                    replay_train(mainDQN,REPLAY_MEMORY,REPLAY)



            rList.append(rall)
            print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(step, count, rall, np.mean(rList)))

if __name__ == '__main__':
    main()