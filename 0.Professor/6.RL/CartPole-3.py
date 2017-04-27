# -*- coding: utf-8 -*-
# https://github.com/openai/gym/wiki/CartPole-v0
import tensorflow as tf
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
input_size = env.observation_space.shape[0]     # 4
output_size = env.action_space.n                # 2

# 꺼내서 사용할 리플레이 갯수
REPLAY = 10

# 리플레이를 저장할 리스트
REPLAY_MEMORY = []

# 미니배치
MINIBATCH = 50

# 하이퍼파라미터
learning_rate = 0.1
num_episodes = 1000
e = 0.1
discount_factor = .9
rList = []
train_error_list = []
episode_list = []

# 네트워크 클래스 구성
class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        # 네트워크 정보 입력
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self.hidden_size = 16
        # 네트워크 생성
        self.build_network()

    def build_network(self):
        # Vanilla Neural Network (Just one hidden layer)
        self.X=tf.placeholder(dtype=tf.float32, shape=[None, self.input_size])
        self.Y=tf.placeholder(dtype=tf.float32, shape=[None, self.output_size])

        self.W1 = tf.Variable(tf.random_normal([self.input_size, self.hidden_size], mean=0.0, stddev=1.0))
        self.W2 = tf.Variable(tf.random_normal([self.hidden_size, self.output_size], mean=0.0, stddev=1.0))

        self.L1=tf.nn.tanh(tf.matmul(self.X, self.W1))

        self.Q_pred = tf.matmul(self.L1, self.W2)

        # 손실 함수
        self.loss = tf.reduce_sum(tf.square(self.Y - self.Q_pred))
        self.train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    # 예측한 Q값 구하기
    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self.Q_pred, feed_dict={self.X: x})

    # 네트워크 학습
    def update(self, x, y):
        loss_value, _ = self.session.run([self.loss, self.train], feed_dict={self.X: x, self.Y: y})
        return loss_value

# 미니배치를 이용한 학습
def replay_train(DQN):
    error_sum = 0
    for sample in random.sample(REPLAY_MEMORY, REPLAY):
        state, action, reward, new_state, done = sample
        Q = DQN.predict(state)
        # DQN 알고리즘으로 학습
        if done:
            Q[0, action] = -100
        else:
            Q[0, action] = reward + discount_factor * np.max(DQN.predict(new_state))

        loss_value = DQN.update(np.reshape(state, [1, DQN.input_size]), Q)
        error_sum += loss_value
    return error_sum / REPLAY


def bot_play(DQN):
    state = env.reset()
    reward_sum = 0
    num_actions = 0
    while True:
        env.render()
        action = np.argmax(DQN.predict(state))
        new_state, reward, done, info = env.step(action)
        reward_sum += reward
        state = new_state
        num_actions += 1

        if done:
            print("=============================================")
            print("Total number of actions: {0}, Total rewards: {1}".format(num_actions, reward_sum))
            break


def draw_error_values():
    fig = plt.figure(figsize=(20, 5))
    plt.ion()
    plt.subplot(111)
    plt.plot(episode_list[0:], train_error_list[0:], 'r', label='Train')
    plt.ylabel('Error Values')
    plt.xlabel('Episodes')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()
    input("Press Enter to close the trained error figure...")
    plt.close(fig)

if __name__ == "__main__":
    with tf.Session() as sess:
        # mainDQN 이라는 DQN 클래스 생성
        mainDQN = DQN(sess, input_size, output_size)

        # 변수 초기화
        sess.run(tf.global_variables_initializer())

        for episode in range(num_episodes):
            state = env.reset()
            e = 1. / ((episode/10)+1)
            rall = 0
            done = False
            num_actions = 0

            while not done and num_actions < 5000:
                env.render()
                # e-greedy 를 사용하여 action값 구함
                if e > np.random.rand(1):
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                # action을 수행함
                new_state, reward, done, _ = env.step(action)

                # state, action, reward, next_state, done 을 메모리에 저장
                REPLAY_MEMORY.append([state, action, reward, new_state, done])

                # 메모리에 50000개 이상의 값이 들어가면 가장 먼저 들어간 것부터 삭제
                if len(REPLAY_MEMORY) > 50000:
                    del REPLAY_MEMORY[0]

                rall += reward
                state = new_state
                num_actions += 1

            # 10 번의 스탭마다 미니배치로 학습
            if not episode == 0 and episode % 10 == 0:
                episode_list.append(episode)
                mean_error_sum = 0
                for _ in range(MINIBATCH):
                    mean_error_sum += replay_train(mainDQN)
                train_error_list.append(mean_error_sum / MINIBATCH)

            rList.append(rall)
            print("Episode {0} finished after {1} actions with r={2}. Running score: {3}".format(episode, num_actions, rall, np.mean(rList)))

        env.reset()
        env.close()
        draw_error_values()
        input("Press Enter to make the trained bot play...")
        bot_play(mainDQN)