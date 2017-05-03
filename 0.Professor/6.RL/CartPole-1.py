# -*- coding: utf-8 -*-
# https://github.com/openai/gym/wiki/CartPole-v0
import tensorflow as tf
import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt

# 리플레이를 저장할 리스트
REPLAY_MEMORY = []

# 미니배치 - 꺼내서 사용할 리플레이 갯수
MINIBATCH = 50

# 하이퍼파라미터
learning_rate = 0.1
num_episodes = 100
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
        self.X = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
        self.Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

        self.W1 = tf.Variable(tf.random_normal([self.input_size, self.hidden_size], mean=0.0, stddev=1.0))
        self.W2 = tf.Variable(tf.random_normal([self.hidden_size, self.output_size], mean=0.0, stddev=1.0))

        self.L1=tf.nn.tanh(tf.matmul(self.X, self.W1))

        self.Qpred = tf.matmul(self.L1, self.W2)

        # 손실 함수
        self.A = tf.placeholder(dtype=tf.int32)
        self.loss = tf.square(self.Y[0, self.A] - self.Qpred[0, self.A])
        self.train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    # 예측한 Q값 구하기
    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self.Qpred, feed_dict={self.X: x})

    # 네트워크 학습
    def update(self, state, new_state, action):
        x = np.reshape(state, [1, input_size])
        # Get the label y
        Q_y = [[0.0, 0.0]]
        if done:
            # it's a terminal state
            Q_y[0][action] = -100
        else:
            x_new_state = np.reshape(new_state, [1, input_size])
            # Obtain the Q_y values by feeding the new state through the network
            Q_new_state = sess.run(self.Qpred, feed_dict={self.X: x_new_state})
            Q_y[0][action] = reward + discount_factor * np.max(Q_new_state)

        loss_value, _ = sess.run([self.loss, self.train], feed_dict={self.X: x, self.Y: Q_y, self.A: action})
        return loss_value


def bot_play(DQN):
    """
    See our trained network in action
    """
    input("Press Enter to make the trained bot play...")
    env = gym.make('CartPole-v0')
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
            time.sleep(1) # delays for 3 second
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
    env = gym.make('CartPole-v0')
    input_size = env.observation_space.shape[0]     # 4
    output_size = env.action_space.n                # 2

    with tf.Session() as sess:
        # DQN 클래스의 mainDQN 인스턴스 생성
        mainDQN = DQN(sess, input_size, output_size)

        # 변수 초기화
        init = tf.global_variables_initializer()
        sess.run(init)

        for episode in range(num_episodes):
            state = env.reset()
            e = 1. / ((episode / 50) + 10)
            rAll = 0
            done = False
            num_actions = 0

            while not done and num_actions < 5000:
                env.render()
                # e-greedy 를 사용하여 action값 구함
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                    print("Episode: {0}, Action: {1}".format(episode, action))
                else:
                    Q_h = mainDQN.predict(state)
                    action = np.argmax(Q_h)
                    print("Episode: {0}, State: {1}, Q_h: {2}, Action: {3}".format(episode, state, Q_h, action))

                # action을 수행함 --> Get new state and reward from environment
                new_state, reward, done, _ = env.step(action)

                loss_value = mainDQN.update(state, new_state, action)

                rAll += reward
                state = new_state
                num_actions += 1

            rList.append(rAll)
            print("Episode {0} finished after {1} actions with r={2}. Running score: {3}".format(episode, num_actions, rAll, np.mean(rList)))
            print()
            time.sleep(1) # delays for 1 second

            episode_list.append(episode)
            train_error_list.append(loss_value)

            # If the last 10's average steps are over 100, it's good enough
            if len(rList) > 10 and np.mean(rList[-10:]) > 100:
                break

        env.reset()
        env.close()
        draw_error_values()
        bot_play(mainDQN)