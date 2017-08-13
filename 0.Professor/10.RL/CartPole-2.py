# -*- coding: utf-8 -*-
# https://github.com/openai/gym/wiki/CartPole-v0
import tensorflow as tf
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

# 리플레이를 저장할 리스트
REPLAY_MEMORY = deque()

# 하이퍼파라미터
INITIAL_EPSILON = 0.5
learning_rate = 0.0001
max_episodes = 10000
discount_factor = 0.9
episode_list = []
train_error_list = []
actions_list = []

# 테스트 에피소드 주기
TEST_PERIOD = 100

# 네트워크 클래스 구성
class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        # 네트워크 정보 입력
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self.hidden_size = 20
        # 네트워크 생성
        self.build_network()

    def build_network(self):
        # Vanilla Neural Network (Just one hidden layer)
        self.X = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
        self.Y = tf.placeholder(shape=[None], dtype=tf.float32)

        W1 = tf.Variable(tf.truncated_normal(shape=[self.input_size, self.hidden_size], mean=0.0, stddev=1.0))
        B1 = tf.Variable(tf.zeros(shape=[self.hidden_size]))
        W2 = tf.Variable(tf.truncated_normal(shape=[self.hidden_size, self.output_size], mean=0.0, stddev=1.0))
        B2 = tf.Variable(tf.zeros(shape=[self.output_size]))

        L1 = tf.nn.relu(tf.matmul(self.X, W1) + B1)

        self.Qpred = tf.matmul(L1, W2) + B2

        # 손실 함수 및 최적화 함수
        self.action = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        Q_action = tf.reduce_sum(tf.multiply(self.Qpred, self.action), reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(self.Y - Q_action))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    # 예측한 Q값 구하기
    def predict(self, state):
        x = np.reshape(state, newshape=[1, self.input_size])
        return self.session.run(self.Qpred, feed_dict={self.X: x})

    # e-greedy 를 사용하여 action값 구함
    def egreedy_action(self, epsilon, env, state):
        if np.random.rand(1) < epsilon:
            action = env.action_space.sample()
            #print("Episode: {0}, Action: {1}".format(episode, action))
        else:
            Q_h = self.predict(state)
            action = np.argmax(Q_h)
            #print("Episode: {0}, State: {1}, Q_h: {2}, Action: {3}".format(episode, state, Q_h, action))
        return action

    def update_from_memory(self, batch_size):
        state_batch = np.ndarray(shape=[batch_size, self.input_size])
        action_batch = np.ndarray(shape=[batch_size, self.output_size])

        minibatch = random.sample(REPLAY_MEMORY, batch_size)
        i = 0
        y_batch = []
        for sample in minibatch:
            state, action, reward, new_state, done = sample         # unpacking

            if done:
                y_batch.append(reward)
            else:
                y_batch.append(reward + discount_factor * np.max(self.predict(new_state)))

            one_hot_action = np.zeros(self.output_size) # [0.0, 0.0]
            one_hot_action[action] = 1

            state_batch[i] = state
            action_batch[i] = one_hot_action
            i += 1

        # DQN 알고리즘으로 학습
        loss_value, _ = self.session.run([self.loss, self.optimizer],
                                 feed_dict={self.X: state_batch, self.Y: y_batch, self.action: action_batch})
        return loss_value

def bot_play(DQN, env):
    """
    See our trained network in action
    """
    state = env.reset()
    reward_sum = 0
    done = False
    while not done:
        env.render()
        action = np.argmax(DQN.predict(state))
        new_state, reward, done, info = env.step(action)
        reward_sum += reward
        state = new_state

    return reward_sum

def draw_error_values():
    fig = plt.figure(figsize=(20, 5))
    plt.ion()
    ax1 = plt.subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(episode_list[0:], train_error_list[0:], 'r', label='Train Error Values')
    ax2.plot(episode_list[0:], actions_list[0:], 'b', label='Number of Actions')
    ax1.set_ylabel('Train Error Values', color='r')
    ax2.set_ylabel('Number of Actions', color='b')
    ax1.set_xlabel('Episodes')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()
    input("Press Enter to close the trained error figure...")
    plt.close(fig)

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    input_size = env.observation_space.shape[0]     # 4
    output_size = env.action_space.n                # 2

    # 미니배치 - 꺼내서 사용할 리플레이 갯수
    BATCH_SIZE = 32

    with tf.Session() as sess:
        # DQN 클래스의 mainDQN 인스턴스 생성
        mainDQN = DQN(sess, input_size, output_size)

        # 변수 초기화
        init = tf.global_variables_initializer()
        sess.run(init)

        for episode in range(max_episodes):
            epsilon = INITIAL_EPSILON
            state = env.reset()
            rAll = 0
            done = False

            while not done:
                epsilon *= 0.99
                # action을 수행함 --> Get new state and reward from environment
                action = mainDQN.egreedy_action(epsilon, env, state)
                new_state, reward, done, _ = env.step(action)

                # state, action, reward, next_state, done 을 메모리에 저장
                REPLAY_MEMORY.append((state, action, reward, new_state, done))

                # 메모리에 10000개 이상의 값이 들어가면 가장 먼저 들어간 것부터 삭제
                if len(REPLAY_MEMORY) > 10000:
                    REPLAY_MEMORY.popleft()

                # REPLAY_MEMORY 크기가 BATCH_SIZE 보다 크면 학습
                if len(REPLAY_MEMORY) > BATCH_SIZE:
                    mean_loss_value = mainDQN.update_from_memory(BATCH_SIZE)

                rAll += reward
                state = new_state

            if len(REPLAY_MEMORY) > BATCH_SIZE:
                episode_list.append(episode)
                train_error_list.append(mean_loss_value)
                actions_list.append(rAll)

            if episode % TEST_PERIOD == 0:
                total_reward = 0
                for i in range(10):
                    total_reward += bot_play(mainDQN, env)

                ave_reward = total_reward / 10
                print("episode: {0}, Epsilon: {1}, Evaluation Average Reward: {2}".format(episode,epsilon, ave_reward))
                if ave_reward >= 200:
                    break

        env.reset()
        env.close()
        draw_error_values()

        input("Press Enter to make the trained bot play...")
        bot_play(mainDQN, env)