# -*- coding: utf-8 -*-
# https://github.com/openai/gym/wiki/CartPole-v0
import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt

# 리플레이를 저장할 리스트
REPLAY_MEMORY = []

# 미니배치 - 꺼내서 사용할 리플레이 갯수
MINIBATCH = 50

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

        self.W1 = tf.Variable(tf.truncated_normal(shape=[self.input_size, self.hidden_size], mean=0.0, stddev=1.0))
        self.B1 = tf.Variable(tf.zeros(shape=[self.hidden_size]))
        self.W2 = tf.Variable(tf.truncated_normal(shape=[self.hidden_size, self.output_size], mean=0.0, stddev=1.0))
        self.B2 = tf.Variable(tf.zeros(shape=[self.output_size]))

        self.L1 = tf.nn.tanh(tf.matmul(self.X, self.W1) + self.B1)

        self.Qpred = tf.matmul(self.L1, self.W2) + self.B2

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

    # 네트워크 학습
    def update(self, state, new_state, action, reward):
        x = np.reshape(state, newshape=[1, self.input_size])
        if done:
            # it's a terminal state
            y = -100
        else:
            # Obtain the Q_y values by feeding the new state through the network
            y = reward + discount_factor * np.max(self.predict(new_state))

        one_hot_action = np.zeros(self.output_size)
        one_hot_action[action] = 1
        one_hot_action = np.reshape(one_hot_action, newshape=[1, self.output_size])

        loss_value, _ = self.session.run([self.loss, self.optimizer], feed_dict={self.X: x, self.Y: [y], self.action: one_hot_action})
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

                loss_value = mainDQN.update(state, new_state, action, reward)

                rAll += reward
                state = new_state

            episode_list.append(episode)
            train_error_list.append(loss_value)
            actions_list.append(rAll)

            if episode % TEST_PERIOD == 0:
                total_reward = 0
                for i in range(10):
                    total_reward += bot_play(mainDQN, env)

                ave_reward = total_reward / 10
                print("episode: {0}, Epsilon: {1}, Evaluation Average Reward: {2}".format(episode,epsilon, ave_reward))
                if ave_reward >= 200:
                    break

            # time.sleep(1) # delays for 1 second between episodes

        env.reset()
        env.close()
        draw_error_values()

        input("Press Enter to make the trained bot play...")
        bot_play(mainDQN, env)