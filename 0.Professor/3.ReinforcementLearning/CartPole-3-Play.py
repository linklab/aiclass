# -*- coding: utf-8 -*-
# https://github.com/openai/gym/wiki/CartPole-v0
# Must read - https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
import tensorflow as tf
import gym
import numpy as np

# 하이퍼파라미터
max_episodes = 10000

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
        with tf.variable_scope(self.net_name):
            # Vanilla Neural Network (Just one hidden layer)
            self.X = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
            self.Y = tf.placeholder(shape=[None], dtype=tf.float32)

            W1 = tf.Variable(tf.truncated_normal(shape=[self.input_size, self.hidden_size], mean=0.0, stddev=1.0))
            B1 = tf.Variable(tf.zeros(shape=[self.hidden_size]))
            W2 = tf.Variable(tf.truncated_normal(shape=[self.hidden_size, self.output_size], mean=0.0, stddev=1.0))
            B2 = tf.Variable(tf.zeros(shape=[self.output_size]))

            L1 = tf.nn.relu(tf.matmul(self.X, W1) + B1)

            self.Qpred = tf.matmul(L1, W2) + B2

    # 예측한 Q값 구하기
    def predict(self, state):
        x = np.reshape(state, newshape=[1, self.input_size])
        return self.session.run(self.Qpred, feed_dict={self.X: x})


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


def restoreModel(session, path='./cartpole.ckpt'):
    tf.train.Saver().restore(sess=session, save_path=path)
    print("Model restored successfully.")


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    input_size = env.observation_space.shape[0]     # 4
    output_size = env.action_space.n                # 2

    # 미니배치 - 꺼내서 사용할 리플레이 갯수
    BATCH_SIZE = 32

    with tf.Session() as sess:
        # DQN 클래스의 mainDQN 인스턴스 생성
        mainDQN = DQN(sess, input_size, output_size, name='main')
        restoreModel(sess, "./cartpole.ckpt")

        for episode in range(max_episodes):
            bot_play(mainDQN, env)

        env.reset()
        env.close()