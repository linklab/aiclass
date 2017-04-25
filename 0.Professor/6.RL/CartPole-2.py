import gym
import tensorflow as tf
import numpy as np
import time
from collections import deque

class DQN:
    def __init__(self, session, input_size, hidden_size, output_size):
        self.session = session
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = 0.1
        self.discount_factor = .99
        self.build_network()
        print("input_size:", self.input_size)
        print("hidden_size:", self.hidden_size)
        print("output_size:", self.output_size)

    def build_network(self):
        self.X = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
        self.Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

        self.W1 = tf.Variable(tf.random_normal([self.input_size, self.hidden_size], mean=0.0, stddev=1.0))
        self.U1 = tf.matmul(self.X, self.W1)        # (1, 4) * (4, 16) = (1, 16)
        self.Z1 = tf.nn.relu(self.U1)               # (1, 16)

        self.W2 = tf.Variable(tf.random_normal([self.hidden_size, self.output_size], mean=0.0, stddev=1.0))
        self.Qpred = tf.matmul(self.Z1, self.W2)    # (1, 16) * (16, 2) = (1, 2)

        self.loss = tf.reduce_sum(tf.square(self.Y - self.Qpred))
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def predict(self, episode, state):
        x = np.reshape(state, [1, input_size])
        Q = self.session.run(self.Qpred, feed_dict={self.X: x})
        return Q

    def update(self, episode, x_stack, y_stack):
        loss_value, _ = self.session.run([self.loss, self.train], feed_dict={self.X: x_stack, self.Y: y_stack})
        if episode % 10 == 0:
            print("Episode: {0}, Loss Value: {1}".format(episode, loss_value))
        return loss_value

    def replay_train(self, episode, train_batch):
        x_stack = np.empty(0).reshape(0, self.input_size)
        y_stack = np.empty(0).reshape(0, self.output_size)

        for state, action, reward, next_state, done in train_batch:
            Q = self.predict(episode, state)

            if done:
                Q[0, action] = reward
            else:
                Q[0, action] = reward + self.discount_factor * np.max(self.predict(new_state))

            x_stack = np.vstack([x_stack, state])
            y_stack = np.vstack([y_stack, Q])

        return self.update(episode, x_stack, y_stack)

    def bot_play(self, main_dqn):
        state = env.reset()
        reward_sum = 0
        num_actions = 0
        while True:
            env.render()
            action = self.predict(0, state)
            num_actions += 1
            state, reward, done, info = env.step(action)
            reward_sum += reward

            if done:
                print("=============================================")
                print("Total number of actions: {0}, Total rewards: {1}".format(num_actions, reward_sum))
                break



if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    env.reset()
    num_episodes = 2000
    REPLAY_MEMORY = 50000
    replay_buffer = deque()
    rList = []

    input_size = env.observation_space.shape[0]     # 4
    output_size = env.action_space.n                # 2

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        main_dqn = DQN(sess, input_size, 16, output_size)

        for episode in range(num_episodes):
            # Reset environment and get first new observation
            state = env.reset()
            rAll = 0
            e = 1. / ((episode / 50) + 10)
            done = False
            num_actions = 0

            # The Q-Network training
            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    Qs = main_dqn.predict(episode, state)
                    action = np.argmax(Qs)
                    print("Episode: {0}, State: {1}, Qs: {2}".format(episode, state, Qs))

                # Get new state and reward from environment
                new_state, reward, done, info = env.step(action)

                if done:
                    # Big Penalty
                    reward = -100

                replay_buffer.append((state, action, reward, new_state, done))
                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = new_state
                num_actions += 1
                if num_actions > 10000:
                    break
                rAll += reward

            rList.append(rAll)
            print("Episode: {0}, Total Step: {1}, Total Rewards: {2}".format(episode, num_actions, rAll))
            if num_actions > 10000:
                break

            if episode % 10 == 1:
                #Get a random batch of experiences.
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 10)
                    loss, _ = main_dqn.replay_train(episode, minibatch)
                print("Loss: {0}".format(loss))

            time.sleep(1) # delays for 1 second

            # If the last 10's average steps are over 70, it's good enough
            if len(rList) > 10 and np.mean(rList[-10:]) > 50:
                break

