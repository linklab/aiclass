# https://github.com/openai/gym/wiki/CartPole-v0
import gym
import tensorflow as tf
import numpy as np
import time

env = gym.make('CartPole-v0')
env.reset()
episode = 0
num_actions = 0

input_size = env.observation_space.shape[0]     # 4
output_size = env.action_space.n                # 2
learning_rate = 0.1

print("input_size:", input_size)
print("output_size:", output_size)

# These lines establish the feed-forward part of the network used to choose actions
X = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

W = tf.Variable(tf.random_normal([input_size, output_size], mean=0.0, stddev=1.0))
Qpred = tf.matmul(X, W) # (1, 4) * (4, 2) = (1, 2)

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Set Q-learning related parameters
discount_factor = .99
num_episodes = 2000

# list to contain total rewards and steps per episode
rList = []

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for episode in range(num_episodes):
        # Reset environment and get first new observation
        state = env.reset()
        rAll = 0
        e = 1. / ((episode / 50) + 10)
        done = False
        num_actions = 0

        # The Q-Network training
        while not done:
            x = np.reshape(state, [1, input_size])
            Qs = sess.run(Qpred, feed_dict={X: x})
            print("Episode: {0}, State: {1}, Qs: {2}".format(episode, state, Qs))
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qs)

            # Get new state and reward from environment
            new_state, reward, done, info = env.step(action)

            if done:
                # Update Q, and no Qs+1, since it's a terminal state
                Qs[0, action] = -100
            else:
                x1 = np.reshape(new_state, [1, input_size])
                # Obtain the Qs1 values by feeding the new state through the network
                Qs1 = sess.run(Qpred, feed_dict={X: x1})
                Qs[0, action] = reward + discount_factor * np.max(Qs1)

            sess.run(train, feed_dict={X: x, Y: Qs})
            rAll += reward
            num_actions += 1
            state = new_state

        rList.append(rAll)
        print("Episode: {0}, Total Step: {1}, Total Rewards: {2}".format(episode, num_actions, rAll))
        time.sleep(1) # delays for 1 second

        # If the last 10's average steps are over 50, it's good enough
        if len(rList) > 10 and np.mean(rList[-10:]) > 50:
            break

    # See our trained network in action
    state = env.reset()
    reward_sum = 0
    num_actions = 0
    while True:
        env.render()

        x = np.reshape(state, [1, input_size])
        Qs = sess.run(Qpred, feed_dict={X: x})
        action = np.argmax(Qs)
        num_actions += 1
        state, reward, done, info = env.step(action)
        reward_sum += reward

        if done:
            print("=============================================")
            print("Total number of actions: {0}, Total rewards: {1}".format(num_actions, reward_sum))
            break