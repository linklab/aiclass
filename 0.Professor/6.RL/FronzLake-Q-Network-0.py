# Dummy Q-Table learning algorithm
from __future__ import print_function

import gym
from gym.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

register(
    id = 'FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={
        'map_name': '4x4',
        'is_slippery': True # <-- Updated for ver.3
    }
)

env = gym.make("FrozenLake-v3")
env.render()

# Input and output size based on the Environment
input_size = env.observation_space.n    # 16
output_size = env.action_space.n        # 4
learning_rate = 0.1

print("input_size:", input_size)
print("output_size:", output_size)

# These lines establish the feed-forward part of the network used to choose actions
X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)

W = tf.Variable(tf.random_uniform([input_size, output_size], minval=0, maxval=0.01))
Qpred = tf.matmul(X, W) # (1, 16) * (16, 4) = (1, 4)

loss = tf.reduce_sum(tf.square(Y-Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Set Q-learning related parameters
discount_factor = .99
max_episodes = 2000

# list to contain total rewards and steps per episode
rList = []

def one_hot(x):
    return np.identity(16)[x:x+1]

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(max_episodes):
        # Reset environment and get first new observation
        state = env.reset()
        rAll = 0
        #e = 0.1 / (i + 1)
        e = 1. / ((i / 50) + 10)
        done = False
        local_loss = []

        # The Q-Network training
        while not done:
            # Choose an action greedily (with e chance of random action) from Q-network
            Qs = sess.run(Qpred, feed_dict={X: one_hot(state)})
            print("Episode: {0}, State: {1}, Qs: {2}".format(i, one_hot(state), Qs))
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qs)

            # Get new state and reward from environment
            new_state, reward, done, info = env.step(action)

            if done:
                # Update Q, and no Qs+1, since it's a terminal state
                Qs[0, action] = reward
            else:
                # Obtain the Qs1 values by feeding the new state through the network
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(new_state)})
                Qs[0, action] = reward + discount_factor * np.max(Qs1)

            sess.run(train, feed_dict={X: one_hot(state), Y: Qs})
            rAll += reward
            state = new_state

        rList.append(rAll)

print("Success rate: " + str(sum(rList)/max_episodes))
plt.plot(rList)
plt.ylim(-0.5, 1.5)
plt.show()