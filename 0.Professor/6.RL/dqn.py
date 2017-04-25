import gym
import tensorflow as tf
import numpy as np
import time
from collections import deque


env = gym.make('CartPole-v0')
env.reset()
episode = 0
num_actions = 0

input_size = env.observation_space.shape[0]     # 4
hidden_size = 16                                # 16
output_size = env.action_space.n                # 2
learning_rate = 0.1

print("input_size:", input_size)
print("hidden_size:", hidden_size)
print("output_size:", output_size)

# These lines establish the feed-forward part of the network used to choose actions


# Set Q-learning related parameters
discount_factor = .99
num_episodes = 2000

# list to contain total rewards and steps per episode
rList = []

replay_buffer = deque()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(num_episodes):
        # Reset environment and get first new observation
        state = env.reset()
        rAll = 0
        #e = 0.1 / (i + 1)
        e = 1. / ((i / 50) + 10)
        done = False

        # The Q-Network training
        while not done:
            num_actions += 1
            x = np.reshape(state, [1, input_size])
            Qs = sess.run(Qpred, feed_dict={X: x})
            print("Episode: {0}, State: {1}, Qs: {2}".format(i, state, Qs))
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
            state = new_state

        rList.append(rAll)
        print("Episode: {0}, Total Step: {1}, Total Rewards: {2}".format(i, num_actions, rAll))
        num_actions = 0
        time.sleep(1) # delays for 1 second

        # If the last 10's average steps are over 70, it's good enough
        if len(rList) > 10 and np.mean(rList[-10:]) > 50:
            break

    # See our trained network in action
    state = env.reset()
    reward_sum = 0
    num_actionss = 0
    while True:
        env.render()

        x = np.reshape(state, [1, input_size])
        Qs = sess.run(Qpred, feed_dict={X: x})
        action = np.argmax(Qs)
        num_actionss += 1
        state, reward, done, info = env.step(action)
        reward_sum += reward

        if done:
            print("=============================================")
            print("Total number of actions: {0}, Total rewards: {1}".format(num_actionss, reward_sum))
            break