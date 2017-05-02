# https://github.com/openai/gym/wiki/CartPole-v0
import gym
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env.reset()
episode = 0

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

# Set Q-learning related parameters
discount_factor = .99
num_episodes = 100

# list to contain total rewards and steps per episode
rList = []
train_error_list = []
episode_list = []

A = tf.placeholder(dtype=tf.int32)
loss = tf.square(Y[0, A] - Qpred[0, A])
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

def bot_play():
    """
        See our trained network in action
    """
    input("Press Enter to make the trained bot play...")
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
            time.sleep(3) # delays for 3 second
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

with tf.Session() as sess:
    # 변수 초기화
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
        while not done and num_actions < 5000:
            env.render()
            x = np.reshape(state, [1, input_size])
            # e-greedy 를 사용하여 action값 구함
            if np.random.rand(1) < e:
                action = env.action_space.sample()
                print("Episode: {0}, Action: {1}".format(episode, action))
            else:
                Q_h = sess.run(Qpred, feed_dict={X: x})
                action = np.argmax(Q_h)
                print("Episode: {0}, State: {1}, Q_h: {2}, Action: {3}".format(episode, state, Q_h, action))

            # action을 수행함 --> Get new state and reward from environment
            new_state, reward, done, _ = env.step(action)

            # Get the label y
            Q_y = [[0.0, 0.0]]
            if done:
                # it's a terminal state
                Q_y[0][action] = -100
            else:
                x_new_state = np.reshape(new_state, [1, input_size])
                # Obtain the Q_y values by feeding the new state through the network
                Q_new_state = sess.run(Qpred, feed_dict={X: x_new_state})
                Q_y[0][action] = reward + discount_factor * np.max(Q_new_state)

            loss_value, _ = sess.run([loss, train], feed_dict={X: x, Y: Q_y, A: action})

            rAll += reward
            num_actions += 1
            state = new_state

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
    bot_play()