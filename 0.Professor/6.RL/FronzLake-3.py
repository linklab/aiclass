# Dummy Q-Table learning algorithm
from __future__ import print_function

import gym
from gym.envs.registration import register
import numpy as np
import random
import matplotlib.pyplot as plt

register(
    id = 'FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={
        'map_name': '4x4',
        'is_slippery': True # <-- Updated for ver.3
    }
)

def rargmax(vector):
    # vector: [ 0.  1.  1.  0.]
    # Return the maximum number of an array element.
    m = np.amax(vector)     # m = 1.
    # Return the list of indices of the elements that are non-zero and the given condition is True
    indices = np.nonzero(vector == m)[0]   # indices = [1, 2]
    return random.choice(indices)

env = gym.make("FrozenLake-v3")
env.render()

print("env.observation_space.n:", env.observation_space.n)
print("env.action_space.n:", env.action_space.n)
Q = np.zeros([env.observation_space.n, env.action_space.n])

#Discount Factor
discount_factor = .99   # <-- Updated for ver.2
num_episodes = 2000
learning_rate = 0.85     # <-- Updated for ver.3

# list to contain total rewards and steps per episode
rList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    #e = 0.1 / (i + 1)
    e = 1. / ((i / 50) + 10)
    done = False

    # The Q-Table learning algorithm
    while not done:
        #Decaying Random Noise <-- Updated for ver.2
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = rargmax(Q[state, :])

        #Decaying Random Noise <-- Updated for ver.2
        #action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))

        # Get new state and reward from environment
        new_state, reward, done, info = env.step(action)

        # Update Q-Table with new knowledge using learning rate
        Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[new_state, :])) # <-- Updated for ver.3

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.plot(rList)
plt.ylim(-0.5, 1.5)
plt.show()