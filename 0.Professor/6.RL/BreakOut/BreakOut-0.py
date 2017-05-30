# https://github.com/openai/gym/wiki/CartPole-v0
import gym
import numpy as np
import time

env = gym.make('Breakout-v0')
env.reset()
episode = 0
reward_sum = 0
num_actions = 0
action_desc = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
#The three FIRE actions, 'FIRE', 'RIGHTFIRE', 'LEFTFIRE', make the game start for human player, but this is unnecessary in learning procedure.

while episode < 10:
    env.render()
    action = env.action_space.sample()
    num_actions += 1
    new_state, reward, done, info = env.step(action)
    print("Action {0}: {1} --> Reward: {2}, Done: {3}, Info: {4}".format(num_actions, action_desc[action], reward, done, info))
    print("New State:", new_state.shape, np.sum(new_state, axis=(0, 1)))
    #print("New State:", new_state.shape, new_state)
    reward_sum += reward
    time.sleep(1)

    if done:
        print("Total reward for this episode {0} was: {1}".format(episode, reward_sum))
        reward_sum = 0
        env.reset()
        input("Press Enter to continue...")
        episode += 1
        num_actions = 1