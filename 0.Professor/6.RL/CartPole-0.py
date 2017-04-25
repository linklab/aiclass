import gym
import os

env = gym.make('CartPole-v0')
env.reset()
episode = 0
reward_sum = 0
num_action = 1

while episode < 10:
    env.render()
    action = env.action_space.sample()
    new_state, reward, done, info = env.step(action)
    print("Action {0}: {1} --> State: {2}, Reward: {3}, Done: {4}, Info: {5}".format(num_action, action, new_state, reward, done, info))
    reward_sum += reward
    num_action += 1
    if done:
        print("Total reward for this episode {0} was: {1}".format(episode, reward_sum))
        reward_sum = 0
        env.reset()
        input("Press Enter to continue...")
        episode += 1
        num_action = 1