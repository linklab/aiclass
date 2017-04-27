# https://github.com/openai/gym/wiki/CartPole-v0
import gym

env = gym.make('CartPole-v0')
env.reset()
episode = 0
reward_sum = 0
num_actions = 0

# x : -0.061586
# θ : -0.75893141
# dx/dt : 0.05793238
# dθ/dt : 1.15547541

while episode < 10:
    env.render()
    action = env.action_space.sample()
    num_actions += 1
    new_state, reward, done, info = env.step(action)
    print("Action {0}: {1} --> State: {2}, Reward: {3}, Done: {4}, Info: {5}".format(num_actions, action, new_state, reward, done, info))
    reward_sum += reward
    if done:
        print("Total reward for this episode {0} was: {1}".format(episode, reward_sum))
        reward_sum = 0
        env.reset()
        input("Press Enter to continue...")
        episode += 1
        num_actions = 1