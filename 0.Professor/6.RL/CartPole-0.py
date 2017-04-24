import gym
env = gym.make('CartPole-v0')
env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 10:
    env.render()
    action = env.action_space.sample()
    new_state, reward, done, info = env.step(action)
    print("Action: {0} --> State: {1}, Reward: {2}, Done: {3}, Info: {4}".format(action, new_state, reward, done, info))
    reward_sum += reward
    if done:
        random_episodes += 1
        print("Reward for this episode was: {0}".format(reward_sum))
        reward_sum = 0
        env.reset()
