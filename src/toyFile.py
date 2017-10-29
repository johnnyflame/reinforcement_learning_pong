# Used to figure out avaialble controls of an openAI gym env



import gym

env = gym.make('Pong-v0')
env.reset()
print("avalable actions: ",env.action_space)


for _ in range(1000):
    env.render()
    env.step(4) # take a random action


