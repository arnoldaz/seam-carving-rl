import time
import gym
from env import SeamCarvingEnv
import matplotlib.pyplot as plt


env = SeamCarvingEnv("./images/clocks.jpeg")
env.reset()

done = False
i = 0

while not done:
    print(i)
    i += 1
    
    action = env.action_space.sample()
    obs, reward, done, _  = env.step(action)


image = env.render_img
plt.imshow(image)
plt.show()

env.close()


# for i in range(500):
#     print(i)
#     img.set_data(env.render())

#     obs, reward, done, _  = env.step(env.action_space.sample())
#     if done:
#         break

#     plt.pause(0.01)

# plt.show()