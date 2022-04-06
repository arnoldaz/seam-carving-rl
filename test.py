import sys
import gym
import numpy as np
from env import SeamCarvingEnv
import matplotlib.pyplot as plt

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines import PPO2


TOTAL_TIMESTEPS = 5e6

env = SeamCarvingEnv("./images/clocks.jpeg")

# img_energy = env.img_energy
_, img_energy = np.mgrid[0:500, 0:500]
# print(grid)

line_count = len(img_energy)
line_length = len(img_energy[0])

print(line_count)
print(line_length)

current_location = 498
current_line = 10

max_obs = np.full((line_count, 1), 256, dtype=np.int)

# a = np.arange(1, 900)
# print(a[(current_location - 4):(current_location + 3)])

low = current_location - 3
high = current_location + 4

min_location = max(low, 0)
max_location = min(high, line_length)

print("min")
print(min_location)
print(max_location)
print("min")
print(low)
print(high)


lines_data = img_energy[current_line:, min_location:max_location]
print(lines_data.shape)
print(lines_data)

x_offset = -low if low < 0 else 0

new = np.zeros((line_count, 7), dtype=np.int)
new[ :lines_data.shape[0], x_offset : lines_data.shape[1] + x_offset] = lines_data

# zero_obs = np.zeros(len(lines_data[0]), dtype=np.int)
# while not len(lines_data) == line_count:
#     lines_data = np.concatenate((lines_data, [zero_obs]))

print(new.shape)
print(new)

# lines_data = np.reshape(lines_data, (line_count, 7))

# np.pad(lines_data, )

# while not len(lines_data[0]) == line_length:
#     for index, line in enumerate(lines_data):
#         print(lines_data[index])
#         # lines_data[index] = np.concatenate(([256], line))
#         lines_data[index] = np.insert(line, 0, 256)

# print(lines_data.shape)
# print(lines_data)

# print(env.img_energy)
# plt.imshow(env.img_energy)
# plt.show()

sys.exit()

# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=int(TOTAL_TIMESTEPS))

# model.save(f".out\\test_{TOTAL_TIMESTEPS}")

# obs = env.reset()
# done = False

# while not done:
#     action = env.action_space.sample()
#     obs, reward, done, _  = env.step(action)

# image = env.render_img
# plt.imshow(image)
# plt.show()

# env.close()
