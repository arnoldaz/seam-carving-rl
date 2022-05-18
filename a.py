import sys
import cv2
import gym
from matplotlib import pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from environment_experiment1 import SeamCarvingEnv

env = SeamCarvingEnv("images/clocks-scaled.png")
obs = env.reset()
done = False

env.current_location = 50

# model = PPO.load("72af20d4_16")

while not done:
    # action, _states = model.predict(obs, deterministic=False)
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    env.render()
    print(rewards)

plt.imshow(env.render_img)
plt.show()