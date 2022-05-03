import sys
import cv2
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from environment import SeamCarvingEnv

env = SeamCarvingEnv("../images/clocks-fix.jpeg")
obs = env.reset()
done = False

env.current_location = 10

# model = PPO.load("72af20d4_16")

while not done:
    # action, _states = model.predict(obs, deterministic=False)
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    env.render()
    print(rewards)


cv2.imwrite("../images-out/ONE_LINE2.png", env.render_image)