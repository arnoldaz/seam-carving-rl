import cv2
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from environment import SeamCarvingEnv

env = SeamCarvingEnv("../images/clocks-fix.jpeg")
obs = env.reset()
done = False

env.current_location = 120
i = 0

while not done:
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    # env.render()
    print(rewards)
    i += 1
    if i == 50:
        cv2.imwrite("../images-out/clocks-env-test4.png", env.get_observations()[0])


cv2.imwrite("../images-out/clocks-env-test5.png", env.render_image)