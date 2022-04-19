
import cv2
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from environment import SeamCarvingEnv

def main():
    """Testing stuff"""

    out_path = "../images-out/clocks-env-test2.png"
    env = SeamCarvingEnv("../images/clocks-fix.jpeg")

    obs = env.reset()
    done = False

    print(env.current_location)
    env.current_location = 70
    print(env.current_location)

    print(env.energy_min)
    print(env.energy_max)

    # cv2.imwrite("../images-out/clocks-env-test10.png", env.observation_image_full)
    # cv2.imwrite("../images-out/clocks-env-test11.png", env.observation_image_50)
    # cv2.imwrite("../images-out/clocks-env-test12.png", env.observation_image_100)

    while not done:
        action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)
        env.render()

    image = env.render_image
    cv2.imwrite(out_path, image)

if __name__ == "__main__":
   main()