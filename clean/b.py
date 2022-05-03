
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from environment import SeamCarvingEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def main():
    """Testing stuff"""
    out_path = "../images-out/clocks-env-test999.png"

    env = SeamCarvingEnv("../images/4k-plane.jpeg")

    print(time.process_time() - start)

    return

    obs = env.reset()
    done = False

    # print(env.current_location)
    # env.current_location = 120
    # print(env.current_location)

    # print(env.energy_min)
    # print(env.energy_max)

    model = PPO.load("e110535a_20")

    while not done:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, done, info = env.step(action)
        # env.render()
        print(rewards)

    image = env.render_image
    cv2.imwrite(out_path, image)

if __name__ == "__main__":
    main()