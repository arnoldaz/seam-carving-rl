import cv2
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines import PPO2
from environment import SeamCarvingEnv
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

def main():
    """Testing stuff"""

    out_path = "../images-out/clocks-env-test7.png"
    env = SeamCarvingEnv("../images/clocks-fix.jpeg")
    # env = make_vec_env(lambda: SeamCarvingEnv("../images/clocks-fix.jpeg"), n_envs=4, vec_env_cls=DummyVecEnv)  
    obs = env.reset()
    # dones = [False, False, False, False]
    done = False

    # print(env.current_location)
    # env.current_location = 120
    # print(env.current_location)

    # print(env.energy_min)
    # print(env.energy_max)

    model = PPO2.load("8a51e22e_FINAL")

    while not done:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, done, info = env.step(action)
        # env.render()
        print(rewards)

    image = env.render_image
    cv2.imwrite(out_path, image)

if __name__ == "__main__":
    main()