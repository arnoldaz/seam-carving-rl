import sys
import gym
import numpy as np
from env import SeamCarvingEnv
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG, DQN, PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import uuid
import os

RANDOM_GUID = str(uuid.uuid4())[:8]

FINAL_SAVE_DIR = ".out"
TENSORBOARD_DIR = ".tensorboard"
STEPS = 1e7
SAVE_PERIOD = 100000

IMAGE_PATH = "../images/clocks-fix.jpeg"

def get_tensorboard_dir(env_name: str):
    full_dir = "{}/{}_{}".format(TENSORBOARD_DIR, env_name, str(STEPS))

    if not os.path.exists(full_dir):
        os.mkdir(full_dir)

    return full_dir

def get_agent_dir(env_name: str, i: int):
    full_dir = "{}/{}_{}_{}".format(FINAL_SAVE_DIR, env_name, str(STEPS), str(i))
    return full_dir

def main():
    # env = SeamCarvingEnv(IMAGE_PATH)
    env = make_vec_env(lambda: SeamCarvingEnv(IMAGE_PATH), n_envs=6, vec_env_cls=SubprocVecEnv)

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=get_tensorboard_dir(RANDOM_GUID))

    for i in range(int(STEPS / SAVE_PERIOD)):
        model.learn(total_timesteps=int(SAVE_PERIOD), reset_num_timesteps=False)

        model.save(get_agent_dir(RANDOM_GUID, i))
        current_steps = (i + 1) * SAVE_PERIOD

        print("Saved {}".format(current_steps))

    print(f"Finished learning after '{STEPS}' steps")

    out_file = f".out\\final_{STEPS}_{RANDOM_GUID}"
    model.save(out_file)

    print(f"Output saved at '{out_file}'")

    env.close()

if __name__ == "__main__":
    main()