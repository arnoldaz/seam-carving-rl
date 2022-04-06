import sys
import gym
import numpy as np
from env import SeamCarvingEnv
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines import ACER, DQN, PPO2, SAC, HER
from stable_baselines.deepq.policies import MlpPolicy as DqnMlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.bit_flipping_env import BitFlippingEnv

import uuid
import os

RANDOM_GUID = str(uuid.uuid4())[:8]

EVALUATE_DIR = "evaluation-out"
SAVE_DIR = "agents-out-v2"
TENSORBOARD_DIR = "tensorboard-out-v2"
STEPS = 2e7
SAVE_PERIOD = 500000


def get_tensorboard_dir(env_name: str):
    full_dir = "{}/{}_{}".format(TENSORBOARD_DIR, env_name, str(STEPS))

    if not os.path.exists(full_dir):
        os.mkdir(full_dir)

    return full_dir

def get_agent_dir(env_name: str, i: int):
    full_dir = "{}/{}_{}_{}".format(SAVE_DIR, env_name, str(STEPS), str(i))
    return full_dir


env = SeamCarvingEnv("./images/clocks.jpeg")
env = make_vec_env(lambda: env, n_envs=3)

# kwargs = { "img_path": "./images/clocks.jpeg" }
# env = make_vec_env(SeamCarvingEnv, n_envs=4, env_kwargs=kwargs)

# model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=get_tensorboard_dir(RANDOM_GUID), full_tensorboard_log=False)


# policy_kwargs = { "n_env": 1, "n_steps": 600, "n_batch": 600 }
# model_class = DQN
# goal_selection_strategy = 'future'
# env = BitFlippingEnv(continuous=False, max_steps=600)


model = ACER(CnnPolicy, env, verbose=1, tensorboard_log=get_tensorboard_dir(RANDOM_GUID), full_tensorboard_log=False)

for i in range(int(STEPS / SAVE_PERIOD)):
    model.learn(total_timesteps=int(SAVE_PERIOD), reset_num_timesteps=False)

    model.save(get_agent_dir(RANDOM_GUID, i))
    current_steps = (i + 1) * SAVE_PERIOD

    print("Saved {}".format(current_steps))

print(f"Finished learning after '{STEPS}' steps")

out_file = f".out\\test_{STEPS}_{RANDOM_GUID}_{env.obs_width}"
model.save(out_file)

print(f"Output saved at '{out_file}'")

# obs = env.reset()
# done = False

# while not done:
#     action = env.action_space.sample()
#     obs, reward, done, _  = env.step(action)

# image = env.render_img
# plt.imshow(image)
# plt.show()

env.close()
