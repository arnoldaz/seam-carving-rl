import gym
from env import SeamCarvingEnv
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import ACER, DQN, PPO2
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env


TOTAL_TIMESTEPS = 1e6

env = SeamCarvingEnv("./images/clocks.jpeg")
env = make_vec_env(lambda: env, n_envs=3)

# check_env(env)

# model = DQN(MlpPolicy, env, verbose=1)
model = ACER.load(f"agents-out-v2\\33ebb653_20000000.0_35", env)

obs = env.reset()
env.current_location = 100
done = False
i = 0

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

image = env.envs[1].render_img
plt.imshow(image)
plt.show()

env.close()
