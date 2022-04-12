import gym
from new.env import SeamCarvingEnv
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import ACER, DQN, PPO2
from stable_baselines.common.env_checker import check_env
from stable_baselines.common import make_vec_env


TOTAL_TIMESTEPS = 1e6

env = SeamCarvingEnv("./images/clocks-fix.jpeg")
env = make_vec_env(lambda: env, n_envs=1)

# check_env(env)

# model = DQN(MlpPolicy, env, verbose=1)
model = PPO2.load(f"agents-out-v2\\64a98684_20000000.0_41", env)

obs = env.reset()
# env.current_location = 130
done = False
i = 0

while not done:
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    # env.render()

image = env.envs[0].render_img
plt.imshow(image)
plt.show()

env.close()
