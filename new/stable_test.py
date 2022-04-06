import gym
from env import SeamCarvingEnv
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env


env = SeamCarvingEnv("../images/clocks-blank.jpeg")
# env = make_vec_env(lambda: env, n_envs=3)

# check_env(env)

# model = DQN(MlpPolicy, env, verbose=1)
model = PPO.load(".out\\final_20000_5b70e91f", env)

obs = env.reset()
env.current_location = 120
done = False
i = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    # env.render()

image = env.render_img
plt.imshow(image)
plt.show()

env.close()
