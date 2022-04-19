import gym
from clean.environment import SeamCarvingEnv
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

IMAGE_PATH = "../images/clocks-fix.jpeg"

env = SeamCarvingEnv(IMAGE_PATH)
# env = make_vec_env(lambda: env, n_envs=3)

# check_env(env)

# model = DQN(MlpPolicy, env, verbose=1)
model = PPO.load(".out\\1dba2724_10000000.0_11", env)

obs = env.reset()
# env.current_location = 80
done = False
i = 0

while not done:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, done, info = env.step(action)
    # env.render()

image = env.render_img
plt.imshow(image)
plt.show()

env.close()
