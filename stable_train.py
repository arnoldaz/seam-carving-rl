import gym
from env import SeamCarvingEnv
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2


TOTAL_TIMESTEPS = 1e6

env = SeamCarvingEnv("./images/clocks.jpeg")

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=int(TOTAL_TIMESTEPS))

model.save(f".out\\test_{TOTAL_TIMESTEPS}")

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, _  = env.step(action)

image = env.render_img
plt.imshow(image)
plt.show()

env.close()
