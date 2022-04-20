import sys

def get_ppo():
    if sys.version_info >= (3, 8):
        from stable_baselines3 import PPO
        return PPO
    else:
        from stable_baselines import PPO2
        return PPO2