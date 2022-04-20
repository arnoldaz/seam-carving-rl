import argparse
import uuid
from pathlib import Path

from stable_baselines import A2C, DQN, PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from environment import SeamCarvingEnv

RANDOM_GUID = str(uuid.uuid4())[:8]
AGENTS_DIR = Path(".agents")
TENSORBOARD_DIR = Path(".tensorboard")


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", type=str, default="PPO", help="Chosen algorithm")
parser.add_argument("-s", "--steps", type=int, default=int(2e7), help="Training steps amount")
parser.add_argument("-p", "--period", type=int, default=int(5e5), help="Save period amount")
parser.add_argument("-n", "--n_env", type=int, default=4, help="Default n_env count")
parser.add_argument("-i", "--image", type=str, default="D:\\Source\\seam-carving\\images\\clocks-fix.jpeg", help="Environment image path")

def get_tensorboard_dir():
    path = TENSORBOARD_DIR / RANDOM_GUID
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_agent_dir(post_str):
    path = AGENTS_DIR / f"{RANDOM_GUID}_{str(post_str)}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def create_model(algorithm_name, image_path, n_envs):
    env = make_vec_env(lambda: SeamCarvingEnv(image_path), n_envs=n_envs, vec_env_cls=DummyVecEnv)   
    model = PPO2("MlpPolicy", env, tensorboard_log=get_tensorboard_dir(), verbose=1, full_tensorboard_log=False)

    return model

def train_model(model: PPO2, steps, save_period):
    for i in range(int(steps / save_period)):
        model.learn(total_timesteps=int(save_period), reset_num_timesteps=False)
        model.save(str(get_agent_dir(i)))

        current_steps = (i + 1) * save_period
        print("Saved {}".format(current_steps))

    model.save(str(get_agent_dir("FINAL")))

def main(args: argparse.Namespace):
    model = create_model(args.algorithm, args.image, args.n_env)
    train_model(model, args.steps, args.period)
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)