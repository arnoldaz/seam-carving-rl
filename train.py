import argparse
import uuid
from pathlib import Path
import time

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from legacy.versioning import get_ppo

from environment import SeamCarvingEnv

RANDOM_GUID = str(uuid.uuid4())[:8]
AGENTS_DIR = Path(".agents")
TENSORBOARD_DIR = Path(".tensorboard")


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", type=str, default="PPO", help="Chosen algorithm")
parser.add_argument("-s", "--steps", type=int, default=int(1e7), help="Training steps amount")
parser.add_argument("-p", "--period", type=int, default=int(2e5), help="Save period amount")
parser.add_argument("-n", "--n_env", type=int, default=4, help="Default n_env count")
parser.add_argument("-i", "--image", type=str, default="D:\\Source\\seam-carving\\images\\clocks-fix.jpeg", help="Environment image path")
parser.add_argument("-v", "--vec_env", action=argparse.BooleanOptionalAction, default=True, help="Use SubprocVecEnv")
parser.add_argument("--name", type=str, default="noname", help="Name")

def get_tensorboard_dir(name):
    path = TENSORBOARD_DIR / f"{name}_{RANDOM_GUID}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_agent_dir(post_str, name):
    path = AGENTS_DIR / f"{name}_{RANDOM_GUID}_{str(post_str)}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def create_model(algorithm_name, image_path, n_envs, vec_env, name):
    print(f"{vec_env=}")
    env = make_vec_env(lambda: SeamCarvingEnv(image_path), n_envs=n_envs, vec_env_cls=SubprocVecEnv if vec_env else DummyVecEnv)   
    model = None

    match algorithm_name:
        case "PPO":
            model = PPO("MlpPolicy", env, tensorboard_log=get_tensorboard_dir(name), verbose=1, batch_size=256, n_steps=1024)
        case "A2C":
            model = A2C("MlpPolicy", env, tensorboard_log=get_tensorboard_dir(name), verbose=1)
        case "DQN":
            model = DQN("MlpPolicy", env, tensorboard_log=get_tensorboard_dir(name), verbose=1)
        case _:
            raise Exception("Bad algorithm name")
            
    return model

def train_model(model: PPO | A2C | DQN, steps, save_period, name):
    start = time.time()
    for i in range(int(steps / save_period)):
        model.learn(total_timesteps=int(save_period), reset_num_timesteps=False)
        model.save(get_agent_dir(i, name))

        current_steps = (i + 1) * save_period
        print("Saved {} {}".format(current_steps, time.time() - start))

    model.save(get_agent_dir("FINAL", name))
    print(f"oh no {time.time() - start} {get_tensorboard_dir(name)}")

def main(args: argparse.Namespace):
    print(f"Passed params: {args.algorithm=} {args.image=} {args.n_env=} {args.vec_env=} {args.steps=} {args.period=} {args.name=}")

    # model = create_model(args.algorithm, args.image, args.n_env, args.vec_env, args.name)
    env = make_vec_env(lambda: SeamCarvingEnv(args.image), n_envs=args.n_env, vec_env_cls=SubprocVecEnv if args.vec_env else DummyVecEnv)
    model = PPO.load(".agents-exp\\balloons-scaled_ac688813_4", env)
    train_model(model, args.steps, args.period, args.name)
    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)