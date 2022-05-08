import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from environment import SeamCarvingEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from seam_carving import draw_seam, remove_seam, get_random_starting_points, add_empty_vertical_lines

def test_dynamic_programming_seam_carving():
    pass

def test_rl_seam_carving():
    pass

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\clocks-fix.jpeg", help="Input image path")
parser.add_argument("-r", "--rloutput", type=str, default="D:\\Source\\seam-carving\\images-out\\rl1.png", help="Output image path")
parser.add_argument("-s", "--seam-carving-output", type=str, default="D:\\Source\\seam-carving\\images-out\\seamcarving1.png", help="Output image path")
parser.add_argument("-m", "--model", type=str, default="3d98e8a2_43", help="PPO model path")

def main(args: argparse.Namespace):
    """Testing stuff"""

    env = SeamCarvingEnv(args.input)
    # env = make_vec_env(lambda: SeamCarvingEnv("../images/clocks-fix.jpeg"), n_envs=4, vec_env_cls=DummyVecEnv)  
    obs = env.reset()
    # dones = [False, False, False, False]
    done = False

    # print(env.current_location)
    env.current_location = 400
    # print(env.current_location)

    # print(env.energy_min)
    # print(env.energy_max)

    model = PPO.load(args.model)

    while not done:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, done, info = env.step(action)
        # env.render()
        # print(rewards)

    # image = env.render_image

    # image = env.render_image
    # plt.imshow(image)
    # plt.show()
    # return

    image = env.image
    path = env.found_path
    new_image = draw_seam(image, path)
    print(f"{env.found_path=} {new_image.shape=}")

    cv2.imwrite(args.rloutput, image)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)