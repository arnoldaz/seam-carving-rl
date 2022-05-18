import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import time

from environment import SeamCarvingEnv

from seam_carving import draw_fat_seam, draw_seam, remove_seam, get_random_starting_points, add_empty_vertical_lines

def test_dynamic_programming_seam_carving():
    pass

def test_rl_seam_carving():
    pass

parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\4k-plane.jpg", help="Input image path")
# parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\Broadway_tower_edit.jpg", help="Input image path")
parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\clocks-scaled.png", help="Input image path")
# parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\balloons-scaled.jpg", help="Input image path")
# parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\clocks-fix.jpeg", help="Input image path")
# parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\4k-plane.jpg", help="Input image path")
parser.add_argument("-r", "--rloutput", type=str, default="D:\\Source\\seam-carving\\images-out\\zzzzzzzzzzzzzzzzzzzzzzzzz.png", help="Output image path")
parser.add_argument("-s", "--seam-carving-output", type=str, default="D:\\Source\\seam-carving\\images-out\\seamcarving1.png", help="Output image path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\balloons-scaled_5cde0973_FINAL", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\exp3_b0c51f66_FINAL", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\clocks-scaled-og-env_1f61fbd6_47", help="PPO model path")
parser.add_argument("-m", "--model", type=str, default=".agents-exp\\clocks-scaled-og-env_1f61fbd6_35", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\14278925_13", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\scaled-exp2_21765320_FINAL", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\clocks-full_b7dccffc_17", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\scaled-exp2-v2_19e5747e_FINAL", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default="clean\\3d98e8a2_43", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default="clean\\e110535a_20", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\2489248c_FINAL", help="PPO model path") # castle good
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\scaled-exp2-v2_19e5747e_15", help="PPO model path") # castle good
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\2489248c_FINAL", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\a73fd37a_50", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\10k-8k_16597ade_FINAL", help="PPO model path")

def main(args: argparse.Namespace):
    """Testing stuff"""

    start = time.time()

    env = SeamCarvingEnv(args.input)
    # env = make_vec_env(lambda: SeamCarvingEnv("../images/clocks-fix.jpeg"), n_envs=4, vec_env_cls=DummyVecEnv)  
    obs = env.reset()
    # dones = [False, False, False, False]
    done = False

    # # print(env.current_location)

    
    # env.current_location = 950
    # env.found_path[0] = env.current_location
    # print(env.current_location)

    # print(env.energy_min)
    # print(env.energy_max)

    model = PPO.load(args.model)

    reward_sum = 0
    while not done:
        action, _states = model.predict(obs, deterministic=False)
        # action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)
        # env.render()
        # print(f"{rewards=} {reward_sum=}")
        reward_sum += rewards

    print(time.time() - start)

    print(f"{reward_sum=}")
    # image = env.render_image

    # image = env.render_image
    # plt.imshow(image)
    # plt.show()
    # return

    img = cv2.cvtColor(env.image, cv2.COLOR_BGR2RGB)
    # img = env.img
    path = env.found_path
    final_img = draw_seam(img, path)
    cv2.imwrite(args.rloutput, final_img)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)