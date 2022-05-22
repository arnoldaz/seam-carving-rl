import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import time

from environment import SeamCarvingEnv

from seam_carving import calc_img_energy, calc_seam_cost_forward, draw_fat_seam, draw_seam, find_index_seam, get_image_seam_from_location, remove_seam, get_random_starting_points, add_empty_vertical_lines

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\4k-plane.jpg", help="Input image path")
# parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\Broadway_tower_edit.jpg", help="Input image path")
# parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\clocks-scaled.png", help="Input image path")
# parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\ballons2-fixed.jpg", help="Input image path")
# parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\eiffel.jpg", help="Input image path")
# parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\clocks-fix.jpeg", help="Input image path")
# parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\4k-plane.jpg", help="Input image path")
parser.add_argument("-r", "--rloutput", type=str, default="D:\\Source\\seam-carving\\images-out\\zzzzzzzzzzzzzzzzzzzzzzzzz.png", help="Output image path")
parser.add_argument("-s", "--seam-carving-output", type=str, default="D:\\Source\\seam-carving\\images-out\\seamcarving1.png", help="Output image path")
parser.add_argument("-m", "--model", type=str, default=".agents-exp\\eiffel_9c7b0525_11", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\balloons2-fixed_ecda09a6_45", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\balloons-scaled_dade243e_FINAL", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\balloons-scaled_5cde0973_FINAL", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\exp3_b0c51f66_FINAL", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\3d98e8a2_45", help="PPO model path") #clocks big
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\clocks-scaled-og-env_1f61fbd6_47", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\clocks-scaled-og-env_1f61fbd6_35", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\14278925_13", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\scaled-exp2_21765320_FINAL", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\clocks-full_b7dccffc_17", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\scaled-exp2-v2_19e5747e_FINAL", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default="clean\\3d98e8a2_43", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default="clean\\e110535a_20", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\2489248c_FINAL", help="PPO model path") # castle good
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\scaled-exp2-v2_19e5747e_15", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\2489248c_FINAL", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\a73fd37a_50", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\10k-8k_16597ade_FINAL", help="PPO model path")

def main(args: argparse.Namespace):
    """Testing seam carving agents result."""

    start = time.time()

    # width = 787
    # width = 1428
    # width = 750
    width = 7633
    locations = []
    for i in range(10, width - 10, int((width - 10) / 14.5)):
        locations.append(i)

    print(locations)

    og_img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    rotated_img = cv2.flip(og_img, 0)
    energy_map = calc_img_energy(rotated_img)
    energy_map_forward, backtrack = calc_seam_cost_forward(energy_map)

    rewards_sum_sum = 0
    for location in locations:
        env = SeamCarvingEnv(args.input)
        obs = env.reset()
        done = False
        
        env.current_location = location
        env.found_path[0] = env.current_location

        min_seam, cost = find_index_seam(energy_map_forward, backtrack, location)
        seam = min_seam[::-1]

        # print(f"{seam=}")

        reward_sum = 0
        i = 0
        current_location = location
        next_location = seam[1]
        while not done:
            if next_location < current_location:
                action = 0
            elif next_location == current_location:
                action = 1
            elif next_location > current_location:
                action = 2

            obs, rewards, done, info = env.step(action)
            reward_sum += rewards

            if done:
                continue

            i += 1
            current_location = next_location
            next_location = seam[i + 1]


        print(time.time() - start)
        print(f"{reward_sum=}")
        rewards_sum_sum += reward_sum

        # img = cv2.cvtColor(env.image, cv2.COLOR_BGR2RGB)
        # path = env.found_path
        # final_img = draw_seam(img, path)
        # plt.imshow(final_img.astype(np.uint8))
        # plt.show()

    print(f"{rewards_sum_sum=}")
    final_avg = rewards_sum_sum / len(locations)
    print(f"{final_avg=}")

    # img = cv2.cvtColor(env.image, cv2.COLOR_BGR2RGB)
    # # img = env.image
    # path = env.found_path
    # final_img = draw_seam(img, path)
    # cv2.imwrite(args.rloutput, final_img)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)