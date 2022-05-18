import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from environment import SeamCarvingEnv
from seam_carving import draw_seam, remove_seam, get_random_starting_points, add_empty_vertical_lines

parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\4k-plane.jpg", help="Input image path")
# parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\clocks-fix.jpeg", help="Input image path")
parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images-out\\temp7\\abc1_149.png", help="Input image path")
# parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\Broadway_tower_edit.jpg", help="Input image path")
parser.add_argument("-r", "--rl-output", type=str, default="D:\\Source\\seam-carving\\images-out\\rl5.png", help="Output image path")
parser.add_argument("-s", "--seam-carving-output", type=str, default="D:\\Source\\seam-carving\\images-out\\seamcarving1.png", help="Output image path")
# parser.add_argument("-m", "--model", type=str, default="clean\\3d98e8a2_43", help="PPO model path") # castle full img
parser.add_argument("-m", "--model", type=str, default="clean\\3d98e8a2_43", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\8d029c42_FINAL", help="PPO model path")
# parser.add_argument("-m", "--model", type=str, default=".agents-exp\\2489248c_FINAL", help="PPO model path")

def main(args: argparse.Namespace):
    """Testing stuff"""

    env = SeamCarvingEnv(args.input)
    obs = env.reset()
    done = False

    start_seam_locations = get_random_starting_points(env.image_width, 150, 123)
    print(f"{start_seam_locations=}")

    model = PPO.load(args.model)

    max_possible_energy = 100 * env.image_height
    passable_rewards = max_possible_energy * 0.97
    try_count = 3

    i = 150
    for start_seam_location in start_seam_locations:
        env.current_location = start_seam_location
        env.found_path[0] = start_seam_location
        print(f"{i} {start_seam_location=}")
        image = env.image[:]

        tries_left = try_count
        path = None
        current_paths = []
        while tries_left > 0:
            rewards_sum = 0
            while not done:
                action, _states = model.predict(obs, deterministic=False)
                obs, rewards, done, info = env.step(action)
                rewards_sum += rewards

            print(f"{i} {rewards_sum=}")
            if rewards_sum >= passable_rewards:
                print(f"{i} Passable path found: {rewards_sum=}")
                path = env.found_path
                break

            tries_left -= 1
            current_paths.append({ "reward": rewards_sum, "path": env.found_path })

            env = SeamCarvingEnv(env.image, block_right_lines=i)
            env.reset()
            # env.current_location = start_seam_location
            # env.found_path[0] = start_seam_location
            done = False
            print(f"{i} Reward not enough, looping {rewards_sum=}")

        if tries_left == 0:
            print(f"{i} Max path calculated: {rewards_sum=}")
            path = max(current_paths, key=lambda x:x["reward"])["path"]

        new_image = remove_seam(image, path)
        final_image = add_empty_vertical_lines(new_image, 1)

        out_image = cv2.cvtColor(np.float32(final_image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"D:\\Source\\seam-carving\\images-out\\temp7\\abc1_{i}.png", out_image)
        out_image_with_seam = cv2.cvtColor(np.float32(draw_seam(image, path)), cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"D:\\Source\\seam-carving\\images-out\\temp7\\abc1_seam_{i}.png", out_image_with_seam)

        i += 1
        env = SeamCarvingEnv(final_image, block_right_lines=i)
        obs = env.reset()
        done = False


    recolored_image = cv2.cvtColor(np.float32(env.image), cv2.COLOR_BGR2RGB)
    cropped_image = recolored_image[:,0:(env.image_width - len(start_seam_locations))]
    cv2.imwrite(f"D:\\Source\\seam-carving\\images-out\\temp7\\abc_FINAL.png", cropped_image)

    return
    out_path = "../images-out/clocks-env-test9999.png"
    env = SeamCarvingEnv("../images/clocks-fix.jpeg")
    # env = make_vec_env(lambda: SeamCarvingEnv("../images/clocks-fix.jpeg"), n_envs=4, vec_env_cls=DummyVecEnv)  
    obs = env.reset()
    # dones = [False, False, False, False]
    done = False

    # print(env.current_location)
    env.current_location = 120
    # print(env.current_location)

    # print(env.energy_min)
    # print(env.energy_max)

    model = PPO.load("dfea11e4_8")

    while not done:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, done, info = env.step(action)
        # env.render()
        print(rewards)

    # image = env.render_image

    image = env.image
    path = env.found_path
    new_image = remove_seam(image, path)
    print(f"{new_image=} {new_image.shape=}")

    env = SeamCarvingEnv(new_image)



    cv2.imwrite(out_path, image)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)