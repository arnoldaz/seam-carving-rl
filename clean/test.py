import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from environment import SeamCarvingEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from seam_carving import remove_seam, get_random_starting_points, add_empty_vertical_lines

def test_dynamic_programming_seam_carving():
    pass

def test_rl_seam_carving():
    pass

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\clocks-fix.jpeg", help="Input image path")
parser.add_argument("-r", "--rl-output", type=str, default="D:\\Source\\seam-carving\\images-out\\rl1.png", help="Output image path")
parser.add_argument("-s", "--seam-carving-output", type=str, default="D:\\Source\\seam-carving\\images-out\\seamcarving1.png", help="Output image path")
parser.add_argument("-m", "--model", type=str, default="dfea11e4_8", help="PPO model path")

def main(args: argparse.Namespace):
    """Testing stuff"""

    env = SeamCarvingEnv(args.input)
    obs = env.reset()
    start_seam_locations = get_random_starting_points(env.WIDTH, 30, 123)
    done = False

    model = PPO.load(args.model)

    i = 0
    for start_seam_location in start_seam_locations:
        env.current_location = start_seam_location
        env.found_path[0] = start_seam_location
        print(f"{i} {start_seam_location=}")

        rewards_sum = 0
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, done, info = env.step(action)
            rewards_sum += rewards
        print(f"{i} {rewards_sum=}")

        image = env.image
        path = env.found_path
        print(f"{i} {path=}")
        new_image = remove_seam(image, path)
        print(f"{i} {new_image.shape=}")
        final_image = add_empty_vertical_lines(new_image, 1)

        out_image = cv2.cvtColor(np.float32(final_image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"D:\\Source\\seam-carving\\images-out\\temp1\\abc_{i}.png", out_image)

        i += 1
        env = SeamCarvingEnv(final_image, block_right_lines=i)
        obs = env.reset()
        done = False


    recolored_image = cv2.cvtColor(np.float32(env.image), cv2.COLOR_BGR2RGB)
    cropped_image = recolored_image[:,0:(env.WIDTH - len(start_seam_locations))]
    cv2.imwrite(f"D:\\Source\\seam-carving\\images-out\\temp1\\abc_FINAL.png", cropped_image)

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