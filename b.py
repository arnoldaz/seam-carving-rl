
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from environment import SeamCarvingEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def main():
    """Testing stuff"""
    out_path = "../images-out/clocks-env-test999.png"

    image = cv2.imread("images/clocks-scaled.png", cv2.IMREAD_COLOR)
    heigth, width, rgb = image.shape

    half_width = width // 2
    start_x = width + 50 - half_width
    end_x = start_x + width
    start_y = 20
    end_y = start_y + heigth

    empty_matrix = np.full((3, heigth * 2, width * 3), 0)
    empty_matrix[:,0:heigth, width:(width * 2)] = image[:,0:heigth, 0:(width)]

    end_image = empty_matrix[start_y:end_y, start_x:end_x]
    plt.imshow(end_image)
    plt.show()
    # print(time.process_time() - start)

    return

    obs = env.reset()
    done = False

    # print(env.current_location)
    # env.current_location = 120
    # print(env.current_location)

    # print(env.energy_min)
    # print(env.energy_max)

    model = PPO.load("e110535a_20")

    while not done:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, done, info = env.step(action)
        # env.render()
        print(rewards)

    image = env.render_image
    cv2.imwrite(out_path, image)

if __name__ == "__main__":
    main()