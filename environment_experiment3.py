import gym
from gym import spaces
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

from seam_carving import calc_img_energy 

class SeamCarvingEnv(gym.Env):
    """Downgraded seam carving environment for experiment 3."""

    # Available actions
    LEFT = 0
    MIDDLE = 1
    RIGHT = 2

    def __init__(self, img_path: str):
        self.image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.image_energy = calc_img_energy(self.image)

        # self.render_img = self.image[:]
        # self.render_img_object = None
        # self.render_initialized = False
        # self.line_color = [255, 255, 255]

        self.line_count = len(self.image_energy)
        self.line_length = len(self.image_energy[0])

        self.image_height, self.image_width, _ = self.image.shape

        self.energy_min = self.image_energy.min()
        self.energy_max = self.image_energy.max()

        self.current_line = 0
        self.current_location = random.randint(0, self.line_length - 1)

        self.found_path = np.full(self.line_count, -1, dtype=int)
        self.found_path[0] = self.current_location

        self.action_space = spaces.Discrete(3) # Going left, mid or right

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.line_count, self.line_length), dtype=np.float32)

        self.obs = np.interp(self.image_energy, (self.image_energy.min(), self.image_energy.max()), (0, 255))
        self.obs_image = self.fill_image_for_observations(self.obs, 255)

        print("In experiment 3 environment")

    def fill_image_for_observations(self, image, max):
        height, width = image.shape

        empty_matrix = np.full((height * 2, width * 3), max)
        empty_matrix[0:height, width:(width * 2)] = image

        return empty_matrix

    def is_done(self):
        return self.current_line >= self.line_count - 1

    def get_observations_for_image(self, image):
        """Cuts out original image size observation from observation image."""
        half_width = self.line_length // 2
        start_x = self.line_length + self.current_location - half_width
        end_x = start_x + self.line_length
        start_y = self.current_line
        end_y = start_y + self.line_count
    
        # print(f"{half_width=} {start_x=} {end_x=} {start_y=} {end_y=}")

        return image[start_y:end_y, start_x:end_x]

    def get_observations(self):
        return self.get_observations_for_image(self.obs_image)

    def step(self, action):
        reward = 0.0
        out_of_bounds = False
        self.current_line += 1

        if action == self.LEFT:
            self.current_location -= 1
            if self.current_location < 0:
                self.current_location = 0
                reward -= 30 
                out_of_bounds = True
        elif action == self.MIDDLE:
            None
        elif action == self.RIGHT:
            self.current_location += 1
            if self.current_location > self.image_width - 1:
                self.current_location = self.image_width - 1
                reward -= 30
                out_of_bounds = True
        else:
            raise Exception("Only 3 actions are supported: LEFT, MIDDLE, RIGHT")

        if not out_of_bounds:
            energy_value = self.image_energy[self.current_line][self.current_location]
            normalized_energy = 100 - np.interp(energy_value, (self.energy_min, self.energy_max), (0, 100))
            reward += normalized_energy

        self.found_path[self.current_line] = self.current_location
        # self.render_img[self.current_line][self.current_location] = self.line_color

        return self.get_observations(), reward, self.is_done(), {}

    def reset(self):
        self.current_line = 0
        self.current_location = random.randint(0, self.line_length - 1)

        self.found_path = np.full(self.line_count, -1, dtype=np.int)
        self.found_path[0] = self.current_location

        # self.render_img = self.image[:]
        # self.render_img_object = None
        # self.render_initialized = False

        return self.get_observations()

    # def render(self):
    #     if not self.render_initialized:
    #         self.render_img_object = plt.imshow(self.get_observations())
    #     else:
    #         self.render_img_object.set_data(self.get_observations())

    #     plt.pause(0.01)

def main():
    """Testing stuff"""

    env = SeamCarvingEnv("images/clocks-scaled.png")
    print(env.obs)

if __name__ == "__main__":
   main()