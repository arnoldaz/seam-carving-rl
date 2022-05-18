import gym
from gym import spaces
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

from seam_carving import calc_img_energy 

class SeamCarvingEnv(gym.Env):
    def __init__(self, img_path: str):
        self.img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.img_energy = calc_img_energy(self.img)

        self.render_img = self.img[:]
        self.render_img_object = None
        self.render_initialized = False
        self.line_color = [255, 255, 255]

        self.line_count = len(self.img_energy)
        self.line_length = len(self.img_energy[0])

        self.current_line = 0
        self.current_location = random.randint(0, self.line_length - 1)

        self.found_path = np.full(self.line_count, -1, dtype=int)
        self.found_path[0] = self.current_location

        self.action_space = spaces.Discrete(3) # Going left, mid or right

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.line_count, self.line_length), dtype=np.float32)

        self.obs = np.interp(self.img_energy, (self.img_energy.min(), self.img_energy.max()), (0, 255))
        self.obs_image = self.fill_image_for_observations(self.obs, 255)

        print("In experiment 2 environment")

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
        self.current_line += 1
        
        # (-1, 0, 1)
        action -= 1

        self.current_location += action
        if self.current_location < 0:
            self.current_location = 0
        if self.current_location > self.line_length - 1:
            self.current_location = self.line_length - 1

        reward -= self.img_energy[self.current_line][self.current_location]

        self.found_path[self.current_line] = self.current_location
        self.render_img[self.current_line][self.current_location] = self.line_color

        return self.get_observations(), reward, self.is_done(), {}

    def reset(self):
        self.current_line = 0
        self.current_location = random.randint(0, self.line_length - 1)

        self.found_path = np.full(self.line_count, -1, dtype=np.int)
        self.found_path[0] = self.current_location

        self.render_img = self.img[:]
        self.render_img_object = None
        self.render_initialized = False

        return self.get_observations()

    def render(self):
        if not self.render_initialized:
            self.render_img_object = plt.imshow(self.get_observations())
        else:
            self.render_img_object.set_data(self.get_observations())

        plt.pause(0.01)

def main():
    """Testing stuff"""

    env = SeamCarvingEnv("images/clocks-scaled.png")
    print(env.obs)

if __name__ == "__main__":
   main()