import gym
from gym import spaces
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from seam_carving import calc_img_energy

class SeamCarvingEnv(gym.Env):
    """Gym environment for image seam carving path finding."""

    metadata = {"render_modes": ["human"]}

    # Available actions
    LEFT = 0
    MIDDLE = 1
    RIGHT = 2

    # Image dimentions, currently resized for easier training
    WIDTH = 160
    HEIGHT = 120

    def __init__(self, image_path: str):
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        scaled_image = cv2.resize(original_image, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_AREA)
        self.image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
        self.image_energy = calc_img_energy(self.image)

        self.energy_min = self.image_energy.min()
        self.energy_max = self.image_energy.max()

        self.render_image = self.image[:]
        self.render_image_object = None
        self.render_initialized = False
        self.path_line_color = [255, 255, 255] # White

        self.current_line = 0
        self.current_location = random.randint(0, self.WIDTH - 1)

        self.found_path = np.full(self.HEIGHT, -1, dtype=int)
        self.found_path[0] = self.current_location

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, self.HEIGHT, self.WIDTH), dtype=np.uint8)

        normalized_energy = np.interp(self.image_energy, (self.image_energy.min(), self.image_energy.max()), (0, 255))
        self.observation_image_full = self.modify_image_for_observations(normalized_energy)

        normalized_energy_50 = np.where(normalized_energy < 50, 0, 255)
        self.observation_image_50 = self.modify_image_for_observations(normalized_energy_50)

        normalized_energy_100 = np.where(normalized_energy < 100, 0, 255)
        self.observation_image_100 = self.modify_image_for_observations(normalized_energy_100)

    def modify_image_for_observations(self, image):
        """Returns 3x2 matrix of original image with clone images flipped."""
        image_flipped_vertical = cv2.flip(image, 0)
        image_flipped_horizontal = cv2.flip(image, 1)
        image_flipped_both = cv2.flip(image, -1)

        image_layer_1 = np.concatenate((image_flipped_horizontal, image, image_flipped_horizontal), axis=1)
        image_layer_2 = np.concatenate((image_flipped_both, image_flipped_vertical, image_flipped_both), axis=1)

        return np.concatenate((image_layer_1, image_layer_2), axis=0)

    def is_done(self):
        """Returns true if episode is done."""
        return self.current_line >= self.HEIGHT - 1

    def get_observations_for_image(self, image):
        """Cuts out original image size observation from observation image."""
        half_width = self.WIDTH // 2
        start_x = self.WIDTH + self.current_location - half_width
        end_x = start_x + self.WIDTH
        start_y = self.current_line
        end_y = start_y + self.HEIGHT
    
        return image[start_y:end_y, start_x:end_x]

    def get_observations(self):
        observation_full = self.get_observations_for_image(self.observation_image_full)
        observation_50 = self.get_observations_for_image(self.observation_image_50)
        observation_100 = self.get_observations_for_image(self.observation_image_100)

        return np.array([observation_full, observation_50, observation_100])

    def normalize_energy_value(self, energy_value: int) -> int:
        """Normalize energy value"""
        return 1 - ((self.energy_max - energy_value) / (self.energy_max - self.energy_min))

    def step(self, action):
        reward = 0.0
        out_of_bounds = False
        self.current_line += 1

        if action == self.LEFT:
            self.current_location -= 1
            if self.current_location < 0:
                self.current_location = 0
                reward -= 100 
                out_of_bounds = True
        elif action == self.MIDDLE:
            None
        elif action == self.RIGHT:
            self.current_location += 1
            if self.current_location > self.WIDTH - 1:
                self.current_location = self.WIDTH - 1
                reward -= 100
                out_of_bounds = True
        else:
            raise Exception("Only 3 actions are supported: LEFT, MIDDLE, RIGHT")

        if not out_of_bounds:
            energy_value = self.image_energy[self.current_line][self.current_location]
            normalized_energy = 100 - np.interp(energy_value, (self.image_energy.min(), self.image_energy.max()), (0, 100))
            reward += normalized_energy

        self.found_path[self.current_line] = self.current_location
        self.render_image[self.current_line][self.current_location] = self.path_line_color

        return self.get_observations(), reward, self.is_done(), {}

    def reset(self):
        self.current_line = 0
        self.current_location = random.randint(0, self.WIDTH - 1)

        self.found_path = np.full(self.HEIGHT, -1, dtype=int)
        self.found_path[0] = self.current_location

        self.render_image = self.image[:]
        self.render_image_object = None
        self.render_initialized = False

        return self.get_observations()

    def render(self, mode="human"):
        if not mode == "human":
            raise NotImplementedError("Only human rendering is available")

        if not self.render_initialized:
            self.render_image_object = plt.imshow(self.render_image)
        else:
            self.render_image_object.set_data(self.render_image)

        plt.pause(0.01)



def main():
    """Testing stuff"""

    out_path = "../images-out/clocks-env-test1.png"
    env = SeamCarvingEnv("../images/clocks-fix.jpeg")

    image = env.observation_image
    cv2.imwrite(out_path, image)

if __name__ == "__main__":
   main()