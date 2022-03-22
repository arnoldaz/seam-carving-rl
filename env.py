import gym
from gym import spaces
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

class SeamCarvingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, img_path: str):
        self.img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.img_energy = self.calculate_img_energy(self.img)

        self.render_img = self.img[:]
        self.render_img_object = None
        self.render_initialized = False
        self.line_color = [255, 255, 255]

        self.line_count = len(self.img_energy)
        self.line_length = len(self.img_energy[0])

        self.current_line = 0
        self.current_location = random.randint(0, self.line_length - 1)

        self.found_path = np.full(self.line_count, -1, dtype=np.int)
        self.found_path[0] = self.current_location

        self.action_space = spaces.Discrete(3) # Going left, mid or right
        self.observation_space = spaces.Box(low=0, high=256, shape=(4, self.line_length), dtype=np.int)
        self.zero_obs = np.zeros(self.line_length, dtype=np.int)

        # Stable baselines doesn't support dict obs space
        # self.observation_space = spaces.Dict({
        #     "lines_data": spaces.Box(low=0, high=256, shape=(self.line_length, 3), dtype=np.int),
        #     "current_location": spaces.Discrete(self.line_length),
        # })

    def is_done(self):
        return True if self.current_line >= self.line_count - 1 else False

    def get_observations(self):
        current_location_array = np.zeros(self.line_length, dtype=np.int)
        current_location_array[self.current_location] = 1

        lines_data = self.img_energy[self.current_line:self.current_line + 3]
        lines_data = np.concatenate(([current_location_array], lines_data))

        # Add zeroes if it's end of the picture
        while not len(lines_data) == 4:
            lines_data = np.concatenate((lines_data, [self.zero_obs]))

        return lines_data

        # return {
        #     "lines_data": self.img_energy[self.current_line:self.current_line + 3],
        #     "current_location": self.current_location,
        # }

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    def calculate_img_energy(self, image):
        image = image.astype("float32")
        energy = np.absolute(cv2.Sobel(image, -1, 1, 0)) + np.absolute(cv2.Sobel(image, -1, 0, 1))
        energy_map = np.sum(energy, axis=2)
        return energy_map

    def step(self, action):
        reward = 0
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
            self.render_img_object = plt.imshow(self.render_img)
        else:
            self.render_img_object.set_data(self.render_img)

        # render_img = self.img[:]
        # for line_number in range(self.line_count):
        #     path_segment = self.found_path[line_number]
        #     render_img[line_number][path_segment] = self.white_color

        plt.pause(0.01)