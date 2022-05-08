import gym
from gym import spaces
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import normalize
import sys 

class SeamCarvingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, img_path: str):
        self.img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.img = cv2.resize(self.img, (160, 120), interpolation=cv2.INTER_AREA)
        self.img_energy = self.calculate_img_energy(self.img)
        self.img_energy = cv2.resize(self.img_energy, (160, 120), interpolation=cv2.INTER_AREA)

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

        self.energy_max = 1000 # approximate
        self.energy_min = 0

        self.action_space = spaces.Discrete(3) # Going left, mid or right

        # Example for using image as input:
        # observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

        self.obs_width = 159
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.line_count, self.obs_width, 3), dtype=np.uint8)

        # =======================
        # 3 next lines of data + first line of location
        # self.observation_space = spaces.Box(low=0, high=256, shape=(4, self.line_length), dtype=np.int)
        # self.zero_obs = np.zeros(self.line_length, dtype=np.int)
        # =======================

        # =======================
        # Stable baselines doesn't support dict obs space
        # self.observation_space = spaces.Dict({
        #     "lines_data": spaces.Box(low=0, high=256, shape=(self.line_length, 3), dtype=np.int),
        #     "current_location": spaces.Discrete(self.line_length),
        # })
        # =======================

        self.test = 0

    def normalize_data(self, data, min, max):
        return (data - min) / (max - min)

    def is_done(self):
        return True if self.current_line >= self.line_count - 1 else False

    def get_observations(self):
        low = self.current_location - self.obs_width // 2
        high = self.current_location + self.obs_width // 2 + 1

        min_location = max(low, 0)
        max_location = min(high, self.line_length)

        # lines_data = self.img_energy[self.current_line:, min_location:max_location]
        lines_data = self.img[self.current_line:, min_location:max_location]

        x_offset = -low if low < 0 else 0

        final_obs = np.full((self.line_count, self.obs_width, 3), 255, dtype=np.int)
        final_obs[:lines_data.shape[0], x_offset:lines_data.shape[1] + x_offset] = lines_data

        # if self.test == 0:
        #     np.set_printoptions(threshold=sys.maxsize)
        #     print("_++++++++++++++++++++")
        #     print(final_obs)
        #     self.test += 1

        return final_obs
        # return final_obs / 255 # normalize

        # =======================
        # current_location_array = np.zeros(self.line_length, dtype=np.int)
        # current_location_array[self.current_location] = 1

        # lines_data = self.img_energy[self.current_line:self.current_line + 3]
        # lines_data = np.concatenate(([current_location_array], lines_data))

        # # Add zeroes if it's end of the picture
        # while not len(lines_data) == 4:
        #     lines_data = np.concatenate((lines_data, [self.zero_obs]))

        # return lines_data
        # =======================

        # =======================
        # return {
        #     "lines_data": self.img_energy[self.current_line:self.current_line + 3],
        #     "current_location": self.current_location,
        # }
        # =======================

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

    def calculate_img_energy(self, image):
        image = image.astype("float32")
        energy = np.absolute(cv2.Sobel(image, -1, 1, 0)) + np.absolute(cv2.Sobel(image, -1, 0, 1))
        energy_map = np.sum(energy, axis=2)
        return energy_map

    def normalize_reward(self, reward):
        """Inverse and normalize reward"""
        return (self.energy_max - reward) / (self.energy_max - self.energy_min)

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
        # reward = self.normalize_reward(reward)

        # reward = self.normalize_reward(-reward)

        # self.test += 1
        # print(f"{self.test} Reward: {reward}, {self.normalize_reward(-reward)}")

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

        # self.test += 1
        # print(f"Episode reset count: {self.test}")

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