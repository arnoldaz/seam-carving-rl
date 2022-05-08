import cv2
import numpy as np

def rgb_to_gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def calc_img_energy(image):
    image = image.astype("float32")
    energy = np.absolute(cv2.Sobel(image, -1, 1, 0)) + np.absolute(cv2.Sobel(image, -1, 0, 1))
    energy_map = np.sum(energy, axis=2)
    return energy_map