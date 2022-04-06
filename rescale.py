from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def calc_img_energy(image):
    image = image.astype("float32")
    energy = np.absolute(cv2.Sobel(image, -1, 1, 0)) + np.absolute(cv2.Sobel(image, -1, 0, 1))
    energy_map = np.sum(energy, axis=2)
    return energy_map


image = cv2.imread("./images/clocks.jpeg", cv2.IMREAD_COLOR)

image = calc_img_energy(image)
image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
# image = rgb2gray(image)
print(image.max())
print(image.min())
print(image)

cv2.imwrite("test.png", image)

# plt.imshow(image)
# plt.show()