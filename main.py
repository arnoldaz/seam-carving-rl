import cv2
import numpy as np
import os
import errno
from os import path
from glob import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt



def calc_seam_cost_forward(energy_map):
    shape = m, n = energy_map.shape
    e_map = np.copy(energy_map).astype('float32')
    backtrack = np.zeros(shape, dtype=int)
    for i in range(1, m):
        for j in range(0, n):
            if j == 0:
                min_idx = np.argmin(e_map[i - 1, j:j + 2])
                min_cost = e_map[i - 1, j + min_idx]
                e_map[i, j] += min_cost
                backtrack[i, j] = j + min_idx
            else:
                min_idx = np.argmin(e_map[i - 1, j - 1:j + 2])
                min_cost = e_map[i - 1, j + min_idx - 1]
                e_map[i, j] += min_cost
                backtrack[i, j] = j + min_idx - 1
    return (e_map, backtrack)



def calc_img_energy(image):
    image = image.astype('float32')
    energy = np.absolute(cv2.Sobel(image, -1, 1, 0)) + np.absolute(cv2.Sobel(image, -1, 0, 1))
    energy_map = np.sum(energy, axis=2)
    return energy_map


def rgbToGrey(arr):
    greyVal = np.dot(arr[...,:3], [0.2989, 0.5870, 0.1140])
    return np.round(greyVal).astype(np.int32)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

fig5 = cv2.imread('./clock.jpeg', cv2.IMREAD_COLOR)
# rgb_fig5 = cv2.cvtColor(fig5, cv2.COLOR_BGR2RGB)

# imgColor = Image.open('./fig5.png')
# img = Image.open('./fig5.png').convert('L')
# data = np.asarray( img, dtype="int32" )
# dataColor = np.asarray(imgColor,dtype="int32")

energy_map = calc_img_energy(fig5)

dataGrey = rgb2gray(fig5)
print(fig5)
print(energy_map)
print(dataGrey)

# energy_map_forward, backtrack = calc_seam_cost_forward(energy_map)
a = plt.imshow(energy_map)
print(a)
print("Figure 5 Shape: %s" % (dataGrey.shape,))
plt.show()