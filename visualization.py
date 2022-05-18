
from pathlib import Path
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
from environment import SeamCarvingEnv
from seam_carving import calc_img_energy
from scipy.special import softmax

out_folder = Path("images-out") / "visual"
out_folder.mkdir(parents=True, exist_ok=True)

# input = "images\\Broadway_tower_edit.jpg"
input = "images\\clocks-scaled.png"
# input = "images\\4k-plane.jpg"

def save_image(image: cv2.Mat, name: str):
    path = f"{str(out_folder)}\\{name}.png"
    cv2.imwrite(path, image)

def main():
    # image = cv2.imread(input, cv2.IMREAD_COLOR)
    # scaled_image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
    # save_image(scaled_image, "scaled_image_og")
    # return

    image = cv2.imread(input, cv2.IMREAD_COLOR)
    save_image(image, "image_og")

    energy = calc_img_energy(image)
    save_image(energy, "energy_og")

    energy_norm = np.interp(energy, (energy.min(), energy.max()), (0, 255))
    save_image(energy_norm, "energy_norm")

    normalized_energy_50 = np.where(energy_norm < 50, 0, 255)
    save_image(normalized_energy_50, "energy_norm_50")

    normalized_energy_100 = np.where(energy_norm < 100, 0, 255)
    save_image(normalized_energy_100, "energy_norm_100")

    energy_softmax = softmax(energy, axis=1)
    print(f"{energy_softmax.shape=} {energy_softmax.min()=} {energy_softmax.max()=}")

    energy_softmax_norm = np.interp(energy_softmax, (0, 1), (0, 255))
    save_image(energy_softmax_norm, "energy_softmax")
    # np.savetxt("images-out\\visual\\file.txt", energy_softmax, fmt='%.2f')
    # np.savetxt("images-out\\visual\\file1.txt", energy_softmax_norm, fmt='%d')

if __name__ == "__main__":
    main()