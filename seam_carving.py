import cv2
import numpy as np
import argparse
import random
import time


def rgb_to_gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def calc_img_energy(image):
    image = image.astype("float32")
    energy = np.absolute(cv2.Sobel(image, -1, 1, 0)) + np.absolute(cv2.Sobel(image, -1, 0, 1))
    energy_map = np.sum(energy, axis=2)
    return energy_map

def calc_seam_cost_forward(energy_map):
    shape = m, n = energy_map.shape
    e_map = np.copy(energy_map).astype("float32")
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

def find_min_seam(energy_map_forward, backtrack):
    shape = m, n = energy_map_forward.shape
    seam = np.zeros(m, dtype=int)
    idx = np.argmin(energy_map_forward[-1])
    cost = energy_map_forward[-1][idx]
    seam[-1] = idx
    for i in range(m - 2, -1, -1):
        idx = backtrack[i + 1, idx]
        seam[i] = idx
    return seam, cost

def find_index_seam(energy_map_forward, backtrack, start_location):
    """Finds seam from given starting location."""
    shape = m, n = energy_map_forward.shape
    seam = np.zeros(m, dtype=int)
    idx = start_location
    cost = energy_map_forward[-1][idx]
    seam[-1] = idx
    for i in range(m - 2, -1, -1):
        idx = backtrack[i + 1, idx]
        seam[i] = idx
    return seam, cost

def draw_seam(image, seam):
    rows = np.arange(0, seam.shape[0], 1)
    blue, green, red = cv2.split(image)
    blue[rows, seam] = 0
    green[rows, seam] = 0
    red[rows, seam] = 255
    img_with_seam = np.zeros((blue.shape[0], blue.shape[1], 3))
    img_with_seam[:,:,0] = blue
    img_with_seam[:,:,1] = green
    img_with_seam[:,:,2] = red
    return img_with_seam

def draw_fat_seam(image, seam):
    rows = np.arange(0, seam.shape[0], 1)
    blue, green, red = cv2.split(image)
    for i in [seam-1, seam, seam+1]:
        try:
            blue[rows, i] = 0
            green[rows, i] = 0
            red[rows, i] = 255
        except:
            pass
    img_with_seam = np.zeros((blue.shape[0], blue.shape[1], 3))
    img_with_seam[:,:,0] = blue
    img_with_seam[:,:,1] = green
    img_with_seam[:,:,2] = red
    return img_with_seam

def remove_seam(image, seam):
    m, n, _ = image.shape
    out_image = np.zeros((m, n - 1, 3)).astype(dtype=int)
    for i in range(m):
        j = seam[i]
        out_image[i, :, 0] = np.delete(image[i, :, 0], j)
        out_image[i, :, 1] = np.delete(image[i, :, 1], j)
        out_image[i, :, 2] = np.delete(image[i, :, 2], j)
    return out_image

def insert_seam(image, seam):
    m, n, num_channels = image.shape
    out_image = np.zeros((m, n + 1, 3)).astype(dtype=int)
    for i in range(m):
        j = seam[i]
        for ch in range(num_channels):
            if j == 0:
                out_image[i, j, ch] = image[i, j, ch]
                out_image[i, j + 1:, ch] = image[i, j:, ch]
                out_image[i, j + 1, ch] = (int(image[i, j, ch]) + int(image[i, j + 1, ch])) / int(2)
            elif j + 1 == n:
                out_image[i, :j + 1, ch] = image[i, :j + 1, ch]
                out_image[i, j + 1, ch] = int(image[i, j, ch])
            else:
                out_image[i, :j, ch] = image[i, :j, ch]
                out_image[i, j + 1:, ch] = image[i, j:, ch]
                out_image[i, j, ch] = (int(image[i, j - 1, ch]) + int(image[i, j + 1, ch])) / int(2)
    return out_image

def remove_vertical_seam(image):
    img = np.copy(image)
    energy_map = calc_img_energy(img)
    energy_map_forward, backtrack = calc_seam_cost_forward(energy_map)
    (min_seam, cost) = find_min_seam(energy_map_forward, backtrack)
    img = remove_seam(img, min_seam)
    return img, cost

def remove_horizontal_seam(image):
    img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    energy_map = calc_img_energy(img)
    energy_map_forward, backtrack = calc_seam_cost_forward(energy_map)
    (min_seam, cost) = find_min_seam(energy_map_forward, backtrack)
    img = remove_seam(img, min_seam)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img, cost

def calc_seam_cost_forward(energy_map):
    shape = m, n = energy_map.shape
    e_map = np.copy(energy_map).astype("float32")
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

def get_image_with_seam(img: cv2.Mat, start_seam_location: int) -> cv2.Mat:
    """Reads image and draws minimum seam from given starting location."""
    # Search is from the bottom, so need to rotate
    img = cv2.rotate(img, cv2.ROTATE_180)

    # Calculate seam normally with inverted image
    energy_map = calc_img_energy(img)
    energy_map_forward, backtrack = calc_seam_cost_forward(energy_map)
    min_seam, cost = find_index_seam(energy_map_forward, backtrack, start_seam_location)
    image_with_seam = draw_seam(img, min_seam)

    # Rotate back to original rotation
    image_with_seam = cv2.rotate(image_with_seam, cv2.ROTATE_180)

    return image_with_seam

def get_random_starting_points(width: int, point_count: int, seed: int = None) -> list[int]:
    """Generates random starting point array assuming that next point has 1 less width (line was seam carved)."""
    final_points = []
    current_width = width
    current_point_count = point_count

    random.seed(seed)

    while current_point_count > 0:
        rand_int = random.randint(0, current_width - 1)
        final_points.append(rand_int)

        current_width -= 1
        current_point_count -= 1

    return final_points

def seam_carve_image_from_points(img: cv2.Mat, start_seam_locations: list[int]) -> cv2.Mat:
    """Seam carve image using given list of starting locations."""
    # Search is from the bottom, so need to rotate
    img = cv2.rotate(img, cv2.ROTATE_180)

    i = 0
    for start_seam_location in start_seam_locations:
        energy_map = calc_img_energy(img)
        energy_map_forward, backtrack = calc_seam_cost_forward(energy_map)
        min_seam, cost = find_index_seam(energy_map_forward, backtrack, start_seam_location)
        print(f"{min_seam=}")
        image_without_seam = remove_seam(img, min_seam)

        cv2.imwrite(f"D:\\Source\\seam-carving\\images-out\\temp\\clocks-middle-{i}.png", cv2.rotate(image_without_seam, cv2.ROTATE_180))
        i += 1

        img = image_without_seam

    # Rotate back to original rotation
    img = cv2.rotate(img, cv2.ROTATE_180)
    
    return img

def add_empty_vertical_lines(img: cv2.Mat, line_count: int) -> cv2.Mat:
    heigth, width, rgb = img.shape
    empty_lines = np.full((heigth, line_count, 3), 255, dtype=int)

    new_img = np.concatenate((img, empty_lines), axis=1)
    # print(f"{new_img.shape=}")
    
    return new_img

def seam_carve_lines(img: cv2.Mat, line_count: int) -> cv2.Mat:
    new_img = np.copy(img)
    for i in range(line_count):
        energy_map = calc_img_energy(new_img)
        energy_map_forward, backtrack = calc_seam_cost_forward(energy_map)
        (min_seam, cost) = find_min_seam(energy_map_forward, backtrack)
        new_img = remove_seam(new_img, min_seam)


    return new_img

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="D:\\Source\\seam-carving\\images\\4k.jpg", help="Input image path")
parser.add_argument("-o", "--output", type=str, default="D:\\Source\\seam-carving\\images-out\\2.png", help="Output image path")
parser.add_argument("-l", "--location", type=int, default=80, help="Starting location for seam")

def main(args: argparse.Namespace):
    """Main function for testing seam calculation."""
    out_path = args.output

    start = time.time()

    og_img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    img = np.copy(og_img)
    energy_map = calc_img_energy(img)
    energy_map_forward, backtrack = calc_seam_cost_forward(energy_map)
    (min_seam, cost) = find_min_seam(energy_map_forward, backtrack)

    print(time.time() - start)

    bgr_img_with_seam = draw_seam(img, min_seam)
    cv2.imwrite("D:\\Source\\seam-carving\\images-out\\xxxxxxy.png", bgr_img_with_seam)
    # cv2.imwrite(out_path, bgr_img_with_seam)
    # img = remove_seam(img, min_seam)


    # image = get_image_with_seam(img, 800)

    # new_img = seam_carve_lines(img, 300)
    # energy = calc_img_energy(img)
    # energy = np.interp(energy_map, (energy_map.min(), energy_map.max()), (0, 255))
    # energy_map_forward, backtrack = calc_seam_cost_forward(energy)
    # energy_map_forward = np.interp(energy_map_forward, (energy_map_forward.min(), energy_map_forward.max()), (0, 255))
    return 
    # a = calc_img_energy(img)
    # a_r = cv2.resize(a, (160, 120), interpolation=cv2.INTER_AREA)
    # a_n = np.interp(a_r, (a_r.min(), a_r.max()), (0, 255))

    # img = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
    # b = calc_img_energy(img)
    # b_n = np.interp(b, (b.min(), b.max()), (0, 255))

    # cv2.imwrite(f"D:\\Source\\seam-carving\\images-out\\TEST1\\a.png", a)
    # cv2.imwrite(f"D:\\Source\\seam-carving\\images-out\\TEST1\\ar.png", a_r)
    # cv2.imwrite(f"D:\\Source\\seam-carving\\images-out\\TEST1\\b.png", b)
    # cv2.imwrite(f"D:\\Source\\seam-carving\\images-out\\TEST1\\an.png", a_n)
    # cv2.imwrite(f"D:\\Source\\seam-carving\\images-out\\TEST1\\bn.png", b_n)

    # return

    # out_image = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)
    # cv2.imwrite(f"D:\\Source\\seam-carving\\images\\clocks-scaled.png", img)
    # return 

    
    # test_img = add_empty_vertical_lines(img, 5)
    # cv2.imwrite("D:\\Source\\seam-carving\\images-out\\4.png", test_img)
    # return

    start_seam_locations = get_random_starting_points(160, 30, 123)
    print(f"{start_seam_locations=}")
    end_image = seam_carve_image_from_points(img, start_seam_locations)

    cv2.imwrite(out_path, end_image)
    return

    test = np.interp(energy_map, (energy_map.min(), energy_map.max()), (0, 255))
    print(test.min())
    print(test.max())
    print(test)
    cv2.imwrite("D:\\Source\\seam-carving\\images-out\\clocks-test6.png", test)
    # energy_map = np.array([
    #     [1, 2, 3],
    #     [5, 6, 7]
    # ])

    columns = np.sum(energy_map, axis=0)
    lines = np.sum(energy_map, axis=1)

    columns = np.reshape(columns, (1, 160))
    lines = np.reshape(lines, (120, 1))

    columns = columns / 160
    lines = lines / 120

    # column = np.reshape(column, (120, 1))
    # lines = np.reshape(lines, (1, 160))

    # print(columns)
    # print(lines)    

    # column = np.swapaxes(column, 0, 1)

    final = np.matmul(lines, columns)
    print(final.min())
    print(final.max())
    final = np.interp(final, (final.min(), final.max()), (0, 255))
    # print(final)
    # energy_map[energy_map < (3000/100*50)] = 0
    # cArray1 = cv2.CreateMat(120, 160, cv2.CV_32FC3)
    # cArray2 = cv2.fromarray(final)
    # cv2.CvtColor(cArray2, cArray1, cv2.CV_GRAY2BGR)
    # cv2.imwrite("cpImage.bmp", cArray1)

    cv2.imwrite(out_path, final)
    return

    # energy_map = cv2.resize(energy_map, (160, 120), interpolation=cv2.INTER_AREA)
    energy_map_forward, backtrack = calc_seam_cost_forward(energy_map)
    (min_seam, cost) = find_min_seam(energy_map_forward, backtrack)
    print(energy_map.max())
    print(energy_map.min())
    bgr_img_with_seam = draw_seam(img, min_seam)
    # cv2.imwrite(out_path, bgr_img_with_seam)
    img = remove_seam(img, min_seam)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)