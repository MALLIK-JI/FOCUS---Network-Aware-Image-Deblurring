
import cv2
import numpy as np
import os

input_dir = "./test_input/test/GoPro/input"
blur_dir = "./test_input/test/GoPro/blurred"
gt_dir = "./test_input/test/GoPro/ground_truth"

os.makedirs(blur_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)

def motion_blur(image, kernel_size=25):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image = cv2.imread(os.path.join(input_dir, filename))

        if image is None:
            continue

        blurred = motion_blur(image)

        cv2.imwrite(os.path.join(gt_dir, filename), image)
        cv2.imwrite(os.path.join(blur_dir, filename), blurred)

        print("Created blurred image:", filename)

print("Done!")
