# ----------------------------------------------------------------------
# Title:    normalize.py
# Details:  calculate Flower classification images
#           means and standard errors for normalization
#
# Author:   Ruihan Xu
# Created:  2020/01/15
# Modified: 2020/01/15
# ----------------------------------------------------------------------

import numpy as np
import cv2
import os


def normalize_params(data_dir, mode="train"):
    img_h, img_w = 56, 56
    imgs = np.zeros([img_w, img_h, 3, 1])
    means, stdevs = [], []

    walk_dir = os.walk(data_dir)
    for root, dirs, files in walk_dir:
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            img = cv2.resize(img, (img_w, img_h))
            img = img[:, :, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)

    imgs = imgs.astype(np.float32) / 255

    for idx in range(3):
        pixels = imgs[:, :, idx, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR -> RGB
    stdevs.reverse()

    print(means)
    print(stdevs)
    return means, stdevs


if __name__ == "__main__":
    normalize_params("data/train", "train")
    # [0.4590731, 0.4196773, 0.3008733], [0.29598123, 0.2657097, 0.28848845]
    normalize_params("data/test", "test")
    # [0.45042223, 0.41608664, 0.29151103], [0.29657817, 0.2652202, 0.28399792]
