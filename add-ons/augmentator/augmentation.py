import os

import cv2
import numpy as np


def augmentor(DATADIR, CATEGORIES, FLIP, NOISE, ROTATE):
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        for img in os.listdir(path):
            image_path = cv2.imread(os.path.join(path,img))
            image_save = os.path.join(path, os.path.basename(img))
            try:
                if ROTATE == True:
                    R_image = cv2.rotate(image_path, cv2.ROTATE_90_CLOCKWISE)
                    cv2.imwrite(image_save + "rotated.jpg", R_image)
                if NOISE == True:
                    mean = 0
                    stddev = 180
                    noise = np.zeros(image_path.shape, np.uint8)
                    cv2.randn(noise, mean, stddev)
                    noisy_image = cv2.add(image_path, noise)
                    cv2.imwrite(image_save + "noised.jpg", noisy_image)
                if FLIP == True:
                    F_image = cv2.flip(image_path, 0)
                    cv2.imwrite(image_save + "flipped.jpg", F_image)
            except Exception as e:
                print("Singularity: Augmentation error occured!")