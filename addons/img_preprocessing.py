import os

import cv2
import numpy as np


def img_preprocessing(DATADIR, CATEGORIES, IMG_SIZE):
    training_data = []

    #formats data
    def create_training_data():
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path,img))
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num])
                except Exception as e:
                    pass

    #creates data as arrays
    create_training_data()
    random.shuffle(training_data)
    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return X, y