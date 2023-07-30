import cv2


def augmentor(DATADIR, CATEGORIES, FLIP, NOISE, ROTATE):
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                if ROTATE == True:
                    R_image = cv2.rotate(path, cv2.ROTATE_90_CLOCKWISE)
                    cv2.imwrite(category + "rotated.jpg", R_image)
                if NOISE == True:
                    mean = 0
                    stddev = 180
                    noise = np.zeros(img.shape, np.uint8)
                    cv2.randn(noise, mean, stddev)
                    noisy_img = cv2.add(img, noise)
                    cv2.imwrite(category + "noised.jpg", R_image)
                if FLIP == True:
                    F_image = cv2.flip(path, 0)
                    cv2.imwrite(category + "flipped.jpg", R_image)
            except Exception as e:
                print("Singularity: Augmentation error occured!")