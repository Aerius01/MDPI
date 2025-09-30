import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
from pipeline.tools.preprocessing.rotate_image import rotate_image


def image_preprocessing(image_paths, img_size):
    print('[PREPROCESSOR]: loading and preprocessing images...')

    # initialize empty variable holders
    images = []
    angles = []
    total_images = len(image_paths)

    # loop over all images
    for i, image_path in enumerate(image_paths):
        # load in image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Otsu threshold
        ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # find contours
        contours, hierarchy = cv2.findContours(otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # create empty image
        largest_area = np.zeros(shape=[100, 100])

        maxi = 0.
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > maxi:
                maxi = area
                idx = i
        contour = contours[idx]
        largest_area = cv2.drawContours(largest_area, [contour], 0, (255, 255, 255), -1)

        # rotate objects vertical
        y, x = np.nonzero(largest_area)
        x = x - np.mean(x)
        y = y - np.mean(y)

        # Compute the angle of the first principal axes
        pca = PCA(n_components=2).fit(np.c_[x, y])
        
        # Check for vertical alignment to prevent division by zero
        if np.isclose(pca.components_[0, 1], 0):
            theta = 90.0
        else:
            theta = -np.tanh(pca.components_[0, 0] / pca.components_[0, 1]) * 180 / np.pi

        # rotate image upright
        img = rotate_image(img, theta)

        # resize each image to fit the expected image size
        img = cv2.resize(img, (img_size, img_size))

        # list all images
        angles.append(theta)
        images.append(img)

    # calculate image means
    images_mean = np.mean(images)

    return angles, images, images_mean
