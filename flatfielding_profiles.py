import cv2
import os
import numpy as np
from imutils import paths
from tqdm import tqdm
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-input", type=str, default="./depth_profiles", help="path to project")
ap.add_argument("-output", type=str, default="./flatfield_profiles", help="path to store output")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["input"]))
imagePaths = sorted(imagePaths)

# to store the images and their names
images = []
img_names = []

# for the logical testing
old_inputPath = None

print('[Progress report]: start flatfielding the profiles...')
# loop over the images
image_counter = 0

for total_counter, imagePath in enumerate(tqdm(imagePaths, desc='[Progress report]: loading images')):
    # storing metadata
    _, _, project, date, time, location, filename = imagePath.split(os.path.sep)

    # set input path
    inputPath = os.path.sep.join([args['input'], project, date, location])

    # initialize old_location and old date
    if old_inputPath is None: old_inputPath = inputPath

    # update counter
    image_counter += 1

    if old_inputPath != inputPath:
        # once loaded, calculate the average image to perform flatfielding
        images = np.array([np.array(img) for img in images])
        ff = np.average(images, axis=0).astype('uint8')

        # flatfield each image
        for index, image in enumerate(tqdm(images, desc='[Progress report]: Flat fielding')):
            # flatfield
            image = np.divide(image, ff)
            image = image * 235

            # save flatfielded image
            cv2.imwrite(os.path.sep.join([outputPath, f'{img_names[index]}.JPEG']), image)

        # to store the images and their names
        images = []
        img_names = []

        # set new old_path
        old_inputPath = inputPath

    # remove filename extension
    img_names.append(os.path.splitext(filename)[0])

    # loading all images of a single profile
    images.append(cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE))

    # create output path
    outputPath = os.path.sep.join([args["output"], project, date, time, location])
    if not os.path.exists(outputPath): os.makedirs(outputPath)

    if image_counter == len(imagePaths):
        # once loaded, calculate the average image to perform flatfielding
        images = np.array([np.array(img) for img in images])
        ff = np.average(images, axis=0).astype('uint8')

        # flatfield each image
        for index, image in enumerate(tqdm(images, desc='[Progress report]: Flat fielding')):
            # flatfield
            image = np.divide(image, ff)
            image = image * 235

            # save flatfielded image
            cv2.imwrite(os.path.sep.join([outputPath, f'{img_names[index]}.JPEG']), image)
