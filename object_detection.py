import cv2
import os
import pandas as pd
from imutils import paths
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from tqdm import tqdm
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-input", type=str, default="./flatfield_profiles", help="path to project")
ap.add_argument("-output", type=str, default="./vignettes", help="path to store output")
ap.add_argument("-img_size", type=int, default=100, help='set vignette output size; default = 100')
args = vars(ap.parse_args())

# list images and sort them by name
imagePaths = list(paths.list_images(args["input"]))
imagePaths = sorted(imagePaths)

# initialize data, location and date variables
data = []
old_location = None
old_date = None
old_time = None
old_depth = None
old_path = None

# loop over the images
image_counter = 0
print('[Progress report]: start processing the profiles...')
for imagePath in tqdm(imagePaths, desc='detecting objects'):
    # storing metadata
    _, _, project, date, time, location, filename = imagePath.split(os.path.sep)

    # storing current path
    current_path = os.path.dirname(imagePath)
    # remove filename extension
    filename = os.path.splitext(filename)[0]

    # get depth measurement
    char_loc = filename.find('_')
    depth = float(filename[:char_loc-1])

    # initialize tracking variables
    if old_location is None: old_location = location
    if old_date is None: old_date = date
    if old_time is None: old_time = time
    if old_path is None: old_path = current_path

    # update counter
    image_counter += 1

    # when moving to the new location, store the data into a dataframe
    if (current_path != old_path) or (image_counter == len(imagePaths)):
        print(f'[Progress report]: scan {date}_{time}_{location} DONE...')
        # assign all data of one location to a panda dataframe
        df = pd.DataFrame(data, columns=['Filename', 'Area', 'MajorAxisLength',
                                         'MinorAxisLength', 'Eccentricity',
                                         'Orientation', 'EquivDiameter', 'Solidity',
                                         'Extent', 'MaxIntensity', 'MeanIntensity',
                                         'MinIntensity', 'Perimeter'])

        # store measurements in output folder
        df.to_csv(os.path.sep.join([outputPath, f'objectMeasurements_{project}_{date}_{time}_{location}.txt']),
                  sep=' ', index=False)

        # update location and date
        old_location = location
        old_date = date
        old_path = current_path

        # empty data
        data = []

    # create output path
    outputPath = os.path.sep.join([args["output"], project, date, time, location])
    if not os.path.exists(outputPath): os.makedirs(outputPath)

    # reading in the scan
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    # thresholding the image
    img_bin = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY_INV)[1]

    # correct for image overlap with preceding image
    if not old_depth is None:
        # calculate the depth at the bottom of the preceding image
        # image is 4.3 cm height
        oldImageDepthEnd = old_depth * 100 + 4.3

        # calculate the depth at the top of the new image
        newImageDepthStart = depth * 100

        # there's only overlap if end depth is deeper than the start depth
        if oldImageDepthEnd - newImageDepthStart > 0:
            # calculate the amount of pixels needs to be ignored
            imageOverlap = ((oldImageDepthEnd - newImageDepthStart) / 4.3) * 2048

            # transform float to integer
            imageOverlap = round(imageOverlap)

            # remove overlap by setting it to white
            img_bin[:imageOverlap, :] = 255

        old_depth = depth

    # set depth after first image
    if old_depth is None: old_depth = depth

    # measure objects after thresholding
    label_img = label(img_bin)

    # remove smaller and larger than object
    label_img = remove_small_objects(label_img, min_size=75)
    label_img = label_img - remove_small_objects(label_img, min_size=5000)

    # reset object index for each
    i = 0
    # measuring objects based on preset conditions
    for region in regionprops(label_img, img):
        # remove filaments, specs that have a very low mean intensity, or very bright solid particles
        if region.eccentricity < 0.97 and region.mean_intensity < 130 and region.major_axis_length > 25 \
                and region.min_intensity < 65:
            # storing measurement data
            data.append([
                f"{filename}_{i}",
                region.area,
                region.major_axis_length,
                region.minor_axis_length,
                region.eccentricity,
                region.orientation,
                region.equivalent_diameter,
                region.solidity,
                region.extent,
                region.max_intensity,
                region.mean_intensity,
                region.min_intensity,
                region.perimeter
            ])

            # cropping object
            row, col = region.centroid
            row = int(row)
            col = int(col)

            # set the padding size
            if region.major_axis_length < 40:
                padding = 25
            elif region.major_axis_length < 50:
                padding = 30
            else:
                padding = 40

            # add padding
            minr = 0 if row - padding < 0 else row - padding
            minc = 0 if col - padding < 0 else col - padding
            maxr = img.shape[0] if row + padding > img.shape[0] else row + padding
            maxc = img.shape[1] if col + padding > img.shape[1] else col + padding

            # crop object
            crop_img = img[minr:maxr, minc:maxc]

            # save cropped object
            cv2.imwrite(os.path.sep.join([outputPath, f'{filename}_{i}.JPEG']), crop_img)

            # update index
            i += 1
print('[Progress report]: object detection DONE')
