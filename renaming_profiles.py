import os
import pandas as pd
import shutil
from dateutil import relativedelta
from datetime import datetime
from imutils import paths
from tqdm import tqdm
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-input", type=str, default="./profiles", help="path to project")
ap.add_argument('-output', type=str, default='./depth_profiles', help='output path; default = depth_profiles')
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["input"]))
imagePaths = sorted(imagePaths)

# to store the images and their names
images = []
img_names = []

# for the logical testing
old_inputPath = None
old_depth = None

print('[Progress report]: renaming image profiles...')
# loop over the images
image_counter = 0
for total_counter, imagePath in enumerate(tqdm(imagePaths)):
    # storing metadata
    _, _, project, date, time, location, filename = imagePath.split(os.path.sep)

    # set input path
    inputPath = os.path.sep.join([args['input'], project, date, time, location])

    # create output path
    outputPath = os.path.sep.join([args["output"], project, date, time, location])
    if not os.path.exists(outputPath): os.makedirs(outputPath)

    # update counter
    image_counter += 1

    # read in depth measurements
    if (inputPath != old_inputPath) or \
            (image_counter == len(imagePaths)):
        for root, dir, files in os.walk(inputPath):
            for file in files:
                if file.endswith('.csv'):
                    # path to csv file
                    csvPath = os.path.sep.join([root, file])

                    # read in first two columns (time and depth) of the csv file
                    # first 6 lines general info
                    # last line is NA
                    # first two column ar time and depth
                    csv = pd.read_csv(csvPath, sep=';', header=6, usecols=[0, 1], names=['time', 'depth'],
                                      index_col='time', skipfooter=1, engine='python')

                    # set time format
                    csv.index = pd.to_datetime(csv.index, format='%d.%m.%Y %H:%M:%S.%f')

                    # update old_inputPath
                    old_inputPath = inputPath

    # get creating time of the image
    #image_time = os.stat(imagePath).st_mtime
    image_time = os.path.getmtime(imagePath)
    image_time = datetime.fromtimestamp(image_time).strftime('%Y-%m-%d %H:%M:%S.%f')
    image_time = pd.to_datetime(image_time)
    #image_time = pd.to_datetime(image_time) - relativedelta.relativedelta(months=1)
    print(image_time)

    # find nearest depth measurement
    index = csv.index.get_loc(image_time, method='nearest')
    depth = csv.loc[csv.index[index], 'depth']
    depth = float(depth.replace(',', '.')) * 10

    # set 3 digits depth
    str_depth = str(depth)[:7]

    # create new filename
    new_filename = os.path.sep.join([outputPath, f'{str_depth}_{project}_{date}_{time}_{location}.tiff'])

    # in case not enough depth measurements are done
    if not os.path.exists(new_filename):
        # rename file
        shutil.copy(imagePath, new_filename)

    else:
        if not index + 1 > len(csv.index) - 1:
            # select second depth measurement
            next_depth = csv.loc[csv.index[index + 1], 'depth']
            next_depth = float(next_depth.replace(',', '.'))

            # average between measurements
            depth_new = depth + next_depth / 2
            str_depth = str(depth_new)[:7]

            # create new filename
            new_filename = os.path.sep.join([outputPath, f'{str_depth}_{project}_{date}_{time}_{location}.tiff'])

            if not os.path.exists(new_filename):
                # rename file
                shutil.copy(imagePath, new_filename)
            else:

                # select second depth measurement
                next_depth = csv.loc[csv.index[index + 2], 'depth']
                next_depth = float(next_depth.replace(',', '.'))

                # average between measurements
                depth_new = depth_new + next_depth / 2
                str_depth = str(depth_new)[:7]

                # create new filename
                new_filename = os.path.sep.join([outputPath, f'{str_depth}_{project}_{date}_{time}_{location}.tiff'])

                if not os.path.exists(new_filename):
                    # rename file
                    shutil.copy(imagePath, new_filename)
                else:
                    # rename file
                    print(imagePath)
                    print(new_filename)

