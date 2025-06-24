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

# listing all images
imagePaths = list(paths.list_images(args["input"]))

# create a new list to be used for sorting, but ensure correct order of the imagePaths
imagePaths_sorting_list = []

# each number gets padded till it's 4 characters long

for name in imagePaths:
    # get online filename
    name = os.path.basename(name)

    # find where the number starts and ends
    start = name.rfind('_')+1
    end = name.rfind('.')

    # calculate number length
    length_number = end - start

    # if less then 4 characters long add padding
    if int(length_number) < 4:
        # extract number
        image_number = name[start:end]

        # amount of padding needed
        padding = 4 - int(length_number)

        # add left side zero padding to the number
        image_number = image_number.zfill(padding + length_number)

        # change name
        name = name[:start] + image_number
    else:
        # else just remove extension
        name = name[:end]

    # append to list for sorting
    imagePaths_sorting_list.append(name)

# sort imagePaths using the sorting list
imagePathsSorted = [imagePaths for _, imagePaths in sorted(zip(imagePaths_sorting_list, imagePaths))]

# for the logical testing
old_inputPath = None
old_depth = None
start_time = None

# loop over the images
image_counter = 0

# progress report
print('[Progress report]: renaming image profiles...')
for imagePath in tqdm(imagePathsSorted):
    # storing metadata
    _, _, project, date, time, location, filename = imagePath.split(os.path.sep)
    # set input path
    inputPath = os.path.sep.join([args['input'], project, date, time, location])

    # create output path
    outputPath = os.path.sep.join([args["output"], project, date, time, location])
    if not os.path.exists(outputPath): os.makedirs(outputPath)

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

                    # reset start_time
                    start_time = None

                    # reset image_counter
                    image_counter = 0

    # get creating time of the image
    if start_time is None:
        # find the date and time in the filename
        start = imagePath.rfind('__') + 2
        if start == 1:
            start = imagePath.rfind(')_') + 2
        end = start + 18
        datetime_str = imagePath[start:end]
        image_time = datetime.strptime(datetime_str, '%Y%m%d_%H%M%S%f')
        image_time = pd.to_datetime(image_time)

    # calculate image creation time
    if image_counter > 0:
        image_time = pd.to_datetime(image_time) + image_counter * \
                     relativedelta.relativedelta(microseconds=(1/2.4)*1000000)

    # update counter
    image_counter += 1

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
