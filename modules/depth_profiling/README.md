# Depth Profiling Module

This module is responsible for assigning a precise depth value to each image captured by the MDPI instrument. It synchronizes the timestamps of the images with data from a pressure sensor to create a detailed depth profile for each imaging run.

## Abstract

The Depth Profiling module is the first main step in the MDPI image processing pipeline. It is responsible for assigning a precise depth value to each image captured by the instrument. It synchronizes the timestamps of the images with data from a pressure sensor to create a detailed depth profile for each imaging run.

## How it Works

The module processes a directory of raw images and a corresponding pressure sensor CSV file to generate a final CSV mapping each image to a specific depth. The process involves several key steps:

1.  **Contextual Setup**: The module receives a `run_config` configuration object, containing information that establishes the context for the profiling run.

2.  **Automatic Camera Format Detection**: The module automatically detects whether the data is from a "new" or "old" MDPI camera format by inspecting the header of the pressure sensor's CSV file. It checks if the first column name is 'Name' (old format) or 'Number' (new format). This is crucial as different camera systems have unique CSV structures, column names, and pressure-to-depth conversion multipliers.

3.  **Image Timestamp Calculation**: Using the recording start time (from metadata) and the user-provided image capture rate (in Hz), the module calculates a precise timestamp for every single image in the sequence.

4.  **Depth Synchronization**: This is the core step of the process. For each calculated image timestamp, the module searches the pressure sensor data to find the depth measurement that was recorded at the nearest point in time. This synchronizes the two data streams.

5.  **Pixel Overlap Calculation**: To aid in subsequent analysis (like object tracking or avoiding duplicate counts), the module calculates the vertical pixel overlap between each consecutive pair of images based on their assigned depths and the physical height of the camera's field of view.

6.  **Output Generation**: The final output is a `depth_profiles.csv` file. This file contains the absolute path to each image, its calculated depth, and the pixel overlap with the next image.

## Important Considerations

*   **Data Accuracy**: The quality of the output highly depends on the accuracy of the instrument's physical parameters, which are defined as constants in `modules/depth_profiling/depth_profile_data.py`. These values must be correctly calibrated for the specific MDPI instrument being used.
