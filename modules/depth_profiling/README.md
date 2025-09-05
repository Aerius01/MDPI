# Depth Profiling Module

This module is responsible for assigning a precise depth value to each image captured by the MDPI instrument. It synchronizes the timestamps of the images with data from a pressure sensor to create a detailed depth profile for each imaging run.

## How it Works

The module processes a directory of raw images and a corresponding pressure sensor CSV file to generate a final CSV mapping each image to a specific depth. The process involves several key steps:

1.  **Metadata Extraction**: The module begins by parsing metadata from the input directory path and filenames. This includes essential information such as the project name, recording date, start time, and location, which establishes the context for the profiling run.

2.  **Automatic Camera Format Detection**: It automatically detects whether the data is from a "new" or "old" MDPI camera format by inspecting the header of the pressure sensor's CSV file. This is crucial as different camera systems have unique CSV structures, column names, and pressure-to-depth conversion multipliers.

3.  **Image Timestamp Calculation**: Using the recording start time (from metadata) and the user-provided image capture rate (in Hz), the module calculates a precise timestamp for every single image in the sequence.

4.  **Depth Synchronization**: This is the core step of the process. For each calculated image timestamp, the module searches the pressure sensor data to find the depth measurement that was recorded at the nearest point in time. This synchronizes the two data streams.

5.  **Pixel Overlap Calculation**: To aid in subsequent analysis (like object tracking or avoiding duplicate counts), the module calculates the vertical pixel overlap between each consecutive pair of images based on their assigned depths and the physical height of the camera's field of view.

6.  **Output Generation**: The final output is a `depth_profiles.csv` file. This file contains the absolute path to each image, its calculated depth, and the pixel overlap with the next image, providing a comprehensive dataset for further scientific analysis.

## Usage

The module can be run as a standalone script from the command line. You must provide the input directory containing the images and the pressure sensor CSV, as well as the camera's capture rate.

```bash
python3 -m modules.depth_profiling -i /path/to/your/image/directory -c 2.4
```

-   `-i` or `--input`: The input directory containing the raw MDPI images and the pressure sensor CSV file.
-   `-o` or `--output`: (Optional) The root directory where the output CSV will be saved. It defaults to `./output`.
-   `-c` or `--capture-rate`: The image capture rate of the MDPI instrument in Hertz (Hz).

The final output file will be saved to a structured path: `<output_directory>/<project>/<date>/<cycle>/<location>/depth_profiles.csv`.

## Important Considerations

*   **Data Accuracy**: The quality of the output highly dependends on accurately reporting the `capture-rate`. Otherwise there will be a mismatch between images and depths.
*   **Configuration Constants**: The physical parameters, such as the height of the camera's field of view, are defined in `modules/common/constants.py`. These values must be correctly calibrated for the specific MDPI instrument being used.
