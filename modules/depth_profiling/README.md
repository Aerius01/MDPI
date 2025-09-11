# Depth Profiling Module

This module is responsible for assigning a precise depth value to each image captured by the MDPI instrument. It synchronizes the timestamps of the images with data from a pressure sensor to create a detailed depth profile for each imaging run.

## Abstract

The Depth Profiling module is the first main step in the MDPI image processing pipeline. It is responsible for assigning a precise depth value to each image captured by the instrument. It synchronizes the timestamps of the images with data from a pressure sensor to create a detailed depth profile for each imaging run.

## How it Works

The module processes a directory of raw images and a corresponding pressure sensor CSV file to generate a final CSV mapping each image to a specific depth. The process involves several key steps:

1.  **Metadata Extraction**: The module begins by parsing metadata from the input directory path and filenames. This includes essential information such as the project name, recording date, start time, and location, which establishes the context for the profiling run.

2.  **Automatic Camera Format Detection**: It automatically detects whether the data is from a "new" or "old" MDPI camera format by inspecting the first row's entry of the pressure sensor's CSV file ('Name' == old format, 'Number' == new format). This is crucial as different camera systems have unique CSV structures, column names, and pressure-to-depth conversion multipliers.

3.  **Image Timestamp Calculation**: Using the recording start time (from metadata) and the user-provided image capture rate (in Hz), the module calculates a precise timestamp for every single image in the sequence.

4.  **Depth Synchronization**: This is the core step of the process. For each calculated image timestamp, the module searches the pressure sensor data to find the depth measurement that was recorded at the nearest point in time. This synchronizes the two data streams.

5.  **Pixel Overlap Calculation**: To aid in subsequent analysis (like object tracking or avoiding duplicate counts), the module calculates the vertical pixel overlap between each consecutive pair of images based on their assigned depths and the physical height of the camera's field of view.

6.  **Output Generation**: The final output is a `depth_profiles.csv` file. This file contains the absolute path to each image, its calculated depth, and the pixel overlap with the next image, providing a comprehensive dataset for further scientific analysis.

## Input Data Structure

For the metadata extraction to work correctly, the input directory and image filenames must follow a specific structure. The path provided via the `-i` flag should point to the `location` directory.

### Directory Structure

The module expects the following directory hierarchy:

```
.../ProjectName/YYYYMMDD/Cycle/Location/
```

-   **ProjectName**: The name of your project (e.g., `Project-Example`).
-   **YYYYMMDD**: The date of the recording in `YearMonthDay` format (e.g., `20230424`).
-   **Cycle**: The recording cycle, typically `day` or `night`.
-   **Location**: The specific location or station for the recording (e.g., `E07-01`).

This `Location` directory must contain all the raw image files and the single pressure sensor CSV for that specific run.

### Image Filename Format

The image filenames must be structured as follows to ensure the recording start time can be parsed correctly:

```
<any_prefix>_YYYYMMDD_HHMMSSmmm_<replicate_number>.<extension>
```

-   **<any_prefix>**: Can be any identifying string (e.g., `Basler_acA2040-25gmNIR (21971232)`).
-   **YYYYMMDD**: The date, which must match the date in the directory path.
-   **HHMMSSmmm**: The time in `HourMinuteSecondMillisecond` format (a 9-digit number). The recording start time is parsed from the *last* image file in the alphabetically sorted list.
-   **<replicate_number>**: A sequential number for the image.
-   **<extension>**: A valid image format, such as `.png`, `.jpg`, `.jpeg`, or `.tiff`.

For example: `Basler_acA2040-25gmNIR (21971232)_20230424_204500123_0.png`

Note: The reason the prefix does not matter is because the parsing of the metadata starts at the end of the string (the extension) and works backwards, splitting the data at the underscores.

## Preliminary Steps

Before running the depth profiling, it is advisable to check the raw image dataset for duplicates. The `duplicate_detection` module can be used for this purpose. This is an optional but recommended step to ensure data quality.

```bash
python3 -m modules.duplicate_detection -i /path/to/your/image/directory
```

## Usage

The module can be run as a standalone script from the command line. You must provide the input directory containing the images and the pressure sensor CSV.

```bash
python3 -m modules.depth_profiling -i /path/to/your/image/directory
```

-   `-i` or `--input`: The input directory containing the raw MDPI images and the pressure sensor CSV file.
-   `-o` or `--output`: (Optional) The root directory where the output CSV will be saved. It defaults to `./output`.

The final output file will be saved to a structured path: `<output_directory>/<project>/<date>/<cycle>/<location>/depth_profiles.csv`.

## Next Steps

The `depth_profiles.csv` file generated by this module is a required input for the next module in the pipeline, `flatfielding`, which corrects for non-uniform illumination in the images.

## Important Considerations

*   **Data Accuracy**: The quality of the output highly depends on the accuracy of the instrument's physical parameters.
*   **Configuration Constants**: The physical parameters are defined as constants in `modules/depth_profiling/depth_profile_data.py`. These values must be correctly calibrated for the specific MDPI instrument being used.
