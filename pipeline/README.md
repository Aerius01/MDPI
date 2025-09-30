# MDPI Processing Pipeline

## Overview

This script (`run_pipeline.py`) serves as the main entry point for running the full MDPI (Plankton Imaging) processing pipeline. It automates the entire workflow from raw data to final concentration plots.

## Pipeline Steps

The pipeline consists of the following sequential steps:

1.  **Depth Profiling**: Processes pressure sensor data to determine the depth of each image.
2.  **Flatfielding**: Corrects for non-uniform illumination in the images.
3.  **Object Detection**: Identifies and crops potential objects (plankton) from the images.
4.  **Object Classification**: Classifies the detected objects using a trained model.
5.  **Concentration Calculation & Plotting**: Calculates the concentration of classified objects at different depths and generates plots.

## Prerequisites

Before running the pipeline, ensure you have the following:

*   **Input Directory**: A directory containing:
    *   Raw MDPI images (e.g., `.tif` files).
    *   A pressure sensor data file (CSV format).
    > **Note:** The input directory must contain exactly one `.csv` file for the pressure sensor data. The pipeline will fail if zero or more than one `.csv` files are found.
*   **Trained Model**: A directory containing the trained TensorFlow model checkpoint files (`model.ckpt.meta`, `model.ckpt.index`, `model.ckpt.data-00000-of-00001`).

## Usage

The script is executed from the command line.

```bash
python3 run_pipeline.py -i <input_directory> -m <model_directory>
```

### Arguments

*   `-i, --input`: **(Required)** Path to the input directory containing the raw MDPI images and the pressure sensor CSV file.
*   `-m, --model`: **(Required)** Path to the directory containing the trained model checkpoint files.

Upon execution, the script will prompt for MDPI configuration details like capture rate and image dimensions.

## Example

Here is an example of how to run the pipeline:

```bash
python3 run_pipeline.py \
  -i ./profiles/Project_Example/20230425/day/E01_01 \
  -m ./model
```
