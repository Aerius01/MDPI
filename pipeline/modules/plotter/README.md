# Plotter Module

## Abstract

The Plotter module serves as the primary tool for visualizing the outputs of the MDPI processing pipeline. It is designed to transform the final `object_data.csv` file into a format suitable for generating various depth profile plots. The module calculates organism concentrations and generates three main types of visualizations: standard concentration profiles, length distribution profiles, and size-class concentration profiles.

## How it Works

The plotting process is a two-stage workflow. First, you must calculate concentration data from the main pipeline's output, and then you can generate the plots.

### Stage 1: Calculating Concentration Data

This stage is handled by the `calculate_concentrations.py` script. It processes the `object_data.csv` file from the classification module to produce aggregated datasets for plotting. Two types of concentration data are generated:

1.  **Standard Concentration:** The script calculates the concentration of each organism class as a function of depth.
    -   **Data Ingestion:** The script loads the `object_data.csv` file, which contains information about every detected object, including its depth, dimensions, and assigned class label.
    -   **Depth Binning:** To create a profile, the water column is divided into discrete vertical sections, or "bins" (e.g., 0.1 meters each). Each detected object is assigned to a specific depth bin based on its recorded depth.
    -   **Counting and Concentration Calculation:** The script counts the number of individuals of each specified class (e.g., 'copepod', 'cladocera') within each depth bin. This count is then converted into a concentration (individuals per cubic meter) by normalizing it based on the volume of water imaged for that bin.
    -   **Output:** The result is a `concentration_data.csv` file, saved in the output directory.
2.  **Size-Class Concentration:** The script also calculates concentration for different size classes of organisms.
    -   **Size Binning:** Organisms are categorized into predefined size classes based on their length.
    -   **Concentration Calculation:** For each size class, the concentration is calculated within each depth bin, similar to the standard concentration.
    -   **Output:** This produces a `sizeclass_concentration_data.csv` file in the output directory.

### Stage 2: Generating Plots

Once the concentration data is calculated, the module generates three types of plots.

#### Concentration Profiles (`plot_profile.py`)

This script uses `concentration_data.csv` to generate individual depth profile plots. For each organism class (eg. 'rotifer', 'clacodera', etc), it creates a horizontal bar chart where the y-axis represents depth and the x-axis represents concentration.

#### Length Profiles (`plot_length_profile.py`)

This script visualizes the length distribution of detected organisms against depth. It uses the raw `object_data.csv` to create a scatter plot where each point represents an individual organism, with its depth on the y-axis and its length on the x-axis.

#### Size-Class Profiles (`plot_size_profiles.py`)

This script uses the `sizeclass_concentration_data.csv` file. It generates depth profiles for each predefined size class, showing the concentration of organisms within that size range at various depths.


## Configuration, Concerns, and Limitations

### Configuration Parameters

Key parameters for calculations and plotting are now configured when the main pipeline is launched, providing flexibility for different datasets or instruments.

-   **Calculation Parameters (`calculate_concentrations.py`):**
    -   `BIN_SIZE`: The vertical height (in meters) of each depth bin for the plotted profiles (Default: 0.1 m).
    -   `MAX_DEPTH`: The maximum depth to include in the profile (Default: 22.0 m).
    -   `IMG_DEPTH` & `IMG_WIDTH`: The physical dimensions (in decimeters) of the camera's field of view, used for calculating the imaged volume. These values are crucial for accurate concentration calculations. Note that in the web-app these are specified in *centimeters*, and then later converted to decimeters before being passed off to execute_pipeline() in the run_pipeline.py script, which expects decimeters.

-   **Plotting Parameters (`run.py`):**
    -   `FIGSIZE`: The dimensions of the output plot image.
    -   `DAY_COLOR`, `NIGHT_COLOR`, `EDGE_COLOR`: The colors used for the bars in the plots.
    -   `FILE_FORMAT`: The file type for the saved plots (e.g., 'png').

### Concerns and Limitations

-   **Instrument Calibration:** The accuracy of the concentration calculations is critically dependent on `IMG_DEPTH` and `IMG_WIDTH`. These must be precisely calibrated for the specific MDPI instrument used. These values are now prompted for when `run_pipeline.py` is executed, with defaults set in the script.
