# Plotter Module

## Abstract

The Plotter module serves as the primary tool for visualizing the outputs of the MDPI processing pipeline. It is designed to transform the final `object_data.csv` file into a format suitable for generating depth profile plots. The module contains separate scripts for calculating organism concentrations and for creating two main types of visualizations: individual depth profiles for a single run (e.g., one day or one night cycle) and combined day-night profiles that compare organism distributions side-by-side.

## How it Works

The plotting process is a two-stage workflow. First, you must calculate concentration data from the main pipeline's output, and then you can generate the plots.

### Stage 1: Calculating Concentration Data

This stage is handled by the `calculate_concentrations.py` script. It takes the detailed `object_data.csv` file (the output of the `object_classification` module) and aggregates the data to calculate organism concentration as a function of depth.

1.  **Data Ingestion:** The script loads the `object_data.csv` file, which contains information about every detected object, including its depth and assigned class label.
2.  **Depth Binning:** To create a profile, the water column is divided into discrete vertical sections, or "bins" (e.g., 0.1 meters each). Each detected object is assigned to a specific depth bin based on its recorded depth.
3.  **Counting and Concentration Calculation:** The script counts the number of individuals of each specified class (e.g., 'copepod', 'cladocera') within each depth bin. This count is then converted into a concentration (individuals per cubic meter) by normalizing it based on the volume of water imaged for that bin.
4.  **Output:** The result of this stage is a new CSV file, `concentration_data.csv`, which is saved in the same directory as the input file. This new file is much smaller and contains the aggregated data required for plotting.

### Stage 2: Generating Plots

Once the `concentration_data.csv` file is created, you can use one of the two plotting scripts to visualize it.

#### Single Profiles (`plot_profile.py`)

This script generates individual depth profile plots. For each class of organism in the concentration data, it creates a horizontal bar chart where the y-axis represents depth and the x-axis represents concentration. This is useful for visualizing the distribution of a specific organism group during a single sampling event (e.g., a day or night cycle).

#### Day-Night Profiles (`plot_day_night_profile.py`)

This script is used to compare the vertical distribution of organisms between day and night. It requires two separate `concentration_data.csv` files—one for a day cycle and one for a night cycle. For each organism class, it generates a single plot with two back-to-back horizontal bar charts: one showing the day distribution and the other showing the night distribution. This allows for direct visual comparison to identify patterns like diel vertical migration.

## How to Use

The module is operated through the command line. Remember to run the concentration calculation first.

### Step 1: Calculate Concentrations

```bash
python3 -m modules.plotter.calculate_concentrations -i <path_to_object_data.csv>
```

-   `-i, --input`: Path to the `object_data.csv` file produced by the `object_classification` module.

This will create a `concentration_data.csv` file in the same directory.

### Step 2A: Plot a Single Profile

```bash
python3 -m modules.plotter.plot_profile -i <path_to_concentration_data.csv>
```

-   `-i, --csv_path`: Path to the `concentration_data.csv` file generated in Step 1.

The output plots (one per species) will be saved as `.png` files in the same directory as the input CSV.

### Step 2B: Plot a Day-Night Profile

```bash
python3 -m modules.plotter.plot_day_night_profile -d <path_to_day_concentration.csv> -n <path_to_night_concentration.csv> -o <output_directory>
```

-   `-d, --day_csv_path`: Path to the `concentration_data.csv` file for the day cycle.
-   `-n, --night_csv_path`: Path to the `concentration_data.csv` file for the night cycle.
-   `-o, --output_path` (Optional): The directory where the output plots will be saved. Defaults to `./output`.

## Constants, Concerns, and Limitations

### Constants

Several key parameters for both the calculation and plotting processes are defined as constants and may need to be adjusted for different datasets or instruments.

-   **`calculate_concentrations.py`:**
    -   `BIN_SIZE` (0.1): The vertical height (in meters) of each depth bin.
    -   `MAX_DEPTH` (18.0): The maximum depth to include in the profile.
    -   `IMG_DEPTH` (10.0) & `IMG_WIDTH` (0.42): The physical dimensions (in meters) of the camera's field of view, used for calculating the imaged volume. The number of individuals in the bin are divided by this volume to produce the concentration.

-   **`plot_*.py` (in `constants.py`):**
    -   `FIGSIZE`: The dimensions of the output plot image.
    -   `DAY_COLOR`, `NIGHT_COLOR`, `EDGE_COLOR`: The colors used for the bars in the plots.
    -   `FILE_FORMAT`: The file type for the saved plots (e.g., 'png').

### Concerns and Limitations

-   **Instrument Calibration:** The accuracy of the concentration calculations is critically dependent on the `IMG_DEPTH` (10.0 meters) and `IMG_WIDTH` (0.42 meters) constants. These must be precisely calibrated for the specific MDPI instrument used to collect the data. **Where are these coming from?**
