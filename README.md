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

## Running the Streamlit Web Interface

In addition to the command-line pipeline, you can use the Streamlit web interface for a more interactive experience. The `start_streamlit.sh` script provides a convenient way to launch the Streamlit app.

### Basic Usage

```bash
./start_streamlit.sh
```

This will:
- Launch the Streamlit web interface at `http://localhost:8501`
- Use the conda environment named `mdpi-env`
- Run from the default project root at `$HOME/MDPI`

### Environment Variables

The script supports several optional environment variables for customization:

---

#### `PROJECT_ROOT`

Set the path to the git project root directory.

**Default:** `$HOME/MDPI`

**When to set:** You cloned or moved the MDPI project to a location other than `$HOME/MDPI` (e.g., `$HOME/Desktop/MDPI` or `/opt/projects/MDPI`).

**Example:**
```bash
PROJECT_ROOT=/path/to/your/MDPI ./start_streamlit.sh
```

---

#### `PORT`

Specify the port number for the Streamlit server.

**Default:** `8501`

**When to set:** Port 8501 is already in use by another application, you need to run multiple instances simultaneously, or organizational/security policies require a different port.

**Example:**
```bash
PORT=8080 ./start_streamlit.sh
```

---

#### `ENV_NAME`

Set the name of your conda environment.

**Default:** `mdpi-env`

**When to set:** You created your conda environment with a different name than `mdpi-env` (e.g., following your own naming convention or managing multiple versions).

**Example:**
```bash
ENV_NAME=my-custom-env ./start_streamlit.sh
```

---

#### `ENV_PREFIX`

Specify the full path to your conda environment. This overrides `ENV_NAME` if set.

**Default:** Auto-detected from common conda installation locations (`~/miniconda3`, `~/anaconda3`, `~/miniforge3`, `~/mambaforge`, `/opt/conda`)

**When to set:** Your conda environment is in a non-standard location that the script cannot auto-detect, or you need explicit control over which environment is used.

**Example:**
```bash
ENV_PREFIX=$HOME/anaconda3/envs/my-env ./start_streamlit.sh
```

---

#### `CONDA_BIN`

Set the path to your conda executable if it's not automatically detected.

**Default:** Auto-detected from common installation paths

**When to set:** You installed conda in a custom location (e.g., `/usr/local/conda`, a network drive), or you're using a conda distribution that isn't in the standard paths checked by the script.

**Example:**
```bash
CONDA_BIN=/opt/conda/bin/conda ./start_streamlit.sh
```

---

### Combined Example

You can set multiple environment variables at once:

```bash
PROJECT_ROOT=/home/user/MDPI PORT=8080 ENV_NAME=plankton-env ./start_streamlit.sh
```

### Notes

- The script will automatically verify that Streamlit is installed in the target environment
- If conda cannot be found, you'll be prompted to set `CONDA_BIN` or install Miniconda/Anaconda
- Press `Ctrl+C` to stop the Streamlit server
