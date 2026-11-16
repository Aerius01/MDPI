# MDPI Processing Pipeline

## Overview

This repository processes the images of one MDPI (Modular Deep-sea Plankton Imager) descent into plankton concentrations per depth. It turns a folder of raw camera images plus the pressure sensor CSV into depth-resolved concentration tables and plots for each plankton class.

There are two ways to run it:

*   **Command line**: `run_pipeline.py`, one input folder per run.
*   **Web interface (GUI)**: a Streamlit app (`app/app.py`, started with `start_streamlit.sh`) that can process several input folders in one go.

Both run exactly the same pipeline.

## Pipeline Steps

The pipeline consists of the following sequential steps. Each module has its own README with details.

1.  **Depth Profiling** ([README](modules/depth_profiling/README.md)): Matches image timestamps to the pressure sensor data to give every image a depth.
2.  **Flatfielding** ([README](modules/flatfielding/README.md)): Corrects for non-uniform illumination in the images.
3.  **Object Detection** ([README](modules/object_detection/README.md)): Finds potential objects (plankton) in the images and crops them into small images called vignettes.
4.  **Object Classification** ([README](modules/object_classification/README.md)): Assigns a class (e.g. copepod) to each vignette using an ONNX model.
5.  **Concentration Calculation & Plotting** ([README](modules/plotter/README.md)): Calculates the concentration of each class per depth bin and generates plots.

## Installation

The pipeline runs in a conda environment called `mdpi-env`, defined in `environment_from_history.yml` (Python 3.14, packages from conda-forge). You need [Miniforge](https://github.com/conda-forge/miniforge) or another conda installation.

```bash
git clone https://github.com/Aerius01/MDPI.git
cd MDPI
conda env create -f environment_from_history.yml   # creates mdpi-env
conda activate mdpi-env
```

Check that it worked:

```bash
python3 run_pipeline.py --help
```

All commands below assume that you are in the repository folder with `mdpi-env` activated.

## Preparing the Input Folder

Each input folder holds the data of **one** descent:

```
E01_01/
├── E01_01.csv                                                    # pressure sensor data (exactly one .csv)
├── Basler acA2040-25gmNIR (21971232)_20230425_140846819_0001.tiff
├── Basler acA2040-25gmNIR (21971232)_20230425_140846819_0002.tiff
└── ...
```

Requirements:

*   **Images**: `.tiff`, `.png`, `.jpg` or `.jpeg` files (note: `.tif` with one "f" is not recognized). The file names must end in `_YYYYmmdd_HHMMSSfff_<replicate>.<ext>`:
    *   `YYYYmmdd_HHMMSSfff` is the date and time at which the recording **started**, down to milliseconds. It is the same for every image of the descent.
    *   `<replicate>` is the image number (a unique integer). Together with the capture rate it determines when each image was taken.
    *   Anything before the date (here `Basler acA2040-25gmNIR (21971232)`) is ignored.
*   **Pressure sensor data**: exactly one `.csv` file. The pipeline stops with an error if there are zero or several. Both the old and the new sensor CSV format are supported; the format is detected automatically from the header.

Example data is not part of the repository (the `profiles/` folder is ignored by git), so ask your supervisor for a sample descent if you do not have one.

## Running from the Command Line

```bash
python3 run_pipeline.py -i <input_directory> [-m <model.onnx>]
```

### Arguments

*   `-i, --input`: **(Required)** Path to the input folder described above.
*   `-m, --model`: *(Optional)* Path to an ONNX classification model. Defaults to the bundled model `model/mdpi_4class_v1.onnx`, which classifies `cladocera`, `copepod`, `junk` and `rotifer`. Custom models must follow the contract described in [Using your own model](modules/object_classification/README.md#using-your-own-model).

### Configuration prompt

After starting, the script shows the instrument configuration and asks whether to keep it:

```
--- MDPI Configuration ---
Default values:
  1) CAPTURE_RATE: 2.4 Hz
  2) IMAGE_HEIGHT_CM: 4.3 cm
  3) IMAGE_DEPTH: 10.0 cm
  4) IMAGE_WIDTH: 4.3 cm
Accept these values? (Y/N):
```

*   `Y` keeps the defaults.
*   `N` asks for each value in turn. Type a new number, or press Enter to keep the current one.

Capture rate and image height are used to assign depths; image height, depth and width give the imaged volume per frame, which the concentrations are based on. Only change them if your camera setup differs.

The model is checked right after this prompt, before any processing starts, and the script prints which model it uses:

```
[PIPELINE]: Using classification model /path/to/MDPI/model/mdpi_4class_v1.onnx
```

### Examples

Process one descent with the bundled model:

```bash
python3 run_pipeline.py -i /data/mdpi/Lake_X/20230425/day/E01_01
```

Process it with your own model:

```bash
python3 run_pipeline.py \
  -i /data/mdpi/Lake_X/20230425/day/E01_01 \
  -m ./my_models/my_model_mdpi.onnx
```

Accept the default configuration without typing (answers the prompt with `Y`):

```bash
echo Y | python3 run_pipeline.py -i /data/mdpi/Lake_X/20230425/day/E01_01
```

Use a different capture rate (5 Hz) and keep the other values. The answers are given in prompt order: `N`, capture rate, image height, image depth, image width (an empty line keeps a value):

```bash
printf 'N\n5\n\n\n\n' | python3 run_pipeline.py -i /data/mdpi/Lake_X/20230425/day/E01_01
```

Process every descent of a day, one after another:

```bash
for dir in /data/mdpi/Lake_X/20230425/day/*/; do
    echo Y | python3 run_pipeline.py -i "$dir" || echo "FAILED: $dir"
done
```

On a laptop, a descent of about 700 images takes a few minutes.

## Output

All results are written to an `output/` folder **inside the input folder**. Running the same folder again overwrites the previous results.

```
E01_01/output/
├── depth_profiles.csv                  # depth assigned to each image
├── flatfielded_images/                 # illumination-corrected images
├── vignettes/                          # one small image per detected object
├── object_data.csv                     # one row per object: depth, size, shape, predicted class, ...
├── object_data.pkl                     # raw classification results (probabilities, uncertainties, model used)
├── concentration_data.csv              # concentration per class and depth bin
├── sizeclass_concentration_data.csv    # concentration per class, size class and depth bin
└── plots/
    ├── copepod/
    │   ├── copepod_2023-04-25_concentration.png
    │   ├── copepod_2023-04-25_length.png
    │   └── copepod_sizeclass_1_2023-04-25.png ...
    └── ...                             # one folder per class of the model
```

In `object_data.csv`, the predicted class is stored in both the `prediction` and the `label` column; see the [classification README](modules/object_classification/README.md) for the uncertainty columns and the pickle file.

To browse the vignettes by predicted class, run `python3 class_viewer.py` and open an `object_data.pkl` through its File menu.

## Running the Streamlit Web Interface

The web interface lets you set the configuration in a form, pick input folders (also several at once, using **Add**), check that each folder is valid before running, and follow the log in the browser.

> **Note:** The web interface always uses the bundled model `model/mdpi_4class_v1.onnx`. To run with your own model, use the command line with `-m`.

### Basic Usage

```bash
./start_streamlit.sh
```

This will:
- Launch the web interface at `http://localhost:8501` (open this address in your browser)
- Use the conda environment named `mdpi-env` (it does not need to be activated)
- Run from the default project root at `$HOME/MDPI`

If you cloned the repository somewhere else, set `PROJECT_ROOT` (see below), e.g. from inside the repository folder:

```bash
PROJECT_ROOT="$PWD" ./start_streamlit.sh
```

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
- The **Browse** button needs a desktop session; on a headless server, paste the folder paths into the text fields instead
- Press `Ctrl+C` in the terminal to stop the Streamlit server

## Troubleshooting

Errors are printed as `[PIPELINE]: Error: ...` and the script exits. Common ones:

| Message | Cause and fix |
|---------|---------------|
| `Model file not found: '...'. Check the model path.` | The path given to `-m` does not exist. |
| `expected a '.onnx' file, found suffix '...'` | `-m` must point to an `.onnx` file. Convert your model to ONNX first. |
| Other errors starting with `Model '...':` | The model does not meet the [model contract](modules/object_classification/README.md#using-your-own-model); the message names the rule that failed. |
| Errors about the `.csv` file | The input folder must contain exactly one `.csv` file. Move any extra CSV files out. |
| Errors about missing images or file names | Check the image extension (`.tiff`, not `.tif`) and the `_YYYYmmdd_HHMMSSfff_<replicate>` naming. |
| `EOFError: EOF when reading a line` | The configuration prompt got no answer, e.g. in a script. Pipe the answer in as shown in the examples (`echo Y \| ...`). |
