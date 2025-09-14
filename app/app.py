import streamlit as st
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import multiprocessing
import queue
import time
import glob


st.set_page_config(layout="wide")


# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Common
from modules.common.constants import CONSTANTS
from modules.common.parser import _parse_path_metadata

# Duplicate detection
from modules.duplicate_detection.utils import process_arguments as duplicate_process_arguments
from modules.duplicate_detection.detector import deduplicate_images

# Depth profiling
from modules.depth_profiling.depth_profile_data import (
    process_arguments as depth_process_arguments,
    CAPTURE_RATE,
    IMAGE_HEIGHT_CM,
)
from modules.depth_profiling.profiler import profile_depths

# Flatfielding
from modules.flatfielding.flatfielding_data import (
    process_arguments as flatfielding_process_arguments,
)
from modules.flatfielding.flatfielding import flatfield_images

# Object detection
from modules.object_detection.detection_data import (
    validate_arguments as detection_validate_arguments,
    THRESHOLD_VALUE,
    THRESHOLD_MAX,
    MIN_OBJECT_SIZE,
    MAX_OBJECT_SIZE,
    MAX_ECCENTRICITY,
    MAX_MEAN_INTENSITY,
    MIN_MAJOR_AXIS_LENGTH,
    MAX_MIN_INTENSITY,
    SMALL_OBJECT_PADDING,
    MEDIUM_OBJECT_PADDING,
    LARGE_OBJECT_PADDING,
    SMALL_OBJECT_THRESHOLD,
    MEDIUM_OBJECT_THRESHOLD,
    OUTPUT_CSV_SEPARATOR
)
from modules.object_detection.__main__ import run_detection
from modules.object_detection.detector import Detector
from modules.object_detection.output_handler import OutputHandler

# Classification
from modules.object_classification.utils import parse_vignette_metadata
from modules.object_classification.classification_data import (
    validate_arguments as cls_validate_arguments,
    CLASSIFICATION_BATCH_SIZE,
    CLASSIFICATION_INPUT_SIZE,
    CLASSIFICATION_INPUT_DEPTH
)
from modules.object_classification.inference_engine import InferenceEngine
from modules.object_classification.processor import ClassificationProcessor
from modules.object_classification.run import run_classification

# Plotter
from modules.plotter.calculate_concentrations import (
    ConcentrationConfig,
    calculate_concentration_data,
)
from modules.plotter.constants import PLOTTING_CONSTANTS
from modules.plotter.plot_profile import PlotConfig, plot_single_profile


def validate_raw_images_path(path):
    """Check if the raw images path is valid."""
    results = []
    if not path or not os.path.isdir(path):
        results.append((False, "Path must be a valid directory."))
        return results

    results.append((True, "Path is a valid directory."))

    try:
        _parse_path_metadata(Path(path))
        results.append((True, "Directory structure is valid."))
    except ValueError as e:
        results.append((False, f"Directory structure error: {e}"))

    image_extensions = ('*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png')
    image_files = [f for ext in image_extensions for f in glob.glob(os.path.join(path, ext))]
    if image_files:
        results.append((True, f"Found {len(image_files)} image file(s)."))
    else:
        results.append((False, "No supported image files found (.tif, .jpg, .png)."))

    csv_files = glob.glob(os.path.join(path, '*.csv'))
    if len(csv_files) == 1:
        results.append((True, "Found one pressure sensor CSV file."))
    else:
        results.append((False, f"Expected 1 CSV file, but found {len(csv_files)}."))
    return results

def validate_output_path(path):
    """Check if the output path is valid."""
    results = []
    if not path or not path.strip():
        return [(False, "Output path cannot be empty.")]

    parent_dir = os.path.dirname(path) or '.'
    if not os.path.isdir(parent_dir):
        return [(False, f"Parent directory does not exist: {parent_dir}")]

    if os.access(parent_dir, os.W_OK):
        results.append((True, "Output directory is writable."))
    else:
        results.append((False, "Output directory is not writable."))

    if os.path.exists(path) and not os.path.isdir(path):
        results.append((False, "Path exists but is a file, not a directory."))
    return results

def validate_model_path(path):
    """Check if the model path is valid."""
    results = []
    if not path or not os.path.isdir(path):
        results.append((False, "Path must be a valid directory."))
        return results
    
    results.append((True, "Path is a valid directory."))

    required_files = ['checkpoint', 'model.ckpt.index', 'model.ckpt.meta']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(path, f))]
    data_files = glob.glob(os.path.join(path, 'model.ckpt.data-*-of-*'))

    if not missing_files and data_files:
        results.append((True, "Found required model checkpoint files."))
    else:
        if not data_files:
            missing_files.append("model.ckpt.data-*-of-*")
        results.append((False, f"Missing model files: {', '.join(missing_files)}"))
    return results

def display_validation(validation_results):
    """Render validation results with icons."""
    for is_valid, message in validation_results:
        if is_valid:
            st.markdown(f"<p style='color:green; margin-bottom:0.1rem; font-size:0.9rem;'>✔ {message}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:red; margin-bottom:0.1rem; font-size:0.9rem;'>❌ {message}</p>", unsafe_allow_html=True)

def display_path_parsing(path):
    """Show how the raw image path will be parsed in a collapsible expander."""
    with st.expander("Path Parsing Details", expanded=True):
        try:
            metadata = _parse_path_metadata(Path(path))
            items = list(metadata.items())
            for i, (key, value) in enumerate(items):
                # Add extra bottom margin to the last item for padding
                margin_bottom = '0.8rem' if i == len(items) - 1 else '0.1rem'
                st.markdown(f"<p style='margin-bottom:{margin_bottom}; font-size:0.9rem; margin-left: 1rem;'><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>", unsafe_allow_html=True)
        except ValueError as e:
            st.markdown(f"<p style='color:orange; margin-bottom:0.1rem; font-size:0.9rem;'>⚠️ Could not parse path: {e}</p>", unsafe_allow_html=True)


def pipeline_worker(log_queue: multiprocessing.Queue, input_dir, output_root, model_dir, capture_rate, image_height_cm, img_depth, img_width):
    """The target function for the pipeline process. No st calls here."""
    try:
        # Resolve absolute paths
        input_dir = str(Path(input_dir).resolve())
        output_root = str(Path(output_root).resolve())
        model_dir = str(Path(model_dir).resolve())

        # 1) Duplicate detection (pre-processing)
        log_queue.put("[PIPELINE]: Running duplicate detection...")
        run_duplicate_detection(input_dir)

        # 2) Depth profiling
        log_queue.put("[PIPELINE]: Running depth profiling...")
        base_output_dir = run_depth_profiling(
            input_dir,
            output_root,
            capture_rate=capture_rate,
            image_height_cm=image_height_cm
        )
        depth_csv = os.path.join(base_output_dir, f"depth_profiles{CONSTANTS.CSV_EXTENSION}")

        # 3) Flatfielding
        log_queue.put("[PIPELINE]: Running flatfielding...")
        flatfield_dir = run_flatfielding(input_dir, depth_csv, output_root)

        # 4) Object detection
        log_queue.put("[PIPELINE]: Running object detection...")
        vignettes_dir = run_detection_step(flatfield_dir, depth_csv, output_root)

        # 5) Classification
        log_queue.put("[PIPELINE]: Running object classification...")
        classification_output_dir = run_classification_step(
            vignettes_dir=vignettes_dir,
            output_root=output_root,
            model_dir=model_dir,
            batch_size=CLASSIFICATION_BATCH_SIZE,
            input_size=CLASSIFICATION_INPUT_SIZE,
            input_depth=CLASSIFICATION_INPUT_DEPTH,
        )

        # 6) Concentration calculation
        log_queue.put("[PIPELINE]: Calculating concentrations...")
        object_data_csv = os.path.join(classification_output_dir, OBJECT_DATA_CSV_FILENAME)
        concentration_csv_path = run_concentration_step(
            object_data_csv=object_data_csv,
            max_depth=DEFAULT_MAX_DEPTH,
            bin_size=DEFAULT_BIN_SIZE,
            img_depth=img_depth,
            img_width=img_width,
        )

        # 7) Plotting
        log_queue.put("[PIPELINE]: Generating plots...")
        run_plotting_step(concentration_csv_path)

        log_queue.put("---DONE---")
        log_queue.put("[PIPELINE]: All steps completed successfully!")
    except Exception as e:
        log_queue.put("---ERROR---")
        log_queue.put(f"[PIPELINE]: Error: {e}")


def select_folder(key: str):
    """Open folder dialog and update session_state."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    if folder_path:
        st.session_state[key] = folder_path

# Global configuration defaults (not in CONSTANTS)
# Volume in L == dm^3
DEFAULT_BIN_SIZE = 1.00 # in decimeters
DEFAULT_MAX_DEPTH = 22.0 # in meters
DEFAULT_IMG_DEPTH = 1.00 # in decimeters
DEFAULT_IMG_WIDTH = 0.42 # in decimeters
CONCENTRATION_OUTPUT_FILENAME = "concentration_data.csv"
OBJECT_DATA_CSV_FILENAME = "object_data.csv"

def run_duplicate_detection(input_dir: str):
    """Run duplicate detection on raw images."""
    args = SimpleNamespace(input=input_dir)
    image_paths = duplicate_process_arguments(args)
    deduplicate_images(image_paths)


def run_depth_profiling(
    input_dir: str,
    output_root: str,
    capture_rate: float,
    image_height_cm: float
) -> str:
    """Run depth profiling and return the base output path for this run."""
    args = SimpleNamespace(input=input_dir, output=output_root)
    # All parameters are now automatically detected and configured
    validated = depth_process_arguments(
        args,
        capture_rate_override=capture_rate,
        image_height_cm_override=image_height_cm
    )

    profile_depths(validated)

    # Depth module writes to <output_root>/<project>/<date>/<cycle>/<location>
    return validated.output_path


def run_flatfielding(raw_input_dir: str, depth_csv_path: str, output_root: str) -> str:
    """Run flatfielding and return the flatfielded images directory path."""
    args = SimpleNamespace(input=raw_input_dir, depth_profiles=depth_csv_path, output=output_root)
    data = flatfielding_process_arguments(args)
    flatfield_images(data)
    return data.output_path


def run_detection_step(flatfield_dir: str, depth_csv_path: str, output_root: str) -> str:
    """Run detection and return the vignettes output directory."""
    args = SimpleNamespace(input=flatfield_dir, depth_profiles=depth_csv_path, output=output_root)
    detection_data = detection_validate_arguments(args)

    detector = Detector(
        threshold_value=THRESHOLD_VALUE,
        threshold_max=THRESHOLD_MAX,
        min_object_size=MIN_OBJECT_SIZE,
        max_object_size=MAX_OBJECT_SIZE,
        max_eccentricity=MAX_ECCENTRICITY,
        max_mean_intensity=MAX_MEAN_INTENSITY,
        min_major_axis_length=MIN_MAJOR_AXIS_LENGTH,
        max_min_intensity=MAX_MIN_INTENSITY,
        small_object_threshold=SMALL_OBJECT_THRESHOLD,
        medium_object_threshold=MEDIUM_OBJECT_THRESHOLD,
        large_object_padding=LARGE_OBJECT_PADDING,
        small_object_padding=SMALL_OBJECT_PADDING,
        medium_object_padding=MEDIUM_OBJECT_PADDING,
        batch_size=CONSTANTS.BATCH_SIZE,
    )
    output_handler = OutputHandler(csv_extension=CONSTANTS.CSV_EXTENSION, csv_separator=OUTPUT_CSV_SEPARATOR)

    run_detection(detection_data, detector, output_handler)
    return detection_data.output_path


def run_classification_step(
    vignettes_dir: str,
    output_root: str,
    model_dir: str,
    batch_size: int,
    input_size: int,
    input_depth: int,
) -> str:
    """Run classification and return the base output directory for files (without 'vignettes')."""
    metadata = parse_vignette_metadata(Path(vignettes_dir))
    args = dict(
        input=vignettes_dir,
        output=output_root,
        model=model_dir,
        batch_size=batch_size,
        input_size=input_size,
        input_depth=input_depth,
    )
    classification_data = cls_validate_arguments(**args, **metadata)

    inference_engine = InferenceEngine(classification_data)
    processor = ClassificationProcessor()
    run_classification(classification_data, inference_engine, processor)

    # Classification writes outputs into <output_root>/<project>/<date>/<cycle>/<location>
    return str(classification_data.output_path)


def run_concentration_step(
    object_data_csv: str,
    max_depth: float,
    bin_size: float,
    img_depth: float,
    img_width: float,
) -> str:
    """Run concentration calculation and return path to saved CSV."""
    # Load classification CSV (merged with detection)
    data = pd.read_csv(object_data_csv, sep=';', dtype={
        'project': str,
        'cycle': str,
        'replicate': str,
        'prediction': str,
        'label': str,
        'FileName': str,
    }, engine='python')

    config = ConcentrationConfig(
        max_depth=max_depth,
        bin_size=bin_size,
        output_file_name=CONCENTRATION_OUTPUT_FILENAME,
        img_depth=img_depth,
        img_width=img_width,
    )
    concentration_df = calculate_concentration_data(data, config)
    output_path = os.path.join(os.path.dirname(object_data_csv), config.output_file_name)
    concentration_df.to_csv(output_path, index=False, sep=';')
    # Use print for worker process, not st
    print(f"[PLOTTER]: Concentration data saved to: {output_path}")
    return output_path


def run_plotting_step(concentration_csv_path: str):
    """Run plotting from a concentration data CSV."""
    config = PlotConfig(
        figsize=PLOTTING_CONSTANTS.FIGSIZE,
        day_color=PLOTTING_CONSTANTS.DAY_COLOR,
        night_color=PLOTTING_CONSTANTS.NIGHT_COLOR,
        edge_color=PLOTTING_CONSTANTS.EDGE_COLOR,
        align=PLOTTING_CONSTANTS.ALIGN,
        file_format=PLOTTING_CONSTANTS.FILE_FORMAT
    )

    input_csv = pd.read_csv(concentration_csv_path, sep=';', engine='python')
    output_path = os.path.dirname(concentration_csv_path)
    plot_single_profile(input_csv, output_path, config)
    # Use print for worker process, not st
    print(f"[PLOTTER]: Plots for {concentration_csv_path} saved in {output_path}.")


def run_pipeline(input_dir, output_root, model_dir, capture_rate, image_height_cm):
    """Main function to run the pipeline."""
    with st.spinner("Running pipeline..."):
        try:
            # Resolve absolute paths
            input_dir = str(Path(input_dir).resolve())
            output_root = str(Path(output_root).resolve())
            model_dir = str(Path(model_dir).resolve())

            # 1) Duplicate detection (pre-processing)
            st.info("[PIPELINE]: Running duplicate detection...")
            run_duplicate_detection(input_dir)

            # 2) Depth profiling
            st.info("[PIPELINE]: Running depth profiling...")
            base_output_dir = run_depth_profiling(
                input_dir,
                output_root,
                capture_rate=capture_rate,
                image_height_cm=image_height_cm
            )
            depth_csv = os.path.join(base_output_dir, f"depth_profiles{CONSTANTS.CSV_EXTENSION}")

            # 3) Flatfielding
            st.info("[PIPELINE]: Running flatfielding...")
            flatfield_dir = run_flatfielding(input_dir, depth_csv, output_root)

            # 4) Object detection
            st.info("[PIPELINE]: Running object detection...")
            vignettes_dir = run_detection_step(flatfield_dir, depth_csv, output_root)

            # 5) Classification
            st.info("[PIPELINE]: Running object classification...")
            classification_output_dir = run_classification_step(
                vignettes_dir=vignettes_dir,
                output_root=output_root,
                model_dir=model_dir,
                batch_size=CLASSIFICATION_BATCH_SIZE,
                input_size=CLASSIFICATION_INPUT_SIZE,
                input_depth=CLASSIFICATION_INPUT_DEPTH,
            )

            # 6) Concentration calculation
            st.info("[PIPELINE]: Calculating concentrations...")
            object_data_csv = os.path.join(classification_output_dir, OBJECT_DATA_CSV_FILENAME)
            concentration_csv_path = run_concentration_step(
                object_data_csv=object_data_csv,
                max_depth=DEFAULT_MAX_DEPTH,
                bin_size=DEFAULT_BIN_SIZE,
                img_depth=DEFAULT_IMG_DEPTH,
                img_width=DEFAULT_IMG_WIDTH,
            )

            # 7) Plotting
            st.info("[PIPELINE]: Generating plots...")
            run_plotting_step(concentration_csv_path)

            st.success("[PIPELINE]: All steps completed successfully!")
        except Exception as e:
            st.error(f"[PIPELINE]: Error: {e}")

# --- Initialize Session State ---
if 'pipeline_process' not in st.session_state:
    st.session_state.pipeline_process = None
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = None
if 'logs' not in st.session_state:
    st.session_state.logs = []


# --- Title & Intro ---
st.title("MDPI Image Processing Pipeline")
st.write("Select input folders and run the pipeline.")

# --- Configuration ---
st.subheader("Configuration")
c_col1, c_col2 = st.columns(2)
with c_col1:
    capture_rate = st.number_input("Capture Rate (Hz)", value=CAPTURE_RATE)
    image_depth_cm = st.number_input("Image Depth (cm)", value=DEFAULT_IMG_DEPTH * 10)
with c_col2:
    image_height_cm = st.number_input("Image Height (cm)", value=IMAGE_HEIGHT_CM)
    image_width_cm = st.number_input("Image Width (cm)", value=DEFAULT_IMG_WIDTH * 10)


# --- Folder Pickers ---
st.subheader("Input Paths")
if 'raw_images' not in st.session_state:
    st.session_state.raw_images = os.path.join(PROJECT_ROOT, "profiles/Project_Example/20230425/day/E01_01")
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = os.path.join(PROJECT_ROOT, "output")
if 'model_dir' not in st.session_state:
    st.session_state.model_dir = os.path.join(PROJECT_ROOT, "model")

st.write("Path to Raw Images Folder")
col1, col2 = st.columns([1, 4])
with col1:
    st.button("Browse", on_click=select_folder, args=("raw_images",), key="browse_raw", use_container_width=True)
with col2:
    st.text_input("Path to Raw Images Folder", key="raw_images", label_visibility="collapsed")
raw_images_validation = validate_raw_images_path(st.session_state.raw_images)
display_validation(raw_images_validation)
if all(v[0] for v in raw_images_validation):
    st.write("")  # Add vertical space
    display_path_parsing(st.session_state.raw_images)
    st.write("")  # Add vertical space
else:
    # Maintain vertical space if expander is not shown
    st.write("")

st.write("Path to Output Folder")
col1, col2 = st.columns([1, 4])
with col1:
    st.button("Browse", on_click=select_folder, args=("output_dir",), key="browse_output", use_container_width=True)
with col2:
    st.text_input("Path to Output Folder", key="output_dir", label_visibility="collapsed")
display_validation(validate_output_path(st.session_state.output_dir))

st.write("")  # Add vertical space
st.write("Path to Pretrained Model Folder")
col1, col2 = st.columns([1, 4])
with col1:
    st.button("Browse", on_click=select_folder, args=("model_dir",), key="browse_model", use_container_width=True)
with col2:
    st.text_input("Path to Pretrained Model Folder", key="model_dir", label_visibility="collapsed")
display_validation(validate_model_path(st.session_state.model_dir))

st.divider()

# --- Run/Stop Buttons ---
def start_pipeline():
    raw_images = st.session_state.raw_images
    output_dir = st.session_state.output_dir
    model_dir = st.session_state.model_dir

    raw_valid = all(v[0] for v in validate_raw_images_path(raw_images))
    out_valid = all(v[0] for v in validate_output_path(output_dir))
    model_valid = all(v[0] for v in validate_model_path(model_dir))

    if not (raw_valid and out_valid and model_valid):
        st.error("Please fix the validation errors before running the pipeline.")
        return
    else:
        log_queue = multiprocessing.Queue()
        # Convert cm to dm before passing to worker
        image_depth_dm = image_depth_cm / 10.0
        image_width_dm = image_width_cm / 10.0
        process = multiprocessing.Process(
            target=pipeline_worker,
            args=(log_queue, raw_images, output_dir, model_dir, capture_rate, image_height_cm, image_depth_dm, image_width_dm)
        )
        process.daemon = True
        st.session_state.pipeline_process = process
        st.session_state.log_queue = log_queue
        st.session_state.logs = ["Starting pipeline..."]
        process.start()

def stop_pipeline():
    process = st.session_state.pipeline_process
    if process and process.is_alive():
        process.terminate()
        # Wait for a short period to allow for graceful termination
        process.join(timeout=2)
        # If the process is still alive, force kill it
        if process.is_alive():
            process.kill()
        
        st.session_state.pipeline_process = None
        st.session_state.log_queue = None
        st.session_state.logs.append("Pipeline stopped by user.")


is_running = st.session_state.pipeline_process is not None and st.session_state.pipeline_process.is_alive()

b_col1, b_col2, _ = st.columns([1, 1, 5])
with b_col1:
    st.button("Run", on_click=start_pipeline, disabled=is_running, use_container_width=True)
with b_col2:
    st.button("Stop", on_click=stop_pipeline, disabled=not is_running, use_container_width=True)


# --- Log Display ---
log_placeholder = st.empty()

if st.session_state.log_queue:
    while True:
        try:
            log_entry = st.session_state.log_queue.get_nowait()
            if log_entry == "---DONE---" or log_entry == "---ERROR---":
                # Signal received. Get the final message that follows.
                try:
                    final_message = st.session_state.log_queue.get_nowait()
                    st.session_state.logs.append(final_message)
                except queue.Empty:
                    # Fallback if worker fails to send final message
                    if log_entry == "---ERROR---":
                        st.session_state.logs.append("[PIPELINE]: An unknown error occurred.")

                # Clean up state
                st.session_state.pipeline_process = None
                st.session_state.log_queue = None
                break  # Exit the log-reading loop
            else:
                st.session_state.logs.append(log_entry)
        except queue.Empty:
            # No more messages for now
            break

log_placeholder.code('\n'.join(st.session_state.logs))

if is_running:
    time.sleep(0.1)
    st.rerun()
