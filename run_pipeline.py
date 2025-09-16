#!/usr/bin/env python3
"""
Unified CLI to run the full MDPI processing pipeline end-to-end.

Steps:
1) Depth profiling
2) Flatfielding
3) Object detection
4) Object classification
5) Concentration calculation

Example:
  python3 run_pipeline.py \
    -i /home/david-james/Desktop/04-MDPI/MDPI/profiles/Project_Example/20230425/day/E01_01 \
    -o /home/david-james/Desktop/04-MDPI/MDPI/output \
    -m /home/david-james/Desktop/04-MDPI/MDPI/model
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Common
from modules.common.constants import CONSTANTS
from modules.common.cli_utils import prompt_for_mdpi_configuration

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
import pandas as pd

# Global configuration defaults (not in CONSTANTS)
# Volume in L == dm^3
DEFAULT_BIN_SIZE = 0.10 # in meters
DEFAULT_MAX_DEPTH = 22.0 # in meters
DEFAULT_IMG_DEPTH = 1.00 # in decimeters
DEFAULT_IMG_WIDTH = 0.42 # in decimeters
CONCENTRATION_OUTPUT_FILENAME = "concentration_data.csv"
OBJECT_DATA_CSV_FILENAME = "object_data.csv"

# Duplicate detection is intentionally disabled to preserve depth matching in later steps.
def run_duplicate_detection(input_dir: str):
    """No-op: duplicate detection disabled."""
    return


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
    print(f"[PLOTTER]: Plots for {concentration_csv_path} saved in {output_path}.")


def execute_pipeline(input_dir, output_root, model_dir, capture_rate, image_height_cm, img_depth, img_width):
    """
    Executes the full MDPI pipeline.
    This function contains the core logic and is called by both the CLI and the Streamlit app.
    """
    # Resolve absolute paths
    input_dir = str(Path(input_dir).resolve())
    output_root = str(Path(output_root).resolve())
    model_dir = str(Path(model_dir).resolve())

    # 1) Duplicate detection intentionally skipped

    # 2) Depth profiling
    print("[PIPELINE]: Running depth profiling...")
    base_output_dir = run_depth_profiling(
        input_dir,
        output_root,
        capture_rate=capture_rate,
        image_height_cm=image_height_cm
    )
    depth_csv = os.path.join(base_output_dir, f"depth_profiles{CONSTANTS.CSV_EXTENSION}")

    # 3) Flatfielding
    print("[PIPELINE]: Running flatfielding...")
    flatfield_dir = run_flatfielding(input_dir, depth_csv, output_root)

    # 4) Object detection
    print("[PIPELINE]: Running object detection...")
    vignettes_dir = run_detection_step(flatfield_dir, depth_csv, output_root)

    # 5) Classification
    print("[PIPELINE]: Running object classification...")
    classification_output_dir = run_classification_step(
        vignettes_dir=vignettes_dir,
        output_root=output_root,
        model_dir=model_dir,
        batch_size=CLASSIFICATION_BATCH_SIZE,
        input_size=CLASSIFICATION_INPUT_SIZE,
        input_depth=CLASSIFICATION_INPUT_DEPTH,
    )

    # 6) Concentration calculation
    print("[PIPELINE]: Calculating concentrations...")
    object_data_csv = os.path.join(classification_output_dir, OBJECT_DATA_CSV_FILENAME)
    concentration_csv_path = run_concentration_step(
        object_data_csv=object_data_csv,
        max_depth=DEFAULT_MAX_DEPTH,
        bin_size=DEFAULT_BIN_SIZE,
        img_depth=img_depth,
        img_width=img_width,
    )

    # 7) Plotting
    print("[PIPELINE]: Generating plots...")
    run_plotting_step(concentration_csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full MDPI pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_pipeline.py -i ./profiles/Project_Example/20230425/day/E01_01 -o ./output -m ./model
        """,
    )

    # Required/primary args
    parser.add_argument("-i", "-\u002Dinput", dest="input", required=True, help="Input directory with raw MDPI images and pressure sensor CSV")
    parser.add_argument("-o", "-\u002Doutput", dest="output", default="./output", help="Root output directory")
    parser.add_argument("-m", "-\u002Dmodel", dest="model", required=True, help="Path to trained model directory containing model.ckpt files")

    args = parser.parse_args()

    capture_rate, image_height_cm, img_depth, img_width = prompt_for_mdpi_configuration(
        CAPTURE_RATE, IMAGE_HEIGHT_CM, DEFAULT_IMG_DEPTH, DEFAULT_IMG_WIDTH
    )

    try:
        execute_pipeline(
            input_dir=args.input,
            output_root=args.output,
            model_dir=args.model,
            capture_rate=capture_rate,
            image_height_cm=image_height_cm,
            img_depth=img_depth,
            img_width=img_width
        )
        print("[PIPELINE]: All steps completed successfully!")
    except Exception as e:
        print(f"[PIPELINE]: Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


